# File: tianyusun1/test2/test2-4.0/models/location.py (V4.5: GAUSSIAN HEATMAPS)

import torch
import torch.nn.functional as F
import random
import numpy as np

# [新增] 尝试导入可视化工具 (如果文件结构允许)
try:
    from data.visualize import draw_gaussian_heatmap
except ImportError:
    draw_gaussian_heatmap = None

class LocationSignalGenerator:
    """
    [V4.5] 高斯热力图位置生成器 (Gaussian Heatmap Generator)
    
    核心能力：
    1. 生成平滑的高斯分布作为位置引导 (Soft Guidance)，提供梯度信息。
    2. 维护全局占用图 (Occupancy Map)，避免物体堆叠。
    3. 支持多样性采样 (Diversity Sampling)。
    """
    def __init__(self, grid_size=8, sigma=1.5):
        self.H = grid_size
        self.W = grid_size
        self.sigma = sigma # 高斯分布的标准差，控制热力图的"晕染"范围
        
        # 1. 初始化高斯模式库
        self.patterns = self._init_gaussian_patterns()
        
        # 2. 定义关系的“候选策略池”
        # 策略名必须与 patterns 中的 key 对应
        self.STRATEGY_POOL = {
            'above': ['sky_center', 'sky_left', 'sky_right'], 
            'below': ['ground_full', 'ground_left', 'ground_right'],
            'on_top': ['ground_center', 'ground_left', 'ground_right'], # 类似 below 但更聚焦
            'inside': ['center_focus', 'core'],
            'surrounds': ['frame'], # 环绕 -> 框架
            'near': ['wing_left', 'wing_right', 'center_focus']
        }
        
        # 关系 ID 映射 (必须与 KG 保持一致)
        self.REL_IDS = {
            'none': 0, 'above': 1, 'below': 2, 'inside': 3,
            'surrounds': 4, 'on_top': 5, 'near': 6
        }

    def _zeros(self):
        return torch.zeros((self.H, self.W), dtype=torch.float32)

    def _generate_gaussian(self, center_x, center_y, sigma_x=None, sigma_y=None):
        """生成一个 2D 高斯分布热力图"""
        if sigma_x is None: sigma_x = self.sigma
        if sigma_y is None: sigma_y = self.sigma
        
        # 创建网格坐标
        x = torch.arange(0, self.W, 1, dtype=torch.float32)
        y = torch.arange(0, self.H, 1, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        
        # 高斯公式: exp(-((x-x0)^2/2sx^2 + (y-y0)^2/2sy^2))
        heatmap = torch.exp(-((x_grid - center_x)**2 / (2 * sigma_x**2) + (y_grid - center_y)**2 / (2 * sigma_y**2)))
        return heatmap

    def _init_gaussian_patterns(self):
        """初始化高斯模式库 (适配 8x8 Grid)"""
        pats = {}
        H, W = self.H, self.W
        
        # --- 天空 (Sky) ---
        # 左上: 中心 (2, 2)
        pats['sky_left']   = self._generate_gaussian(center_x=W*0.25, center_y=H*0.25)
        # 右上: 中心 (6, 2)
        pats['sky_right']  = self._generate_gaussian(center_x=W*0.75, center_y=H*0.25)
        # 中上: 扁平椭圆
        pats['sky_center'] = self._generate_gaussian(center_x=W*0.5, center_y=H*0.2, sigma_x=2.5, sigma_y=1.5)
        
        # --- 地面 (Ground) ---
        # 左下: 中心 (2, 6)
        pats['ground_left']  = self._generate_gaussian(center_x=W*0.25, center_y=H*0.75)
        # 右下: 中心 (6, 6)
        pats['ground_right'] = self._generate_gaussian(center_x=W*0.75, center_y=H*0.75)
        # 中下
        pats['ground_center'] = self._generate_gaussian(center_x=W*0.5, center_y=H*0.75)
        # 底部通栏: 很宽的扁平高斯
        pats['ground_full']  = self._generate_gaussian(center_x=W*0.5, center_y=H*0.85, sigma_x=4.0, sigma_y=1.5)
        
        # --- 核心 (Core / Inside) ---
        pats['core'] = self._generate_gaussian(center_x=W*0.5, center_y=H*0.5, sigma_x=2.0, sigma_y=2.0)
        pats['center_focus'] = self._generate_gaussian(center_x=W*0.5, center_y=H*0.5, sigma_x=1.0, sigma_y=1.0)
        
        # --- 侧翼 (Wings / Near) ---
        pats['wing_left']  = self._generate_gaussian(center_x=W*0.2, center_y=H*0.5, sigma_x=1.0, sigma_y=2.5)
        pats['wing_right'] = self._generate_gaussian(center_x=W*0.8, center_y=H*0.5, sigma_x=1.0, sigma_y=2.5)
        
        # --- 框架 (Frame / Surrounds) ---
        # 反向高斯模拟"四周高中间低"
        center_blob = self._generate_gaussian(center_x=W*0.5, center_y=H*0.5, sigma_x=3.0, sigma_y=3.0)
        pats['frame'] = (1.0 - center_blob).clamp(min=0.0)
        
        return pats

    def infer_stateful_signal(self, my_idx, spatial_row, spatial_col, current_occupancy, mode='greedy', top_k=3):
        """
        [V4.5] 状态感知推理 (支持高斯热力图)
        Args:
            my_idx: 当前物体索引
            spatial_row: 主动关系 (我 -> 他)
            spatial_col: 被动关系 (他 -> 我)
            current_occupancy: [8, 8] 累积热力图
            mode: 'greedy' 或 'sample'
            top_k: 采样候选数
        Returns:
            best_signal: [8, 8]
            new_occupancy: [8, 8]
        """
        # 1. 确定候选策略
        candidates = [] 
        
        # (A) 主动关系
        for rel_id in spatial_row:
            rel = int(rel_id)
            if rel == 0: continue
            rel_name = next((k for k, v in self.REL_IDS.items() if v == rel), None)
            if rel_name in self.STRATEGY_POOL:
                candidates.extend(self.STRATEGY_POOL[rel_name])

        # (B) 被动关系
        for rel_id in spatial_col:
            rel = int(rel_id)
            if rel == 0: continue
            # 逻辑反转
            if rel == self.REL_IDS['above']: candidates.extend(self.STRATEGY_POOL['below'])
            elif rel == self.REL_IDS['below']: candidates.extend(self.STRATEGY_POOL['above'])
            elif rel == self.REL_IDS['inside']: candidates.extend(self.STRATEGY_POOL['surrounds']) 

        # (C) 兜底策略
        if not candidates:
            candidates = ['center_focus', 'wing_left', 'wing_right', 'ground_center']

        # 2. 计算代价 (Soft Overlap Cost)
        unique_candidates = []
        [unique_candidates.append(x) for x in candidates if x not in unique_candidates]
        
        candidate_scores = [] # List of (cost, pattern)
        
        for pat_name in unique_candidates:
            pattern = self.patterns[pat_name]
            
            # 计算软重叠代价: sum(Pattern * Occupancy)
            # 值越大说明重叠越严重
            overlap = (pattern * current_occupancy).sum()
            cost = overlap.item()
            
            candidate_scores.append((cost, pattern))
            
        # 3. 选择策略
        # Cost 越小越好
        candidate_scores.sort(key=lambda x: x[0]) 
        
        best_signal = None
        if not candidate_scores:
            best_signal = self.patterns['center_focus']
        else:
            if mode == 'greedy':
                # 选 Cost 最小的
                best_signal = candidate_scores[0][1]
            elif mode == 'sample':
                # [Diversity] 依概率采样
                k = min(len(candidate_scores), top_k)
                opts = candidate_scores[:k]
                
                # 将 Cost 转换为概率: Prob ~ exp(-Cost * T)
                # T (Temperature) = 5.0 用来放大差异，让低 Cost 的选项概率显著更高
                costs = torch.tensor([x[0] for x in opts], dtype=torch.float32)
                probs = F.softmax(-costs * 5.0, dim=0) 
                
                try:
                    # 按概率采样一个索引
                    idx = torch.multinomial(probs, 1).item()
                    best_signal = opts[idx][1]
                except Exception:
                    best_signal = opts[0][1]

        # 4. 更新状态
        # 累积热力图，表示"这里越来越挤了"
        new_occupancy = current_occupancy + best_signal
        
        # 可选: 归一化当前信号，确保它是标准的分布
        if best_signal.max() > 0:
            best_signal = best_signal / best_signal.max()
            
        return best_signal, new_occupancy

if __name__ == "__main__":
    # 简单测试
    gen = LocationSignalGenerator(grid_size=8)
    occ = torch.zeros((8, 8))
    
    print("Test: Object 1 (Above) -> Sky")
    sig1, occ = gen.infer_stateful_signal(0, [1], [0], occ, mode='greedy')
    
    # [新增] 可视化输出
    if draw_gaussian_heatmap:
        draw_gaussian_heatmap(
            sig1.numpy(), 
            title="Sky Signal (Above)", 
            output_path="outputs/debug_sky_signal.png"
        )
    
    print("Test: Object 2 (Above) -> Sky (Should avoid obj 1 overlap)")
    sig2, occ = gen.infer_stateful_signal(1, [1], [0], occ, mode='greedy')
    
    # [新增] 可视化输出
    if draw_gaussian_heatmap:
        draw_gaussian_heatmap(
            sig2.numpy(), 
            title="Sky Signal 2 (Avoided)", 
            output_path="outputs/debug_sky_avoided.png"
        )
    
    # 简单的控制台打印 (保留作为备用)
    def print_grid(grid):
        for row in grid:
            # 打印数值，保留1位小数，0则显示点
            print(" ".join([f"{x:.1f}" if x>0.1 else " . " for x in row]))
            
    print("\nSignal 1 (Sky Left):")
    print_grid(sig1)
    print("\nSignal 2 (Sky Right - avoided left):")
    print_grid(sig2)