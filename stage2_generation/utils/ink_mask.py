# File: stage2_generation/utils/ink_mask.py (V9.9: Artifacts Fixed & Smooth Edge)

import numpy as np
from PIL import Image, ImageFilter
import torch
from typing import List, Union
import math
import cv2 

class InkWashMaskGenerator:
    """
    [V9.9 修复版] 纹理化势能场生成器
    
    修复记录:
    1. 收紧高斯核 (Divisor 2.5 -> 4.2)，防止墨迹填满整个框导致边缘生硬。
    2. 引入边缘破碎逻辑 (Edge Breakup)，消除人工合成的“方框感”。
    3. 优化枯笔纹理混合算法，防止画面出现黑色噪点空洞。
    4. 增加最小尺寸限制，防止小物体消失。
    """
    
    CLASS_COLORS = {
        2: (255, 0, 0),   # 山 (Red)
        3: (0, 0, 255),   # 水 (Blue)
        4: (0, 255, 255), # 人 (Cyan)
        5: (0, 255, 0),   # 树 (Green)
        6: (255, 255, 0), # 建筑 (Yellow)
        7: (255, 0, 255), # 桥 (Bridge)
        8: (128, 0, 128), # 花 (Purple)
        9: (255, 165, 0), # 鸟 (Orange)
        10: (165, 42, 42) # 兽 (Brown)
    }
    
    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        # 预计算网格
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)
        self.y_grid, self.x_grid = np.meshgrid(y, x, indexing='ij')
        
    def _generate_rotated_gaussian(self, box: np.ndarray) -> np.ndarray:
        class_id = int(box[0])
        cx, cy, w, h = box[1], box[2], box[3], box[4]
        
        # 提取 V8 态势参数
        if len(box) >= 9:
            bx, by = box[5], box[6]
            rot = box[7]
            flow = box[8] # [-1, 1] 枯湿程度
        else:
            bx, by, rot, flow = 0.0, 0.0, 0.0, 0.0

        # 1. 基础几何计算
        center_x = (cx + bx * 0.15) * self.width
        center_y = (cy + by * 0.15) * self.height
        
        # [Fix 1] 收紧核心: 从 2.5 改为 4.2
        # 让高斯分布集中在框的中心，确保边缘处数值自然衰减到 0
        flow_scale = 1.0 + flow * 0.3 
        sigma_x = (w * self.width / 4.2) * flow_scale
        sigma_y = (h * self.height / 4.2) * flow_scale
        
        theta = rot * (math.pi / 2.0)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        dx = self.x_grid - center_x
        dy = self.y_grid - center_y
        dx_rot = dx * cos_t + dy * sin_t
        dy_rot = -dx * sin_t + dy * cos_t
        
        exponent = - (dx_rot**2 / (2 * sigma_x**2 + 1e-6) + 
                      dy_rot**2 / (2 * sigma_y**2 + 1e-6))
        
        # 基础场 (0~1)
        field = np.exp(np.clip(exponent, -20, 0))
        
        # [Fix 2] 消除“方框”伪影 (The Box Artifact Killer)
        # 策略：线性移位 + 边缘随机打碎
        
        # A. 线性移位：切掉底部的 15% 拖尾，让边缘更干脆，但保持内部平滑
        field = np.maximum(0, field - 0.15)
        # 重新归一化，保证中心还是最黑的
        if field.max() > 0:
            field = field / field.max()

        # B. 边缘破碎 (Edge Breakup)：只在边缘区域(0.01~0.5) 随机扣除像素
        # 这模拟了宣纸边缘的随机渗透，打破了计算机生成的完美几何感
        edge_mask = (field > 0.01) & (field < 0.5)
        if edge_mask.any():
            edge_noise = np.random.uniform(0.0, 1.0, field.shape)
            # 约 60% 的概率保留边缘像素，40% 的概率扣掉，制造“毛边”
            field[edge_mask] *= (edge_noise[edge_mask] > 0.4).astype(np.float32)

        # 2. 纹理注入
        field = self._apply_texture(field, flow)
        
        # 3. 上色
        color = self.CLASS_COLORS.get(class_id, (128, 128, 128))
        colored_field = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for i in range(3):
            colored_field[:, :, i] = field * (color[i] / 255.0)
            
        return colored_field

    def _apply_texture(self, field: np.ndarray, flow: float) -> np.ndarray:
        """根据 flow 值给高斯场添加水墨纹理"""
        # 保护：如果场太弱，不处理
        if field.max() < 0.05: return field

        noise = np.random.uniform(0, 1, field.shape).astype(np.float32)
        
        if flow < 0: 
            # === 枯笔 (Dry Mode) ===
            dryness = abs(flow) # 0~1
            # [Fix 3] 降低阈值，防止画面变成全是噪点
            # 限制最大阈值，保证物体主体可见
            threshold = 0.1 + 0.25 * dryness
            
            texture = (noise > threshold).astype(np.float32)
            
            # [Fix 4] 使用混合而非直接相乘，保留物体骨架
            # field = field * texture  <-- OLD (会产生黑色空洞)
            # NEW: 即使是噪点，也保留 30% 的底色
            field = field * (0.3 + 0.7 * texture)
            
            # 适度锐化，模拟燥锋
            field = np.power(field, 0.9)
            
        else:
            # === 湿笔 (Wet Mode) ===
            wetness = flow # 0~1
            k_size = int(5 * wetness) * 2 + 1
            if k_size > 1:
                blurred = cv2.GaussianBlur(field, (k_size, k_size), 0)
                # 湿笔要更柔和
                field = 0.4 * field + 0.6 * blurred
            
            # 纸张纹理
            texture = 1.0 - 0.15 * wetness * noise
            field = field * texture

        return np.clip(field, 0, 1)

    def convert_boxes_to_mask(self, boxes: Union[List[List[float]], torch.Tensor]) -> Image.Image:
        full_canvas = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        # [Fix 5] 增加最小尺寸限制，防止小物体消失
        valid_boxes = []
        for b in boxes:
            if len(b) >= 5 and b[3] > 0 and b[4] > 0:
                # 强制最小宽度/高度为 5%
                b[3] = max(b[3], 0.05)
                b[4] = max(b[4], 0.05)
                valid_boxes.append(b)
                
        # 按面积排序，大的在下
        sorted_boxes = sorted(valid_boxes, key=lambda b: b[3]*b[4], reverse=True)
            
        for box in sorted_boxes:
            class_id = int(box[0])
            if class_id < 2 or class_id > 10: continue
            
            object_field = self._generate_rotated_gaussian(box)
            
            # 积墨融合
            alpha = np.max(object_field, axis=2, keepdims=True)
            # 适当增强 Alpha，但不要太强，保留水墨的半透明叠加感
            alpha = np.clip(alpha * 1.5, 0, 1)
            
            full_canvas = full_canvas * (1 - alpha) + object_field * alpha

        full_canvas = np.clip(full_canvas * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(full_canvas, mode='RGB')