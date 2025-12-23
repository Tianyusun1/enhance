# File: stage2_generation/utils/ink_mask.py (V8.6: Textured Brush Mask)

import numpy as np
from PIL import Image, ImageFilter
import torch
from typing import List, Union
import math
import cv2  # [NEW] 需要 opencv-python 支持高斯模糊

class InkWashMaskGenerator:
    """
    [V8.6 核心组件] 纹理化势能场生成器
    
    不仅生成位置(Gaussian Blob)，还根据 flow (枯湿) 参数生成物理纹理：
    - Flow < 0 (枯): 生成高频噪点，模拟飞白、燥锋。
    - Flow > 0 (湿): 生成水渍晕染，边缘柔和。
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
        # 物理中心偏移 (Bias Shift)
        center_x = (cx + bx * 0.15) * self.width
        center_y = (cy + by * 0.15) * self.height
        
        # 湿笔晕得开，枯笔收得紧
        flow_scale = 1.0 + flow * 0.5 
        sigma_x = (w * self.width / 2.5) * flow_scale
        sigma_y = (h * self.height / 2.5) * flow_scale
        
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
        
        # 2. [关键] 纹理注入 (Texturize)
        # 这里是将“势能参数”转化为“视觉纹理”的关键步骤
        field = self._apply_texture(field, flow)
        
        # 3. 上色
        color = self.CLASS_COLORS.get(class_id, (128, 128, 128))
        colored_field = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for i in range(3):
            colored_field[:, :, i] = field * (color[i] / 255.0)
            
        return colored_field

    def _apply_texture(self, field: np.ndarray, flow: float) -> np.ndarray:
        """根据 flow 值给高斯场添加水墨纹理"""
        # 生成随机噪声底板
        noise = np.random.uniform(0, 1, field.shape).astype(np.float32)
        
        if flow < 0: 
            # === 枯笔 (Dry Mode) ===
            dryness = abs(flow) # 0~1
            # 模拟飞白：利用阈值截断，制造空洞
            # dryness 越大，空洞越多，纹理越破碎
            threshold = 0.2 + 0.3 * dryness
            texture = (noise > threshold).astype(np.float32)
            
            # 边缘锐化：枯笔边缘应该是硬的，且带有断断续续的感觉
            field = field * texture
            # 增强对比度，让墨色更实，背景更白
            field = np.power(field, 0.8)
            
        else:
            # === 湿笔 (Wet Mode) ===
            wetness = flow # 0~1
            # 模拟晕染：叠加多层模糊
            # 湿润度越高，模糊核越大，边缘越柔和
            k_size = int(7 * wetness) * 2 + 1
            if k_size > 1:
                # 只是对场进行模糊，不破坏整体形状
                blurred = cv2.GaussianBlur(field, (k_size, k_size), 0)
                # 混合原场和模糊场，制造水渍扩散感 (中心实，边缘虚)
                field = 0.4 * field + 0.6 * blurred
            
            # 湿笔也会有一些纸张纹理，但比较细腻
            texture = 1.0 - 0.2 * wetness * noise
            field = field * texture

        return np.clip(field, 0, 1)

    def convert_boxes_to_mask(self, boxes: Union[List[List[float]], torch.Tensor]) -> Image.Image:
        full_canvas = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
            
        valid_boxes = [b for b in boxes if len(b) >= 5 and b[3] > 0 and b[4] > 0]
        # 按面积排序，大的在下，小的在上 (Painter's Algorithm)
        sorted_boxes = sorted(valid_boxes, key=lambda b: b[3]*b[4], reverse=True)
            
        for box in sorted_boxes:
            class_id = int(box[0])
            if class_id < 2 or class_id > 10: continue
            
            # 生成单个意象的动态纹理场
            object_field = self._generate_rotated_gaussian(box)
            
            # 积墨融合 (Alpha Blending)
            alpha = np.max(object_field, axis=2, keepdims=True)
            # 稍微增强 alpha，保证遮盖力，但保留半透明感
            alpha = np.clip(alpha * 1.8, 0, 1)
            
            full_canvas = full_canvas * (1 - alpha) + object_field * alpha

        full_canvas = np.clip(full_canvas * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(full_canvas, mode='RGB')