# File: stage2_generation/utils/ink_mask.py (V10.0: Organic Ink & Physics Texture)

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from typing import List, Union
import math
import random
import cv2 

class InkWashMaskGenerator:
    """
    [V10.0 终极融合版] 有机墨迹生成器
    
    核心特性:
    1. Polygon Distortion: 将矩形框转化为不规则多边形，彻底消除人工方框感。
    2. Distance Gradient: 利用距离变换模拟墨水从中心向边缘的自然衰减。
    3. Physics Texture: 保留 V9.9 的枯湿笔触物理模拟。
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
        
    def _rotate_point(self, px, py, cx, cy, angle):
        """绕中心点旋转坐标"""
        theta = angle * math.pi / 2.0  # 假设 angle 是 [0,1] 对应 90度，或者根据你的定义调整
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        nx = cos_t * (px - cx) - sin_t * (py - cy) + cx
        ny = sin_t * (px - cx) + cos_t * (py - cy) + cy
        return nx, ny

    def _distort_box(self, x, y, w, h, rot=0.0, roughness=0.2):
        """
        生成不规则多边形顶点，模拟手绘/墨迹边缘
        """
        points = []
        segments = 8  # 每条边的细分段数
        
        # 矩形四个角点 (未旋转)
        cx, cy = x + w/2, y + h/2
        tl, tr = (x, y), (x + w, y)
        br, bl = (x + w, y + h), (x, y + h)
        
        def get_line_points(start, end):
            res = []
            vx, vy = end[0] - start[0], end[1] - start[1]
            length = math.sqrt(vx**2 + vy**2)
            for i in range(segments):
                alpha = i / segments
                # 基础线性插值
                px = start[0] + vx * alpha
                py = start[1] + vy * alpha
                
                # 垂直方向的随机扰动 (Perpendicular Noise)
                # 只有中间段抖动大，角点抖动小
                noise_scale = math.sin(alpha * math.pi) * roughness * max(w, h) * 0.5
                perp_x, perp_y = -vy, vx
                # 归一化垂直向量
                norm = math.sqrt(perp_x**2 + perp_y**2) + 1e-6
                perp_x /= norm
                perp_y /= norm
                
                noise = (random.random() - 0.5) * 2 * noise_scale
                px += perp_x * noise
                py += perp_y * noise
                
                # 旋转
                if rot != 0:
                    px, py = self._rotate_point(px, py, cx, cy, rot)
                
                res.append((px, py))
            return res

        points.extend(get_line_points(tl, tr))
        points.extend(get_line_points(tr, br))
        points.extend(get_line_points(br, bl))
        points.extend(get_line_points(bl, tl))
        
        return points

    def _apply_texture(self, field: np.ndarray, flow: float) -> np.ndarray:
        """
        [V9.9 保留逻辑] 根据 flow 值给场添加水墨纹理
        """
        if field.max() < 0.05: return field

        noise = np.random.uniform(0, 1, field.shape).astype(np.float32)
        
        if flow < 0: 
            # === 枯笔 (Dry Mode) ===
            dryness = abs(flow)
            threshold = 0.1 + 0.3 * dryness
            texture = (noise > threshold).astype(np.float32)
            # 混合：保留部分底色，防止死黑空洞
            field = field * (0.4 + 0.6 * texture)
            field = np.power(field, 0.8) # 略微锐化
            
        else:
            # === 湿笔 (Wet Mode) ===
            wetness = flow
            # 内部高斯模糊，模拟宣纸晕染
            k_size = int(7 * wetness) * 2 + 1
            if k_size > 1:
                blurred = cv2.GaussianBlur(field, (k_size, k_size), 0)
                field = 0.3 * field + 0.7 * blurred
            
            # 纸张纤维纹理
            texture = 1.0 - 0.2 * wetness * noise
            field = field * texture

        return np.clip(field, 0, 1)

    def convert_boxes_to_mask(self, boxes: Union[List[List[float]], torch.Tensor]) -> Image.Image:
        # 画布初始化
        full_canvas = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        # 过滤与排序
        valid_boxes = []
        for b in boxes:
            if len(b) < 5: continue
            # 强制最小尺寸，防止物体过小消失
            b[3] = max(b[3], 0.06) 
            b[4] = max(b[4], 0.06)
            valid_boxes.append(b)
        
        # 按面积从大到小排序 (背景物体先画)
        sorted_boxes = sorted(valid_boxes, key=lambda b: b[3]*b[4], reverse=True)
            
        for box in sorted_boxes:
            class_id = int(box[0])
            if class_id not in self.CLASS_COLORS: continue
            
            # 解析参数
            cx, cy, w, h = box[1], box[2], box[3], box[4]
            bx, by = (box[5], box[6]) if len(box) >= 7 else (0, 0)
            rot = box[7] if len(box) >= 8 else 0.0
            flow = box[8] if len(box) >= 9 else 0.0 # [-1, 1]
            
            # 转换坐标
            pixel_x = (cx - w/2) * self.width
            pixel_y = (cy - h/2) * self.height
            pixel_w = w * self.width
            pixel_h = h * self.height
            
            # 1. 生成不规则多边形 Mask
            poly_points = self._distort_box(pixel_x, pixel_y, pixel_w, pixel_h, rot, roughness=0.25)
            
            # 使用 PIL 绘制单通道 Mask
            temp_img = Image.new('L', (self.width, self.height), 0)
            ImageDraw.Draw(temp_img).polygon(poly_points, outline=255, fill=255)
            mask_np = np.array(temp_img).astype(np.uint8)
            
            # 2. 距离变换 (Distance Transform) -> 模拟墨水浓度
            # 计算每个像素到背景的距离，形成从中心向外衰减的梯度
            dist_map = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
            max_val = dist_map.max()
            if max_val > 0:
                field = dist_map / max_val
            else:
                continue
            
            # 3. 应用纹理 (枯/湿)
            field = self._apply_texture(field, flow)
            
            # 4. 上色与叠加
            color = self.CLASS_COLORS[class_id]
            object_layer = np.zeros_like(full_canvas)
            for c in range(3):
                object_layer[:, :, c] = field * (color[c] / 255.0)
            
            # Alpha 混合：使用 field 作为 alpha
            # 增强一点 alpha 让主体更实，边缘更虚
            alpha = np.clip(field * 1.8, 0, 1)
            alpha = np.expand_dims(alpha, axis=2)
            
            full_canvas = full_canvas * (1 - alpha) + object_layer * alpha

        # [Final Polish] 全局高斯模糊
        # 这一步至关重要，它将所有图层的边缘融合，彻底消除"贴图感"
        final_img = Image.fromarray(np.clip(full_canvas * 255, 0, 255).astype(np.uint8))
        final_img = final_img.filter(ImageFilter.GaussianBlur(radius=3))
        
        return final_img