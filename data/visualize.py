# File: data/visualize.py (V8.0: Visual Gestalt Compatible)

import os
import math
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
from typing import List, Tuple, Dict, Union

# =============================================================
# [核心同步] 类别颜色定义 (必须与 ink_mask.py 保持完全一致)
# =============================================================
CLASS_COLORS = {
    2: "red",      # mountain (山): 红色
    3: "blue",     # water (水): 蓝色
    4: "cyan",     # people (人): 青色
    5: "green",    # tree (树): 绿色
    6: "yellow",   # building (建筑): 黄色
    7: "magenta",  # bridge (桥): 洋红
    8: "purple",   # flower (花): 紫色
    9: "orange",   # bird (鸟): 橙色
    10: "brown"    # animal (动物): 棕色
}

CLASS_NAMES = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}

def draw_layout(layout_seq: List[Tuple], poem: str, output_path: str, img_size: Tuple[int, int] = (512, 512)):
    """
    绘制带有颜色标注的布局草图。
    [V8.0 Upgrade] 支持可视化 '视觉态势' (Visual Gestalt)：
    - 输入 tuple 长度 >= 9 时，自动绘制重心偏移和旋转方向。
    - 输入格式: (cls_id, cx, cy, w, h, [bias_x, bias_y, rotation, flow])
    """
    try:
        # 创建黑色背景，增强对比度
        img = Image.new('RGB', img_size, (20, 20, 20)) 
        draw = ImageDraw.Draw(img)
        W, H = img_size
        
        try:
            # 尝试加载中文字体，若无则回退
            # 建议系统中有 simhei.ttf 或更改为存在的字体路径
            font = ImageFont.truetype("simhei.ttf", 14) 
        except IOError:
            font = ImageFont.load_default()
            
        for item in layout_seq:
            # [Robustness] 安全检查
            if len(item) < 5: 
                continue
            
            # 1. 基础几何信息
            cls_id = int(item[0])
            cx, cy, w, h = item[1], item[2], item[3], item[4]
            
            # 转换像素坐标
            pixel_cx = cx * W
            pixel_cy = cy * H
            pixel_w = w * W
            pixel_h = h * H
            
            xmin = int(pixel_cx - pixel_w / 2)
            ymin = int(pixel_cy - pixel_h / 2)
            xmax = int(pixel_cx + pixel_w / 2)
            ymax = int(pixel_cy + pixel_h / 2)
            
            color = CLASS_COLORS.get(cls_id, "white")
            cls_name = CLASS_NAMES.get(cls_id, "UNK")
            
            # 绘制包围盒 (BBox)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
            
            # ---------------------------------------------------------
            # [NEW] V8.0 态势可视化逻辑
            # ---------------------------------------------------------
            gestalt_text = ""
            if len(item) >= 9:
                # 解析态势参数
                bias_x, bias_y, rotation, flow = item[5], item[6], item[7], item[8]
                
                # A. 绘制视觉重心 (Visual Center)
                # bias 是相对于半宽/半高的归一化偏移 (-1 ~ 1)
                vc_x = pixel_cx + bias_x * (pixel_w / 2)
                vc_y = pixel_cy + bias_y * (pixel_h / 2)
                
                # 在重心处画一个小实心圆
                r_center = 3
                draw.ellipse(
                    [vc_x - r_center, vc_y - r_center, vc_x + r_center, vc_y + r_center], 
                    fill=color, outline="white"
                )
                
                # B. 绘制旋转/主轴方向 (Rotation)
                # 假设 rotation 在 [-1, 1] 映射到 [-90°, 90°] (即 -pi/2 ~ pi/2)
                angle = rotation * (math.pi / 2)
                line_len = min(pixel_w, pixel_h) * 0.4 # 指示线长度
                
                end_x = pixel_cx + math.sin(angle) * line_len
                end_y = pixel_cy - math.cos(angle) * line_len # 减号因为图像Y轴向下
                
                draw.line([(pixel_cx, pixel_cy), (end_x, end_y)], fill="white", width=1)
                
                # C. 记录 Flow 信息用于文本显示
                gestalt_text = f" | F:{flow:.1f}"

            # ---------------------------------------------------------
            
            # 绘制标签
            label_text = f"{cls_name}{gestalt_text}"
            
            # 文字位置计算 (防止出界)
            text_x = max(0, min(W - 60, xmin))
            text_y = max(0, ymin - 16 if ymin > 20 else ymax + 2)
            
            # 文字背景遮罩 (可选，增加可读性)
            # draw.rectangle([text_x, text_y, text_x + len(label_text)*7, text_y + 14], fill="black")
            draw.text((text_x, text_y), label_text, fill=color, font=font)
        
        # 绘制底部诗句标题
        draw.text((10, H - 25), f"Poem: {poem}", fill="white", font=font)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        # print(f"-> Layout saved to {output_path}")
        
    except Exception as e:
        print(f"[Error in draw_layout]: {e}")

# -------------------------------------------------------------
# 综合热力图绘制函数 (保持不变，用于 Attention 检查)
# -------------------------------------------------------------
def draw_integrated_heatmap(
    layers: List[Tuple[np.ndarray, int]], 
    poem: str, 
    output_path: str, 
    img_size: Tuple[int, int] = (512, 512)
):
    """
    将多个意象的热力图按照语义颜色叠加。
    """
    try:
        W, H = img_size
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        
        for grid, cls_id in layers:
            grid = np.asarray(grid, dtype=np.float32)
            if grid.max() > 0:
                grid = grid / grid.max()
            
            # 上采样至图像尺寸
            pil_grid = Image.fromarray(grid)
            pil_grid = pil_grid.resize(img_size, resample=Image.BILINEAR) 
            grid_large = np.array(pil_grid, dtype=np.float32)
            
            # 获取对应的语义颜色
            color_name = CLASS_COLORS.get(cls_id, "white")
            rgb = ImageColor.getrgb(color_name) 
            
            # 颜色叠加
            for c in range(3):
                canvas[:, :, c] += grid_large * rgb[c]

        # 裁剪并量化
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        img = Image.fromarray(canvas, mode='RGB')
        
        # 绘制网格线
        draw = ImageDraw.Draw(img)
        grid_color = (60, 60, 60)
        num_cells = 8
        for i in range(1, num_cells):
            x = int(i * W / num_cells)
            draw.line([(x, 0), (x, H)], fill=grid_color, width=1)
            y = int(i * H / num_cells)
            draw.line([(0, y), (W, y)], fill=grid_color, width=1)

        # 标题绘制
        text = f"Semantic Heatmap: {poem[:12]}"
        draw.text((10, 10), text, fill="white")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        print(f"  -> Semantic heatmap saved: {output_path}")

    except Exception as e:
        print(f"[Error in draw_integrated_heatmap]: {e}")