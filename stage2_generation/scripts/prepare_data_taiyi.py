# File: stage2_generation/scripts/prepare_data_taiyi.py (V9.2: Gestalt Energy Field & Robust Path Edition)

import sys
import os
import argparse
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2  # 引入 OpenCV 进行形态学计算

# === 路径设置 (保持原有逻辑) ===
current_file_path = os.path.abspath(__file__)
stage2_root = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(stage2_root)
if project_root not in sys.path: sys.path.insert(0, project_root)
if stage2_root not in sys.path: sys.path.append(stage2_root)

# 导入工具
try:
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
except ImportError:
    print("❌ 无法导入 InkWashMaskGenerator，请检查路径。")
    sys.exit(1)

# === [V9.1 修复版] 视觉态势提取器 (完整保留) ===
class FixedVisualGestaltExtractor:
    """
    [V9.1 修复版] 视觉态势提取器
    1. 修正 Flow 截断逻辑，支持负值(枯笔)。
    2. 支持中文路径读取 (cv2.imdecode)。
    """
    def extract(self, image_path: str, box: list) -> tuple:
        """
        输入: 全图路径, 归一化 Box [cx, cy, w, h]
        输出: ([bias_x, bias_y, rotation, flow], validity)
        """
        try:
            # 1. 安全性检查
            if not os.path.exists(image_path):
                return [0.0, 0.0, 0.0, 0.0], 0.0
            
            # [关键修复] 读取灰度图 (支持中文路径)
            try:
                img_array = np.fromfile(image_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            except Exception:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                return [0.0, 0.0, 0.0, 0.0], 0.0
                
            H, W = img.shape
            cx, cy, w, h = box
            
            # 2. 裁切物体 (Crop)
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                return [0.0, 0.0, 0.0, 0.0], 0.0
                
            crop = img[y1:y2, x1:x2]
            
            # 3. 水墨预处理
            ink_map = 255.0 - crop.astype(float)
            ink_map[ink_map < 30] = 0 
            
            total_ink = np.sum(ink_map)
            if total_ink < 100: 
                return [0.0, 0.0, 0.0, 0.0], 0.0

            # === A. 计算 Bias & Rotation ===
            M = cv2.moments(ink_map.astype(np.float32), binaryImage=False)
            
            bias_x, bias_y = 0.0, 0.0
            rotation = 0.0
            
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                h_crop, w_crop = ink_map.shape
                geo_cX = w_crop / 2.0
                geo_cY = h_crop / 2.0
                
                bias_x = (cX - geo_cX) / (geo_cX + 1e-6)
                bias_y = (cY - geo_cY) / (geo_cY + 1e-6)
                bias_x = np.clip(bias_x, -1.0, 1.0)
                bias_y = np.clip(bias_y, -1.0, 1.0)
                
                mu20 = M["mu20"] / M["m00"]
                mu02 = M["mu02"] / M["m00"]
                mu11 = M["mu11"] / M["m00"]
                theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
                rotation = theta / (np.pi / 2)
            
            # === B. 计算 Flow (支持负值枯笔) ===
            h_crop, w_crop = ink_map.shape
            avg_density = total_ink / (w_crop * h_crop * 255.0)
            
            sobelx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            avg_grad = np.mean(grad_mag) / 255.0 
            
            raw_flow = avg_density / (avg_grad + 0.01)
            
            # [核心映射逻辑] Pivot = 0.6
            pivot = 0.6
            if raw_flow > pivot:
                flow = (raw_flow - pivot) / (3.0 - pivot + 1e-6)
                flow = np.clip(flow, 0.05, 1.0)
            else:
                flow = (raw_flow - pivot) / pivot
                flow = np.clip(flow, -1.0, -0.05)
            
            return [float(bias_x), float(bias_y), float(rotation), float(flow)], 1.0
            
        except Exception as e:
            return [0.0, 0.0, 0.0, 0.0], 0.0

# === [NEW V9.2] 软能量场生成器：确保训练与推理逻辑对齐 ===
def generate_soft_energy_field(box_9d, res=64):
    """
    根据态势参数生成 64x64 的高斯软能量掩码。
    box_9d: [cls_id, cx, cy, w, h, bx, by, rot, flow]
    """
    _, cx, cy, bw, bh, bx, by, _, _ = box_9d
    
    # 1. 计算与推理端 PoemInkAttentionProcessor 绝对一致的中心
    # 使用 0.15 偏移系数
    x_c = (cx + bx * 0.15) * res
    y_c = (cy + by * 0.15) * res
    
    # 2. 生成坐标网格
    y_grid, x_grid = np.ogrid[:res, :res]
    dist_sq = (x_grid - x_c)**2 + (y_grid - y_c)**2
    
    # 3. 计算衰减标准差 (基于物体尺寸，/4 确保场强集中)
    sigma = ((bw * res + bh * res) / 4.0) + 1e-6
    
    # 4. 生成高斯分布
    field = np.exp(-dist_sq / (2 * sigma**2))
    return field.astype(np.float32)

def parse_args():
    parser = argparse.ArgumentParser(description="Taiyi V9.2: 准备包含态势能量场的训练数据集")
    default_xlsx = "/home/610-sty/layout2paint/dataset/6800poems.xlsx"
    default_img_dir = "/home/610-sty/layout2paint/dataset" 
    default_lbl_dir = "/home/610-sty/layout2paint/dataset/6800/JPEGImages-pre_new_txt"
    
    parser.add_argument("--xlsx_path", type=str, default=default_xlsx)
    parser.add_argument("--images_dir", type=str, default=default_img_dir)
    parser.add_argument("--labels_dir", type=str, default=default_lbl_dir)
    parser.add_argument("--output_dir", type=str, default="./taiyi_energy_dataset_v9_2") 
    parser.add_argument("--resolution", type=int, default=512) 
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "conditioning_images"), exist_ok=True)
    
    # 1. 初始化组件
    ink_generator = InkWashMaskGenerator(width=args.resolution, height=args.resolution)
    gestalt_extractor = FixedVisualGestaltExtractor()
    print("✅ V9.2 Components (Gestalt Extractor & Ink Generator) initialized.")
    
    # 2. [V9.1 保留逻辑] 全局图片索引扫描
    print(f"🔍 正在扫描图片目录建立索引: {args.images_dir} ...")
    image_index = {}
    for root, dirs, files in os.walk(args.images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_index[file] = os.path.join(root, file)
    print(f"✅ 索引建立完成。共找到 {len(image_index)} 张图片。")

    df = pd.read_excel(args.xlsx_path)
    metadata_entries = []
    success_count = 0

    print(f"🚀 开始处理数据，共 {len(df)} 条...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            raw_img_name = str(row['image']).strip()
            poem = str(row['poem']).strip()
            
            # --- [V9.1 保留逻辑] 智能路径查找 ---
            src_img_path = None
            if os.path.isabs(raw_img_name) and os.path.exists(raw_img_name):
                src_img_path = raw_img_name
            else:
                basename = os.path.basename(raw_img_name)
                src_img_path = image_index.get(basename)
            
            if src_img_path is None: continue

            img_stem = Path(src_img_path).stem
            label_path = os.path.join(args.labels_dir, f"{img_stem}.txt")
            if not os.path.exists(label_path): continue

            # 3. 读取 Box 并提取真实态势
            boxes_9d = [] 
            energy_masks_info = [] # [V9.2] 存储软能量掩码数据
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5: 
                        cls_id, cx, cy, w, h = map(float, parts[:5])
                        
                        # 提取 Flow (包含负值)
                        g_params, valid = gestalt_extractor.extract(src_img_path, [cx, cy, w, h])
                        
                        # 失败处理
                        if valid < 0.5:
                            g_params = [0.0, 0.0, 0.0, 0.5] 
                            
                        full_box = [cls_id, cx, cy, w, h] + g_params
                        boxes_9d.append(full_box)
                        
                        # [V9.2 核心] 生成 64x64 的高斯软能量场
                        # 对应训练端 cross-attention 的空间分辨率
                        soft_mask = generate_soft_energy_field(full_box, res=64)
                        energy_masks_info.append({
                            "class_id": int(cls_id),
                            "mask_data": soft_mask.tolist() # 保存为 list 以序列化到 JSON
                        })
            
            if not boxes_9d: continue

            # 4. 生成渲染 Mask (用于 ControlNet)
            cond_img = ink_generator.convert_boxes_to_mask(boxes_9d) 
            cond_img_name = f"{img_stem}_ink_v9.png"
            cond_img.save(os.path.join(args.output_dir, "conditioning_images", cond_img_name))
            
            # 5. 处理原图 (支持中文路径加载)
            img_array = np.fromfile(src_img_path, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_cv is None: continue
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            target_img = Image.fromarray(img_rgb).resize((args.resolution, args.resolution), Image.BICUBIC)
            
            target_img_name = f"{img_stem}.jpg"
            target_img.save(os.path.join(args.output_dir, "images", target_img_name))

            # 6. [V9.2 升级] 构造元数据，包含 layout_energy 字段
            metadata_entries.append({
                "image": f"images/{target_img_name}",
                "conditioning_image": f"conditioning_images/{cond_img_name}",
                "text": poem,
                "layout_energy": energy_masks_info # <--- 训练脚本 train_taiyi.py 必需字段
            })
            
            success_count += 1
            
        except Exception as e:
            continue

    # 保存 JSONL
    output_jsonl = os.path.join(args.output_dir, "train.jsonl")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"✨ V9.2 能量场数据集准备完成！成功处理: {success_count} 张")
    print(f"📂 输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()