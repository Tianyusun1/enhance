# File: stage2_generation/scripts/prepare_data_taiyi.py (V15.1: Path Fix - Ignore Visualized Boxes)

import sys
import os
import argparse
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2 

# ==========================================
# 1. 路径与环境设置
# ==========================================
current_file_path = os.path.abspath(__file__)
stage2_root = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(stage2_root)
if project_root not in sys.path: sys.path.insert(0, project_root)
if stage2_root not in sys.path: sys.path.append(stage2_root)

try:
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
    print("✅ 成功导入 InkWashMaskGenerator (V10.1)")
except ImportError:
    print("❌ 无法导入 InkWashMaskGenerator，请检查路径。")
    sys.exit(1)

# ==========================================
# 2. 视觉态势提取器
# ==========================================
class FixedVisualGestaltExtractor:
    def extract(self, image_path: str, box: list) -> tuple:
        try:
            if not os.path.exists(image_path):
                return [0.0, 0.0, 0.0, 0.0], 0.0
            
            try:
                img_array = np.fromfile(image_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            except Exception:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None: return [0.0, 0.0, 0.0, 0.0], 0.0
                
            H, W = img.shape
            cx, cy, w, h = box
            
            x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            if (x2 - x1) < 2 or (y2 - y1) < 2: return [0.0, 0.0, 0.0, 0.0], 0.0
                
            crop = img[y1:y2, x1:x2]
            
            ink_map = 255.0 - crop.astype(float)
            ink_map[ink_map < 30] = 0 
            total_ink = np.sum(ink_map)
            if total_ink < 100: return [0.0, 0.0, 0.0, 0.0], 0.0

            M = cv2.moments(ink_map.astype(np.float32), binaryImage=False)
            bias_x, bias_y, rotation = 0.0, 0.0, 0.0
            
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                h_crop, w_crop = ink_map.shape
                bias_x = (cX - w_crop/2.0) / (w_crop/2.0 + 1e-6)
                bias_y = (cY - h_crop/2.0) / (h_crop/2.0 + 1e-6)
                mu20 = M["mu20"] / M["m00"]
                mu02 = M["mu02"] / M["m00"]
                mu11 = M["mu11"] / M["m00"]
                theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
                rotation = theta / (np.pi / 2) 
            
            avg_density = total_ink / (crop.shape[0] * crop.shape[1] * 255.0)
            sobelx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            avg_grad = np.mean(grad_mag) / 255.0 
            
            raw_flow = avg_density / (avg_grad + 0.01)
            
            pivot = 0.6
            if raw_flow > pivot:
                flow = (raw_flow - pivot) / (3.0 - pivot + 1e-6)
                flow = np.clip(flow, 0.05, 1.0) 
            else:
                flow = (raw_flow - pivot) / pivot
                flow = np.clip(flow, -1.0, -0.05) 
            
            return [float(np.clip(bias_x, -1, 1)), float(np.clip(bias_y, -1, 1)), float(rotation), float(flow)], 1.0
            
        except Exception as e:
            return [0.0, 0.0, 0.0, 0.0], 0.0

# ==========================================
# 3. 辅助函数
# ==========================================
def generate_soft_energy_field(box_9d, res=64):
    _, cx, cy, bw, bh, bx, by, _, _ = box_9d
    x_c = (cx + bx * 0.15) * res
    y_c = (cy + by * 0.15) * res
    y_grid, x_grid = np.ogrid[:res, :res]
    dist_sq = (x_grid - x_c)**2 + (y_grid - y_c)**2
    sigma = ((bw * res + bh * res) / 4.0) + 1e-6
    field = np.exp(-dist_sq / (2 * sigma**2))
    return field.astype(np.float32)

def parse_args():
    parser = argparse.ArgumentParser(description="Taiyi V15.1: 修复路径索引问题")
    
    # [FIX] 默认路径调整
    default_xlsx = "/home/610-sty/layout2paint/dataset/6800poems.xlsx"
    # 这里建议直接指到 6800 目录，避免扫描太广
    default_img_dir = "/home/610-sty/layout2paint/dataset/6800" 
    default_lbl_dir = "/home/610-sty/layout2paint/dataset/6800/JPEGImages-pre_new_txt"
    
    parser.add_argument("--xlsx_path", type=str, default=default_xlsx)
    parser.add_argument("--images_dir", type=str, default=default_img_dir)
    parser.add_argument("--labels_dir", type=str, default=default_lbl_dir)
    parser.add_argument("--output_dir", type=str, default="./taiyi_energy_dataset_v9_2") 
    parser.add_argument("--resolution", type=int, default=512) 
    return parser.parse_args()

# ==========================================
# 4. 主程序
# ==========================================
def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "conditioning_images"), exist_ok=True)
    
    ink_generator = InkWashMaskGenerator(width=args.resolution, height=args.resolution)
    gestalt_extractor = FixedVisualGestaltExtractor()
    
    print("=====================================================")
    print("🚀 V15.1 数据准备: 排除 'out_visualize' 脏数据")
    print(f"📂 图片扫描目录: {args.images_dir}")
    print("=====================================================")

    # [核心修复] 建立图片索引 (带过滤)
    print(f"🔍 正在扫描建立索引...")
    image_index = {}
    scan_count = 0
    ignored_count = 0
    
    for root, dirs, files in os.walk(args.images_dir):
        # [关键过滤逻辑] 如果路径中包含 out_visualize 或 visualize，直接跳过
        if "out_visualize" in root or "visualize" in root:
            # print(f"🚫 忽略脏数据目录: {root}") # 调试用，太长可注释
            ignored_count += len(files)
            continue
            
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                # 存入索引
                # 注意：如果文件名重复，这里还是会覆盖，但因为我们排除了脏目录，
                # 理论上只剩下了干净的原图目录。
                image_index[file] = os.path.join(root, file)
                scan_count += 1
                
    print(f"✅ 索引完成: 有效图片 {len(image_index)} 张 (已剔除 {ignored_count} 张带框脏图)")

    df = pd.read_excel(args.xlsx_path)
    metadata_entries = []
    success_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            raw_img_name = str(row['image']).strip()
            poem = str(row['poem']).strip()
            
            src_img_path = None
            if os.path.isabs(raw_img_name) and os.path.exists(raw_img_name):
                # 如果是绝对路径，检查是否包含脏字眼
                if "out_visualize" in raw_img_name:
                    continue
                src_img_path = raw_img_name
            else:
                basename = os.path.basename(raw_img_name)
                src_img_path = image_index.get(basename)
            
            if src_img_path is None: continue

            # 二次确认：确保找到的路径不是脏数据
            if "out_visualize" in src_img_path:
                print(f"⚠️ 警告: 依然捕获到脏数据 {src_img_path}，跳过")
                continue

            img_stem = Path(src_img_path).stem
            label_path = os.path.join(args.labels_dir, f"{img_stem}.txt")
            if not os.path.exists(label_path): continue

            # 读取 Layout
            boxes_9d = [] 
            energy_masks_info = [] 
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5: 
                        cls_id, cx, cy, w, h = map(float, parts[:5])
                        g_params, valid = gestalt_extractor.extract(src_img_path, [cx, cy, w, h])
                        if valid < 0.5: g_params = [0.0, 0.0, 0.0, 0.5] 
                        full_box = [cls_id, cx, cy, w, h] + g_params
                        boxes_9d.append(full_box)
                        soft_mask = generate_soft_energy_field(full_box, res=64)
                        energy_masks_info.append({
                            "class_id": int(cls_id),
                            "mask_data": soft_mask.tolist()
                        })
            
            if not boxes_9d: continue

            # 生成 Conditioning Image (墨块)
            cond_img = ink_generator.convert_boxes_to_mask(boxes_9d) 
            cond_img_name = f"{img_stem}_ink.png"
            cond_img.save(os.path.join(args.output_dir, "conditioning_images", cond_img_name))
            
            # 处理 Target Image (原图)
            img_array = np.fromfile(src_img_path, dtype=np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_cv is None: continue
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            target_img = Image.fromarray(img_rgb).resize((args.resolution, args.resolution), Image.BICUBIC)
            
            target_img_name = f"{img_stem}.jpg"
            target_img.save(os.path.join(args.output_dir, "images", target_img_name))

            metadata_entries.append({
                "image": f"images/{target_img_name}",
                "conditioning_image": f"conditioning_images/{cond_img_name}",
                "text": poem,
                "layout_energy": energy_masks_info 
            })
            
            success_count += 1
            
        except Exception as e:
            continue

    output_jsonl = os.path.join(args.output_dir, "train.jsonl")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"✨ V15.1 数据清洗完毕！有效样本: {success_count}")
    print(f"👉 请立即去 {os.path.join(args.output_dir, 'images')} 检查图片是否还有方框！")

if __name__ == "__main__":
    main()