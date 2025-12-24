# File: stage2_generation/scripts/prepare_data_taiyi.py (V9.1: Final Robust Edition)

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

# === [CRITICAL CLASS] 本地定义修复版的态势提取器 ===
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
                # 湿润区间 (0, 1]
                flow = (raw_flow - pivot) / (3.0 - pivot + 1e-6)
                flow = np.clip(flow, 0.05, 1.0)
            else:
                # 枯燥区间 [-1, 0)
                flow = (raw_flow - pivot) / pivot
                flow = np.clip(flow, -1.0, -0.05)
            
            return [float(bias_x), float(bias_y), float(rotation), float(flow)], 1.0
            
        except Exception as e:
            return [0.0, 0.0, 0.0, 0.0], 0.0

def parse_args():
    parser = argparse.ArgumentParser(description="Taiyi V9.1: 准备包含枯笔质感的数据集 (强健路径版)")
    # 请根据实际环境确认路径，这里使用了 dataset 的上级目录以便全局扫描
    default_xlsx = "/home/610-sty/layout2paint/dataset/6800poems.xlsx"
    default_img_dir = "/home/610-sty/layout2paint/dataset" 
    default_lbl_dir = "/home/610-sty/layout2paint/dataset/6800/JPEGImages-pre_new_txt"
    
    parser.add_argument("--xlsx_path", type=str, default=default_xlsx)
    parser.add_argument("--images_dir", type=str, default=default_img_dir)
    parser.add_argument("--labels_dir", type=str, default=default_lbl_dir)
    # 输出目录
    parser.add_argument("--output_dir", type=str, default="./taiyi_dataset_v9_1_robust") 
    parser.add_argument("--resolution", type=int, default=512) 
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "conditioning_images"), exist_ok=True)
    
    # 1. 初始化 Mask 生成器
    ink_generator = InkWashMaskGenerator(width=args.resolution, height=args.resolution)
    
    # 2. 初始化 态势提取器
    gestalt_extractor = FixedVisualGestaltExtractor()
    print("✅ Fixed Visual Gestalt Extractor (V9.1 with Chinese Path Support) initialized.")
    
    # =========================================================
    # [V9.1 核心修复] 建立全局图片索引 (Global Image Index)
    # 解决路径混乱、子文件夹找不到、中文路径等问题
    # =========================================================
    print(f"🔍 正在扫描图片目录建立索引: {args.images_dir} ...")
    image_index = {}
    scan_count = 0
    # os.walk 会递归扫描所有子文件夹
    for root, dirs, files in os.walk(args.images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                # 建立 文件名 -> 绝对路径 的映射
                image_index[file] = os.path.join(root, file)
                scan_count += 1
    print(f"✅ 索引建立完成。共找到 {scan_count} 张图片。")

    df = pd.read_excel(args.xlsx_path)
    metadata_entries = []
    
    success_count = 0
    missing_count = 0

    print(f"🚀 开始处理数据，共 {len(df)} 条...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            raw_img_name = str(row['image']).strip()
            poem = str(row['poem']).strip()
            
            # --- 智能路径查找 ---
            src_img_path = None
            
            # 策略 1: 绝对路径且存在
            if os.path.isabs(raw_img_name) and os.path.exists(raw_img_name):
                src_img_path = raw_img_name
            
            # 策略 2: 使用索引查找 (文件名匹配)
            # 提取纯文件名 (例如 "6800/a.jpg" -> "a.jpg")
            if src_img_path is None:
                basename = os.path.basename(raw_img_name)
                if basename in image_index:
                    src_img_path = image_index[basename]
            
            # 策略 3: 简单拼接 (Fallback)
            if src_img_path is None:
                fallback = os.path.join(args.images_dir, raw_img_name)
                if os.path.exists(fallback):
                    src_img_path = fallback

            # 还是找不到？记录并跳过
            if src_img_path is None:
                missing_count += 1
                # print(f"⚠️ 跳过: 找不到图片 {raw_img_name}") # 可取消注释以调试
                continue

            # 构造 Label 路径 (Label 通常和图片同名，但在指定文件夹下)
            img_stem = Path(src_img_path).stem
            label_path = os.path.join(args.labels_dir, f"{img_stem}.txt")
            if not os.path.exists(label_path): 
                continue

            # 3. 读取 Box 并提取真实态势
            boxes_9d = [] 
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5: 
                        cls_id, cx, cy, w, h = map(float, parts[:5])
                        
                        # 提取 Flow (包含负值)
                        g_params, valid = gestalt_extractor.extract(src_img_path, [cx, cy, w, h])
                        
                        # 如果提取失败（例如太小或空白），给一个默认湿润值
                        if valid < 0.5:
                            g_params = [0.0, 0.0, 0.0, 0.5] 
                            
                        full_box = [cls_id, cx, cy, w, h] + g_params
                        boxes_9d.append(full_box)
            
            if not boxes_9d: continue

            # 4. 生成 Mask (带枯笔纹理)
            cond_img = ink_generator.convert_boxes_to_mask(boxes_9d) 
            cond_img_name = f"{img_stem}_ink_v9.png"
            cond_img.save(os.path.join(args.output_dir, "conditioning_images", cond_img_name))
            
            # 5. 处理原图 (复制并 Resize)
            # [Fix] 使用 cv2 读取再转 PIL，确保中文路径也能被正确加载
            # 注意：cv2 读取的是 BGR，转 PIL 前需要转 RGB
            try:
                img_array = np.fromfile(src_img_path, dtype=np.uint8)
                img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img_cv is None: continue
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                target_img = Image.fromarray(img_rgb)
            except Exception:
                # Fallback 到 PIL 读取 (如果非中文路径可能更快)
                target_img = Image.open(src_img_path).convert("RGB")

            target_img = target_img.resize((args.resolution, args.resolution), Image.BICUBIC)
            target_img_name = f"{img_stem}.jpg"
            target_img.save(os.path.join(args.output_dir, "images", target_img_name))

            # 6. 构造 Prompt
            chinese_prompt = f"{poem}"

            metadata_entries.append({
                "image": f"images/{target_img_name}",
                "conditioning_image": f"conditioning_images/{cond_img_name}",
                "text": chinese_prompt
            })
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            continue

    # 保存 JSONL
    output_jsonl = os.path.join(args.output_dir, "train.jsonl")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"✨ V9.1 数据准备完成！")
    print(f"✅ 成功处理: {success_count} 张")
    if missing_count > 0:
        print(f"⚠️ 丢失图片: {missing_count} 张 (请检查文件名索引)")
    print(f"📂 输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()