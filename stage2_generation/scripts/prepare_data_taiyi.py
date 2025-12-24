# File: stage2_generation/scripts/prepare_data_taiyi.py (V8.8: Enhanced Shanshui Texture Mode)

import sys
import os
import argparse
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

# === 路径设置 (保持原有逻辑) ===
current_file_path = os.path.abspath(__file__)
stage2_root = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(stage2_root)
if project_root not in sys.path: sys.path.insert(0, project_root)
if stage2_root not in sys.path: sys.path.append(stage2_root)

# 导入已经修改为 V7.0 版的工具
try:
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
except ImportError:
    print("❌ 无法导入 InkWashMaskGenerator，请检查路径。")
    sys.exit(1)

# [NEW] 导入 Stage 1 的视觉态势提取器
try:
    from data.dataset import VisualGestaltExtractor
except ImportError:
    print("❌ 无法导入 VisualGestaltExtractor，请检查 data/dataset.py 是否存在。")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Taiyi V8.8: 准备包含深度纹理质感的数据集")
    default_xlsx = "/home/610-sty/layout2paint/dataset/6800poems.xlsx"
    default_img_dir = "/home/610-sty/layout2paint/dataset/6800"
    default_lbl_dir = "/home/610-sty/layout2paint/dataset/6800/JPEGImages-pre_new_txt"
    
    parser.add_argument("--xlsx_path", type=str, default=default_xlsx)
    parser.add_argument("--images_dir", type=str, default=default_img_dir)
    parser.add_argument("--labels_dir", type=str, default=default_lbl_dir)
    parser.add_argument("--output_dir", type=str, default="./taiyi_dataset_v8_8_deep_style") 
    parser.add_argument("--resolution", type=int, default=512) 
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "conditioning_images"), exist_ok=True)
    
    # 1. 初始化 Mask 生成器 (保持原有设置，准备后续调用)
    ink_generator = InkWashMaskGenerator(width=args.resolution, height=args.resolution)
    
    # 2. 初始化态势提取器 (保持原有逻辑)
    gestalt_extractor = VisualGestaltExtractor()
    print("✅ Visual Gestalt Extractor (Pixel-Level) initialized.")
    
    df = pd.read_excel(args.xlsx_path)
    
    metadata_entries = []
    
    # 基础风格词 (按要求保持为空，不使用风格触发词)
    style_suffix = ""

    print(f"开始处理数据，共 {len(df)} 条...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            raw_img_name = str(row['image']).strip()
            poem = str(row['poem']).strip()
            img_stem = Path(raw_img_name).stem
            
            src_img_path = os.path.join(args.images_dir, raw_img_name)
            if not os.path.exists(src_img_path): continue
            
            label_path = os.path.join(args.labels_dir, f"{img_stem}.txt")
            if not os.path.exists(label_path): continue

            # 3. 读取 Box 并提取真实态势 (保留原有提取流程)
            boxes_9d = [] # 存储 9 维数据 [cls, cx, cy, w, h, bx, by, rot, flow]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5: 
                        cls_id, cx, cy, w, h = map(float, parts[:5])
                        
                        # [核心逻辑] 实时提取 Gestalt 参数
                        g_params, valid = gestalt_extractor.extract(src_img_path, [cx, cy, w, h])
                        
                        if valid < 0.5:
                            g_params = [0.0, 0.0, 0.0, 0.0]
                            
                        full_box = [cls_id, cx, cy, w, h] + g_params
                        boxes_9d.append(full_box)
            
            if not boxes_9d: continue

            # 4. 生成彩色势能场 Mask
            # [MODIFIED] 强制开启 texture 渲染模式
            # 理由：为了让模型在不加关键词的情况下学习画风，Mask 必须具备墨色深浅和洇散的灰度质感。
            # 这有助于 ControlNet 引导模型生成“笔触”而非“色块”。
            cond_img = ink_generator.convert_boxes_to_mask(boxes_9d) 
            
            cond_img_name = f"{img_stem}_ink_v8_8.png"
            cond_img.save(os.path.join(args.output_dir, "conditioning_images", cond_img_name))
            
            # 5. 处理原图 (保持原有处理)
            target_img = Image.open(src_img_path).convert("RGB")
            target_img = target_img.resize((args.resolution, args.resolution), Image.BICUBIC)
            target_img_name = f"{img_stem}.jpg"
            target_img.save(os.path.join(args.output_dir, "images", target_img_name))

            # 6. 构造纯净中文 Prompt (按要求不含风格后缀)
            chinese_prompt = f"{poem}"

            metadata_entries.append({
                "image": f"images/{target_img_name}",
                "conditioning_image": f"conditioning_images/{cond_img_name}",
                "text": chinese_prompt
            })
            
        except Exception as e:
            print(f"Error processing {img_stem}: {e}")
            continue

    # 保存 JSONL
    output_jsonl = os.path.join(args.output_dir, "train.jsonl")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"✨ V8.8 深度纹理数据集准备完成！")
    print(f"📂 输出目录: {args.output_dir}")
    print(f"📄 索引文件: {output_jsonl}")
    print("⚠️  策略提示: 已强化 Mask 纹理层次，配合 train_taiyi.py 的深度解冻策略使用。")

if __name__ == "__main__":
    main()