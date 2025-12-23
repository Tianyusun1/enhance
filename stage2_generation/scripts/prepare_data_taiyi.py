# File: stage2_generation/scripts/prepare_data_taiyi.py (V8.1: Real Gestalt Extraction)

import sys
import os
import argparse
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

# === 路径设置 ===
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
    parser = argparse.ArgumentParser(description="Taiyi V8.1: 准备包含真实物理态势的训练数据")
    default_xlsx = "/home/610-sty/layout2paint/dataset/6800poems.xlsx"
    default_img_dir = "/home/610-sty/layout2paint/dataset/6800"
    default_lbl_dir = "/home/610-sty/layout2paint/dataset/6800/JPEGImages-pre_new_txt"
    
    parser.add_argument("--xlsx_path", type=str, default=default_xlsx)
    parser.add_argument("--images_dir", type=str, default=default_img_dir)
    parser.add_argument("--labels_dir", type=str, default=default_lbl_dir)
    parser.add_argument("--output_dir", type=str, default="./taiyi_dataset_v8_real_gestalt") # 建议区分目录
    parser.add_argument("--resolution", type=int, default=512) 
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "conditioning_images"), exist_ok=True)
    
    # 1. 初始化 Mask 生成器 (绘图用)
    ink_generator = InkWashMaskGenerator(width=args.resolution, height=args.resolution)
    
    # 2. [NEW] 初始化态势提取器 (从原图提取物理参数用)
    gestalt_extractor = VisualGestaltExtractor()
    print("✅ Visual Gestalt Extractor (Pixel-Level) initialized.")
    
    df = pd.read_excel(args.xlsx_path)
    
    metadata_entries = []
    
    # 基础风格词
    style_suffix = "，水墨画，中国画，写意，杰作，高分辨率"

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

            # 3. 读取 Box 并提取真实态势
            boxes_9d = [] # 存储 9 维数据 [cls, cx, cy, w, h, bx, by, rot, flow]
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5: 
                        # 基础几何信息
                        cls_id, cx, cy, w, h = map(float, parts[:5])
                        
                        # [核心升级] 实时从原图提取真实的 Gestalt 参数
                        # extract 接口返回: ([bias_x, bias_y, rot, flow], valid_score)
                        g_params, valid = gestalt_extractor.extract(src_img_path, [cx, cy, w, h])
                        
                        # 数据清洗：如果提取失败（如区域太小、纯白），则使用全0默认值
                        # 这样 InkWashMaskGenerator 会回退到该类别的默认画法
                        if valid < 0.5:
                            g_params = [0.0, 0.0, 0.0, 0.0]
                            
                        # 组装 9 维向量
                        # 注意：这里我们不再依赖 txt 里可能存在的旧态势数据，而是重新从原图提取最新的
                        full_box = [cls_id, cx, cy, w, h] + g_params
                        boxes_9d.append(full_box)
            
            if not boxes_9d: continue

            # 4. 生成彩色势能场 Mask
            # 传入 9 维数据，让 Generator 能够画出真实的重心偏移和墨韵洇散
            # 注意：请确保 utils/ink_mask.py 中的 convert_boxes_to_mask 能处理 len(box)==9 的情况
            cond_img = ink_generator.convert_boxes_to_mask(boxes_9d)
            
            # 关键：确保保存为 RGB 模式
            cond_img_name = f"{img_stem}_ink_v8.png"
            cond_img.save(os.path.join(args.output_dir, "conditioning_images", cond_img_name))
            
            # 5. 处理原图 (Resize 到 512)
            target_img = Image.open(src_img_path).convert("RGB")
            target_img = target_img.resize((args.resolution, args.resolution), Image.BICUBIC)
            target_img_name = f"{img_stem}.jpg"
            target_img.save(os.path.join(args.output_dir, "images", target_img_name))

            # 6. 构造中文 Prompt
            chinese_prompt = f"{poem}{style_suffix}"

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
            
    print(f"✨ V8.1 真实态势数据集准备完成！")
    print(f"📂 输出目录: {args.output_dir}")
    print(f"📄 索引文件: {output_jsonl}")
    print("⚠️  下一步提示: 请检查 stage2_generation/utils/ink_mask.py 是否已支持 9 维输入绘图！")

if __name__ == "__main__":
    main()