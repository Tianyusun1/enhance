# File: tianyusun1/test2/test2-5.2/scripts/infer.py (V6.1: Fix Tuple Error)

# --- 强制添加项目根目录到 Python 模块搜索路径 ---
import sys
import os

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path)) 
sys.path.insert(0, project_root)
# ---------------------------------------------

import torch
import argparse 
from transformers import BertTokenizer
from models.poem2layout import Poem2LayoutGenerator
from inference.greedy_decode import greedy_decode_poem_layout
from data.utils import layout_seq_to_yolo_txt
from data.visualize import draw_layout
import yaml
import re
import string
import random
import copy

# --- 50 句古代诗句 ---
POEMS_50 = [
    "白日依山尽，黄河入海流。",
    "明月松间照，清泉石上流。",
    "野旷天低树，江清月近人。",
    "两岸青山相对出，孤帆一片日边来。",
    "孤舟蓑笠翁，独钓寒江雪。",
    "大漠孤烟直，长河落日圆。",
    "山高月小，水落石出。",
    "月落乌啼霜满天，江枫渔火对愁眠。",
    "落霞与孤鹜齐飞，秋水共长天一色。",
    "渭城朝雨浥轻尘，客舍青青柳色新。",
    "千山鸟飞绝，万径人踪灭。",
    "小楼一夜听春雨，深巷明朝卖杏花。",
    "竹喧归浣女，莲动下渔舟。",
    "云想衣裳花想容，春风拂槛露华浓。",
    "独在异乡为异客，每逢佳节倍思亲。",
    "江流天地外，山色有无中。",
    "青山横北郭，白水绕东城。",
    "柴门闻犬吠，风雪夜归人。",
    "空山新雨后，天气晚来秋。",
    "一水护田将绿绕，两山排闼送青来.",
    "接天莲叶无穷碧，映日荷花别样红。",
    "黄河远上白云间，一片孤城万仞山.",
    "山回路转不见君，雪上空留马行处.",
    "西塞山前白鹭飞，桃花流水鳜鱼肥.",
    "日出江花红胜火，春来江水绿如蓝.",
    "两岸猿声啼不住，轻舟已过万重山.",
    "溪云初起日沉阁，山雨欲来风满楼.",
    "鸡声茅店月，人迹板桥霜.",
    "林表明霁色，城中增暮寒.",
    "清明时节雨纷纷，路上行人欲断魂.",
    "轻舟短棹西湖好，绿水逶迤，芳草长堤.",
    "山光悦鸟性，潭影空人心.",
    "绿树村边合，青山郭外斜.",
    "横看成岭侧成峰，远近高低各不同."
]

def sanitize_filename(text, max_len=30):
    """将诗句转换为安全的文件名"""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned = ''.join(c for c in text if c in valid_chars or '\u4e00' <= c <= '\u9fff')
    cleaned = cleaned.replace(' ', '_').replace('　', '_')
    return cleaned[:max_len].strip('_').replace('__', '_') or "poem"

# ==========================================
# [修复版] 随机对称变换 + 碰撞检测
# ==========================================

def calculate_total_iou(boxes_tensor):
    """计算当前所有框的总重叠面积"""
    if boxes_tensor.size(0) < 2: return 0.0
    
    x1 = boxes_tensor[:, 0] - boxes_tensor[:, 2] / 2
    x2 = boxes_tensor[:, 0] + boxes_tensor[:, 2] / 2
    y1 = boxes_tensor[:, 1] - boxes_tensor[:, 3] / 2
    y2 = boxes_tensor[:, 1] + boxes_tensor[:, 3] / 2
    
    n = boxes_tensor.size(0)
    total_inter = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            total_inter += w * h
            
    return total_inter

def apply_random_symmetry(layout_list, collision_threshold=None):
    """
    [V4 最终版] 完美支持 (Label, X, Y, W, H...) 扁平元组结构
    """
    if not layout_list: return layout_list
    
    # --- 1. 深度格式检测 ---
    first_item = layout_list[0]
    boxes_data = []
    mode = 'unknown' 

    # Case A: 字典模式
    if isinstance(first_item, dict) and 'bbox' in first_item:
        mode = 'dict'
        boxes_data = [item['bbox'] for item in layout_list]

    # Case B: 扁平元组 (Label, X, Y, W, H, ...) <--- 这是你的情况！
    # 特征：长度>=5, 第0个是数字(Label), 第1-4个也是数字(坐标)
    elif isinstance(first_item, (tuple, list)) and len(first_item) >= 5:
        # 假设: Index 0=Label, 1=X, 2=Y, 3=W, 4=H
        mode = 'flat_tuple_label_first'
        # 我们只提取 [x, y, w, h] (即 index 1 到 4)
        boxes_data = [list(item[1:5]) for item in layout_list]
        print(f"   [Debug] Detected format: (Label, X, Y, W, H...)")

    # Case C: 嵌套元组 ([x,y,w,h], label)
    elif isinstance(first_item, (tuple, list)) and \
         len(first_item) > 0 and isinstance(first_item[0], (tuple, list)):
        mode = 'tuple_nested'
        boxes_data = [item[0] for item in layout_list]

    # Case D: 纯坐标列表 [x,y,w,h]
    elif isinstance(first_item, (tuple, list)) and len(first_item) == 4:
        mode = 'list_flat'
        boxes_data = layout_list

    else:
        print(f"⚠️ [Symmetry] Unknown data format: {first_item}. Skipping.")
        return layout_list

    # --- 2. 转换 Tensor ---
    try:
        boxes_tensor = torch.tensor(boxes_data, dtype=torch.float32)
    except Exception as e:
        print(f"⚠️ [Symmetry] Tensor conversion failed: {e}. Skipping.")
        return layout_list

    # 确保维度正确 [N, 4]
    if boxes_tensor.dim() == 1:
        if boxes_tensor.size(0) == 4: boxes_tensor = boxes_tensor.unsqueeze(0)
        else: return layout_list

    N = boxes_tensor.size(0)
    
    # --- 3. [安全锁] 坐标归一化检查 ---
    # 现在取的是真正的 X (index 0 of boxes_tensor)，应该是 0.248 这种小数
    max_val = boxes_tensor[:, 0].max().item() 
    
    should_flip = True
    if max_val > 1.1: 
        print(f"⚠️ [Warning] X-Coordinate still > 1.0 (Max={max_val:.2f}). Skipping flip.")
        should_flip = False

    # --- 4. 执行翻转 ---
    flipped_count = 0
    if should_flip:
        indices = list(range(N))
        for i in indices:
            if random.random() < 0.3:  # 30% 概率
                original_x = boxes_tensor[i, 0].item()
                # 镜像翻转: new_x = 1.0 - x
                boxes_tensor[i, 0] = 1.0 - original_x
                flipped_count += 1

    # --- 5. 数据回写 (Re-assembly) ---
    final_layout = []
    
    if mode == 'flat_tuple_label_first':
        # 你的情况：需要把改好的 [x,y,w,h] 塞回 (label, x, y, w, h, ...)
        for i, item in enumerate(layout_list):
            new_xywh = boxes_tensor[i].tolist() # [new_x, y, w, h]
            
            # 重新拼接元组: (Label,) + (New X, Y, W, H) + (Rest...)
            # item[0]: Label
            # item[5:]: 后面的参数 (如置信度等)
            new_item = (item[0],) + tuple(new_xywh) + item[5:]
            final_layout.append(new_item)

    elif mode == 'dict':
        final_layout = copy.deepcopy(layout_list)
        for i, item in enumerate(final_layout):
            item['bbox'] = boxes_tensor[i].tolist()
            
    elif mode == 'tuple_nested':
        for i, item in enumerate(layout_list):
            new_box = boxes_tensor[i].tolist()
            new_item = (new_box,) + tuple(item[1:])
            final_layout.append(new_item)
            
    elif mode == 'list_flat':
        final_layout = boxes_tensor.tolist()
            
    if flipped_count > 0:
        print(f"✅ [Symmetry] Flipped {flipped_count}/{N} boxes.")
    else:
        print(f"   [Symmetry] Skipped (random) or Safety Lock.")
        
    return final_layout


# ==========================================

def find_best_checkpoint(output_dir):
    """自动查找最佳检查点"""
    if not os.path.exists(output_dir): return None
    files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
    
    if "rl_best_reward.pth" in files:
        return os.path.join(output_dir, "rl_best_reward.pth")

    rl_checkpoints = []
    for f in files:
        if "rl_finetuned" in f:
            match = re.search(r'epoch_(\d+)', f)
            if match: rl_checkpoints.append((int(match.group(1)), os.path.join(output_dir, f)))
    if rl_checkpoints:
        rl_checkpoints.sort(key=lambda x: x[0], reverse=True)
        return rl_checkpoints[0][1]
    
    best_models = [f for f in files if "best_val_loss" in f]
    if best_models: return os.path.join(output_dir, best_models[0])
    
    epoch_models = []
    for f in files:
        if "model_epoch_" in f and "rl" not in f:
            match = re.search(r'epoch_(\d+)', f)
            if match: epoch_models.append((int(match.group(1)), os.path.join(output_dir, f)))
    if epoch_models:
        epoch_models.sort(key=lambda x: x[0], reverse=True)
        return epoch_models[0][1]
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Batch Inference with Symmetry Augmentation")
    parser.add_argument("--poem", type=str, default=None, help="Single poem")
    parser.add_argument("--mode", type=str, default="sample", choices=["greedy", "sample"], help="Decoding mode")
    parser.add_argument("--top_k", type=int, default=2, help="Top-K for sampling")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples per poem")
    parser.add_argument("--checkpoint", type=str, default="/home/610-sty/layout2paint3/outputs/train_v11_bold_explore_rl/rl_best_reward.pth")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(project_root, "configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Init model
    latent_dim = config['model'].get('latent_dim', 32)
    model = Poem2LayoutGenerator(
        bert_path=config['model']['bert_path'],
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        bb_size=config['model']['bb_size'],
        decoder_layers=config['model']['decoder_layers'],
        decoder_heads=config['model']['decoder_heads'],
        dropout=config['model']['dropout'],
        latent_dim=latent_dim
    )

    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        output_dir = config['training']['output_dir']
        checkpoint_path = find_best_checkpoint(output_dir)

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[Warning] No valid checkpoint found. Running with random weights.")
    else:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("✅ Model loaded successfully.")

    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_path'])

    poems_to_process = [args.poem] if args.poem else POEMS_50
    save_dir = config['training']['output_dir']
    os.makedirs(save_dir, exist_ok=True)

    print(f"------------------------------------------------")
    print(f"Starting batch inference with Symmetry Augmentation...")
    print(f"Mode: {args.mode} | Samples: {args.num_samples}")
    print(f"------------------------------------------------")

    for idx, poem in enumerate(poems_to_process, 1):
        print(f"\n[Poem {idx}/{len(poems_to_process)}] {poem}")
        poem_safe_name = sanitize_filename(poem, max_len=25)

        for sample_idx in range(args.num_samples):
            # 1. 原始生成
            raw_layout = greedy_decode_poem_layout(
                model, tokenizer, poem,
                max_elements=config['model'].get('max_elements', 30),
                device=device.type,
                mode=args.mode,
                top_k=args.top_k
            )

            # 2. 应用随机对称增强 (已修复 Tuple 报错)
            final_layout = apply_random_symmetry(raw_layout, collision_threshold=0.02)

            # 3. 保存与绘图
            suffix = f"_{args.mode}"
            if args.num_samples > 1:
                suffix += f"_sample{sample_idx+1}"
            file_base = f"poem{idx:02d}_{poem_safe_name}{suffix}"

            output_txt_path = os.path.join(save_dir, f"{file_base}.txt")
            output_png_path = os.path.join(save_dir, f"{file_base}.png")

            layout_seq_to_yolo_txt(final_layout, output_txt_path)
            draw_layout(final_layout, f"{poem} (Sym-Aug)", output_png_path)

            print(f"  → Saved: {file_base}.png")

    print(f"\n✅ All done! Check {save_dir} for diverse layouts.")

if __name__ == "__main__":
    main()