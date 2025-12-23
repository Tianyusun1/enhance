# File: tianyusun1/test2/test2-5.2/scripts/infer.py (V5.9: Auto-load Best RL Model)

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

# --- 50 句古代诗句（已清洗，含具体描绘）---
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
    "霜落熊升树，林空鹿饮溪.",
    "千峰笋石千株玉，万树松萝万朵云.",
    "烟波江上使人愁。",
    "渔舟逐水爱山春，两岸桃花夹古津.",
    "楼观沧海日，门对浙江潮.",
    "松风吹解带，山月照弹琴.",
    "野渡无人舟自横.",
    "湖光秋月两相和，潭面无风镜未磨.",
    "江碧鸟逾白，山青花欲燃.",
    "石泉流暗壁，草露滴秋根.",
    "晓看红湿处，花重锦官城.",
    "榆柳荫后檐，桃李罗堂前.",
    "木末芙蓉花，山中发红萼.",
    "露从今夜白，月是故乡明.",
    "萧萧梧叶送寒声，江上秋风动客情.",
    "山寺月中寻桂子，郡亭枕上看潮头.",
    "横看成岭侧成峰，远近高低各不同."
]

def sanitize_filename(text, max_len=30):
    """将诗句转换为安全的文件名（移除标点、空格，截断）"""
    # 移除标点和特殊字符
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned = ''.join(c for c in text if c in valid_chars or '\u4e00' <= c <= '\u9fff')  # 保留中英文
    cleaned = cleaned.replace(' ', '_').replace('　', '_')
    # 截断并去除首尾下划线
    return cleaned[:max_len].strip('_').replace('__', '_') or "poem"

def find_best_checkpoint(output_dir):
    """
    自动查找最佳检查点。
    优先级：
    1. rl_best_reward.pth (RL微调后的最佳奖励模型)
    2. rl_finetuned_epoch_X.pth (最新的 RL 模型)
    3. model_best_val_loss.pth (监督训练的最佳 Loss 模型)
    4. model_epoch_X.pth (最新的监督训练模型)
    """
    if not os.path.exists(output_dir):
        return None
    files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]
    
    # [NEW] 优先级 1: RL 最佳奖励模型
    if "rl_best_reward.pth" in files:
        checkpoint_path = os.path.join(output_dir, "rl_best_reward.pth")
        print(f"[Auto-Resume] Found Best RL Reward model: {checkpoint_path}")
        return checkpoint_path

    # 优先级 2: RL 微调过程中的 Checkpoint
    rl_checkpoints = []
    for f in files:
        if "rl_finetuned" in f:
            match = re.search(r'epoch_(\d+)', f)
            if match:
                epoch_num = int(match.group(1))
                rl_checkpoints.append((epoch_num, os.path.join(output_dir, f)))
    if rl_checkpoints:
        rl_checkpoints.sort(key=lambda x: x[0], reverse=True)
        print(f"[Auto-Resume] Found latest RL finetuned model: {rl_checkpoints[0][1]}")
        return rl_checkpoints[0][1]
    
    # 优先级 3: 监督训练最佳验证集模型
    best_models = [f for f in files if "best_val_loss" in f]
    if best_models:
        print(f"[Auto-Resume] Found Best Val Loss model: {os.path.join(output_dir, best_models[0])}")
        return os.path.join(output_dir, best_models[0])
    
    # 优先级 4: 监督训练普通 Checkpoint
    epoch_models = []
    for f in files:
        if "model_epoch_" in f and "rl" not in f:
            match = re.search(r'epoch_(\d+)', f)
            if match:
                epoch_num = int(match.group(1))
                epoch_models.append((epoch_num, os.path.join(output_dir, f)))
    if epoch_models:
        epoch_models.sort(key=lambda x: x[0], reverse=True)
        print(f"[Auto-Resume] Found latest supervised model: {epoch_models[0][1]}")
        return epoch_models[0][1]
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Batch Inference for 50 Ancient Poems")
    parser.add_argument("--poem", type=str, default=None, help="Single poem for inference (optional)")
    parser.add_argument("--mode", type=str, default="sample", choices=["greedy", "sample"], help="Decoding mode")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K for sampling")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples per poem")
    parser.add_argument("--checkpoint", type=str, default="/home/610-sty/layout2paint3/outputs/train_v7_gestal_rl/rl_best_reward.pth", help="Path to specific checkpoint (default: auto-find best)")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(project_root, "configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Init model
    latent_dim = config['model'].get('latent_dim', 32)
    print(f"Initializing model with latent_dim={latent_dim}...")
    model = Poem2LayoutGenerator(
        bert_path=config['model']['bert_path'],
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        bb_size=config['model']['bb_size'],
        decoder_layers=config['model']['decoder_layers'],
        decoder_heads=config['model']['decoder_heads'],
        dropout=config['model']['dropout'],
        latent_dim=latent_dim,
        clustering_loss_weight=config['model'].get('clustering_loss_weight', 1.0)
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
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            print("✅ Model loaded successfully.")
        except RuntimeError as e:
            print(f"\n[ERROR] Checkpoint loading failed: {e}")
            print("Tip: Check if 'latent_dim' in config matches the trained model.")
            return

    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(config['model']['bert_path'])

    # 确定要处理的诗句列表
    if args.poem:
        poems_to_process = [args.poem]
    else:
        poems_to_process = POEMS_50

    save_dir = config['training']['output_dir']
    os.makedirs(save_dir, exist_ok=True)

    print(f"------------------------------------------------")
    print(f"Starting batch inference for {len(poems_to_process)} poem(s)...")
    print(f"Mode: {args.mode} | Top-K: {args.top_k} | Samples per poem: {args.num_samples}")
    print(f"Results will be saved to: {save_dir}")
    print(f"------------------------------------------------")

    for idx, poem in enumerate(poems_to_process, 1):
        print(f"\n[Poem {idx}/{len(poems_to_process)}] {poem}")
        poem_safe_name = sanitize_filename(poem, max_len=25)

        for sample_idx in range(args.num_samples):
            layout = greedy_decode_poem_layout(
                model, tokenizer, poem,
                max_elements=config['model'].get('max_elements', 30),
                device=device.type,
                mode=args.mode,
                top_k=args.top_k
            )

            # 构建唯一文件名
            suffix = f"_{args.mode}"
            if args.num_samples > 1:
                suffix += f"_sample{sample_idx+1}"
            file_base = f"poem{idx:02d}_{poem_safe_name}{suffix}"

            output_txt_path = os.path.join(save_dir, f"{file_base}.txt")
            output_png_path = os.path.join(save_dir, f"{file_base}.png")

            layout_seq_to_yolo_txt(layout, output_txt_path)
            draw_layout(layout, f"{args.mode.capitalize()} Inference: {poem}", output_png_path)

            print(f"  → Saved: {file_base}.png")

    print(f"\n✅ Batch inference completed! All results saved in: {save_dir}")

if __name__ == "__main__":
    main()