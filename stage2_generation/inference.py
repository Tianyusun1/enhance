# File: inference.py (V8.0: End-to-End Data-Driven Gestalt Inference)

import sys
import os
import argparse
import torch
import numpy as np
import yaml
from PIL import Image
from transformers import BertTokenizer
from pathlib import Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# === 路径配置 & 导入 ===
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_file_path) # 假设 inference.py 在根目录
if project_root not in sys.path:
    sys.path.append(project_root)

# 尝试导入项目组件
try:
    from models.poem2layout import Poem2LayoutGenerator
    from inference.greedy_decode import greedy_decode_poem_layout
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
    from data.visualize import draw_layout
except ImportError as e:
    print(f"[Error] 模块导入失败: {e}")
    print("请确保 inference.py 位于项目根目录，或者 PYTHONPATH 设置正确。")
    sys.exit(1)

# =============================================================
# [V8.0 组件] 态势感知注意力处理器 (PoemInkAttentionProcessor)
# =============================================================
class PoemInkAttentionProcessor:
    """
    V8.0 核心：将 8 维布局中的物理态势 (Bias) 注入到 Cross-Attention 中。
    确保生成的画面笔触与 InkMask 的动态墨迹位置一致。
    """
    def __init__(self, dynamic_layout, tokenizer, prompt, device, scale=8.0):
        # dynamic_layout: [N, 9] -> (cls, cx, cy, w, h, bx, by, rot, flow)
        self.layout = dynamic_layout  
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.device = device
        self.scale = scale 

        self.class_to_keyword = {
            2: "山", 3: "水", 4: "人", 5: "树", 6: "屋", 
            7: "桥", 8: "花", 9: "鸟", 10: "兽"
        }

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = attn.to_q(hidden_states)
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # === 态势锚定 (Gestalt Anchoring) ===
        res = int(np.sqrt(sequence_length))
        h, w = res, res
        tokens = self.tokenizer.encode(self.prompt)
        
        for item in self.layout:
            cls_id = int(item[0])
            keyword = self.class_to_keyword.get(cls_id, None)
            if not keyword: continue
            
            # [V8.0] 提取数据驱动的态势参数
            # item: [cls, cx, cy, w, h, bx, by, rot, flow]
            cx, cy, bw, bh = item[1], item[2], item[3], item[4]
            if len(item) >= 7:
                bx, by = item[5], item[6] # Bias Shift
            else:
                bx, by = 0.0, 0.0
            
            keyword_token_ids = self.tokenizer.encode(keyword, add_special_tokens=False)
            token_indices = [i for i, t in enumerate(tokens) if t in keyword_token_ids]
            
            if not token_indices: continue

            # [Alignment Check] 必须与 ink_mask.py (V8.0) 的 center_x 逻辑一致
            # ink_mask V8.0: center_x = (cx + bx * 0.15)
            x_c, y_c = (cx + bx * 0.15) * w, (cy + by * 0.15) * h
            
            x1, y1 = int(x_c - (bw/2)*w), int(y_c - (bh/2)*h)
            x2, y2 = int(x_c + (bw/2)*w), int(y_c + (bh/2)*h)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2) # 修正边界为 w, h

            if x2 > x1 and y2 > y1:
                for idx in token_indices:
                    if idx >= attention_probs.shape[-1]: continue
                    mask = torch.zeros((h, w), device=self.device)
                    # 增强核心区域的注意力
                    mask[y1:y2, x1:x2] = self.scale
                    mask_flat = mask.flatten()
                    attention_probs[:, :, idx] += mask_flat * attention_probs[:, :, idx]

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# =============================================================
# End-to-End Generator
# =============================================================
class EndToEndGenerator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading End-to-End System V8.0 on {self.device}...")

        # 1. 载入配置
        # 优先读取命令行参数中的 config 路径，或者默认路径
        config_path = os.path.join(project_root, "configs", "default.yaml")
        if not os.path.exists(config_path):
            print(f"[Warning] Config not found at {config_path}. Using internal defaults.")
            model_cfg = {'hidden_size': 768, 'bb_size': 128, 'decoder_layers': 6, 'decoder_heads': 8, 'latent_dim': 64}
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            model_cfg = config.get('model', {})

        # 2. 初始化 Stage 1 组件
        print("[Stage 1] Loading Layout Generator...")
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        
        # 初始化模型 (V8.0 参数)
        self.layout_model = Poem2LayoutGenerator(
            bert_path=args.bert_path,
            num_classes=9,
            hidden_size=model_cfg.get('hidden_size', 768),
            bb_size=model_cfg.get('bb_size', 128),
            decoder_layers=model_cfg.get('decoder_layers', 6),
            decoder_heads=model_cfg.get('decoder_heads', 8),
            latent_dim=model_cfg.get('latent_dim', 64),
            gestalt_loss_weight=2.0, # V8.0 Spec
            dropout=0.0
        )
        
        # 加载权重
        checkpoint = torch.load(args.stage1_checkpoint, map_location=self.device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        # 移除 DDP 可能产生的 'module.' 前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            self.layout_model.load_state_dict(state_dict, strict=True)
            print("✅ Stage 1 Model loaded (Strict Mode).")
        except RuntimeError as e:
            print(f"⚠️ Strict loading failed (likely mismatch in V8.0 heads). Trying loose match...")
            self.layout_model.load_state_dict(state_dict, strict=False)
            
        self.layout_model.to(self.device).eval()

        # 3. 初始化 Stage 2 工具
        self.width = 512
        self.height = 512
        # InkMaskGenerator V8.0
        self.ink_gen = InkWashMaskGenerator(width=self.width, height=self.height) 

        # 4. 加载 Stable Diffusion + ControlNet
        print(f"[Stage 2] Loading Dual ControlNets & Taiyi...")
        controlnet_s = ControlNetModel.from_pretrained(
            os.path.join(args.stage2_checkpoint, "controlnet_structure"), torch_dtype=torch.float16
        )
        controlnet_t = ControlNetModel.from_pretrained(
            os.path.join(args.stage2_checkpoint, "controlnet_style"), torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            args.base_model_path, 
            controlnet=[controlnet_s, controlnet_t], 
            torch_dtype=torch.float16,
            safety_checker=None 
        )

        # 加载 LoRA
        lora_path = os.path.join(args.stage2_checkpoint, "unet_lora")
        if os.path.exists(lora_path):
            try:
                self.pipe.load_lora_weights(lora_path)
                print(f"✅ LoRA loaded from {lora_path}")
            except Exception as e:
                print(f"⚠️ LoRA load failed: {e}")
        
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload() # 显存优化

    def infer(self, poem, seed=2024, output_name=None):
        print(f"\n🎨 Generating for: {poem}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        save_dir = Path(self.args.output_dir) / f"{poem[:10]}_{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # === Step 1: Layout Generation (V8.0) ===
        # 使用 greedy_decode 统一接口，获取 8 维布局
        layout_list = greedy_decode_poem_layout(
            self.layout_model, self.tokenizer, poem, 
            max_elements=30, device=self.device.type, mode='sample', top_k=5
        )
        
        if not layout_list:
            print("⚠️ No layout generated.")
            return

        # 转换为 numpy 方便后续处理 [N, 9]
        layout = np.array(layout_list) # (cls, cx, cy, w, h, bx, by, rot, flow)

        # === Step 2: Visualize Layout ===
        # draw_layout 能够自适应 5 维或 9 维输入
        draw_layout(layout, f"Layout: {poem}", str(save_dir / "01_layout.png"))

        # === Step 3: Gestalt Ink Mask (V8.0) ===
        # 利用 V8.0 的 InkMaskGenerator 生成动态势能图
        ink_mask = self.ink_gen.convert_boxes_to_mask(layout)
        ink_mask.save(save_dir / "02_ink_mask.png")

        # === Step 4: Attention Injection (V8.0) ===
        attn_proc = PoemInkAttentionProcessor(
            dynamic_layout=layout, 
            tokenizer=self.pipe.tokenizer, 
            prompt=poem, 
            device=self.device,
            scale=8.0 
        )
        self.pipe.unet.set_attn_processor(attn_proc)

        # === Step 5: Diffusion Generation ===
        prompt = f"{poem}，写意水墨画，中国画风格，杰作，留白，意境"
        neg_prompt = "低质量，模糊，色彩斑驳，边框，水印，文字，现代建筑，照片真实感"
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt, 
            negative_prompt=neg_prompt,
            image=[ink_mask, ink_mask], # Structure + Style Control
            num_inference_steps=35, 
            controlnet_conditioning_scale=[1.1, 0.8], # 结构权重 > 风格权重
            guidance_scale=7.5, 
            generator=generator
        ).images[0]
        
        final_name = output_name if output_name else "03_final_painting.png"
        image.save(save_dir / final_name)
        print(f"✅ Result saved to: {save_dir}")

def main():
    parser = argparse.ArgumentParser()
    # 默认路径配置 (根据你的环境)
    parser.add_argument("--bert_path", type=str, default="/home/610-sty/huggingface/bert-base-chinese")
    parser.add_argument("--stage1_checkpoint", type=str, default="/home/610-sty/layout2paint2/outputs/train_v8/rl_best_reward.pth")
    parser.add_argument("--stage2_checkpoint", type=str, default="/home/610-sty/layout2paint2/outputs/taiyi_ink_controlnet_v2")
    parser.add_argument("--base_model_path", type=str, default="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
    parser.add_argument("--output_dir", type=str, default="inference_results_v8")
    
    parser.add_argument("--poem", type=str, default="明月松间照，清泉石上流。", help="Input poem")
    parser.add_argument("--seed", type=int, default=2024)
    
    args = parser.parse_args()
    
    # 实例化并运行
    engine = EndToEndGenerator(args)
    engine.infer(args.poem, args.seed)

if __name__ == "__main__":
    main()