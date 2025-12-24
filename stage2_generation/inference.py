# File: scripts/infer.py (V9.5: End-to-End Smooth Gestalt Inference)

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

# === 路径配置 & 导入 (完整保留) ===
current_file_path = os.path.abspath(__file__)
# 假设脚本在 scripts/ 或 stage2_generation/ 目录下，向上找两级到项目根目录
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入项目组件
try:
    from models.poem2layout import Poem2LayoutGenerator
    from inference.greedy_decode import greedy_decode_poem_layout
    # 确保 ink_mask 是 V8.6+ 支持纹理的版本
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
    from data.visualize import draw_layout
except ImportError as e:
    print(f"[Error] 模块导入失败: {e}")
    print(f"当前 sys.path: {sys.path}")
    sys.exit(1)

# =============================================================
# [V9.5 组件] 态势感知注意力处理器 (PoemInkAttentionProcessor)
# =============================================================
class PoemInkAttentionProcessor:
    """
    V9.5 核心：将 9 维布局中的物理态势通过高斯能量场注入到 Cross-Attention 中。
    确保生成的画面笔触与 InkMask 的动态墨迹位置一致，且边缘自然衰减。
    """
    def __init__(self, dynamic_layout, tokenizer, prompt, device, scale=5.0):
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

        # === 态势能量场锚定 (Gestalt Energy Anchoring) [V9.5 修改] ===
        tokens = self.tokenizer.encode(self.prompt)
        res = int(np.sqrt(attention_probs.shape[1])) # 动态获取分辨率
        h, w = res, res
        
        # 预计算坐标网格
        yy, xx = torch.meshgrid(
            torch.arange(h, device=self.device), 
            torch.arange(w, device=self.device), 
            indexing='ij'
        )
        
        for item in self.layout:
            cls_id = int(item[0])
            keyword = self.class_to_keyword.get(cls_id, None)
            if not keyword: continue
            
            # 提取态势参数
            cx, cy, bw, bh = item[1], item[2], item[3], item[4]
            bx, by = item[5], item[6] if len(item) >= 7 else (0.0, 0.0)
            
            keyword_token_ids = self.tokenizer.encode(keyword, add_special_tokens=False)
            token_indices = [i for i, t in enumerate(tokens) if t in keyword_token_ids]
            
            if not token_indices: continue

            # 1. 计算对齐中心 (与训练端一致：0.15 偏移系数)
            x_c, y_c = (cx + bx * 0.15) * w, (cy + by * 0.15) * h
            
            # 2. 计算标准差 (基于物体尺寸，/4 确保场强平滑)
            sigma = ((bw * w + bh * h) / 4.0) + 1e-6
            
            # 3. 生成高斯能量场掩码
            dist_sq = (xx - x_c)**2 + (yy - y_c)**2
            gauss_mask = torch.exp(-dist_sq / (2 * sigma**2)) * self.scale
            mask_flat = gauss_mask.flatten()

            # 4. 软注入注意力矩阵
            for idx in token_indices:
                if idx >= attention_probs.shape[-1]: continue
                attention_probs[:, :, idx] += mask_flat * attention_probs[:, :, idx]

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# =============================================================
# End-to-End Generator (V8.8 Updated)
# =============================================================
class EndToEndGenerator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading End-to-End System V9.5 on {self.device}...")

        # 1. 载入配置 (Stage 1)
        config_path = os.path.join(project_root, "configs", "default.yaml")
        if not os.path.exists(config_path):
            print(f"[Warning] Config not found at {config_path}. Using internal defaults.")
            model_cfg = {'hidden_size': 768, 'bb_size': 128, 'decoder_layers': 6, 'decoder_heads': 8, 'latent_dim': 64}
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            model_cfg = config.get('model', {})

        # 2. 初始化 Stage 1 (Poem2Layout)
        print("[Stage 1] Loading Layout Generator...")
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        
        self.layout_model = Poem2LayoutGenerator(
            bert_path=args.bert_path,
            num_classes=9,
            hidden_size=model_cfg.get('hidden_size', 768),
            bb_size=model_cfg.get('bb_size', 128),
            decoder_layers=model_cfg.get('decoder_layers', 6),
            decoder_heads=model_cfg.get('decoder_heads', 8),
            latent_dim=model_cfg.get('latent_dim', 64),
            gestalt_loss_weight=2.0, 
            dropout=0.0
        )
        
        # 加载 Layout 权重
        if os.path.exists(args.stage1_checkpoint):
            checkpoint = torch.load(args.stage1_checkpoint, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.layout_model.load_state_dict(state_dict, strict=False)
            print("✅ Stage 1 Model loaded.")
        else:
            print(f"❌ Stage 1 Checkpoint not found: {args.stage1_checkpoint}")
            
        self.layout_model.to(self.device).eval()

        # 3. 初始化 Stage 2 工具
        self.width = 512
        self.height = 512
        self.ink_gen = InkWashMaskGenerator(width=self.width, height=self.height) 

        # 4. 加载 Stable Diffusion + ControlNet
        print(f"[Stage 2] Loading Single-Stream ControlNet & Taiyi...")
        
        cnet_path = os.path.join(args.stage2_checkpoint, "controlnet_structure")
        try:
            controlnet = ControlNetModel.from_pretrained(cnet_path, torch_dtype=torch.float16)
        except OSError:
            print(f"❌ ControlNet not found at {cnet_path}. Did training finish?")
            sys.exit(1)

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            args.base_model_path, 
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None 
        )

        # 加载 LoRA
        lora_path = os.path.join(args.stage2_checkpoint, "unet_lora")
        if os.path.exists(lora_path):
            try:
                self.pipe.load_lora_weights(lora_path)
                print(f"✅ LoRA loaded from {lora_path} (Strong Style Binding)")
            except Exception as e:
                print(f"⚠️ LoRA load failed: {e}")
        else:
            print(f"⚠️ LoRA path not found: {lora_path}")
        
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()

    def infer(self, poem, seed=2024, output_name=None):
        print(f"\n🎨 Generating for: {poem}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        save_dir = Path(self.args.output_dir) / f"{poem[:10]}_{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # === Step 1: Layout Generation ===
        layout_list = greedy_decode_poem_layout(
            self.layout_model, self.tokenizer, poem, 
            max_elements=30, device=self.device.type, mode='sample', top_k=5
        )
        
        if not layout_list:
            print("⚠️ No layout generated.")
            return

        layout = np.array(layout_list)

        # === Step 2: Visualize Layout ===
        draw_layout(layout, f"Layout: {poem}", str(save_dir / "01_layout.png"))

        # === Step 3: Textured Ink Mask ===
        ink_mask = self.ink_gen.convert_boxes_to_mask(layout)
        ink_mask.save(save_dir / "02_ink_mask.png")

        # === Step 4: Attention Injection (V9.5 Soft Energy) ===
        attn_proc = PoemInkAttentionProcessor(
            dynamic_layout=layout, 
            tokenizer=self.pipe.tokenizer, 
            prompt=poem, 
            device=self.device,
            scale=5.0  # 建议由 8.0 降至 5.0，配合高斯场达到最佳平衡
        )
        self.pipe.unet.set_attn_processor(attn_proc)

        # === Step 5: Diffusion Generation ===
        prompt = poem 
        neg_prompt = "低质量，模糊，色彩斑驳，边框，水印，文字，现代建筑，照片真实感，写实风格，彩色照片"
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt, 
            negative_prompt=neg_prompt,
            image=ink_mask,
            num_inference_steps=35, 
            controlnet_conditioning_scale=1.0, 
            guidance_scale=7.5, 
            generator=generator
        ).images[0]
        
        final_name = output_name if output_name else "03_final_painting.png"
        image.save(save_dir / final_name)
        print(f"✅ Result saved to: {save_dir}/{final_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", type=str, default="/home/610-sty/huggingface/bert-base-chinese")
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
    parser.add_argument("--output_dir", type=str, default="inference_results_v9_5")
    parser.add_argument("--poem", type=str, default="明月松间照，清泉石上流。", help="Input poem")
    parser.add_argument("--seed", type=int, default=2024)
    
    args = parser.parse_args()
    
    # 实例化并运行
    engine = EndToEndGenerator(args)
    engine.infer(args.poem, args.seed)

if __name__ == "__main__":
    main()