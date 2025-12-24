# File: integrated_inference.py (V8.8: Deep Unfreeze Adaptation + Pure Prompt + Texture Control)

import os
import torch
import argparse
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler

# 导入项目内部组件
try:
    from models.poem2layout import Poem2LayoutGenerator
    from inference.greedy_decode import greedy_decode_poem_layout
    from stage2_generation.utils.ink_mask import InkWashMaskGenerator
    from data.visualize import draw_layout
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure project root is in PYTHONPATH.")
    exit(1)

# =============================================================
# 参数解析配置
# =============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Integrated Inference for Poem2Painting (V8.8 Deep Unfreeze)")
    
    # 模型路径参数
    parser.add_argument("--layout_ckpt", type=str, required=True, help="Path to the trained Poem2Layout checkpoint")
    parser.add_argument("--taiyi_model_path", type=str, default="Idea-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1", help="Base Taiyi model")
    
    # V8.8: 对应训练时解冻了 up_blocks 的模型
    parser.add_argument("--controlnet_path", type=str, required=True, help="Path to Structure ControlNet")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to trained UNet LoRA weights (V8.8 Rank 128)")
    
    # 输出设置
    parser.add_argument("--output_base", type=str, default="outputs/integrated_inference_v8_8", help="Directory to save results")
    
    return parser.parse_args()

# =============================================================
# 创新架构：跨模态交叉注意力态势锚定处理器 (Gestalt Attention Processor)
# =============================================================
class PoemInkAttentionProcessor:
    """
    通过干预 Cross-Attention 层实现数学级语义绑定。
    [V8.8 适配]：保持位置锚定，将渲染细节交给解冻后的解码器。
    """
    def __init__(self, dynamic_layout, tokenizer, prompt, device, scale=8.0):
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

        # === 执行态势锚定 (Gestalt Anchoring) ===
        res = int(np.sqrt(sequence_length))
        h, w = res, res
        tokens = self.tokenizer.encode(self.prompt)
        
        for item in self.layout:
            cls_id = int(item[0])
            keyword = self.class_to_keyword.get(cls_id, None)
            if not keyword: continue
            
            cx, cy, bw, bh = item[1], item[2], item[3], item[4]
            bx, by = item[5], item[6] if len(item) >= 7 else (0.0, 0.0)
            
            keyword_token_ids = self.tokenizer.encode(keyword, add_special_tokens=False)
            token_indices = [i for i, t in enumerate(tokens) if t in keyword_token_ids]
            
            if not token_indices: continue

            # 根据态势能计算非对称注意力 Mask
            x_c, y_c = (cx + bx * 0.15) * w, (cy + by * 0.15) * h
            x1, y1 = int(x_c - (bw/2)*w), int(y_c - (bh/2)*h)
            x2, y2 = int(x_c + (bw/2)*w), int(y_c + (bh/2)*h)
            
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                for idx in token_indices:
                    if idx >= attention_probs.shape[-1]: continue
                    mask = torch.zeros((h, w), device=self.device)
                    mask[y1:y2, x1:x2] = self.scale
                    mask_flat = mask.flatten()
                    attention_probs[:, :, idx] += mask_flat * attention_probs[:, :, idx]

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# =============================================================
# 推理主逻辑
# =============================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载 Layout Generator
    print("Loading Poem2Layout Generator...")
    layout_model = Poem2LayoutGenerator(
        bert_path="bert-base-chinese",
        num_classes=9,
        hidden_size=768,
        bb_size=128,
        decoder_layers=6,
        decoder_heads=8,
        latent_dim=64
    ).to(device).eval()
    
    ckpt = torch.load(args.layout_ckpt, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    layout_model.load_state_dict(state_dict, strict=False)
    
    # 2. 加载太乙管线 (V8.8 适配深度解冻权重)
    print("Loading Taiyi Stable Diffusion & ControlNet...")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.taiyi_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    
    # 加载 Rank 128 的强力 LoRA
    print(f"Loading LoRA weights from {args.lora_path}...")
    try:
        pipe.load_lora_weights(args.lora_path)
        print("✅ Shanshui Style LoRA loaded.")
    except Exception as e:
        print(f"⚠️ Failed to load LoRA: {e}")
    
    pipe.enable_model_cpu_offload()
    
    # 3. 初始化水墨 Mask 生成器 (必须开启 texture 模式)
    ink_gen = InkWashMaskGenerator(width=512, height=512)

    POEMS_TEST = [
        "大漠孤烟直，长河落日圆。", 
        "两个黄鹂鸣翠柳，一行白鹭上青天。",
        "忽如一夜春风来，千树万树梨花开。",
        "明月松间照，清泉石上流。"
    ] 

    print(f"Starting Inference for V8.8...")

    for i, poem in enumerate(tqdm(POEMS_TEST)):
        poem_clean = poem[:12].replace("，", "_").replace("。", "").strip()
        save_dir = os.path.join(args.output_base, f"{i+1:02d}_{poem_clean}")
        os.makedirs(save_dir, exist_ok=True)

        # Step 1. 生成布局
        tokenizer_layout = BertTokenizer.from_pretrained("bert-base-chinese") 
        layout_list = greedy_decode_poem_layout(layout_model, tokenizer_layout, poem, device=device)
        if not layout_list: continue
        layout = np.array(layout_list)

        # Step 2. 可视化基础框
        draw_layout(layout, f"Poem: {poem}", os.path.join(save_dir, "01_layout.png"))

        # Step 3. 转换为纹理 Mask (提供物理特征)
        mask_img = ink_gen.convert_boxes_to_mask(layout)
        mask_img.save(os.path.join(save_dir, "02_potential_field.png"))

        # Step 4. 架构注入
        attn_proc = PoemInkAttentionProcessor(
            dynamic_layout=layout, 
            tokenizer=pipe.tokenizer, 
            prompt=poem, 
            device=device,
            scale=8.0
        )
        pipe.unet.set_attn_processor(attn_proc)

        # Step 5. 纯净生成 (不带任何风格词)
        full_prompt = poem 
        
        # [CRITICAL] 极强的负向提示词：通过打压“自然图像”特征来迫使模型走向训练集画风
        neg_prompt = (
            "真实照片，摄影感，3D渲染，锐利边缘，现代感，鲜艳色彩，油画，水粉画，细节过度丰富，"
            "高对比度，写实主义，照片效果，人像摄影，广角镜头，电影感细节"
        )
        
        final_image = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            image=mask_img,
            num_inference_steps=35,
            controlnet_conditioning_scale=1.0,
            # [CRITICAL] 降低引导比例，给微调后的权重更多“山水偏向”的表达自由
            guidance_scale=5.8 
        ).images[0]
        
        final_image.save(os.path.join(save_dir, "03_final_painting.png"))

    print(f"Inference completed. Results saved to {args.output_base}")

if __name__ == "__main__":
    main()