# File: integrated_inference.py (V7.2: 8-dim Gestalt Supported)

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
# 确保项目根目录在 PYTHONPATH 中，或者通过 sys.path.append 添加
from models.poem2layout import Poem2LayoutGenerator
from inference.greedy_decode import greedy_decode_poem_layout
from stage2_generation.utils.ink_mask import InkWashMaskGenerator
from data.visualize import draw_layout

# =============================================================
# 参数解析配置
# =============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Integrated Inference for Poem2Painting (V7.0)")
    
    # 模型路径参数
    parser.add_argument("--layout_ckpt", type=str, required=True, help="Path to the trained Poem2Layout V7.0 checkpoint")
    parser.add_argument("--taiyi_model_path", type=str, default="Idea-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1", help="Base Stable Diffusion model")
    
    # ControlNet & LoRA 路径
    parser.add_argument("--controlnet_seg_path", type=str, required=True, help="Path to Structure ControlNet")
    parser.add_argument("--controlnet_t_path", type=str, required=True, help="Path to Style ControlNet")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to trained LoRA weights")
    
    # 输出设置
    parser.add_argument("--output_base", type=str, default="outputs/integrated_inference", help="Directory to save results")
    
    return parser.parse_args()

# =============================================================
# 创新架构：跨模态交叉注意力态势锚定处理器 (Gestalt Attention Processor)
# =============================================================
class PoemInkAttentionProcessor:
    """
    底层架构创新：通过干预 Cross-Attention 层实现数学级语义绑定。
    [V7.0 更新]：支持态势能参数偏移，使注意力跟随墨迹扩散方向。
    """
    def __init__(self, dynamic_layout, tokenizer, prompt, device, scale=7.0):
        # dynamic_layout layout: numpy array or tensor [N, 9] 
        # (cls, cx, cy, w, h, bx, by, rot, flow)
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
        # 标准 Attention 流程
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
            # item 结构: [cls, cx, cy, w, h, bx, by, rot, flow]
            cls_id = int(item[0])
            keyword = self.class_to_keyword.get(cls_id, None)
            if not keyword: continue
            
            # 提取参数
            # cx, cy, w, h = item[1:5]
            cx, cy, bw, bh = item[1], item[2], item[3], item[4]
            
            # 提取态势偏移参数 (bx, by) -> item[5], item[6]
            if len(item) >= 7:
                bx, by = item[5], item[6]
            else:
                bx, by = 0.0, 0.0 # Fallback
            
            keyword_token_ids = self.tokenizer.encode(keyword, add_special_tokens=False)
            token_indices = [i for i, t in enumerate(tokens) if t in keyword_token_ids]
            
            if not token_indices: continue

            # [架构创新点]：根据态势能计算非对称注意力 Mask
            # 相比普通方框，这里加入了 (bx, by) 的中心偏移
            x_c, y_c = (cx + bx * 0.1) * w, (cy + by * 0.1) * h
            x1, y1 = int(x_c - (bw/2)*w), int(y_c - (bh/2)*h)
            x2, y2 = int(x_c + (bw/2)*w), int(y_c + (bh/2)*h)
            
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                for idx in token_indices:
                    if idx >= attention_probs.shape[-1]: continue
                    # 注意力场增强
                    mask = torch.zeros((h, w), device=self.device)
                    mask[y1:y2, x1:x2] = self.scale
                    mask_flat = mask.flatten()
                    
                    # 乘性增强，强制模型在渲染该区域时‘满脑子都是这个意象’
                    attention_probs[:, :, idx] += mask_flat * attention_probs[:, :, idx]

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# =============================================================
# 推理主逻辑适配
# =============================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Inference on: {device}")
    
    # 初始化模型配置
    config_path = "configs/default.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 1. 加载 Layout Generator (V7.0)
    print("Loading Poem2Layout Generator...")
    layout_model = Poem2LayoutGenerator(
        bert_path=config['model']['bert_path'],
        num_classes=config['model']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        bb_size=config['model']['bb_size'],
        decoder_layers=config['model']['decoder_layers'],
        decoder_heads=config['model']['decoder_heads'],
        latent_dim=config['model'].get('latent_dim', 64)
    ).to(device).eval()
    
    # 加载权重
    ckpt = torch.load(args.layout_ckpt, map_location=device)
    # 兼容处理：检查是否包含 'model_state_dict' 键
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    layout_model.load_state_dict(state_dict)
    
    # 2. 加载太乙管线 (Stable Diffusion + ControlNet)
    print("Loading Taiyi Stable Diffusion & ControlNets...")
    controlnet_seg = ControlNetModel.from_pretrained(args.controlnet_seg_path, torch_dtype=torch.float16)
    controlnet_t = ControlNetModel.from_pretrained(args.controlnet_t_path, torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.taiyi_model_path,
        controlnet=[controlnet_seg, controlnet_t],
        torch_dtype=torch.float16
    ).to(device)
    
    # 加载 LoRA
    print(f"Loading LoRA weights from {args.lora_path}...")
    pipe.load_lora_weights(args.lora_path)
    
    # 3. 初始化水墨 Mask 生成器
    ink_gen = InkWashMaskGenerator(width=512, height=512)

    # 模拟 50 首测试集 (或从文件读取)
    POEMS_50 = [
        "大漠孤烟直，长河落日圆。", 
        "两个黄鹂鸣翠柳，一行白鹭上青天。",
        "忽如一夜春风来，千树万树梨花开。"
    ] 

    print(f"Start Inference for {len(POEMS_50)} poems...")

    for i, poem in enumerate(tqdm(POEMS_50)):
        poem_clean = poem[:12].replace("，", "_").replace("。", "").strip()
        save_dir = os.path.join(args.output_base, f"{i+1:02d}_{poem_clean}")
        os.makedirs(save_dir, exist_ok=True)

        # ---------------------------------------------------------
        # Step 1. 生成 8 维动态布局 (Dynamic Layout Generation)
        # ---------------------------------------------------------
        tokenizer = BertTokenizer.from_pretrained(config['model']['bert_path'])
        layout_list = greedy_decode_poem_layout(layout_model, tokenizer, poem, device=device)
        
        if not layout_list:
            print(f"Warning: No layout generated for poem: {poem}")
            continue
            
        # [CRITICAL] 转换为 Numpy 数组，以便支持 slicing (layout[:, :5])
        # layout shape: [N, 9] -> (cls, cx, cy, w, h, bx, by, rot, flow)
        layout = np.array(layout_list)

        # ---------------------------------------------------------
        # Step 2. 可视化基础框
        # ---------------------------------------------------------
        # 这里传入全量 layout，依赖 visualize.py 中的鲁棒解包 (只取前5维)
        # 或者为了保险，显式切片: layout[:, :5]
        draw_layout(layout, f"Poem: {poem}", os.path.join(save_dir, "01_layout.png"))

        # ---------------------------------------------------------
        # Step 3. 转换为势能场 Mask (Ink Wash Potential Field)
        # ---------------------------------------------------------
        # ink_gen 需要支持 8 维输入来绘制带有扩散趋势的 mask
        mask_img = ink_gen.convert_boxes_to_mask(layout)
        mask_img.save(os.path.join(save_dir, "02_potential_field.png"))

        # ---------------------------------------------------------
        # Step 4. 架构注入：Gestalt Attention Processor
        # ---------------------------------------------------------
        attn_proc = PoemInkAttentionProcessor(
            dynamic_layout=layout, 
            tokenizer=pipe.tokenizer, 
            prompt=poem, 
            device=device,
            scale=8.0 # 强绑定系数
        )
        pipe.unet.set_attn_processor(attn_proc)

        # ---------------------------------------------------------
        # Step 5. 双流协同生成 (Dual-Stream Generation)
        # ---------------------------------------------------------
        # 提示词增强
        full_prompt = f"{poem}，写意水墨画，中国画风格，杰作，留白"
        neg_prompt = "低质量，模糊，色彩斑驳，边框，水印"
        
        final_image = pipe(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            image=[mask_img, mask_img], # 双流控制：结构流 + 风格流
            num_inference_steps=35,
            controlnet_conditioning_scale=[1.2, 0.8], # 结构流略强于风格流
            guidance_scale=7.5
        ).images[0]
        
        final_image.save(os.path.join(save_dir, "03_final_painting.png"))

    print(f"Inference completed. Results saved to {args.output_base}")

if __name__ == "__main__":
    main()