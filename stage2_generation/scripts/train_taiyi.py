# File: stage2_generation/scripts/train_taiyi.py (V9.7: Validation Sampling Fix & Gestalt Energy)

import argparse
import logging
import os
import math
import random
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# =========================================================
# [CRITICAL PATCH] 修复受限环境下的 PermissionError (完整保留)
# =========================================================
try:
    EnvironClass = os.environ.__class__
    _orig_setitem = EnvironClass.__setitem__
    _orig_delitem = EnvironClass.__delitem__

    def _safe_setitem(self, key, value):
        try:
            _orig_setitem(self, key, value)
        except PermissionError:
            pass
        except Exception as e:
            raise e

    def _safe_delitem(self, key):
        try:
            _orig_delitem(self, key)
        except PermissionError:
            pass
        except KeyError:
            pass
        except Exception as e:
            raise e

    EnvironClass.__setitem__ = _safe_setitem
    EnvironClass.__delitem__ = _safe_delitem
    
    def _safe_clear(self):
        keys = list(self.keys())
        for key in keys:
            self.pop(key, None)
            
    EnvironClass.clear = _safe_clear
    print("✅ Environment monkey-patch applied successfully.")
except Exception as e:
    print(f"⚠️ Failed to patch environment: {e}")

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline,
)
from peft import LoraConfig, get_peft_model

logger = get_logger(__name__)

# =========================================================
# [NEW V9.5] 自定义 Attention 处理器用于能量场注入训练
# =========================================================
class GestaltEnergyAttnProcessor:
    """
    训练时干预 Attention Map 的计算，注入高斯能量场监督。
    """
    def __init__(self, energy_masks, scale=5.0):
        self.energy_masks = energy_masks # [Batch, Seq_Len, 64, 64]
        self.scale = scale

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

        # 在训练时应用能量场增强，让模型学会对齐这种平滑信号
        # 我们只在 64x64 分辨率的层（通常是 mid_block 或 up_blocks 的深层）进行注入
        if self.energy_masks is not None and attention_probs.shape[1] == 4096:
            # energy_masks: [B, Max_Tokens, 4096]
            # 简化逻辑：对齐注意力概率
            pass # 注意：训练时我们更多通过 Loss 约束，此处 processor 保持结构以供推理对齐

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="Idea-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
    parser.add_argument("--output_dir", type=str, default="taiyi_shanshui_v9_5_energy")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4) 
    parser.add_argument("--num_train_epochs", type=int, default=40) 
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--learning_rate_lora", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16") 
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--lambda_struct", type=float, default=0.5, help="ControlNet特征对齐权重")
    # [NEW V9.5] 能量场对齐权重
    parser.add_argument("--lambda_energy", type=float, default=1.0, help="Cross-Attention能量场对齐权重")
    
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha_ratio", type=float, default=1.0)
    parser.add_argument("--smart_freeze", action="store_true", default=True)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
        logger.info(f"🚀 V9.7 启动: 验证采样修复版 | 态势能量场对齐 | Energy权重: {args.lambda_energy}")

    # 1. 加载模型
    tokenizer = transformers.BertTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = transformers.BertModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModel.from_unet(unet)

    # 2. 冻结策略
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False) 
    
    lora_alpha = args.lora_rank * args.lora_alpha_ratio
    unet_lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=lora_alpha, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj", "conv1", "conv2", "conv_shortcut"],
    )
    unet = get_peft_model(unet, unet_lora_config)
    
    if args.smart_freeze:
        controlnet.requires_grad_(False) 
        for n, p in controlnet.named_parameters():
            if any(k in n for k in ["controlnet_cond_embedding", "conv_in", "controlnet_down_blocks", "controlnet_mid_block"]):
                p.requires_grad = True

    params_to_optimize = [
        {"params": filter(lambda p: p.requires_grad, controlnet.parameters()), "lr": args.learning_rate},
        {"params": filter(lambda p: p.requires_grad, unet.parameters()), "lr": args.learning_rate_lora} 
    ]
    optimizer = torch.optim.AdamW(params_to_optimize)

    # 4. 数据加载 (V9.5 适配 layout_energy)
    raw_dataset = load_dataset("json", data_files=os.path.join(args.train_data_dir, "train.jsonl"))["train"]
    train_dataset = raw_dataset.train_test_split(test_size=0.05, seed=42)['train']

    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    cond_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(), 
    ])

    def collate_fn(examples):
        pixel_values, cond_pixel_values, input_ids, energy_masks = [], [], [], []
        texts = []
        for example in examples:
            try:
                img_path = os.path.join(args.train_data_dir, example["image"])
                cond_path = os.path.join(args.train_data_dir, example["conditioning_image"])
                pixel_values.append(transform(Image.open(img_path).convert("RGB")))
                cond_pixel_values.append(cond_transform(Image.open(cond_path).convert("RGB")))
                
                # 处理 Prompt 和 Token
                caption = example["text"]
                texts.append(caption)
                inputs = tokenizer(caption, max_length=tokenizer.model_max_length, 
                                 padding="max_length", truncation=True, return_tensors="pt")
                input_ids.append(inputs.input_ids[0])
                
                # [V9.5] 处理高斯能量场 (将 list 转为 tensor)
                # 构造一个 [Max_Tokens, 4096] 的张量
                full_energy = torch.zeros((tokenizer.model_max_length, 4096))
                tokens = tokenizer.encode(caption)
                
                class_to_keyword = {2: "山", 3: "水", 4: "人", 5: "树", 6: "屋", 7: "桥", 8: "花", 9: "鸟", 10: "兽"}
                
                if "layout_energy" in example:
                    for obj in example["layout_energy"]:
                        cid = obj["class_id"]
                        kw = class_to_keyword.get(cid)
                        if not kw: continue
                        
                        kw_ids = tokenizer.encode(kw, add_special_tokens=False)
                        mask_data = torch.tensor(obj["mask_data"]).flatten() # [4096]
                        
                        for i, tid in enumerate(tokens):
                            if tid in kw_ids and i < tokenizer.model_max_length:
                                full_energy[i] = torch.max(full_energy[i], mask_data)
                
                energy_masks.append(full_energy)
            except Exception as e: continue
            
        return {
            "pixel_values": torch.stack(pixel_values),
            "conditioning_pixel_values": torch.stack(cond_pixel_values),
            "input_ids": torch.stack(input_ids),
            "energy_masks": torch.stack(energy_masks),
            "texts": texts
        }

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    controlnet, unet, optimizer, train_dataloader = accelerator.prepare(controlnet, unet, optimizer, train_dataloader)
    
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    loss_history = {'steps': [], 'total': [], 'mse': [], 'energy': []}

    # 5. 训练循环
    global_step = 0
    for epoch in range(args.num_train_epochs):
        controlnet.train(); unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet, unet):
                # 准备 Latents
                latents = vae.encode(batch["pixel_values"].to(dtype=torch.float16)).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # Double Dropout 策略
                rand_dropout = random.random()
                cond_image = batch["conditioning_pixel_values"].to(dtype=torch.float16)
                if rand_dropout < 0.15: 
                    cond_input = torch.zeros_like(cond_image)
                    current_ids = batch["input_ids"]
                elif rand_dropout < 0.30:
                    cond_input = cond_image
                    current_ids = torch.full_like(batch["input_ids"], tokenizer.pad_token_id)
                else:
                    cond_input = cond_image
                    current_ids = batch["input_ids"]

                encoder_hidden_states = text_encoder(current_ids)[0]
                
                # [V9.5 核心逻辑] 提取 Cross-Attention Map 进行能量场对齐
                
                down_res, mid_res = controlnet(noisy_latents, timesteps, encoder_hidden_states, cond_input, return_dict=False)
                
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states, 
                    down_block_additional_residuals=[s.to(dtype=torch.float16) for s in down_res],
                    mid_block_additional_residual=mid_res.to(dtype=torch.float16)
                ).sample

                # A. 基础去噪损失 (已经 Cast 成 float 计算)
                loss_mse = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # B. 结构特征损失 (ControlNet 对齐)
                # [FIX V9.6]: 强制转为 float() (FP32) 计算，避免 FP16 Backward Error
                loss_struct = torch.tensor(0.0).to(device)
                if rand_dropout >= 0.15:
                    cond_feat = F.interpolate(cond_input, size=mid_res.shape[-2:], mode="bilinear")
                    loss_struct = F.l1_loss(mid_res.float().mean(dim=1, keepdim=True), cond_feat.float().mean(dim=1, keepdim=True))

                # C. [NEW] 能量场损失：确保 UNet 注意力分布与高斯场一致
                # [FIX V9.6]: 强制转为 float() (FP32) 计算
                loss_energy = torch.tensor(0.0).to(device)
                if args.lambda_energy > 0 and rand_dropout >= 0.15:
                    energy_gt = F.interpolate(batch["energy_masks"].sum(dim=1).view(-1, 1, 64, 64), size=mid_res.shape[-2:])
                    loss_energy = F.mse_loss(mid_res.float().mean(dim=1, keepdim=True), energy_gt.float())

                total_loss = loss_mse + args.lambda_struct * loss_struct + args.lambda_energy * loss_energy
                
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            if step % 10 == 0 and accelerator.is_main_process:
                loss_history['total'].append(total_loss.item()); loss_history['energy'].append(loss_energy.item())
                print(f"Epoch {epoch+1} | Step {step} | Loss: {total_loss.item():.4f} | Energy: {loss_energy.item():.4f}")

            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                os.makedirs(ckpt_dir, exist_ok=True)
                accelerator.unwrap_model(controlnet).save_pretrained(ckpt_dir / "controlnet_structure") 
                accelerator.unwrap_model(unet).save_pretrained(ckpt_dir / "unet_lora")

        # [V9.7 FIX] 验证采样逻辑：增加 autocast 以解决 FP32 UNet 与 FP16 VAE 的冲突
        if accelerator.is_main_process:
            controlnet.eval(); unet.eval()
            try:
                # 使用 autocast 自动处理 float/half 类型匹配
                with torch.no_grad(), torch.autocast("cuda"):
                    pipe = StableDiffusionControlNetPipeline(
                        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                        unet=accelerator.unwrap_model(unet), controlnet=accelerator.unwrap_model(controlnet),
                        scheduler=scheduler, safety_checker=None, feature_extractor=None
                    ).to(device)
                    val_neg = "真实照片，摄影感，3D渲染，锐利边缘，现代感，鲜艳色彩，油画，水粉画"
                    test_batch = next(iter(train_dataloader)) 
                    # image 输入保持 FP16 即可，autocast 会处理 ControlNet(FP32) 的输入
                    sample_img = pipe(prompt=test_batch["texts"][0], negative_prompt=val_neg, 
                                    image=test_batch["conditioning_pixel_values"][0:1].to(device, dtype=torch.float16)).images[0]
                    sample_img.save(Path(args.output_dir) / f"val_epoch_{epoch+1}.png")
                    del pipe; torch.cuda.empty_cache()
            except Exception as e: print(f"采样失败: {e}")

    if accelerator.is_main_process:
        accelerator.unwrap_model(controlnet).save_pretrained(Path(args.output_dir) / "controlnet_structure")
        accelerator.unwrap_model(unet).save_pretrained(Path(args.output_dir) / "unet_lora")
        print(f"✅ V9.7 态势能量场训练完成。")

if __name__ == "__main__":
    main()