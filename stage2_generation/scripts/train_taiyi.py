# File: stage2_generation/scripts/train_taiyi.py (V9.3: Final Stable Rank-32 Edition)

import argparse
import logging
import os
import math
import random
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="Idea-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1")
    parser.add_argument("--output_dir", type=str, default="taiyi_shanshui_v9_3_output")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4) 
    parser.add_argument("--num_train_epochs", type=int, default=40) 
    
    # [CONFIG] 学习率设置：针对 Rank 32 调优
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="ControlNet的学习率")
    parser.add_argument("--learning_rate_lora", type=float, default=1e-4, help="UNet LoRA学习率")
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16") 
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    
    # [STRATEGY] lambda_struct 设置为 0.05 以保证结构稳定性，防止杂乱
    parser.add_argument("--lambda_struct", type=float, default=0.05, help="结构对齐损失权重")
    
    # [ADAPTED] 核心修改：Rank 32 保证不全黑，Alpha 32 保证稳定
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA的秩 (调整为更稳健的 32)")
    parser.add_argument("--lora_alpha_ratio", type=float, default=1.0, help="LoRA Alpha/Rank 比例")
    
    # [NEW] 回归 Smart Freeze 逻辑，保护原生清晰度，防止变糊
    parser.add_argument("--smart_freeze", action="store_true", default=True, help="默认为True：保护原生SD权重，仅训练侧路")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        log_file = os.path.join(args.output_dir, "train_loss_history.txt")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.logger.addHandler(file_handler)
        logger.info(f"✨ [V9.3 最终稳定版] Rank-32 架构，保护原生清晰度，全功能开启！")

    # 1. 加载模型
    tokenizer = transformers.BertTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = transformers.BertModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # 初始化单流 ControlNet
    controlnet = ControlNetModel.from_unet(unet)

    # 2. 冻结策略与 LoRA 注入
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False) 
    
    # [ENHANCE] LoRA 依然覆盖卷积层，以补偿不再解冻原生 Up-Blocks 带来的画风损失
    lora_alpha = args.lora_rank * args.lora_alpha_ratio
    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj", "conv1", "conv2", "conv_shortcut"],
    )
    unet = get_peft_model(unet, unet_lora_config)
    
    if accelerator.is_main_process:
        print(f"✅ LoRA 注入成功 (Rank={args.lora_rank}, Alpha={lora_alpha})")
        unet.print_trainable_parameters()

    # 显存优化
    try:
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
        controlnet.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()
    except Exception:
        pass

    # =========================================================
    # ControlNet 智能冻结逻辑
    # =========================================================
    if args.smart_freeze:
        controlnet.requires_grad_(False) 
        for n, p in controlnet.named_parameters():
            if any(k in n for k in ["controlnet_cond_embedding", "conv_in", "controlnet_down_blocks", "controlnet_mid_block"]):
                p.requires_grad = True
        if accelerator.is_main_process:
            print(f"❄️ [Smart Freeze] 启用：保护原生底座，仅微调侧路层。")
    else:
        controlnet.requires_grad_(True)

    # 3. 优化器 (管理 ControlNet 侧路与 UNet LoRA)
    params_to_optimize = [
        {"params": filter(lambda p: p.requires_grad, controlnet.parameters()), "lr": args.learning_rate},
        {"params": filter(lambda p: p.requires_grad, unet.parameters()), "lr": args.learning_rate_lora} 
    ]
    optimizer = torch.optim.AdamW(params_to_optimize)

    # 4. 数据加载逻辑 (完整保留)
    raw_dataset = load_dataset("json", data_files=os.path.join(args.train_data_dir, "train.jsonl"))["train"]
    train_testvalid = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_testvalid['train']
    val_dataset = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)['train']

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
        pixel_values, cond_pixel_values, input_ids, raw_texts = [], [], [], []
        for example in examples:
            try:
                img_path = os.path.join(args.train_data_dir, example["image"])
                cond_path = os.path.join(args.train_data_dir, example["conditioning_image"])
                pixel_values.append(transform(Image.open(img_path).convert("RGB")))
                cond_pixel_values.append(cond_transform(Image.open(cond_path).convert("RGB")))
                caption = example["text"]
                inputs = tokenizer(caption, max_length=tokenizer.model_max_length, 
                                 padding="max_length", truncation=True, return_tensors="pt")
                input_ids.append(inputs.input_ids[0])
                raw_texts.append(example["text"])
            except: continue
        return {
            "pixel_values": torch.stack(pixel_values),
            "conditioning_pixel_values": torch.stack(cond_pixel_values),
            "input_ids": torch.stack(input_ids),
            "texts": raw_texts
        }

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    controlnet, unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        controlnet, unet, optimizer, train_dataloader, val_dataloader
    )
    
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)

    empty_tokens = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    empty_input_ids = empty_tokens.input_ids.to(device)

    loss_history = {'steps': [], 'total': [], 'mse': [], 'struct': []}

    def plot_loss_curve(history, save_path):
        if len(history['steps']) < 2: return
        plt.figure(figsize=(10, 6))
        plt.plot(history['steps'], history['total'], label='Total Loss')
        plt.plot(history['steps'], history['mse'], label='MSE (Texture)')
        plt.plot(history['steps'], history['struct'], label='Struct (Layout)')
        plt.title(f"Shanshui V9.3 Training Loss History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        try:
            plt.savefig(save_path)
            plt.close()
        except: pass

    # 5. 训练循环 (完整保留 Double Dropout)
    global_step = 0
    for epoch in range(args.num_train_epochs):
        controlnet.train()
        unet.train()
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet, unet):
                target_images = batch["pixel_values"].to(dtype=torch.float16)
                latents = vae.encode(target_images).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
                scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                cond_image = batch["conditioning_pixel_values"].to(dtype=torch.float16)
                bsz = latents.shape[0]

                # Double Dropout 策略
                rand_dropout = random.random()
                if rand_dropout < 0.15: 
                    cond_input = torch.zeros_like(cond_image)
                    current_input_ids = batch["input_ids"]
                    use_struct_loss = False
                elif rand_dropout < 0.30: 
                    cond_input = cond_image
                    current_input_ids = empty_input_ids.repeat(bsz, 1)
                    use_struct_loss = True
                else: 
                    cond_input = cond_image
                    current_input_ids = batch["input_ids"]
                    use_struct_loss = True
                
                encoder_hidden_states = text_encoder(current_input_ids)[0]
                
                # ControlNet 前向
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, timesteps, encoder_hidden_states, cond_input, return_dict=False
                )

                # UNet 前向
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states, 
                    down_block_additional_residuals=[sample.to(dtype=torch.float16) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float16)
                ).sample

                loss_ddpm = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                loss_struct = torch.tensor(0.0).to(device)
                if use_struct_loss: 
                    cond_resized = F.interpolate(cond_input, size=mid_block_res_sample.shape[-2:], mode="bilinear")
                    loss_struct = F.l1_loss(mid_block_res_sample.mean(dim=1, keepdim=True), cond_resized.mean(dim=1, keepdim=True))
                
                total_loss = loss_ddpm + args.lambda_struct * loss_struct
                
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            # 日志与 Checkpoint
            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                os.makedirs(ckpt_dir, exist_ok=True)
                accelerator.unwrap_model(controlnet).save_pretrained(ckpt_dir / "controlnet_structure") 
                accelerator.unwrap_model(unet).save_pretrained(ckpt_dir / "unet_lora")

            if step % 10 == 0 and accelerator.is_main_process:
                loss_history['steps'].append(global_step)
                loss_history['total'].append(total_loss.item())
                loss_history['mse'].append(loss_ddpm.item())
                loss_history['struct'].append(loss_struct.item())
                print(f"Epoch {epoch+1} | Step {step} | Total Loss: {total_loss.item():.4f}")
                if step % 100 == 0: plot_loss_curve(loss_history, os.path.join(args.output_dir, "loss_curve.png"))

        # [完整保留：验证采样逻辑] 输出 Mask + Sample + Prompt Log
        if accelerator.is_main_process:
            controlnet.eval(); unet.eval()
            try:
                with torch.autocast(device.type, dtype=torch.float16), torch.no_grad():
                    unwrapped_net = accelerator.unwrap_model(controlnet)
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    pipe = StableDiffusionControlNetPipeline(
                        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                        unet=unwrapped_unet, controlnet=unwrapped_net,
                        scheduler=scheduler, safety_checker=None, feature_extractor=None
                    ).to(device)
                    
                    test_batch = next(iter(val_dataloader))
                    test_cond = test_batch["conditioning_pixel_values"][0:1].to(device=device, dtype=torch.float16)
                    test_prompt = test_batch["texts"][0]
                    
                    # 1. 保存对应的 Conditioning Mask
                    mask_pil = transforms.ToPILImage()(test_cond[0].cpu())
                    mask_pil.save(Path(args.output_dir) / f"val_epoch_{epoch+1}_mask.png")
                    
                    # 2. 生成样例 (采样步数设为稳健的 50 步)
                    sample_out = pipe(prompt=test_prompt, image=test_cond, num_inference_steps=50, guidance_scale=7.5).images[0]
                    sample_out.save(Path(args.output_dir) / f"val_epoch_{epoch+1}_sample.png")
                    
                    # 3. 记录日志
                    with open(os.path.join(args.output_dir, "validation_log.txt"), "a") as f:
                        f.write(f"Epoch {epoch+1} | Prompt: {test_prompt}\n")
                    
                    print(f"✅ Epoch {epoch+1} 验证完成。")
                    del pipe; torch.cuda.empty_cache()
            except Exception as e: print(f"采样失败: {e}")

    if accelerator.is_main_process:
        accelerator.unwrap_model(controlnet).save_pretrained(Path(args.output_dir) / "controlnet_structure")
        accelerator.unwrap_model(unet).save_pretrained(Path(args.output_dir) / "unet_lora")
        print(f"✅ V9.3 训练全流程完成。")

if __name__ == "__main__":
    main()