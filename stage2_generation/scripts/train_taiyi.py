# File: stage2_generation/scripts/train_taiyi.py (V8.6: Single-Stream + Textured Mask)

import argparse
import logging
import os
import math
import random
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# =========================================================
# [CRITICAL PATCH] 修复受限环境下的 PermissionError
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
    parser.add_argument("--output_dir", type=str, default="taiyi_controlnet_lora_output")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4) 
    parser.add_argument("--num_train_epochs", type=int, default=10)
    
    # [CONFIG] 学习率设置
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="ControlNet的学习率")
    parser.add_argument("--learning_rate_lora", type=float, default=1e-4, help="UNet LoRA的学习率")
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16") 
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--lambda_struct", type=float, default=0.1, help="结构对齐损失权重")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA的秩")
    
    # [NEW] V8.6 智能冻结开关 (默认开启)
    parser.add_argument("--smart_freeze", action="store_true", default=True, help="开启智能冻结：只训练输入/输出层")
    
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
        logger.info(f"✨ [V8.6 单流纹理版] 启动！")
        logger.info(f"📝 日志文件: {log_file}")
        logger.info(f"📈 实时曲线: {os.path.join(args.output_dir, 'loss_curve.png')}")

    # 1. 加载模型
    tokenizer = transformers.BertTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = transformers.BertModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # [CHANGE] 初始化单流 ControlNet (不再有 controlnet_t)
    if accelerator.is_main_process:
        print("正在初始化单流 ControlNet (Structure Stream)...")
    controlnet = ControlNetModel.from_unet(unet)

    # 2. LoRA 设置 (负责风格)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False) 
    
    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    unet = get_peft_model(unet, unet_lora_config)
    
    if accelerator.is_main_process:
        print("✅ LoRA 注入成功 (负责水墨风格学习)")
        unet.print_trainable_parameters()

    # 显存优化
    try:
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
        # [方案一] 默认开启 Gradient Checkpointing 省显存
        controlnet.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()
    except Exception:
        pass

    # =========================================================
    # [V8.6 核心] 智能冻结逻辑 (Smart Freeze)
    # =========================================================
    if args.smart_freeze:
        controlnet.requires_grad_(False) # 先全冻结
        trainable_names = []
        
        # 1. 解冻输入层 (为了学会看纹理Mask)
        for n, p in controlnet.controlnet_cond_embedding.named_parameters():
            p.requires_grad = True
            trainable_names.append(n)
        for n, p in controlnet.conv_in.named_parameters():
            p.requires_grad = True
            trainable_names.append(n)
            
        # 2. 解冻输出层 (Zero Convolutions)
        for n, p in controlnet.controlnet_down_blocks.named_parameters():
            p.requires_grad = True
            trainable_names.append(n)
        for n, p in controlnet.controlnet_mid_block.named_parameters():
            p.requires_grad = True
            trainable_names.append(n)
            
        if accelerator.is_main_process:
            print(f"❄️ [Smart Freeze] 智能冻结已应用！仅训练 Adapter 层和 Zero Convolution (约 1.5亿参数)。")

    # 3. 优化器 (只优化解冻的参数)
    params_to_optimize = [
        {"params": filter(lambda p: p.requires_grad, controlnet.parameters()), "lr": args.learning_rate},
        {"params": unet.parameters(), "lr": args.learning_rate_lora} 
    ]
    optimizer = torch.optim.AdamW(params_to_optimize)

    # 4. 数据
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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    controlnet, unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        controlnet, unet, optimizer, train_dataloader, val_dataloader
    )
    
    vae.to(device, dtype=torch.float16)
    text_encoder.to(device, dtype=torch.float16)

    # Loss 记录容器
    loss_history = {'steps': [], 'total': [], 'mse': [], 'struct': []}

    def plot_loss_curve(history, save_path):
        if len(history['steps']) < 2: return
        plt.figure(figsize=(10, 6))
        plt.plot(history['steps'], history['total'], label='Total Loss', color='blue', alpha=0.6, linewidth=1)
        plt.plot(history['steps'], history['mse'], label='MSE Loss', color='orange', alpha=0.5, linestyle='--', linewidth=1)
        plt.plot(history['steps'], history['struct'], label='Struct Loss', color='green', alpha=0.8, linewidth=1.5)
        plt.title(f"Training Loss (Step {history['steps'][-1]})")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        try:
            plt.savefig(save_path)
            plt.close()
        except: pass

    # 5. 训练循环
    global_step = 0
    if accelerator.is_main_process:
        print(f"🚀 启动训练流程...")
        
    for epoch in range(args.num_train_epochs):
        controlnet.train()
        unet.train()
        
        train_loss_epoch = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet, unet):
                target_images = batch["pixel_values"].to(dtype=torch.float16)
                latents = vae.encode(target_images).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device).long()
                scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                cond_image = batch["conditioning_pixel_values"].to(dtype=torch.float16)
                
                # [CHANGE] 单流 Dropout 策略
                # 15% 概率完全丢弃 Condition，强迫 LoRA 学习 Text->Image 的映射
                rand_dropout = random.random()
                if rand_dropout < 0.15:
                    cond_input = torch.zeros_like(cond_image) # 空 Mask
                else:
                    cond_input = cond_image # 正常纹理 Mask
                
                # ControlNet 前向 (单流)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states, 
                    cond_input, 
                    return_dict=False
                )

                # UNet 前向 (接受 ControlNet 注入)
                model_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states, 
                    down_block_additional_residuals=[sample.to(dtype=torch.float16) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=torch.float16)
                ).sample

                # 基础生成损失 (MSE)
                loss_ddpm = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # 结构对齐损失 (Struct Loss)
                # 强制 ControlNet 的 mid_block 特征与输入 Mask 空间对齐
                # 只有在没 Dropout (有 Mask 输入) 时才计算
                loss_struct = torch.tensor(0.0).to(device)
                if rand_dropout >= 0.15: 
                    cond_resized = F.interpolate(cond_input, size=mid_block_res_sample.shape[-2:], mode="bilinear")
                    loss_struct = F.l1_loss(mid_block_res_sample.mean(dim=1, keepdim=True), cond_resized.mean(dim=1, keepdim=True))
                
                total_loss = loss_ddpm + args.lambda_struct * loss_struct
                
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss_epoch += total_loss.item()
            global_step += 1
            
            # Checkpoint 保存
            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
                os.makedirs(ckpt_dir, exist_ok=True)
                # 只保存一个 ControlNet
                accelerator.unwrap_model(controlnet).save_pretrained(ckpt_dir / "controlnet_structure") 
                # 保存 LoRA
                accelerator.unwrap_model(unet).save_pretrained(ckpt_dir / "unet_lora")
                print(f"💾 Checkpoint saved at step {global_step}")

            # 日志与绘图
            if step % 10 == 0 and accelerator.is_main_process:
                lr_c = optimizer.param_groups[0]['lr']
                lr_l = optimizer.param_groups[-1]['lr']
                
                loss_history['steps'].append(global_step)
                loss_history['total'].append(total_loss.item())
                loss_history['mse'].append(loss_ddpm.item())
                loss_history['struct'].append(loss_struct.item())
                
                msg = (f"Epoch {epoch+1}/{args.num_train_epochs} | Step {step} | "
                       f"Loss: {total_loss.item():.4f} (MSE: {loss_ddpm.item():.4f} / Struct: {loss_struct.item():.4f}) | "
                       f"LR: {lr_c:.1e}/{lr_l:.1e}")
                print(msg)
                logger.info(msg)
                
                if step % 100 == 0:
                    plot_loss_curve(loss_history, os.path.join(args.output_dir, "loss_curve.png"))

        # === 验证 ===
        if accelerator.is_main_process:
            print(f"🔍 Epoch {epoch}: 验证中...")
            plot_loss_curve(loss_history, os.path.join(args.output_dir, "loss_curve.png"))
        
        controlnet.eval()
        unet.eval()
        
        try:
            if accelerator.is_main_process:
                with torch.autocast(device.type, dtype=torch.float16):
                    with torch.no_grad():
                        unwrapped_net = accelerator.unwrap_model(controlnet)
                        unwrapped_unet = accelerator.unwrap_model(unet)
                        val_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
                        
                        # [CHANGE] Pipeline 只传一个 controlnet
                        pipe = StableDiffusionControlNetPipeline(
                            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                            unet=unwrapped_unet, 
                            controlnet=unwrapped_net, # 单流
                            scheduler=val_scheduler, safety_checker=None, feature_extractor=None
                        ).to(device)
                        
                        # 随机取一个batch做测试
                        test_batch = next(iter(val_dataloader))
                        test_cond = test_batch["conditioning_pixel_values"][0:1].to(device=device, dtype=torch.float16)
                        
                        # 保存输入Mask (Layout)
                        layout_img_pil = transforms.ToPILImage()(test_cond.squeeze(0).cpu())
                        layout_img_pil.save(Path(args.output_dir) / f"layout_epoch_{epoch}_val.png")

                        # 保存生成图 (Sample)
                        sample_out = pipe(
                            prompt="中国水墨山水画", # 固定Prompt测试稳定性
                            image=test_cond, 
                            num_inference_steps=20,
                            guidance_scale=7.5
                        ).images[0]
                        sample_out.save(Path(args.output_dir) / f"sample_epoch_{epoch}_val.png")
                        print(f"✅ 验证图已保存")
                        del pipe
                        torch.cuda.empty_cache()
        except Exception as e:
            print(f"验证采样失败: {e}")

    if accelerator.is_main_process:
        # 保存最终模型
        save_path_c = Path(args.output_dir) / "controlnet_structure"
        os.makedirs(save_path_c, exist_ok=True)
        accelerator.unwrap_model(controlnet).save_pretrained(save_path_c)
        accelerator.unwrap_model(unet).save_pretrained(Path(args.output_dir) / "unet_lora")
        
        plot_loss_curve(loss_history, os.path.join(args.output_dir, "loss_curve_final.png"))
        print(f"✅ 训练完成，Loss 曲线已保存。")

if __name__ == "__main__":
    main()