#!/bin/bash

# ================= 配置区域 =================
# GPU 动态调度已启用，禁止手动设置 CUDA_VISIBLE_DEVICES 防止冲突
# export CUDA_VISIBLE_DEVICES=0 

# 优化显存分配策略
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT=$(pwd)

# [缓存设置]
export HF_HOME="$PROJECT_ROOT/.hf_cache"
mkdir -p "$HF_HOME"

# [输出与数据路径]
OUTPUT_DIR="/home/610-sty/layout2paint3/outputs/taiyi_shanshui_v9_3_rank32"
DATA_DIR="/home/610-sty/layout2paint3/taiyi_dataset_v8_8_deep_style" 

# [基础模型路径]
MODEL_NAME="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"

# Accelerate 配置文件路径
ACCELERATE_CONFIG="stage2_generation/configs/accelerate_config.yaml"

# ===========================================

# 1. 安全检查
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "❌ 错误: 在 $DATA_DIR 中找不到 train.jsonl"
    exit 1
fi

# 2. 检查 Accelerate 配置
if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "⚠️ 生成默认配置..."
    mkdir -p $(dirname "$ACCELERATE_CONFIG")
    cat > "$ACCELERATE_CONFIG" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: NO
mixed_precision: fp16
num_machines: 1
num_processes: 1
use_cpu: false
EOF
fi

# 3. 启动训练 (V9.3 稳健版)
echo "========================================================"
echo "🚀 启动 Stage 2 V9.3 训练 (Rank 32 稳定版)"
echo "   策略亮点: LoRA Rank=32 | lambda_struct=0.05 | Smart Freeze"
echo "========================================================"

accelerate launch --config_file "$ACCELERATE_CONFIG" --mixed_precision="fp16" stage2_generation/scripts/train_taiyi.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATA_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=40 \
  --checkpointing_steps=10000 \
  --mixed_precision="fp16" \
  \
  --learning_rate=2e-5 \
  --learning_rate_lora=1e-4 \
  \
  --lambda_struct=0.05 \
  \
  --lora_rank=32 \
  --lora_alpha_ratio=1.0 \
  \
  --smart_freeze

echo "✅ 训练脚本执行完毕。日志: $OUTPUT_DIR/train_loss_history.txt"