#!/bin/bash

# ================= 配置区域 =================
# 指定 GPU
export CUDA_VISIBLE_DEVICES=0

# 优化显存分配策略，防止碎片化导致 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 项目根目录 (自动获取当前脚本的上上级目录，假设脚本在 stage2_generation/ 中)
# 如果脚本直接在根目录运行，可以使用 $(pwd)
PROJECT_ROOT=$(pwd)

# [缓存设置] 防止撑爆系统盘
export HF_HOME="$PROJECT_ROOT/.hf_cache"
mkdir -p "$HF_HOME"

# [输出与数据路径] (已更新为你的 layout2paint3 路径)
OUTPUT_DIR="/home/610-sty/layout2paint3/outputs/taiyi_ink_controlnet_v8_7_hard_binding"
DATA_DIR="/home/610-sty/layout2paint3/taiyi_dataset_v8_real_gestalt" 

# [基础模型路径]
MODEL_NAME="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"

# Accelerate 配置文件路径
ACCELERATE_CONFIG="stage2_generation/configs/accelerate_config.yaml"

# ===========================================

# 1. 安全检查：确保数据元数据存在
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "❌ 错误: 在 $DATA_DIR 中找不到 train.jsonl"
    echo "   请先运行: python stage2_generation/scripts/prepare_data_taiyi.py"
    exit 1
fi

# 2. 安全检查：生成默认 Accelerate 配置 (如果不存在)
if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "⚠️ 未检测到 Accelerate 配置，正在生成默认 fp16 配置..."
    mkdir -p $(dirname "$ACCELERATE_CONFIG")
    # 自动生成一个适合单卡的 fp16 配置
    cat > "$ACCELERATE_CONFIG" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: NO
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
fi

# 3. 启动训练 (V8.7 配置)
echo "========================================================"
echo "🚀 启动 Stage 2 V8.7 训练 (双向 Dropout + 强风格绑定)"
echo "   基础模型: $MODEL_NAME"
echo "   数据目录: $DATA_DIR"
echo "   输出目录: $OUTPUT_DIR"
echo "   配置亮点: LoRA Rank=64 | Alpha Ratio=2.0 | Smart Freeze"
echo "========================================================"

accelerate launch --config_file "$ACCELERATE_CONFIG" --mixed_precision="fp16" stage2_generation/scripts/train_taiyi.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATA_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --num_train_epochs=20 \
  --checkpointing_steps=2000 \
  --mixed_precision="fp16" \
  \
  --learning_rate=1e-5 \
  --learning_rate_lora=1e-4 \
  \
  --lambda_struct=0.1 \
  \
  --lora_rank=64 \
  --lora_alpha_ratio=2.0 \
  \
  --smart_freeze 

echo "✅ 训练脚本执行完毕。请检查日志: $OUTPUT_DIR/train_loss_history.txt"