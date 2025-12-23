#!/bin/bash

# ================= 配置区域 =================
export CUDA_VISIBLE_DEVICES=0
# 某些显卡可能需要这个环境变量来避免碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 项目根目录 (自动获取)
PROJECT_ROOT=$(pwd)

# [缓存与输出]
export HF_HOME="$PROJECT_ROOT/.hf_cache"
mkdir -p "$HF_HOME"
# 建议输出目录使用绝对路径或明确的相对路径
OUTPUT_DIR="/home/610-sty/layout2paint3/outputs/taiyi_ink_controlnet_v2"

# [模型路径]
MODEL_NAME="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"

# [数据路径配置]
DATA_DIR="/home/610-sty/layout2paint3/taiyi_dataset_v8_real_gestalt" 

# Accelerate 配置
ACCELERATE_CONFIG="stage2_generation/configs/accelerate_config.yaml"

# ===========================================

# 1. 检查数据元数据是否存在
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "❌ 错误: 在 $DATA_DIR 中找不到 train.jsonl"
    echo "请先运行: python stage2_generation/scripts/prepare_data_taiyi.py"
    exit 1
fi

# 2. 检查/生成 Accelerate 配置
if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "⚠️ 生成默认 Accelerate 配置..."
    mkdir -p $(dirname "$ACCELERATE_CONFIG")
    accelerate config default --config_file "$ACCELERATE_CONFIG"
fi

# 3. 开始训练
echo "========================================================"
echo "🚀 启动 Stage 2 训练 (改进版双流 ControlNet)"
echo "   基础模型: $MODEL_NAME"
echo "   数据目录: $DATA_DIR"
echo "   输出目录: $OUTPUT_DIR"
echo "   分辨率: 512 | 混合精度: fp16"
echo "========================================================"

# [修改点] 移除了不支持的 --lambda_perceptual 参数
accelerate launch --config_file "$ACCELERATE_CONFIG" --mixed_precision="fp16" stage2_generation/scripts/train_taiyi.py \
 --pretrained_model_name_or_path="$MODEL_NAME" \
 --train_data_dir="$DATA_DIR" \
 --output_dir="$OUTPUT_DIR" \
 --resolution=512 \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --learning_rate=1e-5 \
 --num_train_epochs=20 \
 --checkpointing_steps=2000 \
 --lambda_struct=0.1 \
 --mixed_precision="fp16"