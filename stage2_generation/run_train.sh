#!/bin/bash

# ================= 配置区域 =================
# GPU 动态调度已启用
# export CUDA_VISIBLE_DEVICES=0 

# 优化显存分配策略 (防止 OOM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# [关键修复] 自动定位项目根目录
# 1. 获取脚本所在的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 2. 推断项目根目录 (假设脚本在 stage2_generation 目录下，根目录则是上一级)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 3. 强制切换工作目录到项目根目录
cd "$PROJECT_ROOT"
echo "📂 工作目录已自动切换至: $(pwd)"

# [缓存设置]
export HF_HOME="$PROJECT_ROOT/.hf_cache"
mkdir -p "$HF_HOME"

# [输出与数据路径]
# 注意：确保你已经运行了新的 prepare_data 脚本并生成了无方框的新数据
DATA_DIR="$PROJECT_ROOT/taiyi_energy_dataset_v9_2" 
OUTPUT_DIR="$PROJECT_ROOT/outputs/taiyi_shanshui_v14_fixed"

# [基础模型路径]
MODEL_NAME="/home/610-sty/huggingface/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"

# Accelerate 配置文件路径 (相对于 PROJECT_ROOT)
ACCELERATE_CONFIG="stage2_generation/configs/accelerate_config.yaml"

# ===========================================

# 1. 安全检查
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
  echo "❌ 错误: 在 $DATA_DIR 中找不到 train.jsonl"
  echo "👉 请先运行: python stage2_generation/scripts/prepare_data_taiyi.py"
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

# 3. 启动训练 (V14.2 终极修复版)
echo "========================================================"
echo "🚀 启动 Stage 2 V14.2 训练 (Clean-Mask & Stable-Joint)"
echo "   策略: LR平衡(2e-5) | 无模糊增强 | ControlNet全量解冻"
echo "========================================================"

# 注意：这里的路径是相对于 PROJECT_ROOT 的
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
  --learning_rate_lora=2e-5 \
  \
  --lambda_struct=0.0 \
  --lambda_energy=0.0 \
  \
  --lora_rank=32 \
  --lora_alpha_ratio=1.0 \
  \
  --snr_gamma=5.0 \
  --offset_noise_scale=0.1

# 注意：已移除 --smart_freeze，让 ControlNet 全量学习颜色语义

echo "✅ 训练脚本执行完毕。检查验证图: $OUTPUT_DIR"