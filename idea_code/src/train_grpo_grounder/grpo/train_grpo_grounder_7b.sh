#!/bin/bash

set -e

# ========== GPU 配置 ==========
# 用法: bash scripts/grpo/train_grpo_grounder_7b.sh [GPU_IDS] [NPROC_PER_GPU]
# 示例:
#   bash scripts/grpo/train_grpo_grounder_7b.sh 0,1,2,3        # 4张卡，每卡1进程
#   bash scripts/grpo/train_grpo_grounder_7b.sh 4,5,6,7 2      # 4张卡，每卡2进程
#   bash scripts/grpo/train_grpo_grounder_7b.sh 0               # 单卡
#   bash scripts/grpo/train_grpo_grounder_7b.sh                  # 默认: 0,1,2,3,4,5,6,7

GPU_IDS="${1:-0,1,2,3,4,5,6,7}"
NPROC_PER_GPU="${2:-1}"

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"

# 计算总进程数 = GPU数量 × 每卡进程数
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}
NPROC_TOTAL=$((NUM_GPUS * NPROC_PER_GPU))

# ========== Model Paths ==========
BASE_MODEL="model_zoo/Qwen2-VL-7B-Instruct"
PRETRAINED_GROUNDER="model_zoo/VideoMind-7B"  # VideoMind预训练的Grounder
OUTPUT_DIR="work_dirs/grpo_grounder_7b"

# ========== GRPO Hyperparameters ==========
NUM_CANDIDATES=8           # N candidates per query
CLIP_EPSILON=0.2          # PPO-style clipping parameter
KL_BETA=0.01              # KL divergence penalty coefficient
OLD_POLICY_SYNC=50        # Sync old policy every N steps

# ========== Training Hyperparameters ==========
LEARNING_RATE=5e-5
NUM_EPOCHS=2
BATCH_SIZE=1
GRAD_ACCUM=4

# ========== Dataset ==========
DATASET="qvhighlights,didemo,internvid_vtime,queryd,tacos"    # Start with QVHighlights as recommended

# ========== 日志文件设置 ==========
# 将所有控制台输出（stdout + stderr）同时写入日志文件
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/train_${TIMESTAMP}.log"
echo "日志文件: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "GRPO Training for VideoMind Grounder (7B)"
echo "=========================================="
echo "GPU IDs: $GPU_IDS (共 ${NUM_GPUS} 张卡)"
echo "每卡进程数: $NPROC_PER_GPU"
echo "总进程数: $NPROC_TOTAL"
echo "Base Model: $BASE_MODEL"
echo "Pretrained Grounder: $PRETRAINED_GROUNDER"
echo "Output Directory: $OUTPUT_DIR"
echo "GRPO Hyperparameters:"
echo "  - Num Candidates: $NUM_CANDIDATES"
echo "  - Clip Epsilon: $CLIP_EPSILON"
echo "  - KL Beta: $KL_BETA"
echo "  - Old Policy Sync: every $OLD_POLICY_SYNC steps"
echo "=========================================="

torchrun --nproc_per_node $NPROC_TOTAL train_grpo_grounder/grpo_train.py \
    --deepspeed train_grpo_grounder/grpo/zero2.json \
    --base_model_path $BASE_MODEL \
    --pretrained_grounder_path $PRETRAINED_GROUNDER \
    --base_model qwen2_vl \
    --conv_type chatml \
    --role grounder \
    --lora_enable True \
    --lora_type qkvo \
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_bias none \
    --tuning_modules none \
    --num_candidates $NUM_CANDIDATES \
    --clip_epsilon $CLIP_EPSILON \
    --kl_beta $KL_BETA \
    --old_policy_sync_interval $OLD_POLICY_SYNC \
    --datasets $DATASET \
    --min_video_len 5 \
    --max_video_len 500 \
    --max_num_words 200 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --output_dir $OUTPUT_DIR \
    --save_full_model False \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --bf16 True \
    --report_to tensorboard

echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
