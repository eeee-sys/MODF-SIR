#!/bin/bash
# ============================================================
# train_planner.sh
# 启动 Planner LoRA SFT 训练的脚本
# ============================================================

set -e

# ── 项目路径 ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 模型参数 ────────────────────────────────────────────────────
MODEL_NAME_OR_PATH="model_zoo/Qwen2.5-Omni-7B"

# ── 数据参数 ────────────────────────────────────────────────────
DATA_CONFIG_DIR="${PROJECT_DIR}/data_config"
TRAIN_DATA="${DATA_CONFIG_DIR}/planner_train_data.json"
VIDEO_FOLDER="IntentTrain/videos"

# ── 输出目录 ────────────────────────────────────────────────────
OUTPUT_DIR="lora_ms"

# ── 视频处理参数 ────────────────────────────────────────────────
RAW_VIDEO=false               # true: 直接输入原视频，不预处理；false: 按下方参数预处理
MAX_FRAMES=150
FPS=1.0
MIN_PIXELS=28224              # 36 * 28 * 28
MAX_PIXELS=50176              # 64 * 28 * 28

# ── LoRA 参数 ───────────────────────────────────────────────────
LORA_R=64
LORA_ALPHA=64
LORA_DROPOUT=0.1
LORA_BIAS="none"
LORA_TARGET_MODULES="q_proj k_proj v_proj o_proj"

# ── 训练参数 ────────────────────────────────────────────────────
NUM_TRAIN_EPOCHS=1
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-5
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01
LR_SCHEDULER="cosine"
LOGGING_STEPS=5
SAVE_STEPS=200
SAVE_TOTAL_LIMIT=3
DATALOADER_NUM_WORKERS=4      # DataLoader 工作进程数

# ── GPU 配置 ────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="0,1,2,3"  # 指定使用的 GPU 编号，逗号分隔
NUM_GPUS=4                       # 使用的 GPU 数量（需与上面一致）

# ── DeepSpeed 配置 ──────────────────────────────────────────────
# 选项: zero2.json (推荐), zero3.json (更省显存但稍慢), 或留空不使用
# 注意：单卡训练时建议留空，多卡训练时启用
if [ ${NUM_GPUS} -gt 1 ]; then
    DEEPSPEED_CONFIG="${PROJECT_DIR}/run_scripts/zero2.json"
else
    DEEPSPEED_CONFIG=""  # 单卡不使用 DeepSpeed
fi

# ── 启动训练 ────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo " Planner LoRA SFT 训练"
echo " 模型:       ${MODEL_NAME_OR_PATH}"
echo " 数据:       ${TRAIN_DATA}"
echo " 输出目录:   ${OUTPUT_DIR}"
echo " 使用 GPU:   ${CUDA_VISIBLE_DEVICES}"
echo " GPU 数量:   ${NUM_GPUS}"
echo " DeepSpeed:  ${DEEPSPEED_CONFIG}"
echo "=================================================================="
echo ""

# 设置可见 GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"

# 构建通用命令参数
CMD_ARGS=(
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --data_files "${TRAIN_DATA}"
    --video_folder "${VIDEO_FOLDER}"
    --max_frames ${MAX_FRAMES}
    --fps ${FPS}
    --min_pixels ${MIN_PIXELS}
    --max_pixels ${MAX_PIXELS}
    --lora_r ${LORA_R}
    --lora_alpha ${LORA_ALPHA}
    --lora_dropout ${LORA_DROPOUT}
    --lora_bias "${LORA_BIAS}"
    --lora_target_modules ${LORA_TARGET_MODULES}
    --num_train_epochs ${NUM_TRAIN_EPOCHS}
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE}
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
    --learning_rate ${LEARNING_RATE}
    --warmup_ratio ${WARMUP_RATIO}
    --weight_decay ${WEIGHT_DECAY}
    --lr_scheduler_type "${LR_SCHEDULER}"
    --logging_steps ${LOGGING_STEPS}
    --save_steps ${SAVE_STEPS}
    --save_total_limit ${SAVE_TOTAL_LIMIT}
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS}
    --bf16
    --gradient_checkpointing
    --debug_print_samples 3
)
if [ "${RAW_VIDEO}" = true ]; then
    CMD_ARGS+=(--raw_video)
fi
if [ -n "${DEEPSPEED_CONFIG}" ] && [ -f "${DEEPSPEED_CONFIG}" ]; then
    CMD_ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

if [ ${NUM_GPUS} -gt 1 ]; then
    # 多卡分布式训练
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=29500 \
        "${SCRIPT_DIR}/train_planner.py" \
        "${CMD_ARGS[@]}"
else
    # 单卡训练
    python "${SCRIPT_DIR}/train_planner.py" \
        "${CMD_ARGS[@]}"
fi

echo ""
echo "=============================================="
echo " Planner LoRA 训练完成！"
echo " LoRA 权重已保存到: ${OUTPUT_DIR}/Planner"
echo "=============================================="
