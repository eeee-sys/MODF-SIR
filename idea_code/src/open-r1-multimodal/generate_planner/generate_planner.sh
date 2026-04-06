#!/bin/bash
# ============================================================
# generate_planner.sh
# 单进程多 GPU 模型并行启动 generate_planner.py 的启动脚本
# 模型通过 device_map="auto" 分片加载到多张 GPU 上
# ============================================================

set -e

MODEL_PATH="Qwen/Qwen3-Omni-30B-A3B-Thinking"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DATA_CONFIG_DIR="${PROJECT_DIR}/data_config"
EMER_INPUT="${DATA_CONFIG_DIR}/emer_rewrite.json"
EMER_OUTPUT="${DATA_CONFIG_DIR}/emer_rewrite_planner_path.json"
SOCIAL_INPUT="${DATA_CONFIG_DIR}/social_iq_v2_rewrite.json"
SOCIAL_OUTPUT="${DATA_CONFIG_DIR}/social_iq_v2_rewrite_planner_path.json"

VIDEO_FOLDER="${PROJECT_DIR}/videos"

GPU_IDS="0,1,2,3,4,5,6,7"

RAW_VIDEO=false         
MAX_FRAMES=150
FPS=1.0
MIN_PIXELS=28224            
MAX_PIXELS=50176              
MAX_NEW_TOKENS=512

# ── 运行推理 ────────────────────────────────────────────────────
run_inference() {
    local INPUT_FILE=$1
    local OUTPUT_FILE=$2
    local DATASET_NAME=$3

    echo ""
    echo "=================================================================="
    echo " 处理数据集: ${DATASET_NAME}"
    echo " 输入文件:   ${INPUT_FILE}"
    echo " 输出文件:   ${OUTPUT_FILE}"
    echo " 使用 GPU:   ${GPU_IDS}"
    echo "=================================================================="

    CMD_ARGS=(
        --model_path "${MODEL_PATH}"
        --input_file "${INPUT_FILE}"
        --output_file "${OUTPUT_FILE}"
        --video_folder "${VIDEO_FOLDER}"
        --gpu_ids "${GPU_IDS}"
        --max_frames ${MAX_FRAMES}
        --fps ${FPS}
        --min_pixels ${MIN_PIXELS}
        --max_pixels ${MAX_PIXELS}
        --max_new_tokens ${MAX_NEW_TOKENS}
    )
    if [ "${RAW_VIDEO}" = true ]; then
        CMD_ARGS+=(--raw_video)
    fi

    python "${SCRIPT_DIR}/generate_planner.py" "${CMD_ARGS[@]}" \
        2>&1 | tee "${SCRIPT_DIR}/generate_planner_${DATASET_NAME}.log"

    echo "🎉 [完成] ${DATASET_NAME} 处理完成: ${OUTPUT_FILE}"
    echo ""
}

run_inference "${EMER_INPUT}" "${EMER_OUTPUT}" "emer_rewrite"
run_inference "${SOCIAL_INPUT}" "${SOCIAL_OUTPUT}" "social_iq_v2_rewrite"

echo ""
echo "=============================================="
echo " 全部完成！"
echo " emer:      ${EMER_OUTPUT}"
echo " social_iq: ${SOCIAL_OUTPUT}"
echo "=============================================="
