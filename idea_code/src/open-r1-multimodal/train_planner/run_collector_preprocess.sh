#!/bin/bash
# ============================================================
# run_collector_preprocess.sh
# ============================================================

set -e


MODEL_PATH="model_zoo/Qwen2.5-Omni-7B"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DATA_CONFIG_DIR="${PROJECT_DIR}/data_config"
SOCIAL_INPUT="${DATA_CONFIG_DIR}/planner_train_data_without_collector"
SOCIAL_OUTPUT="${DATA_CONFIG_DIR}/planner_train_data_with_collector"

VIDEO_FOLDER="IntentTrain/videos"

NUM_CHUNKS=4                
GPU_IDS=(2 4 5 7)         


RAW_VIDEO=false           
MAX_FRAMES=100
FPS=2.0
MIN_PIXELS=28224              # 36 * 28 * 28
MAX_PIXELS=100352              # 128 * 28 * 28
MAX_NEW_TOKENS=1024

# ── 运行推理 ────────────────────────────────────────────────────
run_collector() {
    local INPUT_FILE=$1
    local OUTPUT_FILE=$2
    local DATASET_NAME=$3

    echo ""
    echo "=================================================================="
    echo " Collector 预处理数据集: ${DATASET_NAME}"
    echo " 输入文件:   ${INPUT_FILE}"
    echo " 输出文件:   ${OUTPUT_FILE}"
    echo " GPU 数量:   ${NUM_CHUNKS}"
    echo "=================================================================="

    PIDS=()
    for ((i=0; i<NUM_CHUNKS; i++)); do
        GPU_ID=${GPU_IDS[$i]}
        echo "[启动] Chunk ${i}/${NUM_CHUNKS} on GPU ${GPU_ID}"

        CMD_ARGS=(
            --model_path "${MODEL_PATH}"
            --input_file "${INPUT_FILE}"
            --output_file "${OUTPUT_FILE}"
            --video_folder "${VIDEO_FOLDER}"
            --chunk_id ${i}
            --num_chunks ${NUM_CHUNKS}
            --device "cuda:0"
            --max_frames ${MAX_FRAMES}
            --fps ${FPS}
            --min_pixels ${MIN_PIXELS}
            --max_pixels ${MAX_PIXELS}
            --max_new_tokens ${MAX_NEW_TOKENS}
        )
        if [ "${RAW_VIDEO}" = true ]; then
            CMD_ARGS+=(--raw_video)
        fi

        CUDA_VISIBLE_DEVICES=${GPU_ID} CUDA_LAUNCH_BLOCKING=1 python "${SCRIPT_DIR}/run_collector_preprocess.py" \
            "${CMD_ARGS[@]}" \
            >> "${SCRIPT_DIR}/collector_preprocess_${DATASET_NAME}_${i}.log" 2>&1 &

        PIDS+=($!)
    done

    echo ""
    echo "[等待] 所有 ${NUM_CHUNKS} 个进程完成..."
    FAILED=0
    for ((i=0; i<${#PIDS[@]}; i++)); do
        wait ${PIDS[$i]}
        EXIT_CODE=$?
        if [ ${EXIT_CODE} -ne 0 ]; then
            echo "❌ [Chunk ${i}] 进程退出失败 (exit code: ${EXIT_CODE})"
            FAILED=1
        else
            echo "✅ [Chunk ${i}] 完成"
        fi
    done

    if [ ${FAILED} -eq 1 ]; then
        echo "⚠️ 部分进程失败，请检查日志后决定是否继续合并。"
    fi

    # 合并结果
    echo ""
    echo "[合并] 正在合并 ${NUM_CHUNKS} 个 chunk 结果到 ${OUTPUT_FILE} ..."
    CUDA_LAUNCH_BLOCKING=1 python "${SCRIPT_DIR}/run_collector_preprocess.py" \
        --merge \
        --output_file "${OUTPUT_FILE}" \
        --num_chunks ${NUM_CHUNKS}

    echo "🎉 [完成] ${DATASET_NAME} Collector 预处理完成: ${OUTPUT_FILE}"
    echo ""
}

# ── 处理 social_iq_v2_planner_path_select_900 数据集 ──────────────
run_collector "${SOCIAL_INPUT}" "${SOCIAL_OUTPUT}" "social_iq_v2"

echo ""
echo "=============================================="
echo " Collector 预处理完成！"
echo " social_iq: ${SOCIAL_OUTPUT}"
echo "=============================================="
