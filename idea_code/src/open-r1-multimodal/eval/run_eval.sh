#!/bin/bash

# ======== Model Paths ========
BASE_MODEL_PATH="model_zoo/Qwen2.5-Omni-7B"
PLANNER_LORA_PATH="model_zoo/Planner"
HUMANOMNI_PATH="model_zoo/HumanOmniV2"
GROUNDER_PATH="model_zoo/VideoMind-7B"
GRPO_ADAPTER_PATH="model_zoo/grpo_grounder"

# ======== Grounder Environment ========
GROUNDER_PYTHON="env_grounder/bin/python"

# ======== Data Paths ========
INPUT_FILE="IntentBench/qa.json"
VIDEO_ROOT="IntentBench/videos"
OUTPUT_FILE="eval_results/results.jsonl"
LORA_SAVE_DIR="lora/humanomni_lora"
ID_KEY="id,qid,video"
# ======== GPU Configuration ========
# Each process needs 3 GPUs: [main_gpu, humanomni_gpu, grounder_gpu]
# Configure based on available GPUs:

# Option 1: 1 process × 3 GPUs
MAIN_GPUS=(0)
HUMANOMNI_GPUS=(1)
GROUNDER_GPUS=(2)

# Option 2: 2 processes × 6 GPUs
# MAIN_GPUS=(0 3)
# HUMANOMNI_GPUS=(1 4)
# GROUNDER_GPUS=(2 5)

# Option 3: 4 processes × 12 GPUs (if available)
# MAIN_GPUS=(0 3 6 9)
# HUMANOMNI_GPUS=(1 4 7 10)
# GROUNDER_GPUS=(2 5 8 11)

NUM_CHUNKS=${#MAIN_GPUS[@]}

# ======== Hyperparameters ========
T_MAX=5
TAU=8
ALPHA=0.7
B0=5.0
LR=3e-4

export PYTHONPATH=./
export PYTHONUNBUFFERED=1

mkdir -p eval_results
mkdir -p eval_results/logs
mkdir -p "$LORA_SAVE_DIR"

echo "Starting ${NUM_CHUNKS} parallel processes..."

# Launch processes
for i in "${!MAIN_GPUS[@]}"; do
    MAIN_GPU=${MAIN_GPUS[$i]}
    HUMANOMNI_GPU=${HUMANOMNI_GPUS[$i]}
    GROUNDER_GPU=${GROUNDER_GPUS[$i]}

    echo "Process $i: main=cuda:$MAIN_GPU, humanomni=cuda:$HUMANOMNI_GPU, grounder=cuda:$GROUNDER_GPU"

    LOG_FILE="eval_results/logs/process_${i}.log"
    echo "Logging to: $LOG_FILE"

    python eval/eval_intentbench.py \
        --base_model_path "$BASE_MODEL_PATH" \
        --planner_lora_path "$PLANNER_LORA_PATH" \
        --humanomni_path "$HUMANOMNI_PATH" \
        --grounder_path "$GROUNDER_PATH" \
        --grpo_adapter_path "$GRPO_ADAPTER_PATH" \
        --grounder_python "$GROUNDER_PYTHON" \
        --grounder_cwd "$GROUNDER_CWD" \
        --input_file "$INPUT_FILE" \
        --video_root "$VIDEO_ROOT" \
        --output_file "$OUTPUT_FILE" \
        --lora_save_dir "$LORA_SAVE_DIR" \
        --main_gpu "cuda:$MAIN_GPU" \
        --humanomni_gpu "cuda:$HUMANOMNI_GPU" \
        --grounder_gpu "cuda:$GROUNDER_GPU" \
        --t_max $T_MAX \
        --tau $TAU \
        --alpha $ALPHA \
        --id $ID_KEY \
        --b0 $B0 \
        --lr $LR \
        --num_chunks $NUM_CHUNKS \
        --chunk_idx $i 2>&1 | tee "$LOG_FILE" &
done

# Wait for all processes
wait

echo "All processes completed!"

# Merge output files
if [ $NUM_CHUNKS -gt 1 ]; then
    echo "Merging output files..."
    base="${OUTPUT_FILE%.*}"
    ext="${OUTPUT_FILE##*.}"

    > "$OUTPUT_FILE"  # Clear output file
    for i in $(seq 0 $((NUM_CHUNKS-1))); do
        chunk_file="${base}_chunk${i}.${ext}"
        if [ -f "$chunk_file" ]; then
            cat "$chunk_file" >> "$OUTPUT_FILE"
            echo "Merged $chunk_file"
        fi
    done

    echo "Final output: $OUTPUT_FILE"
fi
