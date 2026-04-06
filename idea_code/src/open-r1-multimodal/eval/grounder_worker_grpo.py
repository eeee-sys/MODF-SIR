# -*- coding: utf-8 -*-


import argparse
import gc
import json
import os
import sys
import glob
import re
import traceback

import torch


def log(msg):
    """诊断信息输出到 stderr，避免污染 stdout 上的 JSON 协议。"""
    print(msg, file=sys.stderr, flush=True)


def is_oom_error(exc):
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cuda out of memory" in text
        or exc.__class__.__name__ == "OutOfMemoryError"
    )


def resolve_grpo_adapter_path(base_path):
    """自动检测 GRPO adapter 路径（与 test_grpo_grounder.py 逻辑一致）。"""
    if os.path.exists(os.path.join(base_path, "adapter_config.json")):
        log(f"[GRPO Worker] Found adapter in {base_path}")
        return base_path

    log(f"[GRPO Worker] Warning: 'adapter_config.json' not found in {base_path}")
    log("[GRPO Worker] Searching for checkpoints...")

    checkpoints = glob.glob(os.path.join(base_path, "checkpoint-*"))
    if not checkpoints:
        raise FileNotFoundError(
            f"Error: Could not find any checkpoint or adapter in {base_path}."
        )

    def get_step(path):
        match = re.search(r"checkpoint-(\d+)", path)
        return int(match.group(1)) if match else -1

    latest_checkpoint = max(checkpoints, key=get_step)
    log(f"[GRPO Worker] Found latest checkpoint: {latest_checkpoint}")

    if not os.path.exists(os.path.join(latest_checkpoint, "adapter_config.json")):
        sub_path = os.path.join(latest_checkpoint, "grpo_grounder")
        if os.path.exists(os.path.join(sub_path, "adapter_config.json")):
            log(f"[GRPO Worker] Relocating to adapter subdirectory: {sub_path}")
            return sub_path
        return latest_checkpoint
    return latest_checkpoint


def get_raw_model(m):
    """Unwrap PeftModel -> base_model -> model 以获取底层模型。"""
    while hasattr(m, 'module'):
        m = m.module
    if hasattr(m, 'base_model'):
        inner = m.base_model
        if hasattr(inner, 'model'):
            return inner.model
    return m


def main():
    parser = argparse.ArgumentParser(description="GRPO Grounder Worker (stdin/stdout JSON)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="基础 Grounder 模型路径 (如 model_zoo/VideoMind-7B)")
    parser.add_argument("--grpo_adapter_path", type=str, required=True,
                        help="GRPO adapter 权重目录 (如 grpo_grounder_7b)")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    log(f"[GRPO Worker] Starting on device={args.device}, model={args.model_path}")
    log(f"[GRPO Worker] GRPO adapter path={args.grpo_adapter_path}")

    # 将 stdout 重定向到 stderr，防止模型加载期间的 print() 污染 JSON 协议
    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    # ---- 导入 (仅在 VideoMind 环境中可用) ----
    from peft import PeftModel
    from videomind.model.builder import build_model
    from videomind.constants import REG_TOKEN, SEG_S_TOKEN, SEG_E_TOKEN, GROUNDER_PROMPT
    from qwen_vl_utils import process_vision_info as qwen_process

    # ---- Step 1: 加载并合并原始 grounder ----
    log("[GRPO Worker] Step 1: Loading base grounder model...")
    model, processor = build_model(
        args.model_path,
        merge_adapter=True,
        dtype=torch.float16
    )

    # ---- Step 2: 注册特殊 token ----
    log("[GRPO Worker] Step 2: Registering special tokens...")
    new_tokens = processor.tokenizer.add_special_tokens(
        dict(additional_special_tokens=[REG_TOKEN, SEG_S_TOKEN, SEG_E_TOKEN])
    )
    log(f"[GRPO Worker] Added {new_tokens} new special token(s)")

    model.config.reg_token_id = processor.tokenizer.convert_tokens_to_ids(REG_TOKEN)
    model.config.seg_s_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_S_TOKEN)
    model.config.seg_e_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_E_TOKEN)
    log(f"[GRPO Worker] reg_token_id = {model.config.reg_token_id}")

    if new_tokens > 0 and len(processor.tokenizer) > model.config.vocab_size:
        log(f"[GRPO Worker] Resizing embeddings: {model.config.vocab_size} -> {len(processor.tokenizer)}")
        model.resize_token_embeddings(len(processor.tokenizer))

    # ---- Step 3: 加载 GRPO adapter ----
    log("[GRPO Worker] Step 3: Loading GRPO adapter...")
    grpo_adapter = resolve_grpo_adapter_path(args.grpo_adapter_path)
    model = PeftModel.from_pretrained(
        model,
        grpo_adapter,
        adapter_name='grpo_grounder',
        is_trainable=False
    )
    model.eval()

    device = next(model.parameters()).device

    # 加载 head 权重
    from safetensors.torch import load_model
    head_weights_path = os.path.join(grpo_adapter, "pytorch_model.safetensors")
    if not os.path.exists(head_weights_path):
        parent_dir = os.path.dirname(grpo_adapter)
        head_weights_path = os.path.join(parent_dir, "pytorch_model.safetensors")

    if os.path.exists(head_weights_path):
        log(f"[GRPO Worker] Loading head weights from: {head_weights_path}")
        load_model(model, head_weights_path, strict=False, device=str(device))
        log("[GRPO Worker] Head weights loaded.")
    else:
        log(f"[GRPO Worker] Warning: pytorch_model.safetensors not found near {grpo_adapter}!")

    raw_model = get_raw_model(model)
    log(f"[GRPO Worker] Underlying model type: {type(raw_model).__name__}")
    log(f"[GRPO Worker] Model loaded on {device}")

    # 恢复 stdout 用于 JSON 通信
    sys.stdout = original_stdout

    # 发送 ready 信号
    ready_msg = json.dumps({"status": "ready"})
    sys.stdout.write(ready_msg + "\n")
    sys.stdout.flush()
    log("[GRPO Worker] Ready, waiting for requests on stdin...")

    # ---- 服务循环 ----
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"[GRPO Worker] Invalid JSON: {e}")
            resp = json.dumps(
                {
                    "pred_spans": [],
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": "",
                    "is_oom": False,
                }
            )
            sys.stdout.write(resp + "\n")
            sys.stdout.flush()
            continue

        # 关闭指令
        if req.get("command") == "shutdown":
            log("[GRPO Worker] Received shutdown command. Exiting.")
            resp = json.dumps({"status": "shutdown"})
            sys.stdout.write(resp + "\n")
            sys.stdout.flush()
            break

        video_path = req["video_path"]
        query_text = req["query"]
        duration = req["duration"]
        log(f"[GRPO Worker] Grounding: video={video_path}, query={query_text}, duration={duration:.1f}s")

        try:
            # 构建消息
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": 64 * 28 * 28,
                        "max_pixels": 96 * 28 * 28,
                        "fps": 1.0,
                        "max_frames": 150
                    },
                    {"type": "text", "text": GROUNDER_PROMPT.format(query_text)}
                ]
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = qwen_process(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True
            ).to(device)

            # 初始化 reg / sal
            raw_model.reg = []
            raw_model.sal = []

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

            # 提取时间区间
            if len(raw_model.reg) > 0:
                bnd = raw_model.reg[0]  # [K, 3]: (start_ratio, end_ratio, score)
                pred = bnd[:, :2].cpu().float() * duration
                pred = pred.clamp(min=0, max=duration).tolist()
                for i in range(len(pred)):
                    if pred[i][0] > pred[i][1]:
                        pred[i] = [pred[i][1], pred[i][0]]
                success = True
                log(f"[GRPO Worker] Predictions: {pred}")
            else:
                log("[GRPO Worker] WARNING: No grounding result, using full video")
                pred = [[0, duration]]
                success = False

            resp = json.dumps({"pred_spans": pred, "success": success})

        except Exception as e:
            log(f"[GRPO Worker] ERROR: {e}")
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            oom = is_oom_error(e)
            if oom and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            resp = json.dumps(
                {
                    "pred_spans": [[0, duration]],
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": tb,
                    "is_oom": oom,
                }
            )

        sys.stdout.write(resp + "\n")
        sys.stdout.flush()

    log("[GRPO Worker] Process exiting.")


if __name__ == "__main__":
    main()
