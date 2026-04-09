# -*- coding: utf-8 -*-
"""
Grounder Worker — standalone process for VideoMind Grounder.

Runs in its OWN Python environment (separate conda env from main process).
Communicates with the main process via stdin/stdout JSON Lines.

Protocol:
  - Reads one JSON object per line from stdin
  - Writes one JSON response per line to stdout
  - Diagnostic/debug messages go to stderr (won't interfere with JSON protocol)

Request JSON:  {"video_path": str, "query": str, "duration": float}
Response JSON: {"pred_spans": [[start, end], ...], "success": bool}
Send:          {"command": "shutdown"} to terminate the worker.

Usage (standalone):
    /path/to/videomind_env/bin/python grounder_worker.py \
        --model_path model_zoo/VideoMind-7B \
        --device cuda:1
"""

import argparse
import json
import sys
import traceback

import torch


def log(msg):
    """Print diagnostic messages to stderr so they don't mix with JSON on stdout."""
    print(msg, file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Grounder Worker (stdin/stdout JSON)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to VideoMind model")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--base_model_prefix", type=str, default="", help="Prefix to fix relative base_model_path")
    args = parser.parse_args()

    log(f"[Grounder Worker] Starting on device={args.device}, model={args.model_path}")

    # Redirect stdout → stderr during imports and model loading,
    # because build_model() and other VideoMind code use print() which
    # would pollute our JSON protocol on stdout.
    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    # ---- imports (VideoMind env only) ----
    from videomind.constants import GROUNDER_PROMPT
    from videomind.dataset.utils import process_vision_info
    from videomind.model.builder import build_model
    from videomind.utils.parser import parse_query

    # ---- load model ----
    log("[Grounder Worker] Loading VideoMind model...")
    from transformers import AutoConfig
    import os
    config = AutoConfig.from_pretrained(args.model_path)
    if hasattr(config, "base_model_path") and not os.path.isabs(config.base_model_path):
        if args.base_model_prefix:
            config.base_model_path = os.path.join(args.base_model_prefix, config.base_model_path)
            log(f"[Grounder Worker] Overriding base_model_path to: {config.base_model_path}")

    model, processor = build_model(args.model_path, config=config, device=args.device)
    model_device = next(model.parameters()).device
    log(f"[Grounder Worker] Model loaded on {model_device}")

    # Restore stdout for JSON communication
    sys.stdout = original_stdout

    # Signal readiness to the main process
    ready_msg = json.dumps({"status": "ready"})
    sys.stdout.write(ready_msg + "\n")
    sys.stdout.flush()
    log("[Grounder Worker] Ready, waiting for requests on stdin...")

    # ---- serve requests ----
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"[Grounder Worker] Invalid JSON: {e}")
            resp = json.dumps({"pred_spans": [], "success": False, "error": str(e)})
            sys.stdout.write(resp + "\n")
            sys.stdout.flush()
            continue

        # Check for shutdown command
        if req.get("command") == "shutdown":
            log("[Grounder Worker] Received shutdown command. Exiting.")
            resp = json.dumps({"status": "shutdown"})
            sys.stdout.write(resp + "\n")
            sys.stdout.flush()
            break

        video_path = req["video_path"]
        query_text = req["query"]
        duration = req["duration"]
        log(f"[Grounder Worker] Grounding: video={video_path}, query={query_text}, duration={duration:.1f}s")

        try:
            query_parsed = parse_query(query_text)

            messages = [{
                'role': 'user',
                'content': [{
                    'type': 'video',
                    'video': video_path,
                    'min_pixels': 36 * 28 * 28,
                    'max_pixels': 64 * 28 * 28,
                    'max_frames': 150,
                    'fps': 1.0
                }, {
                    'type': 'text',
                    'text': GROUNDER_PROMPT.format(query_parsed)
                }]
            }]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
            data = data.to(model_device)

            # set grounder adapter
            if hasattr(model, 'base_model'):
                model.base_model.disable_adapter_layers()
                model.base_model.enable_adapter_layers()
                model.set_adapter('grounder')

            output_ids = model.generate(
                **data,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=None,
                max_new_tokens=256
            )

            output_ids = output_ids[0, data.input_ids.size(1):]
            if output_ids[-1] == processor.tokenizer.eos_token_id:
                output_ids = output_ids[:-1]
            response_text = processor.decode(output_ids, clean_up_tokenization_spaces=False)
            log(f"[Grounder Worker] Model response: {response_text}")

            success = len(model.reg) > 0

            if success:
                blob = model.reg[0].cpu().float()
                pred = blob[:, :2] * duration
                pred = pred.clamp(min=0, max=duration)
                pred = pred.tolist()
                for i in range(len(pred)):
                    if pred[i][0] > pred[i][1]:
                        pred[i] = [pred[i][1], pred[i][0]]
                log(f"[Grounder Worker] Predictions: {pred}")
            else:
                log("[Grounder Worker] WARNING: Failed to parse response, using full video")
                pred = [[0, duration]]

            resp = json.dumps({"pred_spans": pred, "success": success})

        except Exception as e:
            log(f"[Grounder Worker] ERROR: {e}")
            traceback.print_exc(file=sys.stderr)
            resp = json.dumps({"pred_spans": [[0, duration]], "success": False, "error": str(e)})

        sys.stdout.write(resp + "\n")
        sys.stdout.flush()

    log("[Grounder Worker] Process exiting.")


if __name__ == "__main__":
    main()
