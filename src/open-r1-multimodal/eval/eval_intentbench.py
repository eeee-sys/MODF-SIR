# -*- coding: utf-8 -*-
"""
eval_idea3_reviser_7b.py — Multi-agent pipeline with single Qwen2.5-Omni-7B for Collector/Planner/Reviser
Pipeline: Collector -> Planner -> GRPO_Grounder -> HumanOmniV2 (REINFORCE) -> Reviser
Key optimization: Reuse single Qwen2.5-Omni instance for three roles with LoRA switching
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback

import torch

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from peft import LoraConfig, get_peft_model, PeftModel

# Adjust as per your project structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "feature_extraction"))

from qwen_omni_utils import process_mm_info

# ============================================================
# Prompt Templates
# ============================================================
COLLECTOR_SYSTEM_PROMPT = """You are a professional multimodal information Collector. Your task is to carefully watch the provided video and listen to its audio, then extract all information that is relevant to the user's query.

You should focus on:
- Visual details: facial expressions, body language, gestures, movements, scene changes, objects, and interactions between people.
- Audio details: tone of voice, emotions conveyed through speech, background sounds, music, dialogue content, and any auditory cues.
- Temporal information: the order of events, timing of key moments, and any changes over time.

Output your findings as a clear, concise, and organized text summary. Do NOT answer the query directly — only collect and report the relevant information you observe from the video and audio."""

COLLECTOR_USER_TEMPLATE = """Please collect all relevant information from the video and audio that relates to the following query:

Query: "{problem}"

Report your observations in a structured and detailed manner."""

PLANNER_SYSTEM_PROMPT = """You are a professional video content analysis Planner. Your task is to read the user's query about a video and decide whether it is necessary to use an external video grounding tool (GRPO_Grounder) to help answer it.
GRPO_Grounder's function is to retrieve segments from long videos to help locate the golden segments most relevant to the query.

[Decision Guidelines]
- If the question is very simple, or the answer can be easily deduced from the video's surface, background, or overall atmosphere, you should NOT call the Grounder. The subsequent modules can answer it directly.
- If the question involves extremely specific details in the video, subtle movements, dialogue at a specific point in time, or plots in a long video that are hard to capture at once, you MUST call the Grounder to extract key clues. You need to rewrite the original query into a search-friendly, concrete "Grounding query" and send it to the Grounder.

[Output Format Requirements]
You must and ONLY return a valid JSON array. It is strictly forbidden to include any prefixes, explanations, or other Markdown text (do NOT output symbols like ```json).
- Case 1: If Grounder is needed, output: [{"type": "Grounder", "value": "your rewritten retrieval-specific query"}, {"type": "Answer"}]
- Case 2: If Grounder is NOT needed, output: [{"type": "Answer"}]"""

PLANNER_USER_TEMPLATE = """[Collector Information]
Based on the visual and auditory cues from the provided video, the relevant information regarding the query is as follows:
{collector_text}

[User Query]
The user's query is: "{problem}"

Please analyze the complexity of this query in relation to the video content, and output your planning decision as a JSON array."""

HUMANOMNI_SYSTEM_PROMPT = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question.

In reasoning, It is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.
"""

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "emer_ov_mc": " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",
    "judge": " Please answer Yes or No within the <answer> </answer> tags.",
}

REVISER_SYSTEM_PROMPT = """You are a highly capable and rigorous AI agent evaluator. Your task is to review the candidate's answer based on the provided video, audio, and user query.

First, carefully analyze the accuracy, depth, relevance, and logical reasoning of the candidate's answer within the <think> </think> tags.
Then, provide a definitive integer score between 1 and 10 within the <score> </score> tags, where 1 means completely irrelevant/incorrect, and 10 means perfect, highly insightful, and rigorously aligned with the multimodal evidence.

Scoring criteria:
1-2: The answer is completely irrelevant to the query, or based on hallucination (no supporting evidence), or severely contradicts the multimodal evidence.
3-4: The answer is partially relevant but contains multiple factual errors, or the logical reasoning is incoherent, or key evidence is ignored.
5-6: The answer is basically correct but lacks depth, or the reasoning is not rigorous enough, or minor details are omitted.
7-8: The answer is accurate and relevant, reasoning is reasonable, depth is moderate, but may lack some unique insights or fails to fully utilize all evidence.
9-10: The answer is highly accurate, insightful, logically rigorous, fully utilizes and strictly aligns with multimodal evidence, demonstrating excellent analytical ability.

Remember: DO NOT hallucinate. Only rely on the visual and auditory evidence present in the context. Be strict but fair.
CRITICAL: You MUST output your final score strictly enclosed within <score> and </score> tags. Do NOT use <answer> tags for the score. Example: <score>8</score>"""

# ============================================================
# Utilities
# ============================================================
def get_video_duration(video_path):
    import av
    with av.open(video_path) as container:
        duration = container.duration / 1_000_000.0
    return duration

def check_if_video_has_audio(video_path):
    """Aligned with eval_humanomniv2.py: checks audio stream AND minimum length via librosa."""
    try:
        import av
        import librosa
        import warnings
        container = av.open(video_path)
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(video_path, sr=16000)
            if len(y) < 8000:
                return False
        return True
    except:
        return False

def trim_video_ffmpeg(video_path, start, end, output_path):
    # 移除了 "-c:v copy" 和 "-c:a copy" 以强制重新编码。
    # 这样虽然稍微慢一点点，但能保证短片段视频帧的有效性。
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ss", str(start), "-to", str(end),
        output_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # 如果仍然失败，打印错误信息以便调试，并优化备用命令
    if res.returncode != 0:
        print(f"[FFmpeg Warning] Primary trim failed: {res.stderr}")
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(start), "-i", video_path, "-to", str(end - start), output_path
        ], capture_output=True, check=True)
        
    return output_path

def parse_planner_output(text):
    use_grounder, grounder_query = False, ""
    json_match = re.search(r'(\[\s*\{.*?\}\s*\])', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            for item in parsed:
                if item.get("type", "").lower() == "grounder":
                    use_grounder = True
                    grounder_query = item.get("value", "")
        except json.JSONDecodeError:
            pass
    return use_grounder, grounder_query

def extract_score(text):
    # 先尝试匹配 <score>
    match = re.search(r"<score>\s*(\d+)\s*</score>", text)
    if not match:
        # 如果没找到，退而求其次，看看是不是错误地写进了 <answer> 里
        match = re.search(r"<answer>\s*(\d+)\s*</answer>", text)
    
    if match:
        return min(max(int(match.group(1)), 1), 10)
    
    print(f"[Warning] Score extraction failed for text: {text}") # 加上这句方便以后排错
    return 5

def is_oom_error(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()

def format_exception_info(exc):
    if is_oom_error(exc):
        error_type = "OOM"
    else:
        error_type = type(exc).__name__
    error_message = str(exc).strip() or repr(exc)
    tb_text = traceback.format_exc().strip()
    if tb_text:
        tb_text = tb_text[-4000:]
    return error_type, error_message, tb_text

def append_skipped_sample_log(dataset_id, query, video_path, error_type, error_message, traceback_text, chunk_idx):
    log_dir = os.path.join("eval_results")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "skipped_samples_intentbench.log")
    log_entry = {
        "id": dataset_id,
        "query": query,
        "video": video_path,
        "error_type": error_type,
        "error_message": error_message,
        "traceback": traceback_text,
        "chunk_idx": chunk_idx,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# ============================================================
# Main Agent Stages
# ============================================================
def stage1_collector(model, processor, video_path, query, device):
    """Collector uses base_model.thinker with LoRA disabled"""
    has_audio = check_if_video_has_audio(video_path)
    content = [{"type": "video", "video": video_path, 
                "min_pixels": 36 * 28 * 28,
                "max_pixels": 96 * 28 * 28,
                "max_frames": 150, 
                "fps": 1.0}]
    if has_audio:
        content.append({"type": "audio", "audio": video_path})
    text_prompt = COLLECTOR_USER_TEMPLATE.format(problem=query)
    content.append({"type": "text", "text": text_prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": COLLECTOR_SYSTEM_PROMPT}]},
        {"role": "user", "content": content}
    ]
    print(f"collector input: {messages}")

    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True).to(device)

    for k in inputs:
        if isinstance(inputs[k], torch.Tensor) and inputs[k].is_floating_point():
            inputs[k] = inputs[k].to(torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    resp = processor.decode(output_ids[0][inputs.input_ids.size(1):], skip_special_tokens=True)
    return resp

def stage2_planner(model, processor, video_path, query, collector_text, device):
    """Planner uses base_model.thinker with LoRA enabled"""
    has_audio = check_if_video_has_audio(video_path)
    content = [{"type": "video", 
                "video": video_path, 
                "min_pixels": 36 * 28 * 28, 
                "max_pixels": 64 * 28 * 28,
                "max_frames": 150, 
                "fps": 1.0}]
    if has_audio:
        content.append({"type": "audio", "audio": video_path})
    text_prompt = PLANNER_USER_TEMPLATE.format(collector_text=collector_text, problem=query)
    content.append({"type": "text", "text": text_prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": PLANNER_SYSTEM_PROMPT}]},
        {"role": "user", "content": content}
    ]
    print(f"planner input: {messages}")

    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True).to(device)

    for k in inputs:
        if isinstance(inputs[k], torch.Tensor) and inputs[k].is_floating_point():
            inputs[k] = inputs[k].to(torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    resp = processor.decode(output_ids[0][inputs.input_ids.size(1):], skip_special_tokens=True)
    return parse_planner_output(resp), resp

def stage3_grounder(grounder_proc, video_path, query, duration):
    req = json.dumps({"video_path": video_path, "query": query, "duration": duration})
    try:
        grounder_proc.stdin.write(req + "\n")
        grounder_proc.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        raise RuntimeError(f"Grounder worker unavailable while sending request: {exc}") from exc

    resp_line = grounder_proc.stdout.readline().strip()
    if not resp_line:
        raise RuntimeError("Grounder worker returned empty response.")

    try:
        resp = json.loads(resp_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Grounder worker returned invalid JSON: {resp_line}") from exc

    if "error" in resp and resp["error"]:
        raise RuntimeError(f"Grounder worker error: {resp['error']}")

    return resp.get("pred_spans", [[0, duration]]), resp.get("success", False)

def build_humanomni_query(sample):
    """Build the full question text with options and TYPE_TEMPLATE, aligned with eval_humanomniv2.py."""
    problem_type = sample.get("problem_type", "free-form")
    if problem_type in ("multiple choice", "emer_ov_mc"):
        question = sample["problem"] + " Options:\n"
        for op in sample.get("options", []):
            question += op + "\n"
    else:
        question = sample["problem"]
    question += TYPE_TEMPLATE.get(problem_type, TYPE_TEMPLATE["free-form"])
    print(f"[HumanOmni question] {question}")
    return question

def get_humanomni_inputs(processor, video_path, query_text, sample, device):
    """Build model inputs for HumanOmniV2, aligned with eval_humanomniv2.py message format."""
    has_audio = check_if_video_has_audio(video_path)
    data_type = sample.get("data_type", "video")
    content = [{"type": data_type, 
                data_type: video_path,
                "min_pixels": 128 * 28 * 28,
                "max_pixels": 256 * 28 * 28,
                "max_frames": 64,
                "fps": 2.0}]
    if has_audio:
        content.append({"type": "audio", "audio": video_path})
        text_prompt = f"Here is a {data_type}, with the audio from the video.\n" + query_text
    else:
        text_prompt = query_text
    content.append({"type": "text", "text": text_prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": HUMANOMNI_SYSTEM_PROMPT}]},
        {"role": "user", "content": content}
    ]
    print(f"humanomni input: {messages}")
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True).to(device)

    for k in inputs:
        if isinstance(inputs[k], torch.Tensor) and inputs[k].is_floating_point():
            inputs[k] = inputs[k].to(torch.bfloat16)
    return inputs

def revise_answer(model, processor, video_path, query, answer, device):
    """Reviser uses base_model.thinker with LoRA disabled"""
    has_audio = check_if_video_has_audio(video_path)
    content = [{"type": "video", 
                "video": video_path,
                "min_pixels": 64 * 28 * 28,
                "max_pixels": 128 * 28 * 28,
                "max_frames": 100,
                "fps": 1.0}]
    if has_audio:
        content.append({"type": "audio", "audio": video_path})
    user_prompt = f"Query: {query}\nCandidate Answer: {answer}"
    content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": REVISER_SYSTEM_PROMPT}]},
        {"role": "user", "content": content}
    ]

    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True).to(device)

    for k in inputs:
        if isinstance(inputs[k], torch.Tensor) and inputs[k].is_floating_point():
            inputs[k] = inputs[k].to(torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    resp = processor.decode(output_ids[0][inputs.input_ids.size(1):], skip_special_tokens=True)
    return extract_score(resp), resp

# ============================================================
# Main Process
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--planner_lora_path", required=True)
    parser.add_argument("--humanomni_path", required=True)
    parser.add_argument("--grounder_path", required=True)
    parser.add_argument("--grpo_adapter_path", required=True)
    parser.add_argument("--grounder_python", required=True)
    parser.add_argument("--grounder_cwd", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--lora_save_dir", required=True)
    parser.add_argument("--video_root", default="", help="Root directory for video files")
    parser.add_argument("--id_key", default="qid,video", help="Comma separated keys for identifying samples")
    parser.add_argument("--main_gpu", default="cuda:0")
    parser.add_argument("--grounder_gpu", default="cuda:1")
    parser.add_argument("--humanomni_gpu", default="cuda:0")
    parser.add_argument("--t_max", type=int, default=3)
    parser.add_argument("--tau", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--b0", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_chunks", type=int, default=1, help="Total number of parallel processes")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Current process index (0-based)")
    args = parser.parse_args()

    # Modify output file and LoRA dir for multi-process
    if args.num_chunks > 1:
        base, ext = os.path.splitext(args.output_file)
        args.output_file = f"{base}_chunk{args.chunk_idx}{ext}"
        args.lora_save_dir = f"{args.lora_save_dir}_chunk{args.chunk_idx}"
        print(f"[Process {args.chunk_idx}/{args.num_chunks}] Output: {args.output_file}")

    def get_sample_id(item, keys_str):
        for k in keys_str.split(','):
            if k in item and item[k] is not None:
                return str(item[k])
        return "unknown"

    # Load dataset and apply chunking
    with open(args.input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if args.num_chunks > 1:
        chunk_size = len(dataset) // args.num_chunks
        start_idx = args.chunk_idx * chunk_size
        end_idx = start_idx + chunk_size if args.chunk_idx < args.num_chunks - 1 else len(dataset)
        dataset = dataset[start_idx:end_idx]
        print(f"[Process {args.chunk_idx}/{args.num_chunks}] Processing samples {start_idx}-{end_idx}")

    # Resume logic (after chunking)
    # Only successful samples written to output_file are treated as processed.
    # Skipped samples are intentionally excluded so they will be retried next run.
    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        rec_id = get_sample_id(record, args.id_key)
                        if rec_id != "unknown":
                            processed_ids.add(rec_id)
                    except: pass

    print(f"已经处理了{len(processed_ids)}个样本")
    samples_to_process = [item for item in dataset if get_sample_id(item, args.id_key) not in processed_ids]
    print(f"还需要处理{len(samples_to_process)}个样本")
    if not samples_to_process:
        print("[INFO] All samples already processed!")
        return

    # ---- 2. Initialize Models ----
    print(f"\n[INIT] Loading Base Model ({args.base_model_path}) on {args.main_gpu}")
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(args.main_gpu)
    base_processor = Qwen2_5OmniProcessor.from_pretrained(args.base_model_path)

    # Load Planner LoRA onto thinker submodule
    print(f"[INIT] Loading Planner LoRA onto base_model.thinker")
    # base_model.thinker = PeftModel.from_pretrained(
    #     base_model.thinker,
    #     args.planner_lora_path,
    #     adapter_name="planner"
    # )
    base_model.thinker.load_adapter(args.planner_lora_path, adapter_name="planner")
    base_model.eval()

    print(f"[INIT] Loading HumanOmniV2 ({args.humanomni_path}) on {args.humanomni_gpu}")
    humanomni_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.humanomni_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(args.humanomni_gpu)
    humanomni_processor = Qwen2_5OmniProcessor.from_pretrained(args.humanomni_path)

    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    humanomni_model = get_peft_model(humanomni_model, lora_config, adapter_name="initial_dummy")

    humanomni_model.enable_input_require_grads()

    humanomni_model.gradient_checkpointing_enable()
    print(f"[INIT] Starting Grounder process on {args.grounder_gpu}...")
    grounder_script = os.path.join(SCRIPT_DIR, "grounder_worker_grpo.py")
    grounder_env = os.environ.copy()
    grounder_env["CUDA_VISIBLE_DEVICES"] = args.grounder_gpu.replace("cuda:", "")
    grounder_proc = subprocess.Popen([
        args.grounder_python, grounder_script,
        "--model_path", args.grounder_path,
        "--grpo_adapter_path", args.grpo_adapter_path,
        "--device", "cuda:0"
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1, 
                            #cwd=args.grounder_cwd, 
                            env=grounder_env)

    ready_line = grounder_proc.stdout.readline().strip()
    if not ready_line or json.loads(ready_line).get("status") != "ready":
        print("[ERROR] Grounder worker failed to start.")
        sys.exit(1)

    print("[INIT] All models ready!")
    os.makedirs(args.lora_save_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="idea3_reviser7b_")

    # ---- 3. Loop through dataset ----
    for sample in samples_to_process:
        dataset_id = get_sample_id(sample, args.id_key)
        optimizer = None
        query = sample.get("problem")
        video_path = sample.get("video") or sample.get("path")
        if args.video_root and video_path:
            video_path = os.path.join(args.video_root, video_path)
        trim_path = None
        adapter_name = None
        adapter_deleted = False

        try:
            print(f"\n======== Processing [{dataset_id}] ========")
            duration = get_video_duration(video_path)
            print(f"[Video path] {video_path}")
            print(f"[Query] {query}")
            print(f"[Duration] {duration}")

            # ====== PLANNER STAGE ======
            # a) Collector Phase (LoRA disabled)
            base_model.thinker.set_adapter("planner")  # Ensure adapter is active before disabling
            base_model.thinker.disable_adapters()
            collector_text = stage1_collector(base_model.thinker, base_processor, video_path, query, args.main_gpu)
            print(f"[Collector output] {collector_text}")

            # b) Planner Phase (LoRA enabled)
            base_model.thinker.enable_adapters()
            (use_grounder, gnd_query), planner_raw = stage2_planner(base_model.thinker, base_processor, video_path, query, collector_text, args.main_gpu)
            print(f"[Planner output] {planner_raw}")
            print(f"[Planner] Use Grounder: {use_grounder} | query: {gnd_query}")

            # ====== GROUNDER STAGE ======
            generation_video = video_path
            grounded_span = None
            if use_grounder:
                pred_spans, success = stage3_grounder(grounder_proc, video_path, gnd_query or query, duration)
                print(f"[Grounder output] {pred_spans}")
                grounded_span = pred_spans[0]
                trim_path = os.path.join(tmp_dir, f"trim_{dataset_id}.mp4")
                trim_video_ffmpeg(video_path, grounded_span[0], grounded_span[1], trim_path)
                generation_video = trim_path
                print(f"[Grounder] Grounded to {grounded_span[0]:.1f}s - {grounded_span[1]:.1f}s")

            # ====== HUMANOMNI & REINFORCE STAGE ======
            humanomni_query = build_humanomni_query(sample)

            adapter_name = f"sample_{dataset_id}".replace(".", "_")
            humanomni_model.add_adapter(adapter_name, lora_config)
            humanomni_model.set_adapter(adapter_name)

            # Ensure adapter parameters require gradients
            for n, p in humanomni_model.named_parameters():
                if adapter_name in n:
                    p.requires_grad = True

            humanomni_model.train()

            trainable_params = [
                p for n, p in humanomni_model.named_parameters()
                if p.requires_grad and adapter_name in n
            ]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

            b = args.b0
            best_score = -1
            best_answer = ""
            best_raw_resp = ""
            all_history = []
            early_stop = False

            for t in range(1, args.t_max + 1):
                gc.collect(); torch.cuda.empty_cache()

                # --- Generate y_t ---
                humanomni_model.eval()
                inputs = get_humanomni_inputs(humanomni_processor, generation_video, humanomni_query, sample, args.humanomni_gpu)

                with torch.no_grad():
                    output_ids = humanomni_model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.85)

                generated_sequence = output_ids[0][inputs.input_ids.size(1):]
                y_t_text = humanomni_processor.decode(generated_sequence, skip_special_tokens=True)
                print(f"  [Iter {t}/{args.t_max}] Answer = {y_t_text}")

                # --- Reviser scores r_t (LoRA disabled) ---
                base_model.thinker.disable_adapters()
                score_t, reviser_raw = revise_answer(base_model.thinker, base_processor, video_path, query, y_t_text, args.main_gpu)
                print(f"  [Iter {t}/{args.t_max}] Score = {score_t}/10")

                all_history.append({"iter": t, "answer": y_t_text, "score": score_t, "reviser_raw": reviser_raw})

                if score_t > best_score:
                    best_score = score_t
                    best_answer = y_t_text
                    best_raw_resp = reviser_raw

                if score_t >= args.tau:
                    print(f"  -> Score {score_t} >= Tau({args.tau}), breaking loop early.")
                    early_stop = True
                    break

                if t == args.t_max:
                    print(f"  -> Reached T_max, stopping.")
                    break

                # --- RL Update (REINFORCE) ---
                humanomni_model.train()
                optimizer.zero_grad()

                advantage = float(score_t - b)
                advantage_tensor = torch.tensor([advantage], device=args.humanomni_gpu, dtype=torch.bfloat16)

                concat_ids = torch.cat([inputs.input_ids, generated_sequence.unsqueeze(0)], dim=1)
                attention_mask = torch.ones_like(concat_ids)
                labels = torch.cat([
                    torch.full_like(inputs.input_ids, -100),
                    generated_sequence.unsqueeze(0)
                ], dim=1)

                forward_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "attention_mask", "labels"]}
                forward_kwargs["input_ids"] = concat_ids
                forward_kwargs["attention_mask"] = attention_mask
                forward_kwargs["labels"] = labels

                outputs = humanomni_model(**forward_kwargs)

                nll_loss = outputs.loss
                final_loss = nll_loss * advantage_tensor.detach()

                final_loss.backward()
                optimizer.step()

                b = args.alpha * b + (1.0 - args.alpha) * score_t
                print(f"  -> Updated Baseline b = {b:.2f}, Loss = {final_loss.item():.4f} (NLL: {nll_loss.item():.4f}, Adv: {advantage:.2f})")

            # ====== CLEANUP & PERSIST ======
            sample_lora_dir = os.path.join(args.lora_save_dir, dataset_id)
            humanomni_model.save_pretrained(sample_lora_dir, selected_adapters=[adapter_name])

            humanomni_model.delete_adapter(adapter_name)
            adapter_deleted = True

            save_entry = {
                "id": dataset_id,
                "query": query,
                "video": video_path,
                "use_grounder": use_grounder,
                "grounded_span": grounded_span,
                "collector_output": collector_text,
                "planner_output": planner_raw,
                "best_score": best_score,
                "best_answer": best_answer,
                "history": all_history,
                "early_stop": early_stop,
            }
            with open(args.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(save_entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            error_type, error_message, traceback_text = format_exception_info(exc)
            print(f"[Skip] Sample [{dataset_id}] failed with {error_type}: {error_message}")
            append_skipped_sample_log(
                dataset_id=dataset_id,
                query=query,
                video_path=video_path,
                error_type=error_type,
                error_message=error_message,
                traceback_text=traceback_text,
                chunk_idx=args.chunk_idx,
            )
            print("[Skip] Logged to eval_results/skipped_samples_intentbench.log; sample will be retried in future runs.")
            continue
        finally:
            if optimizer is not None:
                del optimizer
            if adapter_name and not adapter_deleted:
                try:
                    humanomni_model.delete_adapter(adapter_name)
                except Exception as cleanup_exc:
                    print(f"[Cleanup Warning] Failed to delete adapter {adapter_name}: {cleanup_exc}")
            if trim_path and os.path.exists(trim_path):
                try:
                    os.remove(trim_path)
                except OSError as cleanup_exc:
                    print(f"[Cleanup Warning] Failed to remove temp file {trim_path}: {cleanup_exc}")
            gc.collect()
            torch.cuda.empty_cache()

    grounder_proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
    grounder_proc.stdin.flush()
    grounder_proc.wait()

if __name__ == "__main__":
    main()
