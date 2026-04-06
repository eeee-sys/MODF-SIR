# -*- coding: utf-8 -*-
"""
eval_idea3_reviser_7b_humanomni+reviser Daily-omni.py
HumanOmni + Reviser ablation pipeline for Daily-omni.
Pipeline: HumanOmniV2 (REINFORCE) -> Reviser
"""

import argparse
import gc
import hashlib
import json
import os
import re
import sys

import torch

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from peft import LoraConfig, get_peft_model

# Adjust as per your project structure
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "feature_extraction"))

from qwen_omni_utils import process_mm_info


# ============================================================
# Prompt Templates
# ============================================================
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
        audio_streams = [stream for stream in container.streams if stream.type == "audio"]
        if not audio_streams:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(video_path, sr=16000)
            if len(y) < 8000:
                return False
        return True
    except Exception:
        return False


def extract_score(text):
    match = re.search(r"<score>\s*(\d+)\s*</score>", text)
    if not match:
        match = re.search(r"<answer>\s*(\d+)\s*</answer>", text)

    if match:
        return min(max(int(match.group(1)), 1), 10)

    print(f"[Warning] Score extraction failed for text: {text}")
    return 5


# ============================================================
# Main Agent Stages
# ============================================================
def build_humanomni_query(sample):
    """Build the full question text with options and TYPE_TEMPLATE for Daily-omni format."""
    question = sample.get("Question", "")
    choices = sample.get("Choice", [])
    if choices:
        question += " Options:\n"
        for choice in choices:
            question += choice + "\n"
        question += TYPE_TEMPLATE["multiple choice"]
    else:
        question += TYPE_TEMPLATE["free-form"]
    print(f"[HumanOmni question] {question}")
    return question


def get_humanomni_inputs(processor, video_path, query_text, sample, device):
    """Build model inputs for HumanOmniV2 with video + audio + query + prompt."""
    has_audio = check_if_video_has_audio(video_path)
    data_type = sample.get("data_type", "video")
    content = [
        {
            "type": data_type,
            data_type: video_path,
            "min_pixels": 128 * 28 * 28,
            "max_pixels": 256 * 28 * 28,
            "max_frames": 64,
            "fps": 2.0,
        }
    ]
    if has_audio:
        content.append({"type": "audio", "audio": video_path})
        text_prompt = f"Here is a {data_type}, with the audio from the video.\n" + query_text
    else:
        text_prompt = query_text
    content.append({"type": "text", "text": text_prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": HUMANOMNI_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    print(f"humanomni input: {messages}")
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    ).to(device)

    for key in inputs:
        if isinstance(inputs[key], torch.Tensor) and inputs[key].is_floating_point():
            inputs[key] = inputs[key].to(torch.bfloat16)
    return inputs


def revise_answer(model, processor, video_path, query, answer, device):
    """Reviser uses base_model.thinker."""
    has_audio = check_if_video_has_audio(video_path)
    content = [
        {
            "type": "video",
            "video": video_path,
            "min_pixels": 64 * 28 * 28,
            "max_pixels": 128 * 28 * 28,
            "max_frames": 100,
            "fps": 1.0,
        }
    ]
    if has_audio:
        content.append({"type": "audio", "audio": video_path})
    user_prompt = f"Query: {query}\nCandidate Answer: {answer}"
    content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": REVISER_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]

    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    ).to(device)

    for key in inputs:
        if isinstance(inputs[key], torch.Tensor) and inputs[key].is_floating_point():
            inputs[key] = inputs[key].to(torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    response = processor.decode(output_ids[0][inputs.input_ids.size(1) :], skip_special_tokens=True)
    return extract_score(response), response


# ============================================================
# Main Process
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--planner_lora_path", default=None, help="Ignored in this ablation script")
    parser.add_argument("--humanomni_path", required=True)
    parser.add_argument("--grounder_path", default=None, help="Ignored in this ablation script")
    parser.add_argument("--grpo_adapter_path", default=None, help="Ignored in this ablation script")
    parser.add_argument("--grounder_python", default=None, help="Ignored in this ablation script")
    parser.add_argument("--grounder_cwd", default=None, help="Ignored in this ablation script")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--lora_save_dir", required=True)
    parser.add_argument("--video_root", default="", help="Root directory for video files")
    parser.add_argument("--main_gpu", default="cuda:0")
    parser.add_argument("--grounder_gpu", default="cuda:1", help="Ignored in this ablation script")
    parser.add_argument("--humanomni_gpu", default="cuda:0")
    parser.add_argument("--t_max", type=int, default=3)
    parser.add_argument("--tau", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--b0", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_chunks", type=int, default=1, help="Total number of parallel processes")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Current process index (0-based)")
    args = parser.parse_args()

    if args.num_chunks > 1:
        base, ext = os.path.splitext(args.output_file)
        args.output_file = f"{base}_chunk{args.chunk_idx}{ext}"
        args.lora_save_dir = f"{args.lora_save_dir}_chunk{args.chunk_idx}"
        print(f"[Process {args.chunk_idx}/{args.num_chunks}] Output: {args.output_file}")

    print("[INFO] Running ablation pipeline: HumanOmniV2 -> Reviser")
    ignored_args = {
        "planner_lora_path": args.planner_lora_path,
        "grounder_path": args.grounder_path,
        "grpo_adapter_path": args.grpo_adapter_path,
        "grounder_python": args.grounder_python,
        "grounder_cwd": args.grounder_cwd,
        "grounder_gpu": args.grounder_gpu,
    }
    active_ignored_args = [name for name, value in ignored_args.items() if value is not None]
    if active_ignored_args:
        print(f"[INFO] Ignoring ablation-incompatible args: {', '.join(active_ignored_args)}")

    def get_sample_id(item):
        q = item.get("Question", "")
        vid = item.get("video_id", "")
        return f"{q}||{vid}"

    def get_short_id(item):
        full_id = get_sample_id(item)
        return hashlib.md5(full_id.encode()).hexdigest()[:12]

    with open(args.input_file, "r", encoding="utf-8") as file_obj:
        dataset = json.load(file_obj)

    if args.num_chunks > 1:
        chunk_size = len(dataset) // args.num_chunks
        start_idx = args.chunk_idx * chunk_size
        end_idx = start_idx + chunk_size if args.chunk_idx < args.num_chunks - 1 else len(dataset)
        dataset = dataset[start_idx:end_idx]
        print(f"[Process {args.chunk_idx}/{args.num_chunks}] Processing samples {start_idx}-{end_idx}")

    processed_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if line.strip():
                    try:
                        record = json.loads(line)
                        record_id = record.get("id", "")
                        if record_id:
                            processed_ids.add(record_id)
                    except Exception:
                        pass

    print(f"已经处理了{len(processed_ids)}个样本")
    samples_to_process = [item for item in dataset if get_sample_id(item) not in processed_ids]
    print(f"还需要处理{len(samples_to_process)}个样本")
    if not samples_to_process:
        print("[INFO] All samples already processed!")
        return

    print(f"\n[INIT] Loading Reviser Base Model ({args.base_model_path}) on {args.main_gpu}")
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.main_gpu)
    base_processor = Qwen2_5OmniProcessor.from_pretrained(args.base_model_path)
    base_model.eval()

    print(f"[INIT] Loading HumanOmniV2 ({args.humanomni_path}) on {args.humanomni_gpu}")
    humanomni_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.humanomni_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.humanomni_gpu)
    humanomni_processor = Qwen2_5OmniProcessor.from_pretrained(args.humanomni_path)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    humanomni_model = get_peft_model(humanomni_model, lora_config, adapter_name="initial_dummy")
    humanomni_model.enable_input_require_grads()
    humanomni_model.gradient_checkpointing_enable()

    print("[INIT] All models ready!")
    os.makedirs(args.lora_save_dir, exist_ok=True)

    for sample in samples_to_process:
        dataset_id = get_sample_id(sample)
        print(f"\n======== Processing [{dataset_id}] ========")
        vid = sample.get("video_id", "")
        video_path = os.path.join(args.video_root, vid, f"{vid}_video.mp4")
        query = sample.get("Question")
        duration = get_video_duration(video_path)
        print(f"[Video path] {video_path}")
        print(f"[Query] {query}")
        print(f"[Duration] {duration}")

        humanomni_query = build_humanomni_query(sample)

        short_id = get_short_id(sample)
        adapter_name = f"sample_{short_id}"
        humanomni_model.add_adapter(adapter_name, lora_config)
        humanomni_model.set_adapter(adapter_name)

        for name, parameter in humanomni_model.named_parameters():
            if adapter_name in name:
                parameter.requires_grad = True

        humanomni_model.train()

        trainable_params = [
            parameter
            for name, parameter in humanomni_model.named_parameters()
            if parameter.requires_grad and adapter_name in name
        ]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

        b = args.b0
        best_score = -1
        best_answer = ""
        best_raw_resp = ""
        all_history = []
        early_stop = False

        for t in range(1, args.t_max + 1):
            gc.collect()
            torch.cuda.empty_cache()

            humanomni_model.eval()
            inputs = get_humanomni_inputs(
                humanomni_processor,
                video_path,
                humanomni_query,
                sample,
                args.humanomni_gpu,
            )

            with torch.no_grad():
                output_ids = humanomni_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.85,
                )

            generated_sequence = output_ids[0][inputs.input_ids.size(1) :]
            y_t_text = humanomni_processor.decode(generated_sequence, skip_special_tokens=True)
            print(f"  [Iter {t}/{args.t_max}] Answer = {y_t_text}")

            score_t, reviser_raw = revise_answer(
                base_model.thinker,
                base_processor,
                video_path,
                query,
                y_t_text,
                args.main_gpu,
            )
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
                print("  -> Reached T_max, stopping.")
                break

            humanomni_model.train()
            optimizer.zero_grad()

            advantage = float(score_t - b)
            advantage_tensor = torch.tensor([advantage], device=args.humanomni_gpu, dtype=torch.bfloat16)

            concat_ids = torch.cat([inputs.input_ids, generated_sequence.unsqueeze(0)], dim=1)
            attention_mask = torch.ones_like(concat_ids)
            labels = torch.cat(
                [
                    torch.full_like(inputs.input_ids, -100),
                    generated_sequence.unsqueeze(0),
                ],
                dim=1,
            )

            forward_kwargs = {key: value for key, value in inputs.items() if key not in ["input_ids", "attention_mask", "labels"]}
            forward_kwargs["input_ids"] = concat_ids
            forward_kwargs["attention_mask"] = attention_mask
            forward_kwargs["labels"] = labels

            outputs = humanomni_model(**forward_kwargs)

            nll_loss = outputs.loss
            final_loss = nll_loss * advantage_tensor.detach()

            final_loss.backward()
            optimizer.step()

            b = args.alpha * b + (1.0 - args.alpha) * score_t
            print(
                f"  -> Updated Baseline b = {b:.2f}, Loss = {final_loss.item():.4f} "
                f"(NLL: {nll_loss.item():.4f}, Adv: {advantage:.2f})"
            )

        sample_lora_dir = os.path.join(args.lora_save_dir, get_short_id(sample))
        humanomni_model.save_pretrained(sample_lora_dir, selected_adapters=[adapter_name])

        humanomni_model.delete_adapter(adapter_name)

        save_entry = {
            "id": dataset_id,
            "Question": sample.get("Question"),
            "video_id": sample.get("video_id"),
            "query": query,
            "video": video_path,
            "collector_output": None,
            "planner_output": None,
            "use_grounder": False,
            "grounded_span": None,
            "best_score": best_score,
            "best_answer": best_answer,
            "best_reviser_output": best_raw_resp,
            "history": all_history,
            "early_stop": early_stop,
        }
        with open(args.output_file, "a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(save_entry, ensure_ascii=False) + "\n")

        del optimizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
