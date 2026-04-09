# -*- coding: utf-8 -*-
import os
import gc
import json
import re
import sys
import argparse
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # src/open-r1-multimodal
sys.path.insert(0, PROJECT_DIR)

from qwen_omni_utils import process_mm_info

def check_if_video_has_audio(video_path):
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
    except:
        return False

# ── System Prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional video content analysis Planner. Your task is to read the user's query about a video and decide whether it is necessary to use an external video grounding tool (GRPO_Grounder) to help answer it.
GRPO_Grounder's function is to retrieve segments from long videos to help locate the golden segments most relevant to the query.

[Decision Guidelines]
- If the question is very simple, or the answer can be easily deduced from the video's surface, background, or overall atmosphere, you should NOT call the Grounder. The subsequent modules can answer it directly.
- If the question involves extremely specific details in the video, subtle movements, dialogue at a specific point in time, or plots in a long video that are hard to capture at once, you MUST call the Grounder to extract key clues. You need to rewrite the original query into a search-friendly, concrete "Grounding query" and send it to the Grounder.

[Output Format Requirements]
You must and ONLY return a valid JSON array. It is strictly forbidden to include any prefixes, explanations, or other Markdown text (do NOT output symbols like ```json).
- Case 1: If Grounder is needed, output: [{"type": "Grounder", "value": "your rewritten retrieval-specific query"}, {"type": "Answer"}]
- Case 2: If Grounder is NOT needed, output: [{"type": "Answer"}]"""

USER_PROMPT_TEMPLATE = """The user's query is: "{problem}" """

def extract_json_from_response(text):
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return json.dumps(result, ensure_ascii=False)
    except json.JSONDecodeError:
        pass

    md_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if md_match:
        try:
            result = json.loads(md_match.group(1))
            if isinstance(result, list):
                return json.dumps(result, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    arr_match = re.search(r'(\[\s*\{.*?\}\s*\])', text, re.DOTALL)
    if arr_match:
        try:
            result = json.loads(arr_match.group(1))
            if isinstance(result, list):
                return json.dumps(result, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    # 4) 全部失败，返回原始文本（后续可人工检查）
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate planner_path using Qwen3-Omni")
    parser.add_argument("--model_path", type=str, required=True, help="Qwen3-Omni 模型路径")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSON 文件路径")
    parser.add_argument("--video_folder", type=str, required=True, help="视频文件的根目录")
    parser.add_argument("--chunk_id", type=int, default=0, help="当前分块 ID")
    parser.add_argument("--num_chunks", type=int, default=1, help="总分块数")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7", help="模型分片使用的 GPU 编号列表，用逗号分隔，如 '0,1,2,3'")
    parser.add_argument("--max_frames", type=int, default=150, help="视频最大采帧数 (设为-1则不控制)")
    parser.add_argument("--fps", type=float, default=1.0, help="视频采样帧率 (设为-1则不控制)")
    parser.add_argument("--min_pixels", type=int, default=28224, help="视频最小像素 (设为-1则不控制)")
    parser.add_argument("--max_pixels", type=int, default=50176, help="视频最大像素 (设为-1则不控制)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="模型最大生成 token 数")
    parser.add_argument("--raw_video", action="store_true", help="是否直接输入原视频（不进行帧采样和像素限制）")
    args = parser.parse_args()

    gpu_list = [int(g.strip()) for g in args.gpu_ids.split(",")]
    print(f"[Chunk {args.chunk_id}] 使用 GPU: {gpu_list}", flush=True)

    # ── 1. 加载数据 ─────────────────────────────────────────────
    print(f"[Chunk {args.chunk_id}] 正在加载数据: {args.input_file}", flush=True)
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    chunk_size = total // args.num_chunks
    start_idx = args.chunk_id * chunk_size
    end_idx = start_idx + chunk_size if args.chunk_id < args.num_chunks - 1 else total
    data_chunk = data[start_idx:end_idx]
    
    # ── 1.5 断点续跑逻辑搭建 ────────────────────────────────────
    base, ext = os.path.splitext(args.output_file)
    chunk_output = f"{base}_chunk_{args.chunk_id}.jsonl"
    processed_keys = set()
    
    if os.path.exists(chunk_output):
        with open(chunk_output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                    # 优先用 path/video，如果都没有用 problem 做去重键
                    key = record.get("path", record.get("video", record.get("problem", "")))
                    processed_keys.add(key)
                except json.JSONDecodeError:
                    pass
        print(f"[Chunk {args.chunk_id}] 检测到断点文件 {chunk_output}，已跳过 {len(processed_keys)} 条记录", flush=True)

    print(f"[Chunk {args.chunk_id}] 总样本: {total}, 本 chunk 范围: [{start_idx}, {end_idx}), 共 {len(data_chunk)} 条", flush=True)

    # ── 2. 加载模型 & Processor（多 GPU 模型并行）──────────────────
    print(f"[Chunk {args.chunk_id}] 正在加载 Qwen3-Omni 模型: {args.model_path}", flush=True)
    print(f"[Chunk {args.chunk_id}] 模型将分布在 GPU {gpu_list} 上", flush=True)

    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    # 构建 max_memory：仅允许模型使用指定的 GPU，其余设为 0
    max_memory = {g: "80GiB" for g in gpu_list}
    for i in range(8):
        if i not in gpu_list:
            max_memory[i] = "0GiB"

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        max_memory=max_memory,
    ).eval()

    # 获取模型的首个设备，用于放置输入 tensor
    if hasattr(model, "hf_device_map"):
        first_device = torch.device(list(model.hf_device_map.values())[0])
    else:
        first_device = torch.device(f"cuda:{gpu_list[0]}")
    print(f"[Chunk {args.chunk_id}] 模型首设备: {first_device}", flush=True)

    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
    print(f"[Chunk {args.chunk_id}] 模型加载完成!", flush=True)

    # ── 3. 推理循环与即时保存 ───────────────────────────────────
    failed_output = os.path.join(SCRIPT_DIR, "failed_sample.json")
    failed_count = 0
    with open(chunk_output, "a", encoding="utf-8") as f_out, \
         open(failed_output, "a", encoding="utf-8") as f_fail:
        for idx, sample in enumerate(tqdm(data_chunk, desc=f"Chunk {args.chunk_id}")):
            # 生成去重键
            sample_key = sample.get("path", sample.get("video", sample.get("problem", "")))
            if sample_key in processed_keys:
                continue

            vid_name = sample.get("path", sample.get("video", ""))
            video_path = os.path.abspath(os.path.join(args.video_folder, vid_name))
            problem = sample.get("problem", "")
            print(f"[Chunk {args.chunk_id}] vid_name: {vid_name}", flush=True)
            print(f"[Chunk {args.chunk_id}] video_path: {video_path}", flush=True)
            print(f"[Chunk {args.chunk_id}] problem: {problem}", flush=True)

            if not os.path.exists(video_path):
                print(f"[Chunk {args.chunk_id}] ⚠️ 视频不存在，记录到 failed_sample.json: {video_path}", flush=True)
                f_fail.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_fail.flush()
                failed_count += 1
                continue

            # 检测音频
            use_audio = check_if_video_has_audio(video_path)

            # 构建对话
            vid_dict = {
                "type": "video",
                "video": video_path,
            }
            if not args.raw_video:
                if args.max_frames > 0: vid_dict["max_frames"] = args.max_frames
                if args.fps > 0: vid_dict["fps"] = args.fps
                if args.min_pixels > 0: vid_dict["min_pixels"] = args.min_pixels
                if args.max_pixels > 0: vid_dict["max_pixels"] = args.max_pixels

            user_content = [vid_dict]
            if use_audio:
                user_content.append({"type": "audio", "audio": video_path})

            user_content.append({"type": "text", "text": USER_PROMPT_TEMPLATE.format(problem=problem)})

            conversation = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]
            print(f"[Chunk {args.chunk_id}] conversation: {conversation}", flush=True)

            try:
                # 预处理多模态信息
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
                text = processor.apply_chat_template([conversation], tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=text,
                    audio=audios if audios else None,
                    images=images if images else None,
                    videos=videos if videos else None,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=use_audio,
                ).to(first_device).to(torch.bfloat16)

                # 生成
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        use_audio_in_video=use_audio,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        thinker_return_dict_in_generate=True,
                    )

                # 解码（thinker_return_dict_in_generate=True 时返回结构化对象）
                generated_ids = output_ids.sequences[0][inputs.input_ids.size(1):]
                response = processor.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                planner_json = extract_json_from_response(response)
                sample["planner_path"] = planner_json
                print(f"[Chunk {args.chunk_id}] planner_json: {planner_json}", flush=True)

                # 写入单条结果
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_out.flush()
                os.fsync(f_out.fileno())

            except Exception as e:
                print(f"[Chunk {args.chunk_id}] ❌ 处理失败: {vid_name}, 错误: {e}", flush=True)
                # 将失败样本原封不动写入 failed_sample.json
                f_fail.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_fail.flush()
                failed_count += 1

            # 清理显存
            gc.collect()
            torch.cuda.empty_cache()

    print(f"[Chunk {args.chunk_id}] ✅ 本 chunk 处理完成: {chunk_output}", flush=True)
    if failed_count > 0:
        print(f"[Chunk {args.chunk_id}] ⚠️ 共 {failed_count} 条失败样本已记录到: {failed_output}", flush=True)


# ── 合并各 chunk 结果的辅助函数 ─────────────────────────────────
def merge_chunks():
    """合并多个 chunk 文件为一个最终输出文件。
    用法: python generate_planner.py --merge --output_file <最终文件> --num_chunks <N>
    """
    parser = argparse.ArgumentParser(description="Merge chunk results")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_chunks", type=int, required=True)
    args = parser.parse_args()

    base, ext = os.path.splitext(args.output_file)
    merged = []
    for i in range(args.num_chunks):
        chunk_file = f"{base}_chunk_{i}.jsonl"
        if not os.path.exists(chunk_file):
            print(f"⚠️ 缺少 chunk 文件: {chunk_file}")
            continue
        count = 0
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    merged.append(json.loads(line))
                    count += 1
        print(f"✅ 已加载 {chunk_file} ({count} 条)")

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)
    print(f"🎉 合并完成! 总共 {len(merged)} 条, 已保存到: {args.output_file}")


if __name__ == "__main__":
    # 判断是否为 merge 模式
    if "--merge" in sys.argv:
        merge_chunks()
    else:
        main()
