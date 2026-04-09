# -*- coding: utf-8 -*-
"""
run_collector_preprocess.py
===========================
使用 Qwen/Qwen2.5-Omni-7B 作为 Collector 角色，遍历数据集中的每条样本，
提取视频和音频中与 Query 相关的文本信息，将其保存为 collector_text 字段。

支持通过 --chunk_id / --num_chunks 参数进行多卡数据并行推理。
"""

import os
import gc
import json
import sys
import argparse
from tqdm import tqdm

# 防止底层多线程冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

# ── 路径设置 ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # src/open-r1-multimodal
sys.path.insert(0, PROJECT_DIR)

from qwen_omni_utils import process_mm_info

# ── 复用音频检测函数 ─────────────────────────────────────────────
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


# ── Collector System Prompt ─────────────────────────────────────
COLLECTOR_SYSTEM_PROMPT = """You are a professional multimodal information Collector. Your task is to carefully watch the provided video and listen to its audio, then extract all information that is relevant to the user's query.

You should focus on:
- Visual details: facial expressions, body language, gestures, movements, scene changes, objects, and interactions between people.
- Audio details: tone of voice, emotions conveyed through speech, background sounds, music, dialogue content, and any auditory cues.
- Temporal information: the order of events, timing of key moments, and any changes over time.

Output your findings as a clear, concise, and organized text summary. Do NOT answer the query directly — only collect and report the relevant information you observe from the video and audio."""

COLLECTOR_USER_TEMPLATE = """Please collect all relevant information from the video and audio that relates to the following query:

Query: "{problem}"

Report your observations in a structured and detailed manner."""


# ── 主函数 ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run Collector preprocessing using Qwen2.5-Omni-7B")
    parser.add_argument("--model_path", type=str, required=True, help="Qwen2.5-Omni-7B 模型路径")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSON 文件路径")
    parser.add_argument("--video_folder", type=str, required=True, help="视频文件的根目录")
    parser.add_argument("--chunk_id", type=int, default=0, help="当前分块 ID")
    parser.add_argument("--num_chunks", type=int, default=1, help="总分块数")
    parser.add_argument("--device", type=str, default="cuda:0", help="运行设备")
    parser.add_argument("--max_frames", type=int, default=150, help="视频最大采帧数 (设为-1则不控制)")
    parser.add_argument("--fps", type=float, default=1.0, help="视频采样帧率 (设为-1则不控制)")
    parser.add_argument("--min_pixels", type=int, default=28224, help="视频最小像素 (设为-1则不控制)")
    parser.add_argument("--max_pixels", type=int, default=50176, help="视频最大像素 (设为-1则不控制)")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="模型最大生成 token 数")
    parser.add_argument("--raw_video", action="store_true", help="是否直接输入原视频（不进行帧采样和像素限制）")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    # ── 1. 加载数据 ─────────────────────────────────────────────
    print(f"[Chunk {args.chunk_id}] 正在加载数据: {args.input_file}", flush=True)
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    chunk_size = total // args.num_chunks
    start_idx = args.chunk_id * chunk_size
    end_idx = start_idx + chunk_size if args.chunk_id < args.num_chunks - 1 else total
    data_chunk = data[start_idx:end_idx]

    # ── 1.5 断点续跑逻辑 ────────────────────────────────────────
    base, ext = os.path.splitext(args.output_file)
    chunk_output = f"{base}_chunk_{args.chunk_id}.jsonl"
    processed_keys = set()

    if os.path.exists(chunk_output):
        with open(chunk_output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = record.get("path", record.get("video", record.get("problem", "")))
                    processed_keys.add(key)
                except json.JSONDecodeError:
                    pass
        print(f"[Chunk {args.chunk_id}] 检测到断点文件 {chunk_output}，已跳过 {len(processed_keys)} 条记录", flush=True)

    print(f"[Chunk {args.chunk_id}] 总样本: {total}, 本 chunk 范围: [{start_idx}, {end_idx}), 共 {len(data_chunk)} 条", flush=True)

    # ── 2. 加载模型 & Processor ─────────────────────────────────
    print(f"[Chunk {args.chunk_id}] 正在加载 Qwen2.5-Omni-7B 模型: {args.model_path}", flush=True)

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device=device)
    model.eval()

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    print(f"[Chunk {args.chunk_id}] 模型加载完成!", flush=True)

    # ── 显存监控函数 ────────────────────────────────────────────
    def print_gpu_memory():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        return f"GPU显存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB"

    # ── 3. 推理循环与即时保存 ───────────────────────────────────
    failed_output = os.path.join(SCRIPT_DIR, f"collector_failed_chunk_{args.chunk_id}.jsonl")
    with open(chunk_output, "a", encoding="utf-8") as f_out, \
         open(failed_output, "a", encoding="utf-8") as f_fail:
        for idx, sample in enumerate(tqdm(data_chunk, desc=f"Collector Chunk {args.chunk_id}")):
            sample_key = sample.get("path", sample.get("video", sample.get("problem", "")))
            if sample_key in processed_keys:
                continue

            # 定期清理显存（每10个样本）
            if idx > 0 and idx % 2 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"[Chunk {args.chunk_id}] 第{idx}个样本，清理显存: {print_gpu_memory()}", flush=True)

            vid_name = sample.get("path", sample.get("video", ""))
            video_path = os.path.abspath(os.path.join(args.video_folder, vid_name))
            problem = sample.get("problem", "")

            if not os.path.exists(video_path):
                print(f"[Chunk {args.chunk_id}] ⚠️ 视频不存在，跳过: {video_path}", flush=True)
                f_fail.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_fail.flush()
                os.fsync(f_fail.fileno())
                continue

            # 检测音频
            use_audio = check_if_video_has_audio(video_path)

            # 构建对话
            vid_dict = {
                "type": "video",
                "video": video_path,
            }
            if not args.raw_video:
                if args.max_frames > 0:
                    vid_dict["max_frames"] = args.max_frames
                if args.fps > 0:
                    vid_dict["fps"] = args.fps
                if args.min_pixels > 0:
                    vid_dict["min_pixels"] = args.min_pixels
                if args.max_pixels > 0:
                    vid_dict["max_pixels"] = args.max_pixels

            user_content = [vid_dict]
            if use_audio:
                user_content.append({"type": "audio", "audio": video_path})

            user_content.append({"type": "text", "text": COLLECTOR_USER_TEMPLATE.format(problem=problem)})

            conversation = [
                {"role": "system", "content": [{"type": "text", "text": COLLECTOR_SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]
            print(f"{conversation}")

            try:
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
                text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                inputs = processor(
                    text=text,
                    audio=audios if audios else None,
                    images=images if images else None,
                    videos=videos if videos else None,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=use_audio,
                ).to(device)

                inputs = {
                        k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v
                        for k, v in inputs.items()
                        }

                with torch.inference_mode():
                    text_ids, audio = model.generate(
                        **inputs,
                        use_audio_in_video=use_audio,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )

                input_len = inputs['input_ids'].size(1)
                generated_ids = text_ids[:, input_len:]

                response = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                collector_text = response[0] if isinstance(response, list) else response
                sample["collector_text"] = collector_text.strip()
                print(f"collector_text: {collector_text}")
                print(f"[Chunk {args.chunk_id}] ✅ {vid_name} collector_text 长度: {len(sample['collector_text'])}", flush=True)

            except RuntimeError as e:
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    print(f"[Chunk {args.chunk_id}] ⚠️ CUDA错误，清理显存后重试: {vid_name}", flush=True)
                    print(f"[Chunk {args.chunk_id}] 错误信息: {e}", flush=True)

                    # 强制清理显存
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print(f"[Chunk {args.chunk_id}] {print_gpu_memory()}", flush=True)

                    # 重试一次
                    try:
                        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
                        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                        inputs = processor(
                            text=text,
                            audio=audios if audios else None,
                            images=images if images else None,
                            videos=videos if videos else None,
                            return_tensors="pt",
                            padding=True,
                            use_audio_in_video=use_audio,
                        ).to(device)

                        inputs = {
                                k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v
                                for k, v in inputs.items()
                                }

                        with torch.inference_mode():
                            text_ids, audio = model.generate(
                                **inputs,
                                use_audio_in_video=use_audio,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                            )

                        input_len = inputs['input_ids'].size(1)
                        generated_ids = text_ids[:, input_len:]

                        response = processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        collector_text = response[0] if isinstance(response, list) else response
                        sample["collector_text"] = collector_text.strip()
                        print(f"[Chunk {args.chunk_id}] ✅ 重试成功: {vid_name}", flush=True)
                    except Exception as retry_e:
                        print(f"[Chunk {args.chunk_id}] ❌ 重试失败: {vid_name}, 错误: {retry_e}", flush=True)
                        f_fail.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        f_fail.flush()
                        os.fsync(f_fail.fileno())
                        continue
                else:
                    raise
            except Exception as e:
                print(f"[Chunk {args.chunk_id}] ❌ 处理失败: {vid_name}, 错误: {e}", flush=True)
                f_fail.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_fail.flush()
                os.fsync(f_fail.fileno())
                continue

            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            f_out.flush()
            os.fsync(f_out.fileno())

            gc.collect()
            torch.cuda.empty_cache()

    print(f"[Chunk {args.chunk_id}] ✅ 本 chunk 处理完成: {chunk_output}", flush=True)


# ── 合并各 chunk 结果 ───────────────────────────────────────────
def merge_chunks():
    parser = argparse.ArgumentParser(description="Merge collector chunk results")
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

    # ── 合并 failed 记录 ──────────────────────────────────────
    failed_merged = []
    for i in range(args.num_chunks):
        failed_file = os.path.join(SCRIPT_DIR, f"collector_failed_chunk_{i}.jsonl")
        if not os.path.exists(failed_file):
            continue
        with open(failed_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    failed_merged.append(json.loads(line))
    failed_output = os.path.join(SCRIPT_DIR, "collector_failed.json")
    with open(failed_output, "w", encoding="utf-8") as f:
        json.dump(failed_merged, f, ensure_ascii=False, indent=4)
    print(f"⚠️ 失败样本合并完成! 总共 {len(failed_merged)} 条, 已保存到: {failed_output}")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge_chunks()
    else:
        main()
