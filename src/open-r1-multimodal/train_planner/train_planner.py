# -*- coding: utf-8 -*-
"""
train_planner.py
================
使用 Qwen/Qwen2.5-Omni-7B 作为基座模型，通过 LoRA 对 Planner 角色进行
有监督微调 (SFT)。训练数据来自预处理后包含 collector_text 和 planner_path
的 JSON 文件。

训练只跑 1 个 Epoch，最终仅保存 LoRA 权重。
"""

import os
import gc
import json
import sys
import math
import logging

import argparse
from dataclasses import dataclass, field

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

# 防止底层多线程冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ── 路径设置 ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # src/open-r1-multimodal
sys.path.insert(0, PROJECT_DIR)

from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from qwen_omni_utils import process_mm_info

logger = logging.getLogger(__name__)

# ── 复用音频检测函数 ─────────────────────────────────────────────
def check_if_video_has_audio(video_path):
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


# ── Planner System Prompt ───────────────────────────────────────
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


# ── 自定义数据集 ────────────────────────────────────────────────
class PlannerDataset(Dataset):
    """读取预处理后的 JSON 数据文件，构建 Planner 训练样本。"""

    def __init__(
        self,
        data_files: List[str],
        video_folder: str,
        processor: Qwen2_5OmniProcessor,
        max_frames: int = 150,
        fps: float = 1.0,
        min_pixels: int = 28224,
        max_pixels: int = 50176,
        raw_video: bool = False,
    ):
        self.processor = processor
        self.video_folder = video_folder
        self.max_frames = max_frames
        self.fps = fps
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.raw_video = raw_video

        # 加载所有数据文件
        self.samples = []
        for fpath in data_files:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for sample in data:
                    # 排除缺少必要字段的样本
                    if "planner_path" not in sample:
                        continue
                    if "collector_text" not in sample:
                        continue
                    self.samples.append(sample)

        logger.info(f"加载了 {len(self.samples)} 个有效训练样本（来自 {len(data_files)} 个文件）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PlannerDataCollator:
    """数据整理器：将原始 sample 转换为模型输入，并严格实现 Label Masking。"""

    def __init__(
        self,
        processor: Qwen2_5OmniProcessor,
        video_folder: str,
        max_frames: int = 150,
        fps: float = 1.0,
        min_pixels: int = 28224,
        max_pixels: int = 50176,
        raw_video: bool = False,
    ):
        self.processor = processor
        self.video_folder = video_folder
        self.max_frames = max_frames
        self.fps = fps
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.raw_video = raw_video

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        对 batch 中的每个 sample 逐条处理（因多模态输入长度不一致，
        batch_size 在训练时通常设为 1 或使用 gradient accumulation）。
        """
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_extras = {}  # 用于存放多模态张量

        for sample in batch:
            result = self._process_single(sample)
            if result is None:
                continue
            all_input_ids.append(result["input_ids"])
            all_attention_mask.append(result["attention_mask"])
            all_labels.append(result["labels"])

            # 收集其他多模态键
            for k, v in result.items():
                if k not in ("input_ids", "attention_mask", "labels"):
                    if k not in all_extras:
                        all_extras[k] = []
                    all_extras[k].append(v)

        if not all_input_ids:
            # 如果整个 batch 都失败了，返回一个空的 dummy
            dummy = torch.zeros(1, 1, dtype=torch.long)
            return {"input_ids": dummy, "attention_mask": dummy, "labels": dummy.fill_(-100)}

        # 对齐 padding（右侧补 pad）
        max_len = max(ids.shape[1] for ids in all_input_ids)
        pad_token_id = self.processor.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, lbl in zip(all_input_ids, all_attention_mask, all_labels):
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((1, pad_len), pad_token_id, dtype=torch.long)], dim=1)
                mask = torch.cat([mask, torch.zeros(1, pad_len, dtype=torch.long)], dim=1)
                lbl = torch.cat([lbl, torch.full((1, pad_len), -100, dtype=torch.long)], dim=1)
            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)
            padded_labels.append(lbl)

        result_dict = {
            "input_ids": torch.cat(padded_input_ids, dim=0),
            "attention_mask": torch.cat(padded_attention_mask, dim=0),
            "labels": torch.cat(padded_labels, dim=0),
        }

        # 合并多模态的额外键
        for k, v_list in all_extras.items():
            if isinstance(v_list[0], torch.Tensor):
                # 尝试 cat（形状不一致时取第一个，因 bs=1 居多）
                try:
                    result_dict[k] = torch.cat(v_list, dim=0)
                except:
                    result_dict[k] = v_list[0]
            elif isinstance(v_list[0], list):
                # 展开嵌套 list
                merged = []
                for vv in v_list:
                    merged.extend(vv)
                result_dict[k] = merged
            else:
                result_dict[k] = v_list[0]

        return result_dict

    def _process_single(self, sample: Dict) -> Optional[Dict[str, torch.Tensor]]:
        """处理单个样本：构建完整 conversation → tokenize → label mask。"""
        vid_name = sample.get("path", sample.get("video", ""))
        video_path = os.path.abspath(os.path.join(self.video_folder, vid_name))
        problem = sample.get("problem", "")
        collector_text = sample.get("collector_text", "")
        planner_path = sample.get("planner_path", "")

        if not os.path.exists(video_path):
            logger.warning(f"视频不存在，跳过: {video_path}")
            return None

        use_audio = check_if_video_has_audio(video_path)

        # ── 构建视频字典 ────────────────────────────────────────
        vid_dict = {"type": "video", "video": video_path}
        if not self.raw_video:
            if self.max_frames > 0:
                vid_dict["max_frames"] = self.max_frames
            if self.fps > 0:
                vid_dict["fps"] = self.fps
            if self.min_pixels > 0:
                vid_dict["min_pixels"] = self.min_pixels
            if self.max_pixels > 0:
                vid_dict["max_pixels"] = self.max_pixels

        user_content = [vid_dict]
        if use_audio:
            user_content.append({"type": "audio", "audio": video_path})
        user_content.append({
            "type": "text",
            "text": PLANNER_USER_TEMPLATE.format(
                collector_text=collector_text,
                problem=problem,
            ),
        })

        # ── 构建完整对话（含 assistant 回复）─────────────────────
        conversation_full = [
            {"role": "system", "content": [{"type": "text", "text": PLANNER_SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": planner_path}]},
        ]

        # ── 构建只含 prompt 部分的对话（用于计算 mask 长度）─────
        conversation_prompt = [
            {"role": "system", "content": [{"type": "text", "text": PLANNER_SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]

        try:
            # 处理多模态信息
            audios, images, videos = process_mm_info(conversation_full, use_audio_in_video=use_audio)

            # ==================================================
            # 完整序列（prompt + assistant 回复）
            # ==================================================
            full_text = self.processor.apply_chat_template(
                conversation_full, tokenize=False, add_generation_prompt=False,
            )
            full_inputs = self.processor(
                text=full_text,
                audio=audios if audios else None,
                images=images if images else None,
                videos=videos if videos else None,
                return_tensors="pt",
                padding=False,
                use_audio_in_video=use_audio,
            )

            # ==================================================
            # 仅 prompt 部分（用于确定需要 mask 的长度）
            # ==================================================
            prompt_text = self.processor.apply_chat_template(
                conversation_prompt, tokenize=False, add_generation_prompt=True,
            )
            prompt_inputs = self.processor(
                text=prompt_text,
                audio=audios if audios else None,
                images=images if images else None,
                videos=videos if videos else None,
                return_tensors="pt",
                padding=False,
                use_audio_in_video=use_audio,
            )

            full_ids = full_inputs["input_ids"]       # [1, full_len]
            prompt_len = prompt_inputs["input_ids"].shape[1]

            # ==================================================
            # Label Masking: prompt 部分设为 -100，仅在回复部分计算 loss
            # ==================================================
            labels = full_ids.clone()
            labels[0, :prompt_len] = -100  # 将 prompt token 的 label 设为 -100

            result = {
                "input_ids": full_ids,
                "attention_mask": full_inputs["attention_mask"],
                "labels": labels,
            }

            # 保留多模态相关的额外键 (如 pixel_values, audio_features 等)
            for k, v in full_inputs.items():
                if k not in ("input_ids", "attention_mask"):
                    result[k] = v

            return result

        except Exception as e:
            logger.error(f"处理样本失败: {vid_name}, 错误: {e}")
            return None


# ── 主函数 ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Planner LoRA on Qwen2.5-Omni-7B")

    # 模型参数
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Qwen2.5-Omni-7B 模型路径")
    parser.add_argument("--output_dir", type=str, required=True, help="LoRA 权重保存目录")

    # 数据参数
    parser.add_argument("--data_files", type=str, nargs="+", required=True, help="训练数据 JSON 文件路径（可传多个）")
    parser.add_argument("--video_folder", type=str, required=True, help="视频文件的根目录")

    # 视频处理参数
    parser.add_argument("--max_frames", type=int, default=150, help="视频最大采帧数")
    parser.add_argument("--fps", type=float, default=1.0, help="视频采样帧率")
    parser.add_argument("--min_pixels", type=int, default=28224, help="视频最小像素")
    parser.add_argument("--max_pixels", type=int, default=50176, help="视频最大像素")
    parser.add_argument("--raw_video", action="store_true", help="是否直接输入原视频（不进行帧采样和像素限制）")

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora_bias", type=str, default="none", help="LoRA bias 类型")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                       default=["q_proj", "k_proj", "v_proj", "o_proj"],
                       help="LoRA 目标模块")

    # 训练参数
    parser.add_argument("--num_train_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每 GPU 训练 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup 比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型")
    parser.add_argument("--logging_steps", type=int, default=5, help="日志打印间隔")
    parser.add_argument("--save_steps", type=int, default=200, help="保存 checkpoint 间隔")
    parser.add_argument("--save_total_limit", type=int, default=3, help="最多保留 checkpoint 数")
    parser.add_argument("--bf16", action="store_true", default=True, help="使用 bf16 训练")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="启用梯度检查点")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="DataLoader worker 数")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed 配置文件路径")

    # 调试
    parser.add_argument("--debug_print_samples", type=int, default=3, help="打印前 N 个样本的 token 和 label 信息")

    args = parser.parse_args()

    # ── 配置日志：同时输出到控制台和文件 ──────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    log_dir = SCRIPT_DIR
    log_file = os.path.join(log_dir, f"train_rank{local_rank}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )

    # ── 1. 加载 Processor ───────────────────────────────────────
    logger.info(f"正在加载 Processor: {args.model_name_or_path}")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name_or_path)

    # 确保 pad_token 存在
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── 2. 构建 Dataset ─────────────────────────────────────────
    logger.info(f"正在加载训练数据: {args.data_files}")
    train_dataset = PlannerDataset(
        data_files=args.data_files,
        video_folder=args.video_folder,
        processor=processor,
        max_frames=args.max_frames,
        fps=args.fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        raw_video=args.raw_video,
    )

    data_collator = PlannerDataCollator(
        processor=processor,
        video_folder=args.video_folder,
        max_frames=args.max_frames,
        fps=args.fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        raw_video=args.raw_video,
    )

    # ── 2.5 调试：打印前几个样本 ────────────────────────────────
    if args.debug_print_samples > 0:
        logger.info(f"=== 调试：打印前 {args.debug_print_samples} 个样本的 token 与 label ===")
        for i in range(min(args.debug_print_samples, len(train_dataset))):
            sample = train_dataset[i]
            result = data_collator._process_single(sample)
            if result is None:
                logger.warning(f"  样本 {i}: 处理失败，跳过")
                continue
            input_ids = result["input_ids"][0]
            labels = result["labels"][0]

            total_tokens = input_ids.shape[0]
            masked_tokens = (labels == -100).sum().item()
            target_tokens = total_tokens - masked_tokens

            logger.info(f"  样本 {i}: 总 token 数 = {total_tokens}, "
                        f"被 Mask 的 Prompt token 数 = {masked_tokens}, "
                        f"参与 Loss 计算的 Target token 数 = {target_tokens}")

            # 解码回答部分
            target_ids = labels[labels != -100]
            if len(target_ids) > 0:
                target_text = processor.tokenizer.decode(target_ids, skip_special_tokens=True)
                logger.info(f"  样本 {i}: Target 解码文本 = {target_text[:200]}")

            # 清理
            del result
            gc.collect()

    # ── 3. 加载模型 ─────────────────────────────────────────────
    logger.info(f"正在加载模型: {args.model_name_or_path}")

    # 临时禁用 torch.load 安全检查（Qwen2.5-Omni 需要加载 speaker embeddings）
    os.environ['HF_HUB_DISABLE_TORCH_LOAD_SAFETY_CHECK'] = '1'

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    model = model.thinker

    # ── 4. 配置 LoRA ────────────────────────────────────────────
    logger.info(f"正在配置 LoRA: r={args.lora_r}, alpha={args.lora_alpha}, "
                f"dropout={args.lora_dropout}, target_modules={args.lora_target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=args.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 启用梯度检查点
    if args.gradient_checkpointing:
        model.enable_input_require_grads() 
        model.gradient_checkpointing_enable()

    # ── 5. 配置训练参数 ─────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,  # 多模态数据必须设为 False
        report_to="tensorboard",
        deepspeed=args.deepspeed,
        save_safetensors=True,
    )

    # ── 6. 初始化 Trainer ───────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # ── 7. 开始训练 ─────────────────────────────────────────────
    logger.info("🚀 开始训练 Planner LoRA ...")
    trainer.train()

    # ── 8. 保存 LoRA 权重 ───────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "Planner")
    logger.info(f"💾 正在保存 LoRA 权重到: {final_dir}")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    logger.info("✅ Planner LoRA 训练完成!")


if __name__ == "__main__":
    main()
