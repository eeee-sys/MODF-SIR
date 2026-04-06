# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.
# GRPO Training Script for VideoMind Grounder

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import nncore
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, HfArgumentParser, TrainingArguments as HFTrainingArguments

from videomind.constants import REG_TOKEN, SEG_E_TOKEN, SEG_S_TOKEN
from videomind.dataset import HybridDataCollator, HybridDataset
from videomind.model import MODELS
from videomind.model.builder import build_model
from train_grpo_grounder.grpo_trainer import GRPOTrainer

logger = logging.getLogger('grpo_training')


@dataclass
class ModelArguments:
    base_model_path: Optional[str] = field(default=None, metadata={"help": "Path to base model (Qwen2-VL)"})
    pretrained_grounder_path: Optional[str] = field(default=None, metadata={"help": "Path to pretrained VideoMind Grounder"})
    base_model: Optional[str] = field(default='qwen2_vl')
    conv_type: Optional[str] = field(default='chatml')
    role: Optional[str] = field(default='grounder')


@dataclass
class DataArguments:
    datasets: Optional[str] = field(default=None)
    min_video_len: Optional[int] = field(default=-1)
    max_video_len: Optional[int] = field(default=-1)
    min_num_words: Optional[int] = field(default=-1)
    max_num_words: Optional[int] = field(default=-1)
    max_retries: Optional[int] = field(default=10)


@dataclass
class GRPOArguments:
    """GRPO-specific hyperparameters"""
    num_candidates: Optional[int] = field(default=8, metadata={"help": "Number of candidates to sample per query"})
    clip_epsilon: Optional[float] = field(default=0.2, metadata={"help": "Clipping parameter for PPO-style objective"})
    kl_beta: Optional[float] = field(default=0.01, metadata={"help": "KL divergence penalty coefficient"})
    old_policy_sync_interval: Optional[int] = field(default=50, metadata={"help": "Sync old policy snapshot every N steps"})


@dataclass
class CustomArguments:
    optim: Optional[str] = field(default='adamw_torch')
    group_by_data_type: Optional[bool] = field(default=True)
    merge_adapter: Optional[bool] = field(default=False)
    lora_enable: Optional[bool] = field(default=True)  # GRPO always uses LoRA
    lora_type: Optional[str] = field(default='qkvo')
    lora_r: Optional[int] = field(default=64)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_bias: Optional[str] = field(default='none')
    lora_lr: Optional[float] = field(default=None)
    head_lr: Optional[float] = field(default=None)
    tuning_modules: Optional[str] = field(default=None)
    save_full_model: Optional[bool] = field(default=False)
    remove_unused_columns: Optional[bool] = field(default=False)


@dataclass
class TrainingArguments(CustomArguments, GRPOArguments, HFTrainingArguments):
    pass


def get_target_modules(model, lora_type, base_model):
    """Get LoRA target modules (same as original VideoMind)"""
    lora_type = lora_type.split('_')
    assert all(t in ('qkvo', 'linear', 'all') for t in lora_type)

    if base_model == 'qwen2_vl':
        # all qkvo layers in the visual encoder and the llm
        qkvo_keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attn.qkv', 'attn.proj']

        target_modules = set()
        for n, m in model.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            if 'all' not in lora_type and 'visual' in n:
                continue
            if 'qkvo' in lora_type and not any(n.endswith(k) for k in qkvo_keys):
                continue
            target_modules.add(n)
    else:
        raise ValueError(f'unknown base model: {base_model}')

    return target_modules


def setup_logging(output_dir, local_rank):
    """Configure logging to write to both console and a log file."""
    grpo_logger = logging.getLogger('grpo_training')
    grpo_logger.setLevel(logging.DEBUG)
    grpo_logger.propagate = False

    # Console handler (all ranks)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_fmt)
    grpo_logger.addHandler(console_handler)

    # File handler (rank 0 only, to avoid duplicate lines)
    if local_rank in (0, -1):
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, 'grpo_training.log')
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            '[%(asctime)s][%(levelname)s][rank%(process)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_fmt)
        grpo_logger.addHandler(file_handler)
        grpo_logger.info(f'Log file: {log_path}')

    return grpo_logger


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ========== Setup logging ==========
    setup_logging(training_args.output_dir, training_args.local_rank)

    assert model_args.role == 'grounder', "GRPO training only supports grounder role"
    assert model_args.pretrained_grounder_path is not None, "Must provide pretrained grounder path"

    config_cls, model_cls = MODELS[model_args.base_model]
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    # ========== Step 1: Load pretrained grounder ==========
    logger.info(f"Loading pretrained grounder from {model_args.pretrained_grounder_path}...")
    
    # First, load config to get base_model_name_or_path
    config = config_cls.from_pretrained(model_args.pretrained_grounder_path, torch_dtype=dtype)
    
    # Ensure base_model_name_or_path is set in config
    # PEFT uses 'base_model_name_or_path' in adapter_config.json
    if not hasattr(config, 'base_model_name_or_path') or config.base_model_name_or_path is None:
        # Fallback: try old naming or use provided path
        if hasattr(config, 'base_model_path') and config.base_model_path is not None:
            config.base_model_name_or_path = config.base_model_path
        elif model_args.base_model_path is not None:
            config.base_model_name_or_path = model_args.base_model_path
        else:
            raise ValueError(
                "base_model_name_or_path not found in pretrained grounder config. "
                "Please provide --base_model_path argument."
            )
    
    logger.info(f"Base model path: {config.base_model_name_or_path}")
    config.update(model_args.__dict__)

    # Load pretrained grounder and MERGE adapter into base weights
    # This bakes the pretrained grounder LoRA into the base model,
    # so the grounder behavior is preserved as the foundation.
    model, processor = build_model(
        model_args.pretrained_grounder_path,
        config=config,
        is_trainable=True,
        merge_adapter=True,
        dtype=dtype
    )
    
    logger.info(f"Successfully loaded and merged grounder adapter into base weights.")
    logger.info(f"  - Base model: {config.base_model_name_or_path}")
    logger.info(f"  - Model type after merge: {type(model).__name__}")
    
    # ========== Step 2: Add grpo_grounder LoRA adapter ==========
    logger.info("Adding grpo_grounder LoRA adapter on top of merged model...")
    
    # Get target modules
    target_modules = get_target_modules(model, training_args.lora_type, model.config.base_model)
    tune_lm_head = True  # Grounder tunes lm_head
    
    logger.info(f'LoRA target modules for grpo_grounder: {target_modules}')
    
    grpo_lora_config = LoraConfig(
        task_type='CAUSAL_LM',
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        target_modules=list(target_modules),
        modules_to_save=['embed_tokens', 'lm_head'] if tune_lm_head else None
    )
    
    # Apply LoRA on the merged model — grpo_grounder is the ONLY adapter
    model = get_peft_model(model, grpo_lora_config, adapter_name='grpo_grounder')
    logger.info(f"Added grpo_grounder adapter. Active adapter: {model.active_adapter}")
    
    # ========== Step 3: Setup tokenizer and special tokens ==========
    new_tokens = processor.tokenizer.add_special_tokens(
        dict(additional_special_tokens=[REG_TOKEN, SEG_S_TOKEN, SEG_E_TOKEN])
    )
    logger.info(f'Added {new_tokens} new token(s)')

    model.config.reg_token_id = processor.tokenizer.convert_tokens_to_ids(REG_TOKEN)
    model.config.seg_s_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_S_TOKEN)
    model.config.seg_e_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_E_TOKEN)

    if new_tokens > 0 and len(processor.tokenizer) > model.config.vocab_size:
        logger.info(f'Expanding vocab size: {model.config.vocab_size} -> {len(processor.tokenizer)}')
        model.resize_token_embeddings(len(processor.tokenizer))
        i_emb = model.get_input_embeddings().weight.data
        o_emb = model.get_output_embeddings().weight.data
        i_emb[-new_tokens:] = i_emb[:-new_tokens].mean(0, keepdim=True)
        o_emb[-new_tokens:] = o_emb[:-new_tokens].mean(0, keepdim=True)

    # ========== Step 4: Setup trainable parameters ==========
    tuning_modules = [] if training_args.tuning_modules is None else training_args.tuning_modules.split(',')

    head_keys = [
        'vis_proj', 'reg_proj', 'vis_fuse', 'vis_norm', 'vis_pos', 'vis_emb', 'reg_emb', 
        'pyramid', 'class_head', 'coord_head', 'coef', 'bundle_loss'
    ]

    # Setup trainable parameters:
    # - grpo_grounder LoRA params: trainable
    # - Grounder head components (vis_proj, reg_proj, etc.): trainable
    # - Everything else (base model weights): frozen
    for n, p in model.named_parameters():
        if 'lora_' in n:  # LoRA parameters (grpo_grounder)
            p.requires_grad = True
        elif 'modules_to_save' in n:  # embed_tokens, lm_head via PeftModel
            p.requires_grad = True
        elif any(k in n for k in head_keys):  # Grounder head components
            p.requires_grad = True
        elif 'projector' in tuning_modules and 'visual.merger' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    if training_args.local_rank in (0, -1):
        logger.info("\n" + "="*80)
        logger.info("Parameter Status:")
        logger.info("="*80)
        for n, p in model.named_parameters():
            logger.debug(f"{str(p.requires_grad):5s} | {str(p.dtype):15s} | {str(tuple(p.shape)):30s} | {n}")
        logger.info("="*80)

        total_params = sum(p.numel() for p in model.parameters())
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = round(learnable_params / total_params * 100, 2) if total_params > 0 else 0
        logger.info(f'Total params: {total_params:,} | Learnable params: {learnable_params:,} ({ratio}%)')

        i_size = model.get_input_embeddings().num_embeddings
        o_size = model.get_output_embeddings().out_features
        assert i_size == o_size, (i_size, o_size)
        logger.info(f'Tokenizer size: {len(processor.tokenizer)} | Vocab size: {model.config.vocab_size} | Embed size: {i_size}')
        logger.info("="*80 + "\n")

    # ========== Step 5: Initialize trainer and start training ==========
    # Note: on-policy GRPO — use detached current logprobs as reference
    # (no deepcopy needed, saves ~50% GPU memory)
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        data_collator=HybridDataCollator(processor.tokenizer),
        train_dataset=HybridDataset(processor, model.config, model_args, data_args, training_args),
        processor=processor,
        head_keys=head_keys,
        num_candidates=training_args.num_candidates,
        clip_epsilon=training_args.clip_epsilon,
        kl_beta=training_args.kl_beta,
        old_policy_sync_interval=training_args.old_policy_sync_interval,
        old_policy_model=None
    )

    has_ckpt = bool(nncore.find(training_args.output_dir, 'checkpoint-*'))
    trainer.train(resume_from_checkpoint=has_ckpt)

    trainer.save_state()
    trainer.gather_and_save_model()
    
    logger.info("\n" + "="*80)
    logger.info("GRPO Training Completed!")
    logger.info(f"Model saved to: {training_args.output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    train()
