# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.
# GRPO (Group Relative Policy Optimization) Trainer for VideoMind Grounder

import copy
import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import warnings

from .custom_trainer import CustomTrainer, gather_lora_params
from nncore.ops import temporal_iou
from videomind.constants import IGNORE_INDEX

logger = logging.getLogger('grpo_training')


def unwrap_model(model):
    """
    Recursively unwrap DDP / DeepSpeed / PeftModel wrappers
    to get the underlying VideoMind model that has `self.reg`.
    """
    # Unwrap DDP / DeepSpeed
    while hasattr(model, 'module'):
        model = model.module
    # Unwrap PeftModel -> base_model -> model
    if hasattr(model, 'base_model'):
        inner = model.base_model
        if hasattr(inner, 'model'):
            return inner.model
    return model


class GRPOTrainer(CustomTrainer):
    """
    Custom Trainer for GRPO training on VideoMind Grounder.

    This trainer implements Group Relative Policy Optimization (GRPO):
    1. Use model.generate() to trigger timestamp-decoder, producing N candidate intervals
    2. Apply softmax on confidence scores to get π(o|q) probability distribution
    3. Compute IoU-based rewards and normalize within group (advantages)
    4. Optimize policy with clipped surrogate objective + KL divergence penalty
    """

    # Timestamp decoder component names used for old policy snapshot
    _DECODER_KEYS = [
        'vis_proj', 'reg_proj', 'vis_fuse', 'vis_norm',
        'vis_emb', 'reg_emb', 'pyramid', 'class_head', 'coord_head', 'coef'
    ]

    def __init__(
        self,
        *args,
        num_candidates: int = 8,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.01,
        old_policy_sync_interval: int = 50,
        old_policy_model: Optional[torch.nn.Module] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_candidates = num_candidates
        self.clip_epsilon = clip_epsilon
        self.kl_beta = kl_beta
        self.old_policy_sync_interval = old_policy_sync_interval
        self.old_policy_model = old_policy_model

        # Initialize old policy snapshot (lightweight, CPU-stored)
        self._init_old_policy()

    # ==================== Old Policy Snapshot Methods ====================

    def _init_old_policy(self):
        """
        深拷贝 timestamp decoder 模块作为独立的旧策略副本。
        这些是完整的 nn.Module 实例，存储在同一设备上，
        不会与训练模型的计算图产生任何交互。
        """
        raw_model = unwrap_model(self.model)
        self.old_decoder = {}
        total_params = 0
        for key in self._DECODER_KEYS:
            module = getattr(raw_model, key, None)
            if module is not None:
                # Deep copy the entire module (independent copy, no shared tensors)
                copied = copy.deepcopy(module)
                # Freeze — old policy should never have gradients
                for p in copied.parameters():
                    p.requires_grad = False
                self.old_decoder[key] = copied
                total_params += sum(p.numel() for p in copied.parameters())
        logger.info(f"[GRPO] Initialized old policy decoder: "
                    f"{len(self.old_decoder)} modules, {total_params:,} params, "
                    f"sync every {self.old_policy_sync_interval} steps")

    def _sync_old_policy(self):
        """
        将当前模型的 timestamp decoder 参数同步到旧策略副本。
        """
        raw_model = unwrap_model(self.model)
        for key in self._DECODER_KEYS:
            src_module = getattr(raw_model, key, None)
            dst_module = self.old_decoder.get(key)
            if src_module is not None and dst_module is not None:
                # Copy state_dict (no in-place modification of training model)
                dst_module.load_state_dict(
                    {k: v.detach().clone() for k, v in src_module.state_dict().items()}
                )
        logger.info(f"[GRPO] Synced old policy at step {self.state.global_step}")

    def _rescore_with_old_policy(
        self,
        raw_model,
        video_grid_thw,
        num_candidates: int
    ) -> Optional[torch.Tensor]:
        """
        用独立的旧策略 decoder 模块重新打分，获取 old_logprobs。

        使用 self.old_decoder 中的独立模块副本运行 decoder 逻辑，
        完全不修改训练模型的任何参数，避免破坏计算图。
        """
        # 从训练模型中获取所需的 cached hidden states（只读）
        if not hasattr(raw_model, 'cache_norm_state') or raw_model.cache_norm_state is None:
            return None
        if not hasattr(raw_model, 'cache_vision_inds') or len(raw_model.cache_vision_inds) == 0:
            return None
        if not hasattr(raw_model.model.norm, 'state') or raw_model.model.norm.state is None:
            return None

        # 确保旧 decoder 模块在正确的设备上（多卡场景下可能是 cuda:0 或 cuda:1）
        target_device = raw_model.model.norm.state.device
        for key, module in self.old_decoder.items():
            module.to(target_device)

        try:
            s, e = raw_model.cache_vision_inds[0][0]
            window = int(video_grid_thw[0][1] * video_grid_thw[0][2] / 4)
            if video_grid_thw[0][0] * window != e - s:
                return None

            norm_state = raw_model.model.norm.state  # [B, L, hidden_size]

            with torch.no_grad():
                # 用旧策略的 reg_proj 处理 reg token
                if hasattr(raw_model, '_cached_reg_inds') and raw_model._cached_reg_inds is not None:
                    reg_inds = raw_model._cached_reg_inds
                    reg_tokens = self.old_decoder['reg_proj'](norm_state[0, reg_inds, None])
                else:
                    reg_tokens = self.old_decoder['reg_proj'](norm_state[0:1, -1:])

                # 用旧策略的 vis_proj 处理 visual tokens
                vis_tokens = raw_model.cache_norm_state[0:1, s:e]
                vis_tokens = vis_tokens.transpose(-1, -2)
                vis_tokens = F.avg_pool1d(vis_tokens.float(), window, stride=window).to(vis_tokens.dtype)
                vis_tokens = vis_tokens.transpose(-1, -2)
                vis_tokens = self.old_decoder['vis_proj'](vis_tokens).repeat(reg_tokens.size(0), 1, 1)

                vis_tokens = self.old_decoder['vis_emb'](vis_tokens)
                reg_tokens = self.old_decoder['reg_emb'](reg_tokens)

                # vis_pos is not trainable (learnable=False), use from raw_model
                pe = raw_model.vis_pos(vis_tokens).to(vis_tokens.dtype)

                joint_tokens = torch.cat((vis_tokens + pe, reg_tokens), dim=1)
                collected = [joint_tokens]
                for blk in self.old_decoder['vis_fuse']:
                    collected.append(blk(collected[-1]))
                collected = collected[1:]
                joint_tokens = torch.cat(collected)
                joint_tokens = self.old_decoder['vis_norm'](joint_tokens)

                video_emb = joint_tokens[:, :-1]

                b, t, c = video_emb.size()
                video_msk = video_emb.new_ones(b, t)

                if t < raw_model.vis_pad_length:
                    emb_pad = video_emb.new_zeros(b, raw_model.vis_pad_length - t, c)
                    msk_pad = video_msk.new_zeros(b, raw_model.vis_pad_length - t)
                    pymid_emb = torch.cat((video_emb, emb_pad), dim=1)
                    pymid_msk = torch.cat((video_msk, msk_pad), dim=1)
                else:
                    pymid_emb, pymid_msk = video_emb, video_msk

                pymid = self.old_decoder['pyramid'](pymid_emb, pymid_msk)
                point = raw_model.generator(pymid)  # generator has no trainable params

                out_class = [self.old_decoder['class_head'](e_).sigmoid() for e_ in pymid]
                out_class = torch.cat(out_class, dim=1)

                out_coord = [self.old_decoder['coef'](self.old_decoder['coord_head'](e_).exp(), i)
                             for i, e_ in enumerate(pymid)]
                out_coord = torch.cat(out_coord, dim=1)

                sal = out_class[0]
                bnd = out_coord[0]

                bnd[:, 0] *= -1
                bnd *= point[:, 3, None].repeat(1, 2)
                bnd += point[:, 0, None].repeat(1, 2)
                bnd /= t
                bnd = torch.cat((bnd, sal), dim=-1)

                _, inds = bnd[:, -1].sort(descending=True)
                bnd = bnd[inds]
                bnd = bnd[:100]

            # 计算 old_logprobs
            N = min(num_candidates, bnd.size(0))
            old_scores = bnd[:N, 2]
            old_logprobs = F.log_softmax(old_scores, dim=0)
            return old_logprobs.detach()

        except Exception as ex:
            logger.warning(f"[GRPO] Old policy rescore failed: {ex}")
            return None

    def compute_iou_reward(
        self,
        pred_intervals: torch.Tensor,
        gt_interval: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU-based reward for predicted intervals.

        Args:
            pred_intervals: [N, 2] predicted intervals (start_ratio, end_ratio)
            gt_interval: [2] ground truth interval (start_ratio, end_ratio)

        Returns:
            rewards: [N] IoU scores
        """
        gt_intervals = gt_interval.unsqueeze(0).expand(pred_intervals.size(0), -1)
        iou = temporal_iou(pred_intervals, gt_intervals)
        return iou.squeeze(-1)  # [N]

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards to zero mean and unit variance within group.

        Args:
            rewards: [N] raw rewards

        Returns:
            normalized_rewards: [N] normalized rewards
        """
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        normalized = (rewards - mean_reward) / (std_reward + 1e-8)
        return normalized

    def compute_kl_divergence(
        self,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between new and reference policy.

        GRPO KL formula:
        KL(π_θ || π_ref) = exp(-log_ratio) + log_ratio - 1
        where log_ratio = log(π_θ / π_ref)

        Args:
            old_logprobs: [N] log probabilities from reference policy (π_ref)
            new_logprobs: [N] log probabilities from current policy (π_θ)

        Returns:
            kl_div: scalar KL divergence
        """
        log_ratio = new_logprobs - old_logprobs
        kl_div = torch.exp(-log_ratio) + log_ratio - 1
        return kl_div.mean()

    def compute_grpo_loss(
        self,
        pred_intervals: torch.Tensor,
        gt_interval: torch.Tensor,
        new_logprobs: torch.Tensor,
        old_logprobs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss with clipped surrogate objective and KL penalty.

        Args:
            pred_intervals: [N, 2] predicted intervals
            gt_interval: [2] ground truth interval
            new_logprobs: [N] log π_θ(o|q) from current policy (softmax of scores)
            old_logprobs: [N] log π_old(o|q) from old policy (softmax of scores)

        Returns:
            dict with 'loss', 'loss_clip', 'loss_kl', 'mean_reward', etc.
        """
        # 1. Compute IoU rewards
        rewards = self.compute_iou_reward(pred_intervals, gt_interval)

        # 2. Normalize rewards (advantage estimation)
        advantages = self.normalize_rewards(rewards)

        # 3. Compute policy ratio: π_θ / π_old = exp(log π_θ - log π_old)
        log_ratio = new_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)

        # 4. Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        loss_clip = -torch.min(surr1, surr2).mean()

        # 5. KL divergence penalty
        loss_kl = self.compute_kl_divergence(old_logprobs, new_logprobs)

        # 6. Total loss
        loss = loss_clip + self.kl_beta * loss_kl

        return {
            'loss': loss,
            'loss_clip': loss_clip,
            'loss_kl': loss_kl,
            'mean_reward': rewards.mean(),
            'mean_advantage': advantages.mean(),
            'policy_ratio': ratio.mean()
        }

    def _prepare_generation_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for model.generate() by extracting the prompt portion.

        In training data, `labels[i] == -100` marks positions that are the prompt
        (not the target response). We extract these positions from input_ids to
        form the generation prompt.

        Args:
            inputs: Training batch dict with 'input_ids', 'labels', 'attention_mask',
                    'pixel_values_videos', 'video_grid_thw', etc.

        Returns:
            gen_kwargs: Dict suitable for model.generate()
        """
        input_ids = inputs['input_ids']       # [B, L]
        labels = inputs['labels']             # [B, L]
        attention_mask = inputs['attention_mask']  # [B, L]

        # Use only first sample in batch (GRPO processes one sample at a time)
        # Find prompt positions: where labels == IGNORE_INDEX (-100)
        prompt_mask = (labels[0] == IGNORE_INDEX)
        prompt_ids = input_ids[0][prompt_mask].unsqueeze(0)         # [1, prompt_len]
        prompt_attn = attention_mask[0][prompt_mask].unsqueeze(0)   # [1, prompt_len]

        gen_kwargs = {
            'input_ids': prompt_ids,
            'attention_mask': prompt_attn,
            'do_sample': False,
            'temperature': None,
            'top_p': None,
            'top_k': None,
            'repetition_penalty': None,
            'max_new_tokens': 256,
        }

        if 'pixel_values_videos' in inputs:
            gen_kwargs['pixel_values_videos'] = inputs['pixel_values_videos']
        if 'video_grid_thw' in inputs:
            gen_kwargs['video_grid_thw'] = inputs['video_grid_thw']
        if 'pixel_values' in inputs:
            gen_kwargs['pixel_values'] = inputs['pixel_values']
        if 'image_grid_thw' in inputs:
            gen_kwargs['image_grid_thw'] = inputs['image_grid_thw']

        # Concise debug info
        try:
            from videomind.constants import REG_TOKEN
            tokenizer = self.processor.tokenizer
            video_pad_id = tokenizer.convert_tokens_to_ids('<|video_pad|>')
            n_video_pad = (prompt_ids[0] == video_pad_id).sum().item()
            reg_token_id = tokenizer.convert_tokens_to_ids(REG_TOKEN)
            has_reg = (input_ids[0] == reg_token_id).any().item()
            logger.info(f"[DEBUG] prompt shape: {prompt_ids.shape}, "
                        f"video_pad: {n_video_pad}, REG in input: {has_reg}")
        except Exception as e:
            logger.warning(f"[DEBUG] Error: {e}")

        return gen_kwargs

    def sample_candidates(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_candidates: int
    ) -> tuple:
        """
        Use model.generate() to trigger the timestamp-decoder and extract
        candidate intervals from self.reg.

        VideoMind's Grounder naturally produces multiple candidate intervals
        (up to 100) with confidence scores in a single generate() call.
        We apply softmax on scores to get π(o|q).

        Args:
            model: The grounder model (may be wrapped by DDP/PeftModel)
            inputs: Training batch dict
            num_candidates: Max number of candidates to use

        Returns:
            intervals: [N, 2] predicted intervals (start_ratio, end_ratio)
            logprobs: [N] log π(o|q) = log softmax(scores)
        """
        # Get underlying model that has self.reg
        raw_model = unwrap_model(model)

        # Prepare generation inputs (prompt only, no ground truth response)
        gen_kwargs = self._prepare_generation_inputs(inputs)

        # Run generate to trigger timestamp-decoder
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                output_ids = model.generate(**gen_kwargs)

            # Diagnostic: decode generated text (truncated)
            try:
                tokenizer = self.processor.tokenizer
                generated = tokenizer.decode(output_ids[0, gen_kwargs['input_ids'].size(1):],
                                             skip_special_tokens=False)
                logger.info(f"[GRPO] Generated ({len(output_ids[0])} tokens): {generated[:200]}")
            except Exception:
                pass

        except Exception as e:
            warnings.warn(f"Generation failed in sample_candidates: {e}")
            if was_training:
                model.train()
            return None, None
        finally:
            if was_training:
                model.train()

        # Extract candidates from self.reg
        if not hasattr(raw_model, 'reg') or len(raw_model.reg) == 0:
            logger.warning(f"[GRPO] model.reg is EMPTY after generate() — timestamp-decoder did not fire")
            return None, None

        blob = raw_model.reg[0]  # [K, 3]: [start_ratio, end_ratio, score]
        N = min(num_candidates, blob.size(0))
        logger.info(f"[GRPO] model.reg has {blob.size(0)} candidates, using top-{N}")

        intervals = blob[:N, :2]       # [N, 2]
        scores = blob[:N, 2]           # [N] raw confidence scores

        # Apply softmax to convert scores → probability distribution π(o|q)
        probs = F.softmax(scores, dim=0)   # [N]
        logprobs = torch.log(probs + 1e-8) # [N] log π(o|q)

        return intervals.detach(), logprobs.detach()

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        GRPO training step.

        Unlike the base Trainer.training_step() which does forward+backward internally,
        we handle the full forward-backward cycle ourselves so we can combine
        SFT loss and GRPO loss into a single backward pass.

        Flow:
        1. Forward pass → SFT loss + cache norm.state (with gradients)
        2. Save training norm state + find reg_token positions
        3. generate() → candidate intervals + old_logprobs (detached)
        4. rescore_candidates() → new_logprobs (WITH gradients, from training hidden states)
        5. GRPO loss with ratio = new/old ≠ 1.0
        6. total_loss = sft_loss + grpo_loss → single backward
        """
        model.train()

        if 'timestamps' not in inputs or inputs['timestamps'] is None:
            # No ground truth timestamps → standard SFT only
            return super().training_step(model, inputs)

        # ========== Step 1: Forward pass to get SFT loss ==========
        # model(**inputs) runs in training mode, computes SFT + bundle_loss
        # It also sets model.norm.state (hidden states before norm, WITH gradients)
        outputs = model(**inputs)
        sft_loss = outputs.loss

        if sft_loss is None:
            warnings.warn("SFT forward returned None loss")
            return super().training_step(model, inputs)

        # ========== Step 2: Save training norm state for rescore ==========
        raw_model = unwrap_model(model)

        # Save the training norm state BEFORE generate() overwrites it
        training_norm_state = raw_model.model.norm.state  # [B, L, hidden_size] - has gradients

        # Find reg_token positions in shifted labels (same as model.py training mode)
        labels = inputs.get('labels')
        reg_inds = None
        if labels is not None and hasattr(raw_model.config, 'reg_token_id'):
            shift_labels = labels[..., 1:].contiguous()
            reg_positions = torch.where(shift_labels[0] == raw_model.config.reg_token_id)[0]
            if len(reg_positions) > 0:
                reg_inds = reg_positions

        # ========== Step 3: Extract ground truth interval ==========
        gt_intervals = inputs['timestamps'][0][0]  # list of [s, e] pairs
        if isinstance(gt_intervals, list):
            gt_interval = torch.tensor(gt_intervals[0], device=sft_loss.device, dtype=torch.float32)
        else:
            gt_interval = gt_intervals[0].to(sft_loss.device).float()

        # ========== Step 4: Sample candidates (for intervals) ==========
        candidate_intervals, _ = self.sample_candidates(
            model, inputs, self.num_candidates
        )

        if candidate_intervals is None:
            # Cannot sample → do backward on SFT loss only
            logger.warning(f"[GRPO] step={self.state.global_step} | "
                          f"No candidates, SFT only: {sft_loss.item():.4f}")
            self.accelerator.backward(sft_loss)
            return sft_loss.detach() / self.args.gradient_accumulation_steps

        # ========== Step 5: Restore training hidden states ==========
        # Restore training hidden states that were overwritten by generate()
        # training_norm_state has shape [B, L, hidden_size] and contains visual tokens
        # at positions cache_vision_inds, which rescore_candidates needs
        raw_model.cache_norm_state = training_norm_state  # visual tokens come from training forward
        raw_model.model.norm.state = training_norm_state  # reg_token hidden state also from training
        raw_model._cached_reg_inds = reg_inds
        video_grid_thw = inputs.get('video_grid_thw')

        # Ensure model is back in training mode
        model.train()

        # ========== Step 5a: OLD policy rescore (no gradients, independent modules) ==========
        # Uses separate deep-copied decoder modules (self.old_decoder) — never touches
        # the training model's parameters, so no in-place modification issues.
        old_logprobs = self._rescore_with_old_policy(
            raw_model, video_grid_thw, self.num_candidates
        )

        if old_logprobs is None:
            logger.warning(f"[GRPO] step={self.state.global_step} | "
                          f"Old policy rescore failed, SFT only: {sft_loss.item():.4f}")
            self.accelerator.backward(sft_loss)
            return sft_loss.detach() / self.args.gradient_accumulation_steps

        # ========== Step 5b: CURRENT policy rescore (WITH gradients) ==========
        # Re-run timestamp-decoder with gradients using cached training hidden states
        rescored_bnd = raw_model.rescore_candidates(video_grid_thw)

        if rescored_bnd is None:
            logger.warning(f"[GRPO] step={self.state.global_step} | "
                          f"Rescore failed, SFT only: {sft_loss.item():.4f}")
            self.accelerator.backward(sft_loss)
            return sft_loss.detach() / self.args.gradient_accumulation_steps

        # Extract new scores with gradients
        # N must be the minimum of ALL sources to avoid shape mismatch:
        # - self.num_candidates (desired count)
        # - candidate_intervals.size(0) (how many sample_candidates returned, may be < num_candidates)
        # - rescored_bnd.size(0) (how many rescore produced)
        # - old_logprobs.size(0) (how many old policy returned)
        N = min(self.num_candidates, candidate_intervals.size(0),
                rescored_bnd.size(0), old_logprobs.size(0))
        new_scores = rescored_bnd[:N, 2]  # [N] confidence scores WITH gradients
        new_logprobs = F.log_softmax(new_scores, dim=0)  # [N] log π_θ(o|q)

        # ========== Step 6: Compute GRPO loss ==========
        candidate_intervals = candidate_intervals[:N].to(sft_loss.device)
        old_logprobs = old_logprobs[:N].to(sft_loss.device)
        gt_interval = gt_interval.to(sft_loss.device)

        loss_dict = self.compute_grpo_loss(
            candidate_intervals,
            gt_interval,
            new_logprobs,
            old_logprobs
        )

        # ========== Step 7: Combine and backward ==========
        total_loss = sft_loss + loss_dict['loss']

        self.accelerator.backward(total_loss)

        # ========== Step 8: Sync old policy periodically ==========
        if (self.state.global_step > 0 and
                self.state.global_step % self.old_policy_sync_interval == 0):
            self._sync_old_policy()

        # ========== Step 9: Print & log metrics ==========
        grpo_metrics = {
            'grpo/loss_clip': loss_dict['loss_clip'].item(),
            'grpo/loss_kl': loss_dict['loss_kl'].item(),
            'grpo/mean_reward': loss_dict['mean_reward'].item(),
            'grpo/mean_advantage': loss_dict['mean_advantage'].item(),
            'grpo/policy_ratio': loss_dict['policy_ratio'].item(),
            'grpo/sft_loss': sft_loss.item(),
        }

        logger.info(f"[GRPO] step={self.state.global_step} | "
                    f"sft={sft_loss.item():.4f} | "
                    f"grpo={loss_dict['loss'].item():.4f} | "
                    f"total={total_loss.item():.4f} | "
                    f"reward={loss_dict['mean_reward'].item():.4f} | "
                    f"ratio={loss_dict['policy_ratio'].item():.4f}")

        if self.state.global_step % self.args.logging_steps == 0:
            self.log(grpo_metrics)

        return total_loss.detach() / self.args.gradient_accumulation_steps

    def _save(self, output_dir: str, state_dict=None):
        """
        Save only the grpo_grounder adapter, not the base grounder adapter.
        """
        if self.args.lora_enable:
            if hasattr(self.model, 'active_adapter'):
                active_adapter = self.model.active_adapter
                if active_adapter == 'grpo_grounder':
                    state_dict = gather_lora_params(self.model, self.args.lora_bias)
                    state_dict = {
                        k: v for k, v in state_dict.items()
                        if 'grpo_grounder' in k or 'modules_to_save' in k
                    }
                    if self.args.should_save:
                        self.model.save_pretrained(
                            output_dir,
                            state_dict=state_dict,
                            selected_adapters=['grpo_grounder']
                        )
                    return

        super()._save(output_dir, state_dict=state_dict)
