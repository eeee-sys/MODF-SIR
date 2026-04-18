<h2 align="center">MODF-SIR: a Multi-agent Omni-modal Distilled Framework for Social Intelligence Reasoning</h2>

<p align="center">
  <a href="https://huggingface.co/Harry-1234/MODF-SIR" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
  <a href="https://huggingface.co/datasets/Harry-1234/IntentRouterTrain/" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a>
  <a href="https://huggingface.co/spaces/Harry-1234/MODF-SIR" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg"></a>
</p>

**MODF-SIR** is a lightweight MLLM-based, distillation-augmented, multi-agent collaborative framework for social intelligence reasoning.

<p align="center"><img width="100%" height="100%" src="assets/method.png"></p>
## 👀 MODF-SIR Overview
Social intelligence reasoning, which involves decoding implicit human intentions and emotional dynamics, remains a significant challenge for Multimodal Large Language Models (MLLMs). When processing unconstrained omni-modal data streams, traditional flat reasoning paradigms often suffer from cognitive overload and hallucination cascades. To overcome these limitations, we propose MODF-SIR. Inspired by the cognitive ``Dual-Process Theory'', MODF-SIR replaces black-box inference with a collaborative multi-agent architecture. Initially, an Endogenous Long-Tail Retriever Agent extracts subtle, long-tail multimodal cues and textualizes them to prevent critical signals from being overshadowed. Guided by these cues, an Asymmetric Knowledge Distilled Router Agent dynamically assigns the reasoning pathway. For complex implicit queries, a GRPO Grounder Agent performs precise spatiotemporal localization. Subsequently, an Omni-Modal Long-Tail Reasoner Agent executes extended Chain-of-Thought (CoT) reasoning. To address the generation-evaluation gap, a Test-Time Adaptation Reviser Agent iteratively refines the reasoning outputs via closed-loop evaluation and dynamic LoRA updates. Extensive experiments demonstrate that MODF-SIR achieves state-of-the-art results across three benchmarks: Daily-Omni, IntentBench, and WorldSense. Notably, our framework significantly surpasses existing open-source video-audio MLLMs and approaches the performance of proprietary models like GPT-4o and Gemini, highlighting its efficacy in human intent modeling.


## 🔥 News
- 🚀 MODF-SIR is ready on [Hugging Face Model](https://huggingface.co/Harry-1234/MODF-SIR). Check it out!
- 📦 Training Data is ready on [Hugging Face Dataset](https://huggingface.co/datasets/Harry-1234/IntentRouterTrain/). Start it!
- 🕹️ Online demo is ready [Hugging Face Space](https://huggingface.co/spaces/Harry-1234/MODF-SIR). Play with it!
- ⭐️ Code, model, dataset and online demo release.

## 🏆 MODF-SIR on Public Benchmarks
<p align="center">
    <img src="assets/dailyomni.png" width="100%" height="100%">
</p>

<p align="center">
    <img src="assets/worldsense.png" width="100%" height="100%">
</p>

<p align="center">
    <img src="assets/intentbench.png" width="100%" height="100%">
</p>

## 🕹️ Demo

Online demo is ready [Hugging Face Space](https://huggingface.co/spaces/Harry-1234/MODF-SIR). Play with it!


https://github.com/user-attachments/assets/60cf2207-d49f-4bea-8713-83d36e9e7c39



## 🚀 Training
Our codebase supports training and evaluating on [10 video datasets and benchmarks] with the following features.

- Hardware settings: NVIDIA GPU A100 / H100, Single-Node / Multi-Node
- Efficient training techniques: DeepSpeed ZeRO, BF16, LoRA, SDPA, FlashAttention2
- Customizing the base LLM and conversation templates
- Monitoring the training process via Tensorboard / Wandb
- Group sampling for mixed dataset training
- Multi-process / multi-device evaluation on public benchmarks

See [TRAIN.md](docs/TRAIN.md) for a quick start guide.

## 🔮 Evaluation

See [EVAL.md](docs/EVAL.md) for details about evaluating MAOmni on benchmarks.
