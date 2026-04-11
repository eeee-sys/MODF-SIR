<h2 align="center">MAOmni: A Self-Correcting Multi-Agent Omni-Modal Reasoning Framework For Affective and Intentional Analysis</h2>

<p align="center">
  <a href="https://huggingface.co/Harry-1234/MAOmni" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
  <a href="https://huggingface.co/datasets/Harry-1234/IntentRouterTrain/" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange"></a>
  <a href="https://huggingface.co/spaces/Harry-1234/MAOmni" target="_blank"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg"></a>
</p>

**MAOmni** is a novel self-correcting multi-agent omni-modal framework endowed with deliberative reasoning capabilities. MAOmni decomposes the reasoning process through a dynamic cognitive workflow orchestrated by five specialized agents, a generative Retriever for global context distillation, an adaptive AKD Router Agent for dynamic reasoning routing, a GRPO Grounder for precise continuous-time spatio-temporal localization, Reasoning Agent for explicit structured logical inference, and a TTA Reviser for test-time adaptive self-correction via ephemeral LoRA tuning.

<p align="center"><img width="100%" height="100%" src="assets/method.png"></p>

## 🔥 News
- 🚀 MAOmni is ready on [Hugging Face Model](https://huggingface.co/Harry-1234/MAOmni). Check it out!
- 📦 Training Data is ready on [Hugging Face Dataset](https://huggingface.co/datasets/Harry-1234/IntentRouterTrain/). Start it!
- 🕹️ Online demo is ready [Hugging Face Space](https://huggingface.co/spaces/Harry-1234/MAOmni). Play with it!
- ⭐️ Code, model, dataset release.

## 🏆 MAOmni on Public Benchmarks
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

Online demo is ready [Hugging Face Space](https://huggingface.co/spaces/Harry-1234/MAOmni). Play with it!


https://github.com/user-attachments/assets/60cf2207-d49f-4bea-8713-83d36e9e7c39



## 🚀 Training
Our codebase supports training and evaluating on [10 video datasets and benchmarks] with the following features.

- Hardware settings: NVIDIA GPU A100/H100, Single-Node / Multi-Node
- Efficient training techniques: DeepSpeed ZeRO, BF16, LoRA, SDPA, FlashAttention2
- Customizing the base LLM and conversation templates
- Monitoring the training process via Tensorboard / Wandb
- Group sampling for mixed dataset training
- Multi-process / multi-device evaluation on public benchmarks

See [TRAIN.md](docs/TRAIN.md) for a quick start guide.

## 🔮 Evaluation

See [EVAL.md](docs/EVAL.md) for details about evaluating MAOmni on benchmarks.
