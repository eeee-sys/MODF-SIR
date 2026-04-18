# Evaluating MODF-SIR

## 🛠️ Environment Setup

Please refer to [TRAIN.md](/docs/TRAIN.md) for setting up the environment.

## 📚 Checkpoint Preparation

Download the [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), [VideoMind-7B](https://huggingface.co/yeliudev/VideoMind-7B) and [MODF-SIR LoRA](https://huggingface.co/Harry-1234/MAOmni), and place them into the `model_zoo` folder.

```
MODF-SIR
└─ model_zoo
   ├─ Qwen2.5-Omni-7B
   ├─ Qwen2-VL-7B-Instruct
   ├─ VideoMind-7B
   ├─ Planner
   └─ GRPO_Grounder
```

## 📦 Dataset Preparation

The desired benchmarks are as following:
- [Daily-Omni](https://huggingface.co/datasets/liarliar/Daily-Omni)
- [WorldSense](https://huggingface.co/datasets/honglyhly/WorldSense)
- [IntentBench](https://huggingface.co/datasets/PhilipC/IntentBench)

## 🔮 Start Evaluation
Use the following commands to evaluate MODF-SIR on different benchmarks. Default is evaluating IntentBench, you may change files path to evaluate on different benchmarks.

```shell
bash src/open-r1-multimodal/eval/run_eval.sh
```

After evaluation, you may use codes from [eval_results](src/open-r1-multimodal/eval_results) to get accuracy on each benchmark. Please modify the files path if necessary.
Parameters for each benchmark can be seen from the supplementary material in our paper.
