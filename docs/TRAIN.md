# Training MAOmni

## 🛠️ Environment Setup

There are two environments to set up, one for GRPO Grounder Agent, one for other agents.

### Install the GRPO Grounder environment

1. Clone the repository from GitHub.

```shell
git clone git@github.com:eeee-sys/MAOmni.git
cd MAOmni
```

2. Initialize conda environment for GRPO Grounder.

```shell
conda create -n grpo_grounder python=3.11 -y
conda activate grpo_grounder
```

3. Install dependencies.

```shell
pip install -r src/requirements_grpo_grounder.txt
```

### Install the main environment

1. Initialize conda environment for main.

```shell
conda create -n maomni_main python=3.11 -y
conda activate maomni_main
```

2. Install dependencies.

```shell
pip install -r src/requirements_main.txt
```

### Prepare base models

Download [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) and [VideoMind-7B](https://huggingface.co/yeliudev/VideoMind-7B), then place them into the `model_zoo` folder.

```
MAOmni
└─ model_zoo
   ├─ Qwen2.5-Omni-7B
   ├─ VideoMind-7B
   └─ Qwen2-VL-7B-Instruct
```

## 📦 Dataset Preparation

The training data used for each role is listed as follows. All the data, including the raw videos, compressed videos, and annotations, could be downloaded on [Hugging Face]

### AKD Router Agent
We select 755 samples from [Hugging Face](https://huggingface.co/datasets/PhilipC/IntentTrain/tree/main), selected samples can be seen in [data_config](/src/open-r1-multimodal/data_config).

### GRPO Grounder Agent

