# LOBA: Language-Optimized Boundary Attention for Medical VQA Segmentation

This repository contains the code for **LOBA**, Localizing Before Answering: A Benchmark for Grounded Medical Visual Question Answering from IJCAI 2025



---

## Environment Setup

### 1. Create environment
```bash
conda create -n loba python=3.10 -y
conda activate loba
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

---

## Dataset Preparation (HealMed-VQA)

The HealMed-VQA dataset is prepared using the provided script.

### Prepare dataset
```bash
python prepare_healmed_vqa.py   --input_root /path/to/raw/MIMIC-CXR   --output_root ./dataset/MIMIC-CXR
```

### Expected structure
```
dataset/
└── healmed_vqa/
    ├── images/
    │   ├── xxx.jpg
    │   └── ...
    └── annotations.json
```

---

## Model Weights

You need the following components:
1. A LLaVA-compatible vision–language backbone
2. A SAM ViT-H checkpoint
3. A LOBA fine-tuned model checkpoint

### SAM ViT-H
Download:
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## Inference

All inference is done via `infer.py`.

### Basic command
```bash
CUDA_VISIBLE_DEVICES=0 python infer.py   --model_path /path/to/loba_model   --vision_pretrained /path/to/sam_vit_h_4b8939.pth   --data_root ./dataset/healmed_vqa   --output_dir ./outputs
```

### Optional flags
```bash
--precision bf16
--load_in_8bit
--load_in_4bit
--max_samples 100
```

---

## Outputs

Inference produces:
- Segmentation masks
- Textual answers
- A JSON file with serialized predictions

```
outputs/
├── predictions.json
├── masks/
│   ├── sample_000.png
│   └── ...
```

---

