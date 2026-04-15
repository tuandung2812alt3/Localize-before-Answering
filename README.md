# **LOBA** — Localizing Before Answering: A Benchmark for Grounded Medical Visual Question Answering

This repository contains the official code for **LOBA**, accepted at **IJCAI 2025**.

> **Paper:** [Localizing Before Answering: A Benchmark for Grounded Medical Visual Question Answering](https://arxiv.org/abs/2505.00744)

LOBA introduces a grounded medical VQA framework that first localizes relevant regions in a medical image before generating a textual answer. The benchmark is built on MIMIC-CXR and provides paired segmentation masks with clinical questions and answers.

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

### Data Annotation
Download the annotation here. The HuggingFace link includes image IDs, question and answer, and the segmentation mask of the Area of Interest:
https://huggingface.co/datasets/tuandung2812/Heal-MedVQA

### Download MIMIC-CXR
Please obtain usage permission and download the MIMIC-CXR dataset from PhysioNet:
```
https://physionet.org/content/mimic-cxr/2.1.0/
```

### Run preparation script
```bash
python prepare_healmed_vqa.py \
  --input_root /path/to/raw/MIMIC-CXR \
  --output_root ./dataset/MIMIC-CXR
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

### SAM ViT-H
Download the SAM ViT-H checkpoint from:
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_ds.py \
  --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
  --data_root ./dataset/healmed_vqa \
  --model_dir ./model
```

---

## Inference

Inference with attention modification is handled by `infer.py`.

### Basic command
```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model_path /path/to/loba_model \
  --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
  --data_root ./dataset/healmed_vqa \
  --output_dir ./outputs
```

---

## Outputs

Running inference produces:
- Segmentation masks
- Textual answers
- A JSON file with serialized predictions

```
outputs/
├── predictions.json
└── masks/
    ├── sample_000.png
    └── ...
```

---

