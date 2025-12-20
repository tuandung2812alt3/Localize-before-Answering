# **LOBA**, Localizing Before Answering: A Benchmark for Grounded Medical Visual Question Answering

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
- Please obtain your usage permission and download the MIMIC-CXR dataset here
```
https://physionet.org/content/mimic-cxr/2.1.0/
```


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


### SAM ViT-H
Download:
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---
## Training

CUDA_VISIBLE_DEVICES=0 python train_ds.py     --vision_pretrained /path/to/sam_vit_h_4b8939.pth   --data_root ./dataset/healmed_vqa   --model_dir ./model



## Inference

The inference function with attention modification is done via `infer.py`.

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

