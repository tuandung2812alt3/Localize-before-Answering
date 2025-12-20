#!/usr/bin/env python3
"""
prepare_data_loba.py

Prepare HEAL-MedVQA-style data for LobA:
1) Load images
2) Load anatomy segmentation masks (per anatomy) (PNG preferred)
3) Load disease bounding boxes (VinDr-like) + optional disease presence filter from reports
4) Map disease -> anatomy via IoU_dis = area(box ∩ anatomy_mask) / area(box), threshold δ (default 0.5)
5) Generate VQA pairs:
   - Closed-ended (Positive): Does {anatomy} have {disease}? -> Yes
   - Closed-ended (Hallucinated): Does {anatomy} have {disease}? -> No
   - Open-ended (Abnormal): List abnormalities in {anatomy}. -> list diseases
   - Open-ended (Normal): Are there any diseases in {anatomy}? -> no abnormalities / no finding

This follows the paper's pipeline and mapping rule (IoU_dis threshold) and question types. :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


# -------------------------
# Config / Defaults
# -------------------------

DEFAULT_ANATOMIES = [
    "right_upper_lung",
    "right_middle_lung",
    "right_lower_lung",
    "left_upper_lung",
    "left_middle_lung",
    "left_lower_lung",
    "heart",
    "aorta",
]

# You can override this with --disease_list_json
DEFAULT_DISEASES = [
    "aortic enlargement",
    "atelectasis",
    "calcification",
    "cardiomegaly",
    "consolidation",
    "interstitial lung disease",
    "infiltration",
    "lung opacification",
    "nodule/mass",
    "other lesion",
    "pleural effusion",
    "pleural thickening",
    "pneumothorax",
    "pulmonary fibrosis",
    "no finding",
]


# -------------------------
# Helpers
# -------------------------

def read_image_size(img_path: str) -> Tuple[int, int]:
    with Image.open(img_path) as im:
        w, h = im.size
    return w, h


def load_mask_png(mask_path: str) -> np.ndarray:
    """
    Load a binary mask from a PNG.
    Nonzero pixels are treated as 1.
    """
    m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return (m > 0).astype(np.uint8)


def clamp_box(x1, y1, x2, y2, w, h) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w)))
    y2 = int(max(0, min(y2, h)))
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def iou_dis(box_xyxy: Tuple[int, int, int, int], anatomy_mask: np.ndarray) -> float:
    """
    IoU_dis as used in the paper:
    intersection_over_disease_area = area( box ∩ anatomy_mask ) / area(box)
    (paper uses intersection over disease area instead of union). :contentReference[oaicite:2]{index=2}
    """
    x1, y1, x2, y2 = box_xyxy
    crop = anatomy_mask[y1:y2, x1:x2]
    inter = int(crop.sum())
    area_box = int((x2 - x1) * (y2 - y1))
    if area_box <= 0:
        return 0.0
    return float(inter) / float(area_box)


def norm_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


@dataclass
class Box:
    disease: str
    x1: int
    y1: int
    x2: int
    y2: int
    score: Optional[float] = None


# -------------------------
# Loading inputs
# -------------------------

def load_boxes_csv(
    csv_path: str,
    image_id_col: str,
    disease_col: str,
    x1_col: str,
    y1_col: str,
    x2_col: str,
    y2_col: str,
    score_col: Optional[str] = None,
) -> Dict[str, List[Box]]:
    """
    Expected: one row per box.
    Returns: dict image_id -> list[Box]
    """
    df = pd.read_csv(csv_path)
    out: Dict[str, List[Box]] = {}
    for _, r in df.iterrows():
        img_id = str(r[image_id_col])
        disease = norm_text(r[disease_col])
        b = Box(
            disease=disease,
            x1=int(r[x1_col]),
            y1=int(r[y1_col]),
            x2=int(r[x2_col]),
            y2=int(r[y2_col]),
            score=float(r[score_col]) if score_col and score_col in df.columns else None
        )
        out.setdefault(img_id, []).append(b)
    return out


def load_presence_json(p: str) -> Dict[str, List[str]]:
    """
    Optional: image_id -> list of diseases present (from report labels).
    Used to filter false positives in disease boxes (paper filters boxes using report-derived labels). :contentReference[oaicite:3]{index=3}
    """
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # allow either dict[str, list[str]] or list[{"image_id":..., "labels":[...]}]
    if isinstance(data, dict):
        return {str(k): [norm_text(x) for x in v] for k, v in data.items()}
    if isinstance(data, list):
        out = {}
        for row in data:
            out[str(row["image_id"])] = [norm_text(x) for x in row.get("labels", [])]
        return out
    raise ValueError("Unsupported presence json format.")


def load_disease_list_json(p: str) -> List[str]:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("disease_list_json must be a JSON list[str].")
    return [norm_text(x) for x in data]


# -------------------------
# Mapping: disease -> anatomy
# -------------------------

def map_disease_to_anatomy(
    img_path: str,
    img_id: str,
    boxes: List[Box],
    anatomy_mask_dir: str,
    anatomy_list: List[str],
    delta: float,
    anatomy_mask_ext: str,
) -> Dict[str, List[str]]:
    """
    For each disease box, find anatomy regions it overlaps with (IoU_dis >= delta).
    Returns: anatomy -> list of diseases that occur in that anatomy.
    """
    w, h = read_image_size(img_path)

    # Preload anatomy masks
    masks: Dict[str, np.ndarray] = {}
    for a in anatomy_list:
        mp = os.path.join(anatomy_mask_dir, a, f"{img_id}{anatomy_mask_ext}")
        if not os.path.exists(mp):
            # allow flat: anatomy_mask_dir/{img_id}_{anatomy}.png
            alt = os.path.join(anatomy_mask_dir, f"{img_id}_{a}{anatomy_mask_ext}")
            if os.path.exists(alt):
                mp = alt
            else:
                continue
        masks[a] = load_mask_png(mp)

    anatomy_to_diseases: Dict[str, List[str]] = {a: [] for a in anatomy_list}

    for b in boxes:
        x1, y1, x2, y2 = clamp_box(b.x1, b.y1, b.x2, b.y2, w, h)
        for a, m in masks.items():
            if m.shape[0] != h or m.shape[1] != w:
                # If mask resolution differs, resize nearest
                m_resized = np.array(Image.fromarray(m * 255).resize((w, h), resample=Image.NEAREST))
                m_bin = (m_resized > 0).astype(np.uint8)
            else:
                m_bin = m
            score = iou_dis((x1, y1, x2, y2), m_bin)
            if score >= delta:
                anatomy_to_diseases[a].append(b.disease)

    # Deduplicate
    for a in anatomy_to_diseases:
        anatomy_to_diseases[a] = sorted(list(set(anatomy_to_diseases[a])))

    return anatomy_to_diseases


# -------------------------
# QA Generation
# -------------------------

def gen_closed_question(anatomy: str, disease: str) -> str:
    # Keep template stable and easy to parse.
    return f"Does the patient have {disease} in the {anatomy}?"


def gen_open_abnormal_question(anatomy: str) -> str:
    return f"List all abnormalities in the {anatomy}."


def gen_open_normal_question(anatomy: str) -> str:
    return f"Are there any diseases in the {anatomy}?"


def answer_closed(is_positive: bool) -> str:
    return "Yes" if is_positive else "No"


def answer_open_list(diseases: List[str]) -> str:
    if not diseases:
        return "It has no abnormalities."
    # comma separated list
    return ", ".join(diseases)


def sample_hallucinated_disease(
    present: List[str],
    disease_pool: List[str],
    rng: random.Random,
) -> Optional[str]:
    candidates = [d for d in disease_pool if d not in present and d != "no finding"]
    if not candidates:
        return None
    return rng.choice(candidates)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True,
                    help="Directory containing images. Filenames should include image_id (e.g., {image_id}.jpg/png).")
    ap.add_argument("--image_exts", type=str, default=".png,.jpg,.jpeg",
                    help="Comma-separated allowed image extensions.")
    ap.add_argument("--anatomy_mask_dir", type=str, required=True,
                    help="Directory containing anatomy masks. Supports anatomy subfolders or flat naming.")
    ap.add_argument("--anatomy_mask_ext", type=str, default=".png",
                    help="Mask file extension (default .png).")
    ap.add_argument("--boxes_csv", type=str, required=True,
                    help="CSV with disease bounding boxes.")
    ap.add_argument("--image_id_col", type=str, default="image_id")
    ap.add_argument("--disease_col", type=str, default="disease")
    ap.add_argument("--x1_col", type=str, default="x1")
    ap.add_argument("--y1_col", type=str, default="y1")
    ap.add_argument("--x2_col", type=str, default="x2")
    ap.add_argument("--y2_col", type=str, default="y2")
    ap.add_argument("--score_col", type=str, default="score",
                    help="Optional score column name. If missing, ignored.")
    ap.add_argument("--presence_json", type=str, default=None,
                    help="Optional JSON image_id -> list of diseases present (from reports) to filter false positives.")
    ap.add_argument("--disease_list_json", type=str, default=None,
                    help="Optional JSON list[str] overriding disease pool for hallucinated sampling.")
    ap.add_argument("--anatomy_list_json", type=str, default=None,
                    help="Optional JSON list[str] overriding default anatomies.")
    ap.add_argument("--delta", type=float, default=0.5,
                    help="IoU_dis threshold δ for mapping disease->anatomy (default 0.5). :contentReference[oaicite:4]{index=4}")
    ap.add_argument("--min_qa_per_image", type=int, default=2,
                    help="Randomly generate between [min,max] QA per image (default 2). :contentReference[oaicite:5]{index=5}")
    ap.add_argument("--max_qa_per_image", type=int, default=5,
                    help="Randomly generate between [min,max] QA per image (default 5). :contentReference[oaicite:6]{index=6}")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--split", type=str, default="train",
                    help="Split name to write into samples (train/val/test).")
    ap.add_argument("--source", type=str, default="healmedvqa",
                    help="Source dataset name tag.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    anatomy_list = DEFAULT_ANATOMIES
    if args.anatomy_list_json:
        anatomy_list = [norm_text(x) for x in json.load(open(args.anatomy_list_json, "r", encoding="utf-8"))]

    disease_pool = [norm_text(x) for x in DEFAULT_DISEASES]
    if args.disease_list_json:
        disease_pool = load_disease_list_json(args.disease_list_json)

    boxes_by_id = load_boxes_csv(
        args.boxes_csv,
        image_id_col=args.image_id_col,
        disease_col=args.disease_col,
        x1_col=args.x1_col,
        y1_col=args.y1_col,
        x2_col=args.x2_col,
        y2_col=args.y2_col,
        score_col=args.score_col if args.score_col else None,
    )

    presence = None
    if args.presence_json:
        presence = load_presence_json(args.presence_json)

    # discover image files
    exts = [e.strip().lower() for e in args.image_exts.split(",") if e.strip()]
    img_index: Dict[str, str] = {}
    for fn in os.listdir(args.images_dir):
        low = fn.lower()
        if any(low.endswith(e) for e in exts):
            img_id = os.path.splitext(fn)[0]
            img_index[img_id] = os.path.join(args.images_dir, fn)

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    n_written = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for img_id, img_path in img_index.items():
            boxes = boxes_by_id.get(img_id, [])
            if not boxes:
                continue

            # Optional: filter boxes using report-derived presence list (reduces FP boxes). :contentReference[oaicite:7]{index=7}
            if presence is not None and img_id in presence:
                present_set = set(presence[img_id])
                boxes = [b for b in boxes if b.disease in present_set]

            if not boxes:
                continue

            anatomy_to_diseases = map_disease_to_anatomy(
                img_path=img_path,
                img_id=img_id,
                boxes=boxes,
                anatomy_mask_dir=args.anatomy_mask_dir,
                anatomy_list=anatomy_list,
                delta=args.delta,
                anatomy_mask_ext=args.anatomy_mask_ext,
            )

            # Build candidate anatomies
            anatomies_with_any = [a for a, ds in anatomy_to_diseases.items() if len(ds) > 0]
            anatomies_all = anatomy_list[:]

            # Generate 2-5 QA pairs per image (paper mentions 2–5). :contentReference[oaicite:8]{index=8}
            k = rng.randint(args.min_qa_per_image, args.max_qa_per_image)

            samples = []
            for _ in range(k):
                qtype = rng.choice(["closed_pos", "closed_neg", "open_abn", "open_norm"])

                if qtype == "closed_pos" and anatomies_with_any:
                    a = rng.choice(anatomies_with_any)
                    d = rng.choice(anatomy_to_diseases[a])
                    q = gen_closed_question(a, d)
                    ans = answer_closed(True)
                    meta = {"anatomy": a, "disease": d}

                elif qtype == "closed_neg":
                    a = rng.choice(anatomies_all)
                    present_here = anatomy_to_diseases.get(a, [])
                    d = sample_hallucinated_disease(present_here, disease_pool, rng)
                    if d is None:
                        continue
                    q = gen_closed_question(a, d)
                    ans = answer_closed(False)
                    meta = {"anatomy": a, "disease": d}

                elif qtype == "open_abn" and anatomies_with_any:
                    a = rng.choice(anatomies_with_any)
                    ds = anatomy_to_diseases[a]
                    q = gen_open_abnormal_question(a)
                    ans = answer_open_list(ds)
                    meta = {"anatomy": a, "diseases": ds}

                else:
                    # open_norm (or fallback)
                    a = rng.choice(anatomies_all)
                    ds = anatomy_to_diseases.get(a, [])
                    q = gen_open_normal_question(a)
                    ans = "It has no abnormalities." if len(ds) == 0 else answer_open_list(ds)
                    meta = {"anatomy": a, "diseases": ds}

                # optional: attach anatomy mask path if exists
                mask_path = os.path.join(args.anatomy_mask_dir, a, f"{img_id}{args.anatomy_mask_ext}")
                if not os.path.exists(mask_path):
                    alt = os.path.join(args.anatomy_mask_dir, f"{img_id}_{a}{args.anatomy_mask_ext}")
                    mask_path = alt if os.path.exists(alt) else None

                record = {
                    "id": f"{img_id}_{len(samples):03d}",
                    "image_id": img_id,
                    "image_path": img_path,
                    "question": q,
                    "answer": ans,
                    "question_type": qtype,
                    "split": args.split,
                    "source": args.source,
                    "anatomy_mask_path": mask_path,
                    "meta": meta,
                }
                samples.append(record)

            for r in samples:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"[OK] Wrote {n_written} samples to {args.out_jsonl}")


if __name__ == "__main__":
    main()
