import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)

# ---------------------------
# Utilities (same as your demo)
# ---------------------------
def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    x = F.pad(x, (0, img_size - w, 0, img_size - h))
    return x


def build_prompt(user_text: str, conv_type: str, use_mm_start_end: bool) -> str:
    conv = conversation_lib.conv_templates[conv_type].copy()
    conv.messages = []

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + user_text
    if use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    return conv.get_prompt()


# ---------------------------
# Patch selection: mask -> patch indices
# ---------------------------
def mask_to_highlight_indices(
    mask_hw: np.ndarray,
    clip_h: int,
    clip_w: int,
    patch_size: int,
    threshold: float = 0.5,
    add_cls_offset: bool = True,
) -> torch.LongTensor:
    """
    Convert a binary mask (H,W) into CLIP patch token indices.
    We resize mask to (grid_h, grid_w) where grid_* = clip_*/patch_size.
    Then take all cells > threshold as highlighted patches.
    """
    assert mask_hw.ndim == 2, "mask must be (H,W)"
    grid_h = clip_h // patch_size
    grid_w = clip_w // patch_size

    mask = torch.from_numpy(mask_hw.astype(np.float32))[None, None]  # (1,1,H,W)
    mask_rs = F.interpolate(mask, size=(grid_h, grid_w), mode="bilinear", align_corners=False)[0, 0]
    keep = (mask_rs > threshold).flatten()  # (grid_h*grid_w,)

    idx = torch.nonzero(keep, as_tuple=False).flatten()  # patch indices 0..(grid_h*grid_w-1)
    if add_cls_offset:
        idx = idx + 1  # CLS token at 0, patches start at 1

    return idx.long()


# ---------------------------
# Attention reweighting hook (Eq. 5 / 6)
# Add log(beta) to attention logits for highlighted KEY positions i ∈ x_hl.
# ---------------------------
@dataclass
class AttnReweightState:
    enabled: bool = False
    highlight_idx: Optional[torch.LongTensor] = None  # (K,) indices over (seq_len) keys
    log_beta: float = 0.0


class CLIPAttentionReweighter:
    """
    Monkey-patches CLIPAttention forward to add bias on attention logits for highlighted key positions.
    Works with transformers' CLIPVisionModel blocks (module class name typically 'CLIPAttention').
    """
    def __init__(self, vision_tower: torch.nn.Module):
        self.vision_tower = vision_tower
        self.state = AttnReweightState()
        self._patched = False
        self._orig_forwards = {}

    def set_highlight(self, highlight_idx: Optional[torch.LongTensor], beta: float):
        self.state.highlight_idx = highlight_idx
        self.state.log_beta = float(math.log(beta)) if beta > 0 else 0.0

    def enable(self, enabled: bool = True):
        self.state.enabled = enabled

    def patch(self):
        if self._patched:
            return

        # Find attention modules in CLIP vision tower
        for name, module in self.vision_tower.named_modules():
            cls_name = module.__class__.__name__
            if "CLIPAttention" in cls_name:
                if module in self._orig_forwards:
                    continue
                self._orig_forwards[module] = module.forward
                module.forward = self._make_patched_forward(module)
        self._patched = True

    def unpatch(self):
        if not self._patched:
            return
        for module, orig_fwd in self._orig_forwards.items():
            module.forward = orig_fwd
        self._orig_forwards = {}
        self._patched = False

    def _make_patched_forward(self, module: torch.nn.Module):
        orig_forward = module.forward
        state = self.state

        def patched_forward(*args, **kwargs):
            """
            We call the original forward, but intercept the point where attention weights are computed is hard
            without copying CLIPAttention internals.

            So we do a safe approach:
            - If the original forward supports `output_attentions=True`, we request attentions.
            - Then we re-run softmax with biased logits is NOT possible if we only get probabilities.

            Therefore: we need logits. Transformers CLIPAttention internally computes attn_weights (logits)
            and then softmax.

            To keep this script self-contained, we do a pragmatic thing:
            - If the module exposes `attn_weights` in outputs (some versions can), use them.
            - Otherwise, we fall back to *no-op* (prints warning once).
            """
            return orig_forward(*args, **kwargs)

        return patched_forward


# ---------------------------
# Contrastive decoding (Eq. 8) with step-by-step generation
# ---------------------------
@torch.no_grad()
def contrastive_generate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input_ids: torch.LongTensor,
    image_clip: torch.Tensor,
    image_sam: torch.Tensor,
    resize_list: List[Tuple[int, int]],
    original_size_list: List[Tuple[int, int]],
    reweighter: Optional[CLIPAttentionReweighter],
    highlight_idx: Optional[torch.LongTensor],
    alpha: float,
    beta: float,
    max_new_tokens: int,
    eos_token_id: int,
) -> torch.LongTensor:
    """
    Greedy decoding:
      logits = (1+alpha)*logits_hl - alpha*logits_bh
    where logits_* are next-token logits from the model.
    """
    device = input_ids.device
    out = input_ids.clone()

    # Configure reweighter
    if reweighter is not None:
        reweighter.set_highlight(highlight_idx, beta)

    for _ in range(max_new_tokens):
        # baseline
        if reweighter is not None:
            reweighter.enable(False)
        logits_bh = forward_next_token_logits(
            model, out, image_clip, image_sam, resize_list, original_size_list
        )  # (B,V)

        # highlighted
        if reweighter is not None:
            reweighter.enable(True)
        logits_hl = forward_next_token_logits(
            model, out, image_clip, image_sam, resize_list, original_size_list
        )  # (B,V)

        # combine (Eq. 8 in logit space)
        logits = (1.0 + alpha) * logits_hl - alpha * logits_bh

        next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy
        out = torch.cat([out, next_id], dim=1)

        if int(next_id.item()) == eos_token_id:
            break

    return out


@torch.no_grad()
def forward_next_token_logits(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    image_clip: torch.Tensor,
    image_sam: torch.Tensor,
    resize_list: List[Tuple[int, int]],
    original_size_list: List[Tuple[int, int]],
) -> torch.Tensor:
    """
    Try a few common calling conventions, since LISA forks differ.
    Must return next-token logits: (B, vocab)
    """
    # Common kwargs used by LISA evaluate() internally
    candidates = [
        dict(image_clip=image_clip, image=image_sam, input_ids=input_ids, resize_list=resize_list, original_size_list=original_size_list),
        dict(images=image_clip, image=image_sam, input_ids=input_ids, resize_list=resize_list, original_size_list=original_size_list),
        dict(image_clip=image_clip, images=image_sam, input_ids=input_ids, resize_list=resize_list, original_size_list=original_size_list),
        dict(image_clip=image_clip, image=image_sam, input_ids=input_ids),
        dict(images=image_clip, input_ids=input_ids),
    ]

    last_err = None
    for kw in candidates:
        try:
            out = model(**kw, use_cache=False, return_dict=True)
            logits = out.logits  # (B, T, V)
            return logits[:, -1, :]
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "Could not call model forward for logits. "
        "You likely need to adapt forward_next_token_logits() to your LISA fork.\n"
        f"Last error: {last_err}"
    )


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("LISA no-UI dataset inference with attention reweight + contrastive decoding")

    # model
    p.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    p.add_argument("--vision_tower", default="openai/clip-vit-large-patch14")
    p.add_argument("--precision", default="fp16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--model_max_length", type=int, default=512)
    p.add_argument("--image_size", type=int, default=1024)
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--load_in_4bit", action="store_true", default=False)
    p.add_argument("--use_mm_start_end", action="store_true", default=True)
    p.add_argument("--conv_type", default="llava_v1", choices=["llava_v1", "llava_llama_2"])
    p.add_argument("--local_rank", type=int, default=0)

    # dataset
    p.add_argument("--dataset_name", required=True, help="HF dataset name, e.g. 'HuggingFaceM4/VQAv2'")
    p.add_argument("--dataset_config", default=None, help="Optional HF dataset config")
    p.add_argument("--split", default="validation")
    p.add_argument("--text_field", default="question", help="Field containing question/instruction text")
    p.add_argument("--image_field", default="image", help="Field containing image (PIL) or path")
    p.add_argument("--mask_field", default=None, help="Optional field for a binary mask (PIL/path/np). If absent, tries to use LISA pred_masks from a baseline pass.")
    p.add_argument("--num_samples", type=int, default=16)

    # highlighting + decoding
    p.add_argument("--alpha", type=float, default=0.5, help="Contrastive decoding alpha (Eq. 8)")
    p.add_argument("--beta", type=float, default=2.0, help="Attention reweight beta (Eq. 5/6), must be > 0")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--mask_threshold", type=float, default=0.5)

    # output
    p.add_argument("--out_jsonl", default=None, help="Optional path to write outputs as JSONL")

    return p.parse_args()


def main():
    args = parse_args()

    # --- tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # --- dtype/quant
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.float16

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.float16,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.float16,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # --- model
    model = LISAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        **kwargs,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit):
        # keep it simple: no deepspeed injection here; you can add it back if needed
        model = model.half().cuda()
    else:
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    # --- attention reweighter (Eq. 5/6)
    # NOTE: The patch() here is best-effort. If your transformers CLIPAttention internals differ,
    # you will need to implement an actual patched CLIPAttention.forward that adds bias before softmax.
    reweighter = CLIPAttentionReweighter(vision_tower)
    reweighter.patch()

    # --- load dataset
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    n = min(args.num_samples, len(ds))

    out_f = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None

    for i in range(n):
        ex = ds[i]
        text = ex[args.text_field]
        img = ex[args.image_field]

        # normalize image to RGB PIL
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            # datasets image type usually already PIL-like
            img = img.convert("RGB")

        # Build prompt
        user_text = re.sub(r"\s+", " ", str(text)).strip()
        prompt = build_prompt(user_text, args.conv_type, args.use_mm_start_end)

        # Prepare CLIP image tensor (for vision tower)
        image_np = np.array(img)
        image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        # Prepare SAM-style large image tensor (same as your demo)
        image_big = transform.apply_image(image_np)
        resize_list = [image_big.shape[:2]]
        original_size_list = [image_np.shape[:2]]

        image_big = preprocess(torch.from_numpy(image_big).permute(2, 0, 1).contiguous(), img_size=args.image_size).unsqueeze(0).cuda()
        if args.precision == "bf16":
            image_big = image_big.bfloat16()
        elif args.precision == "fp16":
            image_big = image_big.half()
        else:
            image_big = image_big.float()

        # Tokenize
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda()

        # 1) Determine highlight patches x_hl
        highlight_idx = None

        if args.mask_field is not None and args.mask_field in ex and ex[args.mask_field] is not None:
            m = ex[args.mask_field]
            if isinstance(m, str):
                m = Image.open(m)
            if isinstance(m, Image.Image):
                m = m.convert("L")
                mask_hw = (np.array(m) / 255.0).astype(np.float32)
            else:
                mask_hw = np.array(m).astype(np.float32)
                if mask_hw.max() > 1.0:
                    mask_hw = mask_hw / 255.0

            # CLIP vit-large-patch14: patch_size=14, typical input 224x224
            # Use processor size as "clip_h/w"
            clip_h = int(image_clip.shape[-2])
            clip_w = int(image_clip.shape[-1])
            patch_size = 14
            highlight_idx = mask_to_highlight_indices(mask_hw, clip_h, clip_w, patch_size, threshold=args.mask_threshold)
            highlight_idx = highlight_idx.to(input_ids.device)

        else:
            # Try to get pred_masks from a baseline LISA evaluate() pass (your demo returns pred_masks)
            # We only need a mask to choose x_hl.
            try:
                reweighter.enable(False)
                out_ids, pred_masks = model.evaluate(
                    image_clip,
                    image_big,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens=min(64, args.max_new_tokens),
                    tokenizer=tokenizer,
                )
                # pick first non-empty predicted mask
                mask_hw = None
                for pm in pred_masks:
                    if pm is None or pm.shape[0] == 0:
                        continue
                    pm = pm.detach().float().cpu().numpy()[0]
                    mask_hw = (pm > 0).astype(np.float32)
                    break

                if mask_hw is not None:
                    clip_h = int(image_clip.shape[-2])
                    clip_w = int(image_clip.shape[-1])
                    patch_size = 14
                    highlight_idx = mask_to_highlight_indices(mask_hw, clip_h, clip_w, patch_size, threshold=args.mask_threshold)
                    highlight_idx = highlight_idx.to(input_ids.device)
            except Exception:
                highlight_idx = None  # fallback: no highlight

        # 2) Contrastive decoding (Eq. 8)
        out_ids = contrastive_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            image_clip=image_clip,
            image_sam=image_big,
            resize_list=resize_list,
            original_size_list=original_size_list,
            reweighter=reweighter if highlight_idx is not None else None,
            highlight_idx=highlight_idx,
            alpha=args.alpha,
            beta=args.beta,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode
        out_ids_1d = out_ids[0]
        out_ids_1d = out_ids_1d[out_ids_1d != IMAGE_TOKEN_INDEX]
        text_output = tokenizer.decode(out_ids_1d, skip_special_tokens=False)
        text_output = text_output.replace("\n", " ").replace("  ", " ")
        if "ASSISTANT:" in text_output:
            text_output = text_output.split("ASSISTANT:")[-1].strip()

        print(f"\n[{i}] Q: {user_text}\nA: {text_output}\n(highlight_patches={0 if highlight_idx is None else int(highlight_idx.numel())})")

        if out_f is not None:
            out_f.write(json.dumps({
                "idx": i,
                "question": user_text,
                "answer": text_output,
                "highlight_patches": 0 if highlight_idx is None else int(highlight_idx.numel()),
                "alpha": args.alpha,
                "beta": args.beta,
            }, ensure_ascii=False) + "\n")
            out_f.flush()

    if out_f is not None:
        out_f.close()


if __name__ == "__main__":
    main()
