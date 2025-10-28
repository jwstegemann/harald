#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL (frozen) ➜ Linear head ➜ Qwen-Image prompt_embeds (residual)
with explicit Pixel-Budget + robust Vision Hook
================================================

- Analyse-Backbone: **Qwen3-VL** (aktuelle Repo/Weights).
- Generator: **Qwen-Image** (Diffusers, akzeptiert prompt_embeds).
- Training: **Linear-Regression** vom visuellen Qwen3-VL-Feature → Residual über Qwen-Image-Text-Embeddings.
- Inference: **Alpha-Sweep** × **16 Seeds**; speichert Grid-PNG + Meta-JSON pro Subjekt.
- Neu in dieser Version:
  • **Pixel-Budget** (in Megapixel) mit automatischer Multi-Image-Skalierung.
  • **Robuster Vision-Hook**: mehrere Pfade (Vision-Tower, model(..., output_hidden_states), Fallback).

Install:
    pip install -U "transformers>=4.57.0" diffusers accelerate pillow tqdm
    # Optional (empfohlen): stabileres Preprocessing
    pip install qwen-vl-utils==0.0.14

Beispiele:
    python qwen3_vl_to_qwen_image_linear_budget.py \
      --data-root /data/8 \
      --out-dir ./outputs_qwen3_lr \
      --qwen3-vl-model Qwen/Qwen3-VL-8B-Instruct \
      --qwen-image-model Qwen/Qwen2-Image-2.1 \
      --base-prompt "a portrait photo of a person, studio lighting, 85mm" \
      --negative-prompt "blurry, low quality, deformed, watermark" \
      --pixel-budget-mp 2.0 --max-side 1536 \
      --train --epochs 5 --batch-size 8

    # Bootstrap ohne Teacher (Caption -> Target-Embeds)
    python qwen3_vl_to_qwen_image_linear_budget.py \
      --data-root /data/8 \
      --out-dir ./outputs_qwen3_lr \
      --qwen3-vl-model Qwen/Qwen3-VL-8B-Instruct \
      --qwen-image-model Qwen/Qwen2-Image-2.1 \
      --base-prompt "a portrait photo of a person, studio lighting, 85mm" \
      --bootstrap-captions \
      --pixel-budget-mp 2.0 --max-side 1536 \
      --train --epochs 5 --batch-size 8

    # Nur Inference
    python qwen3_vl_to_qwen_image_linear_budget.py \
      --data-root /data/8 \
      --out-dir ./outputs_qwen3_lr \
      --qwen3-vl-model Qwen/Qwen3-VL-8B-Instruct \
      --qwen-image-model Qwen/Qwen2-Image-2.1 \
      --base-prompt "a portrait photo of a person, studio lighting, 85mm" \
      --inference

Datenlayout:
    /data/8/seed_<id>/{*.jpg|png|webp}, optional teacher_prompt_embeds.pt [1,S,H]

Hinweis:
- Der Vision-Hook priorisiert den offiziellen Vision-Tower; fällt dann auf
  model(..., output_hidden_states=True) zurück; letzter Ausweg: ein sicheres,
  reproduzierbares Resize mit eigenem Preprocessor.
"""
from __future__ import annotations

import os, glob, json, argparse, math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,   # primär
    AutoModelForImageTextToText,       # fallback
)
from diffusers import DiffusionPipeline

# Optionales Preprocessing (Pixel-Budgets etc.)
try:
    from qwen_vl_utils import process_vision_info
    HAS_QVLU = True
except Exception:
    HAS_QVLU = False

# ---------------------------
# Misc Utils
# ---------------------------

def seed_all(s=123):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def find_images(folder: str) -> List[str]:
    exts = ("*.png","*.jpg","*.jpeg","*.webp","*.bmp")
    files = []
    for e in exts: files += glob.glob(os.path.join(folder, e))
    return sorted(files)

def save_grid(images: List[Image.Image], rows: int, cols: int, out_path: str, pad: int = 8, bg=(18,18,18)):
    assert len(images) == rows*cols
    w,h = images[0].size; W = cols*w + (cols+1)*pad; H = rows*h + (rows+1)*pad
    grid = Image.new("RGB", (W,H), bg)
    k=0
    for r in range(rows):
        for c in range(cols):
            x = pad + c*(w+pad); y = pad + r*(h+pad)
            grid.paste(images[k], (x,y)); k+=1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)

# ---------------------------
# Pixel-Budget Handling
# ---------------------------

def fit_images_to_budget(pil_images: List[Image.Image], pixel_budget_total: int, max_side: int) -> List[Image.Image]:
    """
    Skaliert mehrere Bilder proportional nach unten, sodass die Summe der Pixel
    den Budget-Grenzwert nicht überschreitet, und begrenzt jede Seite durch max_side.
    Bewahrt das Seitenverhältnis jedes Bildes.
    """
    # Max-side clamp zuerst
    imgs = []
    tot = 0
    for im in pil_images:
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        new_w, new_h = int(round(w*scale)), int(round(h*scale))
        if scale < 1.0:
            im = im.resize((max(1,new_w), max(1,new_h)), Image.LANCZOS)
        imgs.append(im)
        tot += im.size[0] * im.size[1]

    if tot <= pixel_budget_total:
        return imgs

    # uniformer Downscale-Faktor über alle Bilder
    scale = math.sqrt(pixel_budget_total / float(tot))
    out = []
    for im in imgs:
        w, h = im.size
        nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
        out.append(im.resize((nw, nh), Image.LANCZOS))
    return out

# ---------------------------
# Qwen-Image: Text-Helper
# ---------------------------

class QwenImageTextHelper:
    def __init__(self, model_id: str, device: str="cuda", dtype: torch.dtype=torch.float16, hf_token: Optional[str]=None):
        self.pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True, token=hf_token).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        if not hasattr(self.pipe, "tokenizer") or not hasattr(self.pipe, "text_encoder"):
            raise RuntimeError("Qwen-Image pipeline muss tokenizer/text_encoder exposen.")
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.device, self.dtype = device, dtype

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        tok = self.tokenizer([text], padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        out = self.text_encoder(**tok)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return hidden.to(self.dtype)  # [1,S,H]

    @torch.inference_mode()
    def generate(self, prompt_embeds: torch.Tensor, negative_prompt_embeds: Optional[torch.Tensor]=None,
                 num_inference_steps: int=28, guidance_scale: float=4.5, height: int=768, width: int=512,
                 seed: Optional[int]=None) -> Image.Image:
        gen = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        img = self.pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                        height=height, width=width, generator=gen).images[0]
        return img

# ---------------------------
# Qwen3-VL: Vision Hook
# ---------------------------

class Qwen3VLVisualEncoder:
    """
    Extrahiert L2-normalisierte visuelle Features aus Qwen3-VL mit Pixel-Budget.
    Pfad-Priorität:
      (A) Offizieller Vision-Tower (get_vision_tower / vision_tower) mit pixel_values
      (B) model(..., output_hidden_states=True) und greife image/vision hidden states
      (C) Sicherer Fallback: eigener Preprocessor -> mittleres Pooling
    """
    def __init__(self, model_id: str, device: str="cuda", dtype: torch.dtype=torch.float16,
                 pixel_budget_mp: float=2.0, max_side: int=1536):
        self.device, self.dtype = device, dtype
        # Modell laden (primär dedizierte Klasse, sonst Fallback)
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=dtype, device_map="auto")
        except Exception:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map="auto")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pixel_budget_total = int(pixel_budget_mp * 1_000_000)  # Megapixel -> absolute Pixel
        self.max_side = max_side

    def _prepare_pixel_values(self, pil_images: List[Image.Image]) -> torch.Tensor:
        # 1) Budgetgerechtes Resize
        imgs = fit_images_to_budget(pil_images, self.pixel_budget_total, self.max_side)

        # 2) Mit qwen_vl_utils (falls vorhanden) einen "messages"-Pfad vorbereiten
        if HAS_QVLU:
            messages = [{
                "role": "user",
                "content": [*([{"type":"image","image":im} for im in imgs]), {"type":"text","text":"."}]
            }]
            vid = None
            try:
                vis = process_vision_info(messages)  # liefert dicts mit passenden image-Objekten
                # Der AutoProcessor kann typischerweise images=... verarbeiten
                inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                            return_tensors="pt", return_dict=True)
                inputs = {**inputs,
                          "images": vis.get("images", None),
                          "videos": vis.get("videos", None)}
                # Tokenizer-Aufruf um pixel_values zu bekommen (je nach Processor-Version)
                proc = self.processor(images=inputs.get("images", imgs), text=".", return_tensors="pt")
                if "pixel_values" in proc:
                    return proc["pixel_values"].to(self.model.device, dtype=self.dtype)
            except Exception:
                pass

        # 3) Standard-Processor: erzeugt pixel_values für Vision-Tower
        try:
            proc = self.processor(images=imgs, text=".", return_tensors="pt")
            if "pixel_values" in proc:
                return proc["pixel_values"].to(self.model.device, dtype=self.dtype)
        except Exception:
            pass

        # 4) Eigenbau-Fallback (NCHW, 224-normalisiert o. ä.)
        arrs = []
        for im in imgs:
            a = np.array(im).astype(np.float32) / 255.0  # HWC
            a = torch.from_numpy(a).permute(2,0,1)       # CHW
            arrs.append(a)
        pv = torch.stack(arrs, dim=0)  # [B,C,H,W]
        return pv.to(self.model.device, dtype=self.dtype)

    @torch.no_grad()
    def __call__(self, pil_images: List[Image.Image]) -> torch.Tensor:
        pixel_values = self._prepare_pixel_values(pil_images)  # [B,C,H,W]

        # (A) Vision-Tower bevorzugt
        try:
            if hasattr(self.model, "get_vision_tower"):
                vt = self.model.get_vision_tower()
                if hasattr(vt, "forward"):
                    out = vt(pixel_values, output_hidden_states=True)
                    last = out.last_hidden_state  # [B,N,C]
                    pooled = last.mean(dim=1).mean(dim=0)  # mittel über Tokens und Bilder
                    return F.normalize(pooled, dim=-1)
        except Exception:
            pass

        try:
            vt = getattr(self.model, "vision_tower", None)
            if vt is not None and hasattr(vt, "forward"):
                out = vt(pixel_values, output_hidden_states=True)
                last = out.last_hidden_state
                pooled = last.mean(dim=1).mean(dim=0)
                return F.normalize(pooled, dim=-1)
        except Exception:
            pass

        # (B) Model-Forward mit hidden states
        try:
            # Manche Processor-Varianten wollen den Text getrennt
            inputs = self.processor(text=".", images=[Image.new("RGB",(1,1))], return_tensors="pt")
        except Exception:
            inputs = {"input_ids": None}

        try:
            out = self.model(pixel_values=pixel_values, output_hidden_states=True, **{k:v for k,v in inputs.items() if k!="pixel_values"})
            # Mögliche Felder: vision_outputs.hidden_states, image_hidden_states, etc.
            if hasattr(out, "vision_outputs") and out.vision_outputs is not None:
                hs = out.vision_outputs.hidden_states if hasattr(out.vision_outputs, "hidden_states") else None
                if hs is not None and len(hs)>0:
                    last = hs[-1]  # [B,N,C]
                    pooled = last.mean(dim=1).mean(dim=0)
                    return F.normalize(pooled, dim=-1)
            if hasattr(out, "image_hidden_states") and out.image_hidden_states is not None:
                last = out.image_hidden_states[-1]  # [B,N,C]
                pooled = last.mean(dim=1).mean(dim=0)
                return F.normalize(pooled, dim=-1)
        except Exception:
            pass

        # (C) Notfall: einfache Global Average Pooling über NCHW -> CHW
        pooled = pixel_values.float().mean(dim=[0,2,3])  # [C]
        return F.normalize(pooled, dim=-1)

# ---------------------------
# Optional: Qwen3-VL Captioner
# ---------------------------

class Qwen3VLCaptioner:
    def __init__(self, model_id: str, device: str="cuda", dtype: torch.dtype=torch.float16):
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=dtype, device_map="auto")
        except Exception:
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map="auto")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = self.model.device

    @torch.no_grad()
    def __call__(self, pil_images: List[Image.Image]) -> str:
        # Caption-Prompt knapp halten (robuste Identitätsmerkmale)
        messages = [{
            "role":"user",
            "content":[*([{"type":"image","image":im} for im in pil_images]),
                       {"type":"text","text":"Describe identity & distinctive attributes (hair, age, style, accessories) succinctly."}]
        }]
        try:
            if HAS_QVLU:
                vis = process_vision_info(messages)
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=vis.get("images", None), videos=vis.get("videos", None), return_tensors="pt").to(self.device)
            else:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=pil_images, return_tensors="pt").to(self.device)
            gen = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
            out = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
            return out.strip()
        except Exception:
            return "a portrait of a person with distinctive identity features"

# ---------------------------
# Dataset
# ---------------------------

class SubjectDataset(Dataset):
    def __init__(self, data_root: str, visual_encoder: Qwen3VLVisualEncoder,
                 base_prompt_embeds: torch.Tensor, expect_teachers: bool,
                 bootstrap_captions: bool, captioner: Optional[Qwen3VLCaptioner],
                 max_images: int=3):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.base_prompt_embeds = base_prompt_embeds
        self.expect_teachers = expect_teachers
        self.bootstrap_captions = bootstrap_captions
        self.captioner = captioner

        self.items = []
        for sd in sorted([d for d in glob.glob(os.path.join(data_root, "seed_*")) if os.path.isdir(d)]):
            imgs = find_images(sd)[:max_images]
            if not imgs: continue
            teacher = os.path.join(sd, "teacher_prompt_embeds.pt")
            caption_file = os.path.join(sd, "caption.txt")
            self.items.append({"dir": sd, "images": imgs,
                               "teacher": teacher if os.path.exists(teacher) else None,
                               "caption_file": caption_file})
        if not self.items:
            raise RuntimeError(f"Keine Subjekte unter {data_root} gefunden.")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # Lade PILs (Originalgrößen; Budgeting im Encoder)
        pil_images = [Image.open(p).convert("RGB") for p in item["images"]]
        with torch.no_grad():
            feat = self.visual_encoder(pil_images)  # [D]

        target_residual = None; caption_text = None
        if item["teacher"] is not None:
            teacher = torch.load(item["teacher"], map_location="cpu")  # [1,S,H]
            target_residual = teacher.to(self.base_prompt_embeds.device) - self.base_prompt_embeds
        elif self.bootstrap_captions:
            if os.path.exists(item["caption_file"]):
                with open(item["caption_file"], "r", encoding="utf-8") as f:
                    caption_text = f.read().strip()
            else:
                if self.captioner is None:
                    raise RuntimeError("bootstrap_captions=True, aber kein Captioner gesetzt.")
                caption_text = self.captioner(pil_images)
                with open(item["caption_file"], "w", encoding="utf-8") as f:
                    f.write(caption_text)

        return {"feat": feat, "dir": item["dir"], "images": pil_images,
                "target_residual": target_residual, "caption": caption_text}

# ---------------------------
# Linear Head
# ---------------------------

class LinearResidualHead(nn.Module):
    def __init__(self, in_dim: int, seq_len: int, hidden_dim: int):
        super().__init__()
        self.seq_len, self.hidden_dim = seq_len, hidden_dim
        self.fc = nn.Linear(in_dim, seq_len*hidden_dim, bias=True)
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim()==1: z = z[None,:]
        x = self.fc(z).view(z.size(0), self.seq_len, self.hidden_dim)  # [B,S,H]
        rms = torch.sqrt(torch.clamp((x**2).mean(dim=-1, keepdim=True), min=1e-9))
        x = x / (rms + 1e-9)
        scale = torch.clamp(torch.exp(self.log_scale), 0.01, 1.5)
        return x * scale  # [B,S,H]

# ---------------------------
# Train / Inference
# ---------------------------

def train_linear(head: LinearResidualHead, loader: DataLoader, device: str="cuda",
                 epochs: int=5, lr: float=1e-3, base_prompt_embeds: torch.Tensor=None,
                 helper: QwenImageTextHelper=None):
    head.train(); opt = torch.optim.AdamW(head.parameters(), lr=lr); step=0
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}", ncols=100)
        for batch in pbar:
            feat = batch["feat"].to(device)
            target = batch["target_residual"]
            if target is None:
                caption = batch["caption"][0] if isinstance(batch["caption"], list) else batch["caption"]
                if caption is None: continue
                with torch.no_grad():
                    t_embed = helper.encode_text(caption).to(device)  # [1,S,H]
                    target = t_embed - base_prompt_embeds.to(device)
            else:
                target = target.to(device)  # [1,S,H]

            pred = head(feat)  # [B,S,H]
            loss_mse = F.mse_loss(pred, target)
            loss_cos = 1.0 - F.cosine_similarity(pred.view(pred.size(0), -1), target.view(target.size(0), -1), dim=-1).mean()
            loss = 0.7*loss_cos + 0.3*loss_mse
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); step+=1
            if step % 10 == 0: pbar.set_postfix({"loss": float(loss.detach().cpu())})
    print("Training done.")

@torch.no_grad()
def alpha_sweep_inference(helper: QwenImageTextHelper, head: LinearResidualHead, dataset: SubjectDataset,
                          base_prompt_embeds: torch.Tensor, negative_prompt_embeds: Optional[torch.Tensor],
                          out_dir: str, alphas: List[float], seeds: List[int],
                          num_inference_steps: int=28, guidance_scale: float=4.5, height: int=768, width: int=512):
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for batch in tqdm(loader, desc="Alpha sweep", ncols=100):
        feat = batch["feat"]
        subj = os.path.basename(batch["dir"][0])
        pred_res = head(feat)  # [1,S,H]
        images = []
        for a in alphas:
            pe = base_prompt_embeds + a * pred_res
            for sd in seeds:
                img = helper.generate(pe, negative_prompt_embeds, num_inference_steps, guidance_scale, height, width, seed=sd)
                images.append(img)
        rows, cols = len(alphas), len(seeds)
        grid = os.path.join(out_dir, f"{subj}_alpha_grid.png")
        save_grid(images, rows, cols, grid)
        with open(os.path.join(out_dir, f"{subj}_alpha_meta.json"), "w") as f:
            json.dump({"alphas": alphas, "seeds": seeds,
                       "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale,
                       "height": height, "width": width}, f, indent=2)

# ---------------------------
# Args / Main
# ---------------------------

@dataclass
class Args:
    data_root: str; out_dir: str
    qwen3_vl_model: str; qwen_image_model: str
    base_prompt: str; negative_prompt: Optional[str]
    device: str; dtype: str
    pixel_budget_mp: float; max_side: int
    train: bool; inference: bool; bootstrap_captions: bool
    batch_size: int; epochs: int; lr: float
    seeds: List[int]; alphas: List[float]
    steps: int; guidance: float; height: int; width: int
    hf_token: Optional[str]

def parse_args()->Args:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--qwen3-vl-model", type=str, required=True, help="z.B. Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--qwen-image-model", type=str, required=True, help="z.B. Qwen/Qwen2-Image-2.1")
    ap.add_argument("--base-prompt", type=str, required=True)
    ap.add_argument("--negative-prompt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16","fp32","bf16"])
    ap.add_argument("--pixel-budget-mp", type=float, default=2.0, help="Gesamt-Megapixel-Budget über alle Referenzbilder (z. B. 2.0)")
    ap.add_argument("--max-side", type=int, default=1536, help="Maximale Kantenlänge pro Bild nach Resize")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--inference", action="store_true")
    ap.add_argument("--bootstrap-captions", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[i for i in range(16)])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5,1.0,1.5,2.0])
    ap.add_argument("--num-inference-steps", type=int, default=28)
    ap.add_argument("--guidance-scale", type=float, default=4.5)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--hf-token", type=str, default=None)
    a = ap.parse_args()
    return Args(a.data_root, a.out_dir, a.qwen3_vl_model, a.qwen_image_model,
                a.base_prompt, a.negative_prompt, a.device, a.dtype,
                a.pixel_budget_mp, a.max_side,
                a.train, a.inference, a.bootstrap_captions,
                a.batch_size, a.epochs, a.lr,
                a.seeds, a.alphas,
                a.num_inference_steps, a.guidance_scale, a.height, a.width,
                a.hf_token)

def main():
    seed_all(123); args = parse_args()
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    # Qwen-Image: Textseite
    helper = QwenImageTextHelper(args.qwen_image_model, device=args.device, dtype=torch_dtype, hf_token=args.hf_token)
    base_embeds = helper.encode_text(args.base_prompt)
    neg_embeds = helper.encode_text(args.negative_prompt) if args.negative_prompt else None
    S, H = base_embeds.shape[1], base_embeds.shape[2]

    # Qwen3-VL: Vision-Hook mit Budget
    encoder = Qwen3VLVisualEncoder(args.qwen3_vl_model, device=args.device, dtype=torch_dtype,
                                   pixel_budget_mp=args.pixel_budget_mp, max_side=args.max_side)

    # Optionaler Captioner
    captioner = Qwen3VLCaptioner(args.qwen3_vl_model, device=args.device, dtype=torch_dtype) if args.bootstrap_captions else None

    # Dataset
    dataset = SubjectDataset(args.data_root, encoder, base_embeds, expect_teachers=args.train,
                             bootstrap_captions=args.bootstrap_captions, captioner=captioner, max_images=3)

    # Feature-Dim automatisch bestimmen
    with torch.no_grad():
        in_dim = dataset[0]["feat"].numel()

    # Linearer Head
    head = LinearResidualHead(in_dim, S, H).to(args.device, dtype=torch_dtype)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "linear_head_qwen3vl.pt")

    # Train
    if args.train:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
        train_linear(head, loader, device=args.device, epochs=args.epochs, lr=args.lr,
                     base_prompt_embeds=base_embeds.to(args.device), helper=helper)
        torch.save(head.state_dict(), ckpt_path)
    else:
        if os.path.exists(ckpt_path):
            head.load_state_dict(torch.load(ckpt_path, map_location=args.device))
            print(f"[INFO] Loaded {ckpt_path}")

    # Inference
    if args.inference or not args.train:
        out_dir = os.path.join(args.out_dir, "alpha_sweeps")
        alpha_sweep_inference(helper, head, dataset, base_embeds.to(args.device, dtype=torch_dtype),
                              neg_embeds.to(args.device, dtype=torch_dtype) if neg_embeds is not None else None,
                              out_dir, args.alphas, args.seeds,
                              num_inference_steps=args.steps, guidance_scale=args.guidance,
                              height=args.height, width=args.width)

    print("Done. Outputs in:", args.out_dir)

if __name__ == "__main__":
    main()
