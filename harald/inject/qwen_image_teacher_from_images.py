#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learn teacher_prompt_embeds.pt from finished images (Qwen-Image)
================================================================

Ziel
----
Für jedes Subjekt (Ordner /data/8/seed_<id>/ mit 1–n Bildern) optimieren wir direkt
die **prompt_embeds** (Form [1, S, H]) des Qwen-Image-Textencoders, sodass die
Diffusionsvorhersage zum Bild passt (Noise-Prediction-Loss, ähnlich Textual Inversion).
Ergebnis wird als `teacher_prompt_embeds.pt` im jeweiligen seed_<id>-Ordner gespeichert.

Funktionsprinzip
----------------
- Qwen-Image-Pipeline (Diffusers) liefert: tokenizer, text_encoder, vae, unet, scheduler, image_processor.
- Wir initialisieren E0 = text_encoder(base_prompt).
- Wir optimieren eine Residual-Parameterisierung: E = E0 + scale * normed(P), P ist lernbar.
- Loss: MSE zwischen vorhergesagtem Rauschen und echtem Rauschen (standard Diffusers Training).
- (Optional) Unconditional/negative prompt Embeddings für CFG.
- Nach T Schritten speichern wir die finalen `teacher_prompt_embeds.pt` = E.

Install:
    pip install -U diffusers transformers accelerate pillow tqdm torch torchvision

Beispiele:
    python qwen_image_teacher_from_images.py \
      --data-root /data/8 \
      --qwen-image-model Qwen/Qwen2-Image-2.1 \
      --base-prompt "a portrait photo of a person, studio lighting, 85mm" \
      --negative-prompt "blurry, low quality, deformed, watermark" \
      --steps 600 --lr 5e-3 --cfg 4.5 --height 768 --width 512

Hinweise:
- Für stabile Gradienten empfehle ich **fp16/bf16** nur, wenn deine GPU das sauber trägt.
- Bei wenig Bildern nutze starke Augmentationen optional (hier einfach gehalten).

Output:
    /data/8/seed_<id>/teacher_prompt_embeds.pt  # torch.Tensor [1, S, H]
"""
from __future__ import annotations

import os, glob, argparse, math, random
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from diffusers import DiffusionPipeline
from torchvision import transforms

def seed_all(s=123):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def find_images(folder: str) -> List[str]:
    exts = ("*.png","*.jpg","*.jpeg","*.webp","*.bmp")
    files = []
    for e in exts: files += glob.glob(os.path.join(folder, e))
    return sorted(files)

def pil_to_latents(pipe, pil_imgs: List[Image.Image], device, dtype):
    ip = pipe.image_processor
    pt = ip.preprocess(pil_imgs).to(device, dtype=dtype)  # [B, C, H, W]
    with torch.no_grad():
        posterior = pipe.vae.encode(pt).latent_dist
        latents = posterior.sample() * pipe.vae.config.scaling_factor
    return latents

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--qwen-image-model", type=str, required=True, help="e.g., Qwen/Qwen2-Image-2.1")
    ap.add_argument("--base-prompt", type=str, required=True)
    ap.add_argument("--negative-prompt", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16","fp32","bf16"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=4.5, help="Guidance scale während Training (CFG); 1.0 ≈ kein CFG")
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--subjects", type=str, nargs="*", default=None, help="Optional: explizite seed_<id> Liste")
    a = ap.parse_args()

    seed_all(123)
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[a.dtype]

    pipe = DiffusionPipeline.from_pretrained(a.qwen_image_model, torch_dtype=torch_dtype, use_safetensors=True).to(a.device)
    pipe.set_progress_bar_config(disable=True)
    unet, vae, tokenizer, text_encoder, scheduler = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder, pipe.scheduler

    def encode_text(txt: str):
        tok = tokenizer([txt], padding="max_length", truncation=True, return_tensors="pt").to(a.device)
        out = text_encoder(**tok)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return hidden  # [1,S,H]

    base_embeds = encode_text(a.base_prompt).detach()
    neg_embeds = encode_text(a.negative_prompt).detach() if a.negative_prompt else None
    S, H = base_embeds.shape[1], base_embeds.shape[2]

    subject_dirs = sorted([d for d in glob.glob(os.path.join(a.data_root, "seed_*")) if os.path.isdir(d)])
    if a.subjects:
        subject_dirs = [os.path.join(a.data_root, s) for s in a.subjects if os.path.isdir(os.path.join(a.data_root, s))]
    if not subject_dirs:
        raise RuntimeError(f"Keine seed_* Ordner unter {a.data_root} gefunden.")

    for sd in subject_dirs:
        imgs_paths = find_images(sd)
        if not imgs_paths:
            print(f"[WARN] Keine Bilder in {sd}, überspringe.")
            continue
        pil_imgs = [Image.open(p).convert("RGB").resize((a.width, a.height), Image.LANCZOS) for p in imgs_paths]

        P = torch.zeros((1, S, H), device=a.device, dtype=torch_dtype, requires_grad=True)
        log_scale = torch.nn.Parameter(torch.zeros(1, device=a.device, dtype=torch_dtype), requires_grad=True)
        opt = torch.optim.AdamW([P, log_scale], lr=a.lr)

        latents_cache = pil_to_latents(pipe, pil_imgs, a.device, torch_dtype)

        pbar = tqdm(range(a.steps), desc=f"Optimize {os.path.basename(sd)}", ncols=100)
        for step in pbar:
            idxs = np.random.choice(len(pil_imgs), size=min(a.batch_size, len(pil_imgs)), replace=False)
            latents = latents_cache[idxs].clone()

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),), device=a.device, dtype=torch.long)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            norm = torch.sqrt(torch.clamp((P**2).mean(), min=1e-12))
            P_hat = P / (norm + 1e-9)
            scale = torch.clamp(torch.exp(log_scale), 0.01, 1.5)
            prompt_embeds = (base_embeds + scale * P_hat).to(a.device)

            if a.cfg and a.cfg > 1.0 and neg_embeds is not None:
                latent_in = torch.cat([noisy_latents, noisy_latents], dim=0)
                ts_in = torch.cat([timesteps, timesteps], dim=0)
                hidden = torch.cat([neg_embeds.expand_as(prompt_embeds), prompt_embeds], dim=0)
                latent_in = scheduler.scale_model_input(latent_in, ts_in)
                pred = unet(latent_in, ts_in, encoder_hidden_states=hidden).sample
                uncond, cond = pred.chunk(2, dim=0)
                noise_pred = uncond + a.cfg * (cond - uncond)
            else:
                latent_in = scheduler.scale_model_input(noisy_latents, timesteps)
                noise_pred = unet(latent_in, timesteps, encoder_hidden_states=prompt_embeds).sample

            if hasattr(scheduler.config, "prediction_type") and scheduler.config.prediction_type == "v_prediction":
                target = scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = noise

            loss = F.mse_loss(noise_pred.float(), target.float())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if (step+1) % 25 == 0:
                pbar.set_postfix({"loss": float(loss.detach().cpu()), "scale": float(scale.detach().cpu())})

            if (step+1) % a.save_every == 0 or (step+1) == a.steps:
                out_path = os.path.join(sd, "teacher_prompt_embeds.pt")
                final_E = (base_embeds + scale.detach() * (P.detach() / (torch.sqrt(torch.clamp((P.detach()**2).mean(), min=1e-12)) + 1e-9))).to("cpu")
                torch.save(final_E, out_path)

        print(f"[OK] Saved teacher_prompt_embeds.pt -> {sd}")

if __name__ == "__main__":
    main()
