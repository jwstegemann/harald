#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL to Qwen-Image Linear Injection - Minimal POC
======================================================

Minimal end-to-end proof-of-concept for fine-tuning approach inspired by
Textual Inversion. Maps visual features from Qwen3-VL to Qwen-Image prompt
embeddings via a linear residual head.

Usage (on modal.com):
    modal run -m harald.modal_runner::injection_poc

Requirements:
- Follows harald/config.py patterns (HF_CACHE, HF_SECRET, GPU/CUDA)
- Reads from /mnt/dataset/X/seed_*/photo_view_*.png
- Writes to /mnt/output/injection_poc/
- Single image per identity (simplified)
- Deterministic seeding throughout
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from diffusers import DiffusionPipeline

# ============================================================================
# Constants & Config
# ============================================================================

# Fixed image size per project config
IMAGE_SIZE = 448

# Default model IDs
QWEN3_VL_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_IMAGE_MODEL = "Qwen/Qwen-Image"

# Default generation params (Qwen-Image recommended)
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 4.0
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024

# Maximum sequence length for text embeddings
# Use 256 to preserve caption details while staying memory-efficient
MAX_SEQ_LEN = 256

# ============================================================================
# Utilities
# ============================================================================


def seed_all(seed: int = 123):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_photo_images(folder: str) -> List[str]:
    """Find photo_view_*.png files in a directory."""
    pattern = os.path.join(folder, "photo_view_*.png")
    files = glob.glob(pattern)
    return sorted(files)


def save_grid(
    images: List[Image.Image],
    rows: int,
    cols: int,
    out_path: str,
    pad: int = 8,
    bg=(18, 18, 18),
):
    """Save a grid of images."""
    assert len(images) == rows * cols, (
        f"Expected {rows * cols} images, got {len(images)}"
    )
    if not images:
        raise ValueError("No images to save")

    w, h = images[0].size
    W = cols * w + (cols + 1) * pad
    H = rows * h + (rows + 1) * pad
    grid = Image.new("RGB", (W, H), bg)

    k = 0
    for r in range(rows):
        for c in range(cols):
            x = pad + c * (w + pad)
            y = pad + r * (h + pad)
            grid.paste(images[k], (x, y))
            k += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)
    print(f"Saved grid to {out_path}")


# ============================================================================
# Qwen3-VL Visual Feature Extractor
# ============================================================================


class Qwen3VLFeatureExtractor:
    """
    Extract L2-normalized visual features from Qwen3-VL.
    Simplified version: single image, fixed resize, standard API.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        image_size: int = IMAGE_SIZE,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.image_size = image_size

        # Convert torch.dtype to string for transformers API
        if dtype == torch.bfloat16:
            dtype_str = "bfloat16"
        elif dtype == torch.float16:
            dtype_str = "float16"
        elif dtype == torch.float32:
            dtype_str = "float32"
        else:
            dtype_str = "auto"

        print(f"Loading Qwen3-VL model: {model_id} (dtype={dtype_str})")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype_str,  # String format for transformers
            device_map="auto",  # Auto device mapping works well
            attn_implementation="flash_attention_2",  # Efficient attention for long sequences
            token=hf_token,
            cache_dir=cache_dir,
        )
        self.model.eval()
        print(f"Qwen3-VL loaded in inference mode with Flash Attention 2")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            token=hf_token,
            cache_dir=cache_dir,
        )
        print(f"Qwen3-VL loaded on {self.model.device}")

    def _preprocess_image(self, pil_image: Image.Image) -> Image.Image:
        """Resize image to fixed size, preserving aspect ratio with padding."""
        # Simple center crop/resize to square
        w, h = pil_image.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        pil_image = pil_image.crop((left, top, left + s, top + s))
        pil_image = pil_image.resize((self.image_size, self.image_size), Image.LANCZOS)
        return pil_image

    @torch.no_grad()
    def extract_features(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Extract visual features from a single image.
        Returns L2-normalized feature vector [D].
        """
        # Preprocess
        img = self._preprocess_image(pil_image)

        # Process with Qwen3-VL
        # Use a minimal text prompt to get vision features
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "."},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Forward pass to get hidden states
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract vision features from hidden states
        # Strategy: take last hidden state and pool
        hidden_states = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]

        # Global average pooling over sequence dimension
        pooled = hidden_states.mean(dim=1).squeeze(0)  # [hidden_dim]

        # L2 normalization
        pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled.to(self.dtype)


# ============================================================================
# Qwen-Image Text Encoding Helper
# ============================================================================


class QwenImageTextHelper:
    """
    Helper to encode text prompts to embeddings for Qwen-Image pipeline
    and generate images from prompt embeddings.

    Note: Qwen-Image may not support prompt_embeds directly. This implementation
    attempts to access the text encoder components for embedding injection.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,  # Qwen-Image prefers bfloat16
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.device = device
        self.dtype = dtype

        print(f"Loading Qwen-Image pipeline: {model_id}")
        # DiffusionPipeline uses torch_dtype (not dtype string) and doesn't support device_map="auto"
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,  # DiffusionPipeline expects torch.dtype object
            token=hf_token,
            cache_dir=cache_dir,
        ).to(device)  # Use .to() for device placement

        self.pipe.set_progress_bar_config(disable=True)
        print(f"Qwen-Image pipeline loaded on {device}")

    @torch.no_grad()
    def encode_text(
        self, text: str, max_sequence_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text to prompt embeddings using the pipeline's encode_prompt method.

        Args:
            text: Text prompt to encode
            max_sequence_length: Maximum sequence length (default 512, max 1024)

        Returns:
            Tuple of (prompt_embeds, prompt_embeds_mask):
                - prompt_embeds: [1, seq_len, hidden_dim]
                - prompt_embeds_mask: [1, seq_len]
        """
        # Use the pipeline's official encode_prompt method
        prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
            prompt=text,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )

        # Ensure mask is on correct device (it may return as CPU tensor)
        prompt_embeds_mask = prompt_embeds_mask.to(self.device)

        # Return embeddings in their original dtype from encode_prompt
        # The pipeline handles dtype conversion internally
        return prompt_embeds, prompt_embeds_mask

    @torch.inference_mode()
    def generate(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate image from prompt embeddings using Qwen-Image pipeline.

        Args:
            prompt_embeds: Prompt embeddings [1, seq_len, hidden_dim]
            prompt_embeds_mask: Prompt embeddings mask [1, seq_len]
            negative_prompt_embeds: Optional negative prompt embeddings
            negative_prompt_embeds_mask: Optional negative prompt embeddings mask
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale (Qwen-Image uses true_cfg_scale)
            height: Output image height
            width: Output image width
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image
        """
        # Use explicit "cuda" device for generator (official example pattern)
        generator = (
            torch.Generator(device="cuda").manual_seed(seed)
            if seed is not None
            else None
        )

        # Qwen-Image pipeline call with prompt_embeds and masks
        img = self.pipe(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            true_cfg_scale=guidance_scale,  # Qwen-Image specific parameter
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=generator,
        ).images[0]

        return img


# ============================================================================
# Dataset
# ============================================================================


class SinglePhotoDataset(Dataset):
    """
    Dataset for loading single photo per identity from seed_* directories.
    """

    def __init__(
        self,
        seed_dirs: List[str],
        feature_extractor: Qwen3VLFeatureExtractor,
        base_prompt_embeds: torch.Tensor,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.base_prompt_embeds = base_prompt_embeds

        self.items = []
        for seed_dir in seed_dirs:
            if not os.path.isdir(seed_dir):
                print(f"Warning: {seed_dir} is not a directory, skipping")
                continue

            photos = find_photo_images(seed_dir)
            if not photos:
                print(f"Warning: No photo images found in {seed_dir}, skipping")
                continue

            # Take first photo image
            photo_path = photos[0]
            seed_name = os.path.basename(seed_dir)

            self.items.append(
                {
                    "seed_dir": seed_dir,
                    "seed_name": seed_name,
                    "photo_path": photo_path,
                }
            )

        if not self.items:
            raise RuntimeError(f"No valid items found in provided seed directories")

        print(f"Dataset: {len(self.items)} identities loaded")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]

        # Load image
        pil_image = Image.open(item["photo_path"]).convert("RGB")

        # Extract features
        with torch.no_grad():
            feat = self.feature_extractor.extract_features(pil_image)

        return {
            "feat": feat,
            "seed_name": item["seed_name"],
            "seed_dir": item["seed_dir"],
        }


class PreExtractedFeaturesDataset(Dataset):
    """
    Dataset that works with pre-extracted features stored in memory.
    Also includes caption texts for training.
    """

    def __init__(
        self,
        features: dict[str, torch.Tensor],
        captions: dict[str, str],
        device: str = "cuda",
    ):
        super().__init__()
        self.features = features
        self.captions = captions
        self.seed_names = list(features.keys())
        self.device = device

        if not self.seed_names:
            raise RuntimeError("No features provided")

        print(f"PreExtractedFeaturesDataset: {len(self.seed_names)} identities")

    def __len__(self) -> int:
        return len(self.seed_names)

    def __getitem__(self, idx: int):
        seed_name = self.seed_names[idx]
        feat = self.features[seed_name]  # CPU tensor
        caption = self.captions[seed_name]

        return {
            "feat": feat,  # Will be moved to GPU in training loop
            "seed_name": seed_name,
            "caption": caption,
        }


# ============================================================================
# Linear Residual Head
# ============================================================================


class LinearResidualHead(nn.Module):
    """
    Linear head that maps visual features to residual over base prompt embeddings.
    Output shape: [batch, seq_len, hidden_dim]
    """

    def __init__(self, in_dim: int, seq_len: int, hidden_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Linear projection
        self.fc = nn.Linear(in_dim, seq_len * hidden_dim, bias=True)

        # Learnable scale parameter (log space for stability)
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: visual features [batch, in_dim] or [in_dim]
        Returns:
            residual embeddings [batch, seq_len, hidden_dim]
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Validate input
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("WARNING: NaN/Inf in input features, returning zeros")
            return torch.zeros(
                z.size(0), self.seq_len, self.hidden_dim, device=z.device, dtype=z.dtype
            )

        # Project to embedding space
        x = self.fc(z).view(z.size(0), self.seq_len, self.hidden_dim)

        # Check after linear projection
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: NaN/Inf after fc layer, returning zeros")
            return torch.zeros_like(x)

        # RMS normalization with extra safety
        x_squared = x**2
        if torch.isnan(x_squared).any():
            print("WARNING: NaN in x squared, returning zeros")
            return torch.zeros_like(x)

        rms = torch.sqrt(torch.clamp(x_squared.mean(dim=-1, keepdim=True), min=1e-9))
        x = x / (rms + 1e-9)

        # Check after normalization
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: NaN/Inf after RMS normalization, returning zeros")
            return torch.zeros(
                x.size(0), self.seq_len, self.hidden_dim, device=x.device, dtype=x.dtype
            )

        # Apply learnable scale (clamped for stability)
        scale = torch.clamp(torch.exp(self.log_scale), 0.01, 1.5)
        result = x * scale

        # Final check
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("WARNING: NaN/Inf in final result, returning zeros")
            return torch.zeros_like(result)

        return result


# ============================================================================
# Training
# ============================================================================


def train_linear_head(
    head: LinearResidualHead,
    loader: DataLoader,
    base_prompt_embeds: torch.Tensor,
    helper: QwenImageTextHelper,
    device: str = "cuda",
    epochs: int = 3,
    lr: float = 1e-3,
):
    """Train the linear head to predict caption embeddings as residuals."""
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    print(f"\nStarting training: {epochs} epochs, lr={lr}")
    print(f"Dataset size: {len(loader.dataset)}")

    step = 0
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100)
        epoch_loss = 0.0

        for batch in pbar:
            feat = batch["feat"].to(device)
            captions = batch["caption"]  # List of caption strings

            # Predict residual
            pred_residual = head(feat)  # [batch, seq_len, hidden_dim]

            # Check for NaN in predictions
            if torch.isnan(pred_residual).any():
                print(f"\nERROR: NaN in pred_residual at step {step}, skipping batch")
                continue

            # Encode captions to get target embeddings
            # Use MAX_SEQ_LEN to preserve caption details (not truncate to short base prompt)
            # Process each caption in batch
            caption_embeds_list = []
            for caption in captions:
                cap_embeds, _ = helper.encode_text(
                    caption, max_sequence_length=MAX_SEQ_LEN
                )
                caption_embeds_list.append(cap_embeds)

            # Stack and move to device
            caption_embeds = torch.cat(caption_embeds_list, dim=0).to(device)

            # Align sequence lengths by padding (preserve all information, no truncation)
            base_seq_len = base_prompt_embeds.shape[1]
            caption_seq_len = caption_embeds.shape[1]
            target_seq_len = max(base_seq_len, caption_seq_len)

            # Pad base_prompt_embeds if needed
            if base_seq_len < target_seq_len:
                padding = torch.zeros(
                    base_prompt_embeds.shape[0],
                    target_seq_len - base_seq_len,
                    base_prompt_embeds.shape[2],
                    device=base_prompt_embeds.device,
                    dtype=base_prompt_embeds.dtype,
                )
                base_prompt_embeds = torch.cat([base_prompt_embeds, padding], dim=1)
                print(
                    f"[DEBUG] Padded base_prompt from {base_seq_len} to {target_seq_len} tokens"
                )

            # Pad caption_embeds if needed
            if caption_seq_len < target_seq_len:
                padding = torch.zeros(
                    caption_embeds.shape[0],
                    target_seq_len - caption_seq_len,
                    caption_embeds.shape[2],
                    device=caption_embeds.device,
                    dtype=caption_embeds.dtype,
                )
                caption_embeds = torch.cat([caption_embeds, padding], dim=1)
                print(
                    f"[DEBUG] Padded caption from {caption_seq_len} to {target_seq_len} tokens"
                )

            # Debug: Print shapes after alignment
            print(f"[DEBUG] Final caption_embeds shape: {caption_embeds.shape}")
            print(f"[DEBUG] Final base_prompt_embeds shape: {base_prompt_embeds.shape}")

            # Target: residual from base to caption embeddings
            # Now both have same sequence length!
            target_residual = caption_embeds - base_prompt_embeds

            # Pad pred_residual to match target_residual if needed
            # (LinearResidualHead has fixed output size from initialization)
            pred_seq_len = pred_residual.shape[1]
            target_seq_len = target_residual.shape[1]

            if pred_seq_len < target_seq_len:
                padding = torch.zeros(
                    pred_residual.shape[0],
                    target_seq_len - pred_seq_len,
                    pred_residual.shape[2],
                    device=pred_residual.device,
                    dtype=pred_residual.dtype,
                )
                pred_residual = torch.cat([pred_residual, padding], dim=1)
                print(
                    f"[DEBUG] Padded pred_residual from {pred_seq_len} to {target_seq_len} tokens"
                )

            # MSE loss only (cosine similarity with zero targets is numerically unstable)
            loss = F.mse_loss(pred_residual, target_residual)

            # Add L2 regularization on residuals to keep them small
            l2_reg = 0.01 * (pred_residual**2).mean()
            loss = loss + l2_reg

            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"\nERROR: NaN in loss at step {step}, skipping batch")
                continue

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)

            optimizer.step()

            # Check for NaN in weights after update
            has_nan_weights = False
            for name, param in head.named_parameters():
                if torch.isnan(param).any():
                    print(f"\nERROR: NaN in {name} after optimizer step at step {step}")
                    has_nan_weights = True
            if has_nan_weights:
                print("Stopping training due to NaN in weights")
                return

            # Logging
            step += 1
            epoch_loss += loss.item()
            if step % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

    print("Training complete!")


# ============================================================================
# Inference
# ============================================================================


@torch.no_grad()
def run_inference(
    helper: QwenImageTextHelper,
    head: LinearResidualHead,
    dataset: SinglePhotoDataset,
    base_prompt_embeds: torch.Tensor,
    base_prompt_embeds_mask: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor],
    negative_prompt_embeds_mask: Optional[torch.Tensor],
    out_dir: str,
    alphas: List[float],
    seeds: List[int],
    num_inference_steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
):
    """
    Run inference with alpha sweep: generate images at different alpha values
    (strength of residual injection) for each identity.

    Note: The mask remains constant during residual injection - only embeddings are modified.
    """
    os.makedirs(out_dir, exist_ok=True)
    head.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"\nRunning inference on {len(dataset)} identities")
    print(f"Alphas: {alphas}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {out_dir}")

    for batch in tqdm(loader, desc="Generating images", ncols=100):
        feat = batch["feat"].to(base_prompt_embeds.device)
        seed_name = batch["seed_name"][0]

        # Predict residual
        pred_residual = head(feat)  # [1, seq_len, hidden_dim]

        # Debug: Print residual statistics
        print(f"\n{seed_name} - Residual stats:")
        print(f"  Shape: {pred_residual.shape}")
        print(
            f"  Mean: {pred_residual.mean().item():.6f}, Std: {pred_residual.std().item():.6f}"
        )
        print(
            f"  Min: {pred_residual.min().item():.6f}, Max: {pred_residual.max().item():.6f}"
        )

        # Check for NaNs in residual
        if torch.isnan(pred_residual).any():
            print(f"WARNING: NaN detected in residual for {seed_name}, skipping")
            continue

        # Generate images for each (alpha, seed) combination
        images = []
        for alpha in alphas:
            # Inject residual with alpha scaling (mask stays constant!)
            if alpha == 0.0:
                # Pure base embeddings (no injection)
                prompt_embeds = base_prompt_embeds
            else:
                prompt_embeds = base_prompt_embeds + alpha * pred_residual

            # Validate embeddings
            if torch.isnan(prompt_embeds).any() or torch.isinf(prompt_embeds).any():
                print(
                    f"WARNING: Invalid values in prompt_embeds for {seed_name} alpha={alpha}, using base embeddings"
                )
                prompt_embeds = base_prompt_embeds.clone()

            # Debug: Print embedding statistics for first alpha
            if alpha == alphas[0]:
                print(f"  Injected embeds (alpha={alpha}):")
                print(
                    f"    Mean: {prompt_embeds.mean().item():.6f}, Std: {prompt_embeds.std().item():.6f}"
                )
                print(
                    f"    Min: {prompt_embeds.min().item():.6f}, Max: {prompt_embeds.max().item():.6f}"
                )

            for seed in seeds:
                img = helper.generate(
                    prompt_embeds,
                    base_prompt_embeds_mask,
                    negative_prompt_embeds,
                    negative_prompt_embeds_mask,
                    num_inference_steps,
                    guidance_scale,
                    height,
                    width,
                    seed=seed,
                )

                # Save individual image immediately (no conversion)
                individual_path = os.path.join(
                    out_dir, f"{seed_name}_alpha_{alpha}_seed_{seed}.png"
                )
                img.save(individual_path)
                print(f"  Saved: {individual_path}")

                images.append(img)

        # Save grid
        rows, cols = len(alphas), len(seeds)
        grid_path = os.path.join(out_dir, f"{seed_name}_alpha_grid.png")
        save_grid(images, rows, cols, grid_path)

        # Save metadata
        meta = {
            "seed_name": seed_name,
            "alphas": alphas,
            "seeds": seeds,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
        }
        meta_path = os.path.join(out_dir, f"{seed_name}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\nInference complete! Results saved to {out_dir}")


# ============================================================================
# CLI & Main
# ============================================================================


@dataclass
class Args:
    seed_dirs: List[str]
    out_dir: str
    qwen3_vl_model: str
    qwen_image_model: str
    base_prompt: str
    negative_prompt: Optional[str]
    device: str
    dtype: str
    train: bool
    epochs: int
    batch_size: int
    lr: float
    alphas: List[float]
    gen_seeds: List[int]
    steps: int
    guidance: float
    height: int
    width: int
    hf_token: Optional[str]
    cache_dir: Optional[str]


def parse_args() -> Args:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen3-VL to Qwen-Image Linear Injection POC"
    )

    # Data
    parser.add_argument(
        "--seed-dirs",
        type=str,
        required=True,
        help="Comma-separated list of seed_* directories (e.g., /mnt/dataset/1/seed_123,/mnt/dataset/1/seed_456)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/output/injection_poc",
        help="Output directory for checkpoints and generated images",
    )

    # Models
    parser.add_argument(
        "--qwen3-vl-model",
        type=str,
        default=QWEN3_VL_MODEL,
        help="Qwen3-VL model ID",
    )
    parser.add_argument(
        "--qwen-image-model",
        type=str,
        default=QWEN_IMAGE_MODEL,
        help="Qwen-Image model ID",
    )

    # Prompts
    parser.add_argument(
        "--base-prompt",
        type=str,
        default="a portrait photo of a person, studio lighting, 85mm",
        help="Base prompt for image generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, deformed, watermark",
        help="Negative prompt for image generation",
    )

    # Training
    parser.add_argument("--train", action="store_true", help="Train the linear head")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Inference
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5],
        help="Alpha values for residual scaling (0.0 = pure base embeddings, no injection)",
    )
    parser.add_argument(
        "--gen-seeds",
        type=int,
        nargs="+",
        default=[1234, 5678, 9999],
        help="Random seeds for generation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=DEFAULT_GUIDANCE,
        help="Guidance scale",
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_HEIGHT, help="Generated image height"
    )
    parser.add_argument(
        "--width", type=int, default=DEFAULT_WIDTH, help="Generated image width"
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "fp32", "bf16"],
        help="Data type (bf16 recommended for Qwen-Image)",
    )
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token")
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="HuggingFace cache directory"
    )

    args = parser.parse_args()

    # Parse seed directories
    seed_dirs = [d.strip() for d in args.seed_dirs.split(",")]

    return Args(
        seed_dirs=seed_dirs,
        out_dir=args.out_dir,
        qwen3_vl_model=args.qwen3_vl_model,
        qwen_image_model=args.qwen_image_model,
        base_prompt=args.base_prompt,
        negative_prompt=args.negative_prompt,
        device=args.device,
        dtype=args.dtype,
        train=args.train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alphas=args.alphas,
        gen_seeds=args.gen_seeds,
        steps=args.steps,
        guidance=args.guidance,
        height=args.height,
        width=args.width,
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
    )


def extract_all_features(
    seed_dirs: List[str],
    feature_extractor: Qwen3VLFeatureExtractor,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """
    Extract visual features and load captions from all seed directories.
    Store in CPU memory to free GPU before loading generative model.

    Returns:
        Tuple of (features dict, captions dict) mapping seed_name → feature/caption
    """
    print("\n" + "=" * 80)
    print("PHASE 1: FEATURE EXTRACTION & CAPTION LOADING")
    print("=" * 80)

    features = {}
    captions = {}

    for seed_dir in tqdm(seed_dirs, desc="Extracting features", ncols=100):
        if not os.path.isdir(seed_dir):
            print(f"Warning: {seed_dir} is not a directory, skipping")
            continue

        photos = find_photo_images(seed_dir)
        if not photos:
            print(f"Warning: No photo images found in {seed_dir}, skipping")
            continue

        # Take first photo
        photo_path = photos[0]
        seed_name = os.path.basename(seed_dir)

        # Load and extract features
        pil_image = Image.open(photo_path).convert("RGB")
        with torch.no_grad():
            feat = feature_extractor.extract_features(pil_image)

        # Validate features
        if torch.isnan(feat).any() or torch.isinf(feat).any():
            print(f"WARNING: Invalid features for {seed_name}, skipping")
            continue

        # Load caption from corresponding .txt file
        caption_path = photo_path.replace(".png", ".txt")
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            print(f"WARNING: No caption file for {seed_name}, using default")
            caption = "a portrait photo of a person"

        # Store on CPU to free GPU memory
        features[seed_name] = feat.cpu()
        captions[seed_name] = caption

    print(f"\nExtracted features for {len(features)} identities")

    # Print feature statistics
    if features:
        all_feats = torch.stack(list(features.values()))
        print(f"Feature statistics:")
        print(f"  Shape per identity: {feat.shape}")
        print(
            f"  Mean: {all_feats.mean().item():.6f}, Std: {all_feats.std().item():.6f}"
        )
        print(f"  Min: {all_feats.min().item():.6f}, Max: {all_feats.max().item():.6f}")

    print(f"Loaded captions for {len(captions)} identities")
    return features, captions


def main():
    """Main entry point for the POC."""
    # Set deterministic seed
    seed_all(123)

    # Parse arguments
    args = parse_args()

    # Get HF token from environment if not provided
    hf_token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN")

    # Get cache directory (follow harald/config.py pattern)
    cache_dir = args.cache_dir or os.environ.get("HF_HUB_CACHE", str(Path("/models")))

    # Dtype mapping
    dtype_map = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    print("=" * 80)
    print("Qwen3-VL to Qwen-Image Linear Injection POC")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"HF Cache: {cache_dir}")
    print(f"Output: {args.out_dir}")
    print(f"Seed directories: {len(args.seed_dirs)}")
    for sd in args.seed_dirs:
        print(f"  - {sd}")
    print("=" * 80)

    # ========================================================================
    # PHASE 1: Feature Extraction with Qwen3-VL
    # ========================================================================
    print("\nInitializing Qwen3-VL for feature extraction...")
    feature_extractor = Qwen3VLFeatureExtractor(
        args.qwen3_vl_model,
        device=args.device,
        dtype=torch_dtype,
        image_size=IMAGE_SIZE,
        hf_token=hf_token,
        cache_dir=cache_dir,
    )

    # Extract all features and store in CPU memory
    all_features, all_captions = extract_all_features(args.seed_dirs, feature_extractor)

    # Get feature dimension
    feat_dim = next(iter(all_features.values())).numel()
    print(f"Feature dimension: {feat_dim}")

    # Free GPU memory by deleting feature extractor
    print("\nFreeing GPU memory...")
    del feature_extractor
    torch.cuda.empty_cache()
    print("Qwen3-VL unloaded from GPU")

    # ========================================================================
    # PHASE 2: Text Encoding and Generation with Qwen-Image
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: TEXT ENCODING & MODEL SETUP")
    print("=" * 80)

    # Initialize Qwen-Image helper
    helper = QwenImageTextHelper(
        args.qwen_image_model,
        device=args.device,
        dtype=torch_dtype,
        hf_token=hf_token,
        cache_dir=cache_dir,
    )

    # TEST: Generate baseline image with text prompt (no embeddings)
    print("\n" + "=" * 80)
    print("BASELINE TEST: Text-based generation")
    print("=" * 80)
    print("Testing pipeline with direct text prompt (official example pattern)...")
    baseline_img = helper.pipe(
        prompt=args.base_prompt,
        negative_prompt=" ",
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        true_cfg_scale=args.guidance,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]
    baseline_path = os.path.join(args.out_dir, "test_baseline_text.png")
    os.makedirs(args.out_dir, exist_ok=True)
    baseline_img.save(baseline_path)
    print(f"Baseline image saved to: {baseline_path}")
    print("If this image is not black, the pipeline works!")

    # Encode base prompts with fixed max_sequence_length to preserve caption details
    print(
        f"\nEncoding base prompt with max_seq_len={MAX_SEQ_LEN}: '{args.base_prompt}'"
    )
    base_embeds, base_mask = helper.encode_text(
        args.base_prompt, max_sequence_length=MAX_SEQ_LEN
    )
    seq_len, hidden_dim = base_embeds.shape[1], base_embeds.shape[2]
    print(
        f"Base embeddings shape: {base_embeds.shape} (seq_len={seq_len}, hidden_dim={hidden_dim})"
    )
    print(f"Base mask shape: {base_mask.shape}")

    # Always use single space for negative prompt (Qwen-Image official example pattern)
    negative_prompt_text = " "
    print(f"Encoding negative prompt: '{negative_prompt_text}'")
    neg_embeds, neg_mask = helper.encode_text(
        negative_prompt_text, max_sequence_length=MAX_SEQ_LEN
    )

    # Create dataset from pre-extracted features
    print("\nCreating dataset from pre-extracted features...")
    dataset = PreExtractedFeaturesDataset(
        all_features, all_captions, device=args.device
    )

    # Initialize linear head
    head = LinearResidualHead(feat_dim, seq_len, hidden_dim).to(
        args.device, dtype=torch_dtype
    )
    print(f"Linear head parameters: {sum(p.numel() for p in head.parameters()):,}")

    # Setup checkpoint path
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "linear_head.pt")

    # Training
    if args.train:
        print("\n" + "=" * 80)
        print("TRAINING")
        print("=" * 80)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        train_linear_head(
            head,
            loader,
            base_embeds.to(args.device),
            helper,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
        )

        # Save checkpoint
        torch.save(head.state_dict(), ckpt_path)
        print(f"\nCheckpoint saved to {ckpt_path}")
    else:
        # Load checkpoint if exists
        if os.path.exists(ckpt_path):
            head.load_state_dict(torch.load(ckpt_path, map_location=args.device))
            print(f"\nLoaded checkpoint from {ckpt_path}")
        else:
            print(
                f"\nWarning: No checkpoint found at {ckpt_path}, using untrained head"
            )

    # Inference
    print("\n" + "=" * 80)
    print("INFERENCE")
    print("=" * 80)

    inference_dir = os.path.join(args.out_dir, "generations")
    run_inference(
        helper,
        head,
        dataset,
        base_embeds.to(args.device, dtype=torch_dtype),
        base_mask,
        neg_embeds.to(args.device, dtype=torch_dtype)
        if neg_embeds is not None
        else None,
        neg_mask,
        inference_dir,
        args.alphas,
        args.gen_seeds,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
    )

    print("\n" + "=" * 80)
    print("POC COMPLETE")
    print("=" * 80)
    print(f"Outputs saved to: {args.out_dir}")
    print(f"  - Checkpoints: {ckpt_dir}")
    print(f"  - Generations: {inference_dir}")


if __name__ == "__main__":
    main()
