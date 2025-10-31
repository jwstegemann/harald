"""
Teacher Inversion Trainer: Learn slot-pure residual embeddings via diffusion noise prediction.

This module implements the core teacher inversion training loop as specified in tasks.md Phase 2.
Key features:
- Learns embedding perturbation P and scale parameter via noise prediction loss
- Hard slot-maskierung: off-slot residuals forced to exactly zero
- RMS normalization + scale clamping for fp16/bf16 stability
- CFG support with fixed negative prompt
- Alpha-QA grid generation for visual validation
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from ..models.qwen_image import QwenImagePipeline
from ..core.slot_tokenizer import SlotConfig
from ..core.metrics import (
    compute_rms,
    compute_slot_off_slot_rms,
    log_metrics_json,
    format_metrics_for_logging,
)
from ..core.shape_validator import validate_device_dtype
from ..core.determinism import seed_all
from ..phase1 import save_grid


class TeacherInversionTrainer:
    """
    Trainer for learning slot-pure teacher embeddings via diffusion noise prediction.

    Training flow:
    1. Initialize learnable parameters P (residual) and log_scale
    2. For each step:
       - Sample batch of images (cached latents)
       - Sample timesteps and noise
       - Compute noisy latents
       - Re-parametrize: E = E_base + scale * RMS_norm(P)
       - Forward through UNet (with optional CFG)
       - Compute loss (MSE + RMS-reg + off-slot-sparsity)
       - Backward + optimize
    3. Post-process: hard slot-maskierung (off-slot → 0)
    4. Save teacher_prompt_embeds.pt
    5. Generate alpha-QA grid
    """

    def __init__(
        self,
        pipeline: QwenImagePipeline,
        base_prompt: str,
        slot_config: SlotConfig,
        device: torch.device,
        dtype: torch.dtype,
        cfg_scale: float = 1.0,
        negative_prompt: str = " ",
    ):
        """
        Initialize teacher inversion trainer.

        Args:
            pipeline: QwenImagePipeline instance
            base_prompt: Base prompt with slot (e.g., "a portrait photo of ~ID~, studio lighting")
            slot_config: Slot configuration (must have slot position 's' set)
            device: Target device (CUDA)
            dtype: Target dtype
            cfg_scale: Classifier-free guidance scale (default: 4.5)
            negative_prompt: Negative prompt (default: " " - empty string)
        """
        self.pipeline = pipeline
        self.base_prompt = base_prompt
        self.slot_config = slot_config
        self.device = device
        self.dtype = dtype
        self.cfg_scale = cfg_scale
        self.negative_prompt = negative_prompt

        # Encode base prompt (fixed throughout training)
        print("Encoding base prompt...")
        self.E_base, self.E_base_mask = pipeline.encode_text(base_prompt)

        # Clone to remove inference mode flag and enable gradient flow
        # Without cloning, tensors retain inference flag from text_encoder
        self.E_base = self.E_base.clone()
        self.E_base_mask = self.E_base_mask.clone()

        # Find slot position in base prompt
        if slot_config.s is None:
            self._find_slot_position()

        self.s = slot_config.s
        self.L = slot_config.L

        # Extract dimensions
        self.B, self.S, self.H = self.E_base.shape
        assert self.B == 1, "Base prompt embeddings must have batch size 1"

        # Encode negative prompt (fixed)
        self.E_neg, self.E_neg_mask = pipeline.encode_text(negative_prompt)

        # Clone to remove inference mode flag
        self.E_neg = self.E_neg.clone()
        self.E_neg_mask = self.E_neg_mask.clone()

        # Pad negative embeddings to match base prompt length if needed
        if self.E_neg.shape[1] != self.E_base.shape[1]:
            B, S_neg, H = self.E_neg.shape
            S_base = self.E_base.shape[1]

            if S_neg < S_base:
                # Pad with zeros to match length
                padding_length = S_base - S_neg
                padding = torch.zeros(B, padding_length, H, device=self.E_neg.device, dtype=self.E_neg.dtype)
                self.E_neg = torch.cat([self.E_neg, padding], dim=1)

                # Pad mask with zeros (indicating padded positions)
                mask_padding = torch.zeros(B, padding_length, device=self.E_neg_mask.device, dtype=self.E_neg_mask.dtype)
                self.E_neg_mask = torch.cat([self.E_neg_mask, mask_padding], dim=1)

                print(f"  ℹ Padded negative prompt embeddings: {S_neg} → {S_base} tokens")
            else:
                # Truncate if longer (unlikely but handle it)
                self.E_neg = self.E_neg[:, :S_base, :]
                self.E_neg_mask = self.E_neg_mask[:, :S_base]
                print(f"  ℹ Truncated negative prompt embeddings: {S_neg} → {S_base} tokens")

        # Final validation
        if self.E_neg.shape != self.E_base.shape:
            print(
                f"ERROR: Negative prompt embeddings shape mismatch after padding.\n"
                f"  E_base:  {self.E_base.shape}\n"
                f"  E_neg:   {self.E_neg.shape}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Enable gradient checkpointing to save memory
        if hasattr(self.pipeline.transformer, "enable_gradient_checkpointing"):
            self.pipeline.transformer.enable_gradient_checkpointing()
            print("  ✓ Enabled gradient checkpointing on transformer")
        elif hasattr(self.pipeline.transformer, "gradient_checkpointing_enable"):
            self.pipeline.transformer.gradient_checkpointing_enable()
            print("  ✓ Enabled gradient checkpointing on transformer")

        print(f"✓ Teacher trainer initialized")
        print(f"  Base prompt:   '{base_prompt}'")
        print(f"  Slot position: s={self.s}, L={self.L}")
        print(f"  Embeddings:    shape={self.E_base.shape}, device={self.device}, dtype={self.dtype}")
        print(f"  CFG scale:     {cfg_scale}")
        print()

    def _pack_latents(
        self, latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Pack latents from [B, C, H, W] to [B, num_patches, C*4] for transformer.

        QwenImage transformer expects packed latents where each 2x2 spatial patch
        is flattened into the channel dimension.

        Args:
            latents: VAE latents [B, C, H, W] (typically C=16)

        Returns:
            Packed latents [B, (H//2)*(W//2), C*4] (typically C*4=64)
        """
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

        batch_size, num_channels, height, width = latents.shape

        return QwenImagePipeline._pack_latents(
            latents,
            batch_size=batch_size,
            num_channels_latents=num_channels,
            height=height,
            width=width,
        )

    def _unpack_latents(
        self, latents: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        Unpack latents from [B, num_patches, 64] to [B, 16, H, W] for VAE.

        Args:
            latents: Packed latents [B, num_patches, 64]
            height: Target height (must be divisible by 8)
            width: Target width (must be divisible by 8)

        Returns:
            Unpacked latents [B, 16, H, W]
        """
        from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

        return QwenImagePipeline._unpack_latents(
            latents,
            height=height,
            width=width,
            vae_scale_factor=8,  # Standard VAE scale factor
        )

    def _find_slot_position(self):
        """Find slot position in base prompt and update slot_config."""
        from ..core.slot_tokenizer import tokenize_and_find_slot

        _, positions = tokenize_and_find_slot(
            self.pipeline.tokenizer,
            self.base_prompt,
            self.slot_config,
        )

        if len(positions) != 1:
            print(
                f"ERROR: Base prompt must contain exactly one slot occurrence.\n"
                f"  Found {len(positions)} occurrences at positions {positions}",
                file=sys.stderr,
            )
            sys.exit(1)

        self.slot_config.s = positions[0]
        print(f"  ✓ Slot position found: s={self.slot_config.s}")

    def prepare_images(
        self,
        image_paths: List[Path],
        target_size: Tuple[int, int] = (1024, 1024),
    ) -> torch.Tensor:
        """
        Prepare and cache image latents for training.

        Args:
            image_paths: List of image file paths (2-5 images)
            target_size: Target image size (width, height)

        Returns:
            Tensor of cached latents [N, C, H_lat, W_lat]

        Notes:
            - Loads images, center crops, resizes
            - Encodes to latents via VAE
            - Caches on GPU to avoid re-encoding each step
        """
        if not (2 <= len(image_paths) <= 5):
            print(
                f"WARNING: Expected 2-5 images, got {len(image_paths)}. Continuing anyway.",
                file=sys.stderr,
            )

        print(f"Preparing {len(image_paths)} images...")
        images = []

        for img_path in image_paths:
            if not img_path.exists():
                print(
                    f"ERROR: Image not found: {img_path}",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Load and preprocess
            img = Image.open(img_path).convert("RGB")

            # Center crop to square
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))

            # Resize
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(img)

        # Store image dimensions for img_shapes calculation in forward_step()
        # target_size is (width, height) tuple
        self.image_width = target_size[0]
        self.image_height = target_size[1]

        # Convert to tensors
        import torchvision.transforms.functional as TF

        pixel_values = torch.stack([TF.to_tensor(img) for img in images])  # [N, 3, H, W]
        pixel_values = pixel_values * 2.0 - 1.0  # Normalize to [-1, 1]
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        # Add frame dimension for Qwen-Image video VAE: [N, 3, H, W] → [N, 3, 1, H, W]
        # Qwen-Image VAE expects 5D tensor (batch, channels, frames, height, width)
        pixel_values = pixel_values.unsqueeze(2)  # Insert frame dimension at position 2

        # Encode to latents
        print("Encoding images to latents...")
        with torch.no_grad():
            encoder_output = self.pipeline.vae.encode(pixel_values, return_dict=True)
            latents = encoder_output.latent_dist.sample()
            # Note: Qwen-Image VAE uses latents_mean/latents_std, not scaling_factor
            # Latents are already normalized internally, no external scaling needed

            print(f"  Raw latents shape (5D): {latents.shape}")

            # Remove frame dimension: [N, C, F, H, W] → [N, C, H, W]
            # VAE outputs 5D tensor with frame dimension, but we need 4D for packing
            latents = latents.squeeze(2)
            print(f"  Squeezed latents shape (4D): {latents.shape}")

            # Pack latents for transformer: [N, C, H, W] → [N, num_patches, C*4]
            # QwenImage transformer expects packed 2x2 patches, not spatial latents
            latents = self._pack_latents(latents)

        print(f"  ✓ Cached {len(latents)} packed latents: {latents.shape}")
        return latents

    def reparametrize(
        self,
        P: torch.Tensor,
        log_scale: torch.nn.Parameter,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-parametrize raw residual P to stabilized residual R with hard slot masking.

        Args:
            P: Raw learnable residual [1, S, H]
            log_scale: Log-scale parameter (scalar)
            eps: Epsilon for numerical stability

        Returns:
            Tuple of (E, R):
                - E: Modified embeddings [1, S, H] = E_base + R
                - R: Slot-pure residual [1, S, H] with off-slot exactly zero

        Notes:
            - Hard slot mask applied BEFORE normalization (gradients never flow to off-slot)
            - RMS normalization on slot-only part prevents explosion/collapse in fp16/bf16
            - Scale is clamped to [0.01, 2.0] for stability
        """
        # 1) Hard slot mask - zero out off-slot positions
        slot_mask = torch.zeros_like(P)
        slot_mask[:, self.s : self.s + self.L, :] = 1.0

        # 2) Only keep slot parameters (off-slot becomes zero, no gradients flow there)
        P_slot = P * slot_mask

        # 3) RMS normalization on slot part only
        rms = compute_rms(P_slot, dim=None, eps=eps)
        P_hat = P_slot / (rms + eps)

        # 4) Apply scale
        scale = torch.exp(log_scale).clamp(0.01, 2.0)
        R = scale * P_hat

        # 5) Safety: ensure off-slot stays exactly zero
        R = R * slot_mask

        # Modified embeddings
        E = self.E_base + R

        return E, R

    def sample_timesteps(
        self,
        batch_size: int,
        weighting_scheme: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample timesteps for Flow Matching training (direct computation).

        Args:
            batch_size: Number of timesteps to sample
            weighting_scheme: Sampling distribution ("uniform", "logit_normal", "mode")
            logit_mean: Mean for logit_normal distribution (default: 0.0)
            logit_std: Std for logit_normal distribution (default: 1.0)

        Returns:
            Timestep values [batch_size] for transformer input

        Notes:
            - Samples u ∈ [0,1] and computes sigmas directly
            - Restricts to central 85% of range (u ∈ [0.05, 0.9]) for stability
            - Does NOT depend on scheduler.timesteps (which may be in inference mode)
            - Returns timestep values for transformer, not sigma values
            - Wider range than 70% to capture low-noise details for small concepts
        """
        # Sample u values from [0, 1] according to distribution
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=None,
        )
        u = u.to(device=self.device)

        # Restrict to central 85% of timestep range (avoid extreme ends)
        # Maps [0, 1] → [0.05, 0.9]
        u = 0.05 + u * 0.85

        # Compute sigmas directly from u (Flow Matching formula)
        # u=0.05 → sigma=0.95 (high noise), u=0.9 → sigma=0.1 (low noise)
        sigmas = 1.0 - u

        # Apply shift correction if configured
        shift = self.pipeline.scheduler.config.shift
        if shift != 1.0:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        # Convert sigmas to timestep values for transformer
        num_train_timesteps = self.pipeline.scheduler.config.num_train_timesteps
        timesteps = sigmas * num_train_timesteps

        return timesteps

    def get_sigmas(
        self,
        timesteps: torch.Tensor,
        n_dim: int = 4,
    ) -> torch.Tensor:
        """
        Compute sigma values from timestep values (direct formula).

        This is the CORRECT pattern for Flow Matching training - compute sigmas
        directly from timestep values using the mathematical formula, NOT by
        looking up in scheduler arrays.

        Args:
            timesteps: Timestep VALUES [batch_size] (sigma * num_train_timesteps)
            n_dim: Number of dimensions to broadcast to (default: 4 for latents)

        Returns:
            Sigmas tensor broadcastable to latent shape

        Notes:
            - Sigmas are used for noise computation and loss weighting
            - CRITICAL: Direct computation, independent of scheduler state
            - Works regardless of whether set_timesteps() was called
        """
        # Get scheduler config
        num_train_timesteps = self.pipeline.scheduler.config.num_train_timesteps

        # Convert timestep values to sigmas using direct formula
        # timestep = sigma * num_train_timesteps → sigma = timestep / num_train_timesteps
        sigmas = timesteps.to(dtype=self.dtype, device=self.device) / num_train_timesteps

        # Note: We don't need to invert the shift transformation here because
        # timesteps already encode the shifted sigmas (computed in sample_timesteps)

        # Broadcast to n_dim (e.g., [B] -> [B, 1, 1, 1] for 4D latents)
        while len(sigmas.shape) < n_dim:
            sigmas = sigmas.unsqueeze(-1)

        return sigmas

    def compute_loss_weights(
        self,
        sigmas: torch.Tensor,
        weighting_scheme: str = "cosmap",
    ) -> torch.Tensor:
        """
        Compute per-timestep loss weights using SD3/Qwen-Image approach.

        Args:
            sigmas: Sigma values [batch_size, 1, 1, 1]
            weighting_scheme: Weighting scheme ("none", "cosmap", "sigma_sqrt")

        Returns:
            Loss weights tensor (same shape as sigmas)

        Notes:
            - "none": Uniform weighting (all timesteps equal)
            - "cosmap": Cosine mapping from SD3 paper (recommended)
            - "sigma_sqrt": Inverse variance weighting
        """
        import math

        if weighting_scheme == "sigma_sqrt":
            # Inverse variance weighting (emphasizes low-noise timesteps)
            # Add epsilon for numerical stability (prevents division by zero)
            weighting = (sigmas**2 + 1e-5)**-1.0
        elif weighting_scheme == "cosmap":
            # Cosine-based weighting from SD3 paper
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
        else:
            # "none" - uniform weighting
            weighting = torch.ones_like(sigmas)

        return weighting

    def forward_step(
        self,
        E: torch.Tensor,
        E_mask: torch.Tensor,
        latents_batch: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through diffusion model with optional CFG.

        Args:
            E: Modified prompt embeddings [1, S, H]
            E_mask: Prompt embeddings mask [1, S]
            latents_batch: Batch of latents [batch, C, H_lat, W_lat]
            timesteps: Timestep values [batch] from scheduler (for transformer)
            noise: Noise tensor (same shape as latents_batch)

        Returns:
            Tuple of (pred, target):
                - pred: Transformer prediction (velocity field for Flow Matching)
                - target: Ground truth velocity (noise - clean)

        Notes:
            - Uses manual noisy latent computation (Flow Matching training pattern)
            - Does NOT use scheduler.scale_noise() (avoids CUDA errors)
            - Implements CFG if cfg_scale > 1.0
            - Computes sigmas internally using official lookup pattern
        """
        batch_size = latents_batch.shape[0]

        # Compute sigmas from timesteps using official lookup pattern
        sigmas = self.get_sigmas(timesteps, n_dim=latents_batch.ndim)

        # Compute noisy latents manually (Flow Matching formula)
        # This is the official training pattern - do NOT use scheduler.scale_noise()
        noisy_latents = (1.0 - sigmas) * latents_batch + sigmas * noise

        # Expand embeddings to batch size
        E_expanded = E.expand(batch_size, -1, -1)
        E_mask_expanded = E_mask.expand(batch_size, -1)

        # Clone to ensure fresh tensors without inference mode flag
        # expand() creates views which may carry forward inference flags from operations
        E_expanded = E_expanded.clone()
        E_mask_expanded = E_mask_expanded.clone()

        # Compute transformer metadata (required for RoPE position embeddings)
        # txt_seq_lens: use full padded sequence length for consistent RoPE frequencies in CFG
        txt_seq_lens = [E_expanded.shape[1]] * batch_size

        # img_shapes: (frame, height_latent, width_latent) per batch item
        # Formula: original_size → VAE (÷8) → pack (÷2)
        # Example: 768×512 → 96×64 → 48×32 latent patches
        height_latent = self.image_height // 8 // 2  # VAE scale factor = 8, packing = 2
        width_latent = self.image_width // 8 // 2
        img_shapes = [[(1, height_latent, width_latent)]] * batch_size

        # Transformer forward (conditional)
        # Use enable_grad() to override any inference mode contexts from diffusers
        with torch.enable_grad():
            model_pred_cond = self.pipeline.transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=E_expanded,
                encoder_hidden_states_mask=E_mask_expanded,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]

        # Optional CFG
        if self.cfg_scale > 1.0:
            # Unconditional prediction
            E_neg_expanded = self.E_neg.expand(batch_size, -1, -1)
            E_neg_mask_expanded = self.E_neg_mask.expand(batch_size, -1)

            # Clone for CFG to avoid inference mode issues
            E_neg_expanded = E_neg_expanded.clone()
            E_neg_mask_expanded = E_neg_mask_expanded.clone()

            # Compute txt_seq_lens for negative prompt (use full padded length for CFG consistency)
            txt_seq_lens_neg = [E_neg_expanded.shape[1]] * batch_size

            # Use enable_grad() for unconditional pass as well
            with torch.enable_grad():
                model_pred_uncond = self.pipeline.transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=E_neg_expanded,
                    encoder_hidden_states_mask=E_neg_mask_expanded,
                    img_shapes=img_shapes,  # Same as conditional (same latent shape)
                    txt_seq_lens=txt_seq_lens_neg,
                    return_dict=False,
                )[0]

            # CFG combination
            model_pred = model_pred_uncond + self.cfg_scale * (model_pred_cond - model_pred_uncond)
        else:
            model_pred = model_pred_cond

        # Flow Matching target: velocity = noise - clean
        # (NOT epsilon like DDPM - Flow Matching always predicts velocity field)
        target = noise - latents_batch

        return model_pred, target

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        R: torch.Tensor,
        sigmas: Optional[torch.Tensor] = None,
        weighting_scheme: str = "cosmap",
        loss_weights: Dict[str, float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted loss with MSE, RMS-regularization, and off-slot sparsity.

        Args:
            pred: Model prediction
            target: Ground truth
            R: Current residual [1, S, H]
            sigmas: Sigma values for timestep weighting [batch, 1, 1, 1]
            weighting_scheme: Weighting scheme for MSE loss ("none", "cosmap", "sigma_sqrt")
            loss_weights: Dict with keys "mse", "rms", "offslot" (default: {1.0, 0.1, 1.0})

        Returns:
            Dict with all loss components and total loss

        Notes:
            - MSE: main noise prediction loss (weighted by timestep/sigma)
            - RMS: light regularization (weight=0.1) - prevents scale extremes
            - Off-slot: forces residual to zero outside slot
        """
        if loss_weights is None:
            loss_weights = {"mse": 1.0, "rms": 0.1, "offslot": 1.0}

        # MSE loss (main) with optional timestep weighting
        if sigmas is not None and weighting_scheme != "none":
            # Qwen-Image / SD3 approach: weight by sigma to prevent late timesteps from dominating
            loss_weighting = self.compute_loss_weights(sigmas, weighting_scheme)

            # Compute per-sample weighted MSE
            mse_per_sample = (
                (loss_weighting.float() * (pred.float() - target.float()) ** 2)
                .reshape(pred.shape[0], -1)
                .mean(dim=1)
            )
            loss_mse = mse_per_sample.mean()
        else:
            # Uniform weighting (original behavior)
            loss_mse = F.mse_loss(pred, target, reduction="mean")

        # RMS regularization (slot only)
        R_slot = R[0, self.s : self.s + self.L, :]  # [L, H]
        rms_slot = compute_rms(R_slot)

        # Target RMS 0.7 (lower than base to preserve prompt semantics)
        target_rms = 0.7
        loss_rms = F.l1_loss(rms_slot, torch.tensor(target_rms, device=self.device))

        # Off-slot sparsity (force to zero)
        R_2d = R[0]  # [S, H]
        R_off_slot_before = R_2d[: self.s, :]
        R_off_slot_after = R_2d[self.s + self.L :, :]

        if R_off_slot_before.numel() > 0 or R_off_slot_after.numel() > 0:
            R_off_slot = torch.cat([R_off_slot_before, R_off_slot_after], dim=0)
            loss_offslot = torch.mean(R_off_slot**2)
            rms_off = compute_rms(R_off_slot)
        else:
            loss_offslot = torch.tensor(0.0, device=self.device)
            rms_off = torch.tensor(0.0, device=self.device)

        # Total weighted loss
        total_loss = (
            loss_weights["mse"] * loss_mse
            + loss_weights["rms"] * loss_rms
            + loss_weights["offslot"] * loss_offslot
        )

        return {
            "total": total_loss,
            "mse": loss_mse,
            "rms": loss_rms,
            "offslot": loss_offslot,
            "slot_rms": rms_slot,
            "off_rms": rms_off,
        }

    def load_and_parse_prompt(self, image_paths: List[Path]) -> Optional[str]:
        """
        Load prompt from first photo_view image and extract person description.

        Prompts follow the pattern:
        "portrait single subject. upper-body shot of [PERSON_DESC]. (perfect..."

        Returns:
            Person-specific description or None if not found
        """
        import re

        # Find first photo_view_*.txt file
        image_dir = image_paths[0].parent
        prompt_file = image_dir / "photo_view_01.txt"

        if not prompt_file.exists():
            # Fallback: Try any photo_view_*.txt
            txt_files = sorted(image_dir.glob("photo_view_*.txt"))
            if txt_files:
                prompt_file = txt_files[0]
            else:
                return None

        with open(prompt_file, 'r') as f:
            full_prompt = f.read().strip()

        # Extract person description between "shot of " and ". (perfect"
        match = re.search(r'shot of (.+?)\s*\.\s*\(perfect', full_prompt, re.DOTALL)
        if match:
            person_desc = match.group(1).strip()
            return person_desc

        # Fallback: If pattern doesn't match, return None
        print(f"  Warning: Could not parse prompt pattern from: {full_prompt[:100]}...")
        return None

    def compute_p_init_from_description(self, description: str) -> torch.Tensor:
        """
        Compute initial P from person description.

        Args:
            description: Person-specific part, e.g. "60yo woman, blonde hair, ..."

        Returns:
            P_init [1, S, H] - initialized parameter
        """
        # Encode the person description only
        E_desc, E_desc_mask = self.pipeline.encode_text(description)
        E_desc = E_desc.to(self.device, dtype=self.dtype)

        # Get description length (actual tokens, not padding)
        desc_len = E_desc_mask.sum().item()

        if desc_len == 0:
            # Fallback: zero init
            return torch.zeros(1, self.S, self.H, device=self.device, dtype=self.dtype)

        # Average multiple token groups for robustness
        groups = []

        # First tokens (often subject: "60yo woman")
        if desc_len >= 2:
            groups.append(E_desc[0, :min(4, desc_len), :].mean(dim=0, keepdim=True))

        # Middle tokens (often key features)
        if desc_len >= 6:
            mid_start = max(0, (desc_len - 4) // 2)
            mid_end = min(desc_len, mid_start + 4)
            groups.append(E_desc[0, mid_start:mid_end, :].mean(dim=0, keepdim=True))

        # Last tokens (often detailed features)
        if desc_len >= 2:
            last_start = max(0, desc_len - 4)
            groups.append(E_desc[0, last_start:desc_len, :].mean(dim=0, keepdim=True))

        # Average all groups to get representative embedding for each slot token
        if groups:
            E_slot_init = torch.cat(groups, dim=0).mean(dim=0)  # [H]
            # Expand to L tokens
            E_slot_init = E_slot_init.unsqueeze(0).expand(self.L, -1)  # [L, H]
        else:
            return torch.zeros(1, self.S, self.H, device=self.device, dtype=self.dtype)

        # Compute residual at slot position
        E_base_slot = self.E_base[0, self.s:self.s+self.L, :]  # [L, H]
        R_slot = E_slot_init - E_base_slot  # [L, H]

        # Invert reparametrization: R → P
        # Forward: P_slot → normalize(RMS=1) → scale → R
        # Backward: R → un-scale → un-normalize → P_slot

        scale_init = 1.0  # Will be learned
        P_hat_slot = R_slot / scale_init

        # Scale to target RMS (0.7)
        rms_init = compute_rms(P_hat_slot)
        if rms_init > 1e-8:
            P_slot = P_hat_slot * (0.7 / rms_init)
        else:
            P_slot = P_hat_slot

        # Create full P tensor (zeros everywhere except slot)
        P_init = torch.zeros(1, self.S, self.H, device=self.device, dtype=self.dtype)
        P_init[0, self.s:self.s+self.L, :] = P_slot

        return P_init

    def train_identity(
        self,
        image_paths: List[Path],
        steps: int = 600,
        lr: float = 1e-4,
        batch_size: int = 2,
        grad_accum_steps: int = 4,
        grad_clip: float = 5.0,
        loss_weights: Optional[Dict[str, float]] = None,
        timestep_sampling_scheme: str = "logit_normal",
        loss_weighting_scheme: str = "cosmap",
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Train teacher embeddings for a single identity.

        Args:
            image_paths: List of 2-5 image paths for this identity
            steps: Number of optimization steps (default: 600)
            lr: Learning rate (default: 1e-4)
            batch_size: Batch size (default: 2)
            grad_accum_steps: Gradient accumulation steps (default: 4)
            grad_clip: Gradient clipping threshold (default: 5.0)
            loss_weights: Optional custom loss weights
            timestep_sampling_scheme: Timestep sampling ("uniform", "logit_normal", "mode") (default: "logit_normal")
            loss_weighting_scheme: Loss weighting ("none", "cosmap", "sigma_sqrt") (default: "cosmap")
            output_dir: Output directory for intermediate QA grids (required for step 100 checkpoint)

        Returns:
            Dict with training metrics and final parameters

        Raises:
            SystemExit: On NaN/Inf or divergence

        Notes:
            - Uses Qwen-Image/SD3 approach for timestep sampling and loss weighting
            - "logit_normal" + "cosmap" is the recommended combination (official Qwen training)
            - Loss weighting prevents late (high-noise) timesteps from dominating gradients
        """
        # Prepare latents (cached)
        latents = self.prepare_images(image_paths)
        num_images = len(latents)

        # Initialize learnable parameters
        # Try to initialize P from person description in prompt
        person_desc = self.load_and_parse_prompt(image_paths)
        if person_desc is not None:
            print(f"  Initializing P from prompt: '{person_desc[:60]}{'...' if len(person_desc) > 60 else ''}'")
            P_init = self.compute_p_init_from_description(person_desc)
            P = torch.nn.Parameter(P_init.detach().clone()).requires_grad_(True)
        else:
            print("  No prompt found, using zero initialization")
            P = torch.nn.Parameter(torch.zeros(1, self.S, self.H, device=self.device, dtype=self.dtype)).requires_grad_(True)

        log_scale = torch.nn.Parameter(torch.tensor(0.0, device=self.device, dtype=self.dtype))

        # Verify parameters are trainable
        print(f"  P.requires_grad: {P.requires_grad}")
        print(f"  log_scale.requires_grad: {log_scale.requires_grad}")

        # Optimizer
        optimizer = torch.optim.AdamW([P, log_scale], lr=lr)

        # Learning rate scheduler with warmup
        warmup_steps = int(steps * 0.15)  # 15% warmup (90 steps for 600 total)

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps  # Linear warmup from 0 to 1
            return 1.0  # Constant after warmup

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda)

        # Training loop
        print(f"Training for {steps} steps...")
        print(f"  Images: {num_images}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {grad_accum_steps}")
        print(f"  Learning rate: {lr}")
        print(f"  Timestep sampling: {timestep_sampling_scheme}")
        print(f"  Loss weighting: {loss_weighting_scheme}")
        print()

        loss_history = []
        best_loss = float("inf")
        metrics_log = []  # Detailed metrics every 10 steps

        pbar = tqdm(range(steps), desc="Training")

        for step in pbar:
            # Zero gradients once per accumulation
            optimizer.zero_grad()
            accumulated_loss = 0.0
            accumulated_losses = {"mse": 0.0, "rms": 0.0, "offslot": 0.0, "slot_rms": 0.0, "off_rms": 0.0}

            # Accumulate gradients over multiple micro-batches
            for k in range(grad_accum_steps):
                # Sample micro-batch
                idx = torch.randperm(num_images)[:batch_size]
                latents_batch = latents[idx]

                # Sample timesteps using Qwen-Image/SD3 approach (official pattern)
                timesteps = self.sample_timesteps(
                    batch_size=batch_size,
                    weighting_scheme=timestep_sampling_scheme,
                )

                # Compute sigmas for loss weighting (official lookup pattern)
                sigmas = self.get_sigmas(timesteps, n_dim=latents_batch.ndim)

                # Sample noise
                noise = torch.randn_like(latents_batch)

                # Re-parametrize
                E, R = self.reparametrize(P, log_scale)

                # Forward (computes sigmas internally with official lookup pattern)
                pred, target = self.forward_step(
                    E,
                    self.E_base_mask,
                    latents_batch,
                    timesteps,
                    noise,
                )

                # Compute loss with timestep weighting
                losses = self.compute_loss(
                    pred, target, R, sigmas, loss_weighting_scheme, loss_weights
                )
                loss = losses["total"]

                # Scale loss by accumulation steps and backward
                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

                # Accumulate losses for logging
                accumulated_loss += loss.item()
                for key in accumulated_losses:
                    accumulated_losses[key] += losses[key].item()

            # NOTE: No gradient normalization needed here!
            # The losses are computed with reduction="mean", so gradients are already averaged.
            # The expand+clone pattern DOES sum gradients, BUT each individual gradient is
            # already divided by batch_size (because loss is averaged), so they cancel out.
            # Previous incorrect normalization: P.grad.div_(batch_size) made gradients too small!

            # Average accumulated losses
            avg_loss = accumulated_loss / grad_accum_steps
            for key in accumulated_losses:
                accumulated_losses[key] /= grad_accum_steps

            # Compute gradient norm BEFORE clipping (honest measurement)
            total_norm = torch.norm(torch.stack([
                torch.norm(P.grad.detach(), 2),
                torch.norm(log_scale.grad.detach(), 2),
            ]), 2).item()

            # Gradient clipping (only clip once)
            torch.nn.utils.clip_grad_norm_([P, log_scale], grad_clip)
            was_clipped = total_norm > grad_clip

            # Single optimizer step per accumulation
            optimizer.step()
            scheduler.step()  # Update learning rate

            # Log detailed metrics every 10 steps
            if step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                current_scale = torch.exp(log_scale).clamp(0.01, 2.0).item()

                metric_entry = {
                    "step": step,
                    "loss_total": avg_loss,
                    "loss_mse": accumulated_losses["mse"],
                    "loss_rms": accumulated_losses["rms"],
                    "loss_offslot": accumulated_losses["offslot"],
                    "slot_rms": accumulated_losses["slot_rms"],
                    "off_rms": accumulated_losses["off_rms"],
                    "scale": current_scale,
                    "lr": current_lr,
                    "grad_norm": total_norm,
                    "was_clipped": bool(was_clipped),
                }
                metrics_log.append(metric_entry)

            # Check for NaN/Inf (use averaged loss)
            if torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss)):
                print(
                    f"\nERROR: NaN or Inf detected at step {step}.\n"
                    f"  Loss: {avg_loss:.4f}\n"
                    f"  Scale: {torch.exp(log_scale).item()}\n"
                    "  Try lowering LR or CFG scale.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Track metrics
            loss_history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            # Update progress bar (use accumulated RMS metrics)
            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "scale": f"{torch.exp(log_scale).item():.3f}",
                    "slot_rms": f"{accumulated_losses['slot_rms']:.3f}",
                    "off_rms": f"{accumulated_losses['off_rms']:.6f}",
                }
            )

            # Generate intermediate QA grid at step 100
            if step == 100:
                print(f"\n  Generating intermediate QA grid (step {step})...")
                with torch.no_grad():
                    E_checkpoint, _ = self.reparametrize(P, log_scale)
                    E_checkpoint = E_checkpoint.detach()

                # Use output directory for intermediate QA
                if output_dir is None:
                    raise ValueError("output_dir must be provided for intermediate QA generation at step 100")
                qa_dir = output_dir / f"qa_step{step}"
                qa_dir.mkdir(exist_ok=True, parents=True)

                # Generate QA grid with reduced seeds for speed
                self.alpha_qa(
                    E_teacher=E_checkpoint,
                    alphas=[0.5, 1.0, 1.5, 2.0],
                    seeds=[4, 5],  # Reduced to 2 seeds (8 images instead of 16)
                    output_dir=qa_dir,
                )

                # Cleanup
                del E_checkpoint
                torch.cuda.empty_cache()
                print(f"  ✓ Intermediate QA saved to {qa_dir / 'alpha_qa_grid.png'}\n")

            # Check for plateau (early stopping)
            if step > 100 and step % 100 == 0:
                recent_losses = loss_history[-100:]
                if max(recent_losses) - min(recent_losses) < 0.01 * best_loss:
                    print(f"\n✓ Loss plateaued at step {step}. Stopping early.")
                    break

            # Check for divergence
            if step > 50:
                recent_losses = loss_history[-50:]
                if all(recent_losses[i] > recent_losses[i - 1] for i in range(1, len(recent_losses))):
                    print(
                        f"\nERROR: Loss diverging (increasing for 50 consecutive steps).\n"
                        f"  Current loss: {avg_loss:.4f}\n"
                        f"  Best loss: {best_loss:.4f}\n"
                        "  Try halving LR or reducing CFG.",
                        file=sys.stderr,
                    )
                    sys.exit(1)

        print(f"\n✓ Training complete. Best loss: {best_loss:.4f}")
        print()

        # Final metrics
        with torch.no_grad():
            E_final, R_final = self.reparametrize(P, log_scale)
            final_rms = compute_slot_off_slot_rms(R_final, self.s, self.L)

        metrics = {
            "steps_completed": step + 1,
            "final_loss": avg_loss,
            "best_loss": best_loss,
            "final_scale": torch.exp(log_scale).item(),
            **final_rms,
        }

        # Return parameters and metrics
        return {
            "P": P.detach(),
            "log_scale": log_scale.detach(),
            "E_final": E_final.detach(),
            "R_final": R_final.detach(),
            "metrics": metrics,
            "metrics_log": metrics_log,  # Detailed metrics every 10 steps
        }

    def post_process_and_save(
        self,
        training_result: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        """
        Post-process: hard slot-maskierung and save teacher embeddings.

        Args:
            training_result: Result dict from train_identity()
            output_path: Path to save teacher_prompt_embeds.pt

        Returns:
            Path to saved file

        Notes:
            - Sets off-slot residual exactly to zero (hard maskierung)
            - Validates off-slot RMS < 1e-6
            - Saves with metadata
        """
        print("Post-processing: Hard slot maskierung...")

        E_final = training_result["E_final"]
        R_final = training_result["R_final"]

        # Compute delta
        Delta = E_final - self.E_base  # [1, S, H]

        # Hard maskierung: set off-slot to exactly zero
        Delta_masked = Delta.clone()
        Delta_masked[0, : self.s, :] = 0.0
        Delta_masked[0, self.s + self.L :, :] = 0.0

        # Construct teacher embeddings
        E_teacher = self.E_base + Delta_masked

        # Validate off-slot RMS
        rms_check = compute_slot_off_slot_rms(Delta_masked, self.s, self.L)

        if rms_check["off_slot_rms"] > 1e-6:
            print(
                f"WARNING: Off-slot RMS after maskierung > 1e-6: {rms_check['off_slot_rms']:.8f}",
                file=sys.stderr,
            )

        print(f"  ✓ Off-slot RMS: {rms_check['off_slot_rms']:.8f} (target: <1e-6)")
        print(f"  ✓ Slot RMS:     {rms_check['slot_rms']:.4f}")
        print()

        # Save to CPU
        E_teacher_cpu = E_teacher.cpu()

        # Metadata
        metadata = {
            "base_prompt": self.base_prompt,
            "slot_string": self.slot_config.slot_string,
            "s": self.s,
            "L": self.L,
            "S": self.S,
            "H": self.H,
            "dtype": str(self.dtype),
            "cfg_scale": self.cfg_scale,
            **training_result["metrics"],
        }

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "teacher_prompt_embeds": E_teacher_cpu,
                "metadata": metadata,
            },
            output_path,
        )

        print(f"✓ Saved teacher embeddings to {output_path}")
        print(f"  Shape: {E_teacher_cpu.shape}")
        print(f"  dtype: {E_teacher_cpu.dtype}")
        print()

        # Save detailed training metrics log
        if "metrics_log" in training_result and len(training_result["metrics_log"]) > 0:
            metrics_log_path = output_path.parent / "training_metrics.json"
            with open(metrics_log_path, "w") as f:
                json.dump(training_result["metrics_log"], f, indent=2)
            print(f"✓ Saved training metrics to {metrics_log_path}")
            print(f"  Entries: {len(training_result['metrics_log'])}")
            print()

        return output_path

    def alpha_qa(
        self,
        E_teacher: torch.Tensor,
        alphas: List[float] = [0.5, 1.0, 1.5, 2.0],
        seeds: List[int] = list(range(4, 20)),
        output_dir: Path = None,
        num_inference_steps: int = 30,
        true_cfg_scale: float = 2.5,
        height: int = 1024,
        width: int = 1024,
    ) -> Path:
        """
        Generate alpha-QA grid for visual validation.

        Args:
            E_teacher: Teacher embeddings [1, S, H]
            alphas: List of alpha values to test
            seeds: List of seeds
            output_dir: Output directory
            num_inference_steps: Diffusion steps
            true_cfg_scale: Traditional CFG scale (2.5 recommended for inference)
            height: Image height
            width: Image width

        Returns:
            Path to grid PNG

        Notes:
            - E_eval(α) = E_base + α * (E_teacher - E_base)
            - Grid rows = alphas, cols = seeds
            - Uses traditional classifier-free guidance (true_cfg_scale)
            - Wrapper converts to pipeline's true_cfg_scale parameter
        """
        print("Generating alpha-QA grid...")
        print(f"  Alphas: {alphas}")
        print(f"  Seeds: {len(seeds)} seeds")
        print()

        E_teacher = E_teacher.to(self.device, dtype=self.dtype)

        # Encode negative prompt (required for CFG)
        neg_embeds, neg_mask = self.pipeline.encode_negative_prompt(negative_prompt=" ")

        all_images = []

        for alpha in alphas:
            print(f"  Alpha {alpha}...")
            row_images = []

            for seed in seeds:
                seed_all(seed)

                # Interpolate
                E_eval = self.E_base + alpha * (E_teacher - self.E_base)

                # Generate
                img = self.pipeline.generate(
                    prompt_embeds=E_eval,
                    prompt_embeds_mask=self.E_base_mask,
                    negative_prompt_embeds=neg_embeds,
                    negative_prompt_embeds_mask=neg_mask,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=true_cfg_scale,
                    height=height,
                    width=width,
                    seed=seed,
                )

                row_images.append(img)

            all_images.extend(row_images)

        # Save grid
        grid_path = output_dir / "alpha_qa_grid.png"
        save_grid(all_images, rows=len(alphas), cols=len(seeds), out_path=grid_path)

        # Save metadata
        meta = {
            "alphas": alphas,
            "seeds": seeds,
            "base_prompt": self.base_prompt,
            "slot_string": self.slot_config.slot_string,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "height": height,
            "width": width,
        }

        meta_path = output_dir / "alpha_qa_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✓ Saved alpha-QA grid to {grid_path}")
        print(f"✓ Saved metadata to {meta_path}")
        print()

        return grid_path
