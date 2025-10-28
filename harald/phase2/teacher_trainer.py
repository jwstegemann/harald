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
        cfg_scale: float = 4.5,
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

        # Validate shapes
        if self.E_neg.shape != self.E_base.shape:
            print(
                f"ERROR: Negative prompt embeddings shape mismatch.\n"
                f"  E_base:  {self.E_base.shape}\n"
                f"  E_neg:   {self.E_neg.shape}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"✓ Teacher trainer initialized")
        print(f"  Base prompt:   '{base_prompt}'")
        print(f"  Slot position: s={self.s}, L={self.L}")
        print(f"  Embeddings:    shape={self.E_base.shape}, device={self.device}, dtype={self.dtype}")
        print(f"  CFG scale:     {cfg_scale}")
        print()

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
        target_size: Tuple[int, int] = (768, 512),
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

        # Convert to tensors
        import torchvision.transforms.functional as TF

        pixel_values = torch.stack([TF.to_tensor(img) for img in images])  # [N, 3, H, W]
        pixel_values = pixel_values * 2.0 - 1.0  # Normalize to [-1, 1]
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        # Encode to latents
        print("Encoding images to latents...")
        with torch.no_grad():
            latents = self.pipeline.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.pipeline.vae.config.scaling_factor

        print(f"  ✓ Cached {len(latents)} latents: {latents.shape}")
        return latents

    def reparametrize(
        self,
        P: torch.Tensor,
        log_scale: torch.nn.Parameter,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Re-parametrize raw residual P to stabilized residual R.

        Args:
            P: Raw learnable residual [1, S, H]
            log_scale: Log-scale parameter (scalar)
            eps: Epsilon for numerical stability

        Returns:
            Tuple of (E, R):
                - E: Modified embeddings [1, S, H] = E_base + R
                - R: Stabilized residual [1, S, H] = scale * RMS_norm(P)

        Notes:
            - RMS normalization prevents explosion/collapse in fp16/bf16
            - Scale is clamped to [0.01, 1.5] for stability
        """
        # RMS normalization
        rms = compute_rms(P, dim=None, eps=eps)
        P_hat = P / (rms + eps)

        # Scale with clamp
        scale = torch.exp(log_scale).clamp(0.01, 1.5)

        # Stabilized residual
        R = scale * P_hat

        # Modified embeddings
        E = self.E_base + R

        return E, R

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
            timesteps: Timesteps [batch]
            noise: Noise tensor (same shape as latents_batch)

        Returns:
            Tuple of (pred, target):
                - pred: UNet prediction (noise or v)
                - target: Ground truth (noise or v)

        Notes:
            - Handles scheduler's prediction_type (epsilon vs. v_prediction)
            - Implements CFG if cfg_scale > 1.0
        """
        batch_size = latents_batch.shape[0]

        # Add noise to latents
        noisy_latents = self.pipeline.scheduler.add_noise(latents_batch, noise, timesteps)

        # Expand embeddings to batch size
        E_expanded = E.expand(batch_size, -1, -1)
        E_mask_expanded = E_mask.expand(batch_size, -1)

        # Transformer forward (conditional)
        model_pred_cond = self.pipeline.transformer(
            noisy_latents,
            timesteps,
            encoder_hidden_states=E_expanded,
            encoder_attention_mask=E_mask_expanded,
            return_dict=False,
        )[0]

        # Optional CFG
        if self.cfg_scale > 1.0:
            # Unconditional prediction
            E_neg_expanded = self.E_neg.expand(batch_size, -1, -1)
            E_neg_mask_expanded = self.E_neg_mask.expand(batch_size, -1)

            model_pred_uncond = self.pipeline.transformer(
                noisy_latents,
                timesteps,
                encoder_hidden_states=E_neg_expanded,
                encoder_attention_mask=E_neg_mask_expanded,
                return_dict=False,
            )[0]

            # CFG combination
            model_pred = model_pred_uncond + self.cfg_scale * (model_pred_cond - model_pred_uncond)
        else:
            model_pred = model_pred_cond

        # Determine target based on scheduler's prediction type
        if self.pipeline.scheduler.config.prediction_type == "v_prediction":
            target = self.pipeline.scheduler.get_velocity(latents_batch, noise, timesteps)
        else:  # epsilon
            target = noise

        return model_pred, target

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        R: torch.Tensor,
        loss_weights: Dict[str, float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted loss with MSE, RMS-regularization, and off-slot sparsity.

        Args:
            pred: Model prediction
            target: Ground truth
            R: Current residual [1, S, H]
            loss_weights: Dict with keys "mse", "rms", "offslot" (defaults provided)

        Returns:
            Dict with all loss components and total loss

        Notes:
            - MSE: main noise prediction loss
            - RMS: keeps slot RMS in reasonable range
            - Off-slot: forces residual to zero outside slot
        """
        if loss_weights is None:
            loss_weights = {"mse": 1.0, "rms": 0.02, "offslot": 0.05}

        # MSE loss (main)
        loss_mse = F.mse_loss(pred, target, reduction="mean")

        # RMS regularization (slot only)
        R_slot = R[0, self.s : self.s + self.L, :]  # [L, H]
        rms_slot = compute_rms(R_slot)

        # Target RMS around 0.8-1.2 (adjust based on observed values)
        target_rms = 1.0
        loss_rms = F.l1_loss(rms_slot, torch.tensor(target_rms, device=self.device))

        # Off-slot sparsity (force to zero)
        R_2d = R[0]  # [S, H]
        R_off_slot_before = R_2d[: self.s, :]
        R_off_slot_after = R_2d[self.s + self.L :, :]

        if R_off_slot_before.numel() > 0 or R_off_slot_after.numel() > 0:
            R_off_slot = torch.cat([R_off_slot_before, R_off_slot_after], dim=0)
            loss_offslot = torch.mean(R_off_slot**2)
        else:
            loss_offslot = torch.tensor(0.0, device=self.device)

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
        }

    def train_identity(
        self,
        image_paths: List[Path],
        steps: int = 600,
        lr: float = 5e-3,
        batch_size: int = 2,
        grad_clip: float = 1.0,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Train teacher embeddings for a single identity.

        Args:
            image_paths: List of 2-5 image paths for this identity
            steps: Number of optimization steps (default: 600)
            lr: Learning rate (default: 5e-3)
            batch_size: Batch size (default: 2)
            grad_clip: Gradient clipping threshold (default: 1.0)
            loss_weights: Optional custom loss weights

        Returns:
            Dict with training metrics and final parameters

        Raises:
            SystemExit: On NaN/Inf or divergence
        """
        # Prepare latents (cached)
        latents = self.prepare_images(image_paths)
        num_images = len(latents)

        # Initialize learnable parameters
        P = torch.zeros(1, self.S, self.H, device=self.device, dtype=self.dtype, requires_grad=True)
        log_scale = torch.nn.Parameter(torch.tensor(0.0, device=self.device, dtype=torch.float32))

        # Optimizer
        optimizer = torch.optim.AdamW([P, log_scale], lr=lr)

        # Training loop
        print(f"Training for {steps} steps...")
        print(f"  Images: {num_images}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print()

        loss_history = []
        best_loss = float("inf")

        pbar = tqdm(range(steps), desc="Training")

        for step in pbar:
            # Sample batch
            idx = torch.randperm(num_images)[:batch_size]
            latents_batch = latents[idx]

            # Sample timesteps (uniform)
            timesteps = torch.randint(
                0,
                self.pipeline.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            ).long()

            # Sample noise
            noise = torch.randn_like(latents_batch)

            # Re-parametrize
            E, R = self.reparametrize(P, log_scale)

            # Forward
            pred, target = self.forward_step(
                E,
                self.E_base_mask,
                latents_batch,
                timesteps,
                noise,
            )

            # Compute loss
            losses = self.compute_loss(pred, target, R, loss_weights)
            loss = losses["total"]

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([P, log_scale], grad_clip)

            # Optimizer step
            optimizer.step()

            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"\nERROR: NaN or Inf detected at step {step}.\n"
                    f"  Loss: {loss.item()}\n"
                    f"  Scale: {torch.exp(log_scale).item()}\n"
                    "  Try lowering LR or CFG scale.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Track metrics
            loss_val = loss.item()
            loss_history.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val

            # Compute RMS metrics
            with torch.no_grad():
                rms_metrics = compute_slot_off_slot_rms(R, self.s, self.L)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss_val:.4f}",
                    "scale": f"{torch.exp(log_scale).item():.3f}",
                    "slot_rms": f"{rms_metrics['slot_rms']:.3f}",
                    "off_rms": f"{rms_metrics['off_slot_rms']:.6f}",
                }
            )

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
                        f"  Current loss: {loss_val:.4f}\n"
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
            "final_loss": loss_val,
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

        return output_path

    def alpha_qa(
        self,
        E_teacher: torch.Tensor,
        alphas: List[float] = [0.5, 1.0, 1.5, 2.0],
        seeds: List[int] = list(range(4, 20)),
        output_dir: Path = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
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
            guidance_scale: CFG scale
            height: Image height
            width: Image width

        Returns:
            Path to grid PNG

        Notes:
            - E_eval(α) = E_base + α * (E_teacher - E_base)
            - Grid rows = alphas, cols = seeds
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
                    guidance_scale=guidance_scale,
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
            "guidance_scale": guidance_scale,
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
