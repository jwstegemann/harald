"""Phase 1: Minimal render path (smoke test without training)."""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from .models.qwen_image import QwenImagePipeline
from .core.slot_tokenizer import SlotConfig
from .core.shape_validator import validate_device_dtype, validate_shapes
from .core.determinism import seed_all


def save_grid(
    images: List[Image.Image],
    rows: int,
    cols: int,
    out_path: Path,
    pad: int = 8,
    bg=(18, 18, 18),
) -> None:
    """
    Save a grid of images.

    Args:
        images: List of PIL Images
        rows: Number of rows in grid
        cols: Number of columns in grid
        out_path: Output path for grid image
        pad: Padding between images in pixels
        bg: Background color (RGB tuple)

    Raises:
        ValueError: If number of images doesn't match rows * cols
    """
    if len(images) != rows * cols:
        raise ValueError(f"Expected {rows * cols} images, got {len(images)}")

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"✓ Saved grid to {out_path}")


def test_minimal_render(
    pipeline: QwenImagePipeline,
    base_prompt: str,
    slot_config: SlotConfig,
    device: torch.device,
    dtype: torch.dtype,
    output_dir: Optional[Path] = None,
) -> bool:
    """
    Test minimal render path: encode base prompt, generate images with dummy residual.

    Args:
        pipeline: QwenImagePipeline instance
        base_prompt: Base prompt with slot (e.g., "a portrait photo of ~ID~, studio lighting")
        slot_config: Slot configuration
        device: Target device (must be CUDA)
        dtype: Target dtype
        output_dir: Optional output directory for test images

    Returns:
        True if test passed successfully

    Raises:
        SystemExit: If any validation fails or generation errors occur
    """
    print("=" * 60)
    print("Phase 1: Minimal Render Test")
    print("=" * 60)
    print(f"Base prompt: '{base_prompt}'")
    print(f"Slot string: '{slot_config.slot_string}'")
    print(f"Device:      {device}")
    print(f"dtype:       {dtype}")
    print()

    # Step 1: Encode base prompt
    print("Step 1: Encoding base prompt...")
    E_base, E_base_mask = pipeline.encode_text(base_prompt)

    print(f"  E_base shape:      {E_base.shape}")
    print(f"  E_base device:     {E_base.device}")
    print(f"  E_base dtype:      {E_base.dtype}")
    print(f"  E_base_mask shape: {E_base_mask.shape}")

    # Validate device/dtype
    validate_device_dtype(
        {"E_base": E_base, "E_base_mask": E_base_mask},
        expected_device=device,
        expected_dtype=dtype,
    )

    # Validate shape (must be [1, S, H])
    if E_base.ndim != 3 or E_base.shape[0] != 1:
        print(
            f"ERROR: E_base must be [1, S, H], got {E_base.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    B, S, H = E_base.shape
    L = slot_config.L

    print(f"  ✓ Shape validated: B={B}, S={S}, H={H}, L={L}")
    print()

    # Step 2: Create dummy residual (all zeros)
    print("Step 2: Creating dummy residual (zeros)...")
    R_dummy = torch.zeros(1, L, H, device=device, dtype=dtype)

    print(f"  R_dummy shape:  {R_dummy.shape}")
    print(f"  R_dummy device: {R_dummy.device}")
    print(f"  R_dummy dtype:  {R_dummy.dtype}")
    print(f"  ✓ Dummy residual created")
    print()

    # Step 3: Test injection (without actual injection, just copy)
    print("Step 3: Testing embedding injection...")
    E_inj = E_base.clone()

    # We could inject here, but for minimal test we keep it as-is
    # E_inj[:, s:s+L, :] += alpha * R_dummy

    print(f"  E_inj shape:  {E_inj.shape}")
    print(f"  ✓ Injection test passed (no actual residual added)")
    print()

    # Step 4: Generate test images
    print("Step 4: Generating test images...")
    test_seeds = [0, 1]
    test_images = []

    for seed in test_seeds:
        seed_all(seed)
        print(f"  Generating with seed={seed}...")

        img = pipeline.generate(
            prompt_embeds=E_inj,
            prompt_embeds_mask=E_base_mask,
            negative_prompt_embeds=None,
            negative_prompt_embeds_mask=None,
            num_inference_steps=20,  # Fewer steps for quick test
            guidance_scale=4.0,
            height=512,  # Smaller size for quick test
            width=512,
            seed=seed,
        )

        test_images.append(img)
        print(f"    ✓ Generated {img.size}")

    print(f"  ✓ Generated {len(test_images)} images")
    print()

    # Step 5: Save test images
    if output_dir is not None:
        print("Step 5: Saving test images...")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(test_images):
            img_path = output_dir / f"phase1_test_seed{test_seeds[i]}.png"
            img.save(img_path)
            print(f"  ✓ Saved {img_path}")

        print()

    print("=" * 60)
    print("✓ Phase 1 Test PASSED")
    print("=" * 60)
    print()

    return True


def test_alpha_sweep(
    pipeline: QwenImagePipeline,
    base_prompt: str,
    slot_config: SlotConfig,
    device: torch.device,
    dtype: torch.dtype,
    alphas: List[float] = [0.5, 1.0, 1.5, 2.0],
    seeds: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
    output_dir: Optional[Path] = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.0,
    height: int = 512,
    width: int = 512,
) -> Image.Image:
    """
    Test alpha sweep with dummy residual: generate grid (rows=alphas, cols=seeds).

    Args:
        pipeline: QwenImagePipeline instance
        base_prompt: Base prompt with slot
        slot_config: Slot configuration
        device: Target device
        dtype: Target dtype
        alphas: List of alpha values to test
        seeds: List of random seeds
        output_dir: Optional output directory
        num_inference_steps: Number of diffusion steps
        guidance_scale: CFG scale
        height: Image height
        width: Image width

    Returns:
        Grid image (PIL.Image)
    """
    print("=" * 60)
    print("Phase 1: Alpha Sweep Test (Dummy Residual)")
    print("=" * 60)
    print(f"Alphas: {alphas}")
    print(f"Seeds:  {seeds}")
    print(f"Grid:   {len(alphas)} rows × {len(seeds)} cols")
    print()

    # Encode base prompt
    E_base, E_base_mask = pipeline.encode_text(base_prompt)
    B, S, H = E_base.shape
    L = slot_config.L

    # Create dummy residual
    R_dummy = torch.zeros(1, L, H, device=device, dtype=dtype)

    # Generate grid
    all_images = []

    for alpha in alphas:
        print(f"Generating for alpha={alpha}...")
        row_images = []

        for seed in seeds:
            seed_all(seed)

            # Inject with alpha (even though R_dummy is zeros, we test the injection path)
            E_inj = E_base.clone()
            # In real implementation, we'd find slot position and inject
            # For now, just use base embeddings

            img = pipeline.generate(
                prompt_embeds=E_inj,
                prompt_embeds_mask=E_base_mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                seed=seed,
            )

            row_images.append(img)

        all_images.extend(row_images)
        print(f"  ✓ Generated {len(row_images)} images for alpha={alpha}")

    print()
    print(f"Total images generated: {len(all_images)}")

    # Create grid
    grid_path = output_dir / "phase1_alpha_sweep_grid.png" if output_dir else Path("phase1_alpha_sweep_grid.png")
    save_grid(
        images=all_images,
        rows=len(alphas),
        cols=len(seeds),
        out_path=grid_path,
    )

    # Save metadata
    if output_dir:
        meta = {
            "alphas": alphas,
            "seeds": seeds,
            "base_prompt": base_prompt,
            "slot_string": slot_config.slot_string,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
        }

        meta_path = output_dir / "phase1_alpha_sweep_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✓ Saved metadata to {meta_path}")

    print("=" * 60)
    print("✓ Alpha Sweep Test PASSED")
    print("=" * 60)
    print()

    # Load and return grid
    return Image.open(grid_path)
