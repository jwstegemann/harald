"""Phase 2 Runner: Batch processing for teacher inversion across multiple identities."""

import json
from pathlib import Path
from typing import List, Dict, Any

from .teacher_trainer import TeacherInversionTrainer
from ..models.qwen_image import QwenImagePipeline
from ..core.slot_tokenizer import SlotConfig


def run_phase2_for_identities(
    seed_dirs: List[Path],
    base_prompt: str,
    output_dir: Path,
    pipeline: QwenImagePipeline,
    slot_config: SlotConfig,
    device,
    dtype,
    steps: int = 600,
    lr: float = 1e-4,
    cfg_scale: float = 1.0,
    batch_size: int = 2,
    negative_prompt: str = " ",
    alpha_qa_alphas: List[float] = [0.5, 1.0, 1.5, 2.0],
    alpha_qa_seeds: List[int] = list(range(4, 8)),
) -> List[Dict[str, Any]]:
    """
    Run Phase 2 teacher inversion for multiple identities.

    Args:
        seed_dirs: List of seed_* directories
        base_prompt: Base prompt with slot
        output_dir: Output directory for all results
        pipeline: QwenImagePipeline instance
        slot_config: Slot configuration
        device: Target device
        dtype: Target dtype
        steps: Training steps per identity
        lr: Learning rate
        cfg_scale: CFG scale
        batch_size: Batch size for training
        negative_prompt: Negative prompt
        alpha_qa_alphas: Alphas for QA grid
        alpha_qa_seeds: Seeds for QA grid

    Returns:
        List of result dicts (one per identity)

    Notes:
        - Processes each identity sequentially
        - Saves teacher_prompt_embeds.pt per identity
        - Generates alpha-QA grid per identity
        - Collects metrics in summary JSON
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = TeacherInversionTrainer(
        pipeline=pipeline,
        base_prompt=base_prompt,
        slot_config=slot_config,
        device=device,
        dtype=dtype,
        cfg_scale=cfg_scale,
        negative_prompt=negative_prompt,
    )

    results = []

    print("=" * 60)
    print(f"Phase 2: Teacher Inversion - {len(seed_dirs)} identities")
    print("=" * 60)
    print()

    for i, seed_dir in enumerate(seed_dirs):
        print(f"\n{'=' * 60}")
        print(f"Identity {i + 1}/{len(seed_dirs)}: {seed_dir.name}")
        print(f"{'=' * 60}\n")

        # Find image files (photo_view_* only, not comic_view_*)
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]:
            # Filter for photo_view_* pattern
            for img_file in seed_dir.glob(ext):
                if img_file.name.startswith("photo_view_"):
                    image_files.append(img_file)

        if len(image_files) < 2:
            print(f"WARNING: {seed_dir} has <2 photo_view_* images. Skipping.")
            print(f"  Found {len(image_files)} photo_view_* images")
            continue

        # Take first 5 images (or all if <5)
        image_files = sorted(image_files)[:5]

        print(f"Using {len(image_files)} images:")
        for img_file in image_files:
            print(f"  - {img_file.name}")
        print()

        # Create output directory for this identity
        identity_output_dir = output_dir / seed_dir.name
        identity_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Train
            training_result = trainer.train_identity(
                image_paths=image_files,
                steps=steps,
                lr=lr,
                batch_size=batch_size,
            )

            # Post-process and save
            teacher_path = identity_output_dir / "teacher_prompt_embeds.pt"
            trainer.post_process_and_save(training_result, teacher_path)

            # Alpha-QA
            E_teacher = training_result["E_final"]
            qa_grid_path = trainer.alpha_qa(
                E_teacher=E_teacher,
                alphas=alpha_qa_alphas,
                seeds=alpha_qa_seeds,
                output_dir=identity_output_dir,
            )

            # Record result
            result = {
                "seed_dir": str(seed_dir),
                "seed_name": seed_dir.name,
                "num_images": len(image_files),
                "teacher_path": str(teacher_path),
                "qa_grid_path": str(qa_grid_path),
                "success": True,
                **training_result["metrics"],
            }

            results.append(result)

            print(f"✓ Identity {seed_dir.name} complete.\n")

        except Exception as e:
            import traceback
            print(f"ERROR: Failed to process {seed_dir.name}")
            print(f"  {e}")
            print()
            print("Full traceback:")
            traceback.print_exc()
            print()

            result = {
                "seed_dir": str(seed_dir),
                "seed_name": seed_dir.name,
                "success": False,
                "error": str(e),
            }

            results.append(result)

    # Save summary
    summary_path = output_dir / "phase2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Phase 2 Complete")
    print("=" * 60)
    print(f"Total identities: {len(seed_dirs)}")
    print(f"Successful:       {sum(1 for r in results if r.get('success', False))}")
    print(f"Failed:           {sum(1 for r in results if not r.get('success', False))}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 60)
    print()

    return results
