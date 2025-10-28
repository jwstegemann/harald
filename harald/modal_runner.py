# modal_runner.py — Modal entrypoints inside the package (use: modal run -m cida.modal_runner:function)
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from pathlib import PurePosixPath
from typing import Any, Dict, Union

import modal

# --- Import project runtime from the same package ----------------------------------
from . import config as cfg  # type: ignore

app = cfg.app
image = cfg.image
HF_SECRET = cfg.HF_SECRET
HF_CACHE = cfg.HF_CACHE
hf_cache_vol = cfg.hf_cache_vol

# Optional GPU default
try:
    GPU_DEFAULT = cfg.GPU_DEFAULT  # type: ignore[attr-defined]
except Exception:
    GPU_DEFAULT = "L40S"


# Collect optional dataset shard volumes if they exist in config.
# NOTE: Purposely avoid narrow typing (dict[str, Volume]) to satisfy strict checkers that expect
# dict[str|PurePosixPath, CloudBucketMount|Volume]. We keep it untyped (dict) here.
def _collect_dataset_volumes() -> Dict[Union[str, PurePosixPath], Any]:
    vols = {}  # type: ignore[var-annotated]
    # Mount the HF cache volume to its path (HF_CACHE must be the mount path)
    vols[str(HF_CACHE)] = hf_cache_vol
    # dataset shards (0..9) are optional; mount to /data/shard_{i} by default, or env override
    for i in range(10):
        if i == 5:
            name = "da4taset"
        else:
            name = f"dataset_shard_{i}_vol"
        vol = getattr(cfg, name, None)
        if vol is not None:
            mount_path = Path(
                os.environ.get(f"CIDA_DATASET_SHARD_{i}_PATH", f"/mnt/dataset/{i}")
            )
            vols[str(mount_path)] = vol
    # preprocessed shards (0..9) are optional; mount to /mnt/preprocessed/{i} by default
    for i in range(10):
        name = f"preprocessed_shard_{i}_vol"
        vol = getattr(cfg, name, None)
        if vol is not None:
            mount_path = Path(
                os.environ.get(
                    f"CIDA_PREPROCESSED_SHARD_{i}_PATH", f"/mnt/preprocessed/{i}"
                )
            )
            vols[str(mount_path)] = vol

    # add output
    vols["/mnt/output"] = cfg.output_vol

    # add tensorboard
    vols["/mnt/tensorboard"] = cfg.tensorboard_vol

    # add cache
    vols["/mnt/cache"] = cfg.cache_vol

    return vols  # type: ignore


VOLUMES = _collect_dataset_volumes()

# ----------------------------- helpers --------------------------------------------


def _find_repo_root() -> Path:
    # Try to locate the project root (where tests/ or pyproject.toml live).
    # Starting from this file's directory (package), walk upwards.
    cur = Path(__file__).resolve().parent
    for _ in range(6):  # go up a few levels max
        if (cur / "pyproject.toml").exists() or (cur / "tests").exists():
            return cur
        cur = cur.parent
    # fallback: package parent
    return Path(__file__).resolve().parent


def _prep_runtime():
    # Ensure package root on sys.path and set env for HF cache
    root = _find_repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(str(root))
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _run_pytest(mark_expr: str) -> int:
    import pytest  # type: ignore

    _prep_runtime()
    print(f"[modal] pytest -m {mark_expr}")
    return int(pytest.main(["-m", mark_expr]))


def _run_pytest_file(file_path: str, mark_expr: str = "") -> int:
    import pytest  # type: ignore

    _prep_runtime()

    # Build pytest arguments
    args = [file_path, "-s"]

    # Add marker expression if provided
    if mark_expr:
        args.extend(["-m", mark_expr])
        print(f"[modal] pytest {file_path} -m {mark_expr}")
    else:
        print(f"[modal] pytest {file_path}")

    return int(pytest.main(args))


def _maybe_import_and_call(
    module_path: str, fn_name: str = "main", argv: Optional[List[str]] = None
) -> int:
    # Import module and call its `main` if available; fallback to subprocess -m.
    _prep_runtime()
    sys.argv = [module_path] + (argv or [])
    try:
        module = __import__(module_path, fromlist=["*"])
        target = getattr(module, fn_name, None)
        if callable(target):
            print(f"[modal] calling {module_path}.{fn_name}({argv})")
            target()
            return 0
    except Exception as e:
        print(
            f"[modal] import path failed ({module_path}): {e}; falling back to subprocess"
        )
        import traceback

        traceback.print_exc()  # prints the full stack trace to stderr
        print(f"!!! Exception: {e.__class__.__name__}:", e)
        raise
    # subprocess fallback
    cmd = ["python", "-m", module_path] + (argv or [])
    print(f"[modal] {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.call(cmd)


# ----------------------------- functions ------------------------------------------


@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes=VOLUMES,
    timeout=60 * 30,
    cpu=2,
    memory=4096,
)
def run_cpu_tests(mark_expr: str = "not gpu and not slow and not huge") -> int:
    """Run the fast, CPU-only test suite (default markers)."""
    return _run_pytest(mark_expr)


@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes=VOLUMES,
    timeout=60 * 60,
    gpu=GPU_DEFAULT,
    memory=12288,
)
def run_gpu_tests(mark_expr: str = "gpu and not huge") -> int:
    """Run the GPU-marked test suite."""
    return _run_pytest(mark_expr)


@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes=VOLUMES,
    timeout=60 * 60,
    gpu=GPU_DEFAULT,
    memory=12288,
)
def run_test_file(file_path: str, mark_expr: str = "") -> int:
    """Run tests in a specific file with optional marker filtering.

    Args:
        file_path: Relative path to test file (e.g., 'tests/test_label_fix.py')
        mark_expr: Optional pytest marker expression (e.g., 'gpu', 'not slow', etc.)
                  If empty, all tests in the file are run.

    Examples:
        modal run -m cida.modal_runner::run_test_file -- tests/test_label_fix.py
        modal run -m cida.modal_runner::run_test_file -- tests/test_preprocessing_integration.py gpu
        modal run -m cida.modal_runner::run_test_file -- tests/test_transforms_phase1.py "not slow"
    """
    return _run_pytest_file(file_path, mark_expr)


@app.function(
    image=image,
    secrets=[HF_SECRET],
    volumes=VOLUMES,
    timeout=60 * 60 * 24,
    gpu="H200",
    # ephemeral_disk= 7 * 8 * 1024,
    cpu=2.0,
    memory=32768,
)
def injection_poc(
    seed_dirs: str = "",
    train: bool = False,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 1e-3,
    alphas: str = "0.0,0.5,1.0,1.5",
    gen_seeds: str = "1234,5678,9999",
    base_prompt: str = "a portrait photo of a person, studio lighting, 85mm",
    negative_prompt: str = "blurry, low quality, deformed, watermark",
    steps: int = 50,
    guidance: float = 4.0,
    height: int = 928,
    width: int = 1664,
) -> int:
    """Phase 0: POC of injection using Qwen-VL

    Args:
        seed_dirs: Comma-separated list of seed_* directory paths
        train: Enable training mode
        epochs: Number of training epochs (default: 3)
        batch_size: Training batch size (default: 2)
        lr: Learning rate (default: 1e-3)
        alphas: Comma-separated alpha values for residual scaling (default: "0.5,1.0,1.5")
        gen_seeds: Comma-separated random seeds for generation (default: "1234,5678,9999")
        base_prompt: Base prompt for generation
        negative_prompt: Negative prompt for generation
        steps: Number of diffusion steps (default: 28)
        guidance: Guidance scale (default: 4.5)
        height: Generated image height (default: 768)
        width: Generated image width (default: 512)
    """
    argv = []

    # Required argument
    if not seed_dirs:
        raise ValueError(
            "seed_dirs is required. Provide comma-separated paths to seed_* directories."
        )
    argv.extend(["--seed-dirs", seed_dirs])

    # Training flags
    if train:
        argv.append("--train")
    argv.extend(["--epochs", str(epochs)])
    argv.extend(["--batch-size", str(batch_size)])
    argv.extend(["--lr", str(lr)])

    # Inference parameters
    argv.extend(["--alphas"] + alphas.split(","))
    argv.extend(["--gen-seeds"] + gen_seeds.split(","))
    argv.extend(["--steps", str(steps)])
    argv.extend(["--guidance", str(guidance)])
    argv.extend(["--height", str(height)])
    argv.extend(["--width", str(width)])

    # Prompts
    argv.extend(["--base-prompt", base_prompt])
    argv.extend(["--negative-prompt", negative_prompt])

    return _maybe_import_and_call("harald.inject.poc", "main", argv)
