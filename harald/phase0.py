"""Phase 0: Environment checks and basic specifications."""

import sys
from pathlib import Path
from typing import List, Dict, Any

import torch

from .core.slot_tokenizer import SlotConfig, determine_slot_config


def check_cuda_availability() -> Dict[str, Any]:
    """
    Check CUDA availability and gather GPU information.

    Returns:
        Dict with CUDA version, driver, GPU name, and VRAM info

    Raises:
        SystemExit: If CUDA is not available (hard fail)
    """
    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA is not available. This system requires NVIDIA GPU with CUDA.\n"
            "  torch.cuda.is_available() returned False.\n"
            "  CPU-only execution is NOT supported.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Gather CUDA information
    cuda_info = {
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "num_gpus": torch.cuda.device_count(),
        "gpus": [],
    }

    # Gather info for each GPU
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpu_info = {
            "id": i,
            "name": props.name,
            "total_memory_gb": props.total_memory / (1024**3),
            "compute_capability": f"{props.major}.{props.minor}",
        }
        cuda_info["gpus"].append(gpu_info)

    # Log to stdout
    print("=" * 60)
    print("CUDA Environment Check")
    print("=" * 60)
    print(f"CUDA Version:        {cuda_info['cuda_version']}")
    print(f"cuDNN Version:       {cuda_info['cudnn_version']}")
    print(f"Number of GPUs:      {cuda_info['num_gpus']}")
    print()

    for gpu in cuda_info["gpus"]:
        print(f"GPU {gpu['id']}: {gpu['name']}")
        print(f"  Total Memory:      {gpu['total_memory_gb']:.2f} GB")
        print(f"  Compute Capability: {gpu['compute_capability']}")
        print()

    print("=" * 60)

    return cuda_info


def determine_dtype(prefer_bf16: bool = True) -> torch.dtype:
    """
    Determine appropriate dtype based on GPU capabilities.

    Args:
        prefer_bf16: If True, prefer bfloat16 over float16 when supported

    Returns:
        torch.dtype (bf16, fp16, or fp32)

    Notes:
        - Checks GPU support for bfloat16
        - Falls back to fp16, then fp32 if needed
        - Logs chosen dtype to stdout
    """
    # Check bfloat16 support
    bf16_supported = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    if prefer_bf16 and bf16_supported:
        dtype = torch.bfloat16
        dtype_name = "bfloat16 (bf16)"
    elif not prefer_bf16 and bf16_supported:
        # User doesn't prefer bf16, but it's available - use fp16
        dtype = torch.float16
        dtype_name = "float16 (fp16)"
    elif bf16_supported:
        dtype = torch.bfloat16
        dtype_name = "bfloat16 (bf16)"
    else:
        # bf16 not supported, try fp16
        dtype = torch.float16
        dtype_name = "float16 (fp16)"

    print("=" * 60)
    print("dtype Selection")
    print("=" * 60)
    print(f"bfloat16 supported:  {bf16_supported}")
    print(f"Prefer bfloat16:     {prefer_bf16}")
    print(f"Selected dtype:      {dtype_name}")
    print("=" * 60)
    print()

    return dtype


def scan_dataset(base_paths: List[str]) -> List[Path]:
    """
    Scan dataset directories for valid seed_* folders with ≥2 images.

    Args:
        base_paths: List of base directory paths to scan (e.g., ["/mnt/dataset/0", "/mnt/dataset/1"])

    Returns:
        List of Path objects for valid seed_* directories

    Notes:
        - Scans for directories matching pattern: seed_*
        - Filters to only include directories with ≥2 image files
        - Supported image formats: .png, .jpg, .jpeg, .webp, .bmp
        - Logs statistics to stdout
    """
    valid_seed_dirs = []
    total_scanned = 0
    total_with_images = 0

    print("=" * 60)
    print("Dataset Scan")
    print("=" * 60)

    for base_path_str in base_paths:
        base_path = Path(base_path_str)

        if not base_path.exists():
            print(f"WARNING: Base path does not exist: {base_path}")
            continue

        if not base_path.is_dir():
            print(f"WARNING: Base path is not a directory: {base_path}")
            continue

        # Find all seed_* directories
        seed_dirs = sorted(base_path.glob("seed_*"))

        for seed_dir in seed_dirs:
            if not seed_dir.is_dir():
                continue

            total_scanned += 1

            # Count image files
            image_files = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]:
                image_files.extend(seed_dir.glob(ext))

            num_images = len(image_files)

            if num_images >= 2:
                valid_seed_dirs.append(seed_dir)
                total_with_images += 1

        print(f"Scanned: {base_path}")
        print(f"  Found {len(seed_dirs)} seed_* directories")

    print()
    print(f"Total seed_* directories scanned: {total_scanned}")
    print(f"Directories with ≥2 images:       {total_with_images}")
    print(f"Valid directories selected:       {len(valid_seed_dirs)}")
    print("=" * 60)
    print()

    if len(valid_seed_dirs) == 0:
        print(
            "ERROR: No valid seed_* directories found with ≥2 images.\n"
            f"  Scanned paths: {base_paths}\n"
            "  Please check dataset structure.",
            file=sys.stderr,
        )
        sys.exit(1)

    return valid_seed_dirs


def initialize_slot_config(
    tokenizer,
    slot_string: str = "~ID~",
    target_length: int = 4,
) -> SlotConfig:
    """
    Initialize slot configuration by finding a slot string that tokenizes to target_length.

    Args:
        tokenizer: HuggingFace tokenizer from Qwen-Image pipeline
        slot_string: Preferred slot string (default: "~ID~")
        target_length: Target token length (default: 4)

    Returns:
        SlotConfig with slot_string, T_slot, L

    Raises:
        SystemExit: If no valid slot string found with target_length

    Notes:
        - First tries the provided slot_string
        - If that doesn't match target_length, tries alternatives
        - Logs slot configuration to stdout
    """
    print("=" * 60)
    print("Slot Configuration Initialization")
    print("=" * 60)
    print(f"Target slot length:  {target_length} tokens")
    print(f"Preferred string:    '{slot_string}'")
    print()

    # Try the preferred slot string first
    candidates = [slot_string, "~identity~", "~ID_token~", "~SLOT~", "~PERSON~"]

    try:
        slot_config = determine_slot_config(
            tokenizer=tokenizer,
            target_length=target_length,
            candidate_strings=candidates,
        )

        print(f"✓ Slot configuration determined:")
        print(f"  Slot string:       '{slot_config.slot_string}'")
        print(f"  Token IDs (T_slot): {slot_config.T_slot}")
        print(f"  Length (L):        {slot_config.L}")
        print()
        print(f"Tokenizer info:")
        print(f"  Model max length:  {tokenizer.model_max_length}")
        print(f"  Vocab size:        {tokenizer.vocab_size}")
        print("=" * 60)
        print()

        return slot_config

    except ValueError as e:
        print(
            f"ERROR: Could not find slot string with length {target_length}.\n"
            f"  Tried candidates: {candidates}\n"
            f"  Tokenizer: {tokenizer.__class__.__name__}\n"
            f"  Error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
