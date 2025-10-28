"""Metrics computation and logging utilities."""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F


def compute_rms(tensor: torch.Tensor, dim: Optional[int] = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Root Mean Square (RMS) of a tensor.

    Args:
        tensor: Input tensor
        dim: Dimension(s) to reduce over (None = all dimensions)
        eps: Epsilon for numerical stability

    Returns:
        RMS value(s)
    """
    return torch.sqrt(torch.mean(tensor**2, dim=dim) + eps)


def compute_slot_off_slot_rms(
    residual: torch.Tensor,
    s: int,
    L: int,
) -> Dict[str, float]:
    """
    Compute RMS separately for slot region and off-slot regions.

    Args:
        residual: Residual tensor [1, S, H] or [S, H]
        s: Slot start position
        L: Slot length

    Returns:
        Dict with keys: "slot_rms", "off_slot_rms", "total_rms"
    """
    if residual.ndim == 3:
        residual = residual[0]  # [S, H]

    S, H = residual.shape

    # Extract regions
    slot_region = residual[s : s + L, :]  # [L, H]
    off_slot_before = residual[:s, :]  # [s, H]
    off_slot_after = residual[s + L :, :]  # [S-(s+L), H]

    # Compute RMS
    slot_rms = compute_rms(slot_region).item()
    total_rms = compute_rms(residual).item()

    # Off-slot RMS (concatenate both regions)
    if off_slot_before.numel() > 0 or off_slot_after.numel() > 0:
        off_slot_combined = torch.cat([off_slot_before, off_slot_after], dim=0)
        off_slot_rms = compute_rms(off_slot_combined).item()
    else:
        off_slot_rms = 0.0

    return {
        "slot_rms": slot_rms,
        "off_slot_rms": off_slot_rms,
        "total_rms": total_rms,
    }


def compute_cosine_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Compute cosine similarity between two tensors.

    Args:
        pred: Predicted tensor
        target: Target tensor
        dim: Dimension along which to compute similarity

    Returns:
        Cosine similarity value(s) in range [-1, 1]
    """
    return F.cosine_similarity(pred, target, dim=dim)


def log_metrics_json(
    metrics: Dict[str, Any],
    output_path: Path,
    mode: str = "w",
) -> None:
    """
    Log metrics to a JSON file.

    Args:
        metrics: Dict of metrics to log
        output_path: Path to JSON file
        mode: File open mode ("w" = overwrite, "a" = append)

    Notes:
        - If mode="a", each call appends a new JSON object (one per line)
        - If mode="w", overwrites the file with a single JSON object
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "w":
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    elif mode == "a":
        with open(output_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'w' or 'a'.")


def format_metrics_for_logging(
    step: int,
    loss: float,
    scale: float,
    rms_metrics: Dict[str, float],
    lr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Format metrics for logging during training.

    Args:
        step: Training step
        loss: Total loss
        scale: Current scale value (exp(log_scale))
        rms_metrics: Dict from compute_slot_off_slot_rms()
        lr: Optional learning rate

    Returns:
        Formatted metrics dict
    """
    metrics = {
        "step": step,
        "loss": loss,
        "scale": scale,
        **rms_metrics,
    }

    if lr is not None:
        metrics["lr"] = lr

    return metrics
