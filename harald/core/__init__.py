"""Core utilities for Harald identity injection system."""

from .determinism import seed_all
from .slot_tokenizer import SlotConfig, tokenize_and_find_slot, find_all_slot_positions
from .shape_validator import validate_shapes, validate_device_dtype
from .metrics import compute_rms, compute_slot_off_slot_rms, compute_cosine_similarity, log_metrics_json

__all__ = [
    "seed_all",
    "SlotConfig",
    "tokenize_and_find_slot",
    "find_all_slot_positions",
    "validate_shapes",
    "validate_device_dtype",
    "compute_rms",
    "compute_slot_off_slot_rms",
    "compute_cosine_similarity",
    "log_metrics_json",
]
