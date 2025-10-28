"""Determinism utilities for reproducible training and inference."""

import random

import numpy as np
import torch


def seed_all(seed: int = 123):
    """
    Set deterministic seeds for reproducibility across all libraries.

    Args:
        seed: Random seed to use (default: 123)

    Notes:
        - Sets seeds for Python random, NumPy, PyTorch (CPU + CUDA)
        - Enables deterministic cuDNN algorithms (may impact performance)
        - Same seed should produce identical results across runs
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
