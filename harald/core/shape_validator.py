"""Shape, device, and dtype validation utilities with hard failures."""

import sys
from typing import Dict, Any

import torch


def validate_device_dtype(
    tensors: Dict[str, torch.Tensor],
    expected_device: torch.device,
    expected_dtype: torch.dtype,
) -> None:
    """
    Validate that all tensors have the expected device and dtype.

    Args:
        tensors: Dict mapping tensor names to tensors
        expected_device: Expected device (e.g., torch.device("cuda"))
        expected_dtype: Expected dtype (e.g., torch.float16)

    Raises:
        SystemExit: If any tensor has wrong device or dtype (hard fail)
    """
    for name, tensor in tensors.items():
        if tensor.device != expected_device:
            print(
                f"ERROR: Tensor '{name}' has wrong device.\n"
                f"  Expected: {expected_device}\n"
                f"  Got:      {tensor.device}\n"
                f"  Shape:    {tensor.shape}\n"
                f"  dtype:    {tensor.dtype}",
                file=sys.stderr,
            )
            sys.exit(1)

        if tensor.dtype != expected_dtype:
            print(
                f"ERROR: Tensor '{name}' has wrong dtype.\n"
                f"  Expected: {expected_dtype}\n"
                f"  Got:      {tensor.dtype}\n"
                f"  Device:   {tensor.device}\n"
                f"  Shape:    {tensor.shape}",
                file=sys.stderr,
            )
            sys.exit(1)


def validate_shapes(
    tensors: Dict[str, torch.Tensor],
    expected_shapes: Dict[str, tuple],
) -> None:
    """
    Validate that tensors have expected shapes.

    Args:
        tensors: Dict mapping tensor names to tensors
        expected_shapes: Dict mapping tensor names to expected shapes
                        Use None for dimensions that can vary (e.g., batch size)

    Raises:
        SystemExit: If any tensor has wrong shape (hard fail)

    Example:
        validate_shapes(
            {"E_base": E_base, "R_slot": R_slot},
            {"E_base": (1, 256, 2048), "R_slot": (1, 4, 2048)}
        )
    """
    for name, tensor in tensors.items():
        if name not in expected_shapes:
            continue

        expected = expected_shapes[name]
        actual = tuple(tensor.shape)

        # Check dimension count
        if len(actual) != len(expected):
            print(
                f"ERROR: Tensor '{name}' has wrong number of dimensions.\n"
                f"  Expected: {len(expected)} dims {expected}\n"
                f"  Got:      {len(actual)} dims {actual}\n"
                f"  Device:   {tensor.device}\n"
                f"  dtype:    {tensor.dtype}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Check each dimension (None = any size allowed)
        for i, (exp_dim, act_dim) in enumerate(zip(expected, actual)):
            if exp_dim is not None and exp_dim != act_dim:
                print(
                    f"ERROR: Tensor '{name}' has wrong size at dimension {i}.\n"
                    f"  Expected: {expected}\n"
                    f"  Got:      {actual}\n"
                    f"  Device:   {tensor.device}\n"
                    f"  dtype:    {tensor.dtype}",
                    file=sys.stderr,
                )
                sys.exit(1)


def validate_injection_shapes(
    E_base: torch.Tensor,
    R_slot: torch.Tensor,
    L: int,
    H: int,
) -> None:
    """
    Validate shapes for injection operation.

    Args:
        E_base: Base embeddings [B, S, H]
        R_slot: Residual embeddings [B, L, H]
        L: Slot length in tokens
        H: Hidden dimension

    Raises:
        SystemExit: If shapes incompatible for injection
    """
    # Check E_base shape
    if E_base.ndim != 3:
        print(
            f"ERROR: E_base must be 3D [B, S, H], got {E_base.ndim}D: {E_base.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    B, S, H_base = E_base.shape

    if H_base != H:
        print(
            f"ERROR: E_base hidden dimension mismatch.\n"
            f"  Expected H={H}\n"
            f"  Got H_base={H_base}\n"
            f"  E_base shape: {E_base.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check R_slot shape
    if R_slot.ndim != 3:
        print(
            f"ERROR: R_slot must be 3D [B, L, H], got {R_slot.ndim}D: {R_slot.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    B_r, L_r, H_r = R_slot.shape

    if B_r != B:
        print(
            f"ERROR: Batch size mismatch.\n"
            f"  E_base batch: {B}\n"
            f"  R_slot batch: {B_r}",
            file=sys.stderr,
        )
        sys.exit(1)

    if L_r != L:
        print(
            f"ERROR: R_slot slot length mismatch.\n"
            f"  Expected L={L}\n"
            f"  Got L_r={L_r}\n"
            f"  R_slot shape: {R_slot.shape}",
            file=sys.stderr,
        )
        sys.exit(1)

    if H_r != H:
        print(
            f"ERROR: R_slot hidden dimension mismatch.\n"
            f"  Expected H={H}\n"
            f"  Got H_r={H_r}\n"
            f"  R_slot shape: {R_slot.shape}",
            file=sys.stderr,
        )
        sys.exit(1)
