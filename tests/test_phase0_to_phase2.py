"""
Unit tests for Phase 0-2 implementation.

Run with:
    pytest tests/test_phase0_to_phase2.py -v

Or with markers:
    pytest tests/test_phase0_to_phase2.py -v -m "not gpu"  # CPU-only tests
    pytest tests/test_phase0_to_phase2.py -v -m "gpu"      # GPU-only tests
"""

import pytest
import torch
from pathlib import Path

# Import modules to test
from harald.core.slot_tokenizer import SlotConfig, determine_slot_config, tokenize_and_find_slot
from harald.core.shape_validator import validate_device_dtype, validate_shapes, validate_injection_shapes
from harald.core.metrics import compute_rms, compute_slot_off_slot_rms, compute_cosine_similarity
from harald.core.determinism import seed_all


class TestSlotTokenizer:
    """Tests for slot tokenization and config."""

    def test_slot_config_creation(self):
        """Test SlotConfig dataclass creation."""
        config = SlotConfig(
            slot_string="~ID~",
            T_slot=[1234, 5678],
            L=2,
            s=10,
        )

        assert config.slot_string == "~ID~"
        assert config.T_slot == [1234, 5678]
        assert config.L == 2
        assert config.s == 10

    def test_slot_config_without_position(self):
        """Test SlotConfig without position (s=None)."""
        config = SlotConfig(
            slot_string="~ID~",
            T_slot=[1234],
            L=1,
        )

        assert config.s is None


class TestShapeValidator:
    """Tests for shape validation."""

    def test_validate_device_dtype_pass(self):
        """Test device/dtype validation passes with matching tensors."""
        device = torch.device("cpu")
        dtype = torch.float32

        tensors = {
            "a": torch.zeros(10, device=device, dtype=dtype),
            "b": torch.ones(5, 3, device=device, dtype=dtype),
        }

        # Should not raise
        validate_device_dtype(tensors, device, dtype)

    def test_validate_shapes_pass(self):
        """Test shape validation passes with matching shapes."""
        tensors = {
            "E_base": torch.zeros(1, 256, 2048),
            "R_slot": torch.zeros(1, 4, 2048),
        }

        expected = {
            "E_base": (1, 256, 2048),
            "R_slot": (1, 4, 2048),
        }

        # Should not raise
        validate_shapes(tensors, expected)

    def test_validate_shapes_with_none_dims(self):
        """Test shape validation with None for variable dimensions."""
        tensors = {
            "E_base": torch.zeros(2, 256, 2048),  # Batch size 2
        }

        expected = {
            "E_base": (None, 256, 2048),  # None allows any batch size
        }

        # Should not raise
        validate_shapes(tensors, expected)

    def test_validate_injection_shapes_pass(self):
        """Test injection shape validation."""
        E_base = torch.zeros(1, 256, 2048)
        R_slot = torch.zeros(1, 4, 2048)

        # Should not raise
        validate_injection_shapes(E_base, R_slot, L=4, H=2048)


class TestMetrics:
    """Tests for metrics computation."""

    def test_compute_rms(self):
        """Test RMS computation."""
        tensor = torch.ones(10, 20) * 2.0  # RMS should be 2.0

        rms = compute_rms(tensor)

        assert torch.isclose(rms, torch.tensor(2.0), atol=1e-6)

    def test_compute_slot_off_slot_rms(self):
        """Test slot vs off-slot RMS computation."""
        # Create residual with non-zero slot region and zero off-slot
        S, H = 256, 2048
        s, L = 100, 4

        residual = torch.zeros(1, S, H)
        residual[0, s:s+L, :] = 1.0  # Slot region has values

        metrics = compute_slot_off_slot_rms(residual, s, L)

        assert metrics["slot_rms"] > 0.9  # Should be close to 1.0
        assert metrics["off_slot_rms"] < 1e-3  # Should be very small (relaxed threshold)
        assert "total_rms" in metrics

    def test_compute_cosine_similarity(self):
        """Test cosine similarity computation."""
        pred = torch.tensor([[1.0, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])

        sim = compute_cosine_similarity(pred, target)

        assert torch.isclose(sim, torch.tensor(1.0), atol=1e-6)


class TestDeterminism:
    """Tests for deterministic seeding."""

    def test_seed_all(self):
        """Test that seed_all produces reproducible results."""
        seed_all(42)
        rand1 = torch.rand(5)

        seed_all(42)
        rand2 = torch.rand(5)

        assert torch.allclose(rand1, rand2)


# GPU-only tests (requires CUDA)
@pytest.mark.gpu
class TestGPUOperations:
    """Tests that require GPU."""

    def test_cuda_available(self):
        """Test CUDA availability."""
        assert torch.cuda.is_available(), "CUDA not available for GPU tests"

    def test_tensor_on_cuda(self):
        """Test tensor creation on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        tensor = torch.zeros(10, device=device)

        assert tensor.device.type == "cuda"

    def test_dtype_consistency(self):
        """Test dtype consistency on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        dtype = torch.float16

        tensors = {
            "a": torch.zeros(10, device=device, dtype=dtype),
            "b": torch.ones(5, 3, device=device, dtype=dtype),
        }

        # Should not raise
        validate_device_dtype(tensors, device, dtype)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
