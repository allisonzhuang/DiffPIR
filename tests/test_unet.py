"""Tests for models/unet.py — U-Net wrapper implementing DenoiserPrior.

The U-Net is the most expensive component; tests use mock models to keep
execution fast while verifying the wrapper logic and interface compliance.
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from configs import DiffusionConfig
from interfaces import DenoiserPrior
from models.unet import UNetDenoiser, build_unet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockUNet(nn.Module):
    """Minimal U-Net mock that returns zeros as noise prediction.
    Simulates ε_θ(x_t, t) = 0, so predict_x0 returns x_t / √ᾱ_t.
    """
    def forward(self, x_t: Tensor, t: Tensor | int) -> Tensor:
        return torch.zeros_like(x_t)


class MockUNetEcho(nn.Module):
    """Mock U-Net that returns the input as the noise prediction.
    Simulates ε_θ(x_t, t) = x_t.
    """
    def forward(self, x_t: Tensor, t: Tensor | int) -> Tensor:
        return x_t


@pytest.fixture
def noise_schedule():
    """Minimal noise schedule for testing the wrapper."""
    from models.diffusion import make_noise_schedule
    return make_noise_schedule(DiffusionConfig())


@pytest.fixture
def mock_denoiser(noise_schedule):
    """UNetDenoiser wrapping a zero-output mock U-Net."""
    return UNetDenoiser(model=MockUNet(), noise_schedule=noise_schedule)


# ---------------------------------------------------------------------------
# ABC compliance
# ---------------------------------------------------------------------------

class TestUNetDenoiserInterface:

    def test_is_denoiser_prior_subclass(self):
        """UNetDenoiser must be a subclass of DenoiserPrior.
        This ensures it can be used polymorphically in the HQS loop.
        """
        assert issubclass(UNetDenoiser, DenoiserPrior)

    def test_isinstance_check(self, mock_denoiser):
        """An instance of UNetDenoiser must pass isinstance(x, DenoiserPrior).
        This is the runtime contract checked by the HQS functions.
        """
        assert isinstance(mock_denoiser, DenoiserPrior)


# ---------------------------------------------------------------------------
# denoise method
# ---------------------------------------------------------------------------

class TestUNetDenoiserDenoise:

    def test_output_shape(self, mock_denoiser, noise_schedule):
        """denoise must return a tensor with the exact same shape as x_t.
        Shape mismatch would crash the data step in the HQS loop.
        """
        x_t = torch.randn(2, 3, 32, 32)
        result = mock_denoiser.denoise(x_t, t=500, noise_schedule=noise_schedule)
        assert result.shape == x_t.shape

    def test_output_dtype_float32(self, mock_denoiser, noise_schedule):
        """denoise output must be float32.  The entire pipeline assumes float32;
        dtype drift (e.g. float64 from a division) would be a silent bug.
        """
        x_t = torch.randn(1, 3, 16, 16)
        result = mock_denoiser.denoise(x_t, t=100, noise_schedule=noise_schedule)
        assert result.dtype == torch.float32

    def test_batch_independence(self, noise_schedule):
        """Each sample in the batch should be processed independently.
        The result for sample i should not change when sample j is modified.
        """
        model = MockUNet()
        denoiser = UNetDenoiser(model=model, noise_schedule=noise_schedule)
        x_t = torch.randn(4, 3, 16, 16)
        result_full = denoiser.denoise(x_t, t=500, noise_schedule=noise_schedule)
        result_single = denoiser.denoise(x_t[0:1], t=500, noise_schedule=noise_schedule)
        torch.testing.assert_close(result_full[0:1], result_single, atol=1e-6, rtol=1e-6)

    def test_with_zero_noise_prediction(self, noise_schedule):
        """When the U-Net predicts ε=0, predict_x0 gives x̂₀ = x_t / √ᾱ_t.
        This verifies the wrapper correctly chains the U-Net output through
        predict_x0 (Eq. 11).
        """
        import math
        model = MockUNet()
        denoiser = UNetDenoiser(model=model, noise_schedule=noise_schedule)
        x_t = torch.ones(1, 1, 4, 4)
        t = 500
        ab_t = noise_schedule["alpha_bars"][t - 1].item()
        result = denoiser.denoise(x_t, t=t, noise_schedule=noise_schedule)
        expected = x_t / math.sqrt(ab_t)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_various_timesteps(self, mock_denoiser, noise_schedule):
        """denoise should work for any valid timestep t ∈ {1, …, T}.
        Tests boundary timesteps (t=1, t=T) and a mid-range value.
        """
        x_t = torch.randn(1, 3, 8, 8)
        for t in [1, 500, 1000]:
            result = mock_denoiser.denoise(x_t, t=t, noise_schedule=noise_schedule)
            assert result.shape == x_t.shape
            assert torch.isfinite(result).all(), f"Non-finite output at t={t}"


# ---------------------------------------------------------------------------
# build_unet
# ---------------------------------------------------------------------------

class TestBuildUnet:

    def test_returns_nn_module(self):
        """build_unet must return an nn.Module suitable for training."""
        cfg = DiffusionConfig()
        model = build_unet(cfg)
        assert isinstance(model, nn.Module)

    def test_float32_parameters(self):
        """All model parameters should be float32 (as required by the paper's
        noise schedule and training objective).
        """
        cfg = DiffusionConfig()
        model = build_unet(cfg)
        for param in model.parameters():
            assert param.dtype == torch.float32
