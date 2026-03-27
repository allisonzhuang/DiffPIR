"""Shared pytest fixtures for all DiffPIR test modules.

Centralises construction of noise schedules, config objects, and small test
tensors so individual test files stay focused on assertions.
"""

import pytest
import torch

from configs import DiffusionConfig, SolverConfig
from models.diffusion import make_noise_schedule, build_ddim_timestep_sequence, precompute_rho_schedule


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@pytest.fixture
def diffusion_cfg():
    """Default DDPM diffusion config (T=1000, linear schedule)."""
    return DiffusionConfig()


@pytest.fixture
def solver_cfg():
    """Default DiffPIR solver config (Table 3 baseline)."""
    return SolverConfig()


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

@pytest.fixture
def noise_schedule(diffusion_cfg):
    """Pre-computed DDPM noise schedule from the default DiffusionConfig."""
    return make_noise_schedule(diffusion_cfg)


# ---------------------------------------------------------------------------
# Small test tensors
# ---------------------------------------------------------------------------

@pytest.fixture
def small_image():
    """A small seeded (B=2, C=3, H=16, W=16) test image for shape/dtype checks."""
    torch.manual_seed(0)
    return torch.randn(2, 3, 16, 16)


@pytest.fixture
def single_image():
    """A single (B=1, C=3, H=32, W=32) test image."""
    torch.manual_seed(1)
    return torch.randn(1, 3, 32, 32)
