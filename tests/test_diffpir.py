"""Tests for restoration/diffpir.py — DiffPIR outer sampling loop.

Uses mock denoisers and solvers to verify loop orchestration, shape contracts,
and iteration counts without requiring pretrained model weights.
"""

import pytest
import torch

from configs import DiffusionConfig, SolverConfig
from interfaces import DenoiserPrior, PnPSolver
from restoration.diffpir import diffpir_restore, build_noise_scheduler
from restoration.pnp import hqs_step


# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------

class IdentityDenoiser(DenoiserPrior):
    """Returns x_t unchanged (identity prior — x̂₀ = x_t)."""

    def __init__(self):
        self.call_count = 0

    def denoise(self, x_t, t, noise_schedule):
        self.call_count += 1
        return x_t


class IdentitySolver(PnPSolver):
    """Returns x0_prior unchanged (no data correction)."""

    def data_step(self, x0_prior, y, rho_t):
        return x0_prior


class HalfDenoiser(DenoiserPrior):
    """Returns 0.5 * x_t — non-trivial but numerically stable."""

    def denoise(self, x_t, t, noise_schedule):
        return 0.5 * x_t


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def noise_schedule():
    return build_noise_scheduler(DiffusionConfig())


@pytest.fixture
def fast_cfg():
    return SolverConfig(n_steps=5, lambda_=8.0, sigma_n=0.05, zeta=0.0)


# ---------------------------------------------------------------------------
# build_noise_scheduler (smoke tests — detailed math in test_diffusion.py)
# ---------------------------------------------------------------------------

class TestBuildNoiseScheduler:

    def test_returns_dict(self):
        s = build_noise_scheduler(DiffusionConfig())
        assert isinstance(s, dict)

    def test_required_keys(self):
        s = build_noise_scheduler(DiffusionConfig())
        assert {"T", "beta", "alpha", "alpha_bar", "sigma_bar"}.issubset(s.keys())


# ---------------------------------------------------------------------------
# diffpir_restore — shape and dtype
# ---------------------------------------------------------------------------

class TestDiffpirRestoreContracts:

    def test_output_shape(self, fast_cfg, noise_schedule):
        """Output must have the same shape as the degraded input y."""
        y = torch.randn(1, 3, 32, 32)
        result = diffpir_restore(fast_cfg, y, IdentityDenoiser(), IdentitySolver(),
                                 hqs_step, noise_schedule)
        assert result.shape == y.shape

    def test_output_dtype_float32(self, fast_cfg, noise_schedule):
        """Output must be float32."""
        y = torch.randn(1, 3, 16, 16)
        result = diffpir_restore(fast_cfg, y, IdentityDenoiser(), IdentitySolver(),
                                 hqs_step, noise_schedule)
        assert result.dtype == torch.float32

    def test_batch_dimension_preserved(self, fast_cfg, noise_schedule):
        """Batch size must be preserved end-to-end."""
        y = torch.randn(4, 3, 16, 16)
        result = diffpir_restore(fast_cfg, y, IdentityDenoiser(), IdentitySolver(),
                                 hqs_step, noise_schedule)
        assert result.shape[0] == 4

    def test_finite_output(self, fast_cfg, noise_schedule):
        """Output must not contain NaN or Inf."""
        y = torch.randn(1, 3, 16, 16)
        result = diffpir_restore(fast_cfg, y, IdentityDenoiser(), IdentitySolver(),
                                 hqs_step, noise_schedule)
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# diffpir_restore — iteration count
# ---------------------------------------------------------------------------

class TestDiffpirRestoreIterations:

    def test_denoiser_called_n_steps_times(self, fast_cfg, noise_schedule):
        """The denoiser must be called exactly n_steps times."""
        y = torch.randn(1, 3, 16, 16)
        denoiser = IdentityDenoiser()
        diffpir_restore(fast_cfg, y, denoiser, IdentitySolver(), hqs_step, noise_schedule)
        assert denoiser.call_count == fast_cfg.n_steps

    def test_various_n_steps(self, noise_schedule):
        """Denoiser call count must equal n_steps for any NFE value."""
        for n in [1, 3, 10, 20]:
            cfg = SolverConfig(n_steps=n, zeta=0.0, sigma_n=0.05)
            denoiser = IdentityDenoiser()
            y = torch.randn(1, 3, 8, 8)
            diffpir_restore(cfg, y, denoiser, IdentitySolver(), hqs_step, noise_schedule)
            assert denoiser.call_count == n


# ---------------------------------------------------------------------------
# diffpir_restore — determinism
# ---------------------------------------------------------------------------

class TestDiffpirRestoreDeterminism:

    def test_deterministic_with_zeta_zero(self, noise_schedule):
        """With ζ=0 and the same x_T, two runs must produce identical outputs."""
        cfg = SolverConfig(n_steps=3, zeta=0.0, sigma_n=0.05)
        torch.manual_seed(0)
        y = torch.randn(1, 3, 16, 16)

        torch.manual_seed(7)
        r1 = diffpir_restore(cfg, y, IdentityDenoiser(), IdentitySolver(),
                             hqs_step, noise_schedule)
        torch.manual_seed(7)
        r2 = diffpir_restore(cfg, y, IdentityDenoiser(), IdentitySolver(),
                             hqs_step, noise_schedule)
        torch.testing.assert_close(r1, r2)

    def test_stochastic_with_zeta_positive(self, noise_schedule):
        """With ζ>0, two runs without seeding should (almost certainly) differ."""
        cfg = SolverConfig(n_steps=5, zeta=0.5, sigma_n=0.05)
        y = torch.randn(1, 3, 16, 16)
        r1 = diffpir_restore(cfg, y, IdentityDenoiser(), IdentitySolver(),
                             hqs_step, noise_schedule)
        r2 = diffpir_restore(cfg, y, IdentityDenoiser(), IdentitySolver(),
                             hqs_step, noise_schedule)
        assert not torch.allclose(r1, r2)


# ---------------------------------------------------------------------------
# diffpir_restore — plug-and-play extensibility
# ---------------------------------------------------------------------------

class TestDiffpirRestorePlugAndPlay:

    def test_accepts_any_denoiser_prior(self, fast_cfg, noise_schedule):
        """diffpir_restore must work with any DenoiserPrior subclass."""
        y = torch.randn(1, 3, 16, 16)
        result = diffpir_restore(fast_cfg, y, HalfDenoiser(), IdentitySolver(),
                                 hqs_step, noise_schedule)
        assert result.shape == y.shape
        assert torch.isfinite(result).all()

    def test_accepts_any_pnp_step(self, fast_cfg, noise_schedule):
        """diffpir_restore must work with any callable pnp_step."""
        from restoration.pnp import drs_step
        y = torch.randn(1, 3, 16, 16)
        result = diffpir_restore(fast_cfg, y, IdentityDenoiser(), IdentitySolver(),
                                 drs_step, noise_schedule)
        assert result.shape == y.shape

    def test_accepts_any_pnp_solver(self, fast_cfg, noise_schedule):
        """diffpir_restore must work with any PnPSolver subclass."""

        class ScaleSolver(PnPSolver):
            def data_step(self, x0_prior, y, rho_t):
                return x0_prior * 0.9

        y = torch.randn(1, 3, 16, 16)
        result = diffpir_restore(fast_cfg, y, IdentityDenoiser(), ScaleSolver(),
                                 hqs_step, noise_schedule)
        assert result.shape == y.shape
