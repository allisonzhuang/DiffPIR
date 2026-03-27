"""Tests for restoration/hqs.py — Half-Quadratic Splitting step functions.

The HQS functions are thin orchestration wrappers: they delegate to the
DenoiserPrior, PnPSolver, and diffusion math.  Tests verify correct
delegation, shape/dtype contracts, and the re-noising step formula (Eq. 15).
"""

import math
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from interfaces import DenoiserPrior, PnPSolver
from restoration.hqs import hqs_prior_step, hqs_data_step, hqs_renoise_step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class IdentityDenoiser(DenoiserPrior):
    """Mock denoiser that returns x_t unchanged (identity prior)."""
    def denoise(self, x_t, t, noise_schedule):
        return x_t


class IdentitySolver(PnPSolver):
    """Mock solver that returns x0_prior unchanged (no data correction)."""
    def data_step(self, x0_prior, y, degradation, rho_t):
        return x0_prior


class ScaledDenoiser(DenoiserPrior):
    """Mock denoiser that returns 0.5 * x_t — easy to verify output numerically."""
    def denoise(self, x_t, t, noise_schedule):
        return 0.5 * x_t


@pytest.fixture
def small_image():
    torch.manual_seed(42)
    return torch.randn(2, 3, 16, 16)


@pytest.fixture
def noise_schedule():
    from models.diffusion import make_noise_schedule
    from configs import DiffusionConfig
    return make_noise_schedule(DiffusionConfig())


# ---------------------------------------------------------------------------
# hqs_prior_step
# ---------------------------------------------------------------------------

class TestHqsPriorStep:

    def test_output_shape(self, small_image, noise_schedule):
        """hqs_prior_step must return a tensor with the same shape as x_t.
        Shape mismatch between the prior and data steps would crash the HQS loop.
        """
        denoiser = IdentityDenoiser()
        result = hqs_prior_step(denoiser, small_image, t=500, noise_schedule=noise_schedule)
        assert result.shape == small_image.shape

    def test_output_dtype_float32(self, small_image, noise_schedule):
        """Output must be float32 for consistency with the noise schedule."""
        denoiser = IdentityDenoiser()
        result = hqs_prior_step(denoiser, small_image, t=500, noise_schedule=noise_schedule)
        assert result.dtype == torch.float32

    def test_delegates_to_denoiser(self, small_image, noise_schedule):
        """hqs_prior_step must call denoiser.denoise exactly once with the
        correct arguments.  This is a delegation contract — the prior step
        should not contain any denoising logic itself.
        """
        mock = MagicMock(spec=DenoiserPrior)
        mock.denoise.return_value = small_image
        hqs_prior_step(mock, small_image, t=500, noise_schedule=noise_schedule)
        mock.denoise.assert_called_once()
        args, kwargs = mock.denoise.call_args
        # Verify x_t and t were passed (either positional or keyword)
        call_values = list(args) + list(kwargs.values())
        assert any(isinstance(v, Tensor) for v in call_values)

    def test_identity_denoiser_returns_input(self, small_image, noise_schedule):
        """With the identity denoiser (returns x_t), the prior step should
        return x_t unchanged.  This verifies no extra transformation is applied.
        """
        denoiser = IdentityDenoiser()
        result = hqs_prior_step(denoiser, small_image, t=500, noise_schedule=noise_schedule)
        torch.testing.assert_close(result, small_image)

    def test_scaled_denoiser_output(self, small_image, noise_schedule):
        """With the ScaledDenoiser (returns 0.5*x_t), the prior step should
        return 0.5 * x_t.  Tests that the wrapper passes through the denoiser
        output without modification.
        """
        denoiser = ScaledDenoiser()
        result = hqs_prior_step(denoiser, small_image, t=500, noise_schedule=noise_schedule)
        torch.testing.assert_close(result, 0.5 * small_image)


# ---------------------------------------------------------------------------
# hqs_data_step
# ---------------------------------------------------------------------------

class TestHqsDataStep:

    def test_output_shape(self, small_image):
        """hqs_data_step must return a tensor with the same shape as x0_prior."""
        solver = IdentitySolver()
        result = hqs_data_step(solver, small_image, y=small_image,
                               degradation=None, rho_t=1.0)
        assert result.shape == small_image.shape

    def test_output_dtype_float32(self, small_image):
        """Output must be float32."""
        solver = IdentitySolver()
        result = hqs_data_step(solver, small_image, y=small_image,
                               degradation=None, rho_t=1.0)
        assert result.dtype == torch.float32

    def test_delegates_to_solver(self, small_image):
        """hqs_data_step must call solver.data_step exactly once.
        No data-step logic should live in the wrapper.
        """
        mock = MagicMock(spec=PnPSolver)
        mock.data_step.return_value = small_image
        hqs_data_step(mock, small_image, y=small_image, degradation=None, rho_t=1.0)
        mock.data_step.assert_called_once()

    def test_passes_rho_t_to_solver(self, small_image):
        """The penalty coefficient ρ_t must reach the solver unchanged.
        Corrupted ρ_t would silently break the data-prior balance.
        """
        mock = MagicMock(spec=PnPSolver)
        mock.data_step.return_value = small_image
        hqs_data_step(mock, small_image, y=small_image, degradation=None, rho_t=42.0)
        _, kwargs = mock.data_step.call_args
        args = mock.data_step.call_args[0]
        # rho_t should appear as 42.0 somewhere in the call
        all_args = list(args) + list(kwargs.values())
        assert 42.0 in all_args or any(
            isinstance(a, float) and a == pytest.approx(42.0) for a in all_args
        )


# ---------------------------------------------------------------------------
# hqs_renoise_step
# ---------------------------------------------------------------------------

class TestHqsRenoiseStep:

    def test_output_shape(self, small_image):
        """hqs_renoise_step must return a tensor with the same shape as x_t.
        This is the output fed to the next iteration of the HQS loop.
        """
        x0_data = torch.randn_like(small_image)
        result = hqs_renoise_step(small_image, x0_data,
                                  alpha_bar_t=0.5, alpha_bar_prev=0.6, zeta=0.3)
        assert result.shape == small_image.shape

    def test_output_dtype_float32(self, small_image):
        """Output must be float32."""
        x0_data = torch.randn_like(small_image)
        result = hqs_renoise_step(small_image, x0_data,
                                  alpha_bar_t=0.5, alpha_bar_prev=0.6, zeta=0.0)
        assert result.dtype == torch.float32

    def test_deterministic_when_zeta_zero(self, small_image):
        """With ζ=0, the re-noising step is deterministic (Eq. 15 with no
        stochastic term).  Two calls with identical inputs must give identical
        outputs.  This is important for reproducible DDIM sampling.
        """
        x0_data = torch.randn_like(small_image)
        r1 = hqs_renoise_step(small_image, x0_data, 0.5, 0.6, zeta=0.0)
        r2 = hqs_renoise_step(small_image, x0_data, 0.5, 0.6, zeta=0.0)
        torch.testing.assert_close(r1, r2)

    def test_stochastic_when_zeta_positive(self, small_image):
        """With ζ > 0, the √ζ · ε_t term injects fresh noise.  Two calls
        should produce different outputs (with overwhelming probability).
        """
        x0_data = torch.randn_like(small_image)
        r1 = hqs_renoise_step(small_image, x0_data, 0.5, 0.6, zeta=0.5)
        r2 = hqs_renoise_step(small_image, x0_data, 0.5, 0.6, zeta=0.5)
        assert not torch.allclose(r1, r2)

    def test_formula_zeta_zero(self, small_image):
        """Verify the deterministic formula (Eq. 15 with ζ=0):
        x_{t-1} = √ᾱ_{t-1}·x̂₀ + √(1 − ᾱ_{t-1})·ε̂

        where ε̂ = (x_t − √ᾱ_t·x̂₀) / √(1−ᾱ_t).

        This hand-check catches coefficient errors in the implementation.
        """
        ab_t = 0.4
        ab_prev = 0.6
        x0_data = torch.randn_like(small_image)
        # Manually compute ε̂
        eps_hat = (small_image - math.sqrt(ab_t) * x0_data) / math.sqrt(1 - ab_t)
        expected = math.sqrt(ab_prev) * x0_data + math.sqrt(1 - ab_prev) * eps_hat
        result = hqs_renoise_step(small_image, x0_data, ab_t, ab_prev, zeta=0.0)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_finite_output_at_extreme_alpha_bars(self, small_image):
        """At extreme ᾱ_t values (near 0 or 1), the re-noising step must
        remain numerically stable (no NaN/Inf from division by near-zero).
        """
        x0_data = torch.randn_like(small_image)
        # Near t=T: ᾱ_t ≈ 0
        result = hqs_renoise_step(small_image, x0_data,
                                  alpha_bar_t=0.001, alpha_bar_prev=0.002, zeta=0.0)
        assert torch.isfinite(result).all(), "Non-finite at ᾱ_t ≈ 0"

        # Near t=1: ᾱ_t ≈ 1
        result = hqs_renoise_step(small_image, x0_data,
                                  alpha_bar_t=0.999, alpha_bar_prev=0.9995, zeta=0.0)
        assert torch.isfinite(result).all(), "Non-finite at ᾱ_t ≈ 1"


# ---------------------------------------------------------------------------
# Integration: HQS loop convergence on trivial problem
# ---------------------------------------------------------------------------

class TestHqsIntegration:

    def test_identity_degradation_converges(self, noise_schedule):
        """With an identity denoiser (prior returns x_t) and identity solver
        (data step returns x0_prior), one full HQS iteration should produce
        a valid output of the correct shape.

        This is a minimal smoke test that the three steps compose correctly.
        """
        x_t = torch.randn(1, 3, 16, 16)
        denoiser = IdentityDenoiser()
        solver = IdentitySolver()

        # Prior step
        x0_prior = hqs_prior_step(denoiser, x_t, t=500, noise_schedule=noise_schedule)
        # Data step
        x0_data = hqs_data_step(solver, x0_prior, y=x_t, degradation=None, rho_t=1.0)
        # Re-noise step
        ab_t = noise_schedule["alpha_bars"][499].item()
        ab_prev = noise_schedule["alpha_bars"][498].item()
        x_prev = hqs_renoise_step(x_t, x0_data, ab_t, ab_prev, zeta=0.0)

        assert x_prev.shape == x_t.shape
        assert x_prev.dtype == torch.float32
        assert torch.isfinite(x_prev).all()

    def test_multi_step_hqs_stays_finite(self, noise_schedule):
        """Running multiple consecutive HQS iterations (simulating a short
        sampling chain) must remain numerically stable throughout.

        Uses identity denoiser/solver with decreasing timesteps to mimic
        the actual DiffPIR reverse process.  All intermediate x_t must be
        finite — a single NaN/Inf would poison all subsequent iterations.
        """
        from models.diffusion import build_ddim_timestep_sequence
        timesteps = build_ddim_timestep_sequence(t_start=1000, n_steps=10, T=1000)

        torch.manual_seed(42)
        x_t = torch.randn(1, 3, 16, 16)
        denoiser = IdentityDenoiser()
        solver = IdentitySolver()

        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_prev = timesteps[i + 1]
            ab_t = noise_schedule["alpha_bars"][t_curr - 1].item()
            ab_prev = noise_schedule["alpha_bars"][t_prev - 1].item()

            x0_prior = hqs_prior_step(denoiser, x_t, t=t_curr,
                                       noise_schedule=noise_schedule)
            x0_data = hqs_data_step(solver, x0_prior, y=x_t,
                                     degradation=None, rho_t=1.0)
            x_t = hqs_renoise_step(x_t, x0_data, ab_t, ab_prev, zeta=0.0)

            assert torch.isfinite(x_t).all(), \
                f"Non-finite values at iteration {i} (t={t_curr}→{t_prev})"

    def test_renoise_step_reduces_noise_level(self, noise_schedule):
        """After one HQS iteration, x_{t-1} should be at a lower noise level
        than x_t.  We verify this by checking that x_{t-1} has lower variance
        around the clean estimate than x_t does.

        This tests the fundamental property that the reverse process moves
        toward the clean image manifold.
        """
        t_curr = 500
        t_prev = 490
        ab_t = noise_schedule["alpha_bars"][t_curr - 1].item()
        ab_prev = noise_schedule["alpha_bars"][t_prev - 1].item()

        # Construct x_t from a known x0
        torch.manual_seed(0)
        x0_true = torch.randn(1, 3, 16, 16)
        eps = torch.randn_like(x0_true)
        x_t = math.sqrt(ab_t) * x0_true + math.sqrt(1 - ab_t) * eps

        # Use x0_true as the "perfect" denoiser output
        x_prev = hqs_renoise_step(x_t, x0_true, ab_t, ab_prev, zeta=0.0)

        # x_prev should be closer to x0_true than x_t is
        dist_before = (x_t - x0_true).pow(2).mean().item()
        dist_after = (x_prev - x0_true).pow(2).mean().item()
        assert dist_after < dist_before, \
            f"Re-noising didn't reduce distance to x0: {dist_before:.4f} → {dist_after:.4f}"
