"""Tests for degradations/sr.py — super-resolution degradation and data solvers.

Verifies the bicubic downsampling forward model (Eq. 29), the IBP iterative
solver (Eq. 30), and the FFT closed-form solver (Eq. 31), as well as the
SRPnPSolver routing logic.
"""

import pytest
import torch

from configs import SRConfig
from interfaces import PnPSolver
from degradations.sr import (
    SRDegradation,
    SRPnPSolver,
    sr_data_step_ibp,
    sr_data_step_fft,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hr_image():
    """High-resolution test image, divisible by scale factor 4."""
    torch.manual_seed(0)
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def sr_cfg():
    return SRConfig(scale_factor=4, solver="fft")


@pytest.fixture
def sr_degradation(sr_cfg):
    return SRDegradation(sr_cfg)


# ---------------------------------------------------------------------------
# SRDegradation.apply (Eq. 29)
# ---------------------------------------------------------------------------

class TestSRDegradationApply:

    def test_output_shape(self, hr_image, sr_degradation):
        """Output must be (B, C, H//sf, W//sf).  A shape error here means the
        downsampling factor is wrong, which would break everything downstream.
        """
        y = sr_degradation.apply(hr_image)
        assert y.shape == (1, 3, 16, 16)

    def test_output_dtype_float32(self, hr_image, sr_degradation):
        """Output must be float32."""
        y = sr_degradation.apply(hr_image)
        assert y.dtype == torch.float32

    def test_different_scale_factors(self):
        """Verify correct downsampling for various scale factors."""
        x = torch.randn(1, 3, 64, 64)
        for sf in [2, 4, 8]:
            deg = SRDegradation(SRConfig(scale_factor=sf))
            y = deg.apply(x)
            assert y.shape == (1, 3, 64 // sf, 64 // sf), f"Wrong shape for sf={sf}"

    def test_batch_dimension_preserved(self, sr_degradation):
        """Batch dimension must be preserved through downsampling."""
        x = torch.randn(4, 3, 32, 32)
        y = sr_degradation.apply(x)
        assert y.shape[0] == 4

    def test_constant_image_preserved(self, sr_degradation):
        """A constant-valued image should remain constant after bicubic
        downsampling (bicubic interpolation preserves constants).
        """
        x = torch.ones(1, 3, 64, 64) * 5.0
        y = sr_degradation.apply(x)
        torch.testing.assert_close(y, torch.ones(1, 3, 16, 16) * 5.0, atol=0.01, rtol=0.01)


# ---------------------------------------------------------------------------
# SRDegradation.upsample
# ---------------------------------------------------------------------------

class TestSRDegradationUpsample:

    def test_output_shape(self, sr_degradation):
        """upsample must restore the original HR spatial dimensions."""
        y = torch.randn(1, 3, 16, 16)
        x_up = sr_degradation.upsample(y, target_size=(64, 64))
        assert x_up.shape == (1, 3, 64, 64)

    def test_output_dtype_float32(self, sr_degradation):
        """Output must be float32."""
        y = torch.randn(1, 3, 16, 16)
        x_up = sr_degradation.upsample(y, target_size=(64, 64))
        assert x_up.dtype == torch.float32

    def test_roundtrip_constant_image(self, sr_degradation):
        """Downsampling then upsampling a constant image should recover the
        constant (bicubic preserves constants in both directions).
        """
        x = torch.ones(1, 3, 64, 64) * 3.0
        y = sr_degradation.apply(x)
        x_up = sr_degradation.upsample(y, target_size=(64, 64))
        torch.testing.assert_close(x_up, x, atol=0.05, rtol=0.05)


# ---------------------------------------------------------------------------
# sr_data_step_ibp (Eq. 30)
# ---------------------------------------------------------------------------

class TestSRDataStepIBP:

    def test_output_shape(self, hr_image, sr_degradation):
        """IBP output must have the same HR shape as x0_prior."""
        y = sr_degradation.apply(hr_image)
        result = sr_data_step_ibp(hr_image, y, sr_degradation, gamma_t=0.5, n_iter=3)
        assert result.shape == hr_image.shape

    def test_output_dtype_float32(self, hr_image, sr_degradation):
        """Output must be float32."""
        y = sr_degradation.apply(hr_image)
        result = sr_data_step_ibp(hr_image, y, sr_degradation, gamma_t=0.5, n_iter=3)
        assert result.dtype == torch.float32

    def test_reduces_residual(self, sr_degradation):
        """After IBP iterations, the data consistency residual ||y − x₀↓_{sf}||
        should decrease compared to the initial prior.  This is the fundamental
        convergence property of iterative back-projection.
        """
        torch.manual_seed(42)
        # Start from a "perfect" HR image so we know the true LR
        x_gt = torch.randn(1, 3, 64, 64)
        y = sr_degradation.apply(x_gt)

        # Use a noisy prior
        x0_prior = x_gt + torch.randn_like(x_gt) * 0.5

        # Measure residual before IBP
        residual_before = (y - sr_degradation.apply(x0_prior)).norm().item()

        # Run IBP
        x0_ibp = sr_data_step_ibp(x0_prior, y, sr_degradation, gamma_t=0.5, n_iter=6)
        residual_after = (y - sr_degradation.apply(x0_ibp)).norm().item()

        assert residual_after < residual_before, \
            f"IBP did not reduce residual: {residual_before:.4f} → {residual_after:.4f}"

    def test_more_iterations_better_consistency(self, sr_degradation):
        """More IBP iterations should bring the result closer to data consistency.
        This tests that the iterative refinement actually converges.
        """
        torch.manual_seed(0)
        x_gt = torch.randn(1, 3, 32, 32)
        y = sr_degradation.apply(x_gt)
        x0_prior = x_gt + torch.randn_like(x_gt) * 0.3

        result_3 = sr_data_step_ibp(x0_prior, y, sr_degradation, gamma_t=0.5, n_iter=3)
        result_10 = sr_data_step_ibp(x0_prior, y, sr_degradation, gamma_t=0.5, n_iter=10)

        res_3 = (y - sr_degradation.apply(result_3)).norm().item()
        res_10 = (y - sr_degradation.apply(result_10)).norm().item()
        assert res_10 <= res_3 + 1e-6


# ---------------------------------------------------------------------------
# sr_data_step_fft (Eq. 31)
# ---------------------------------------------------------------------------

class TestSRDataStepFFT:

    def test_output_shape(self, hr_image):
        """FFT solver output must have the same HR shape as x0_prior."""
        sf = 4
        y = torch.randn(1, 3, 16, 16)
        result = sr_data_step_fft(hr_image, y, scale_factor=sf, rho_t=1.0)
        assert result.shape == hr_image.shape

    def test_output_dtype_float32(self, hr_image):
        """Output must be float32."""
        y = torch.randn(1, 3, 16, 16)
        result = sr_data_step_fft(hr_image, y, scale_factor=4, rho_t=1.0)
        assert result.dtype == torch.float32

    def test_large_rho_recovers_prior(self, hr_image):
        """As ρ_t → ∞, the FFT solver should return z₀ (prior dominates).
        This is the same limiting behaviour as Eq. 28 for deblurring.
        """
        y = torch.randn(1, 3, 16, 16)
        result = sr_data_step_fft(hr_image, y, scale_factor=4, rho_t=1e6)
        torch.testing.assert_close(result, hr_image, atol=0.05, rtol=0.05)

    def test_finite_output(self, hr_image):
        """Output must contain no NaN or Inf values.  The FFT solver involves
        division in Fourier domain which can blow up numerically.
        """
        y = torch.randn(1, 3, 16, 16)
        result = sr_data_step_fft(hr_image, y, scale_factor=4, rho_t=0.1)
        assert torch.isfinite(result).all()

    def test_output_real_valued(self, hr_image):
        """Output must be real (not complex).  Common FFT bug: forgetting
        to take the real part of ifft.
        """
        y = torch.randn(1, 3, 16, 16)
        result = sr_data_step_fft(hr_image, y, scale_factor=4, rho_t=1.0)
        assert not torch.is_complex(result)


# ---------------------------------------------------------------------------
# SRPnPSolver
# ---------------------------------------------------------------------------

class TestSRPnPSolver:

    def test_implements_pnp_solver(self):
        """SRPnPSolver must be a valid PnPSolver subclass."""
        assert issubclass(SRPnPSolver, PnPSolver)
        solver = SRPnPSolver()
        assert isinstance(solver, PnPSolver)

    def test_routes_to_fft_by_default(self, hr_image):
        """With cfg.solver='fft' (the default), SRPnPSolver should route to
        sr_data_step_fft.  Verify by checking the output matches.
        """
        cfg = SRConfig(scale_factor=4, solver="fft")
        deg = SRDegradation(cfg)
        y = deg.apply(hr_image)

        solver = SRPnPSolver()
        result = solver.data_step(hr_image, y, deg, rho_t=1.0)
        expected = sr_data_step_fft(hr_image, y, scale_factor=4, rho_t=1.0)
        torch.testing.assert_close(result, expected)

    def test_routes_to_ibp(self, hr_image):
        """With cfg.solver='ibp', SRPnPSolver should route to sr_data_step_ibp."""
        cfg = SRConfig(scale_factor=4, solver="ibp")
        deg = SRDegradation(cfg)
        y = deg.apply(hr_image)

        solver = SRPnPSolver()
        result = solver.data_step(hr_image, y, deg, rho_t=1.0)
        # Verify shape (can't easily match exact output without knowing gamma_t)
        assert result.shape == hr_image.shape
        assert result.dtype == torch.float32

    def test_output_shape(self, hr_image):
        """Solver output shape must match x0_prior."""
        cfg = SRConfig(scale_factor=4)
        deg = SRDegradation(cfg)
        y = deg.apply(hr_image)
        result = SRPnPSolver().data_step(hr_image, y, deg, rho_t=1.0)
        assert result.shape == hr_image.shape


# ---------------------------------------------------------------------------
# SR data step consistency and convergence
# ---------------------------------------------------------------------------

class TestSRDataStepConsistency:
    """Verify that the SR data steps produce measurement-consistent solutions."""

    def test_ibp_zero_iterations_returns_prior(self):
        """With n_iter=0 (if supported) or n_iter=1 and gamma_t=0, IBP should
        return close to the prior (no back-projection correction applied).

        More precisely, with gamma_t=0: x₀ ← z₀ - 0·(residual) = z₀.
        """
        cfg = SRConfig(scale_factor=4, solver="ibp")
        deg = SRDegradation(cfg)
        x0 = torch.randn(1, 3, 32, 32)
        y = deg.apply(x0)
        result = sr_data_step_ibp(x0, y, deg, gamma_t=0.0, n_iter=6)
        torch.testing.assert_close(result, x0, atol=1e-6, rtol=1e-6)

    def test_ibp_constant_image_stays_constant(self):
        """IBP on a constant image (whose LR is also constant) should return
        the same constant.  Bicubic down/up of a constant is the constant.
        """
        cfg = SRConfig(scale_factor=4)
        deg = SRDegradation(cfg)
        x_const = torch.ones(1, 3, 64, 64) * 5.0
        y = deg.apply(x_const)
        result = sr_data_step_ibp(x_const, y, deg, gamma_t=0.5, n_iter=6)
        torch.testing.assert_close(result, x_const, atol=0.1, rtol=0.1)

    def test_fft_constant_image_stays_constant(self):
        """FFT solver on a constant image should preserve the constant.
        The Fourier transform of a constant is a delta at DC; the formula
        should reduce to the identity for constant inputs.
        """
        x_const = torch.ones(1, 3, 64, 64) * 5.0
        y = torch.ones(1, 3, 16, 16) * 5.0  # LR of a constant is a constant
        result = sr_data_step_fft(x_const, y, scale_factor=4, rho_t=1.0)
        torch.testing.assert_close(result, x_const, atol=0.1, rtol=0.1)

    def test_fft_and_ibp_agree_directionally(self):
        """Both solvers should move the prior closer to data consistency.
        Given a noisy prior and clean LR measurement, both should reduce
        ||y - result↓||.
        """
        torch.manual_seed(0)
        cfg_fft = SRConfig(scale_factor=4, solver="fft")
        cfg_ibp = SRConfig(scale_factor=4, solver="ibp")
        deg_fft = SRDegradation(cfg_fft)
        deg_ibp = SRDegradation(cfg_ibp)

        x_gt = torch.randn(1, 3, 32, 32)
        y = deg_fft.apply(x_gt)
        x0_prior = x_gt + torch.randn_like(x_gt) * 0.5

        residual_before = (y - deg_fft.apply(x0_prior)).pow(2).sum().item()

        result_fft = sr_data_step_fft(x0_prior, y, scale_factor=4, rho_t=0.1)
        residual_fft = (y - deg_fft.apply(result_fft)).pow(2).sum().item()

        result_ibp = sr_data_step_ibp(x0_prior, y, deg_ibp, gamma_t=0.5, n_iter=6)
        residual_ibp = (y - deg_ibp.apply(result_ibp)).pow(2).sum().item()

        assert residual_fft < residual_before, \
            f"FFT solver didn't improve consistency: {residual_before:.4f} → {residual_fft:.4f}"
        assert residual_ibp < residual_before, \
            f"IBP solver didn't improve consistency: {residual_before:.4f} → {residual_ibp:.4f}"


class TestSRDegradationEdgeCases:
    """Edge cases for the SR degradation operator."""

    def test_non_square_image(self):
        """SR degradation must work on non-square images as long as both
        dimensions are divisible by the scale factor.
        """
        cfg = SRConfig(scale_factor=4)
        deg = SRDegradation(cfg)
        x = torch.randn(1, 3, 64, 128)  # non-square
        y = deg.apply(x)
        assert y.shape == (1, 3, 16, 32)

    def test_upsample_non_square(self):
        """upsample must work on non-square images."""
        cfg = SRConfig(scale_factor=4)
        deg = SRDegradation(cfg)
        y = torch.randn(1, 3, 16, 32)
        x_up = deg.upsample(y, target_size=(64, 128))
        assert x_up.shape == (1, 3, 64, 128)

    def test_scale_factor_2(self):
        """Verify correctness for scale_factor=2 (different from default 4)."""
        cfg = SRConfig(scale_factor=2)
        deg = SRDegradation(cfg)
        x = torch.randn(1, 3, 32, 32)
        y = deg.apply(x)
        assert y.shape == (1, 3, 16, 16)
        x_up = deg.upsample(y, target_size=(32, 32))
        assert x_up.shape == (1, 3, 32, 32)
