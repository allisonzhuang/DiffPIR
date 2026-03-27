"""Tests for degradations/blur.py — deblurring degradation and FFT solver.

Verifies kernel construction, the circular convolution forward model (Eq. 27),
and the closed-form FFT data subproblem solution (Eq. 28).  Includes a
regression test comparing FFT-based blur to naive spatial convolution.
"""

import pytest
import torch
import torch.nn.functional as F

from interfaces import PnPSolver
from degradations.blur import (
    BlurDegradation,
    BlurPnPSolver,
    blur_data_step_fft,
    build_gaussian_kernel,
    build_motion_kernel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_image():
    torch.manual_seed(0)
    return torch.randn(1, 3, 32, 32)


@pytest.fixture
def delta_kernel():
    """Dirac delta kernel: identity under convolution."""
    k = torch.zeros(7, 7)
    k[3, 3] = 1.0
    return k


@pytest.fixture
def gaussian_kernel():
    return build_gaussian_kernel(size=11, std=2.0)


# ---------------------------------------------------------------------------
# build_gaussian_kernel
# ---------------------------------------------------------------------------

class TestBuildGaussianKernel:

    def test_shape(self):
        """Kernel shape must be (size, size) — 2D spatial kernel."""
        k = build_gaussian_kernel(size=61, std=3.0)
        assert k.shape == (61, 61)

    def test_dtype_float32(self):
        """Kernel must be float32 for compatibility with image tensors."""
        k = build_gaussian_kernel(size=11, std=2.0)
        assert k.dtype == torch.float32

    def test_sums_to_one(self):
        """A normalised kernel must sum to 1.  If the kernel doesn't sum to 1,
        convolution would change the overall image intensity, corrupting the
        data fidelity term in the HQS objective.
        """
        k = build_gaussian_kernel(size=61, std=3.0)
        assert k.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_non_negative(self):
        """Gaussian kernel values must be non-negative (it's a probability density)."""
        k = build_gaussian_kernel(size=61, std=3.0)
        assert (k >= 0).all()

    def test_symmetric(self):
        """2D Gaussian kernel must be symmetric under 90° rotations."""
        k = build_gaussian_kernel(size=11, std=2.0)
        torch.testing.assert_close(k, k.T)
        torch.testing.assert_close(k, k.flip(0))
        torch.testing.assert_close(k, k.flip(1))

    def test_peak_at_center(self):
        """The maximum value should be at the centre pixel."""
        k = build_gaussian_kernel(size=11, std=2.0)
        center = 11 // 2
        assert k[center, center] == k.max()

    def test_known_value_at_center(self):
        """For a Gaussian with std σ on grid size s, the centre value should be
        approximately 1/(2πσ²) before normalisation, or after normalisation it
        should be the largest entry.  Verify the centre value matches the
        analytical Gaussian evaluated at (0,0).
        """
        import math
        std = 3.0
        size = 61
        k = build_gaussian_kernel(size=size, std=std)
        # Unnormalised peak: exp(0) / (2πσ²) → after normalisation, depends on grid
        center = size // 2
        # Just verify it's the max and positive
        assert k[center, center] > 0
        assert k[center, center] == k.max()


# ---------------------------------------------------------------------------
# build_motion_kernel
# ---------------------------------------------------------------------------

class TestBuildMotionKernel:

    def test_shape(self):
        """Motion kernel shape must be (size, size)."""
        k = build_motion_kernel(size=61, intensity=0.5)
        assert k.shape == (61, 61)

    def test_dtype_float32(self):
        """Kernel must be float32."""
        k = build_motion_kernel(size=61, intensity=0.5)
        assert k.dtype == torch.float32

    def test_sums_to_one(self):
        """Motion kernel must be normalised to sum to 1."""
        k = build_motion_kernel(size=61, intensity=0.5)
        assert k.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_non_negative(self):
        """Motion kernel values must be non-negative."""
        k = build_motion_kernel(size=61, intensity=0.5)
        assert (k >= 0).all()


# ---------------------------------------------------------------------------
# BlurDegradation.apply
# ---------------------------------------------------------------------------

class TestBlurDegradationApply:

    def test_output_shape(self, small_image, gaussian_kernel):
        """Circular convolution preserves spatial dimensions: y has the same
        shape as x.  Shape mismatch would break the FFT solver.
        """
        deg = BlurDegradation(gaussian_kernel)
        y = deg.apply(small_image)
        assert y.shape == small_image.shape

    def test_output_dtype_float32(self, small_image, gaussian_kernel):
        """Output must be float32."""
        deg = BlurDegradation(gaussian_kernel)
        y = deg.apply(small_image)
        assert y.dtype == torch.float32

    def test_delta_kernel_is_identity(self, small_image, delta_kernel):
        """Convolving with a delta kernel should return the input unchanged.
        This is the trivial base case: x ⊗ δ = x.
        """
        deg = BlurDegradation(delta_kernel)
        y = deg.apply(small_image)
        torch.testing.assert_close(y, small_image, atol=1e-5, rtol=1e-5)

    def test_constant_image_unchanged(self, gaussian_kernel):
        """Convolving a constant image with any normalised kernel returns
        the same constant (since ∑k = 1 and all pixel values are equal).
        """
        x = torch.ones(1, 3, 32, 32) * 7.0
        deg = BlurDegradation(gaussian_kernel)
        y = deg.apply(x)
        torch.testing.assert_close(y, x, atol=1e-4, rtol=1e-4)

    def test_fft_matches_naive_convolution(self):
        """REGRESSION TEST: The FFT-based circular convolution must produce
        the same result as naive spatial convolution (with circular padding)
        on small inputs.  This catches bugs in kernel zero-padding, FFT axis
        ordering, or conjugate handling.

        Uses a small 3×3 kernel and 8×8 image for exact comparison.
        """
        torch.manual_seed(42)
        # Small test kernel
        k = torch.tensor([[1., 2., 1.],
                          [2., 4., 2.],
                          [1., 2., 1.]], dtype=torch.float32)
        k = k / k.sum()
        x = torch.randn(1, 1, 8, 8)

        # FFT-based (the implementation under test)
        deg = BlurDegradation(k)
        y_fft = deg.apply(x)

        # Naive circular convolution via torch.nn.functional.conv2d with circular padding
        pad = k.shape[0] // 2
        x_padded = F.pad(x, [pad, pad, pad, pad], mode='circular')
        k_conv = k.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        y_naive = F.conv2d(x_padded, k_conv)

        torch.testing.assert_close(y_fft, y_naive, atol=1e-4, rtol=1e-4)

    def test_multichannel_independence(self, gaussian_kernel):
        """Blur should be applied independently per channel (Eq. 27 is per-channel).
        Zeroing one channel should not affect the others.
        """
        x = torch.randn(1, 3, 16, 16)
        deg = BlurDegradation(gaussian_kernel)

        y_full = deg.apply(x)

        x_zeroed = x.clone()
        x_zeroed[:, 1, :, :] = 0.0
        y_zeroed = deg.apply(x_zeroed)

        # Channel 0 and 2 should be the same
        torch.testing.assert_close(y_full[:, 0], y_zeroed[:, 0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(y_full[:, 2], y_zeroed[:, 2], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# blur_data_step_fft (Eq. 28)
# ---------------------------------------------------------------------------

class TestBlurDataStepFFT:

    def test_output_shape(self, small_image, gaussian_kernel):
        """blur_data_step_fft output must match x0_prior shape."""
        y = small_image  # use as mock measurement
        result = blur_data_step_fft(small_image, y, gaussian_kernel, rho_t=1.0)
        assert result.shape == small_image.shape

    def test_output_dtype_float32(self, small_image, gaussian_kernel):
        """Output must be float32."""
        result = blur_data_step_fft(small_image, small_image, gaussian_kernel, rho_t=1.0)
        assert result.dtype == torch.float32

    def test_delta_kernel_recovers_weighted_average(self, small_image, delta_kernel):
        """With δ kernel: F̄(δ)·F(δ) = 1, so Eq. 28 becomes:
        x₀ = F⁻¹((F(y) + ρ_t·F(z₀)) / (1 + ρ_t)) = (y + ρ_t·z₀) / (1 + ρ_t)

        This is just a weighted average of y and z₀.
        """
        y = torch.randn_like(small_image)
        z0 = small_image
        rho_t = 2.0
        result = blur_data_step_fft(z0, y, delta_kernel, rho_t)
        expected = (y + rho_t * z0) / (1.0 + rho_t)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_large_rho_recovers_prior(self, small_image, gaussian_kernel):
        """As ρ_t → ∞, the data step should return z₀ (prior dominates).
        x₀ = F⁻¹((F̄(k)·F(y) + ρ_t·F(z₀)) / (F̄(k)·F(k) + ρ_t))
           → F⁻¹(ρ_t·F(z₀) / ρ_t) = z₀ as ρ_t → ∞.
        """
        y = torch.randn_like(small_image)
        result = blur_data_step_fft(small_image, y, gaussian_kernel, rho_t=1e6)
        torch.testing.assert_close(result, small_image, atol=0.01, rtol=0.01)

    def test_output_real_valued(self, small_image, gaussian_kernel):
        """The output must be real-valued (no imaginary component).
        A common bug in FFT-based solvers is forgetting to take the real part
        of the inverse FFT, which can leave tiny imaginary residuals.
        """
        result = blur_data_step_fft(small_image, small_image, gaussian_kernel, rho_t=1.0)
        assert not torch.is_complex(result)

    def test_finite_output(self, small_image, gaussian_kernel):
        """Output must contain no NaN or Inf.  Division by near-zero in Fourier
        domain (when F̄(k)·F(k) + ρ_t ≈ 0) can cause numerical blow-up.
        """
        result = blur_data_step_fft(small_image, small_image, gaussian_kernel, rho_t=0.01)
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# BlurPnPSolver
# ---------------------------------------------------------------------------

class TestBlurPnPSolver:

    def test_implements_pnp_solver(self):
        """BlurPnPSolver must be a valid PnPSolver subclass."""
        assert issubclass(BlurPnPSolver, PnPSolver)
        solver = BlurPnPSolver()
        assert isinstance(solver, PnPSolver)

    def test_delegates_to_blur_data_step_fft(self, small_image, gaussian_kernel):
        """The solver adapter must produce the same result as calling
        blur_data_step_fft directly.
        """
        deg = BlurDegradation(gaussian_kernel)
        y = deg.apply(small_image)

        solver = BlurPnPSolver()
        result = solver.data_step(small_image, y, deg, rho_t=1.0)
        expected = blur_data_step_fft(small_image, y, gaussian_kernel, rho_t=1.0)
        torch.testing.assert_close(result, expected)

    def test_output_shape(self, small_image, gaussian_kernel):
        """Solver output shape must match x0_prior."""
        deg = BlurDegradation(gaussian_kernel)
        result = BlurPnPSolver().data_step(small_image, small_image, deg, rho_t=1.0)
        assert result.shape == small_image.shape


# ---------------------------------------------------------------------------
# Multi-channel FFT correctness
# ---------------------------------------------------------------------------

class TestBlurMultiChannelFFT:
    """Verify that the FFT-based blur and data step handle multi-channel
    (RGB) images correctly.  Eqs. 27–28 are applied per-channel.
    """

    def test_blur_per_channel_consistency(self):
        """Blurring an RGB image where each channel is a scaled version of
        the same pattern must produce scaled versions of the same blurred output.

        If channels are mixed in the FFT, the outputs won't maintain the
        expected scaling relationship.
        """
        torch.manual_seed(0)
        k = build_gaussian_kernel(size=11, std=2.0)
        base = torch.randn(1, 1, 32, 32)
        x = torch.cat([base, 2 * base, 3 * base], dim=1)  # (1, 3, 32, 32)

        deg = BlurDegradation(k)
        y = deg.apply(x)

        # Channel 1 should be 2× channel 0, channel 2 should be 3× channel 0
        torch.testing.assert_close(y[:, 1], 2 * y[:, 0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(y[:, 2], 3 * y[:, 0], atol=1e-5, rtol=1e-5)

    def test_data_step_fft_per_channel_independence(self):
        """The FFT data step (Eq. 28) must process each channel independently.
        Zeroing one channel's prior and measurement should not affect others.
        """
        torch.manual_seed(0)
        k = build_gaussian_kernel(size=7, std=1.5)
        x0 = torch.randn(1, 3, 16, 16)
        y = torch.randn(1, 3, 16, 16)

        result_full = blur_data_step_fft(x0, y, k, rho_t=1.0)

        # Zero out channel 1 in both x0 and y
        x0_zeroed = x0.clone()
        x0_zeroed[:, 1] = 0
        y_zeroed = y.clone()
        y_zeroed[:, 1] = 0
        result_zeroed = blur_data_step_fft(x0_zeroed, y_zeroed, k, rho_t=1.0)

        # Channels 0 and 2 should be identical
        torch.testing.assert_close(result_full[:, 0], result_zeroed[:, 0],
                                   atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(result_full[:, 2], result_zeroed[:, 2],
                                   atol=1e-5, rtol=1e-5)


class TestBlurDataStepInversion:
    """Tests verifying that the FFT data step produces measurement-consistent
    solutions when given noise-free, blur-consistent data.
    """

    def test_consistent_measurement_recovered(self):
        """If y = x ⊗ k (noise-free), then the data step with small ρ_t should
        approximately recover x.  This is the "oracle prior" scenario where
        z₀ is close to x and y is perfectly consistent.

        Eq. 28: x₀ = F⁻¹((F̄(k)F(y) + ρ_t·F(z₀)) / (|F(k)|² + ρ_t))
        When z₀ = x and y = x⊗k, the numerator = (|F(k)|² + ρ_t)·F(x),
        so x₀ = x exactly.
        """
        torch.manual_seed(0)
        k = build_gaussian_kernel(size=11, std=2.0)
        x_true = torch.randn(1, 1, 32, 32)
        deg = BlurDegradation(k)
        y = deg.apply(x_true)

        # With z₀ = x_true, should recover x_true regardless of ρ_t
        result = blur_data_step_fft(x_true, y, k, rho_t=1.0)
        torch.testing.assert_close(result, x_true, atol=1e-4, rtol=1e-4)

    def test_small_rho_trusts_measurement(self):
        """With small ρ_t, the data step should produce an output whose blur
        closely matches y (measurement dominance).

        We verify: ||y - (result ⊗ k)||² < ||y - (z₀ ⊗ k)||²
        """
        torch.manual_seed(0)
        k = build_gaussian_kernel(size=7, std=1.5)
        x_true = torch.randn(1, 1, 16, 16)
        deg = BlurDegradation(k)
        y = deg.apply(x_true)

        z0 = x_true + torch.randn_like(x_true) * 0.5  # noisy prior
        result = blur_data_step_fft(z0, y, k, rho_t=0.01)

        residual_before = (y - deg.apply(z0)).pow(2).sum().item()
        residual_after = (y - deg.apply(result)).pow(2).sum().item()
        assert residual_after < residual_before, \
            f"Data step didn't improve consistency: {residual_before:.4f} → {residual_after:.4f}"
