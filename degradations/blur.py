"""
degradations/blur.py — Image deblurring degradation operator and data subproblem solver.

The deblurring forward model (circular convolution) is:
    y = x ⊗ k + n                                     (Eq. 27)

The closed-form data subproblem solution via FFT is:
    x₀ = F⁻¹( (F̄(k)·F(y) + ρ_t·F(z₀)) / (F̄(k)·F(k) + ρ_t) )   (Eq. 28)

where F and F⁻¹ denote the 2D FFT and its inverse.

Paper: Appendix B.2, Eqs. 27–28.
"""

from __future__ import annotations

import torch
from torch import Tensor

from configs import BlurConfig
from interfaces import PnPSolver


# ---------------------------------------------------------------------------
# Degradation operator
# ---------------------------------------------------------------------------

class BlurDegradation:
    """Image blurring degradation: circular convolution y = x ⊗ k.

    State:
        kernel: Blur kernel k, shape (H_k, W_k), float32.
        cfg: BlurConfig describing blur mode and kernel parameters.
    """

    def __init__(self, kernel: Tensor, cfg: BlurConfig) -> None:
        """Initialise with a pre-computed blur kernel.

        Args:
            kernel: Blur kernel k, shape (H_k, W_k), float32, sums to 1.
            cfg: BlurConfig.
        """
        self.kernel = kernel
        self.cfg = cfg

    def apply(self, x: Tensor) -> Tensor:
        """Apply the blur operator: y = x ⊗ k (circular convolution).

        Args:
            x: Clean image, shape (B, C, H, W), float32.

        Returns:
            y: Blurred image, same shape as x, float32.

        Paper: Eq. 27 (noiseless forward pass; noise is added separately).
        """
        raise NotImplementedError


def build_gaussian_kernel(
    size: int,
    std: float,
    device: str | None = None,
) -> Tensor:
    """Build a normalised 2D Gaussian blur kernel.

    Args:
        size: Kernel side length in pixels (e.g. 61).
        std: Gaussian standard deviation σ_k in pixels.
        device: Target device string.

    Returns:
        kernel: Float32 tensor, shape (size, size), sums to 1.

    Paper: Section 4.1 — 61×61 Gaussian kernel with σ_k = 3.0.
    """
    raise NotImplementedError


def build_motion_kernel(
    size: int,
    intensity: float,
    device: str | None = None,
) -> Tensor:
    """Build a random motion blur kernel following the paper's protocol.

    Args:
        size: Kernel side length in pixels (e.g. 61).
        intensity: Kernel intensity value (paper uses 0.5).
        device: Target device string.

    Returns:
        kernel: Float32 tensor, shape (size, size), sums to 1.

    Paper: Section 4.1 — random motion kernels, 61×61, intensity 0.5,
        generated following [8].
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Data subproblem solver
# ---------------------------------------------------------------------------

def blur_data_step_fft(
    x0_prior: Tensor,
    y: Tensor,
    kernel: Tensor,
    rho_t: float,
) -> Tensor:
    """Closed-form FFT solution to the deblurring data subproblem.

    Assumes circular convolution.  Solves:
        argmin_x ||y − x⊗k||² + ρ_t ||x − x̂₀_prior||²

    via the analytic formula in the Fourier domain:
        x₀ = F⁻¹( (F̄(k)·F(y) + ρ_t·F(z₀)) / (F̄(k)·F(k) + ρ_t) )  (Eq. 28)

    Args:
        x0_prior: Prior estimate from the denoiser, (B, C, H, W), float32.
        y: Blurred measurement, same spatial shape as x0_prior, float32.
        kernel: Blur kernel k, shape (H_k, W_k), float32.
        rho_t: Penalty coefficient ρ_t at the current timestep.

    Returns:
        x0_data: Data-consistent estimate, same shape as x0_prior, float32.

    Paper: Eq. 28.
    """
    raise NotImplementedError


class BlurPnPSolver(PnPSolver):
    """PnPSolver adapter for the deblurring data subproblem.

    Delegates to ``blur_data_step_fft``.  Holds no additional state.
    """

    def data_step(
        self,
        x0_prior: Tensor,
        y: Tensor,
        degradation: BlurDegradation,
        rho_t: float,
    ) -> Tensor:
        """Solve the deblurring data subproblem via FFT (Eq. 28).

        Args:
            x0_prior: Prior estimate, (B, C, H, W), float32.
            y: Blurred measurement, same spatial shape, float32.
            degradation: BlurDegradation providing the kernel k.
            rho_t: Penalty coefficient ρ_t.

        Returns:
            x0_data: Data-consistent estimate, same shape as x0_prior, float32.

        Paper: Eq. 28.
        """
        raise NotImplementedError
