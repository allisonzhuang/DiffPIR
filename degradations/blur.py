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
import torch.nn.functional as F
from torch import Tensor

from configs import BlurConfig
from interfaces import PnPSolver
from motionblur.motionblur import Kernel


def fft(x: Tensor) -> Tensor:
    return torch.fft.fft2(x)


def ifft(x: Tensor) -> Tensor:
    return torch.fft.ifft2(x).real


def kernel_fft(k: Tensor, h: int, w: int) -> Tensor:
    kh, kw = k.shape[-2:]

    pad_h = h - kh
    pad_w = w - kw
    k = F.pad(k, (0, pad_w, 0, pad_h))

    k = torch.roll(k, shifts=(-(kh // 2), -(kw // 2)), dims=(-2, -1))

    return fft(k)


# ---------------------------------------------------------------------------
# Degradation operator
# ---------------------------------------------------------------------------


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
    ij = torch.linspace(
        -(size - 1) / 2, (size - 1) / 2, steps=size, device=device
    )
    x, y = torch.meshgrid(ij, ij, indexing="ij")

    k = torch.exp(-(x**2 + y**2) / (2 * std**2))
    k = k / k.sum()

    return k.to(torch.float32)


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
    k = torch.from_numpy(Kernel((size, size), intensity).kernelMatrix)
    k = k.to(device).to(torch.float32)

    return k


def build_blur_kernel(cfg: BlurConfig, device: str | None = None) -> Tensor:
    if cfg.blur_mode == "gaussian":
        return build_gaussian_kernel(cfg.kernel_size, cfg.gaussian_std, device)
    elif cfg.blur_mode == "motion":
        return build_motion_kernel(
            cfg.kernel_size, cfg.motion_intensity, device
        )
    else:
        raise ValueError(f"Unrecognized blur mode: {cfg.blur_mode}")


class BlurDegradation:
    """Image blurring degradation: circular convolution y = x ⊗ k.

    State:
        kernel: Blur kernel k, shape (H_k, W_k), float32.
    """

    def __init__(self, cfg: BlurConfig, device: str | None = None) -> None:
        """Initialise with a pre-computed blur kernel.

        Args:
            kernel: Blur kernel k, shape (H_k, W_k), float32, sums to 1.
        """
        self.kernel = build_blur_kernel(cfg, device)

    def apply(self, x: Tensor) -> Tensor:
        """Apply the blur operator: y = x ⊗ k (circular convolution).

        Args:
            x: Clean image, shape (B, C, H, W), float32.

        Returns:
            y: Blurred image, same shape as x, float32.

        Paper: Eq. 27 (noiseless forward pass; noise is added separately).
        """
        h, w = x.shape[-2:]

        x_hat = fft(x)
        k_hat = kernel_fft(self.kernel, h, w)

        y_hat = k_hat * x_hat
        y = ifft(y_hat)

        return y


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
    h, w = x0_prior.shape[-2:]

    k_hat = kernel_fft(kernel, h, w)
    k_hat_conj = torch.conj(k_hat)
    y_hat = fft(y)
    z_hat = fft(x0_prior)

    x_hat = (k_hat_conj * y_hat + rho_t * z_hat) / (k_hat_conj * k_hat + rho_t)

    x = ifft(x_hat)
    return x


class BlurPnPSolver(PnPSolver):
    """PnPSolver adapter for the deblurring data subproblem.

    Delegates to ``blur_data_step_fft``.  Holds no additional state.
    """

    def __init__(self, kernel: Tensor) -> None:
        self.kernel = kernel

    def data_step(
        self,
        x0_prior: Tensor,
        y: Tensor,
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

        return blur_data_step_fft(x0_prior, y, self.kernel, rho_t)
