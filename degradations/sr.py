"""
degradations/sr.py — Super-resolution degradation operator and data subproblem solvers.

The SR forward model is:
    y = x ↓_{sf}^{bicubic} + n                        (Eq. 29)

Two data subproblem solvers are provided:

  IBP (iterative back-projection, Eq. 30):
    x₀ ← z₀ − γ_t · (y − z₀ ↓_{sf}^{bicubic}) ↑_{sf}^{bicubic}

  FFT (closed-form, Eq. 31 — preferred):
    x₀ = F⁻¹( (1/ρ_t) · (d − F(k) ⊙_s ((F(k)·d) ↓_s / ((F(k)·F(k)) ↓_s + ρ_t))) )

Paper: Appendix B.2, Eqs. 29–31.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from configs import SRConfig
from interfaces import PnPSolver


# ---------------------------------------------------------------------------
# Degradation operator
# ---------------------------------------------------------------------------


class SRDegradation:
    """Super-resolution degradation: bicubic downsampling y = x ↓_{sf}^{bicubic}.

    State:
        cfg: SRConfig describing scale factor and solver type.
    """

    def __init__(self, cfg: SRConfig) -> None:
        """Initialise with the SR configuration.

        Args:
            cfg: SRConfig specifying scale_factor and solver.
        """
        self.cfg = cfg

    def apply(self, x: Tensor) -> Tensor:
        """Bicubic downsample x by scale factor sf: y = x ↓_{sf}^{bicubic}.

        Args:
            x: High-resolution clean image, shape (B, C, H, W), float32.

        Returns:
            y: Low-resolution measurement, shape (B, C, H//sf, W//sf), float32.

        Paper: Eq. 29.
        """
        sf = 1 / self.cfg.scale_factor
        return F.interpolate(x, scale_factor=sf, mode="bicubic", antialias=True)

    def upsample(self, y: Tensor, target_size: tuple[int, int]) -> Tensor:
        """Bicubic upsample the low-resolution image back to HR size.

        Used in the IBP data step (Eq. 30).

        Args:
            y: Low-resolution image, shape (B, C, H_lr, W_lr), float32.
            target_size: (H_hr, W_hr) target spatial dimensions.

        Returns:
            x_up: Upsampled image, shape (B, C, H_hr, W_hr), float32.

        Paper: Eq. 30 — ↑_{sf}^{bicubic} operator.
        """
        return F.interpolate(y, target_size, mode="bicubic")


# ---------------------------------------------------------------------------
# Data subproblem solvers
# ---------------------------------------------------------------------------


def sr_data_step_ibp(
    x0_prior: Tensor,
    y: Tensor,
    degradation: SRDegradation,
    gamma_t: float,
    n_iter: int = 6,
) -> Tensor:
    """Iterative back-projection (IBP) solution to the SR data subproblem.

    Performs n_iter gradient steps toward consistency with y:
        x₀ ← z₀ − γ_t · (y − z₀ ↓_{sf}) ↑_{sf}                    (Eq. 30)

    Args:
        x0_prior: Prior estimate, shape (B, C, H, W), float32.
        y: LR measurement, shape (B, C, H//sf, W//sf), float32.
        degradation: SRDegradation providing apply() and upsample().
        gamma_t: Step size γ_t = γ / (1 + ρ_t).  Decreases over iterations.
        n_iter: Number of IBP iterations per timestep.

    Returns:
        x0_data: Data-consistent estimate, same shape as x0_prior, float32.

    Paper: Eq. 30.
    """
    x = x0_prior
    h, w = x.shape[-2:]

    for _ in range(n_iter):
        x = x + gamma_t * degradation.upsample(y - degradation.apply(x), (h, w))

    return x


def get_bicubic_kernel_fft(
    sf: int, h: int, w: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Generates the FFT of the approximated PyTorch bicubic kernel."""
    a = -0.75  # PyTorch bicubic coefficient

    def cubic(x: Tensor) -> Tensor:
        x = x.abs()
        res = torch.zeros_like(x)
        m1 = x <= 1
        res[m1] = (a + 2) * x[m1] ** 3 - (a + 3) * x[m1] ** 2 + 1
        m2 = (x > 1) & (x < 2)
        res[m2] = a * x[m2] ** 3 - 5 * a * x[m2] ** 2 + 8 * a * x[m2] - 4 * a
        return res

    # Sample the 1D kernel
    radius = 2 * sf
    x_grid = torch.arange(-radius, radius + 1, dtype=dtype, device=device) / sf
    k1d = cubic(x_grid)
    k1d = k1d / k1d.sum()

    k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)
    kh, kw = k2d.shape

    padded_k = F.pad(k2d, (0, w - kw, 0, h - kh))
    padded_k = torch.roll(
        padded_k, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1)
    )

    k_hat = torch.fft.fftn(padded_k, dim=(0, 1))
    return k_hat.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)


def sr_data_step_fft(
    x0_prior: Tensor,
    y: Tensor,
    scale_factor: int,
    rho_t: float,
) -> Tensor:
    """Closed-form FFT solution to the SR data subproblem (preferred).

    Uses an approximated bicubic kernel k and solves the data subproblem in
    the Fourier domain using the block-downsampling trick from [57]:

        d = F(k) · F(y ↑_{sf}) + ρ_t · F(z₀)
        x₀ = F⁻¹( (1/ρ_t) · (d − F(k) ⊙_s ((F(k)·d) ↓_s / ((F(k)·F(k)) ↓_s + ρ_t))) )
                                                                       (Eq. 31)

    where ⊙_s denotes distinct-block element-wise multiplication and ↓_s
    denotes block-averaging downsampling over s×s blocks.

    Args:
        x0_prior: Prior estimate, shape (B, C, H, W), float32.
        y: LR measurement, shape (B, C, H//sf, W//sf), float32.
        scale_factor: Downsampling factor sf.
        rho_t: Penalty coefficient ρ_t at the current timestep.

    Returns:
        x0_data: Data-consistent estimate, same shape as x0_prior, float32.

    Paper: Eq. 31; [57] Section 3.
    """
    b, c, h_lr, w_lr = y.shape
    h, w = x0_prior.shape[-2:]
    sf = scale_factor

    k_hat = get_bicubic_kernel_fft(sf, h, w, x0_prior.device, x0_prior.dtype)
    k_hat_conj = torch.conj(k_hat)

    y_up = torch.zeros_like(x0_prior)
    y_up[..., ::sf, ::sf] = y

    y_up_hat = torch.fft.fftn(y_up, dim=(-2, -1))
    z_hat = torch.fft.fftn(x0_prior, dim=(-2, -1))

    d = k_hat_conj * y_up_hat + rho_t * z_hat

    term1 = k_hat * d
    num = term1.view(b, c, sf, h_lr, sf, w_lr).mean(dim=(2, 4))

    term2 = (k_hat_conj * k_hat).real
    den = term2.view(1, 1, sf, h_lr, sf, w_lr).mean(dim=(2, 4)) + rho_t

    ratio = num / den
    ratio_expanded = ratio.view(b, c, 1, h_lr, 1, w_lr)
    k_hat_conj_aux = k_hat_conj.view(1, 1, sf, h_lr, sf, w_lr)

    M = (k_hat_conj_aux * ratio_expanded).view(b, c, h, w)

    x_hat = (d - M) / rho_t
    return torch.fft.ifftn(x_hat, dim=(-2, -1)).real


class SRPnPSolver(PnPSolver):
    """PnPSolver adapter for the super-resolution data subproblem.

    Routes to ``sr_data_step_fft`` or ``sr_data_step_ibp`` based on
    ``degradation.cfg.solver``.
    """

    def data_step(
        self,
        x0_prior: Tensor,
        y: Tensor,
        degradation: SRDegradation,
        rho_t: float,
    ) -> Tensor:
        """Solve the SR data subproblem (Eq. 30 or 31).

        Args:
            x0_prior: Prior estimate, (B, C, H, W), float32.
            y: LR measurement, (B, C, H//sf, W//sf), float32.
            degradation: SRDegradation providing cfg.solver, cfg.scale_factor, etc.
            rho_t: Penalty coefficient ρ_t.

        Returns:
            x0_data: Data-consistent HR estimate, same shape as x0_prior, float32.

        Paper: Eqs. 30–31.
        """

        if degradation.cfg.solver == "fft":
            return sr_data_step_fft(
                x0_prior, y, degradation.cfg.scale_factor, rho_t
            )

        n_iter = degradation.cfg.ibp_n_iter
        gamma_t = 1 / (1 + rho_t)

        return sr_data_step_ibp(x0_prior, y, degradation, gamma_t, n_iter)
