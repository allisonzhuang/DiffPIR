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
        raise NotImplementedError

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
        raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
        raise NotImplementedError
