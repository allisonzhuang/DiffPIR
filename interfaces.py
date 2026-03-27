"""
interfaces.py — Abstract base classes for DiffPIR components.

Swapping the denoiser prior or PnP solver never requires editing this file
or any existing file — only adding a new subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import Tensor


class DenoiserPrior(ABC):
    """Abstract generative denoiser prior used as the plug-and-play prior in HQS.

    Implementations wrap any off-the-shelf denoiser (DDPM U-Net, BM3D, DnCNN, …)
    and expose a uniform interface.  The prior subproblem solved here is:

        x̂₀⁽ᵗ⁾ = argmin_z  1/(2σ_t²) ||z - x_t||²  +  P(z)        (Eq. 12a)

    whose analytic solution for a score-based model is:

        x̂₀⁽ᵗ⁾ = (1/√ᾱ_t) (x_t − √(1−ᾱ_t) · ε_θ(x_t, t))          (Eq. 11)
    """

    @abstractmethod
    def denoise(
        self,
        x_t: Tensor,
        t: int,
        noise_schedule: Dict[str, Tensor],
    ) -> Tensor:
        """Solve the prior subproblem: return the predicted clean image x̂₀.

        Args:
            x_t: Noisy image at diffusion timestep t, shape (B, C, H, W), float32.
            t: Integer diffusion timestep (1-indexed, T → 1).
            noise_schedule: Pre-computed schedule dict with keys
                ``alpha_bars`` (shape T), ``betas`` (shape T), etc.

        Returns:
            x0_pred: Predicted clean image x̂₀⁽ᵗ⁾, same shape as x_t, float32.

        Paper: Algorithm 1 line 3, Eq. 11.
        """
        raise NotImplementedError


class PnPSolver(ABC):
    """Abstract plug-and-play solver for the data subproblem in HQS.

    Implementations provide a closed-form or iterative solution to:

        x̂₀⁽ᵗ⁾ = argmin_x  ||y − H(x)||²  +  ρ_t ||x − x̂₀⁽ᵗ⁾_prior||²   (Eq. 12b)

    for a specific degradation operator H.  New degradation types are added
    by subclassing this class in a new file — no existing files are modified.
    """

    @abstractmethod
    def data_step(
        self,
        x0_prior: Tensor,
        y: Tensor,
        degradation: Any,
        rho_t: float,
    ) -> Tensor:
        """Solve the data subproblem given the prior estimate x̂₀_prior.

        Args:
            x0_prior: Prior estimate of the clean image from the denoiser,
                shape (B, C, H, W), float32.
            y: Degraded measurement, shape varies by task, float32.
            degradation: Degradation object (InpaintingDegradation, BlurDegradation,
                SRDegradation, …) providing the operator H and its parameters.
            rho_t: Penalty coefficient ρ_t = λσ_n²/σ_t² at the current timestep.

        Returns:
            x0_data: Data-consistent estimate of the clean image, same shape
                as x0_prior, float32.

        Paper: Algorithm 1 line 4, Eq. 12b.
        """
        raise NotImplementedError
