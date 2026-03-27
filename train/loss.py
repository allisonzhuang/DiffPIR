"""
train/loss.py — Diffusion training loss functions.

Paper: Section 2.1, Eq. 4 — denoising score matching objective.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def diffusion_loss(
    eps_pred: Tensor,
    eps_target: Tensor,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Compute the (optionally weighted) denoising score matching loss.

    Implements the simplified training objective from Ho et al. [24] with
    optional time-dependent weighting γ(t) as in Eq. 4:

        L = E_t { γ(t) · E_{x₀,ε} [ ||ε_θ(x_t, t) − ε||² ] }

    Args:
        eps_pred: Predicted noise ε_θ(x_t, t), shape (B, C, H, W), float32.
        eps_target: Ground-truth noise ε ~ N(0, I), same shape, float32.
        weights: Optional per-sample or per-timestep weights γ(t),
            shape (B,) or scalar.  If None, uniform weighting (γ=1).

    Returns:
        loss: Scalar loss tensor, float32.

    Paper: Eq. 4.
    """
    raise NotImplementedError
