"""
degradations/inpaint.py — Inpainting degradation operator and data subproblem solver.

The inpainting forward model is:
    y = M ⊙ x                           (Eq. 25)

The closed-form data subproblem solution is:
    x₀ = (M ⊙ y + ρ_t · z₀) / (M + ρ_t)   (Eq. 26)

Paper: Appendix B.2, Eqs. 25–26.
"""

from __future__ import annotations

import torch
from torch import Tensor

from configs import InpaintConfig
from interfaces import PnPSolver


# ---------------------------------------------------------------------------
# Degradation operator
# ---------------------------------------------------------------------------


class InpaintingDegradation:
    """Inpainting degradation: element-wise masking y = M ⊙ x.

    State:
        mask: Binary mask tensor M, shape (1, 1, H, W) or (B, 1, H, W), float32.
            1 = known pixel, 0 = missing pixel.
    """

    def __init__(self, mask: Tensor) -> None:
        """Initialise with a pre-computed mask tensor.

        Args:
            mask: Binary mask M, shape (1, 1, H, W) or (B, 1, H, W), float32.
                1 = observed, 0 = missing.
        """
        self.mask = mask

    def apply(self, x: Tensor) -> Tensor:
        """Apply the masking operator: y = M ⊙ x.

        Args:
            x: Clean image, shape (B, C, H, W), float32.

        Returns:
            y: Masked image, same shape as x, float32.

        Paper: Eq. 25.
        """
        return self.mask * x


def build_box_mask(
    height: int,
    width: int,
    box_size: int,
    device: str | None = None,
) -> Tensor:
    """Create a centred square box mask (0 inside the box, 1 outside).

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        box_size: Side length of the masked box region.
        device: Target device string.

    Returns:
        mask: Float32 tensor, shape (1, 1, H, W), values in {0, 1}.

    Paper: Section 4.1 — 128×128 box region following [8].
    """

    M = torch.ones((1, 1, height, width), device=device, dtype=torch.float32)

    i = (height - box_size) // 2
    j = (width - box_size) // 2

    M[..., i : i + box_size, j : j + box_size] = 0

    return M


def build_random_mask(
    height: int,
    width: int,
    fraction: float,
    device: str | None = None,
) -> Tensor:
    """Create a random pixel mask with a given missing fraction.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        fraction: Fraction of pixels to mask (set to 0).
        device: Target device string.

    Returns:
        mask: Float32 tensor, shape (1, 1, H, W), values in {0, 1}.

    Paper: Section 4.1 — random masks masking half the total pixels.
    """
    M = torch.rand((1, 1, height, width), device=device, dtype=torch.float32)

    return (M > fraction).float()


def build_mask(
    cfg: InpaintConfig, height: int, width: int, device: str | None = None
) -> Tensor:
    if cfg.mask_type == "box":
        return build_box_mask(height, width, cfg.mask_box_size, device)
    elif cfg.mask_type == "random":
        return build_random_mask(
            height, width, cfg.mask_random_fraction, device
        )
    elif cfg.mask_type == "file":
        raise NotImplementedError

    raise ValueError(f"Unrecognized mask type: {cfg.mask_type}")


# ---------------------------------------------------------------------------
# Data subproblem solver
# ---------------------------------------------------------------------------


def inpaint_data_step(
    x0_prior: Tensor,
    y: Tensor,
    mask: Tensor,
    rho_t: float,
) -> Tensor:
    """Closed-form solution to the inpainting data subproblem.

    Solves argmin_x ||y − M⊙x||² + ρ_t ||x − x̂₀_prior||² element-wise:

        x₀ = (M ⊙ y + ρ_t · x̂₀_prior) / (M + ρ_t)               (Eq. 26)

    Division and addition are element-wise.

    Args:
        x0_prior: Prior estimate from the denoiser, (B, C, H, W), float32.
        y: Masked measurement M ⊙ x_gt, same shape as x0_prior, float32.
        mask: Binary mask M, broadcastable to (B, C, H, W), float32.
        rho_t: Penalty coefficient ρ_t at the current timestep.

    Returns:
        x0_data: Data-consistent estimate, same shape as x0_prior, float32.

    Paper: Eq. 26.
    """

    x = mask * y + rho_t * x0_prior
    x = x / (mask + rho_t)

    return x


class InpaintingPnPSolver(PnPSolver):
    """PnPSolver adapter for the inpainting data subproblem.

    Delegates to ``inpaint_data_step``.  Holds no additional state.
    """

    def data_step(
        self,
        x0_prior: Tensor,
        y: Tensor,
        degradation: InpaintingDegradation,
        rho_t: float,
    ) -> Tensor:
        """Solve the inpainting data subproblem (Eq. 26).

        Args:
            x0_prior: Prior estimate, (B, C, H, W), float32.
            y: Masked measurement, same shape as x0_prior, float32.
            degradation: InpaintingDegradation providing the mask M.
            rho_t: Penalty coefficient ρ_t.

        Returns:
            x0_data: Data-consistent estimate, same shape as x0_prior, float32.

        Paper: Eq. 26.
        """

        return inpaint_data_step(x0_prior, y, degradation.mask, rho_t)
