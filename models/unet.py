"""
models/unet.py — U-Net wrapper implementing the DenoiserPrior interface.

The U-Net is a pre-trained DDPM noise predictor ε_θ(x_t, t).  This module
wraps it to satisfy the DenoiserPrior ABC and exposes a loader for
pre-trained checkpoints (DPIR [57] weights for FFHQ; guided-diffusion [14]
weights for ImageNet, as used in the paper).

Paper: Section 3.1; Algorithm 1 line 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from configs import DiffusionConfig
from interfaces import DenoiserPrior
from models.diffusion import predict_x0


class UNetDenoiser(DenoiserPrior):
    """Wraps a pre-trained DDPM U-Net as a plug-and-play denoiser prior.

    The U-Net predicts the noise ε_θ(x_t, t) added to x_t.  This class
    converts that prediction to x̂₀ via ``predict_x0`` and satisfies the
    ``DenoiserPrior`` interface required by the HQS loop.

    State:
        model: Pre-trained PyTorch U-Net (noise predictor).
        noise_schedule: Pre-computed schedule dict from ``make_noise_schedule``.
    """

    def __init__(
        self,
        model: nn.Module,
        noise_schedule: Dict[str, Tensor],
    ) -> None:
        """Initialise the wrapper.

        Args:
            model: Pre-trained U-Net that takes (x_t, t) and returns ε_θ.
            noise_schedule: Pre-computed schedule from ``make_noise_schedule``.
        """
        self.model = model
        self.noise_schedule = noise_schedule

    def denoise(
        self,
        x_t: Tensor,
        t: int,
        noise_schedule: Dict[str, Tensor],
    ) -> Tensor:
        """Run one forward pass of the U-Net and return the x̂₀ prediction.

        Calls the U-Net to get ε_θ(x_t, t), then inverts via ``predict_x0``
        (Eq. 11) to obtain x̂₀⁽ᵗ⁾.

        Args:
            x_t: Noisy image at timestep t, shape (B, C, H, W), float32.
            t: Integer diffusion timestep (1-indexed).
            noise_schedule: Pre-computed schedule dict (same as stored at init).

        Returns:
            x0_pred: Predicted clean image x̂₀⁽ᵗ⁾, same shape as x_t, float32.

        Paper: Algorithm 1 line 3; Eq. 11.
        """
        raise NotImplementedError


def load_pretrained_unet(
    ckpt_path: Path,
    cfg: DiffusionConfig,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load a pre-trained DDPM U-Net from a checkpoint file.

    Supports the checkpoint format used by DPIR [57] (FFHQ) and
    guided-diffusion [14] (ImageNet), which are the two models used in the
    paper's experiments.

    Args:
        ckpt_path: Path to the checkpoint file (.pth or .pt).
        cfg: DiffusionConfig used to verify schedule compatibility.
        device: Target device for the loaded model.

    Returns:
        model: Loaded U-Net in eval mode, on the specified device.

    Paper: Section 4.1 — uses pre-trained models from [14] and [7].
    """
    raise NotImplementedError


def build_unet(cfg: DiffusionConfig) -> nn.Module:
    """Instantiate a U-Net architecture compatible with the DDPM training objective.

    Used when training from scratch rather than loading pre-trained weights.
    Architecture follows Ho et al. [24] with the improvements from [41].

    Args:
        cfg: DiffusionConfig specifying T and dtype.

    Returns:
        model: Uninitialised U-Net ready for training, float32.

    Paper: Section 2.2; training objective Eq. 4.
    """
    raise NotImplementedError
