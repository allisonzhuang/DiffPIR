"""
configs.py — Python dataclasses for all DiffPIR configuration.

No YAML, no argparse, no external config files.  All hyperparameters live here.
Each degradation type has its own dedicated config.
Hyperparameter values from Table 3 (Appendix B.1) are noted in comments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import torch


# ---------------------------------------------------------------------------
# Diffusion schedule
# ---------------------------------------------------------------------------

@dataclass
class DiffusionConfig:
    """Configuration for the DDPM/DDIM noise schedule.

    Paper: Section 2.2, Eqs. 5–6.  The linear beta schedule from Ho et al. [24]
    is the default; T=1000 betas span [beta_start, beta_end].
    """

    T: int = 1000
    """Total number of diffusion timesteps used during training."""

    beta_start: float = 1e-4
    """Starting value of the linear noise schedule β_1."""

    beta_end: float = 2e-2
    """Ending value of the linear noise schedule β_T."""

    schedule_type: Literal["linear", "cosine"] = "linear"
    """Noise schedule type.  'linear' matches DDPM [24]; 'cosine' matches [41]."""

    dtype: torch.dtype = torch.float32
    """All diffusion tensors are kept in float32 as required by the paper."""


# ---------------------------------------------------------------------------
# HQS / DiffPIR sampling loop
# ---------------------------------------------------------------------------

@dataclass
class SolverConfig:
    """Configuration for the DiffPIR HQS sampling loop.

    Hyperparameter defaults match Table 3 for NFE=100, FFHQ, σ_n=0.05.
    """

    lambda_: float = 8.0
    """Regularisation strength λ controlling the balance between data and prior.

    Paper: Eq. 1, ablated in Figure 8.  Larger λ → stronger data fidelity.
    """

    zeta: float = 0.3
    """Noise injection variance ζ controlling stochasticity of the reverse process.

    Paper: Eq. 15, ablated in Figure 8.  ζ=0 → deterministic (DDIM-like).
    """

    sigma_n: float = 0.05
    """Known measurement noise standard deviation σ_n.

    Paper: Eq. 1.  Set to 0.0 for noiseless experiments (Table 2).
    """

    n_steps: int = 100
    """Number of reverse diffusion steps (NFEs).  Paper uses 20 or 100."""

    t_start: int = 1000
    """Starting timestep t_start for the reverse diffusion.

    Paper: Section 4.4, Figure 7.  Can be < T to skip early noisy steps.
    """


# ---------------------------------------------------------------------------
# Degradation configs — one per task
# ---------------------------------------------------------------------------

@dataclass
class InpaintConfig:
    """Configuration for the image inpainting degradation.

    Paper: Appendix B.2, Eq. 25–26.  Only the noiseless case (σ_n=0) is
    considered for inpainting in the paper.
    """

    mask_type: Literal["box", "random", "file"] = "box"
    """Inpainting mask type.  Paper uses 128×128 box or random 50% pixel masks."""

    mask_box_size: int = 128
    """Size of the square box mask in pixels.  Paper: 128×128 centred."""

    mask_random_fraction: float = 0.5
    """Fraction of pixels masked for random inpainting.  Paper: 0.5."""

    mask_path: Optional[Path] = None
    """Path to a custom mask image file (used when mask_type='file')."""


@dataclass
class BlurConfig:
    """Configuration for the image deblurring degradation.

    Paper: Appendix B.2, Eqs. 27–28.  Deblurring is solved via FFT using
    the circular convolution assumption.
    """

    blur_mode: Literal["gaussian", "motion"] = "gaussian"
    """Blur kernel type.  Paper uses 61×61 Gaussian (std=3) or random motion kernels."""

    kernel_size: int = 61
    """Blur kernel spatial size.  Paper: 61×61 for both Gaussian and motion blur."""

    gaussian_std: float = 3.0
    """Standard deviation of the Gaussian blur kernel.

    Paper: σ_k = 3.0.
    """

    motion_intensity: float = 0.5
    """Intensity value for the random motion blur kernel (used when blur_mode='motion').

    Paper: intensity 0.5, kernel size 61×61.
    """


@dataclass
class SRConfig:
    """Configuration for the super-resolution degradation.

    Paper: Appendix B.2, Eqs. 29–31.  Bicubic downsampling is applied to the
    ground-truth image to produce the low-resolution measurement.
    """

    scale_factor: int = 4
    """Bicubic downsampling factor sf.  Paper uses 4×, 8×, 16×."""

    solver: Literal["ibp", "fft"] = "fft"
    """Data subproblem solver.

    'ibp': iterative back-projection (Eq. 30) — more hyperparameters.
    'fft': closed-form FFT solution (Eq. 31) — fewer hyperparameters, preferred.
    """

    ibp_gamma: float = 0.9
    """Step size γ for IBP iterations (Eq. 30).  Decays as γ_t = γ/(1+ρ_t)."""

    ibp_n_iter: int = 6
    """Number of IBP iterations per timestep (Eq. 30)."""


# ---------------------------------------------------------------------------
# Type alias for the union of all degradation configs
# ---------------------------------------------------------------------------

DegradationConfig = Union[InpaintConfig, BlurConfig, SRConfig]
"""Union type for all supported degradation configurations.

Use isinstance checks or a task-specific factory to route to the right solver.
"""


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for training the diffusion model from scratch.

    DiffPIR itself is training-free (it uses pre-trained weights), but this
    config is provided for completeness and for ablation experiments that
    require fine-tuning the denoiser.
    """

    lr: float = 2e-5
    """Learning rate for the Adam optimiser."""

    batch_size: int = 16
    """Training batch size."""

    n_epochs: int = 100
    """Total number of training epochs."""

    ckpt_dir: Path = Path("checkpoints")
    """Directory where model checkpoints are saved."""

    log_dir: Path = Path("logs")
    """Directory for TensorBoard / WandB logs."""

    keep_last_n: int = 5
    """Number of most-recent checkpoints to retain."""

    seed: int = 42
    """Global random seed for reproducibility."""
