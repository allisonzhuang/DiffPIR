"""
train/trainer.py — Training loop orchestration for the diffusion model.

Trainer holds state (model, loss, optimizer, checkpoint manager, config) and
delegates all logic to standalone functions and the CheckpointManager / Loss
modules.  No training or checkpointing logic lives in the class itself.

DiffPIR is training-free in its paper evaluation, but a Trainer is provided
for ablations that require fine-tuning the denoiser prior.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from configs import DiffusionConfig, TrainingConfig
from train.checkpoint import CheckpointManager
from train.loss import diffusion_loss


# ---------------------------------------------------------------------------
# Standalone training functions
# ---------------------------------------------------------------------------

def run_training_step(
    model: nn.Module,
    batch: Dict[str, Tensor],
    noise_schedule: Dict[str, Tensor],
    optimizer: Optimizer,
    cfg: DiffusionConfig,
) -> float:
    """Execute one gradient update step.

    Samples a random timestep t for each sample in the batch, adds noise,
    runs the forward pass, computes the loss (Eq. 4), and updates the model.

    Args:
        model: U-Net noise predictor ε_θ.
        batch: Dictionary with key ``'image'`` containing a (B, C, H, W) float32
            tensor of clean images x₀.
        noise_schedule: Pre-computed schedule from ``make_noise_schedule``.
        optimizer: Gradient-based optimiser.
        cfg: DiffusionConfig specifying T and dtype.

    Returns:
        loss_val: Scalar loss value for this step (Python float).

    Paper: Eq. 4 training objective.
    """
    raise NotImplementedError


def run_validation(
    model: nn.Module,
    dataloader: DataLoader,
    noise_schedule: Dict[str, Tensor],
    cfg: DiffusionConfig,
    device: torch.device | str,
) -> Dict[str, float]:
    """Compute average loss over the validation set.

    Args:
        model: U-Net noise predictor ε_θ in eval mode.
        dataloader: DataLoader yielding validation batches.
        noise_schedule: Pre-computed schedule from ``make_noise_schedule``.
        cfg: DiffusionConfig.
        device: Target device.

    Returns:
        metrics: Dictionary with at least ``'val_loss'`` (mean validation loss).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates the diffusion model training loop.

    Delegates all per-step logic to ``run_training_step``, loss computation
    to ``diffusion_loss``, and persistence to ``CheckpointManager``.

    State:
        model: U-Net to be trained.
        optimizer: Gradient optimiser.
        ckpt_manager: Handles checkpoint saving / loading.
        noise_schedule: Pre-computed DDPM schedule tensors.
        cfg: Joint training + diffusion config.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        ckpt_manager: CheckpointManager,
        noise_schedule: Dict[str, Tensor],
        diffusion_cfg: DiffusionConfig,
        training_cfg: TrainingConfig,
    ) -> None:
        """Initialise the Trainer.

        Args:
            model: U-Net noise predictor.
            optimizer: Optimiser (e.g. Adam).
            ckpt_manager: CheckpointManager instance.
            noise_schedule: Pre-computed schedule from ``make_noise_schedule``.
            diffusion_cfg: DiffusionConfig.
            training_cfg: TrainingConfig.
        """
        self.model = model
        self.optimizer = optimizer
        self.ckpt_manager = ckpt_manager
        self.noise_schedule = noise_schedule
        self.diffusion_cfg = diffusion_cfg
        self.training_cfg = training_cfg

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """Run the full training loop for ``training_cfg.n_epochs`` epochs.

        Delegates per-step updates to ``run_training_step`` and checkpointing
        to ``self.ckpt_manager``.  Does not implement training logic itself.

        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation; evaluated each epoch.
        """
        raise NotImplementedError

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation and return metrics.

        Delegates to ``run_validation``.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            metrics: Dictionary of metric name → value.
        """
        raise NotImplementedError
