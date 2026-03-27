"""
train/checkpoint.py — Checkpoint saving and loading utilities.

CheckpointManager holds state (directory, retention policy) and delegates
the actual I/O to standalone functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Standalone I/O functions
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    """Serialise a state dictionary to disk using torch.save.

    Args:
        path: Destination file path (.pt or .pth).
        state: Dictionary containing model state_dict, optimizer state_dict,
            step, and any other metadata.
    """
    raise NotImplementedError


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Deserialise a checkpoint from disk.

    Args:
        path: Path to the checkpoint file.

    Returns:
        state: Dictionary as saved by ``save_checkpoint``.
    """
    raise NotImplementedError


def list_checkpoints(ckpt_dir: Path) -> list[Path]:
    """List all checkpoint files in a directory, sorted by step (ascending).

    Args:
        ckpt_dir: Directory containing checkpoint files.

    Returns:
        paths: Sorted list of checkpoint file paths.
    """
    raise NotImplementedError


def prune_old_checkpoints(ckpt_dir: Path, keep_last_n: int) -> None:
    """Delete all but the most recent ``keep_last_n`` checkpoints.

    Args:
        ckpt_dir: Directory containing checkpoint files.
        keep_last_n: Number of checkpoints to keep.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CheckpointManager class
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages saving and loading of model checkpoints.

    State:
        ckpt_dir: Directory where checkpoints are written.
        keep_last_n: Number of most-recent checkpoints to retain.
    """

    def __init__(self, ckpt_dir: Path, keep_last_n: int = 5) -> None:
        """Initialise the manager.

        Args:
            ckpt_dir: Directory for checkpoint files.  Created if absent.
            keep_last_n: Maximum number of checkpoints to keep on disk.
        """
        self.ckpt_dir = ckpt_dir
        self.keep_last_n = keep_last_n

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save a checkpoint and prune old ones.

        Args:
            model: Model whose state_dict will be saved.
            optimizer: Optimizer whose state_dict will be saved.
            step: Global training step (used to name the file).
            metrics: Optional dict of metric name → value to store alongside.

        Returns:
            path: Path to the newly written checkpoint file.
        """
        raise NotImplementedError

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
    ) -> int:
        """Load the most recent checkpoint into model (and optionally optimizer).

        Args:
            model: Model to load weights into.
            optimizer: Optional optimizer to restore state into.

        Returns:
            step: Global training step stored in the checkpoint.
        """
        raise NotImplementedError

    def load(
        self,
        path: Path,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
    ) -> int:
        """Load a specific checkpoint by path.

        Args:
            path: Path to the checkpoint file.
            model: Model to load weights into.
            optimizer: Optional optimizer to restore state into.

        Returns:
            step: Global training step stored in the checkpoint.
        """
        raise NotImplementedError
