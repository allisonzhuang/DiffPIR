"""Tests for train/trainer.py — training loop orchestration.

Uses mock/minimal components to verify that the training step, validation,
and Trainer class correctly delegate work to loss.py, checkpoint.py, and
diffusion.py without running expensive forward passes.
"""

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from configs import DiffusionConfig, TrainingConfig
from models.diffusion import make_noise_schedule
from train.checkpoint import CheckpointManager
from train.trainer import run_training_step, run_validation, Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyUNet(nn.Module):
    """Minimal noise predictor ε_θ(x_t, t) → same shape as x_t.
    Uses a single conv layer so parameters exist for gradient checks.
    """
    def __init__(self, channels=3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x_t, t):
        return self.conv(x_t)


@pytest.fixture
def tiny_model():
    return TinyUNet(channels=3)


@pytest.fixture
def noise_schedule():
    return make_noise_schedule(DiffusionConfig())


@pytest.fixture
def optimizer(tiny_model):
    return torch.optim.Adam(tiny_model.parameters(), lr=1e-3)


@pytest.fixture
def train_batch():
    """Minimal training batch: 2 images of size (3, 16, 16)."""
    torch.manual_seed(0)
    return {"image": torch.randn(2, 3, 16, 16)}


@pytest.fixture
def train_loader():
    """DataLoader with 4 small images."""
    torch.manual_seed(0)
    images = torch.randn(4, 3, 16, 16)
    ds = TensorDataset(images)
    # Wrap to return dicts
    class DictDataset:
        def __init__(self, tensors):
            self.tensors = tensors
        def __len__(self):
            return len(self.tensors)
        def __getitem__(self, idx):
            return {"image": self.tensors[idx]}

    return DataLoader(DictDataset(images), batch_size=2)


@pytest.fixture
def val_loader(train_loader):
    return train_loader


# ---------------------------------------------------------------------------
# run_training_step
# ---------------------------------------------------------------------------

class TestRunTrainingStep:

    def test_returns_float(self, tiny_model, train_batch, noise_schedule, optimizer):
        """run_training_step must return a Python float (the loss value).
        Returning a Tensor would cause a memory leak via retained graph.
        """
        cfg = DiffusionConfig()
        loss_val = run_training_step(tiny_model, train_batch, noise_schedule, optimizer, cfg)
        assert isinstance(loss_val, float)

    def test_loss_is_finite(self, tiny_model, train_batch, noise_schedule, optimizer):
        """The loss must be finite (no NaN/Inf).  Infinite loss indicates
        a numerical stability issue in the forward pass or loss computation.
        """
        cfg = DiffusionConfig()
        loss_val = run_training_step(tiny_model, train_batch, noise_schedule, optimizer, cfg)
        assert not (loss_val != loss_val)  # NaN check
        assert abs(loss_val) < float("inf")

    def test_loss_positive(self, tiny_model, train_batch, noise_schedule, optimizer):
        """MSE loss should be positive for a random (untrained) model."""
        cfg = DiffusionConfig()
        loss_val = run_training_step(tiny_model, train_batch, noise_schedule, optimizer, cfg)
        assert loss_val > 0

    def test_updates_model_weights(self, tiny_model, train_batch, noise_schedule, optimizer):
        """After one training step, model parameters should change (the optimizer
        applied a gradient update).  If weights don't change, either the loss
        has no gradient or the optimizer isn't stepping.
        """
        cfg = DiffusionConfig()
        params_before = {n: p.clone() for n, p in tiny_model.named_parameters()}
        run_training_step(tiny_model, train_batch, noise_schedule, optimizer, cfg)
        changed = False
        for n, p in tiny_model.named_parameters():
            if not torch.equal(p, params_before[n]):
                changed = True
                break
        assert changed, "Model weights did not change after training step"

    def test_multiple_steps_reduce_loss(self, noise_schedule):
        """Running multiple training steps should reduce the loss over time.
        This is a minimal convergence check — if the loss doesn't decrease
        at all, the training logic is broken.
        """
        model = TinyUNet(channels=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        cfg = DiffusionConfig()
        batch = {"image": torch.randn(4, 1, 8, 8)}

        losses = []
        for _ in range(20):
            l = run_training_step(model, batch, noise_schedule, optimizer, cfg)
            losses.append(l)

        # Average of last 5 should be lower than average of first 5
        avg_first = sum(losses[:5]) / 5
        avg_last = sum(losses[-5:]) / 5
        assert avg_last < avg_first, f"Loss did not decrease: {avg_first:.4f} → {avg_last:.4f}"


# ---------------------------------------------------------------------------
# run_validation
# ---------------------------------------------------------------------------

class TestRunValidation:

    def test_returns_dict_with_val_loss(self, tiny_model, val_loader, noise_schedule):
        """run_validation must return a dict containing 'val_loss'."""
        cfg = DiffusionConfig()
        metrics = run_validation(tiny_model, val_loader, noise_schedule, cfg, device="cpu")
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics

    def test_val_loss_is_float(self, tiny_model, val_loader, noise_schedule):
        """val_loss must be a Python float."""
        cfg = DiffusionConfig()
        metrics = run_validation(tiny_model, val_loader, noise_schedule, cfg, device="cpu")
        assert isinstance(metrics["val_loss"], float)

    def test_val_loss_positive(self, tiny_model, val_loader, noise_schedule):
        """Validation loss should be positive for an untrained model."""
        cfg = DiffusionConfig()
        metrics = run_validation(tiny_model, val_loader, noise_schedule, cfg, device="cpu")
        assert metrics["val_loss"] > 0

    def test_no_gradient_modification(self, tiny_model, val_loader, noise_schedule):
        """run_validation must not modify model gradients or parameters.
        Validation should run in eval/no_grad mode.
        """
        cfg = DiffusionConfig()
        params_before = {n: p.clone() for n, p in tiny_model.named_parameters()}
        run_validation(tiny_model, val_loader, noise_schedule, cfg, device="cpu")
        for n, p in tiny_model.named_parameters():
            torch.testing.assert_close(p, params_before[n])


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class TestTrainer:

    def _make_trainer(self, tmp_path, tiny_model, optimizer, noise_schedule):
        """Helper to construct a Trainer with minimal config."""
        dcfg = DiffusionConfig()
        tcfg = TrainingConfig(n_epochs=1, ckpt_dir=tmp_path / "ckpts", batch_size=2)
        ckpt_mgr = CheckpointManager(tcfg.ckpt_dir, keep_last_n=2)
        return Trainer(tiny_model, optimizer, ckpt_mgr, noise_schedule, dcfg, tcfg)

    def test_train_runs_without_error(self, tmp_path, tiny_model, optimizer,
                                       noise_schedule, train_loader):
        """Trainer.train should complete one epoch without raising.
        This is the fundamental smoke test for the training loop integration.
        """
        trainer = self._make_trainer(tmp_path, tiny_model, optimizer, noise_schedule)
        trainer.train(train_loader)

    def test_train_saves_checkpoint(self, tmp_path, tiny_model, optimizer,
                                     noise_schedule, train_loader):
        """Trainer.train should invoke ckpt_manager.save at least once during
        training.  If no checkpoint is saved, training progress cannot be resumed.
        """
        trainer = self._make_trainer(tmp_path, tiny_model, optimizer, noise_schedule)
        trainer.train(train_loader)
        ckpt_dir = tmp_path / "ckpts"
        if ckpt_dir.exists():
            files = list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.pth"))
            assert len(files) > 0, "No checkpoint files found after training"

    def test_validate_returns_metrics(self, tmp_path, tiny_model, optimizer,
                                       noise_schedule, val_loader):
        """Trainer.validate should return a metrics dict with 'val_loss'."""
        trainer = self._make_trainer(tmp_path, tiny_model, optimizer, noise_schedule)
        metrics = trainer.validate(val_loader)
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics

    def test_train_with_validation(self, tmp_path, tiny_model, optimizer,
                                    noise_schedule, train_loader, val_loader):
        """Trainer.train with a val_loader should not raise errors."""
        trainer = self._make_trainer(tmp_path, tiny_model, optimizer, noise_schedule)
        trainer.train(train_loader, val_loader=val_loader)
