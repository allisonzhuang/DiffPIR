"""Tests for train/checkpoint.py — checkpoint I/O and management.

Verifies save/load round-tripping, checkpoint listing/pruning, and the
CheckpointManager class that orchestrates these operations.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from train.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
    prune_old_checkpoints,
    CheckpointManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Minimal model for checkpoint testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def _make_checkpoint_files(ckpt_dir: Path, steps: list[int]):
    """Create dummy checkpoint files at the given steps."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters())
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        }
        save_checkpoint(ckpt_dir / f"ckpt_{step:06d}.pt", state)


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

class TestSaveLoadCheckpoint:

    def test_roundtrip(self, tmp_path):
        """save_checkpoint followed by load_checkpoint must recover the exact
        state dictionary.  This is the fundamental serialisation contract.
        A failure here means training progress would be lost.
        """
        model = TinyModel()
        state = {
            "model_state_dict": model.state_dict(),
            "step": 42,
            "loss": 0.123,
        }
        path = tmp_path / "test_ckpt.pt"
        save_checkpoint(path, state)
        loaded = load_checkpoint(path)

        assert loaded["step"] == 42
        assert loaded["loss"] == pytest.approx(0.123)
        for key in model.state_dict():
            torch.testing.assert_close(
                loaded["model_state_dict"][key],
                state["model_state_dict"][key],
            )

    def test_creates_file(self, tmp_path):
        """save_checkpoint must create a file on disk."""
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, {"step": 1})
        assert path.exists()

    def test_file_non_empty(self, tmp_path):
        """The saved file must have non-zero size (not an empty write)."""
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, {"step": 1})
        assert path.stat().st_size > 0

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading a non-existent file should raise an error, not return None."""
        with pytest.raises(Exception):  # FileNotFoundError or RuntimeError
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_roundtrip_with_optimizer(self, tmp_path):
        """Optimizer state must survive the round-trip — this includes
        momentum buffers and step counts that are critical for training resumption.
        """
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Do one step to populate optimizer state
        loss = model(torch.randn(1, 4)).sum()
        loss.backward()
        optimizer.step()

        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": 100,
        }
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, state)
        loaded = load_checkpoint(path)

        # Verify optimizer state keys match
        assert set(loaded["optimizer_state_dict"].keys()) == set(state["optimizer_state_dict"].keys())


# ---------------------------------------------------------------------------
# list_checkpoints
# ---------------------------------------------------------------------------

class TestListCheckpoints:

    def test_returns_sorted_by_step(self, tmp_path):
        """list_checkpoints must return paths sorted by step number (ascending).
        Unsorted listing would break load_latest and prune logic.
        """
        _make_checkpoint_files(tmp_path, [50, 10, 30, 20, 40])
        paths = list_checkpoints(tmp_path)
        steps = [int(p.stem.split("_")[-1]) for p in paths]
        assert steps == sorted(steps)

    def test_empty_directory(self, tmp_path):
        """Empty directory should return an empty list, not raise."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        paths = list_checkpoints(tmp_path)
        assert paths == []

    def test_returns_paths(self, tmp_path):
        """Each element should be a Path object."""
        _make_checkpoint_files(tmp_path, [1, 2, 3])
        paths = list_checkpoints(tmp_path)
        assert all(isinstance(p, Path) for p in paths)
        assert len(paths) == 3


# ---------------------------------------------------------------------------
# prune_old_checkpoints
# ---------------------------------------------------------------------------

class TestPruneOldCheckpoints:

    def test_keeps_n_files(self, tmp_path):
        """After pruning with keep_last_n=2, exactly 2 files should remain."""
        _make_checkpoint_files(tmp_path, [10, 20, 30, 40, 50])
        prune_old_checkpoints(tmp_path, keep_last_n=2)
        remaining = list_checkpoints(tmp_path)
        assert len(remaining) == 2

    def test_keeps_most_recent(self, tmp_path):
        """The surviving checkpoints should be the most recent ones (highest step)."""
        _make_checkpoint_files(tmp_path, [10, 20, 30, 40, 50])
        prune_old_checkpoints(tmp_path, keep_last_n=2)
        remaining = list_checkpoints(tmp_path)
        steps = [int(p.stem.split("_")[-1]) for p in remaining]
        assert sorted(steps) == [40, 50]

    def test_no_op_when_fewer_than_n(self, tmp_path):
        """If fewer than keep_last_n checkpoints exist, none should be deleted."""
        _make_checkpoint_files(tmp_path, [10, 20])
        prune_old_checkpoints(tmp_path, keep_last_n=5)
        remaining = list_checkpoints(tmp_path)
        assert len(remaining) == 2


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class TestCheckpointManager:

    def test_save_creates_file(self, tmp_path):
        """CheckpointManager.save must write a checkpoint file to ckpt_dir."""
        manager = CheckpointManager(tmp_path, keep_last_n=3)
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters())
        path = manager.save(model, optimizer, step=1)
        assert path.exists()

    def test_save_returns_path(self, tmp_path):
        """save() must return a Path to the new checkpoint."""
        manager = CheckpointManager(tmp_path, keep_last_n=3)
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters())
        path = manager.save(model, optimizer, step=1)
        assert isinstance(path, Path)

    def test_save_prunes_old(self, tmp_path):
        """After saving more than keep_last_n checkpoints, old ones should
        be automatically pruned.
        """
        manager = CheckpointManager(tmp_path, keep_last_n=2)
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters())
        for step in range(5):
            manager.save(model, optimizer, step=step)
        remaining = list_checkpoints(tmp_path)
        assert len(remaining) <= 2

    def test_load_latest_restores_step(self, tmp_path):
        """load_latest must return the step number from the most recent checkpoint."""
        manager = CheckpointManager(tmp_path, keep_last_n=5)
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters())
        manager.save(model, optimizer, step=10)
        manager.save(model, optimizer, step=20)

        model2 = TinyModel()
        step = manager.load_latest(model2)
        assert step == 20

    def test_load_restores_model_weights(self, tmp_path):
        """load must restore exact model weights from the checkpoint.
        This catches bugs where load ignores the state_dict or loads into
        the wrong module.
        """
        manager = CheckpointManager(tmp_path, keep_last_n=5)
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters())
        # Randomise weights
        torch.manual_seed(0)
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(torch.randn_like(p))
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        path = manager.save(model, optimizer, step=1)

        # Create a fresh model (different random weights)
        model2 = TinyModel()
        manager.load(path, model2)

        for key in original_state:
            torch.testing.assert_close(model2.state_dict()[key], original_state[key])

    def test_save_with_metrics(self, tmp_path):
        """save() with metrics should include them in the checkpoint."""
        manager = CheckpointManager(tmp_path, keep_last_n=3)
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters())
        path = manager.save(model, optimizer, step=1, metrics={"psnr": 27.36})
        loaded = load_checkpoint(path)
        assert "metrics" in loaded or "psnr" in str(loaded)

    def test_multiple_saves_creates_unique_files(self, tmp_path):
        """Each save at a different step should create a distinct file.
        Overwriting would lose previous checkpoints silently.
        """
        manager = CheckpointManager(tmp_path, keep_last_n=10)
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters())
        paths = []
        for step in [10, 20, 30]:
            p = manager.save(model, optimizer, step=step)
            paths.append(p)
        # All paths should be distinct
        assert len(set(paths)) == 3

    def test_load_specific_checkpoint(self, tmp_path):
        """manager.load(path, model) should load the specific checkpoint at
        the given path, not the latest one.
        """
        manager = CheckpointManager(tmp_path, keep_last_n=10)
        model1 = TinyModel()
        optimizer = torch.optim.Adam(model1.parameters())

        # Save two checkpoints with different weights
        torch.manual_seed(0)
        with torch.no_grad():
            for p in model1.parameters():
                p.copy_(torch.randn_like(p))
        state_at_10 = {k: v.clone() for k, v in model1.state_dict().items()}
        path_10 = manager.save(model1, optimizer, step=10)

        torch.manual_seed(99)
        with torch.no_grad():
            for p in model1.parameters():
                p.copy_(torch.randn_like(p))
        manager.save(model1, optimizer, step=20)

        # Load step 10 specifically
        model2 = TinyModel()
        step = manager.load(path_10, model2)
        assert step == 10
        for key in state_at_10:
            torch.testing.assert_close(model2.state_dict()[key], state_at_10[key])
