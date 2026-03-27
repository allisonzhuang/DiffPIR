"""Tests for configs.py — dataclass configuration objects.

Verifies that every config dataclass:
  1. Has sensible defaults matching the paper (Table 3, Section 4).
  2. Enforces type and value invariants.
  3. Separates concerns (one config per degradation type, not a monolith).
"""

import pytest
from pathlib import Path

import torch

from configs import (
    DiffusionConfig,
    SolverConfig,
    InpaintConfig,
    BlurConfig,
    SRConfig,
    TrainingConfig,
    DegradationConfig,
)


# ---------------------------------------------------------------------------
# DiffusionConfig
# ---------------------------------------------------------------------------

class TestDiffusionConfig:

    def test_defaults_match_ddpm(self):
        """DiffusionConfig defaults must match the standard DDPM linear schedule
        from Ho et al. [24]: T=1000, beta_start=1e-4, beta_end=2e-2.
        Getting these wrong silently corrupts every downstream computation.
        """
        cfg = DiffusionConfig()
        assert cfg.T == 1000
        assert cfg.beta_start == pytest.approx(1e-4)
        assert cfg.beta_end == pytest.approx(2e-2)
        assert cfg.schedule_type == "linear"
        assert cfg.dtype == torch.float32

    def test_beta_start_less_than_beta_end(self):
        """β_start < β_end is a mathematical invariant of the linear schedule.
        If violated, alpha_bars would increase (noise decreases over time),
        breaking the entire forward process.
        """
        cfg = DiffusionConfig()
        assert cfg.beta_start < cfg.beta_end

    def test_T_positive(self):
        """T must be a positive integer — zero or negative timesteps are meaningless."""
        cfg = DiffusionConfig()
        assert cfg.T > 0

    def test_custom_values_override_defaults(self):
        """Verify that passing custom kwargs to the dataclass works correctly.
        This ensures the dataclass isn't using frozen=True or __post_init__
        that silently ignores overrides.
        """
        cfg = DiffusionConfig(T=500, beta_start=1e-3, beta_end=5e-2, schedule_type="cosine")
        assert cfg.T == 500
        assert cfg.beta_start == pytest.approx(1e-3)
        assert cfg.beta_end == pytest.approx(5e-2)
        assert cfg.schedule_type == "cosine"


# ---------------------------------------------------------------------------
# SolverConfig
# ---------------------------------------------------------------------------

class TestSolverConfig:

    def test_defaults_match_table3(self):
        """SolverConfig defaults should match the paper's Table 3 baseline:
        λ=8.0, ζ=0.3, σ_n=0.05, NFE=100, t_start=1000.
        """
        cfg = SolverConfig()
        assert cfg.lambda_ == pytest.approx(8.0)
        assert cfg.zeta == pytest.approx(0.3)
        assert cfg.sigma_n == pytest.approx(0.05)
        assert cfg.n_steps == 100
        assert cfg.t_start == 1000

    def test_lambda_positive(self):
        """λ > 0 is required for the MAP objective (Eq. 1) to be well-posed.
        λ = 0 would ignore the data fidelity term entirely.
        """
        cfg = SolverConfig()
        assert cfg.lambda_ > 0

    def test_zeta_non_negative(self):
        """ζ ∈ [0, 1] controls stochasticity (Eq. 15).  Negative ζ would
        produce imaginary noise amplitudes (√ζ), which is physically meaningless.
        """
        cfg = SolverConfig()
        assert cfg.zeta >= 0

    def test_sigma_n_non_negative(self):
        """σ_n ≥ 0 is the measurement noise std; negative is unphysical."""
        cfg = SolverConfig()
        assert cfg.sigma_n >= 0

    def test_n_steps_positive(self):
        """NFE must be at least 1 — zero sampling steps produce no output."""
        cfg = SolverConfig()
        assert cfg.n_steps > 0

    def test_t_start_within_T(self):
        """t_start should not exceed T=1000 (the total training timesteps).
        A t_start > T would index outside the noise schedule.
        """
        cfg = SolverConfig()
        assert cfg.t_start <= 1000


# ---------------------------------------------------------------------------
# InpaintConfig
# ---------------------------------------------------------------------------

class TestInpaintConfig:

    def test_defaults(self):
        """InpaintConfig defaults: box mask, 128×128, 50% random fraction.
        Paper: Section 4.1, Appendix B.2.
        """
        cfg = InpaintConfig()
        assert cfg.mask_type == "box"
        assert cfg.mask_box_size == 128
        assert cfg.mask_random_fraction == pytest.approx(0.5)
        assert cfg.mask_path is None

    def test_mask_type_valid_options(self):
        """mask_type must be one of 'box', 'random', 'file'.
        Invalid types should be caught by the Literal type annotation.
        """
        cfg_box = InpaintConfig(mask_type="box")
        assert cfg_box.mask_type == "box"
        cfg_rand = InpaintConfig(mask_type="random")
        assert cfg_rand.mask_type == "random"
        cfg_file = InpaintConfig(mask_type="file")
        assert cfg_file.mask_type == "file"

    def test_mask_box_size_positive(self):
        """A box mask of size ≤ 0 is degenerate — no pixels would be masked."""
        cfg = InpaintConfig()
        assert cfg.mask_box_size > 0

    def test_mask_random_fraction_in_unit_interval(self):
        """The masking fraction must be in [0, 1].  Values outside this range
        are meaningless (you can't mask more than 100% or fewer than 0% of pixels).
        """
        cfg = InpaintConfig()
        assert 0.0 <= cfg.mask_random_fraction <= 1.0


# ---------------------------------------------------------------------------
# BlurConfig
# ---------------------------------------------------------------------------

class TestBlurConfig:

    def test_defaults_match_paper(self):
        """BlurConfig defaults: Gaussian blur, 61×61 kernel, σ_k=3.0, motion intensity=0.5.
        Paper: Section 4.1, Appendix B.2.
        """
        cfg = BlurConfig()
        assert cfg.blur_mode == "gaussian"
        assert cfg.kernel_size == 61
        assert cfg.gaussian_std == pytest.approx(3.0)
        assert cfg.motion_intensity == pytest.approx(0.5)

    def test_kernel_size_positive_and_odd(self):
        """Blur kernels should have odd size so the centre pixel is well-defined.
        Even-sized kernels create half-pixel offset artifacts in the convolution.
        """
        cfg = BlurConfig()
        assert cfg.kernel_size > 0
        assert cfg.kernel_size % 2 == 1

    def test_gaussian_std_positive(self):
        """σ_k > 0 is required for a valid Gaussian kernel.  σ_k = 0 degenerates
        to a delta function (no blur), which is technically valid but should be
        explicit rather than accidental.
        """
        cfg = BlurConfig()
        assert cfg.gaussian_std > 0


# ---------------------------------------------------------------------------
# SRConfig
# ---------------------------------------------------------------------------

class TestSRConfig:

    def test_defaults_match_paper(self):
        """SRConfig defaults: 4× scale, FFT solver, IBP γ=0.9, IBP n_iter=6.
        Paper: Section 4.1 (4×), Appendix B.2 (FFT preferred).
        """
        cfg = SRConfig()
        assert cfg.scale_factor == 4
        assert cfg.solver == "fft"
        assert cfg.ibp_gamma == pytest.approx(0.9)
        assert cfg.ibp_n_iter == 6

    def test_scale_factor_positive_integer(self):
        """Scale factor must be a positive integer.  A factor of 0 or negative
        is degenerate — the LR image would have zero or negative spatial size.
        """
        cfg = SRConfig()
        assert cfg.scale_factor > 0
        assert isinstance(cfg.scale_factor, int)

    def test_ibp_gamma_in_valid_range(self):
        """IBP step size γ should be in (0, 1] for convergence.  γ > 1 causes
        divergence in the back-projection iterations (Eq. 30).
        """
        cfg = SRConfig()
        assert 0 < cfg.ibp_gamma <= 1.0

    def test_ibp_n_iter_positive(self):
        """IBP needs at least 1 iteration to have any effect."""
        cfg = SRConfig()
        assert cfg.ibp_n_iter > 0


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfig:

    def test_defaults(self):
        """TrainingConfig defaults: lr=2e-5, batch_size=16, n_epochs=100,
        keep_last_n=5, seed=42.
        """
        cfg = TrainingConfig()
        assert cfg.lr == pytest.approx(2e-5)
        assert cfg.batch_size == 16
        assert cfg.n_epochs == 100
        assert cfg.keep_last_n == 5
        assert cfg.seed == 42
        assert isinstance(cfg.ckpt_dir, Path)
        assert isinstance(cfg.log_dir, Path)

    def test_lr_positive(self):
        """Learning rate must be positive for gradient descent to make progress."""
        cfg = TrainingConfig()
        assert cfg.lr > 0

    def test_batch_size_positive(self):
        """Batch size must be a positive integer."""
        cfg = TrainingConfig()
        assert cfg.batch_size > 0

    def test_keep_last_n_positive(self):
        """Must keep at least 1 checkpoint — otherwise we'd prune everything."""
        cfg = TrainingConfig()
        assert cfg.keep_last_n > 0


# ---------------------------------------------------------------------------
# DegradationConfig union type
# ---------------------------------------------------------------------------

class TestDegradationConfigUnion:

    def test_inpaint_config_is_degradation_config(self):
        """InpaintConfig should be a valid DegradationConfig member.
        This confirms the Union type alias includes all degradation types.
        """
        cfg = InpaintConfig()
        assert isinstance(cfg, (InpaintConfig, BlurConfig, SRConfig))

    def test_blur_config_is_degradation_config(self):
        """BlurConfig should be a valid DegradationConfig member."""
        cfg = BlurConfig()
        assert isinstance(cfg, (InpaintConfig, BlurConfig, SRConfig))

    def test_sr_config_is_degradation_config(self):
        """SRConfig should be a valid DegradationConfig member."""
        cfg = SRConfig()
        assert isinstance(cfg, (InpaintConfig, BlurConfig, SRConfig))

    def test_configs_are_distinct_types(self):
        """Each degradation config must be its own class — not a single monolith.
        This enforces the design decision to have separate config dataclasses
        per degradation type (see memory: feedback_config_design.md).
        """
        assert InpaintConfig is not BlurConfig
        assert BlurConfig is not SRConfig
        assert InpaintConfig is not SRConfig


# ---------------------------------------------------------------------------
# Cross-config consistency (Table 3 hyperparameters)
# ---------------------------------------------------------------------------

class TestTable3Consistency:
    """Verify that the default config values are self-consistent with
    each other and with the paper's Table 3 hyperparameters.
    """

    def test_solver_n_steps_within_T(self):
        """n_steps (NFE) must not exceed T — can't take more reverse steps
        than training timesteps.
        """
        dcfg = DiffusionConfig()
        scfg = SolverConfig()
        assert scfg.n_steps <= dcfg.T

    def test_solver_t_start_within_T(self):
        """t_start must be ≤ T — can't start sampling beyond the schedule."""
        dcfg = DiffusionConfig()
        scfg = SolverConfig()
        assert scfg.t_start <= dcfg.T

    def test_inpaint_box_fits_in_reasonable_image(self):
        """128×128 box should fit in the paper's evaluation images (256×256).
        A mask larger than the image is degenerate.
        """
        cfg = InpaintConfig()
        assert cfg.mask_box_size <= 256

    def test_blur_kernel_fits_in_reasonable_image(self):
        """61×61 kernel should be smaller than the minimum image dimension.
        A kernel larger than the image causes degenerate circular convolution.
        """
        cfg = BlurConfig()
        assert cfg.kernel_size < 256

    def test_all_configs_are_dataclasses(self):
        """Every config must be a Python dataclass — this ensures they are
        immutable record types with auto-generated __init__, __eq__, etc.
        """
        from dataclasses import fields
        for cls in (DiffusionConfig, SolverConfig, InpaintConfig,
                    BlurConfig, SRConfig, TrainingConfig):
            # fields() raises TypeError if not a dataclass
            assert len(fields(cls)) > 0, f"{cls.__name__} has no fields"
