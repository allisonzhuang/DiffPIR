"""Tests for restoration/diffpir.py — DiffPIR outer sampling loop.

Tests use mock denoisers and solvers to keep execution fast while verifying
that the loop orchestration, shape contracts, and iteration counts are correct.
"""

from unittest.mock import MagicMock, call

import pytest
import torch
from torch import Tensor

from configs import DiffusionConfig, SolverConfig
from interfaces import DenoiserPrior, PnPSolver
from models.diffusion import make_noise_schedule, build_ddim_timestep_sequence, precompute_rho_schedule
from restoration.diffpir import diffpir_restore, initialize_x_T


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class PassthroughDenoiser(DenoiserPrior):
    """Returns x_t unchanged — identity prior for integration testing."""
    def __init__(self):
        self.call_count = 0

    def denoise(self, x_t, t, noise_schedule):
        self.call_count += 1
        return x_t


class PassthroughSolver(PnPSolver):
    """Returns x0_prior unchanged — no data correction."""
    def data_step(self, x0_prior, y, degradation, rho_t):
        return x0_prior


@pytest.fixture
def solver_cfg():
    return SolverConfig(n_steps=5, lambda_=8.0, sigma_n=0.05, zeta=0.0, t_start=1000)


@pytest.fixture
def noise_schedule():
    return make_noise_schedule(DiffusionConfig())


@pytest.fixture
def timesteps(solver_cfg):
    return build_ddim_timestep_sequence(
        t_start=solver_cfg.t_start,
        n_steps=solver_cfg.n_steps,
        T=1000,
    )


@pytest.fixture
def rho_schedule(solver_cfg, noise_schedule, timesteps):
    return precompute_rho_schedule(solver_cfg, noise_schedule, timesteps)


# ---------------------------------------------------------------------------
# initialize_x_T
# ---------------------------------------------------------------------------

class TestInitializeXT:

    def test_output_shape(self):
        """initialize_x_T must return a tensor of the exact requested shape.
        Shape (B, C, H, W) is the entry point of the entire reverse process.
        """
        shape = (2, 3, 64, 64)
        x_T = initialize_x_T(shape, device="cpu")
        assert x_T.shape == shape

    def test_output_dtype_float32(self):
        """x_T must be float32 — the noise schedule and all downstream ops
        assume float32.
        """
        x_T = initialize_x_T((1, 3, 32, 32), device="cpu")
        assert x_T.dtype == torch.float32

    def test_is_standard_normal(self):
        """x_T ~ N(0, I).  Over many samples, the empirical mean should be ≈ 0
        and empirical variance ≈ 1.  This verifies the initialisation distribution.
        """
        torch.manual_seed(0)
        shape = (10000, 1, 1, 1)
        x_T = initialize_x_T(shape, device="cpu")
        assert x_T.mean().abs().item() < 0.05
        assert x_T.var().item() == pytest.approx(1.0, abs=0.05)

    def test_different_calls_produce_different_noise(self):
        """Two calls without seed control should produce different tensors
        (with overwhelming probability).
        """
        x1 = initialize_x_T((1, 3, 32, 32), device="cpu")
        x2 = initialize_x_T((1, 3, 32, 32), device="cpu")
        assert not torch.allclose(x1, x2)


# ---------------------------------------------------------------------------
# diffpir_restore
# ---------------------------------------------------------------------------

class TestDiffpirRestore:

    def test_output_shape(self, solver_cfg, noise_schedule, timesteps, rho_schedule):
        """diffpir_restore must return a tensor with the same spatial shape as x_T.
        This is the restored image — shape mismatch would be a critical bug.
        """
        x_T = torch.randn(1, 3, 32, 32)
        y = torch.randn(1, 3, 32, 32)
        denoiser = PassthroughDenoiser()
        solver = PassthroughSolver()
        result = diffpir_restore(
            x_T, y, denoiser, solver, degradation=None,
            cfg=solver_cfg, noise_schedule=noise_schedule,
            timesteps=timesteps, rho_schedule=rho_schedule,
        )
        assert result.shape == x_T.shape

    def test_output_dtype_float32(self, solver_cfg, noise_schedule, timesteps, rho_schedule):
        """Output must be float32."""
        x_T = torch.randn(1, 3, 16, 16)
        y = torch.randn(1, 3, 16, 16)
        result = diffpir_restore(
            x_T, y, PassthroughDenoiser(), PassthroughSolver(),
            degradation=None, cfg=solver_cfg, noise_schedule=noise_schedule,
            timesteps=timesteps, rho_schedule=rho_schedule,
        )
        assert result.dtype == torch.float32

    def test_calls_denoiser_n_steps_times(self, solver_cfg, noise_schedule, timesteps, rho_schedule):
        """The loop should call the denoiser exactly n_steps times — once per
        timestep.  Fewer calls means skipped steps; more means wasted NFEs.
        """
        x_T = torch.randn(1, 3, 16, 16)
        y = torch.randn(1, 3, 16, 16)
        denoiser = PassthroughDenoiser()
        diffpir_restore(
            x_T, y, denoiser, PassthroughSolver(),
            degradation=None, cfg=solver_cfg, noise_schedule=noise_schedule,
            timesteps=timesteps, rho_schedule=rho_schedule,
        )
        assert denoiser.call_count == solver_cfg.n_steps

    def test_finite_output(self, solver_cfg, noise_schedule, timesteps, rho_schedule):
        """The output must contain no NaN or Inf values.  Numerical instability
        in the loop (e.g. from division by near-zero ᾱ_t) would propagate here.
        """
        x_T = torch.randn(1, 3, 16, 16)
        y = torch.randn(1, 3, 16, 16)
        result = diffpir_restore(
            x_T, y, PassthroughDenoiser(), PassthroughSolver(),
            degradation=None, cfg=solver_cfg, noise_schedule=noise_schedule,
            timesteps=timesteps, rho_schedule=rho_schedule,
        )
        assert torch.isfinite(result).all()

    def test_deterministic_with_zeta_zero(self, noise_schedule):
        """With ζ=0 (deterministic DDIM) and seeded x_T, two runs must produce
        identical outputs.  This verifies no stray randomness leaks in.
        """
        cfg = SolverConfig(n_steps=3, zeta=0.0, sigma_n=0.05)
        ts = build_ddim_timestep_sequence(cfg.t_start, cfg.n_steps, 1000)
        rhos = precompute_rho_schedule(cfg, noise_schedule, ts)

        torch.manual_seed(0)
        x_T = torch.randn(1, 3, 16, 16)
        y = torch.randn(1, 3, 16, 16)

        r1 = diffpir_restore(x_T, y, PassthroughDenoiser(), PassthroughSolver(),
                             None, cfg, noise_schedule, ts, rhos)
        r2 = diffpir_restore(x_T, y, PassthroughDenoiser(), PassthroughSolver(),
                             None, cfg, noise_schedule, ts, rhos)
        torch.testing.assert_close(r1, r2)


# ---------------------------------------------------------------------------
# Integration: mock denoiser satisfying DenoiserPrior
# ---------------------------------------------------------------------------

class TestDiffpirIntegration:

    def test_accepts_any_denoiser_prior(self, solver_cfg, noise_schedule, timesteps, rho_schedule):
        """diffpir_restore should work with ANY object that satisfies the
        DenoiserPrior interface, not just UNetDenoiser.  This is the core
        plug-and-play property.
        """

        class HalfDenoiser(DenoiserPrior):
            def denoise(self, x_t, t, noise_schedule):
                return x_t * 0.5

        x_T = torch.randn(1, 3, 16, 16)
        y = torch.randn(1, 3, 16, 16)
        result = diffpir_restore(
            x_T, y, HalfDenoiser(), PassthroughSolver(),
            degradation=None, cfg=solver_cfg, noise_schedule=noise_schedule,
            timesteps=timesteps, rho_schedule=rho_schedule,
        )
        assert result.shape == x_T.shape
        assert result.dtype == torch.float32

    def test_accepts_any_pnp_solver(self, solver_cfg, noise_schedule, timesteps, rho_schedule):
        """diffpir_restore should work with ANY PnPSolver, not just the
        built-in ones.  This verifies Axis 2 extensibility (SPEC Section 6).
        """

        class ScaleSolver(PnPSolver):
            def data_step(self, x0_prior, y, degradation, rho_t):
                return x0_prior * 0.9

        x_T = torch.randn(1, 3, 16, 16)
        y = torch.randn(1, 3, 16, 16)
        result = diffpir_restore(
            x_T, y, PassthroughDenoiser(), ScaleSolver(),
            degradation=None, cfg=solver_cfg, noise_schedule=noise_schedule,
            timesteps=timesteps, rho_schedule=rho_schedule,
        )
        assert result.shape == x_T.shape

    def test_different_n_steps(self, noise_schedule):
        """diffpir_restore should work correctly for various NFE values.
        The number of denoiser calls should always equal n_steps.
        """
        for n_steps in [1, 3, 10, 20]:
            cfg = SolverConfig(n_steps=n_steps, zeta=0.0, sigma_n=0.05)
            ts = build_ddim_timestep_sequence(cfg.t_start, cfg.n_steps, 1000)
            rhos = precompute_rho_schedule(cfg, noise_schedule, ts)
            denoiser = PassthroughDenoiser()
            x_T = torch.randn(1, 3, 8, 8)
            y = torch.randn(1, 3, 8, 8)
            result = diffpir_restore(
                x_T, y, denoiser, PassthroughSolver(),
                None, cfg, noise_schedule, ts, rhos,
            )
            assert result.shape == x_T.shape
            assert denoiser.call_count == n_steps

    def test_batch_output_correct(self, noise_schedule):
        """diffpir_restore should handle batch dimension correctly.
        Output batch size must match input batch size.
        """
        cfg = SolverConfig(n_steps=3, zeta=0.0, sigma_n=0.05)
        ts = build_ddim_timestep_sequence(cfg.t_start, cfg.n_steps, 1000)
        rhos = precompute_rho_schedule(cfg, noise_schedule, ts)
        x_T = torch.randn(4, 3, 8, 8)
        y = torch.randn(4, 3, 8, 8)
        result = diffpir_restore(
            x_T, y, PassthroughDenoiser(), PassthroughSolver(),
            None, cfg, noise_schedule, ts, rhos,
        )
        assert result.shape == (4, 3, 8, 8)
