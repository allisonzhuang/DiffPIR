"""Tests for models/diffusion.py — DDPM/DDIM noise schedule and diffusion math.

These tests verify the mathematical properties that the noise schedule and
diffusion utilities must satisfy for DiffPIR to work correctly.  Many are
derived directly from the closed-form expressions in the paper (Eqs. 5–15).
"""

import math

import pytest
import torch

from configs import DiffusionConfig, SolverConfig
from models.diffusion import (
    make_noise_schedule,
    compute_sigma_t,
    compute_rho_t,
    predict_x0,
    add_noise,
    ddim_step,
    compute_effective_noise,
    build_ddim_timestep_sequence,
    precompute_rho_schedule,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_cfg():
    return DiffusionConfig()


@pytest.fixture
def noise_schedule(default_cfg):
    return make_noise_schedule(default_cfg)


@pytest.fixture
def small_image():
    """A small (B=2, C=3, H=16, W=16) test image for shape/dtype checks."""
    torch.manual_seed(0)
    return torch.randn(2, 3, 16, 16)


# ---------------------------------------------------------------------------
# make_noise_schedule
# ---------------------------------------------------------------------------

class TestMakeNoiseSchedule:

    def test_output_keys(self, noise_schedule):
        """make_noise_schedule must return a dict with exactly the documented keys:
        betas, alphas, alpha_bars, sigma_ts.  Missing keys would crash any
        downstream consumer.
        """
        required = {"betas", "alphas", "alpha_bars", "sigma_ts"}
        assert required.issubset(noise_schedule.keys())

    def test_shapes_match_T(self, default_cfg, noise_schedule):
        """All schedule tensors must have shape (T,).  A shape mismatch would
        cause indexing errors in the sampling loop.
        """
        T = default_cfg.T
        for key in ("betas", "alphas", "alpha_bars", "sigma_ts"):
            assert noise_schedule[key].shape == (T,), f"{key} shape mismatch"

    def test_dtype_float32(self, noise_schedule):
        """All schedule tensors must be float32.  Mixed precision or float64
        would silently degrade numerical accuracy in downstream operations.
        """
        for key in ("betas", "alphas", "alpha_bars", "sigma_ts"):
            assert noise_schedule[key].dtype == torch.float32, f"{key} dtype mismatch"

    def test_betas_in_range(self, default_cfg, noise_schedule):
        """β_t must lie in [beta_start, beta_end] for the linear schedule.
        Values outside this range indicate a bug in linspace construction.
        """
        betas = noise_schedule["betas"]
        assert betas.min() >= default_cfg.beta_start - 1e-7
        assert betas.max() <= default_cfg.beta_end + 1e-7

    def test_betas_monotonically_increasing(self, noise_schedule):
        """For the linear schedule, β_t increases monotonically from β_1 to β_T.
        Non-monotone betas would create irregular noise injection.
        """
        betas = noise_schedule["betas"]
        diffs = betas[1:] - betas[:-1]
        assert (diffs >= -1e-7).all(), "betas not monotonically increasing"

    def test_alphas_equal_one_minus_betas(self, noise_schedule):
        """α_t = 1 − β_t (Eq. 5).  This identity must hold exactly (up to
        float32 precision).  A violation means either betas or alphas is wrong.
        """
        alphas = noise_schedule["alphas"]
        betas = noise_schedule["betas"]
        torch.testing.assert_close(alphas, 1.0 - betas)

    def test_alpha_bars_are_cumprod(self, noise_schedule):
        """ᾱ_t = ∏_{s=1}^{t} α_s (Eq. 6).  Must equal torch.cumprod(alphas).
        If this fails, predict_x0 and add_noise will silently produce wrong results.
        """
        alphas = noise_schedule["alphas"]
        expected = torch.cumprod(alphas, dim=0)
        torch.testing.assert_close(noise_schedule["alpha_bars"], expected)

    def test_alpha_bars_monotonically_decreasing(self, noise_schedule):
        """ᾱ_t decreases monotonically from ~1 (almost no noise) to ~0 (pure noise).
        This is the fundamental property of the forward diffusion process.
        If violated, the signal-to-noise ratio would not decrease over time.
        """
        ab = noise_schedule["alpha_bars"]
        diffs = ab[1:] - ab[:-1]
        assert (diffs <= 1e-7).all(), "alpha_bars not monotonically decreasing"

    def test_alpha_bars_boundary_values(self, noise_schedule):
        """ᾱ_1 should be close to 1 (almost no noise at t=1) and ᾱ_T close to 0
        (nearly pure noise at t=T).  These boundary conditions ensure the
        forward process spans the full noise range.
        """
        ab = noise_schedule["alpha_bars"]
        assert ab[0] > 0.99, f"ᾱ_1 = {ab[0]:.4f}, expected > 0.99"
        assert ab[-1] < 0.01, f"ᾱ_T = {ab[-1]:.4f}, expected < 0.01"

    def test_sigma_ts_formula(self, noise_schedule):
        """σ_t = √((1−ᾱ_t) / ᾱ_t) (Section 3.1).  This connects the VP-SDE
        to the VE-SDE parameterisation used in HQS.
        """
        ab = noise_schedule["alpha_bars"]
        expected = torch.sqrt((1.0 - ab) / ab)
        torch.testing.assert_close(noise_schedule["sigma_ts"], expected)

    def test_sigma_ts_monotonically_increasing(self, noise_schedule):
        """σ_t should increase monotonically, since ᾱ_t decreases.
        This ensures the noise level interpretation is consistent.
        """
        st = noise_schedule["sigma_ts"]
        diffs = st[1:] - st[:-1]
        assert (diffs >= -1e-6).all(), "sigma_ts not monotonically increasing"


# ---------------------------------------------------------------------------
# compute_sigma_t
# ---------------------------------------------------------------------------

class TestComputeSigmaT:

    def test_known_value_alpha_bar_half(self):
        """When ᾱ_t = 0.5, σ_t = √((1−0.5)/0.5) = 1.0.
        Hand-computed reference value to catch formula bugs.
        """
        result = compute_sigma_t(torch.tensor(0.5))
        assert result.item() == pytest.approx(1.0, abs=1e-6)

    def test_known_value_alpha_bar_near_one(self):
        """When ᾱ_t = 0.99, σ_t = √(0.01/0.99) ≈ 0.1005.
        Near-unity ᾱ_t tests the low-noise regime (early timesteps).
        """
        result = compute_sigma_t(torch.tensor(0.99))
        expected = math.sqrt(0.01 / 0.99)
        assert result.item() == pytest.approx(expected, abs=1e-5)

    def test_known_value_alpha_bar_near_zero(self):
        """When ᾱ_t = 0.01, σ_t = √(0.99/0.01) ≈ 9.9499.
        Near-zero ᾱ_t tests the high-noise regime (late timesteps).
        Numerical stability matters here — division by small ᾱ_t.
        """
        result = compute_sigma_t(torch.tensor(0.01))
        expected = math.sqrt(0.99 / 0.01)
        assert result.item() == pytest.approx(expected, abs=1e-3)

    def test_output_dtype_float32(self):
        """compute_sigma_t must return float32 to maintain precision consistency."""
        result = compute_sigma_t(torch.tensor(0.5, dtype=torch.float32))
        assert result.dtype == torch.float32

    def test_positive_for_any_valid_alpha_bar(self):
        """σ_t > 0 for all ᾱ_t ∈ (0, 1).  σ_t = 0 only at ᾱ_t = 1 (impossible
        in practice since β_1 > 0).
        """
        alpha_bars = torch.linspace(0.001, 0.999, 100)
        sigmas = compute_sigma_t(alpha_bars)
        assert (sigmas > 0).all()


# ---------------------------------------------------------------------------
# compute_rho_t
# ---------------------------------------------------------------------------

class TestComputeRhoT:

    def test_known_value(self):
        """ρ_t = λ · σ_n² / σ_t².  For λ=8, σ_n=0.05, σ_t=1.0:
        ρ_t = 8 × 0.0025 / 1.0 = 0.02.  Hand-computed reference.
        """
        result = compute_rho_t(lambda_=8.0, sigma_n=0.05, sigma_t=1.0)
        assert result.item() == pytest.approx(0.02, abs=1e-6)

    def test_positive_for_positive_inputs(self):
        """ρ_t > 0 whenever λ > 0, σ_n > 0, σ_t > 0.  Negative ρ_t would
        reverse the penalty direction in the data subproblem (Eq. 12b).
        """
        result = compute_rho_t(lambda_=7.0, sigma_n=0.05, sigma_t=2.0)
        assert result.item() > 0

    def test_zero_sigma_n_gives_zero_rho(self):
        """When σ_n = 0 (noiseless measurement, as in inpainting), ρ_t = 0.
        This means the data step fully trusts the measurement y.
        """
        result = compute_rho_t(lambda_=8.0, sigma_n=0.0, sigma_t=1.0)
        assert result.item() == pytest.approx(0.0, abs=1e-10)

    def test_rho_decreases_as_sigma_t_increases(self):
        """ρ_t ∝ 1/σ_t², so larger σ_t (more noise) → smaller ρ_t (less
        data trust).  This is the key scheduling insight of DiffPIR.
        """
        rho_small = compute_rho_t(lambda_=8.0, sigma_n=0.05, sigma_t=0.5)
        rho_large = compute_rho_t(lambda_=8.0, sigma_n=0.05, sigma_t=2.0)
        assert rho_small.item() > rho_large.item()

    def test_output_is_tensor(self):
        """compute_rho_t must return a tensor (not a plain float) for
        downstream compatibility with PyTorch operations.
        """
        result = compute_rho_t(lambda_=8.0, sigma_n=0.05, sigma_t=1.0)
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# predict_x0
# ---------------------------------------------------------------------------

class TestPredictX0:

    def test_shape_preserved(self, small_image):
        """predict_x0 output must have the same shape as x_t.
        Shape mismatch would crash the data step.
        """
        eps = torch.randn_like(small_image)
        result = predict_x0(small_image, eps, alpha_bar_t=0.5)
        assert result.shape == small_image.shape

    def test_dtype_float32(self, small_image):
        """Output must be float32 to maintain precision through the HQS loop."""
        eps = torch.randn_like(small_image)
        result = predict_x0(small_image, eps, alpha_bar_t=0.5)
        assert result.dtype == torch.float32

    def test_invertibility_with_add_noise(self, noise_schedule, small_image):
        """predict_x0 should invert add_noise: given x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε,
        predict_x0(x_t, ε, ᾱ_t) should recover x₀ exactly.

        This is the fundamental invertibility property (Eq. 11): the U-Net
        predicts ε, and predict_x0 inverts the forward process to get x₀.
        If this fails, the entire DiffPIR algorithm is broken.
        """
        t = 500  # mid-range timestep
        ab_t = noise_schedule["alpha_bars"][t - 1]
        eps = torch.randn_like(small_image)
        # Forward: x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε
        x_t = math.sqrt(ab_t.item()) * small_image + math.sqrt(1 - ab_t.item()) * eps
        # Invert: should recover x₀
        x0_recovered = predict_x0(x_t, eps, ab_t)
        torch.testing.assert_close(x0_recovered, small_image, atol=1e-5, rtol=1e-5)

    def test_known_values_alpha_bar_one(self, small_image):
        """When ᾱ_t = 1 (no noise, t=0): x̂₀ = (x_t − 0·ε) / 1 = x_t.
        The predicted clean image should equal the input (numerical stability at t=0).
        """
        eps = torch.randn_like(small_image)
        result = predict_x0(small_image, eps, alpha_bar_t=1.0)
        torch.testing.assert_close(result, small_image, atol=1e-6, rtol=1e-6)

    def test_zero_eps_returns_scaled_input(self, small_image):
        """When ε_θ = 0, x̂₀ = x_t / √ᾱ_t.  This tests the formula path
        where the noise prediction is zero (a degenerate but valid case).
        """
        eps = torch.zeros_like(small_image)
        ab_t = 0.25
        result = predict_x0(small_image, eps, alpha_bar_t=ab_t)
        expected = small_image / math.sqrt(ab_t)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# add_noise
# ---------------------------------------------------------------------------

class TestAddNoise:

    def test_shape_preserved(self, noise_schedule, small_image):
        """add_noise output must have the same shape as x₀."""
        result = add_noise(small_image, t=500, noise_schedule=noise_schedule)
        assert result.shape == small_image.shape

    def test_dtype_float32(self, noise_schedule, small_image):
        """Output must be float32."""
        result = add_noise(small_image, t=500, noise_schedule=noise_schedule)
        assert result.dtype == torch.float32

    def test_deterministic_with_provided_noise(self, noise_schedule, small_image):
        """When noise is provided explicitly, add_noise must use it (not sample
        new noise).  This is critical for reproducibility and test isolation.
        """
        noise = torch.ones_like(small_image)
        r1 = add_noise(small_image, t=500, noise_schedule=noise_schedule, noise=noise)
        r2 = add_noise(small_image, t=500, noise_schedule=noise_schedule, noise=noise)
        torch.testing.assert_close(r1, r2)

    def test_forward_process_mean(self, noise_schedule, small_image):
        """E[x_t] = √ᾱ_t · x₀ (Eq. 6).  The empirical mean over many noise
        realisations should converge to the theoretical mean.  This is the
        first-moment check for the forward process.
        """
        t = 200
        ab_t = noise_schedule["alpha_bars"][t - 1].item()
        n_samples = 5000
        samples = torch.stack([
            add_noise(small_image, t=t, noise_schedule=noise_schedule)
            for _ in range(n_samples)
        ])
        empirical_mean = samples.mean(dim=0)
        theoretical_mean = math.sqrt(ab_t) * small_image
        torch.testing.assert_close(
            empirical_mean, theoretical_mean, atol=0.05, rtol=0.05,
        )

    def test_forward_process_variance(self, noise_schedule):
        """Var[x_t] = (1 − ᾱ_t) (Eq. 6, when x₀ is fixed).  For a scalar x₀=0,
        Var[x_t] should converge to (1−ᾱ_t) over many samples.
        This is the second-moment check.
        """
        t = 500
        ab_t = noise_schedule["alpha_bars"][t - 1].item()
        x0 = torch.zeros(1, 1, 8, 8)
        n_samples = 10000
        samples = torch.stack([
            add_noise(x0, t=t, noise_schedule=noise_schedule)
            for _ in range(n_samples)
        ])
        empirical_var = samples.var(dim=0).mean().item()
        theoretical_var = 1.0 - ab_t
        assert empirical_var == pytest.approx(theoretical_var, abs=0.02)

    def test_noise_at_t1_is_nearly_clean(self, noise_schedule, small_image):
        """At t=1, ᾱ_1 ≈ 1, so x_1 ≈ x₀.  The added noise should be negligible."""
        noise = torch.randn_like(small_image)
        result = add_noise(small_image, t=1, noise_schedule=noise_schedule, noise=noise)
        ab_1 = noise_schedule["alpha_bars"][0].item()
        # x_1 = √ᾱ_1 · x₀ + √(1−ᾱ_1) · ε ≈ x₀ + tiny noise
        assert (result - small_image).abs().max() < 0.1


# ---------------------------------------------------------------------------
# compute_effective_noise and ddim_step
# ---------------------------------------------------------------------------

class TestComputeEffectiveNoise:

    def test_shape_preserved(self, small_image):
        """compute_effective_noise output shape must match x_t."""
        x0 = torch.randn_like(small_image)
        result = compute_effective_noise(small_image, x0, alpha_bar_t=0.5)
        assert result.shape == small_image.shape

    def test_recovers_original_noise(self, noise_schedule, small_image):
        """If x_t was constructed from x₀ and ε via add_noise, then
        compute_effective_noise(x_t, x₀, ᾱ_t) should recover ε exactly.

        This is the algebraic identity:
        ε̂ = (x_t − √ᾱ_t·x₀) / √(1−ᾱ_t) = (√ᾱ_t·x₀ + √(1−ᾱ_t)·ε − √ᾱ_t·x₀) / √(1−ᾱ_t) = ε
        """
        t = 300
        ab_t = noise_schedule["alpha_bars"][t - 1]
        eps = torch.randn_like(small_image)
        x_t = add_noise(small_image, t=t, noise_schedule=noise_schedule, noise=eps)
        eps_hat = compute_effective_noise(x_t, small_image, ab_t)
        torch.testing.assert_close(eps_hat, eps, atol=1e-5, rtol=1e-5)


class TestDdimStep:

    def test_shape_preserved(self, small_image):
        """ddim_step output must match x_t shape."""
        x0 = torch.randn_like(small_image)
        eps_hat = torch.randn_like(small_image)
        result = ddim_step(small_image, x0, eps_hat, alpha_bar_t=0.5,
                           alpha_bar_prev=0.6, zeta=0.0)
        assert result.shape == small_image.shape

    def test_deterministic_when_zeta_zero(self, small_image):
        """With ζ = 0, ddim_step is deterministic (Eq. 15 with no stochastic term).
        Two calls with identical inputs must produce identical outputs.
        """
        x0 = torch.randn_like(small_image)
        eps_hat = torch.randn_like(small_image)
        r1 = ddim_step(small_image, x0, eps_hat, 0.5, 0.6, zeta=0.0)
        r2 = ddim_step(small_image, x0, eps_hat, 0.5, 0.6, zeta=0.0)
        torch.testing.assert_close(r1, r2)

    def test_stochastic_when_zeta_positive(self, small_image):
        """With ζ > 0, ddim_step injects fresh noise (Eq. 15, √ζ · ε_t term).
        Two calls should produce different outputs (with overwhelming probability).
        """
        x0 = torch.randn_like(small_image)
        eps_hat = torch.randn_like(small_image)
        r1 = ddim_step(small_image, x0, eps_hat, 0.5, 0.6, zeta=0.3)
        r2 = ddim_step(small_image, x0, eps_hat, 0.5, 0.6, zeta=0.3)
        assert not torch.allclose(r1, r2)

    def test_ddim_step_formula_zeta_zero(self, small_image):
        """Verify the deterministic DDIM formula (Eq. 15 with ζ=0):
        x_{t-1} = √ᾱ_{t-1}·x̂₀ + √(1 − ᾱ_{t-1})·ε̂

        Hand-check with specific values to catch sign or coefficient errors.
        """
        ab_t = 0.4
        ab_prev = 0.6
        x0 = small_image
        eps_hat = torch.randn_like(small_image)
        result = ddim_step(small_image, x0, eps_hat, ab_t, ab_prev, zeta=0.0)
        expected = math.sqrt(ab_prev) * x0 + math.sqrt(1.0 - ab_prev) * eps_hat
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# build_ddim_timestep_sequence
# ---------------------------------------------------------------------------

class TestBuildDdimTimestepSequence:

    def test_correct_length(self):
        """Output list must have exactly n_steps elements."""
        seq = build_ddim_timestep_sequence(t_start=1000, n_steps=100, T=1000)
        assert len(seq) == 100

    def test_strictly_decreasing(self):
        """Timesteps must be strictly decreasing — the reverse process goes
        from high noise (large t) to clean (small t).  Non-decreasing steps
        would revisit noise levels and waste NFEs.
        """
        seq = build_ddim_timestep_sequence(t_start=1000, n_steps=100, T=1000)
        for i in range(len(seq) - 1):
            assert seq[i] > seq[i + 1], f"seq[{i}]={seq[i]} <= seq[{i+1}]={seq[i+1]}"

    def test_first_element_at_most_t_start(self):
        """The first timestep should not exceed t_start."""
        seq = build_ddim_timestep_sequence(t_start=800, n_steps=50, T=1000)
        assert seq[0] <= 800

    def test_all_positive(self):
        """All timesteps must be positive integers (1-indexed)."""
        seq = build_ddim_timestep_sequence(t_start=1000, n_steps=100, T=1000)
        assert all(t >= 1 for t in seq)

    def test_all_within_T(self):
        """No timestep should exceed T."""
        seq = build_ddim_timestep_sequence(t_start=1000, n_steps=100, T=1000)
        assert all(t <= 1000 for t in seq)

    def test_quadratic_concentrates_at_low_t(self):
        """The quadratic subsequence from DDIM [51] concentrates more steps
        in the low-noise region (small t).  The spacing between consecutive
        timesteps should generally increase as t increases.
        """
        seq = build_ddim_timestep_sequence(t_start=1000, n_steps=100, T=1000)
        # Check that the bottom half of timesteps has denser spacing
        mid = len(seq) // 2
        top_gaps = [seq[i] - seq[i + 1] for i in range(mid)]
        bot_gaps = [seq[i] - seq[i + 1] for i in range(mid, len(seq) - 1)]
        avg_top_gap = sum(top_gaps) / len(top_gaps)
        avg_bot_gap = sum(bot_gaps) / len(bot_gaps)
        assert avg_top_gap > avg_bot_gap, "quadratic should concentrate steps at low t"

    def test_small_n_steps(self):
        """Edge case: n_steps=1 should still produce a valid single-element list."""
        seq = build_ddim_timestep_sequence(t_start=1000, n_steps=1, T=1000)
        assert len(seq) == 1
        assert 1 <= seq[0] <= 1000


# ---------------------------------------------------------------------------
# precompute_rho_schedule
# ---------------------------------------------------------------------------

class TestPrecomputeRhoSchedule:

    def test_correct_length(self, noise_schedule):
        """Must return one ρ_t per timestep in the sequence."""
        timesteps = build_ddim_timestep_sequence(t_start=1000, n_steps=50, T=1000)
        cfg = SolverConfig(n_steps=50)
        rhos = precompute_rho_schedule(cfg, noise_schedule, timesteps)
        assert len(rhos) == 50

    def test_all_positive_with_positive_sigma_n(self, noise_schedule):
        """ρ_t > 0 when σ_n > 0 and λ > 0.  Negative ρ_t is physically meaningless."""
        timesteps = build_ddim_timestep_sequence(t_start=1000, n_steps=20, T=1000)
        cfg = SolverConfig(lambda_=8.0, sigma_n=0.05, n_steps=20)
        rhos = precompute_rho_schedule(cfg, noise_schedule, timesteps)
        assert all(r > 0 for r in rhos)

    def test_all_zero_when_sigma_n_zero(self, noise_schedule):
        """When σ_n = 0 (noiseless, e.g. inpainting), all ρ_t = 0."""
        timesteps = build_ddim_timestep_sequence(t_start=1000, n_steps=20, T=1000)
        cfg = SolverConfig(lambda_=6.0, sigma_n=0.0, n_steps=20)
        rhos = precompute_rho_schedule(cfg, noise_schedule, timesteps)
        assert all(r == pytest.approx(0.0, abs=1e-10) for r in rhos)

    def test_rho_values_are_floats(self, noise_schedule):
        """ρ_t values should be Python floats (or float-convertible) for
        compatibility with the data step functions.
        """
        timesteps = build_ddim_timestep_sequence(t_start=1000, n_steps=10, T=1000)
        cfg = SolverConfig(n_steps=10)
        rhos = precompute_rho_schedule(cfg, noise_schedule, timesteps)
        for r in rhos:
            assert isinstance(float(r), float)

    def test_hand_computed_single_value(self, noise_schedule):
        """Verify ρ_t for a specific timestep against hand computation.

        For the first timestep in a 10-step sequence with λ=8, σ_n=0.05:
        ρ_t = λ · σ_n² / σ_t² = 8 · 0.0025 / σ_t²

        σ_t = √((1−ᾱ_t)/ᾱ_t), so ρ_t = 8 · 0.0025 · ᾱ_t / (1−ᾱ_t)
        """
        timesteps = build_ddim_timestep_sequence(t_start=1000, n_steps=10, T=1000)
        cfg = SolverConfig(lambda_=8.0, sigma_n=0.05, n_steps=10)
        rhos = precompute_rho_schedule(cfg, noise_schedule, timesteps)

        # Manually compute for the first timestep
        t0 = timesteps[0]
        ab = noise_schedule["alpha_bars"][t0 - 1].item()
        sigma_t_sq = (1.0 - ab) / ab
        expected_rho = 8.0 * 0.05**2 / sigma_t_sq
        assert rhos[0] == pytest.approx(expected_rho, rel=1e-5)


# ---------------------------------------------------------------------------
# Regression: sigma_n → timestep mapping (Eq. 7)
# ---------------------------------------------------------------------------

class TestSigmaNToTimestepRegression:
    """Regression tests verifying the σ_n → timestep mapping at specific
    points, hand-computed from the DDPM linear schedule formulas.

    Paper: Section 3.1, connection between σ_t and the schedule.
    """

    def test_sigma_t_at_midpoint_timestep(self, noise_schedule):
        """At t=500 (midpoint of T=1000), σ_t should be a specific value
        derivable from the linear beta schedule.

        Hand computation:
          β_t = linspace(1e-4, 2e-2, 1000) → β_500 ≈ 0.01001
          α_s = 1 - β_s for s=1..500
          ᾱ_500 = ∏ α_s
          σ_500 = √((1 - ᾱ_500) / ᾱ_500)

        We verify against the schedule's stored values to catch divergence
        between make_noise_schedule and compute_sigma_t.
        """
        ab_500 = noise_schedule["alpha_bars"][499].item()
        sigma_from_schedule = noise_schedule["sigma_ts"][499].item()
        sigma_from_func = compute_sigma_t(torch.tensor(ab_500)).item()
        assert sigma_from_schedule == pytest.approx(sigma_from_func, rel=1e-5)
        # σ_500 should be > 1 (more noise than signal at midpoint)
        assert sigma_from_func > 1.0

    def test_rho_t_hand_computed_for_deblur_config(self, noise_schedule):
        """Verify ρ_t for the deblur (Gaussian) config from Table 3:
        λ=7.0, σ_n=0.05, at a specific timestep.

        For t=100 (1-indexed): ρ_t = 7.0 × 0.05² / σ_t²
        σ_t = σ_ts[99] from the schedule.
        """
        sigma_t = noise_schedule["sigma_ts"][99].item()
        rho_expected = 7.0 * 0.05**2 / sigma_t**2
        rho_computed = compute_rho_t(lambda_=7.0, sigma_n=0.05, sigma_t=sigma_t)
        assert rho_computed.item() == pytest.approx(rho_expected, rel=1e-5)

    def test_rho_schedule_decreases_over_reverse_process(self, noise_schedule):
        """ρ_t = λσ_n²/σ_t² should increase as t decreases (σ_t decreases),
        meaning the data step trusts measurements more at later (lower-noise)
        stages.  This is the core scheduling insight of DiffPIR (Algorithm 1).
        """
        timesteps = build_ddim_timestep_sequence(t_start=1000, n_steps=50, T=1000)
        cfg = SolverConfig(lambda_=8.0, sigma_n=0.05, n_steps=50)
        rhos = precompute_rho_schedule(cfg, noise_schedule, timesteps)
        # rhos should generally increase (since timesteps decrease → σ_t decreases)
        # Allow small violations due to quadratic spacing
        increases = sum(1 for i in range(len(rhos) - 1) if rhos[i + 1] >= rhos[i])
        assert increases > len(rhos) * 0.8, "ρ_t should mostly increase over reverse process"


# ---------------------------------------------------------------------------
# Numerical stability at extreme timesteps
# ---------------------------------------------------------------------------

class TestDiffusionNumericalStability:
    """Verify numerical stability of diffusion math at boundary timesteps
    where ᾱ_t is near 0 or 1, and divisions can blow up.
    """

    def test_predict_x0_stable_at_t_near_T(self, noise_schedule, small_image):
        """At t≈T, ᾱ_t ≈ 0.  predict_x0 divides by √ᾱ_t, which could blow up.
        The output must remain finite even when ᾱ_t is very small.
        """
        eps = torch.randn_like(small_image)
        ab_T = noise_schedule["alpha_bars"][-1]
        result = predict_x0(small_image, eps, ab_T)
        assert torch.isfinite(result).all(), "predict_x0 blew up at t≈T"

    def test_compute_effective_noise_stable_at_small_alpha_bar(self, small_image):
        """compute_effective_noise divides by √(1−ᾱ_t).  When ᾱ_t ≈ 1,
        (1−ᾱ_t) ≈ 0 and division could blow up.  Must remain finite.
        """
        x0 = torch.randn_like(small_image)
        # ᾱ_t very close to 1 but not exactly 1
        result = compute_effective_noise(small_image, x0, alpha_bar_t=0.9999)
        assert torch.isfinite(result).all(), "Effective noise blew up at ᾱ_t≈1"

    def test_ddim_step_stable_at_extreme_alpha_bars(self, small_image):
        """ddim_step with ᾱ_{t-1} very close to 0 or 1 must not produce NaN/Inf.
        The formula involves √(1 − ᾱ_{t-1} − ζ) which can become negative
        if ζ > 1 − ᾱ_{t-1}.
        """
        x0 = torch.randn_like(small_image)
        eps_hat = torch.randn_like(small_image)

        # ᾱ_{t-1} ≈ 0: √(1 − 0.001 − 0) ≈ 1 (safe)
        result = ddim_step(small_image, x0, eps_hat, 0.001, 0.002, zeta=0.0)
        assert torch.isfinite(result).all(), "ddim_step blew up at ᾱ_prev≈0"

        # ᾱ_{t-1} ≈ 1: √(1 − 0.999 − 0) ≈ 0.03 (safe but small)
        result = ddim_step(small_image, x0, eps_hat, 0.998, 0.999, zeta=0.0)
        assert torch.isfinite(result).all(), "ddim_step blew up at ᾱ_prev≈1"

    def test_add_noise_at_t_equals_T(self, noise_schedule, small_image):
        """At t=T, ᾱ_T ≈ 0, so x_T ≈ pure noise.  The signal component
        √ᾱ_T · x₀ should be negligible compared to the noise component.
        """
        noise = torch.randn_like(small_image)
        x_T = add_noise(small_image, t=1000, noise_schedule=noise_schedule, noise=noise)
        ab_T = noise_schedule["alpha_bars"][-1].item()
        # Signal power ≈ ᾱ_T · ||x₀||², noise power ≈ (1−ᾱ_T) · ||ε||²
        signal_power = ab_T * (small_image ** 2).mean().item()
        noise_power = (1 - ab_T) * (noise ** 2).mean().item()
        assert noise_power > 100 * signal_power, \
            f"At t=T, noise should dominate: signal={signal_power:.6f}, noise={noise_power:.6f}"
