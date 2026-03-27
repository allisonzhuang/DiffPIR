"""Tests for build_noise_scheduler (restoration/diffpir.py).

Verifies the mathematical properties of the DDPM noise schedule (Eqs. 5–6).
"""

import math
import pytest
import torch

from configs import DiffusionConfig
from restoration.diffpir import build_noise_scheduler


@pytest.fixture
def schedule():
    return build_noise_scheduler(DiffusionConfig())


class TestBuildNoiseScheduler:

    def test_output_keys(self, schedule):
        """Returned dict must contain the documented keys."""
        assert {"T", "beta", "alpha", "alpha_bar", "sigma_bar"}.issubset(schedule.keys())

    def test_lengths(self, schedule):
        """Each list must have T+1 entries (index 0 … T inclusive)."""
        T = schedule["T"]
        for key in ("beta", "alpha", "alpha_bar", "sigma_bar"):
            assert len(schedule[key]) == T + 1, f"{key} has wrong length"

    def test_beta_range(self, schedule):
        """β_t must lie in (0, 1) for all t."""
        for t, b in enumerate(schedule["beta"]):
            assert 0 < b < 1, f"beta[{t}]={b} out of range"

    def test_alpha_equals_one_minus_beta(self, schedule):
        """α_t = 1 − β_t for all t."""
        for t in range(schedule["T"] + 1):
            assert schedule["alpha"][t] == pytest.approx(1 - schedule["beta"][t], abs=1e-7)

    def test_alpha_bar_monotone_decreasing(self, schedule):
        """ᾱ_t must decrease monotonically (more noise at later timesteps)."""
        ab = schedule["alpha_bar"]
        for t in range(1, schedule["T"] + 1):
            assert ab[t] < ab[t - 1], f"alpha_bar not decreasing at t={t}"

    def test_alpha_bar_in_unit_interval(self, schedule):
        """ᾱ_t ∈ (0, 1) for all t."""
        for t, ab in enumerate(schedule["alpha_bar"]):
            assert 0 < ab <= 1, f"alpha_bar[{t}]={ab} out of (0,1]"

    def test_sigma_bar_non_negative(self, schedule):
        """σ_t = √((1−ᾱ_t)/ᾱ_t) must be ≥ 0."""
        for t, s in enumerate(schedule["sigma_bar"]):
            assert s >= 0, f"sigma_bar[{t}]={s} is negative"

    def test_sigma_bar_formula(self, schedule):
        """Verify σ_t = √((1−ᾱ_t)/ᾱ_t) numerically."""
        for t in range(1, schedule["T"] + 1):
            ab = schedule["alpha_bar"][t]
            expected = math.sqrt((1 - ab) / ab)
            assert schedule["sigma_bar"][t] == pytest.approx(expected, rel=1e-5)

    def test_beta_linear(self, schedule):
        """β_t should be a linear interpolation from beta_start to beta_end."""
        T = schedule["T"]
        cfg = DiffusionConfig()
        for t in range(T + 1):
            expected = cfg.beta_start + t * (cfg.beta_end - cfg.beta_start) / T
            assert schedule["beta"][t] == pytest.approx(expected, rel=1e-6)

    def test_alpha_bar_decreasing_to_near_zero(self, schedule):
        """ᾱ_T should be very small (close to 0) — fully noisy at T."""
        assert schedule["alpha_bar"][schedule["T"]] < 0.01

    def test_sigma_bar_large_at_T(self, schedule):
        """σ_T should be large — signal is almost entirely noise at T."""
        assert schedule["sigma_bar"][schedule["T"]] > 1.0

    def test_custom_config(self):
        """build_noise_scheduler should respect non-default DiffusionConfig values."""
        cfg = DiffusionConfig(T=200, beta_start=1e-3, beta_end=1e-2)
        s = build_noise_scheduler(cfg)
        assert s["T"] == 200
        assert len(s["beta"]) == 201
        assert s["beta"][0] == pytest.approx(1e-3, rel=1e-6)
        assert s["beta"][200] == pytest.approx(1e-2, rel=1e-6)
