"""Tests for restoration/pnp.py — hqs_step and drs_step.

Both functions accept two callable proximal operators (prox_f, prox_g) and an
iterate x, and return the next iterate.  Tests verify shape/dtype contracts,
correct delegation order, and basic convergence properties.
"""

import pytest
import torch
from torch import Tensor

from restoration.pnp import hqs_step, drs_step


@pytest.fixture
def x():
    torch.manual_seed(42)
    return torch.randn(2, 3, 16, 16)


def identity(z: Tensor) -> Tensor:
    return z


def half(z: Tensor) -> Tensor:
    return 0.5 * z


# ---------------------------------------------------------------------------
# hqs_step
# ---------------------------------------------------------------------------

class TestHqsStep:

    def test_output_shape(self, x):
        """hqs_step must return a tensor with the same shape as x."""
        result = hqs_step(identity, identity, x)
        assert result.shape == x.shape

    def test_output_dtype(self, x):
        """Output dtype must match input dtype."""
        result = hqs_step(identity, identity, x)
        assert result.dtype == x.dtype

    def test_identity_proxes_return_input(self, x):
        """With identity proxes, hqs_step should return x unchanged."""
        result = hqs_step(identity, identity, x)
        torch.testing.assert_close(result, x)

    def test_applies_prox_g_then_prox_f(self, x):
        """HQS order: z = prox_g(x), result = prox_f(z).

        With prox_g = half and prox_f = identity:
          z = 0.5 * x
          result = z = 0.5 * x
        """
        result = hqs_step(identity, half, x)
        torch.testing.assert_close(result, 0.5 * x)

    def test_prox_f_applied_last(self, x):
        """prox_f is applied after prox_g (HQS convention: prior first, data second).

        With prox_g = identity and prox_f = half:
          z = x
          result = 0.5 * x
        """
        result = hqs_step(half, identity, x)
        torch.testing.assert_close(result, 0.5 * x)

    def test_composition(self, x):
        """With prox_g = half and prox_f = half:
          z = 0.5 * x
          result = 0.25 * x
        """
        result = hqs_step(half, half, x)
        torch.testing.assert_close(result, 0.25 * x)

    def test_finite_output(self, x):
        """Output must contain no NaN or Inf."""
        result = hqs_step(identity, identity, x)
        assert torch.isfinite(result).all()

    def test_delegation_count(self, x):
        """Each prox is called exactly once per hqs_step call."""
        calls = {"f": 0, "g": 0}

        def counting_f(z):
            calls["f"] += 1
            return z

        def counting_g(z):
            calls["g"] += 1
            return z

        hqs_step(counting_f, counting_g, x)
        assert calls["f"] == 1
        assert calls["g"] == 1


# ---------------------------------------------------------------------------
# drs_step
# ---------------------------------------------------------------------------

class TestDrsStep:

    def test_output_shape(self, x):
        """drs_step must return a tensor with the same shape as x."""
        result = drs_step(identity, identity, x)
        assert result.shape == x.shape

    def test_output_dtype(self, x):
        """Output dtype must match input dtype."""
        result = drs_step(identity, identity, x)
        assert result.dtype == x.dtype

    def test_identity_proxes_return_input(self, x):
        """With identity proxes, drs_step should return x unchanged.

        DRS fixed point with prox_f = prox_g = I:
          z = 2*x - x = x
          x_hat = 2*x - x = x
          return (x + x) / 2 = x
        """
        result = drs_step(identity, identity, x)
        torch.testing.assert_close(result, x)

    def test_finite_output(self, x):
        """Output must contain no NaN or Inf."""
        result = drs_step(identity, identity, x)
        assert torch.isfinite(result).all()

    def test_delegation_count(self, x):
        """Each prox is called exactly once per drs_step call."""
        calls = {"f": 0, "g": 0}

        def counting_f(z):
            calls["f"] += 1
            return z

        def counting_g(z):
            calls["g"] += 1
            return z

        drs_step(counting_f, counting_g, x)
        assert calls["f"] == 1
        assert calls["g"] == 1

    def test_different_from_hqs(self, x):
        """DRS and HQS produce different results for the same pair of non-trivial proxes.

        With prox_f = half and prox_g = half:
          HQS: z = half(x) = 0.5x, result = half(z) = 0.25x
          DRS: z = 2*half(x) - x = 0, x_hat = 2*half(0) - 0 = 0, return (x+0)/2 = 0.5x
        """
        result_hqs = hqs_step(half, half, x)
        result_drs = drs_step(half, half, x)
        assert not torch.allclose(result_hqs, result_drs)
