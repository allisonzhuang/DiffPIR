"""Tests for train/loss.py — diffusion training loss (Eq. 4).

Verifies the denoising score matching objective:
    L = E_t { γ(t) · E_{x₀,ε} [ ||ε_θ(x_t, t) − ε||² ] }

The loss is the backbone of diffusion model training; these tests ensure
correct reduction, weighting, and boundary behaviour.
"""

import pytest
import torch

from train.loss import diffusion_loss


# ---------------------------------------------------------------------------
# Basic contract tests
# ---------------------------------------------------------------------------

class TestDiffusionLossContract:

    def test_scalar_output(self):
        """diffusion_loss must return a scalar tensor (0-dim).  A non-scalar
        loss would break optimizer.step() and logging.
        """
        eps_pred = torch.randn(4, 3, 16, 16)
        eps_target = torch.randn(4, 3, 16, 16)
        loss = diffusion_loss(eps_pred, eps_target)
        assert loss.dim() == 0

    def test_output_dtype_float32(self):
        """Loss must be float32 for stable gradient computation."""
        eps_pred = torch.randn(4, 3, 16, 16)
        eps_target = torch.randn(4, 3, 16, 16)
        loss = diffusion_loss(eps_pred, eps_target)
        assert loss.dtype == torch.float32

    def test_requires_grad(self):
        """The loss must support backpropagation (requires_grad=True when
        inputs require grad).
        """
        eps_pred = torch.randn(4, 3, 8, 8, requires_grad=True)
        eps_target = torch.randn(4, 3, 8, 8)
        loss = diffusion_loss(eps_pred, eps_target)
        loss.backward()
        assert eps_pred.grad is not None


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------

class TestDiffusionLossMath:

    def test_zero_when_perfect_prediction(self):
        """When ε_θ = ε (perfect noise prediction), loss = 0.  This is the
        global minimum of the MSE objective.
        """
        eps = torch.randn(2, 3, 16, 16)
        loss = diffusion_loss(eps, eps)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_for_imperfect_prediction(self):
        """When ε_θ ≠ ε, loss > 0.  The MSE between distinct tensors is
        strictly positive.
        """
        torch.manual_seed(0)
        eps_pred = torch.randn(2, 3, 16, 16)
        eps_target = torch.randn(2, 3, 16, 16)
        loss = diffusion_loss(eps_pred, eps_target)
        assert loss.item() > 0

    def test_known_value(self):
        """Hand-computed test case:
        eps_pred = [[1, 0], [0, 1]], eps_target = [[0, 0], [0, 0]]
        MSE = mean(1² + 0² + 0² + 1²) = 2/4 = 0.5

        This catches reduction mode errors (sum vs mean).
        """
        eps_pred = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])   # (1,1,2,2)
        eps_target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
        loss = diffusion_loss(eps_pred, eps_target)
        assert loss.item() == pytest.approx(0.5, abs=1e-6)

    def test_symmetric(self):
        """MSE is symmetric: L(a, b) = L(b, a).  Asymmetric loss would indicate
        the wrong formula.
        """
        torch.manual_seed(42)
        a = torch.randn(2, 3, 8, 8)
        b = torch.randn(2, 3, 8, 8)
        assert diffusion_loss(a, b).item() == pytest.approx(diffusion_loss(b, a).item(), abs=1e-7)

    def test_scales_quadratically(self):
        """Doubling the error should quadruple the loss (MSE scales as ||e||²).
        L(2ε, 0) = 4 · L(ε, 0).
        """
        eps = torch.randn(2, 3, 8, 8)
        zero = torch.zeros_like(eps)
        loss_1x = diffusion_loss(eps, zero)
        loss_2x = diffusion_loss(2 * eps, zero)
        assert loss_2x.item() == pytest.approx(4.0 * loss_1x.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# Weighted loss
# ---------------------------------------------------------------------------

class TestDiffusionLossWeighted:

    def test_uniform_weights_equal_unweighted(self):
        """Weights of all 1s should give the same result as no weights.
        This verifies the weighting path doesn't accidentally scale the loss.
        """
        torch.manual_seed(0)
        eps_pred = torch.randn(4, 3, 8, 8)
        eps_target = torch.randn(4, 3, 8, 8)
        loss_no_w = diffusion_loss(eps_pred, eps_target)
        loss_w1 = diffusion_loss(eps_pred, eps_target, weights=torch.ones(4))
        assert loss_no_w.item() == pytest.approx(loss_w1.item(), abs=1e-6)

    def test_zero_weight_zeroes_contribution(self):
        """A sample with weight=0 should not contribute to the loss.
        If we zero-weight one sample and set the other to perfect prediction,
        the loss should be 0.
        """
        eps_pred = torch.randn(2, 1, 4, 4)
        eps_target = eps_pred.clone()
        # Make sample 0 have a large error
        eps_pred_modified = eps_pred.clone()
        eps_pred_modified[0] += 10.0
        weights = torch.tensor([0.0, 1.0])
        loss = diffusion_loss(eps_pred_modified, eps_target, weights=weights)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_higher_weight_increases_contribution(self):
        """A sample with higher weight should have proportionally larger
        influence on the total loss.
        """
        torch.manual_seed(0)
        eps_pred = torch.randn(2, 1, 4, 4)
        eps_target = torch.zeros_like(eps_pred)
        w_uniform = torch.tensor([1.0, 1.0])
        w_biased = torch.tensor([2.0, 1.0])
        loss_uniform = diffusion_loss(eps_pred, eps_target, weights=w_uniform)
        loss_biased = diffusion_loss(eps_pred, eps_target, weights=w_biased)
        # If sample 0 has a different MSE than sample 1, the losses should differ
        if (eps_pred[0].norm() - eps_pred[1].norm()).abs() > 0.01:
            assert loss_uniform.item() != pytest.approx(loss_biased.item(), abs=1e-4)


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------

class TestDiffusionLossGradients:
    """Verify that the loss produces meaningful gradients for training."""

    def test_gradient_nonzero_for_imperfect_prediction(self):
        """When ε_θ ≠ ε, the gradient ∂L/∂ε_θ must be nonzero.  Zero gradients
        would mean the model cannot learn from its errors.
        """
        eps_pred = torch.randn(2, 3, 8, 8, requires_grad=True)
        eps_target = torch.randn(2, 3, 8, 8)
        loss = diffusion_loss(eps_pred, eps_target)
        loss.backward()
        assert eps_pred.grad is not None
        assert eps_pred.grad.abs().sum().item() > 0

    def test_gradient_zero_at_optimum(self):
        """When ε_θ = ε (perfect prediction), the gradient should be zero
        at the global minimum of the MSE objective.
        """
        eps = torch.randn(2, 3, 8, 8)
        eps_pred = eps.clone().requires_grad_(True)
        loss = diffusion_loss(eps_pred, eps)
        loss.backward()
        torch.testing.assert_close(eps_pred.grad, torch.zeros_like(eps_pred.grad),
                                   atol=1e-6, rtol=1e-6)

    def test_gradient_direction_correct(self):
        """The gradient should point from ε_θ toward ε.  Specifically,
        ∂L/∂ε_θ ∝ (ε_θ − ε), so the gradient should be proportional to
        the error vector.
        """
        eps_target = torch.zeros(1, 1, 4, 4)
        eps_pred = torch.ones(1, 1, 4, 4, requires_grad=True)
        loss = diffusion_loss(eps_pred, eps_target)
        loss.backward()
        # Gradient should be proportional to (eps_pred - eps_target) = eps_pred
        # For MSE mean reduction: grad = 2*(eps_pred - eps_target) / numel
        expected_grad = 2 * (eps_pred.data - eps_target) / eps_pred.numel()
        torch.testing.assert_close(eps_pred.grad, expected_grad, atol=1e-6, rtol=1e-6)
