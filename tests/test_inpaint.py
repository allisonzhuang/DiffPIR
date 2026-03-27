"""Tests for degradations/inpaint.py — inpainting degradation and solver.

Verifies the inpainting forward model y = M ⊙ x (Eq. 25) and the closed-form
data subproblem solution x₀ = (M⊙y + ρ_t·z₀) / (M + ρ_t) (Eq. 26).
"""

import pytest
import torch

from configs import InpaintConfig
from interfaces import PnPSolver
from degradations.inpaint import (
    InpaintingDegradation,
    InpaintingPnPSolver,
    build_box_mask,
    build_random_mask,
    inpaint_data_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ones_image():
    """All-ones image for easy manual verification."""
    return torch.ones(1, 3, 64, 64)


@pytest.fixture
def random_image():
    torch.manual_seed(0)
    return torch.randn(2, 3, 64, 64)


# ---------------------------------------------------------------------------
# build_box_mask
# ---------------------------------------------------------------------------

class TestBuildBoxMask:

    def test_shape(self):
        """build_box_mask must return shape (1, 1, H, W) for broadcasting."""
        mask = build_box_mask(256, 256, box_size=128)
        assert mask.shape == (1, 1, 256, 256)

    def test_dtype_float32(self):
        """Mask must be float32 for element-wise multiplication with images."""
        mask = build_box_mask(64, 64, box_size=32)
        assert mask.dtype == torch.float32

    def test_binary_values(self):
        """Mask entries must be exactly 0 or 1 — no intermediate values."""
        mask = build_box_mask(64, 64, box_size=32)
        unique = torch.unique(mask)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_zeros_inside_centred_box(self):
        """The centred box_size×box_size region should be 0 (masked/missing).
        Paper uses 128×128 centred box on 256×256 images.
        """
        H, W, bs = 64, 64, 32
        mask = build_box_mask(H, W, box_size=bs)
        cy, cx = H // 2, W // 2
        half = bs // 2
        box_region = mask[0, 0, cy - half:cy + half, cx - half:cx + half]
        assert box_region.sum().item() == 0.0

    def test_ones_outside_box(self):
        """Pixels outside the box should be 1 (observed/known)."""
        H, W, bs = 64, 64, 32
        mask = build_box_mask(H, W, box_size=bs)
        total_ones = mask.sum().item()
        expected_ones = H * W - bs * bs
        assert total_ones == pytest.approx(expected_ones, abs=1.0)

    def test_correct_mask_count(self):
        """The number of masked pixels should equal box_size²."""
        mask = build_box_mask(128, 128, box_size=64)
        n_masked = (mask == 0).sum().item()
        assert n_masked == 64 * 64


# ---------------------------------------------------------------------------
# build_random_mask
# ---------------------------------------------------------------------------

class TestBuildRandomMask:

    def test_shape(self):
        """build_random_mask must return shape (1, 1, H, W)."""
        mask = build_random_mask(64, 64, fraction=0.5)
        assert mask.shape == (1, 1, 64, 64)

    def test_dtype_float32(self):
        """Mask must be float32."""
        mask = build_random_mask(64, 64, fraction=0.5)
        assert mask.dtype == torch.float32

    def test_binary_values(self):
        """Mask entries must be exactly 0 or 1."""
        mask = build_random_mask(64, 64, fraction=0.5)
        unique = torch.unique(mask)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_approximate_fraction(self):
        """The fraction of masked (=0) pixels should be approximately equal
        to the requested fraction.  Allow 5% tolerance for randomness.
        """
        mask = build_random_mask(256, 256, fraction=0.5)
        n_masked = (mask == 0).sum().item()
        total = 256 * 256
        assert n_masked / total == pytest.approx(0.5, abs=0.05)

    def test_fraction_zero_means_all_observed(self):
        """fraction=0 should produce an all-ones mask (nothing masked)."""
        mask = build_random_mask(32, 32, fraction=0.0)
        assert mask.sum().item() == 32 * 32

    def test_fraction_one_means_all_masked(self):
        """fraction=1 should produce an all-zeros mask (everything masked)."""
        mask = build_random_mask(32, 32, fraction=1.0)
        assert mask.sum().item() == 0.0


# ---------------------------------------------------------------------------
# InpaintingDegradation.apply
# ---------------------------------------------------------------------------

class TestInpaintingDegradationApply:

    def test_zeros_masked_pixels(self, ones_image):
        """InpaintingDegradation.apply must set masked (M=0) pixels to 0.
        y = M ⊙ x, so wherever M=0, y=0 regardless of x.
        """
        mask = torch.zeros(1, 1, 64, 64)  # all masked
        deg = InpaintingDegradation(mask, InpaintConfig())
        y = deg.apply(ones_image)
        assert y.sum().item() == 0.0

    def test_preserves_known_pixels(self, random_image):
        """InpaintingDegradation.apply must leave observed (M=1) pixels unchanged.
        y = M ⊙ x = x when M=1 everywhere.
        """
        mask = torch.ones(1, 1, 64, 64)  # all observed
        deg = InpaintingDegradation(mask, InpaintConfig())
        y = deg.apply(random_image)
        torch.testing.assert_close(y, random_image)

    def test_output_shape(self, random_image):
        """Output shape must match input shape."""
        mask = build_box_mask(64, 64, box_size=32)
        deg = InpaintingDegradation(mask, InpaintConfig())
        y = deg.apply(random_image)
        assert y.shape == random_image.shape

    def test_output_dtype_float32(self, random_image):
        """Output must be float32."""
        mask = build_box_mask(64, 64, box_size=32)
        deg = InpaintingDegradation(mask, InpaintConfig())
        y = deg.apply(random_image)
        assert y.dtype == torch.float32

    def test_adjoint_property(self, random_image):
        """For the masking operator H = diag(M), H^T H = diag(M).
        This is because applying the mask twice is the same as applying it once
        (idempotent): M ⊙ (M ⊙ x) = M ⊙ x.
        """
        mask = build_box_mask(64, 64, box_size=32)
        deg = InpaintingDegradation(mask, InpaintConfig())
        y = deg.apply(random_image)
        y2 = deg.apply(y)
        torch.testing.assert_close(y, y2)


# ---------------------------------------------------------------------------
# inpaint_data_step (Eq. 26)
# ---------------------------------------------------------------------------

class TestInpaintDataStep:

    def test_output_shape(self, random_image):
        """inpaint_data_step output must have the same shape as x0_prior."""
        mask = build_box_mask(64, 64, box_size=32)
        y = mask * random_image
        result = inpaint_data_step(random_image, y, mask, rho_t=1.0)
        assert result.shape == random_image.shape

    def test_output_dtype_float32(self, random_image):
        """Output must be float32."""
        mask = build_box_mask(64, 64, box_size=32)
        y = mask * random_image
        result = inpaint_data_step(random_image, y, mask, rho_t=1.0)
        assert result.dtype == torch.float32

    def test_eq26_hand_computed(self):
        """Verify Eq. 26 against hand computation.

        Given: mask M=[[1,0],[0,1]], y=[[5,0],[0,3]], z₀=[[2,4],[6,8]], ρ_t=2.0
        Expected per Eq. 26: x₀ = (M⊙y + ρ_t·z₀) / (M + ρ_t)

        For pixel (0,0): M=1, y=5, z₀=2 → (5 + 4) / (1+2) = 3.0
        For pixel (0,1): M=0, y=0, z₀=4 → (0 + 8) / (0+2) = 4.0
        For pixel (1,0): M=0, y=0, z₀=6 → (0 + 12) / (0+2) = 6.0
        For pixel (1,1): M=1, y=3, z₀=8 → (3 + 16) / (1+2) = 19/3 ≈ 6.333

        This catches off-by-one errors in the formula implementation.
        """
        mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # (1,1,2,2)
        y = torch.tensor([[[[5.0, 0.0], [0.0, 3.0]]]])
        z0 = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
        rho_t = 2.0

        result = inpaint_data_step(z0, y, mask, rho_t)
        expected = torch.tensor([[[[3.0, 4.0], [6.0, 19.0 / 3.0]]]])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_known_pixels_biased_toward_y(self):
        """At observed pixels (M=1), the result should interpolate between y
        and z₀, weighted by ρ_t.  As ρ_t → 0, the result → y at those pixels.
        """
        mask = torch.ones(1, 1, 4, 4)
        y = torch.ones(1, 1, 4, 4) * 10.0
        z0 = torch.zeros(1, 1, 4, 4)
        result = inpaint_data_step(z0, y, mask, rho_t=0.001)
        # (1*10 + 0.001*0) / (1 + 0.001) ≈ 9.99
        assert (result > 9.9).all()

    def test_masked_pixels_equal_prior(self):
        """At masked pixels (M=0), x₀ = (0 + ρ_t·z₀) / (0 + ρ_t) = z₀.
        The data step should not modify masked pixels — only the prior matters.
        """
        mask = torch.zeros(1, 1, 4, 4)
        y = torch.ones(1, 1, 4, 4) * 100.0  # should be ignored
        z0 = torch.ones(1, 1, 4, 4) * 7.0
        result = inpaint_data_step(z0, y, mask, rho_t=1.0)
        torch.testing.assert_close(result, z0, atol=1e-6, rtol=1e-6)

    def test_large_rho_recovers_prior(self):
        """As ρ_t → ∞, the solution → z₀ everywhere (prior dominates).
        x₀ = (M⊙y + ρ_t·z₀) / (M + ρ_t) → z₀ as ρ_t → ∞.
        """
        mask = torch.ones(1, 1, 4, 4)
        y = torch.ones(1, 1, 4, 4) * 10.0
        z0 = torch.ones(1, 1, 4, 4) * 5.0
        result = inpaint_data_step(z0, y, mask, rho_t=1e6)
        torch.testing.assert_close(result, z0, atol=0.01, rtol=0.01)


# ---------------------------------------------------------------------------
# InpaintingPnPSolver
# ---------------------------------------------------------------------------

class TestInpaintingPnPSolver:

    def test_implements_pnp_solver(self):
        """InpaintingPnPSolver must be a valid PnPSolver subclass.
        This is the interface contract that allows it to be used in the HQS loop.
        """
        assert issubclass(InpaintingPnPSolver, PnPSolver)
        solver = InpaintingPnPSolver()
        assert isinstance(solver, PnPSolver)

    def test_delegates_to_inpaint_data_step(self):
        """The solver should produce the same output as calling
        inpaint_data_step directly with the degradation's mask.
        """
        mask = build_box_mask(32, 32, box_size=16)
        cfg = InpaintConfig()
        deg = InpaintingDegradation(mask, cfg)

        torch.manual_seed(0)
        x0_prior = torch.randn(1, 3, 32, 32)
        y = mask * x0_prior

        solver = InpaintingPnPSolver()
        result = solver.data_step(x0_prior, y, deg, rho_t=1.0)
        expected = inpaint_data_step(x0_prior, y, mask, rho_t=1.0)
        torch.testing.assert_close(result, expected)

    def test_output_shape(self):
        """Solver output shape must match x0_prior shape."""
        mask = build_box_mask(32, 32, box_size=16)
        deg = InpaintingDegradation(mask, InpaintConfig())
        x0 = torch.randn(2, 3, 32, 32)
        y = mask * x0
        result = InpaintingPnPSolver().data_step(x0, y, deg, rho_t=1.0)
        assert result.shape == x0.shape


# ---------------------------------------------------------------------------
# Mask broadcasting and edge cases
# ---------------------------------------------------------------------------

class TestInpaintBroadcasting:
    """Verify that mask broadcasting works correctly across batch and channel
    dimensions.  The mask has shape (1,1,H,W) but must work with (B,C,H,W) images.
    """

    def test_mask_broadcasts_over_batch(self):
        """A single mask (1,1,H,W) must apply identically to every sample in
        the batch.  If broadcasting is broken, different batch elements get
        different masks, corrupting the data fidelity term.
        """
        mask = build_box_mask(32, 32, box_size=16)
        x = torch.randn(4, 3, 32, 32)
        deg = InpaintingDegradation(mask, InpaintConfig())
        y = deg.apply(x)
        # All batch elements should have the same mask pattern
        for b in range(4):
            masked_pixels = (y[b] == 0).all(dim=0)  # across channels
            expected_masked = (mask[0, 0] == 0)
            assert torch.equal(masked_pixels, expected_masked), \
                f"Mask pattern differs for batch element {b}"

    def test_mask_broadcasts_over_channels(self):
        """The (1,1,H,W) mask must apply the same pattern to all C channels.
        If a pixel is masked, it must be masked in R, G, and B simultaneously.
        """
        mask = build_box_mask(32, 32, box_size=16)
        x = torch.ones(1, 3, 32, 32)
        deg = InpaintingDegradation(mask, InpaintConfig())
        y = deg.apply(x)
        # All channels at a masked pixel should be zero
        for c in range(3):
            torch.testing.assert_close(y[0, c], mask[0, 0] * x[0, c])

    def test_inpaint_data_step_broadcasts_mask(self):
        """inpaint_data_step with mask (1,1,H,W) and image (B,C,H,W) must
        broadcast correctly.  Verify the element-wise formula holds per-pixel.
        """
        mask = build_random_mask(8, 8, fraction=0.5)
        x0 = torch.randn(2, 3, 8, 8)
        y = mask * x0
        rho_t = 1.5
        result = inpaint_data_step(x0, y, mask, rho_t)
        expected = (mask * y + rho_t * x0) / (mask + rho_t)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


class TestInpaintEdgeCases:
    """Edge cases that could break the inpainting data step formula."""

    def test_all_pixels_observed_reduces_to_weighted_avg(self):
        """When M=1 everywhere, Eq. 26 becomes x₀ = (y + ρ_t·z₀)/(1 + ρ_t).
        The solution interpolates between the measurement and the prior.
        """
        mask = torch.ones(1, 1, 8, 8)
        y = torch.ones(1, 1, 8, 8) * 3.0
        z0 = torch.ones(1, 1, 8, 8) * 7.0
        rho_t = 1.0
        result = inpaint_data_step(z0, y, mask, rho_t)
        expected_val = (3.0 + 1.0 * 7.0) / (1.0 + 1.0)  # = 5.0
        torch.testing.assert_close(result, torch.full_like(result, expected_val),
                                   atol=1e-6, rtol=1e-6)

    def test_all_pixels_masked_returns_prior(self):
        """When M=0 everywhere, x₀ = (0 + ρ_t·z₀)/(0 + ρ_t) = z₀.
        The data step must return the prior when no measurements exist.
        """
        mask = torch.zeros(1, 1, 8, 8)
        z0 = torch.randn(1, 3, 8, 8)
        y = torch.zeros(1, 3, 8, 8)
        result = inpaint_data_step(z0, y, mask, rho_t=5.0)
        torch.testing.assert_close(result, z0, atol=1e-6, rtol=1e-6)

    def test_rho_zero_at_observed_pixels_returns_y(self):
        """When ρ_t=0 and M=1: x₀ = (y + 0) / (1 + 0) = y.
        Zero penalty means full trust in the measurement (noiseless case).
        This is the inpainting regime σ_n=0 used in the paper.
        """
        mask = torch.ones(1, 1, 4, 4)
        y = torch.randn(1, 3, 4, 4)
        z0 = torch.randn(1, 3, 4, 4)
        # ρ_t very close to 0 (avoid exact 0 to prevent division issues)
        result = inpaint_data_step(z0, y, mask, rho_t=1e-10)
        torch.testing.assert_close(result, y, atol=1e-4, rtol=1e-4)
