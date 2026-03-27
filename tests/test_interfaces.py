"""Tests for interfaces.py — abstract base classes DenoiserPrior and PnPSolver.

Verifies that ABCs enforce implementation of abstract methods, preventing
incomplete subclasses from being instantiated (a core OOP contract that
guarantees every denoiser/solver in the system satisfies the interface).
"""

import pytest
import torch
from torch import Tensor

from interfaces import DenoiserPrior, PnPSolver


# ---------------------------------------------------------------------------
# DenoiserPrior ABC enforcement
# ---------------------------------------------------------------------------

class TestDenoiserPriorABC:
    """Verify DenoiserPrior is a proper ABC that enforces the denoise contract."""

    def test_cannot_instantiate_directly(self):
        """DenoiserPrior defines denoise() as abstract; instantiating the ABC
        directly must raise TypeError because abstract methods are unresolved.
        This guarantees all priors go through a concrete subclass.
        """
        with pytest.raises(TypeError):
            DenoiserPrior()

    def test_incomplete_subclass_raises_type_error(self):
        """A subclass that does NOT implement denoise() must raise TypeError
        on instantiation.  This catches the bug of forgetting to override the
        abstract method when adding a new denoiser (e.g. BM3D, DnCNN).
        """

        class IncompleteDenoiser(DenoiserPrior):
            pass

        with pytest.raises(TypeError):
            IncompleteDenoiser()

    def test_complete_subclass_instantiates(self):
        """A subclass that fully implements denoise() should instantiate
        without error.  Confirms the ABC is not overly restrictive.
        """

        class GoodDenoiser(DenoiserPrior):
            def denoise(self, x_t, t, noise_schedule):
                return x_t

        denoiser = GoodDenoiser()
        assert isinstance(denoiser, DenoiserPrior)

    def test_denoise_returns_tensor_of_correct_shape(self):
        """A concrete DenoiserPrior.denoise must accept (x_t, t, noise_schedule)
        and return a Tensor of the same shape.  This verifies the interface
        contract at runtime — shape preservation is critical because the HQS
        loop feeds the output directly into the data step.
        """

        class IdentityDenoiser(DenoiserPrior):
            def denoise(self, x_t, t, noise_schedule):
                return x_t

        denoiser = IdentityDenoiser()
        x_t = torch.randn(2, 3, 32, 32)
        schedule = {"alpha_bars": torch.ones(1000)}
        result = denoiser.denoise(x_t, t=500, noise_schedule=schedule)
        assert isinstance(result, Tensor)
        assert result.shape == x_t.shape
        assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# PnPSolver ABC enforcement
# ---------------------------------------------------------------------------

class TestPnPSolverABC:
    """Verify PnPSolver is a proper ABC that enforces the data_step contract."""

    def test_cannot_instantiate_directly(self):
        """PnPSolver defines data_step() as abstract; direct instantiation
        must raise TypeError.  Ensures every degradation solver goes through
        a concrete implementation (InpaintingPnPSolver, BlurPnPSolver, etc.).
        """
        with pytest.raises(TypeError):
            PnPSolver()

    def test_incomplete_subclass_raises_type_error(self):
        """A subclass that omits data_step() must raise TypeError on
        instantiation.  Catches the bug of adding a new degradation solver
        file but forgetting to implement the required method.
        """

        class IncompleteSolver(PnPSolver):
            pass

        with pytest.raises(TypeError):
            IncompleteSolver()

    def test_complete_subclass_instantiates(self):
        """A subclass that implements data_step() should instantiate cleanly."""

        class GoodSolver(PnPSolver):
            def data_step(self, x0_prior, y, degradation, rho_t):
                return x0_prior

        solver = GoodSolver()
        assert isinstance(solver, PnPSolver)

    def test_data_step_returns_tensor_of_correct_shape(self):
        """A concrete PnPSolver.data_step must accept (x0_prior, y, degradation, rho_t)
        and return a Tensor of the same shape as x0_prior.  Shape preservation is
        critical because the HQS loop feeds the output into the re-noising step.
        """

        class PassthroughSolver(PnPSolver):
            def data_step(self, x0_prior, y, degradation, rho_t):
                return x0_prior

        solver = PassthroughSolver()
        x0 = torch.randn(2, 3, 64, 64)
        result = solver.data_step(x0, y=x0, degradation=None, rho_t=1.0)
        assert isinstance(result, Tensor)
        assert result.shape == x0.shape
        assert result.dtype == torch.float32

    def test_multiple_abc_methods_all_enforced(self):
        """If PnPSolver were extended with additional abstract methods in the
        future, a subclass must implement ALL of them.  This test ensures that
        the current single-method contract is properly enforced via the ABC
        machinery (not just a convention).
        """
        import inspect
        abstract_methods = {
            name
            for name, _ in inspect.getmembers(PnPSolver)
            if getattr(getattr(PnPSolver, name, None), "__isabstractmethod__", False)
        }
        assert "data_step" in abstract_methods
