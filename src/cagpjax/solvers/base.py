"""Base classes for linear solvers and methods."""

from abc import ABC, abstractmethod

from cola.ops import LinearOperator
from flax import nnx
from jaxtyping import Array, Float
from typing_extensions import Self

from ..typing import ScalarFloat


class AbstractLinearSolver(ABC):
    """
    Base class for linear solvers.

    These solvers are used to exactly or approximately solve the linear
    system $Ax = b$ for $x$, where $A$ is a positive (semi-)definite (PSD)
    linear operator.

    Solvers should always be constructed by a `AbstractLinearSolverMethod`.
    """

    @abstractmethod
    def solve(self, b: Float[Array, "N #K"]) -> Float[Array, "N #K"]:
        """Compute a solution to the linear system $Ax = b$.

        Arguments:
            b: Right-hand side of the linear system.
        """
        pass

    @abstractmethod
    def logdet(self) -> ScalarFloat:
        """Compute the logarithm of the (pseudo-)determinant of $A$."""
        pass

    @abstractmethod
    def inv_congruence_transform(
        self, B: LinearOperator | Float[Array, "N K"]
    ) -> LinearOperator | Float[Array, "K K"]:
        """Compute the inverse congruence transform $B^T x$ for $x$ in $Ax = B$.

        Arguments:
            B: Linear operator or array to be applied.

        Returns:
            Linear operator or array resulting from the congruence transform.
        """
        pass

    def inv_quad(self, b: Float[Array, "N #1"]) -> ScalarFloat:
        """Compute the inverse quadratic form $b^T x$, for $x$ in $Ax = b$.

        Arguments:
            b: Right-hand side of the linear system.
        """
        return self.inv_congruence_transform(b[:, None]).squeeze()

    @abstractmethod
    def trace_solve(self, B: Self) -> ScalarFloat:
        r"""Compute $\mathrm{trace}(X)$ in $AX=B$ for PSD $B$.

        Arguments:
            B: An `AbstractLinearSolver` of the same type as `self` representing
                the PSD linear operator $B$.
        """
        pass


class AbstractLinearSolverMethod(nnx.Module):
    """
    Base class for linear solver methods.

    These methods are used to construct `AbstractLinearSolver` instances.
    """

    @abstractmethod
    def __call__(self, A: LinearOperator) -> AbstractLinearSolver:
        """Construct a solver from the positive (semi-)definite linear operator."""
        pass
