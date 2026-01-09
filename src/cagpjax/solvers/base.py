"""Base classes for linear solvers and methods."""

from abc import abstractmethod

from cola.ops import LinearOperator
from flax import nnx
from jaxtyping import Array, Float
from typing_extensions import Generic, TypeVar

from ..typing import ScalarFloat

_LinearSolverState = TypeVar("_LinearSolverState")


class AbstractLinearSolver(nnx.Module, Generic[_LinearSolverState]):
    """
    Base class for linear solvers.

    These solvers are used to exactly or approximately solve the linear
    system $Ax = b$ for $x$, where $A$ is a positive (semi-)definite (PSD)
    linear operator.
    """

    @abstractmethod
    def init(self, A: LinearOperator) -> _LinearSolverState:
        """Construct a solver state.

        Arguments:
            A: Positive (semi-)definite linear operator.

        Returns:
            State of the linear solver, which stores any necessary intermediate values.
        """
        pass

    @abstractmethod
    def solve(
        self, state: _LinearSolverState, b: Float[Array, "N #K"]
    ) -> Float[Array, "N #K"]:
        """Compute a solution to the linear system $Ax = b$.

        Arguments:
            state: State of the linear solver returned by `init`.
            b: Right-hand side of the linear system.
        """
        pass

    @abstractmethod
    def unwhiten(
        self, state: _LinearSolverState, z: Float[Array, "N #K"]
    ) -> Float[Array, "N #K"]:
        """Given an IID standard normal vector $z$, return $x$ with covariance $A$.

        Arguments:
            state: State of the linear solver returned by `init`.
            z: IID standard normal vector.
        """
        pass

    @abstractmethod
    def logdet(self, state: _LinearSolverState) -> ScalarFloat:
        """Compute the logarithm of the (pseudo-)determinant of $A$.

        Arguments:
            state: State of the linear solver returned by `init`.
        """
        pass

    @abstractmethod
    def inv_congruence_transform(
        self, state: _LinearSolverState, B: LinearOperator | Float[Array, "N K"]
    ) -> LinearOperator | Float[Array, "K K"]:
        """Compute the inverse congruence transform $B^T x$ for $x$ in $Ax = B$.

        Arguments:
            state: State of the linear solver returned by `init`.
            B: Linear operator or array to be applied.

        Returns:
            Linear operator or array resulting from the congruence transform.
        """
        pass

    def inv_quad(
        self, state: _LinearSolverState, b: Float[Array, "N #1"]
    ) -> ScalarFloat:
        """Compute the inverse quadratic form $b^T x$, for $x$ in $Ax = b$.

        Arguments:
            state: State of the linear solver returned by `init`.
            b: Right-hand side of the linear system.
        """
        return self.inv_congruence_transform(state, b[:, None]).squeeze()

    @abstractmethod
    def trace_solve(
        self, state: _LinearSolverState, state_other: _LinearSolverState
    ) -> ScalarFloat:
        r"""Compute $\mathrm{trace}(X)$ in $AX=B$ for PSD $B$.

        Arguments:
            state: State of the linear solver returned by applying `init` to the PSD linear operator
                $A$.
            state_other: Another state obtained by applying `init` to the PSD linear operator
                $B$.
        """
        pass
