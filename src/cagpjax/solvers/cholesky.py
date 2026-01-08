"""Linear solvers based on Cholesky decomposition."""

import cola
import jax.numpy as jnp
from cola.ops import LinearOperator
from jaxtyping import Array, Float
from typing_extensions import Self, override

from ..linalg import lower_cholesky
from ..typing import ScalarFloat
from .base import AbstractLinearSolver, AbstractLinearSolverState


class Cholesky(AbstractLinearSolver):
    """
    Solve a linear system using the Cholesky decomposition.

    Due to numerical imprecision, Cholesky factorization may fail even for
    positive-definite $A$. Optionally, a small amount of `jitter` ($\\epsilon$) can
    be added to $A$ to ensure positive-definiteness. Note that the resulting system
    solved is slightly different from the original system.

    Attributes:
        jitter: Small amount of jitter to add to $A$ to ensure positive-definiteness.
    """

    jitter: ScalarFloat | None

    def __init__(self, jitter: ScalarFloat | None = None):
        self.jitter = jitter

    @override
    def __call__(self, A: LinearOperator) -> AbstractLinearSolverState:
        return CholeskySolver(A, jitter=self.jitter)


class CholeskySolver(AbstractLinearSolverState):
    """
    Solve a linear system by computing the Cholesky decomposition.
    """

    lchol: LinearOperator

    def __init__(self, A: LinearOperator, jitter: ScalarFloat | None = None):
        self.lchol = lower_cholesky(A, jitter)

    @override
    def unwhiten(self, z: Float[Array, "N #K"]) -> Float[Array, "N #K"]:
        return self.lchol @ z

    @override
    def solve(self, b: Float[Array, "N #K"]) -> Float[Array, "N #K"]:
        Linv = cola.linalg.inv(self.lchol)
        return Linv.T @ (Linv @ b)

    @override
    def logdet(self) -> ScalarFloat:
        return 2 * jnp.sum(jnp.log(cola.linalg.diag(self.lchol)))

    @override
    def inv_congruence_transform(
        self, B: LinearOperator | Float[Array, "K N"]
    ) -> LinearOperator | Float[Array, "K K"]:
        Linv = cola.linalg.inv(self.lchol)
        right_term = Linv @ B
        return right_term.T @ right_term

    @override
    def trace_solve(self, B: Self) -> ScalarFloat:
        L = cola.linalg.inv(self.lchol) @ B.lchol.to_dense()
        return jnp.sum(jnp.square(L))
