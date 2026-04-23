"""Linear solvers based on Cholesky decomposition."""

import cola
import equinox as eqx
import jax.numpy as jnp
from cola.ops import LinearOperator
from jaxtyping import Array, Float
from typing_extensions import TypeAlias, override

from ..linalg import lower_cholesky
from ..typing import ScalarFloat
from .base import AbstractLinearSolver, LinearOperatorLike

CholeskyState: TypeAlias = LinearOperatorLike


class Cholesky(AbstractLinearSolver[CholeskyState]):
    """
    Solve a linear system using the Cholesky decomposition.

    Due to numerical imprecision, Cholesky factorization may fail even for
    positive-definite $A$. Optionally, a small amount of `jitter` ($\\epsilon$) can
    be added to $A$ to ensure positive-definiteness. Note that the resulting system
    solved is slightly different from the original system.

    Attributes:
        jitter: Small amount of jitter to add to $A$ to ensure positive-definiteness.
    """

    jitter: ScalarFloat | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        if self.jitter is not None and self.jitter < 0:
            raise ValueError("jitter must be non-negative")

    @override
    def init(self, A: LinearOperatorLike) -> CholeskyState:
        return lower_cholesky(A, jitter=self.jitter)

    @override
    def unwhiten(
        self, state: CholeskyState, z: Float[Array, "N #K"]
    ) -> Float[Array, "N #K"]:
        return state @ z

    @override
    def solve(
        self, state: CholeskyState, b: Float[Array, "N #K"]
    ) -> Float[Array, "N #K"]:
        L = state.to_dense()
        y = jnp.linalg.solve(L, b)
        return jnp.linalg.solve(L.T, y)

    @override
    def logdet(self, state: CholeskyState) -> ScalarFloat:
        L = state.to_dense()
        return 2 * jnp.sum(jnp.log(jnp.diag(L)))

    @override
    def inv_congruence_transform(
        self, state: CholeskyState, B: LinearOperatorLike | Float[Array, "K N"]
    ) -> LinearOperatorLike | Float[Array, "K K"]:
        L = state.to_dense()
        B_mat = B.to_dense() if isinstance(B, LinearOperator) else B
        Y = jnp.linalg.solve(L, B_mat)
        result = Y.T @ Y
        return cola.lazify(result) if isinstance(B, LinearOperator) else result

    @override
    def trace_solve(
        self, state: CholeskyState, state_other: CholeskyState
    ) -> ScalarFloat:
        L = state.to_dense()
        X = jnp.linalg.solve(L, state_other.to_dense())
        return jnp.sum(jnp.square(X))
