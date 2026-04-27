"""Linear solvers based on Cholesky decomposition."""

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float
from typing_extensions import TypeAlias, override

from ..interop import lazify
from ..linalg import lower_cholesky
from ..typing import ScalarFloat
from .base import AbstractLinearSolver, LinearOperatorLike, SupportsDenseOperator

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

    @staticmethod
    def _to_dense_matrix(A):
        if isinstance(A, lx.AbstractLinearOperator):
            return A.as_matrix()
        if hasattr(A, "to_dense"):
            return jnp.asarray(A.to_dense())
        if hasattr(A, "as_matrix"):
            return jnp.asarray(A.as_matrix())
        return jnp.asarray(A)

    @override
    def init(self, A: LinearOperatorLike) -> CholeskyState:
        return lower_cholesky(A, jitter=self.jitter)

    @override
    def unwhiten(
        self, state: CholeskyState, z: Float[Array, "N #K"]
    ) -> Float[Array, "N #K"]:
        return self._to_dense_matrix(state) @ z

    @override
    def solve(
        self, state: CholeskyState, b: Float[Array, "N #K"]
    ) -> Float[Array, "N #K"]:
        L = self._to_dense_matrix(state)
        y = jnp.linalg.solve(L, b)
        return jnp.linalg.solve(L.T, y)

    @override
    def logdet(self, state: CholeskyState) -> ScalarFloat:
        L = self._to_dense_matrix(state)
        return 2 * jnp.sum(jnp.log(jnp.diag(L)))

    @override
    def inv_congruence_transform(
        self, state: CholeskyState, B: LinearOperatorLike | Float[Array, "K N"]
    ) -> LinearOperatorLike | Float[Array, "K K"]:
        L = self._to_dense_matrix(state)
        B_is_operator = (
            isinstance(B, lx.AbstractLinearOperator)
            or isinstance(B, SupportsDenseOperator)
            or hasattr(B, "as_matrix")
        )
        B_mat = self._to_dense_matrix(B) if B_is_operator else B
        Y = jnp.linalg.solve(L, B_mat)
        result = Y.T @ Y
        return lazify(result) if B_is_operator else result

    @override
    def trace_solve(
        self, state: CholeskyState, state_other: CholeskyState
    ) -> ScalarFloat:
        L = self._to_dense_matrix(state)
        X = jnp.linalg.solve(L, self._to_dense_matrix(state_other))
        return jnp.sum(jnp.square(X))
