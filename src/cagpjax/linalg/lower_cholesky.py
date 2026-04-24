"""Lower Cholesky decomposition of positive semidefinite operators."""

import cola
import jax.numpy as jnp
from cola.ops import Diagonal, Identity, LinearOperator, ScalarMul, Triangular

from ..typing import ScalarFloat
from .utils import _add_jitter


def lower_cholesky(
    A: LinearOperator, jitter: ScalarFloat | None = None
) -> LinearOperator:
    """Lower Cholesky decomposition of a positive semidefinite operator.

    Args:
        A: Positive semidefinite operator
        jitter: Positive jitter to add to the operator.

    Returns:
        Lower Cholesky factor of A.
    """
    if jitter is None:
        return _lower_cholesky(cola.PSD(A))
    return _lower_cholesky_jittered(A, jitter)


def _lower_cholesky_jittered(A: LinearOperator, jitter: ScalarFloat) -> LinearOperator:
    A_jittered = _add_jitter(A, jitter)
    return _lower_cholesky(cola.PSD(A_jittered))


def _lower_cholesky(A: LinearOperator) -> LinearOperator:
    if isinstance(A, Diagonal):
        return Diagonal(jnp.sqrt(A.diag))
    if isinstance(A, ScalarMul):
        return ScalarMul(jnp.sqrt(A.c), A.shape, A.dtype, A.device)
    if isinstance(A, Identity):
        return A
    return Triangular(jnp.linalg.cholesky(A.to_dense()), lower=True)
