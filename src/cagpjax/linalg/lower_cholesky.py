"""Lower Cholesky decomposition of positive semidefinite operators."""

from typing import Any

import cola
import jax.numpy as jnp
from cola.ops import Diagonal, Identity, LinearOperator, ScalarMul, Triangular
from typing_extensions import overload

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


@overload
def _lower_cholesky(A: LinearOperator) -> Triangular:
    return Triangular(jnp.linalg.cholesky(A.to_dense()), lower=True)


@overload
def _lower_cholesky(A: Diagonal) -> Diagonal:  # pyright: ignore[reportOverlappingOverload]
    return Diagonal(jnp.sqrt(A.diag))


@overload
def _lower_cholesky(A: ScalarMul) -> ScalarMul:  # pyright: ignore[reportOverlappingOverload]
    return ScalarMul(jnp.sqrt(A.c), A.shape, A.dtype, A.device)


@overload
def _lower_cholesky(A: Identity) -> Identity:  # pyright: ignore[reportOverlappingOverload]
    return A


@cola.dispatch
def _lower_cholesky(A: LinearOperator) -> Any:
    pass
