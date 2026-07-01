"""Lower Cholesky decomposition of positive semidefinite operators."""

from typing import Any

import jax.numpy as jnp
import lineax as lx

from ..interop import to_lineax
from ..typing import ScalarFloat
from .utils import _add_jitter


def lower_cholesky(
    A: Any, jitter: ScalarFloat | None = None
) -> lx.AbstractLinearOperator:
    """Lower Cholesky decomposition of a positive semidefinite operator.

    Args:
        A: Positive semidefinite operator
        jitter: Positive jitter to add to the operator.

    Returns:
        Lower Cholesky factor of A.
    """
    if jitter is None:
        return _lower_cholesky(A)
    return _lower_cholesky_jittered(A, jitter)


def _lower_cholesky_jittered(A: Any, jitter: ScalarFloat) -> lx.AbstractLinearOperator:
    A_jittered = _add_jitter(A, jitter)
    return _lower_cholesky(A_jittered)


def _lower_cholesky(A: Any) -> lx.AbstractLinearOperator:
    A_lx = to_lineax(A)
    if lx.is_diagonal(A_lx):
        return lx.DiagonalLinearOperator(jnp.sqrt(lx.diagonal(A_lx)))
    return lx.MatrixLinearOperator(jnp.linalg.cholesky(A_lx.as_matrix()))
