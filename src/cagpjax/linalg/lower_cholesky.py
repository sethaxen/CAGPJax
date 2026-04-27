"""Lower Cholesky decomposition of positive semidefinite operators."""

from typing import Any, cast

import jax.numpy as jnp
import lineax as lx

from ..interop import lazify, to_lineax
from ..typing import ScalarFloat
from .utils import _add_jitter


def lower_cholesky(A: Any, jitter: ScalarFloat | None = None) -> Any:
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


def _lower_cholesky_jittered(A: Any, jitter: ScalarFloat) -> Any:
    A_jittered = _add_jitter(A, jitter)
    return _lower_cholesky(cast(Any, A_jittered))


def _lower_cholesky(A: Any) -> Any:
    A_lx = to_lineax(A)
    if lx.is_diagonal(A_lx):
        L = lx.DiagonalLinearOperator(jnp.sqrt(lx.diagonal(A_lx)))
    else:
        L = lx.MatrixLinearOperator(jnp.linalg.cholesky(A_lx.as_matrix()))
    if isinstance(A, lx.AbstractLinearOperator):
        return L
    return lazify(L)
