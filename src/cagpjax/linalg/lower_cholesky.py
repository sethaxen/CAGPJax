"""Lower Cholesky decomposition of positive semidefinite operators."""

import cola
import gpjax.lower_cholesky
from cola.ops import LinearOperator

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
        return gpjax.lower_cholesky.lower_cholesky(cola.PSD(A))
    return _lower_cholesky_jittered(A, jitter)


def _lower_cholesky_jittered(A: LinearOperator, jitter: ScalarFloat) -> LinearOperator:
    A_jittered = _add_jitter(A, jitter)
    return gpjax.lower_cholesky.lower_cholesky(cola.PSD(A_jittered))
