"""Lower Cholesky decomposition of positive semidefinite operators."""

from typing import Any

import cola
import gpjax.lower_cholesky
from cola.ops import Diagonal, LinearOperator
from typing_extensions import overload

from ..operators import diag_like
from ..typing import ScalarFloat


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


# fallback implementation
@overload
def _add_jitter(A: LinearOperator, jitter: ScalarFloat) -> LinearOperator:
    return A + diag_like(A, jitter)


@overload
def _add_jitter(A: Diagonal, jitter: ScalarFloat) -> Diagonal:  # pyright: ignore[reportOverlappingOverload]
    return Diagonal(A.diag + jitter)


@cola.dispatch
def _add_jitter(A: Any, jitter: ScalarFloat) -> Any:
    pass
