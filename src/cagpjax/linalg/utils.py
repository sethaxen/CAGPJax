"""Linear algebra utilities."""

from typing import Any

import cola
from cola.ops import Diagonal, Identity, LinearOperator, ScalarMul
from jaxtyping import Array, Float
from typing_extensions import overload

from ..operators import diag_like
from ..typing import ScalarFloat


# fallback implementation
@overload
def _add_jitter(
    A: LinearOperator, jitter: ScalarFloat | Float[Array, "N"]
) -> LinearOperator:
    return A + diag_like(A, jitter)


@overload
def _add_jitter(A: ScalarMul, jitter: ScalarFloat) -> ScalarMul:  # pyright: ignore[reportOverlappingOverload]
    return ScalarMul(A.c + jitter, A.shape, A.dtype, A.device)


@overload
def _add_jitter(A: Identity, jitter: ScalarFloat) -> ScalarMul:  # pyright: ignore[reportOverlappingOverload]
    return ScalarMul(1.0 + jitter, A.shape, A.dtype, A.device)


@overload
def _add_jitter(  # pyright: ignore[reportOverlappingOverload]
    A: ScalarMul | Diagonal | Identity, jitter: ScalarFloat | Float[Array, "N"]
) -> Diagonal:
    return Diagonal(cola.linalg.diag(A) + jitter)


@cola.dispatch
def _add_jitter(A: Any, jitter: Any) -> Any:
    pass
