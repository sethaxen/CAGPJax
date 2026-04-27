"""Linear algebra utilities."""

import jax.numpy as jnp
import lineax as lx
from cola.ops import Diagonal, Identity, LinearOperator, ScalarMul
from jaxtyping import Array, Float

from ..interop import lazify, to_lineax
from ..operators import diag_like
from ..typing import ScalarFloat


def _structured_cola_diagonal(
    A: Diagonal | Identity | ScalarMul,
) -> Float[Array, " N"]:
    """Diagonal of a structured cola operator without ``cola.linalg.diag``."""
    if isinstance(A, Diagonal):
        return A.diag
    if isinstance(A, Identity):
        return jnp.ones((A.shape[0],), dtype=A.dtype)
    return jnp.full((A.shape[0],), A.c, dtype=A.dtype)


def _add_jitter(
    A: LinearOperator | lx.AbstractLinearOperator,
    jitter: ScalarFloat | Float[Array, "N"],
) -> LinearOperator | lx.AbstractLinearOperator:
    """Add scalar/vector jitter to an operator diagonal."""
    if isinstance(A, lx.AbstractLinearOperator) and lx.is_diagonal(A):
        diagonal = lx.diagonal(A)
        return lx.DiagonalLinearOperator(diagonal + jitter)
    if isinstance(A, ScalarMul):
        if jnp.isscalar(jitter):
            return ScalarMul(A.c + jitter, A.shape, A.dtype, A.device)
        return Diagonal(_structured_cola_diagonal(A) + jitter)
    if isinstance(A, Identity):
        if jnp.isscalar(jitter):
            return ScalarMul(1.0 + jitter, A.shape, A.dtype, A.device)
        return Diagonal(_structured_cola_diagonal(A) + jitter)
    if isinstance(A, Diagonal):
        return Diagonal(_structured_cola_diagonal(A) + jitter)
    if isinstance(A, lx.AbstractLinearOperator):
        return A + diag_like(A, jitter)
    return lazify(to_lineax(A) + diag_like(A, jitter))
