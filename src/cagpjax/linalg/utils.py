"""Linear algebra utilities."""

import cola
import jax.numpy as jnp
import lineax as lx
from cola.ops import Diagonal, Identity, LinearOperator, ScalarMul
from jaxtyping import Array, Float

from ..interop import lazify, to_lineax
from ..operators import diag_like
from ..typing import ScalarFloat


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
        return Diagonal(cola.linalg.diag(A) + jitter)
    if isinstance(A, Identity):
        if jnp.isscalar(jitter):
            return ScalarMul(1.0 + jitter, A.shape, A.dtype, A.device)
        return Diagonal(cola.linalg.diag(A) + jitter)
    if isinstance(A, Diagonal):
        return Diagonal(cola.linalg.diag(A) + jitter)
    if isinstance(A, lx.AbstractLinearOperator):
        return A + diag_like(A, jitter)
    return lazify(to_lineax(A) + diag_like(A, jitter))
