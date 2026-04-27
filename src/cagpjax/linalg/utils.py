"""Linear algebra utilities."""

from typing import Any

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from ..interop import lazify, to_lineax
from ..operators import diag_like
from ..typing import ScalarFloat


def _add_jitter(
    A: Any,
    jitter: ScalarFloat | Float[Array, "N"],
) -> Any:
    """Add scalar/vector jitter to an operator diagonal."""
    A_lx = to_lineax(A)
    if lx.is_diagonal(A_lx):
        dtype = A_lx.in_structure().dtype
        diagonal = lx.diagonal(A_lx).astype(dtype)
        jitter_value = jnp.asarray(jitter, dtype=dtype)
        result = lx.DiagonalLinearOperator(diagonal + jitter_value)
        if isinstance(A, lx.AbstractLinearOperator):
            return result
        return lazify(result)
    result = A_lx + diag_like(A_lx, jitter)
    if isinstance(A, lx.AbstractLinearOperator):
        return result
    return lazify(result)
