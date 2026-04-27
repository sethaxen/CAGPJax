from typing import Any

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from ..interop import to_lineax
from ..typing import ScalarFloat


def diag_like(
    operator: Any, values: ScalarFloat | Float[Array, "N"]
) -> lx.DiagonalLinearOperator:
    """Create a diagonal Lineax operator with matching structure and dtype.

    Args:
        operator: Reference operator (Lineax operator or array via ``to_lineax``).
        values: Scalar for a scalar matrix or array of diagonal values for a diagonal matrix.

    Returns:
        ``DiagonalLinearOperator`` with shape and dtype consistent with ``operator``.
    """
    op = to_lineax(operator)
    n = op.in_size()
    dtype = op.in_structure().dtype
    if jnp.isscalar(values):
        diagonal = jnp.full((n,), values, dtype=dtype)
    else:
        diagonal = jnp.asarray(values, dtype=dtype)
    return lx.DiagonalLinearOperator(diagonal)
