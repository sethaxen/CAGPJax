from typing import Any

import cola.ops
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from ..typing import ScalarFloat


def diag_like(
    operator: Any, values: ScalarFloat | Float[Array, "N"]
) -> lx.DiagonalLinearOperator | cola.ops.Diagonal | cola.ops.ScalarMul:
    """Create a diagonal operator with matching structure and dtype.

    Args:
        operator: Reference operator.
        values: Scalar for a scalar matrix or array of diagonal values for a diagonal matrix.

    Returns:
        Diagonal-like operator matching the backend of `operator`.
    """
    if isinstance(operator, lx.AbstractLinearOperator):
        n = operator.in_size()
        dtype = operator.in_structure().dtype
        if jnp.isscalar(values):
            diagonal = jnp.full((n,), values, dtype=dtype)
        else:
            diagonal = jnp.asarray(values, dtype=dtype)
        return lx.DiagonalLinearOperator(diagonal)

    if not isinstance(operator, cola.ops.LinearOperator):
        raise TypeError(
            "diag_like expects a lineax.AbstractLinearOperator or cola.ops.LinearOperator."
        )

    device = operator.device
    dtype = operator.dtype
    if jnp.isscalar(values):
        return cola.ops.ScalarMul(values, operator.shape, dtype=dtype, device=device)

    values = values.astype(dtype).to_device(device)
    return cola.ops.Diagonal(values)
