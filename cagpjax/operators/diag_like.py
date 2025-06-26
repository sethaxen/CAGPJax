import jax.numpy as jnp
from cola.ops import Diagonal, LinearOperator, ScalarMul
from jaxtyping import Array, Float

from ..typing import ScalarFloat


def diag_like(
    operator: LinearOperator, values: ScalarFloat | Float[Array, "N"]
) -> Diagonal | ScalarMul:
    """Create a diagonal operator with the same shape, dtype, and device as the operator.

    Args:
        operator: Linear operator.
        values: Scalar for a scalar matrix or array of diagonal values for a diagonal matrix.

    Returns:
            Diagonal or scalar operator.
    """
    device = operator.device
    dtype = operator.dtype
    if jnp.isscalar(values):
        return ScalarMul(values, operator.shape, dtype=dtype, device=device)
    else:
        return Diagonal(values.astype(dtype)).to(device)
