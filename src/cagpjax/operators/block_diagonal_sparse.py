"""Block-diagonal sparse linear operator."""

import jax.numpy as jnp
from cola.ops import LinearOperator
from jaxtyping import Array, Float


class BlockDiagonalSparse(LinearOperator):
    """Block-diagonal sparse linear operator.

    This operator represents a block-diagonal matrix structure where the blocks are contiguous, and
    each contains a row vector, so that exactly one value is non-zero in each column.

    Args:
        nz_values: Non-zero values organized as blocks, shape (n_blocks, block_size).
        n: Total number of columns in the matrix. Must be >= n_blocks * block_size.

    Examples
    --------
    ```python
    >>> import jax.numpy as jnp
    >>> from cagpjax.operators import BlockDiagonalSparse
    >>>
    >>> # Create a 3x6 block-diagonal matrix with 3 blocks of size 2
    >>> nz_values = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> op = BlockDiagonalSparse(nz_values, n=6)
    >>> print(op.shape)
    (3, 6)
    >>>
    >>> # Apply to a vector
    >>> x = jnp.ones(6)
    >>> result = op @ x
    ```
    """

    def __init__(self, nz_values: Float[Array, "n_blocks block_size"], n: int):
        n_blocks, block_size = nz_values.shape
        if n < n_blocks * block_size:
            raise ValueError(
                f"n ({n}) must be >= n_blocks * block_size ({n_blocks * block_size})"
            )
        super().__init__(nz_values.dtype, (n_blocks, n))
        self.nz_values = nz_values

    def _matmat(self, X: Float[Array, "N #M"]) -> Float[Array, "K #M"]:
        n_blocks, block_size = self.nz_values.shape
        n_used = n_blocks * block_size

        # block-wise multiplication for used portion
        X_used = X[:n_used, ...].reshape(n_blocks, block_size, -1)
        res = jnp.einsum("ik,ikj->ij", self.nz_values, X_used)

        return res.reshape(-1, *X.shape[1:])
