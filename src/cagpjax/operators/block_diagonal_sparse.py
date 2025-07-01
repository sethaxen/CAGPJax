"""Block-diagonal sparse linear operator."""

import jax.numpy as jnp
from cola.ops import LinearOperator
from jaxtyping import Array, Float


class BlockDiagonalSparse(LinearOperator):
    """Block-diagonal sparse linear operator.

    This operator represents a block-diagonal matrix structure where the blocks are contiguous, and
    each contains a row vector, so that exactly one value is non-zero in each column.

    Args:
        nz_values: Non-zero values to be distributed across diagonal blocks.
        n_blocks: Number of diagonal blocks in the matrix.

    Examples
    --------
    ```python
    >>> import jax.numpy as jnp
    >>> from cagpjax.operators import BlockDiagonalSparse
    >>>
    >>> # Create a 3x6 block-diagonal matrix with 3 blocks
    >>> nz_values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> op = BlockDiagonalSparse(nz_values, n_blocks=3)
    >>> print(op.shape)
    (3, 6)
    >>>
    >>> # Apply to a vector
    >>> x = jnp.ones(6)
    >>> result = op @ x
    ```
    """

    def __init__(self, nz_values: Float[Array, "N"], n_blocks: int):
        n = nz_values.shape[0]
        block_size = n // n_blocks
        n_used = n_blocks * block_size
        super().__init__(nz_values.dtype, (n_blocks, n))
        self.nz_values = nz_values[:n_used].reshape(n_blocks, block_size)

    def _matmat(self, X: Float[Array, "N #M"]) -> Float[Array, "K #M"]:
        n_blocks, n = self.shape
        n_used = self.nz_values.size
        block_size = n_used // n_blocks

        # block-wise multiplication for used portion
        X_used = X[:n_used, ...].reshape(n_blocks, block_size, -1)
        res = jnp.einsum("ik,ikj->ij", self.nz_values, X_used)

        return res.reshape(-1, *X.shape[1:])
