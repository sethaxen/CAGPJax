"""Block-diagonal sparse linear operator."""

import math

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
        super().__init__(nz_values.dtype, (n_blocks, n))
        self.nz_values = nz_values

    def _matmat(self, X: Float[Array, "N #M"]) -> Float[Array, "K #M"]:
        # figure out size of main blocks
        n_blocks, n = self.shape
        block_size = math.ceil(n / n_blocks)
        n_blocks_main = n // block_size
        n_main = n_blocks_main * block_size

        # block-wise multiplication
        blocks_main = self.nz_values[:n_main].reshape(n_blocks_main, block_size)
        X_main = X[:n_main, ...].reshape(n_blocks_main, block_size, -1)
        res = jnp.einsum("ik,ikj->ij", blocks_main, X_main)

        # handle overhang if any
        if n > n_main:
            n_overhang = n - n_main
            X_overhang = X[n_main:, ...].reshape(n_overhang, -1)
            block_overhang = self.nz_values[n_blocks_main * block_size :]
            res_overhang = block_overhang[None, :] @ X_overhang
            res = jnp.concatenate([res, res_overhang], axis=0)

        return res.reshape(-1, *X.shape[1:])
