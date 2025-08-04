"""Block-diagonal sparse linear operator."""

import jax.numpy as jnp
from cola.ops import LinearOperator
from jaxtyping import Array, Float


class BlockDiagonalSparse(LinearOperator):
    """Block-diagonal sparse linear operator.

    This operator represents a block-diagonal matrix structure where the blocks are contiguous, and
    each contains a column vector, so that exactly one value is non-zero in each row.

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
    (6, 3)
    >>>
    >>> # Apply to identity matrices
    >>> op @ jnp.eye(3)
    Array([[1., 0., 0.],
           [2., 0., 0.],
           [0., 3., 0.],
           [0., 4., 0.],
           [0., 0., 5.],
           [0., 0., 6.]], dtype=float32)
    ```
    """

    def __init__(self, nz_values: Float[Array, "N"], n_blocks: int):
        n = nz_values.shape[0]
        super().__init__(nz_values.dtype, (n, n_blocks))
        self.nz_values = nz_values

    def _matmat(self, X: Float[Array, "K M"]) -> Float[Array, "N M"]:
        n, n_blocks = self.shape
        block_size = n // n_blocks
        n_blocks_main = n_blocks if n % n_blocks == 0 else n_blocks - 1
        n_main = n_blocks_main * block_size
        m = X.shape[1]

        # block-wise multiplication for main blocks
        if n_blocks_main > 0:
            blocks_main = self.nz_values[:n_main].reshape(n_blocks_main, block_size)
            X_main = X[:n_blocks_main, :]
            res_main = (blocks_main[..., None] * X_main[:, None, :]).reshape(n_main, m)
        else:
            res_main = jnp.empty((0, m), dtype=X.dtype)

        # handle overhang if any
        if n > n_main:
            n_overhang = n - n_main
            X_overhang = X[n_blocks_main, :]
            block_overhang = self.nz_values[n_main:]
            res_overhang = jnp.outer(block_overhang, X_overhang).reshape(n_overhang, m)
            res = jnp.concatenate([res_main, res_overhang], axis=0)
        else:
            res = res_main

        return res

    def _rmatmat(self, X: Float[Array, "M N"]) -> Float[Array, "M K"]:
        # figure out size of main blocks
        n, n_blocks = self.shape
        block_size = n // n_blocks
        n_blocks_main = n_blocks if n % n_blocks == 0 else n_blocks - 1
        n_main = n_blocks_main * block_size
        m = X.shape[0]

        # block-wise multiplication for main blocks
        if n_blocks_main > 0:
            blocks_main = self.nz_values[:n_main].reshape(n_blocks_main, block_size)
            X_main = X[:, :n_main].reshape(m, n_blocks_main, block_size)
            res_main = jnp.einsum("ik,jik->ji", blocks_main, X_main)
        else:
            res_main = jnp.empty((m, 0), dtype=X.dtype)

        # handle overhang if any
        if n > n_main:
            n_overhang = n - n_main
            X_overhang = X[:, n_main:].reshape(m, n_overhang)
            block_overhang = self.nz_values[n_main:]
            res_overhang = (X_overhang @ block_overhang)[:, None]
            res = jnp.concatenate([res_main, res_overhang], axis=1)
        else:
            res = res_main

        return res
