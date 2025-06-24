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
        nnz_values: Non-zero values to be distributed across diagonal blocks.
        n_blocks: Number of diagonal blocks in the matrix.

    Example:
        >>> import jax.numpy as jnp
        >>> from cagpjax.operators import BlockDiagonalSparse
        >>>
        >>> # Create a 3x6 block-diagonal matrix with 3 blocks
        >>> nz_values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> op = BlockDiagonalSparse(nz_values, n_blocks=3)
        >>> print(op.shape)  # (3, 6)
        >>>
        >>> # Apply to a vector
        >>> x = jnp.ones(6)
        >>> result = op @ x
    """

    def __init__(self, nnz_values: Float[Array, "N"], n_blocks: int):
        n = nnz_values.shape[0]
        super().__init__(nnz_values.dtype, (n_blocks, n))
        block_size: int = math.ceil(n / n_blocks)
        n_blocks_main = n // block_size
        self.blocks_main = nnz_values[: n_blocks_main * block_size].reshape(
            n_blocks_main, block_size
        )
        self.block_overhang = nnz_values[n_blocks_main * block_size :]
        self.n_blocks = n_blocks

    def _matmat(self, X: Float[Array, "N #M"]) -> Float[Array, "K #M"]:
        n_main = self.blocks_main.size
        X_main = X[:n_main, ...].reshape(*self.blocks_main.shape, -1)
        res = jnp.einsum("ik,ikj->ij", self.blocks_main, X_main)

        n_overhang = self.shape[1] - n_main
        if n_overhang > 0:
            X_overhang = X[n_main:, ...].reshape(n_overhang, -1)
            res_overhang = self.block_overhang[None, :] @ X_overhang
            res = jnp.concatenate([res, res_overhang], axis=0)

        return res.reshape(-1, *X.shape[1:])
