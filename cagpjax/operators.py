"""Block-sparse linear operators for computation-aware Gaussian processes."""

from typing import Optional

import jax.numpy as jnp
from cola.ops import LinearOperator
from jaxtyping import Array, Float


class BlockDiagonalSparseOperator(LinearOperator):
    """Block-diagonal sparse linear operator.

    This operator represents a block-diagonal matrix with contiguous blocks
    of identical size.

    Args:
        blocks: Array of shape (n_blocks, block_rows, block_cols) containing the
            block values for the contiguous block-diagonal matrix.
        shape: Optional tuple specifying the shape of the linear operator. If None,
            inferred from the blocks assuming perfect tiling. This must be provided if blocks
            overhang the intended size of the operator.

    Example:
        >>> import jax.numpy as jnp
        >>> blocks = jnp.ones((3, 2, 2))    # 3 contiguous 2x2 blocks
        >>> op = BlockDiagonalSparseOperator(blocks)
        >>> x = jnp.ones((6, 3))
        >>> result = op @ x  # Matrix multiplication: (6, 6) @ (6, 3) -> (6, 3)
    """

    def __init__(
        self,
        blocks: Float[Array, "n_blocks block_rows block_cols"],
        shape: Optional[tuple[int, int]] = None,
    ) -> None:
        # Compute the natural shape assuming perfect tiling
        n_blocks, block_rows, block_cols = blocks.shape
        natural_rows = n_blocks * block_rows
        natural_cols = n_blocks * block_cols

        if shape is None:
            shape = (natural_rows, natural_cols)

        super().__init__(blocks.dtype, shape)
        self.blocks = blocks

        # Validate that the shape makes sense
        if self.shape[0] < natural_rows - block_rows:
            raise ValueError("Shape too small for blocks in row dimension")
        if self.shape[1] < natural_cols - block_cols:
            raise ValueError("Shape too small for blocks in column dimension")

    def _matmat(
        self, X: Float[Array, "n_cols n_samples"]
    ) -> Float[Array, "n_rows n_samples"]:
        """Matrix multiplication with the sparse operator."""
        n_rows, n_cols = self.shape

        # Initialize result with appropriate dtype (promotion of blocks and other)
        result_dtype = jnp.result_type(self.dtype, X.dtype)
        result = jnp.zeros((n_rows, X.shape[1]), dtype=result_dtype)

        # Get block dimensions
        n_blocks, block_rows, block_cols = self.blocks.shape

        # Handle perfect blocks first (vectorized)
        n_perfect_blocks = min(n_rows // block_rows, n_cols // block_cols, n_blocks)

        if n_perfect_blocks > 0:
            # Reshape input for vectorized block multiplication
            input_blocks = X[: n_perfect_blocks * block_cols].reshape(
                n_perfect_blocks, block_cols, -1
            )

            # Vectorized block matrix multiplication
            output_blocks = jnp.einsum(
                "bij,bjk->bik", self.blocks[:n_perfect_blocks], input_blocks
            )

            # Reshape and place in result
            output_flat = output_blocks.reshape(n_perfect_blocks * block_rows, -1)
            result = result.at[: n_perfect_blocks * block_rows].set(output_flat)

        # Handle overhang block if present
        if n_perfect_blocks < n_blocks and n_perfect_blocks * block_rows < n_rows:
            block_idx = n_perfect_blocks
            row_start = block_idx * block_rows
            col_start = block_idx * block_cols

            # Determine actual dimensions for overhang
            actual_rows = min(block_rows, n_rows - row_start)
            actual_cols = min(block_cols, n_cols - col_start)

            if actual_cols > 0:
                # Extract the relevant block and input slice
                block_slice = self.blocks[block_idx][:actual_rows, :actual_cols]
                input_slice = X[col_start : col_start + actual_cols]

                # Compute and place overhang result
                overhang_result = block_slice @ input_slice
                result = result.at[row_start : row_start + actual_rows].set(
                    overhang_result
                )

        return result

    def to_dense(self) -> Float[Array, "n_rows n_cols"]:
        """Convert to dense matrix representation.

        Returns:
            Dense matrix representation of the sparse operator with same dtype as blocks.
        """
        n_rows, n_cols = self.shape
        n_blocks, block_rows, block_cols = self.blocks.shape

        # Create dense matrix with same dtype and on same device as blocks
        dense = jnp.zeros((n_rows, n_cols), dtype=self.blocks.dtype)

        for i in range(n_blocks):
            row_start = i * block_rows
            row_end = min(row_start + block_rows, n_rows)
            col_start = i * block_cols
            col_end = min(col_start + block_cols, n_cols)

            if row_start < n_rows and col_start < n_cols:
                actual_rows = row_end - row_start
                actual_cols = col_end - col_start

                block_slice = self.blocks[i][:actual_rows, :actual_cols]
                dense = dense.at[row_start:row_end, col_start:col_end].set(block_slice)

        return dense
