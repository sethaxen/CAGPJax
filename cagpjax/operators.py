"""Block-sparse linear operators for computation-aware Gaussian processes."""

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float


class BlockDiagonalSparseOperator:
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
        self.blocks = blocks

        # Compute the natural shape assuming perfect tiling
        n_blocks, block_rows, block_cols = blocks.shape
        natural_rows = n_blocks * block_rows
        natural_cols = n_blocks * block_cols

        if shape is None:
            shape = (natural_rows, natural_cols)

        self._shape = shape

        # Validate that the shape makes sense
        if shape[0] < natural_rows - block_rows:
            raise ValueError("Shape too small for blocks in row dimension")
        if shape[1] < natural_cols - block_cols:
            raise ValueError("Shape too small for blocks in column dimension")

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the represented matrix."""
        return self._shape

    def __matmul__(
        self, other: Float[Array, "n_cols n_samples"]
    ) -> Float[Array, "n_rows n_samples"]:
        """Matrix multiplication with the sparse operator.

        Optimized implementation leveraging the contiguous block structure for
        maximum performance with JAX.

        Args:
            other: Array to multiply with, shape (n_cols, n_samples).

        Returns:
            Result of matrix multiplication, shape (n_rows, n_samples).
        """
        n_rows, n_cols = self.shape

        if other.shape[0] != n_cols:
            raise ValueError(f"Input dimension mismatch: {other.shape[0]} != {n_cols}")

        # Initialize result with appropriate dtype (promotion of blocks and other)
        result_dtype = jnp.result_type(self.blocks.dtype, other.dtype)
        result = jnp.zeros((n_rows, other.shape[1]), dtype=result_dtype)

        # Get block dimensions
        n_blocks, block_rows, block_cols = self.blocks.shape

        # Handle perfect blocks first (vectorized)
        n_perfect_blocks = min(n_rows // block_rows, n_cols // block_cols, n_blocks)

        if n_perfect_blocks > 0:
            # Reshape input for vectorized block multiplication
            input_blocks = other[: n_perfect_blocks * block_cols].reshape(
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
                input_slice = other[col_start : col_start + actual_cols]

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

    def transpose(self) -> "BlockDiagonalSparseOperator":
        """Transpose of the sparse operator.

        Returns:
            Transposed sparse operator with swapped block dimensions and shape.
        """
        # Transpose each block
        transposed_blocks = jnp.transpose(self.blocks, (0, 2, 1))

        # Swap shape dimensions
        n_rows, n_cols = self.shape
        transposed_shape = (n_cols, n_rows)

        return BlockDiagonalSparseOperator(transposed_blocks, transposed_shape)
