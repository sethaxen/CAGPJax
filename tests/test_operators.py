"""Tests for block-sparse linear operators."""

import jax
import jax.numpy as jnp
import pytest

from cagpjax.operators import BlockDiagonalSparseOperator


class TestBlockDiagonalSparseOperator:
    """Test cases for BlockDiagonalSparseOperator with contiguous blocks."""

    def test_init_basic(self):
        """Test basic initialization."""
        blocks = jnp.ones((2, 3, 3))
        op = BlockDiagonalSparseOperator(blocks, shape=(6, 6))

        assert op.shape == (6, 6)
        assert op.blocks.shape == (2, 3, 3)

    def test_auto_shape_inference(self):
        """Test automatic shape inference when shape=None."""
        blocks = jnp.ones((3, 2, 2))  # 3 blocks of 2x2
        op = BlockDiagonalSparseOperator(blocks)  # No shape provided

        # Should infer (3*2, 3*2) = (6, 6)
        expected_shape = (6, 6)
        assert op.shape == expected_shape

    def test_shape_validation(self):
        """Test shape validation."""
        blocks = jnp.ones((2, 3, 3))

        # Shape too small should fail
        with pytest.raises(ValueError, match="Shape too small"):
            BlockDiagonalSparseOperator(blocks, shape=(2, 6))

    def test_matmul_perfect_tiling(self):
        """Test matrix multiplication with perfect block tiling."""
        # Create 2 contiguous blocks
        blocks = jnp.array(
            [
                [[2.0, 0.0], [0.0, 2.0]],  # 2*I
                [[3.0, 0.0], [0.0, 3.0]],  # 3*I
            ]
        )
        op = BlockDiagonalSparseOperator(blocks)  # Natural shape (4, 4)

        x = jnp.ones((4, 2))
        result = op @ x

        expected = jnp.array(
            [
                [2.0, 2.0],  # First block: 2*[1,1]
                [2.0, 2.0],
                [3.0, 3.0],  # Second block: 3*[1,1]
                [3.0, 3.0],
            ]
        )

        assert jnp.allclose(result, expected)

    def test_matmul_with_overhang(self):
        """Test matrix multiplication with overhang in the final block."""
        # 2 blocks of size 3x3, but matrix is 5x5 (overhang in last block)
        blocks = jnp.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # I
                [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],  # 2*I
            ]
        )
        op = BlockDiagonalSparseOperator(blocks, shape=(5, 5))

        x = jnp.ones((5, 1))
        result = op @ x

        # First block (rows 0-2), second block (rows 3-4, partial)
        expected = jnp.array([[1.0], [1.0], [1.0], [2.0], [2.0]])

        assert jnp.allclose(result, expected)

    def test_matmul_rectangular_blocks(self):
        """Test matrix multiplication with rectangular blocks."""
        # 2 blocks of size 2x3
        blocks = jnp.array(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]]
        )
        op = BlockDiagonalSparseOperator(blocks)  # Natural shape (4, 6)

        x = jnp.ones((6, 1))
        result = op @ x

        # First block: [1+2+3, 4+5+6] = [6, 15]
        # Second block: [2+3+4, 5+6+7] = [9, 18]
        expected = jnp.array([[6.0], [15.0], [9.0], [18.0]])

        assert jnp.allclose(result, expected)

    def test_to_dense(self):
        """Test conversion to dense matrix."""
        blocks = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        op = BlockDiagonalSparseOperator(blocks)  # Natural shape (4, 4)

        dense = op.to_dense()

        expected = jnp.array(
            [
                [1.0, 2.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 6.0],
                [0.0, 0.0, 7.0, 8.0],
            ]
        )

        assert jnp.allclose(dense, expected)

    def test_to_dense_with_overhang(self):
        """Test dense conversion with overhang."""
        blocks = jnp.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        op = BlockDiagonalSparseOperator(blocks, shape=(3, 3))  # Overhang

        dense = op.to_dense()

        expected = jnp.array(
            [
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 0.0],
                [0.0, 0.0, 5.0],  # Only partial second block
            ]
        )

        assert jnp.allclose(dense, expected)

    def test_jit_compatibility(self):
        """Test that operations are JIT-compatible."""
        blocks = jnp.ones((2, 2, 2))  # Each block is all ones
        op = BlockDiagonalSparseOperator(blocks)  # Natural shape (4, 4)

        @jax.jit
        def matmul_op(x):
            return op @ x

        x = jnp.ones((4, 3))
        result = matmul_op(x)

        assert result.shape == (4, 3)
        # Each block is [[1,1],[1,1]] @ [1,1] = [2,2]
        assert jnp.allclose(result, 2 * jnp.ones((4, 3)))

    def test_empty_blocks(self):
        """Test with zero blocks."""
        blocks = jnp.ones((0, 2, 2))
        op = BlockDiagonalSparseOperator(blocks, shape=(4, 4))

        x = jnp.ones((4, 2))
        result = op @ x

        # Should return zeros
        assert jnp.allclose(result, jnp.zeros((4, 2)))

    def test_single_block(self):
        """Test with a single block."""
        blocks = jnp.array([[[2.0, 3.0], [4.0, 5.0]]])
        op = BlockDiagonalSparseOperator(blocks, shape=(4, 4))

        x = jnp.ones((4, 1))
        result = op @ x

        # Only first block active, rest zeros
        expected = jnp.array([[5.0], [9.0], [0.0], [0.0]])
        assert jnp.allclose(result, expected)

    def test_dimension_mismatch(self):
        """Test dimension mismatch error."""
        blocks = jnp.ones((1, 2, 2))
        op = BlockDiagonalSparseOperator(blocks, shape=(2, 2))

        x = jnp.ones((3, 1))  # Wrong input dimension

        with pytest.raises(Exception, match="dimension mismatch"):
            op @ x  # type: ignore[unused-expression]

    def test_vectorized_performance(self):
        """Test that vectorized operations work correctly for multiple perfect blocks."""
        # Create 3 identical blocks for vectorization
        blocks = jnp.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 2.0], [3.0, 4.0]],
            ]
        )
        op = BlockDiagonalSparseOperator(blocks)  # Natural shape (6, 6)

        x = jnp.ones((6, 2))
        result = op @ x

        # Each block produces [3, 7] when multiplied by [1, 1]
        expected = jnp.tile(jnp.array([[3.0, 3.0], [7.0, 7.0]]), (3, 1))

        assert jnp.allclose(result, expected)

    def test_rectangular_matrix_with_overhang(self):
        """Test rectangular matrix with both row and column overhang."""
        blocks = jnp.array(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]]
        )
        # Natural would be (4, 6), but set to (3, 5) for overhang
        op = BlockDiagonalSparseOperator(blocks, shape=(3, 5))

        x = jnp.ones((5, 1))
        result = op @ x

        # First block: full [6, 15], second block: partial [5] (only first row, first 2 cols)
        expected = jnp.array([[6.0], [15.0], [5.0]])

        assert jnp.allclose(result, expected)

    def test_dtype_consistency(self):
        """Test that output dtypes are consistent with input dtypes."""
        # Test with float32 blocks
        blocks_f32 = jnp.ones((2, 2, 2), dtype=jnp.float32)
        op = BlockDiagonalSparseOperator(blocks_f32)

        # Test matmul dtype consistency
        x_f32 = jnp.ones((4, 2), dtype=jnp.float32)
        result = op @ x_f32
        assert result.dtype == jnp.float32

        # Test mixed dtypes with integer (should promote to float)
        x_int = jnp.ones((4, 2), dtype=jnp.int32)
        result_mixed = op @ x_int
        assert result_mixed.dtype == jnp.float32  # Should promote int32 to float32

        # Test to_dense dtype consistency
        dense = op.to_dense()
        assert dense.dtype == blocks_f32.dtype

        # Test with different float dtypes if available
        # Note: JAX may downcast to float32 depending on configuration
        blocks_f16 = jnp.ones((2, 2, 2), dtype=jnp.float16)
        op_f16 = BlockDiagonalSparseOperator(blocks_f16)
        dense_f16 = op_f16.to_dense()
        assert dense_f16.dtype == blocks_f16.dtype
