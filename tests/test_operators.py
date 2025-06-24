"""Tests for custom linear operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cola.ops import LinearOperator

from cagpjax.operators import BlockDiagonalSparse

jax.config.update("jax_enable_x64", True)


def _test_mul_consistency(op: LinearOperator):
    """Test that the linear operator is consistent with multiplication by identity."""
    mat = op.to_dense()
    np.testing.assert_allclose(jnp.eye(op.shape[0]) @ op, mat)
    np.testing.assert_allclose(op @ jnp.eye(op.shape[1]), mat)


def _test_transpose_consistency(op: LinearOperator):
    """Test that the transpose is consistent."""
    op_transpose = op.T
    assert op_transpose.shape == op.shape[::-1]
    np.testing.assert_allclose(op_transpose.to_dense(), op.to_dense().T)


def _test_dtype_consistency(op: LinearOperator):
    """Test that the linear operator has the correct dtype."""
    assert op.dtype == op.to_dense().dtype
    x = jnp.ones(op.shape[1], dtype=op.dtype)
    y = jnp.ones(op.shape[0], dtype=op.dtype)
    assert (op @ x).dtype == op.dtype
    assert (op.T @ y).dtype == op.dtype


def _test_linear_operator_consistency(op: LinearOperator):
    """Test that the linear operator is self-consistent."""
    _test_mul_consistency(op)
    _test_transpose_consistency(op)
    _test_dtype_consistency(op)


class TestBlockDiagonalSparse:
    """Test cases for BlockDiagonalSparse."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("shape", [(2, 5), (4, 8)])
    def test_basic(self, shape, dtype, key=jax.random.key(42)):
        """Test initialization and basic properties."""
        n_blocks, n_nz_values = shape
        nz_values = jax.random.normal(key, (n_nz_values,), dtype=dtype)
        op = BlockDiagonalSparse(nz_values, n_blocks)
        assert op.shape == shape
        assert op.dtype == dtype
        _test_linear_operator_consistency(op)

    @pytest.mark.parametrize("shape", [(2, 5), (5, 10)])
    def test_grad(self, shape, dtype=jnp.float64, key=jax.random.key(42)):
        """Test gradient containing the linear operator."""
        n_blocks, n_nz_values = shape
        key, subkey = jax.random.split(key)
        nz_values = jax.random.normal(subkey, (n_nz_values,), dtype=dtype)
        op = BlockDiagonalSparse(nz_values, n_blocks)

        f = lambda x: jnp.prod(jnp.sin(op @ x))
        x = jax.random.normal(key, (n_nz_values,), dtype=dtype)
        jax.test_util.check_grads(f, (x,), order=1)
