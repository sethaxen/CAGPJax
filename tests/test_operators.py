"""Tests for custom linear operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cola.ops import Dense, Diagonal, LinearOperator, ScalarMul

from cagpjax.operators import BlockDiagonalSparse
from cagpjax.operators.diag_like import diag_like

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

    @pytest.fixture(
        params=[
            pytest.param((5, 2), id="has_overhang"),
            pytest.param((9, 4), id="has_overhang"),
            pytest.param((6, 3), id="no_overhang"),
        ]
    )
    def shape(self, request):
        return request.param

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_basic(self, shape, dtype, key=jax.random.key(42)):
        """Test initialization and basic properties."""
        n_nz_values, n_blocks = shape
        nz_values = jax.random.normal(key, (n_nz_values,), dtype=dtype)
        op = BlockDiagonalSparse(nz_values, n_blocks)
        assert op.shape == shape
        assert op.dtype == dtype
        _test_linear_operator_consistency(op)

    def test_grad(self, shape, dtype=jnp.float64, key=jax.random.key(42)):
        """Test gradient containing the linear operator."""
        n_nz_values, n_blocks = shape
        key, subkey = jax.random.split(key)
        nz_values = jax.random.normal(subkey, (n_nz_values,), dtype=dtype)
        op = BlockDiagonalSparse(nz_values, n_blocks)

        f1 = lambda x: jnp.prod(jnp.sin(op @ x))
        x1 = jax.random.normal(key, (n_blocks,), dtype=dtype)
        jax.test_util.check_grads(f1, (x1,), order=1)

        f2 = lambda x: jnp.prod(jnp.sin(op.T @ x))
        x2 = jax.random.normal(key, (n_nz_values,), dtype=dtype)
        jax.test_util.check_grads(f2, (x2,), order=1)


class TestDiagLike:
    """Test cases for ``diag_like``."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize(
        "vals_n", [(2.5, 5), (jnp.array(3.3), 6), (jnp.arange(1.0, 6.0) ** 2, None)]
    )
    def test_scalar_values(self, vals_n, dtype):
        """Test ``diag_like``."""
        vals, n = vals_n
        if n is None:
            n = len(vals)
        ref_op = Dense(jnp.ones((n, n), dtype=dtype))
        diag_op = diag_like(ref_op, vals)
        assert diag_op.shape == ref_op.shape
        assert diag_op.dtype == ref_op.dtype
        assert diag_op.device == ref_op.device
        _test_linear_operator_consistency(diag_op)
        if isinstance(vals, jnp.ndarray) and vals.ndim == 1:
            assert isinstance(diag_op, Diagonal)
            np.testing.assert_allclose(diag_op.to_dense(), jnp.diag(vals))
        else:
            assert isinstance(diag_op, ScalarMul)
            np.testing.assert_allclose(diag_op.to_dense(), jnp.diag(jnp.full(n, vals)))
