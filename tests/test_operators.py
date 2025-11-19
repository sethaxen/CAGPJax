"""Tests for custom linear operators."""

import cola
import gpjax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cola.ops import Dense, Diagonal, LinearOperator, ScalarMul

from cagpjax.operators import BlockDiagonalSparse
from cagpjax.operators.annotations import ScaledOrthogonal
from cagpjax.operators.diag_like import diag_like
from cagpjax.operators.utils import lazify

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


class TestUtils:
    # lazify needs to be tested for a cola.ops.Dense, cola.ops.Diagonal, and jnp.ndarray
    # if gpjax.linalg is available, it should also be tested for gpjax.linalg.Dense, gpjax.linalg.Diagonal, gpjax.linalg.Identity, and gpjax.linalg.Triangular
    # lazify should return a cola.ops.LinearOperator, and have the same shape and dtype as the input

    @pytest.fixture(params=[5, 6])
    def nrows(self, request) -> int:
        return request.param

    @pytest.fixture(params=[3, 7])
    def ncols(self, request) -> int:
        return request.param

    @pytest.fixture(params=[jnp.float32, jnp.float64])
    def dtype(self, request):
        return request.param

    def test_lazify_ndarray(self, nrows, ncols, dtype, key=jax.random.key(42)):
        arr = jax.random.normal(key, (nrows, ncols), dtype=dtype)
        lazy_op = lazify(arr)
        assert isinstance(lazy_op, cola.ops.Dense)
        assert lazy_op.shape == (nrows, ncols)
        assert lazy_op.dtype == dtype
        np.testing.assert_allclose(lazy_op.to_dense(), arr)

    def test_lazify_dense(self, nrows, ncols, dtype, key=jax.random.key(42)):
        op = cola.ops.Dense(jax.random.normal(key, (nrows, ncols), dtype=dtype))
        lazy_op = lazify(op)
        assert lazy_op is op

    def test_lazify_diagonal(self, nrows, dtype, key=jax.random.key(42)):
        diag = jax.random.normal(key, (nrows,), dtype=dtype)
        op = cola.ops.Diagonal(diag)
        lazy_op = lazify(op)
        assert lazy_op is op

    try:  # test support for GPJax v0.12.0
        import gpjax.linalg

        def test_lazify_gpjax_dense(self, nrows, ncols, dtype, key=jax.random.key(42)):
            op = gpjax.linalg.Dense(jax.random.normal(key, (nrows, ncols), dtype=dtype))
            lazy_op = lazify(op)
            assert isinstance(lazy_op, cola.ops.Dense)
            assert lazy_op.shape == (nrows, ncols)
            assert lazy_op.dtype == dtype
            np.testing.assert_allclose(lazy_op.to_dense(), op.to_dense())

        def test_lazify_gpjax_diagonal(self, nrows, dtype, key=jax.random.key(42)):
            diag = jax.random.normal(key, (nrows,), dtype=dtype)
            op = gpjax.linalg.Diagonal(diag)
            lazy_op = lazify(op)
            assert isinstance(lazy_op, cola.ops.Diagonal)
            assert lazy_op.shape == (nrows, nrows)
            assert lazy_op.dtype == dtype
            np.testing.assert_allclose(lazy_op.to_dense(), jnp.diag(diag))

        def test_lazify_gpjax_identity(self, nrows, dtype, key=jax.random.key(42)):
            op = gpjax.linalg.Identity(nrows, dtype=dtype)
            lazy_op = lazify(op)
            assert isinstance(lazy_op, cola.ops.Identity)
            assert lazy_op.shape == (nrows, nrows)
            assert lazy_op.dtype == dtype

        @pytest.mark.parametrize("lower", [True, False])
        def test_lazify_gpjax_triangular(
            self, nrows, dtype, lower, key=jax.random.key(42)
        ):
            op = gpjax.linalg.Triangular(
                jax.random.normal(key, (nrows, nrows), dtype=dtype), lower=lower
            )
            lazy_op = lazify(op)
            assert isinstance(lazy_op, cola.ops.Triangular)
            assert lazy_op.shape == (nrows, nrows)
            assert lazy_op.dtype == dtype
            assert lazy_op.lower == lower
            np.testing.assert_allclose(lazy_op.to_dense(), op.to_dense())

    except ImportError:
        pass


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
        assert op.isa(ScaledOrthogonal)
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
