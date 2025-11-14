"""Tests for custom linear operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cola.ops import Dense, Diagonal, LinearOperator, ScalarMul
from gpjax.kernels import RBF

from cagpjax.operators import BlockDiagonalSparse
from cagpjax.operators.annotations import ScaledOrthogonal
from cagpjax.operators.diag_like import diag_like
from cagpjax.operators.lazy_kernel import LazyKernel

jax.config.update("jax_enable_x64", True)


def _test_mul_consistency(op: LinearOperator, **kwargs):
    """Test that the linear operator is consistent with multiplication by identity."""
    mat = op.to_dense()
    np.testing.assert_allclose(jnp.eye(op.shape[0]) @ op, mat, **kwargs)
    np.testing.assert_allclose(op @ jnp.eye(op.shape[1]), mat, **kwargs)


def _test_transpose_consistency(op: LinearOperator, **kwargs):
    """Test that the transpose is consistent."""
    op_transpose = op.T
    assert op_transpose.shape == op.shape[::-1]
    np.testing.assert_allclose(op_transpose.to_dense(), op.to_dense().T, **kwargs)


def _test_dtype_consistency(op: LinearOperator, **kwargs):
    """Test that the linear operator has the correct dtype."""
    assert op.dtype == op.to_dense().dtype
    x = jnp.ones(op.shape[1], dtype=op.dtype)
    y = jnp.ones(op.shape[0], dtype=op.dtype)
    assert (op @ x).dtype == op.dtype
    assert (op.T @ y).dtype == op.dtype


def _test_linear_operator_consistency(op: LinearOperator, **kwargs):
    """Test that the linear operator is self-consistent."""
    _test_mul_consistency(op, **kwargs)
    _test_transpose_consistency(op, **kwargs)
    _test_dtype_consistency(op, **kwargs)


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


class TestLazyKernel:
    """Test cases for LazyKernel."""

    @pytest.fixture(params=[jnp.float64])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def n_dims(self):
        return 2

    @pytest.fixture(params=[(9, 5), (7, 10)], ids=["(9, 5)", "(7, 10)"])
    def shape(self, request):
        return request.param

    @pytest.fixture(params=[1, 3, None])
    def batch_size(self, request, shape):
        if request.param is None:
            return None
        num_elements = request.param
        return num_elements * max(*shape)

    @pytest.fixture(params=[0.0, 2**5])
    def max_memory_mb(self, request):
        return request.param

    @pytest.fixture
    def kernel(self, dtype):
        """Create RBF kernel for testing."""
        return RBF(
            lengthscale=jnp.array(1.0, dtype=dtype),
            variance=jnp.array(1.0, dtype=dtype),
        )

    @pytest.fixture
    def inputs(self, shape, n_dims, dtype, key=jax.random.key(42)):
        """Create first set of input points."""
        _, subkey1, subkey2 = jax.random.split(key, 3)
        x1 = jax.random.normal(subkey1, (shape[0], n_dims), dtype=dtype)
        x2 = jax.random.normal(subkey2, (shape[1], n_dims), dtype=dtype)
        return x1, x2

    @pytest.fixture
    def op(self, kernel, inputs, batch_size, max_memory_mb):
        """Create LazyKernel with all valid parameter combinations."""
        return LazyKernel(
            kernel, *inputs, batch_size=batch_size, max_memory_mb=max_memory_mb
        )

    def test_initialization(self, op, inputs, kernel, batch_size, max_memory_mb):
        """Test initialization for all parameter combinations."""
        x1, x2 = inputs
        assert op.dtype == x1.dtype
        assert op.kernel is kernel
        assert op.x1 is x1
        assert op.x2 is x2
        if batch_size is None:
            if max_memory_mb == 0.0:
                assert op.batch_size_row == 1
                assert op.batch_size_col == 1
            else:
                assert op.batch_size_row > 1
                assert op.batch_size_col > 1
        else:
            assert op.batch_size_row == batch_size
            assert op.batch_size_col == batch_size

    def test_linear_operator_consistency(self, op):
        """Test LinearOperator consistency for all parameter combinations."""
        _test_linear_operator_consistency(op)

    @pytest.mark.parametrize("grad,checkpoint", [(False, False), (True, True)])
    @pytest.mark.parametrize("n,dtype", [(20_000, jnp.float64), (40_000, jnp.float32)])
    def test_large_kernel_matrix_grad(
        self,
        grad,
        checkpoint,
        n,
        dtype,
        max_memory_mb=2**8,  # 256MB
        key=jax.random.key(42),
    ):
        """Test gradient computation for large kernel matrix."""

        (*subkeys,) = jax.random.split(key, 4)
        x1 = jax.random.normal(subkeys[0], (n, 2), dtype=dtype)
        x2 = jax.random.normal(subkeys[1], (n, 2), dtype=dtype)
        v = jax.random.normal(subkeys[2], (n,), dtype=dtype)

        def loss(params):
            kernel = RBF(lengthscale=params[0], variance=params[1])
            op = LazyKernel(
                kernel, x1, x2, max_memory_mb=max_memory_mb, checkpoint=checkpoint
            )
            return jnp.vdot(v, op @ v)

        params = jax.random.uniform(subkeys[3], (2,), dtype=dtype)
        if not grad:
            assert jnp.isfinite(loss(params))
        else:
            grads = jax.grad(loss)(params)
            assert grads.dtype == dtype
            assert jnp.isfinite(grads).all()
