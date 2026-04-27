"""Tests for custom linear operators."""

import jax
import jax.numpy as jnp
import jax.test_util
import lineax as lx
import numpy as np
import pytest
from gpjax.kernels import RBF
from gpjax.kernels.computations import DenseKernelComputation
from gpjax.parameters import Real

from cagpjax.operators import BlockDiagonalSparse
from cagpjax.operators.diag_like import diag_like
from cagpjax.operators.lazy_kernel import LazyKernel

jax.config.update("jax_enable_x64", True)


def _test_mul_consistency(op, **kwargs):
    """Test that the linear operator is consistent with multiplication by identity."""
    mat = op.as_matrix()
    eye_in = jnp.eye(op.in_size(), dtype=mat.dtype)
    right = jax.vmap(op.mv, in_axes=1, out_axes=1)(eye_in)
    np.testing.assert_allclose(right, mat, **kwargs)


def _test_transpose_consistency(op, **kwargs):
    """Test that the transpose is consistent."""
    np.testing.assert_allclose(op.transpose().as_matrix(), op.as_matrix().T, **kwargs)


def _test_structure_consistency(op, **kwargs):
    """Test that the linear operator has the correct structure."""
    in_structure = op.in_structure()
    out_structure = op.out_structure()
    mat = op.as_matrix()
    assert in_structure.dtype == mat.dtype
    assert in_structure.shape == (mat.shape[1],)
    assert out_structure.dtype == mat.dtype
    assert out_structure.shape == (mat.shape[0],)


def _test_linear_operator_consistency(op, **kwargs):
    """Test that the linear operator is self-consistent."""
    _test_mul_consistency(op, **kwargs)
    _test_transpose_consistency(op, **kwargs)
    _test_structure_consistency(op, **kwargs)


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
        assert op.out_size() == n_nz_values
        assert op.in_size() == n_blocks
        assert op.in_structure().dtype == dtype
        assert op.out_structure().dtype == dtype
        _test_linear_operator_consistency(op)

    def test_grad(self, shape, dtype=jnp.float64, key=jax.random.key(42)):
        """Test gradient containing the linear operator."""
        n_nz_values, n_blocks = shape
        key, subkey = jax.random.split(key)
        nz_values = jax.random.normal(subkey, (n_nz_values,), dtype=dtype)
        op = BlockDiagonalSparse(nz_values, n_blocks)

        f1 = lambda x: jnp.prod(jnp.sin(op.mv(x)))
        x1 = jax.random.normal(key, (n_blocks,), dtype=dtype)
        jax.test_util.check_grads(f1, (x1,), order=1)

        f2 = lambda x: jnp.prod(jnp.sin(op.transpose().mv(x)))
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
        ref_op = lx.MatrixLinearOperator(jnp.ones((n, n), dtype=dtype))
        diag_op = diag_like(ref_op, vals)
        _test_linear_operator_consistency(diag_op)
        expected_vals = vals
        assert isinstance(diag_op, lx.DiagonalLinearOperator)
        if isinstance(vals, jnp.ndarray) and vals.ndim == 1:
            np.testing.assert_allclose(diag_op.as_matrix(), jnp.diag(expected_vals))
        else:
            np.testing.assert_allclose(diag_op.as_matrix(), jnp.diag(jnp.full(n, vals)))

        assert diag_op.in_size() == n
        assert diag_op.out_size() == n
        assert diag_op.in_structure().dtype == dtype
        assert diag_op.out_structure().dtype == dtype


class TestLazyKernel:
    """Test cases for LazyKernel."""

    @pytest.fixture(params=[jnp.float32, jnp.float64])
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
            lengthscale=Real(jnp.array(1.0, dtype=dtype)),
            variance=Real(jnp.array(1.0, dtype=dtype)),
            compute_engine=DenseKernelComputation(),
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

    @pytest.mark.parametrize("checkpoint", [True, False])
    def test_initialization(
        self, inputs, kernel, batch_size, max_memory_mb, checkpoint
    ):
        """Test initialization for all parameter combinations."""
        op = LazyKernel(
            kernel,
            *inputs,
            batch_size=batch_size,
            max_memory_mb=max_memory_mb,
            checkpoint=checkpoint,
        )
        x1, x2 = inputs
        assert op.in_structure().dtype == kernel(x1[0], x2[0]).dtype
        assert op.out_structure().dtype == kernel(x1[0], x2[0]).dtype
        assert op.in_size() == x2.shape[0]
        assert op.out_size() == x1.shape[0]
        assert op.kernel is kernel
        assert op.x1 is x1
        assert op.x2 is x2
        assert op.checkpoint is checkpoint
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

    @jax.default_matmul_precision("highest")
    def test_linear_operator_consistency(self, op, atol=1e-6):
        """Test LinearOperator consistency for all parameter combinations."""
        _test_linear_operator_consistency(op, atol=atol)

    @jax.default_matmul_precision("highest")
    def test_consistency_with_dense(self, op, kernel, inputs):
        """Test consistency with dense kernel matrix."""
        x1, x2 = inputs
        expected = kernel.cross_covariance(x1, x2)
        expected = expected.as_matrix() if hasattr(expected, "as_matrix") else expected
        assert jnp.allclose(op.as_matrix(), expected)
        assert lx.is_symmetric(op) == bool(jnp.array_equal(x1, x2))
        assert lx.is_positive_semidefinite(op) == bool(jnp.array_equal(x1, x2))

    def test_diagonal_when_singleton_inputs(self, kernel, dtype):
        x = jnp.zeros((1, 2), dtype=dtype)
        op = LazyKernel(kernel, x, x)
        assert lx.is_diagonal(op)
        assert lx.is_tridiagonal(op)

    @pytest.mark.parametrize("grad,checkpoint", [(False, False), (True, True)])
    @pytest.mark.parametrize("n,dtype", [(20_000, jnp.float64), (40_000, jnp.float32)])
    def test_large_kernel_matrix_with_grad(
        self,
        grad,
        checkpoint,
        n,
        dtype,
        max_memory_mb=2**8,  # 256MB
        key=jax.random.key(42),
    ):
        """Test finite loss and (optional) gradient computation."""

        (*subkeys,) = jax.random.split(key, 4)
        x1 = jax.random.normal(subkeys[0], (n, 2), dtype=dtype)
        x2 = jax.random.normal(subkeys[1], (n, 2), dtype=dtype)
        v = jax.random.normal(subkeys[2], (n,), dtype=dtype)

        lengthscale, variance = jax.random.uniform(subkeys[3], (2,), dtype=dtype)

        kernel = RBF(
            lengthscale=lengthscale,
            variance=variance,
            compute_engine=DenseKernelComputation(),
        )

        def loss(kernel):
            op = LazyKernel(
                kernel, x1, x2, max_memory_mb=max_memory_mb, checkpoint=checkpoint
            )
            with jax.default_matmul_precision("highest"):
                return jnp.vdot(v, op.mv(v))

        assert jnp.isfinite(loss(kernel))
        if grad:
            rtol = 1e-2 if dtype == jnp.float32 else 1e-6
            jax.test_util.check_grads(
                loss, (kernel,), order=1, modes=["rev"], rtol=rtol
            )
