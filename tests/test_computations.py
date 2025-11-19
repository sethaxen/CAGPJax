"""Tests for kernel computation methods."""

import cola
import jax
import jax.numpy as jnp
import pytest
from gpjax.kernels import RBF

from cagpjax.computations import LazyKernelComputation
from cagpjax.operators import LazyKernel
from cagpjax.operators.utils import lazify

jax.config.update("jax_enable_x64", True)


class TestLazyKernelComputation:
    """Test cases for LazyKernelComputation."""

    @pytest.fixture(params=[0.0, 2**5])
    def max_memory_mb(self, request):
        return request.param

    @pytest.fixture(params=[1, 3, None])
    def batch_size(self, request):
        return request.param

    @pytest.fixture(params=[False, True])
    def checkpoint(self, request):
        return request.param

    @pytest.fixture(params=[jnp.float64, jnp.float32])
    def dtype(self, request):
        return request.param

    @pytest.fixture
    def kernel(self, dtype):
        return RBF(
            lengthscale=jnp.array(1.0, dtype=dtype),
            variance=jnp.array(1.0, dtype=dtype),
        )

    @pytest.fixture(params=[(9, 5), (7, 10)], ids=["(9, 5)", "(7, 10)"])
    def shape(self, request):
        return request.param

    @pytest.fixture
    def n_dims(self):
        return 2

    @pytest.fixture
    def inputs(self, shape, n_dims, dtype, key=jax.random.key(42)):
        """Create first set of input points."""
        _, subkey1, subkey2 = jax.random.split(key, 3)
        x1 = jax.random.normal(subkey1, (shape[0], n_dims), dtype=dtype)
        x2 = jax.random.normal(subkey2, (shape[1], n_dims), dtype=dtype)
        return x1, x2

    def test_initialization(self, batch_size, max_memory_mb, checkpoint):
        """Test initialization for all parameter combinations."""
        comp = LazyKernelComputation(
            batch_size=batch_size, max_memory_mb=max_memory_mb, checkpoint=checkpoint
        )
        assert isinstance(comp, LazyKernelComputation)
        assert comp.batch_size == batch_size
        assert comp.max_memory_mb == max_memory_mb
        assert comp.checkpoint == checkpoint

    @jax.default_matmul_precision("highest")
    def test_gram(self, kernel, inputs, dtype, batch_size, max_memory_mb, checkpoint):
        """Test Gram matrix computation."""
        comp = LazyKernelComputation(
            batch_size=batch_size, max_memory_mb=max_memory_mb, checkpoint=checkpoint
        )
        x1, _ = inputs
        gram = lazify(comp.gram(kernel, x1))
        assert isinstance(gram, LazyKernel)
        assert gram.shape == (x1.shape[0], x1.shape[0])
        assert gram.dtype == dtype
        assert gram.checkpoint == checkpoint
        assert gram.isa(cola.PSD)

        gram_lazy = LazyKernel(
            kernel,
            x1,
            x1,
            batch_size=batch_size,
            max_memory_mb=max_memory_mb,
        )
        assert gram.batch_size_row == gram_lazy.batch_size_row
        assert gram.batch_size_col == gram_lazy.batch_size_col

        assert jnp.allclose(cola.densify(gram), kernel.gram(x1).to_dense())

    @jax.default_matmul_precision("highest")
    def test_cross_covariance(
        self, kernel, inputs, dtype, batch_size, max_memory_mb, checkpoint
    ):
        """Test cross-covariance matrix computation."""
        comp = LazyKernelComputation(
            batch_size=batch_size, max_memory_mb=max_memory_mb, checkpoint=checkpoint
        )
        x1, x2 = inputs
        cross_cov = comp.cross_covariance(kernel, x1, x2)
        assert isinstance(cross_cov, LazyKernel)
        assert cross_cov.shape == (x1.shape[0], x2.shape[0])
        assert cross_cov.dtype == dtype
        assert cross_cov.checkpoint == checkpoint

        cross_cov_lazy = LazyKernel(
            kernel,
            x1,
            x2,
            batch_size=batch_size,
            max_memory_mb=max_memory_mb,
        )
        assert cross_cov.batch_size_row == cross_cov_lazy.batch_size_row
        assert cross_cov.batch_size_col == cross_cov_lazy.batch_size_col

        assert jnp.allclose(
            cola.densify(cross_cov), cola.densify(kernel.cross_covariance(x1, x2))
        )
