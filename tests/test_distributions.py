import math

import cola
import jax
import jax.numpy as jnp
import jax.scipy.stats
import numpyro
import pytest
from numpyro.distributions import MultivariateNormal

import cagpjax
from cagpjax.distributions import GaussianDistribution
from cagpjax.operators.utils import lazify
from cagpjax.solvers import Cholesky, PseudoInverse

jax.config.update("jax_enable_x64", True)


class TestGaussianDistribution:
    """Test suite for GaussianDistribution class."""

    @pytest.fixture(params=[jnp.float32, jnp.float64])
    def dtype(self, request):
        return request.param

    @pytest.fixture(params=[1, 2, 3])
    def n(self, request):
        return request.param

    @pytest.fixture
    def loc(self, n, dtype, key=jax.random.key(42)):
        return jax.random.normal(key, (n,), dtype=dtype)

    @pytest.fixture
    def scale(self, n, dtype, key=jax.random.key(98)):
        A = jax.random.normal(key, (n, n), dtype=dtype)
        return lazify(A @ A.T)

    @pytest.fixture(params=[Cholesky, PseudoInverse])
    def solver(self, request):
        """Method for solving the linear system of equations."""
        if request.param is Cholesky:
            return request.param(1e-6)
        elif request.param is PseudoInverse:
            return request.param()
        else:
            raise ValueError(f"Invalid solver method class: {request.param}")

    def test_construction(self, loc, scale, solver):
        """Test construction of GaussianDistribution."""
        dist = GaussianDistribution(loc, scale, solver)
        assert isinstance(dist, GaussianDistribution)
        assert isinstance(dist, numpyro.distributions.Distribution)
        assert dist.loc is loc
        assert dist.scale is scale
        assert dist.solver is solver

    def test_numpyro_compatibility(self, dtype, n, loc, scale):
        """Test basic properties of GaussianDistribution."""
        dist = GaussianDistribution(loc, scale)
        assert dist.support == numpyro.distributions.constraints.real_vector
        assert dist.batch_shape == ()
        assert dist.event_shape == (n,)
        assert dist.event_dim == 1
        assert dist.shape() == (n,)

    def test_moments(self, dtype, n, loc, scale):
        """Test moment accessors of GaussianDistribution."""
        dist = GaussianDistribution(loc, scale)
        assert dist.mean.dtype == dtype
        assert jnp.allclose(dist.mean, loc)
        assert dist.variance.dtype == dtype
        assert dist.variance.shape == (n,)
        assert jnp.allclose(dist.variance, cola.diag(scale))
        assert dist.stddev.dtype == dtype
        assert dist.stddev.shape == (n,)
        assert jnp.allclose(dist.stddev, jnp.sqrt(cola.diag(scale)))

    def test_log_prob_consistency_with_numpyro(
        self, dtype, loc, scale, solver, key=jax.random.key(76)
    ):
        """Test log_prob of GaussianDistribution matches numpyro's MultivariateNormal."""
        y = jax.random.normal(key, loc.shape, dtype=dtype)
        dist = GaussianDistribution(loc, scale, solver)
        lp = dist.log_prob(y)
        assert lp.dtype == dtype
        scale_dense = cola.densify(scale)
        lp_ref = MultivariateNormal(loc, scale_dense).log_prob(y)
        assert jnp.allclose(lp, lp_ref, rtol=1e-3 if dtype == jnp.float32 else 1e-5)

    @pytest.mark.parametrize(
        "sample_shape",
        [(), (100_000,), (100_000, 3), (100_000, 3, 4)],
    )
    def test_sample(
        self, sample_shape, dtype, loc, scale, solver, key=jax.random.key(92)
    ):
        """Test sample of GaussianDistribution."""
        dist = GaussianDistribution(loc, scale, solver)
        x = dist.sample(key, sample_shape)
        assert x.shape == sample_shape + loc.shape
        assert x.dtype == dtype
        if len(sample_shape) > 0:
            nsample = math.prod(sample_shape)
            # tolerance needed for 99% CIs asymptotically
            atol_mean = jax.scipy.stats.norm.ppf(
                0.995, scale=dist.stddev / math.sqrt(nsample)
            )
            atol_std = jax.scipy.stats.norm.ppf(
                0.995, scale=dist.stddev / math.sqrt(2 * nsample)
            )
            if dtype == jnp.float32:  # slightly relax tolerances for float32
                atol_mean *= 1.2
                atol_std *= 1.2
            sample_axis = tuple(range(len(sample_shape)))
            assert jnp.allclose(
                jnp.mean(x, axis=sample_axis), dist.mean, atol=atol_mean
            )
            assert jnp.allclose(
                jnp.std(x, axis=sample_axis), dist.stddev, atol=atol_std
            )
