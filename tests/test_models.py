"""Tests for Gaussian process models."""

import gpjax as gpjax
import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from gpjax.distributions import GaussianDistribution
from gpjax.gps import ConjugatePosterior
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Constant
from gpjax.objectives import elbo
from gpjax.parameters import Real

import cagpjax
from cagpjax.models import ComputationAwareGP
from cagpjax.policies import BlockSparsePolicy, LanczosPolicy
from cagpjax.solvers import Cholesky, PseudoInverse

jax.config.update("jax_enable_x64", True)


class TestComputationAwareGP:
    """Test suite for ComputationAwareGP class."""

    @pytest.fixture(params=[jnp.float32, jnp.float64])
    def dtype(self, request):
        return request.param

    @pytest.fixture(params=[10, 30])
    def n_train(self, request):
        return request.param

    @pytest.fixture()
    def n_test(self):
        return 20

    @pytest.fixture(params=[3, 5])
    def n_actions(self, request):
        return request.param

    @pytest.fixture(params=[LanczosPolicy, BlockSparsePolicy])
    def policy(self, request, n_train, n_actions, dtype, key=jax.random.key(98)):
        policy_class = request.param
        if policy_class is LanczosPolicy:
            return policy_class(n_actions=n_actions, key=key)
        elif policy_class is BlockSparsePolicy:
            return policy_class(n_actions=n_actions, n=n_train, key=key, dtype=dtype)
        else:
            raise ValueError(f"Invalid policy class: {policy_class}")

    @pytest.fixture(params=[Cholesky, PseudoInverse])
    def solver_method(self, request):
        """Method for solving the linear system of equations."""
        if request.param is Cholesky:
            return request.param(1e-6)
        elif request.param is PseudoInverse:
            return request.param()
        else:
            raise ValueError(f"Invalid solver method class: {request.param}")

    def generate_dataset(self, n, bounds, dtype, key, noise=0.1):
        """Create simple 1D training data for testing."""
        x_train = jnp.linspace(bounds[0], bounds[1], n, dtype=dtype).reshape(-1, 1)
        y_train = (
            jnp.sin(x_train.squeeze())
            + noise * jax.random.normal(key, (n,), dtype=dtype)
        ).reshape(-1, 1)

        return gpjax.Dataset(X=x_train, y=y_train)

    @pytest.fixture
    def train_data(self, n_train, dtype, key=jax.random.key(42)):
        """Create simple 1D training data for testing."""
        return self.generate_dataset(n_train, (-2.0, 2.0), dtype, key)

    @pytest.fixture
    def train_data_alt(self, n_train, dtype, key=jax.random.key(37)):
        """Create 2nd simple 1D training data for testing."""
        return self.generate_dataset(n_train, (-3.0, 1.0), dtype, key)

    @pytest.fixture
    def test_data(self, n_test, dtype, key=jax.random.key(74)):
        """Create simple 1D test data for testing."""
        return self.generate_dataset(n_test, (-2.5, 2.5), dtype, key)

    @pytest.fixture(params=[jnp.array, Real])
    def constant_type(self, request):
        return request.param

    @pytest.fixture
    def posterior(self, n_train, dtype, constant_type):
        """Create GP posterior."""
        kernel = RBF(
            lengthscale=jnp.array(1.0, dtype=dtype),
            variance=jnp.array(1.0, dtype=dtype),
        )
        likelihood = Gaussian(
            num_datapoints=n_train, obs_stddev=jnp.array(0.1, dtype=dtype)
        )
        mean_function = Constant(constant_type(jnp.array(3.0, dtype=dtype)))
        prior = gpjax.gps.Prior(kernel=kernel, mean_function=mean_function)
        posterior = ConjugatePosterior(prior=prior, likelihood=likelihood)

        return posterior

    @pytest.fixture
    def conditioned_cagp(self, policy, posterior, train_data, solver_method):
        """Create CAGP policy."""
        cagp = ComputationAwareGP(
            posterior=posterior, policy=policy, solver_method=solver_method
        )
        cagp.condition(train_data)
        return cagp

    @pytest.mark.skipif(constant_type is nnx.Variable, reason="Test is redundant")
    def test_initialization(self, policy, posterior, solver_method):
        """Test that CAGP initializes correctly."""
        cagp = ComputationAwareGP(
            posterior=posterior, policy=policy, solver_method=solver_method
        )
        assert cagp.posterior is posterior
        assert cagp.policy is policy
        assert isinstance(cagp.solver_method, solver_method.__class__)
        if isinstance(solver_method, Cholesky):
            assert cagp.solver_method.jitter == solver_method.jitter
        elif isinstance(solver_method, PseudoInverse):
            assert cagp.solver_method.rtol == solver_method.rtol
        assert not cagp.is_conditioned

    def test_condition(self, policy, posterior, train_data, solver_method):
        """Test that CAGP can be conditioned on data."""
        import warnings

        # Convert the JAX dtype warning to an error so test fails if it occurs
        warnings.filterwarnings(
            "error", message=".*scatter inputs have incompatible types.*"
        )

        cagp = ComputationAwareGP(
            posterior=posterior, policy=policy, solver_method=solver_method
        )
        cagp.condition(train_data)
        assert cagp.is_conditioned
        assert isinstance(
            cagp._posterior_params, cagpjax.models.cagp._ProjectedPosteriorParameters
        )

    @pytest.mark.parametrize(
        "dataset_replacements", [{"X": None}, {"y": None}, {"X": None, "y": None}]
    )
    def test_condition_requires_supervised_data(
        self, policy, posterior, train_data, dataset_replacements
    ):
        """Test that CAGP raises an error if conditioned on unsupervised data."""
        cagp = ComputationAwareGP(posterior=posterior, policy=policy)
        train_data_entries = {"X": train_data.X, "y": train_data.y}
        train_data_entries.update(dataset_replacements)
        train_data_new = gpjax.Dataset(**train_data_entries)
        with pytest.raises(ValueError, match="Training data must be supervised"):
            cagp.condition(train_data_new)

    def test_reconditioning_with_new_data(
        self, policy, posterior, train_data, train_data_alt, solver_method
    ):
        """Test that CAGP can be reconditioned with new data."""
        cagp = ComputationAwareGP(
            posterior=posterior, policy=policy, solver_method=solver_method
        )
        cagp.condition(train_data)
        assert cagp._posterior_params is not None  # help pyright
        assert jnp.allclose(cagp._posterior_params.x, train_data.X)
        cagp.condition(train_data_alt)
        assert cagp._posterior_params is not None  # help pyright
        assert jnp.allclose(cagp._posterior_params.x, train_data_alt.X)

    def test_predict_test_inputs(self, conditioned_cagp, test_data, n_test, dtype):
        """Test that CAGP predict method with test inputs works."""
        x_test = test_data.X
        pred = conditioned_cagp.predict(x_test)
        assert isinstance(pred, GaussianDistribution)
        assert pred.mean.shape == (n_test,)
        assert pred.mean.dtype == dtype
        assert pred.scale.shape == (n_test, n_test)
        assert pred.scale.dtype == dtype

    def test_predict_no_inputs(self, conditioned_cagp, train_data, dtype):
        """Test that CAGP predict method with no inputs evaluates at training inputs."""
        if dtype == jnp.float32:
            pytest.skip("Skipping float32 test due to numerical precision limitations")

        x_train = train_data.X
        pred = conditioned_cagp.predict()
        pred2 = conditioned_cagp.predict(x_train)

        # Use tight tolerance for float64 where numerical precision is sufficient
        assert jnp.allclose(pred.mean, pred2.mean)
        assert jnp.allclose(pred.scale.to_dense(), pred2.scale.to_dense(), atol=1e-7)

    def test_predict_exact_when_n_actions_equals_n(
        self,
        train_data,
        test_data,
        posterior,
        n_train,
        dtype,
        solver_method,
        key=jax.random.key(42),
    ):
        """Test that when n_actions=N, CAGP produces the same mean and variance as exact GP."""
        if dtype == jnp.float32:
            pytest.skip("Skipping float32 test due to numerical precision limitations")

        x_test = test_data.X
        policy = LanczosPolicy(n_actions=n_train, key=key)
        cagp = ComputationAwareGP(
            posterior=posterior, policy=policy, solver_method=solver_method
        )
        cagp.condition(train_data)
        pred = cagp.predict(x_test)

        pred_exact = posterior.predict(x_test, train_data)

        assert pred.mean.shape == pred_exact.mean.shape
        assert pred.scale.shape == pred_exact.scale.shape
        assert jnp.allclose(pred.mean, pred_exact.mean, atol=1e-4)
        assert jnp.allclose(
            pred.scale.to_dense(), pred_exact.scale.to_dense(), atol=1e-5
        )

    @pytest.mark.skipif(constant_type is nnx.Variable, reason="Test is redundant")
    def test_prior_kl_consistency(self, conditioned_cagp, train_data, dtype):
        """Test that custom ``prior_kl`` matches KL computed from result of ``predict``."""
        if dtype == jnp.float32:
            pytest.skip("Skipping float32 test due to numerical precision limitations")

        kl = conditioned_cagp.prior_kl()
        assert isinstance(kl, jnp.ndarray)
        assert kl.dtype == dtype
        assert jnp.isscalar(kl)
        assert jnp.isfinite(kl)
        assert kl >= 0.0

        # compare with KL computed explicitly from predictive distributions
        q = conditioned_cagp.predict()
        p = conditioned_cagp.posterior.prior.predict(train_data.X)
        kl_explicit = q.kl_divergence(p)

        assert jnp.allclose(kl, kl_explicit, rtol=1e-4)

    @pytest.mark.skipif(constant_type is nnx.Variable, reason="Test is redundant")
    def test_prior_kl_gradient_sparse_actions(
        self,
        posterior,
        n_train,
        train_data,
        dtype,
        solver_method,
        key=jax.random.key(42),
    ):
        """Test that ``prior_kl`` gradient wrt sparse action parameters is correct."""
        if dtype == jnp.float32:
            pytest.skip("Skipping float32 test due to numerical precision limitations")

        def kl_objective(nz_values):
            policy = BlockSparsePolicy(n_actions=n_train, nz_values=nz_values)
            cagp = ComputationAwareGP(
                posterior=posterior, policy=policy, solver_method=solver_method
            )
            cagp.condition(train_data)
            return cagp.prior_kl()

        nz_values = jax.random.normal(key, (n_train,), dtype=dtype)
        jax.test_util.check_grads(kl_objective, (nz_values,), order=1, modes=["rev"])

    def test_integration_elbo(self, conditioned_cagp, posterior, train_data, dtype):
        """Test that ``elbo`` objective from GPJax is computed correctly."""
        if dtype == jnp.float32:
            pytest.skip("Skipping float32 test due to numerical precision limitations")

        elbo_value = elbo(conditioned_cagp, train_data)
        assert isinstance(elbo_value, jnp.ndarray)
        assert elbo_value.dtype == dtype
        assert jnp.isscalar(elbo_value)
        assert jnp.isfinite(elbo_value)

        # compute CaGP posterior mean and variance
        q = conditioned_cagp.predict()
        mean_q = q.mean
        var_q = q.variance

        # compute ELBO explicitly
        var_exp = posterior.likelihood.expected_log_likelihood(
            train_data.y, mean_q[:, None], var_q[:, None]
        )
        elbo_value_explicit = var_exp.sum() - conditioned_cagp.prior_kl()

        assert jnp.allclose(elbo_value, elbo_value_explicit)
