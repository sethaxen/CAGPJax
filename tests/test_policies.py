"""Test the linear solver policies."""

import cola
import gpjax.kernels
import jax
import jax.numpy as jnp
import pytest
from cola.ops import Dense, LinearOperator, Transpose
from gpjax.dataset import Dataset
from gpjax.parameters import Real, Static

from cagpjax.operators import BlockDiagonalSparse
from cagpjax.policies import BlockSparsePolicy, LanczosPolicy, PseudoInputPolicy

jax.config.update("jax_enable_x64", True)


def _test_batch_policy_actions_consistency(policy, op: LinearOperator):
    """Test a batch policy."""
    actions = policy.to_actions(op)
    assert isinstance(actions, LinearOperator)
    assert actions.shape == (op.shape[0], policy.n_actions)
    assert actions.dtype == op.dtype


@pytest.fixture(params=[[10, jnp.float32], [20, jnp.float64]])
def psd_linear_operator(request, key=jax.random.key(42)):
    """Create a sample linear operator for testing."""
    n, dtype = request.param
    B = jax.random.normal(key, (n, n), dtype=dtype)
    A = B @ B.T
    return cola.PSD(Dense(A))


class TestLanczosPolicy:
    """Test the LanczosPolicy concrete implementation."""

    @pytest.mark.parametrize("key", [None, jax.random.key(42)])
    @pytest.mark.parametrize("n_actions", [2, 4])
    def test_init_with_valid_params(self, n_actions, key):
        """Test initialization with valid parameters."""
        actions = LanczosPolicy(n_actions=n_actions, key=key)
        assert actions.n_actions == n_actions
        if key is None:
            assert actions.key is None
        else:
            assert actions.key is not None
            assert jnp.array_equal(actions.key, key)

    def test_actions_consistency(self, psd_linear_operator, key=jax.random.key(42)):
        """Test that the actions are consistent."""
        actions = LanczosPolicy(n_actions=2, key=key)
        _test_batch_policy_actions_consistency(actions, psd_linear_operator)

    def test_reproducibility(
        self, psd_linear_operator, n_actions=5, key=jax.random.key(42)
    ):
        """Test that using the same key produces the same results."""
        actions1 = LanczosPolicy(n_actions=n_actions, key=key)
        actions2 = LanczosPolicy(n_actions=n_actions, key=key)

        result1 = actions1.to_actions(psd_linear_operator)
        result2 = actions2.to_actions(psd_linear_operator)

        assert jnp.array_equal(
            result1 @ jnp.eye(n_actions), result2 @ jnp.eye(n_actions)
        )

    @pytest.mark.parametrize("n_actions", [8, None])
    def test_eigenvectors_match_dense_computation(
        self, psd_linear_operator, n_actions, key=jax.random.key(42)
    ):
        """Test that the eigenvectors match those from dense eigendecomposition."""

        if n_actions is None:
            n_actions = psd_linear_operator.shape[0]
            atol = 1e-8 if psd_linear_operator.dtype == jnp.float64 else 1e-5
            nvecs_check = n_actions
        else:
            atol = 1e-4
            nvecs_check = 2

        # Get eigenvectors using LanczosPolicy
        actions = LanczosPolicy(n_actions=n_actions, key=key)
        cg_vecs = actions.to_actions(psd_linear_operator)

        # Get reference eigenvectors using dense computation
        _, eigenvecs = jnp.linalg.eigh(psd_linear_operator.to_dense())
        # Get the largest nvecs_check eigenvectors (eigh returns them in ascending order)
        ref_vecs = eigenvecs[:, -nvecs_check:]

        # Convert LanczosPolicy result to dense array for comparison
        cg_vecs_dense = cg_vecs.to_dense()[:, -nvecs_check:]

        # Compare eigenvectors (they should match up to sign)
        # Computing the product should give a diagonal matrix with +/-1 entries
        product = cg_vecs_dense.T @ ref_vecs
        abs_product = jnp.abs(product)
        expected_identity = jnp.eye(nvecs_check, dtype=cg_vecs_dense.dtype)

        assert jnp.allclose(abs_product, expected_identity, atol=atol), (
            "CG eigenvectors don't match reference eigenvectors (up to sign)"
        )

    @pytest.mark.parametrize("grad_rtol", [None, 1e-9])
    @pytest.mark.parametrize("key", [jax.random.key(42), jax.random.key(89)])
    def test_eigenvector_gradient_degenerate(
        self, grad_rtol, key, n=10, dtype=jnp.float64
    ):
        """Test that the gradient is zero for a degenerate matrix."""
        policy = LanczosPolicy(n_actions=n, grad_rtol=grad_rtol, key=key)
        assert policy.grad_rtol == grad_rtol
        x = jnp.ones(n, dtype=dtype)
        scale = jnp.concatenate(
            [jnp.ones(n - 2, dtype=dtype), jnp.full(2, 3.0, dtype=dtype)]
        )

        # This loss function is constant within floating point precision, so its gradient
        # when considering all eigenvectors should be zero.
        # If matrix is degenerate, gradient will contain NaNs or (because Lanczos is approximate)
        # be numerically unstable.
        # Increasing grad_rtol should stabilize the gradient.
        def loss(op_diag):
            op = cola.lazify(jnp.diag(op_diag))
            actions = policy.to_actions(op)
            z = actions @ ((actions.T @ x) * scale)
            return jnp.sum(jnp.square(z))

        op_diag = jnp.concatenate(
            [jnp.linspace(0, 1, n - 2, dtype=dtype) * 0.5, jnp.ones(2, dtype=dtype)]
        )
        grad = jax.grad(loss)(op_diag)
        if grad_rtol is None:
            assert not jnp.isclose(jnp.abs(grad).max(), 0.0, atol=1e-3)
        else:
            assert jnp.isclose(jnp.abs(grad).max(), 0.0, atol=1e-5)


class TestBlockSparsePolicy:
    """Test the BlockSparsePolicy concrete implementation."""

    @pytest.mark.parametrize("key", [None, jax.random.key(42)])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("n_actions", [2, 3])
    @pytest.mark.parametrize("n", [10, 20])
    def test_init_basic(self, n_actions, n, key, dtype):
        """Test basic initialization."""
        policy = BlockSparsePolicy(n_actions=n_actions, n=n, key=key, dtype=dtype)

        assert policy.n_actions == n_actions
        assert isinstance(policy.nz_values, Real)
        assert policy.nz_values.value.shape == (n,)
        assert policy.nz_values.value.dtype == dtype

    @pytest.mark.parametrize("n_actions", [2, 3])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("n", [10, 20])
    def test_init_with_provided_values(
        self, n_actions, n, dtype, key=jax.random.key(42)
    ):
        """Test initialization with provided nz_values."""
        nz_values = jax.random.normal(key, (n,), dtype=dtype)

        policy = BlockSparsePolicy(n_actions=n_actions, nz_values=nz_values)
        assert policy.n_actions == n_actions
        assert isinstance(policy.nz_values, Real)
        assert policy.nz_values.value.dtype == dtype
        assert jnp.allclose(policy.nz_values.value, nz_values)

        policy_static = BlockSparsePolicy(
            n_actions=n_actions, nz_values=Static(nz_values)
        )
        assert isinstance(policy_static.nz_values, Static)
        assert policy_static.nz_values.value.dtype == dtype
        assert jnp.allclose(policy_static.nz_values.value, nz_values)

    @pytest.mark.parametrize("n_actions", [2, 3])
    def test_to_actions_consistency(
        self, psd_linear_operator, n_actions, key=jax.random.key(42)
    ):
        """Test to_actions consistency and return type."""
        n = psd_linear_operator.shape[0]
        dtype = psd_linear_operator.dtype
        policy = BlockSparsePolicy(n_actions=n_actions, n=n, key=key, dtype=dtype)
        _test_batch_policy_actions_consistency(policy, psd_linear_operator)
        action = policy.to_actions(psd_linear_operator)
        assert isinstance(action, Transpose)
        assert isinstance(action.T, BlockDiagonalSparse)

    @pytest.mark.parametrize("n_actions", [2, 3])
    def test_to_actions_reproducible(
        self, psd_linear_operator, n_actions, key=jax.random.key(42)
    ):
        """Test that to_actions produces reproducible results with same values."""
        n = psd_linear_operator.shape[0]
        dtype = psd_linear_operator.dtype
        policy = BlockSparsePolicy(n_actions=n_actions, n=n, key=key, dtype=dtype)
        actions1 = policy.to_actions(psd_linear_operator)
        actions2 = policy.to_actions(psd_linear_operator)
        assert jnp.allclose(actions1.to_dense(), actions2.to_dense())


class TestPseudoInputPolicy:
    """Test the PseudoInputPolicy."""

    @pytest.fixture(params=[jnp.float32, jnp.float64])
    def dtype(self, request):
        """Data type for testing."""
        return request.param

    @pytest.fixture(params=[1, 2])
    def input_dim(self, request):
        """Dimension of the input space."""
        return request.param

    @pytest.fixture(params=[gpjax.kernels.RBF, gpjax.kernels.Matern32])
    def kernel(self, request, input_dim, dtype, key=jax.random.key(42)):
        """Create a kernel for testing."""
        lengthscale = jax.random.uniform(key, (input_dim,), dtype=dtype)
        variance = jax.random.uniform(key, (), dtype=dtype)
        return request.param(lengthscale=lengthscale, variance=variance)

    @pytest.fixture(params=[(10, 5), (15, 8)])
    def inputs(self, request, input_dim, dtype, key=jax.random.key(42)):
        """Create a training dataset."""
        n, n_pseudo = request.param
        key, subkey = jax.random.split(key)
        train_inputs = jax.random.normal(subkey, (n, input_dim), dtype=dtype)
        pseudo_inputs = jax.random.normal(key, (n_pseudo, input_dim), dtype=dtype)
        return train_inputs, pseudo_inputs

    @pytest.mark.parametrize("pseudo_input_type", [jnp.ndarray, Static, Real])
    @pytest.mark.parametrize("train_input_type", [jnp.ndarray, Dataset])
    def test_basic_properties(
        self,
        inputs,
        kernel,
        train_input_type,
        pseudo_input_type,
    ):
        """Test basic initialization."""
        train_inputs, pseudo_inputs = inputs
        train_data = (
            Dataset(X=train_inputs) if train_input_type is Dataset else train_inputs
        )
        pseudo_input_wrapped = (
            pseudo_input_type(pseudo_inputs)
            if pseudo_input_type is not jnp.ndarray
            else pseudo_inputs
        )
        policy = PseudoInputPolicy(pseudo_input_wrapped, train_data, kernel)

        # check correctly initialized
        assert isinstance(policy, PseudoInputPolicy)
        assert policy.n_actions == pseudo_inputs.shape[0]
        assert policy.kernel is kernel
        assert isinstance(policy.train_inputs, jnp.ndarray)
        assert jnp.array_equal(policy.train_inputs, train_inputs)
        if pseudo_input_type is jnp.ndarray:
            assert isinstance(policy.pseudo_inputs, Static)
        else:
            assert isinstance(policy.pseudo_inputs, pseudo_input_type)
        assert jnp.array_equal(policy.pseudo_inputs.value, pseudo_inputs)

    def test_actions_is_cross_covariance(self, inputs, kernel, dtype):
        """Test actions are the cross-covariance between the training inputs and pseudo-inputs."""
        train_inputs, pseudo_inputs = inputs
        policy = PseudoInputPolicy(pseudo_inputs, train_inputs, kernel)
        op = kernel.gram(train_inputs)
        actions = policy.to_actions(op)
        assert isinstance(actions, LinearOperator)
        assert actions.dtype == dtype
        assert jnp.allclose(
            actions.to_dense(), kernel.cross_covariance(train_inputs, pseudo_inputs)
        )

    def test_actions_consistency(self, inputs, kernel):
        """Test to_actions consistency and return type."""
        train_inputs, pseudo_inputs = inputs
        policy = PseudoInputPolicy(pseudo_inputs, train_inputs, kernel)
        op = kernel.gram(train_inputs)
        _test_batch_policy_actions_consistency(policy, op)
