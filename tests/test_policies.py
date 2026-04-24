"""Test the linear solver policies."""

import cola
import gpjax.kernels
import jax
import jax.numpy as jnp
import lineax as lx
import paramax
import pytest
from cola.ops import Dense, LinearOperator
from gpjax.dataset import Dataset
from gpjax.kernels.computations import DenseKernelComputation
from gpjax.parameters import Real

from cagpjax.interop import lazify
from cagpjax.linalg import OrthogonalizationMethod
from cagpjax.operators import BlockDiagonalSparse
from cagpjax.policies import (
    BlockSparsePolicy,
    LanczosPolicy,
    OrthogonalizationPolicy,
    PseudoInputPolicy,
)

jax.config.update("jax_enable_x64", True)


def _test_batch_policy_actions_consistency(
    policy, op: LinearOperator, key: jax.Array | None = None
):
    """Test a batch policy."""
    actions = policy.to_actions(op, key=key)
    assert isinstance(actions, (LinearOperator, lx.AbstractLinearOperator))
    if isinstance(actions, lx.AbstractLinearOperator):
        assert actions.out_size() == op.shape[0]
        assert actions.in_size() == policy.n_actions
        assert actions.in_structure().dtype == op.dtype
    else:
        assert actions.shape == (op.shape[0], policy.n_actions)
        assert actions.dtype == op.dtype


def _matrix(op):
    if isinstance(op, lx.AbstractLinearOperator):
        return op.as_matrix()
    return op.to_dense()


@pytest.fixture(params=[[10, jnp.float32], [20, jnp.float64]])
def psd_linear_operator(request, key=jax.random.key(42)):
    """Create a sample linear operator for testing."""
    n, dtype = request.param
    B = jax.random.normal(key, (n, n), dtype=dtype)
    A = B @ B.T
    return cola.PSD(Dense(A))


def make_constant_sampler(value):
    def constant_sampler(key, shape, dtype):
        del key
        return jnp.full(shape, value, dtype=dtype)

    return constant_sampler


class TestLanczosPolicy:
    """Test the LanczosPolicy concrete implementation."""

    @pytest.mark.parametrize("n_actions", [2, 4])
    def test_init_with_valid_params(self, n_actions):
        """Test initialization with valid parameters."""
        actions = LanczosPolicy(n_actions=n_actions)
        assert actions.n_actions == n_actions

    def test_init_errors_on_nonpositive_actions(self):
        """Test n_actions validation from base policy."""
        with pytest.raises(ValueError, match="n_actions must be at least 1"):
            LanczosPolicy(n_actions=0)

    @pytest.mark.parametrize("key", [jax.random.key(42)])
    def test_actions_consistency(self, psd_linear_operator, key):
        """Test that the actions are consistent."""
        actions = LanczosPolicy(n_actions=2)
        _test_batch_policy_actions_consistency(actions, psd_linear_operator, key=key)

    def test_reproducibility(
        self, psd_linear_operator, n_actions=5, key=jax.random.key(42)
    ):
        """Test that using the same key produces the same results."""
        actions1 = LanczosPolicy(n_actions=n_actions)
        actions2 = LanczosPolicy(n_actions=n_actions)

        result1 = actions1.to_actions(psd_linear_operator, key=key)
        result2 = actions2.to_actions(psd_linear_operator, key=key)

        assert jnp.array_equal(
            jnp.asarray(result1 @ jnp.eye(n_actions)),
            jnp.asarray(result2 @ jnp.eye(n_actions)),
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
        actions = LanczosPolicy(n_actions=n_actions)
        cg_vecs = actions.to_actions(psd_linear_operator, key=key)

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
        policy = LanczosPolicy(n_actions=n, grad_rtol=grad_rtol)
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
            actions = policy.to_actions(op, key=key)
            z = jnp.asarray(actions @ ((actions.T @ x) * scale))
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

    @pytest.mark.parametrize("key", [jax.random.key(42)])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("n_actions", [2, 3])
    @pytest.mark.parametrize("n", [10, 20])
    def test_init_basic(self, n_actions, n, key, dtype):
        """Test basic initialization."""
        policy = BlockSparsePolicy.from_random(
            key=key, num_datapoints=n, n_actions=n_actions, dtype=dtype
        )

        assert policy.n_actions == n_actions
        assert paramax.unwrap(policy.nz_values).shape == (n,)
        assert paramax.unwrap(policy.nz_values).dtype == dtype

    @pytest.mark.parametrize("n_actions", [1, 3])
    def test_from_random_errors_on_zero_datapoints(self, n_actions):
        """Test validation for num_datapoints."""
        with pytest.raises(ValueError, match="num_datapoints must be at least 1"):
            BlockSparsePolicy.from_random(
                key=jax.random.key(123),
                num_datapoints=0,
                n_actions=n_actions,
            )

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
        assert paramax.unwrap(policy.nz_values).dtype == dtype
        assert jnp.allclose(paramax.unwrap(policy.nz_values), nz_values)

        policy_static = BlockSparsePolicy(
            n_actions=n_actions, nz_values=paramax.non_trainable(nz_values)
        )
        assert jnp.allclose(paramax.unwrap(policy_static.nz_values), nz_values)
        assert paramax.unwrap(policy_static.nz_values).dtype == dtype

    @pytest.mark.parametrize("n_actions", [2, 3])
    def test_to_actions_consistency(
        self, psd_linear_operator, n_actions, key=jax.random.key(42)
    ):
        """Test to_actions consistency and return type."""
        n = psd_linear_operator.shape[0]
        dtype = psd_linear_operator.dtype
        policy = BlockSparsePolicy.from_random(
            key=key, num_datapoints=n, n_actions=n_actions, dtype=dtype
        )
        _test_batch_policy_actions_consistency(policy, psd_linear_operator)
        action = policy.to_actions(psd_linear_operator)
        assert isinstance(action, BlockDiagonalSparse)

    @pytest.mark.parametrize("n_actions", [2, 3])
    def test_to_actions_reproducible(
        self, psd_linear_operator, n_actions, key=jax.random.key(42)
    ):
        """Test that to_actions produces reproducible results with same values."""
        n = psd_linear_operator.shape[0]
        dtype = psd_linear_operator.dtype
        policy = BlockSparsePolicy.from_random(
            key=key, num_datapoints=n, n_actions=n_actions, dtype=dtype
        )
        actions1 = policy.to_actions(psd_linear_operator)
        actions2 = policy.to_actions(psd_linear_operator)
        assert jnp.allclose(actions1.as_matrix(), actions2.as_matrix())

    @pytest.mark.parametrize("distribution", ["normal", "rademacher", "constant"])
    def test_from_random_with_distribution_sampler(
        self,
        distribution,
        num_datapoints=12,
        n_actions=2,
        dtype=jnp.float64,
    ):
        """Test selecting built-in distributions via module-level sampler helper."""
        if distribution == "normal":
            sampler = jax.random.normal
        elif distribution == "rademacher":
            sampler = jax.random.rademacher
        elif distribution == "constant":
            sampler = make_constant_sampler(3.0)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")

        policy = BlockSparsePolicy.from_random(
            key=jax.random.key(13),
            num_datapoints=num_datapoints,
            n_actions=n_actions,
            sampler=sampler,
            dtype=dtype,
        )
        values = paramax.unwrap(policy.nz_values)
        assert values.shape == (num_datapoints,)
        assert values.dtype == dtype
        # confirm values are block-normalized
        block_size = num_datapoints // n_actions
        pad_size = num_datapoints - n_actions * block_size
        padded_values = jnp.concatenate([values, jnp.zeros(pad_size, dtype=dtype)])
        values_blocked = padded_values.reshape(n_actions, block_size)
        assert jnp.allclose(jnp.linalg.vector_norm(values_blocked, axis=1), 1.0)
        # check specific invariants
        if distribution == "rademacher":
            block_size = num_datapoints // n_actions
            scale = 1.0 / jnp.sqrt(block_size).astype(dtype)
            assert jnp.all(jnp.isin(values, jnp.array([-scale, scale], dtype=dtype)))
        elif distribution == "constant":
            assert jnp.all(values == values[0])


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

    @pytest.mark.parametrize("pseudo_input_type", [jnp.ndarray, Real])
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
        assert jnp.array_equal(paramax.unwrap(policy.train_inputs), train_inputs)
        assert isinstance(policy.pseudo_inputs, pseudo_input_type)
        assert jnp.array_equal(paramax.unwrap(policy.pseudo_inputs), pseudo_inputs)

    def test_actions_is_cross_covariance(self, inputs, kernel, dtype):
        """Test actions are the cross-covariance between the training inputs and pseudo-inputs."""
        train_inputs, pseudo_inputs = inputs
        policy = PseudoInputPolicy(pseudo_inputs, train_inputs, kernel)
        op = lazify(kernel.gram(train_inputs))
        expected = kernel.cross_covariance(train_inputs, pseudo_inputs)
        actions = policy.to_actions(op)
        assert isinstance(actions, lx.MatrixLinearOperator)
        assert actions.in_structure().dtype == expected.dtype
        assert jnp.allclose(actions.as_matrix(), expected)

    def test_actions_consistency(self, inputs, kernel):
        """Test to_actions consistency and return type."""
        train_inputs, pseudo_inputs = inputs
        policy = PseudoInputPolicy(pseudo_inputs, train_inputs, kernel)
        op = lazify(kernel.gram(train_inputs))
        _test_batch_policy_actions_consistency(policy, op)

    def test_errors_when_dataset_has_no_inputs(self):
        """Test dataset input validation."""
        kernel = gpjax.kernels.RBF(
            lengthscale=Real(jnp.ones(1)),
            variance=Real(jnp.array(1.0)),
            compute_engine=DenseKernelComputation(),
        )
        with pytest.raises(ValueError, match="Dataset must contain training inputs"):
            PseudoInputPolicy(
                pseudo_inputs=jnp.zeros((2, 1)),
                train_inputs_or_dataset=Dataset(X=None, y=jnp.zeros((2, 1))),
                kernel=kernel,
            )

    def test_errors_on_shape_mismatch(self):
        """Test pseudo-input and training-input shape validation."""
        kernel = gpjax.kernels.RBF(
            lengthscale=Real(jnp.ones(1)),
            variance=Real(jnp.array(1.0)),
            compute_engine=DenseKernelComputation(),
        )
        with pytest.raises(
            ValueError,
            match=(
                "Training inputs and pseudo-inputs must have the same trailing dimensions"
            ),
        ):
            PseudoInputPolicy(
                pseudo_inputs=jnp.zeros((3, 2)),
                train_inputs_or_dataset=jnp.zeros((5, 1)),
                kernel=kernel,
            )


class TestOrthogonalizationPolicy:
    """Test the OrthogonalizationPolicy concrete implementation."""

    @pytest.mark.parametrize("method", list(OrthogonalizationMethod))
    @pytest.mark.parametrize("n_reortho", [0, 1, 2])
    def test_init_and_properties(self, method, n_reortho, key=jax.random.key(42)):
        """Test initialization and basic properties."""
        base_policy = LanczosPolicy(n_actions=3)
        policy = OrthogonalizationPolicy(
            base_policy=base_policy, method=method, n_reortho=n_reortho
        )

        assert policy.base_policy is base_policy
        assert policy.method == method
        assert policy.n_reortho == n_reortho
        assert policy.n_actions == base_policy.n_actions

    def test_init_errors_on_negative_reorthogonalization(self):
        """Test n_reortho validation."""
        base_policy = LanczosPolicy(n_actions=2)
        with pytest.raises(ValueError, match="n_reortho must be non-negative"):
            OrthogonalizationPolicy(base_policy=base_policy, n_reortho=-1)

    @pytest.mark.parametrize(
        "method,n_reortho",
        [
            (OrthogonalizationMethod.QR, 0),
            (OrthogonalizationMethod.CGS, 1),
            (OrthogonalizationMethod.MGS, 1),
        ],
    )
    @pytest.mark.parametrize("n", [10, 20])
    @pytest.mark.parametrize("n_pseudo", [5, 10])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_wrapping_pseudoinput_with_replicates(
        self, n, n_pseudo, method, n_reortho, dtype, input_dim=1, key=jax.random.key(42)
    ):
        """Test PseudoInputPolicy with repeated pseudo-inputs (produces rank-deficient actions)."""
        n_pseudo_repeated = n_pseudo // 2
        _, subkey1, subkey2 = jax.random.split(key, 3)

        # Create inputs with replicates to induce rank deficiency
        pseudo_inputs = jax.random.normal(
            subkey1, (n_pseudo - n_pseudo_repeated, input_dim), dtype=dtype
        )
        pseudo_inputs = jnp.concatenate(
            [pseudo_inputs, pseudo_inputs[:n_pseudo_repeated, :]], axis=0
        )
        train_inputs = jax.random.normal(subkey2, (n, input_dim), dtype=dtype)

        kernel = gpjax.kernels.RBF(
            lengthscale=Real(jnp.ones(input_dim, dtype=dtype)),
            variance=Real(jnp.array(1.0, dtype=dtype)),
            compute_engine=DenseKernelComputation(),
        )
        base_policy = PseudoInputPolicy(pseudo_inputs, train_inputs, kernel)
        policy = OrthogonalizationPolicy(
            base_policy=base_policy, method=method, n_reortho=n_reortho
        )

        op = lazify(kernel.gram(train_inputs))
        _test_batch_policy_actions_consistency(policy, op)

        # Verify orthogonality is maintained despite rank deficiency
        base_actions = base_policy.to_actions(op)
        actions = policy.to_actions(op)
        base_matrix = _matrix(base_actions)
        action_matrix = _matrix(actions)
        assert action_matrix.shape == base_matrix.shape
        assert action_matrix.dtype == base_matrix.dtype
        assert not jnp.allclose(action_matrix, base_matrix)
        if dtype == jnp.float64:
            projector = action_matrix @ action_matrix.T
            assert jnp.allclose(projector @ base_matrix, base_matrix)

    def test_lanczos_passes_through(self, psd_linear_operator, key=jax.random.key(42)):
        """Test wrapping LanczosPolicy preserves orthogonality."""
        base_policy = LanczosPolicy(n_actions=3)
        policy = OrthogonalizationPolicy(base_policy=base_policy)

        base_actions = base_policy.to_actions(psd_linear_operator, key=key)
        ortho_actions = policy.to_actions(psd_linear_operator, key=key)

        assert isinstance(ortho_actions, type(base_actions))
        assert _matrix(ortho_actions).shape == _matrix(base_actions).shape
        assert jnp.array_equal(_matrix(ortho_actions), _matrix(base_actions))

    def test_block_sparse_passes_through(
        self, psd_linear_operator, key=jax.random.key(42)
    ):
        """Test wrapping BlockSparsePolicy preserves orthogonality."""
        n, dtype = psd_linear_operator.shape[0], psd_linear_operator.dtype
        base_policy = BlockSparsePolicy.from_random(
            key=key, num_datapoints=n, n_actions=3, dtype=dtype
        )
        policy = OrthogonalizationPolicy(base_policy=base_policy)

        base_actions = base_policy.to_actions(psd_linear_operator)
        ortho_actions = policy.to_actions(psd_linear_operator)

        assert isinstance(ortho_actions, BlockDiagonalSparse)
        assert ortho_actions.out_size() == base_actions.out_size()
        assert ortho_actions.in_size() == base_actions.in_size()
        assert jnp.array_equal(ortho_actions.nz_values, base_actions.nz_values)
