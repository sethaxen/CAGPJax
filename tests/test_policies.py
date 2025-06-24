"""Test the linear solver policies."""

import cola
import jax
import jax.numpy as jnp
import pytest
from cola.ops import Dense, LinearOperator

from cagpjax.policies import LanczosPolicy

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

    @pytest.mark.parametrize("n_actions", [2, 4])
    def test_eigenvectors_match_dense_computation(
        self, psd_linear_operator, n_actions, key=jax.random.key(42)
    ):
        """Test that the eigenvectors match those from dense eigendecomposition."""

        # Get eigenvectors using LanczosPolicy
        actions = LanczosPolicy(n_actions=n_actions, key=key)
        cg_vecs = actions.to_actions(psd_linear_operator)

        # Get reference eigenvectors using dense computation
        _, eigenvecs = jnp.linalg.eigh(psd_linear_operator.to_dense())
        # Get the largest n_actions eigenvectors (eigh returns them in ascending order)
        ref_vecs = eigenvecs[:, -n_actions:]

        # Convert LanczosPolicy result to dense array for comparison
        cg_vecs_dense = cg_vecs @ jnp.eye(n_actions)

        # Compare eigenvectors (they should match up to sign)
        # Computing the product should give a diagonal matrix with +/-1 entries
        product = cg_vecs_dense.T @ ref_vecs
        abs_product = jnp.abs(product)
        expected_identity = jnp.eye(n_actions)

        atol = 1e-6 if psd_linear_operator.dtype == jnp.float64 else 1e-3
        assert jnp.allclose(abs_product, expected_identity, atol=atol), (
            "CG eigenvectors don't match reference eigenvectors (up to sign)"
        )
