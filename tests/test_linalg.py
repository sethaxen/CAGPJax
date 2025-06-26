"""Tests for linear algebra functions."""

import cola
import jax
import jax.numpy as jnp
import pytest
from cola.ops import Diagonal, LinearOperator

from cagpjax.linalg import congruence_transform
from cagpjax.operators import BlockDiagonalSparse

jax.config.update("jax_enable_x64", True)


class TestCongruenceTransform:
    """Tests for ``congruence_transform``."""

    @pytest.mark.parametrize("A_wrap", [jnp.asarray, cola.lazify])
    @pytest.mark.parametrize("B_wrap", [jnp.asarray, cola.lazify])
    @pytest.mark.parametrize("m, n", [(3, 5), (4, 6)])
    def test_congruence_fallback(
        self, A_wrap, B_wrap, m, n, dtype=jnp.float32, key=jax.random.key(42)
    ):
        """Test fallback case where ``A`` and ``B`` can be linear operators or dense arrays."""
        key, subkey = jax.random.split(key)
        A_dense = jax.random.normal(key, (m, n), dtype=dtype)
        B_dense = jax.random.normal(subkey, (n, n), dtype=dtype)
        A = A_wrap(A_dense)
        B = B_wrap(B_dense)
        C = congruence_transform(A, B)
        assert C.shape == (m, m)
        assert C.dtype == dtype
        if isinstance(A, LinearOperator) and isinstance(B, LinearOperator):
            assert isinstance(C, LinearOperator)
            C_dense = C.to_dense()
        else:
            assert isinstance(C, jnp.ndarray)
            C_dense = C

        assert jnp.allclose(C_dense, A_dense @ B_dense @ A_dense.T)

    @pytest.mark.parametrize("n", [3, 6])
    def test_congruence_both_diagonal(
        self, n, dtype=jnp.float32, key=jax.random.key(42)
    ):
        """Test overload where both ``A`` and ``B`` are ``Diagonal``."""
        key, subkey = jax.random.split(key)
        A_diag = jax.random.normal(key, (n,), dtype=dtype)
        B_diag = jax.random.normal(subkey, (n,), dtype=dtype)
        A = Diagonal(A_diag)
        B = Diagonal(B_diag)
        C = congruence_transform(A, B)
        assert isinstance(C, Diagonal)
        C_diag = jnp.diagonal(congruence_transform(jnp.diag(A_diag), jnp.diag(B_diag)))
        assert jnp.allclose(cola.linalg.diag(C), C_diag)

    @pytest.mark.parametrize("n, n_blocks", [(7, 3), (10, 2)])
    def test_congruence_block_diagonal_sparse_diagonal(
        self, n, n_blocks, dtype=jnp.float64, key=jax.random.key(42)
    ):
        """Test overload where ``A`` is ``BlockDiagonalSparse`` and ``B`` is ``Diagonal``."""
        key, subkey = jax.random.split(key)
        A = BlockDiagonalSparse(
            jax.random.normal(subkey, (n,), dtype=dtype), n_blocks=n_blocks
        )
        B = Diagonal(jax.random.normal(subkey, (n,), dtype=dtype))
        C = congruence_transform(A, B)
        assert isinstance(C, Diagonal)
        assert C.shape == (n_blocks, n_blocks)
        assert C.dtype == dtype
        assert jnp.allclose(
            C.to_dense(), congruence_transform(A.to_dense(), B.to_dense())
        )
