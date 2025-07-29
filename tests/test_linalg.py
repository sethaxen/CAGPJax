"""Tests for linear algebra functions."""

import cola
import jax
import jax.numpy as jnp
import pytest
from cola.ops import Dense, Diagonal, Identity, LinearOperator, ScalarMul, Triangular

from cagpjax.linalg import Eigh, congruence_transform, eigh, lower_cholesky
from cagpjax.linalg.eigh import EighResult
from cagpjax.linalg.utils import _add_jitter
from cagpjax.operators import BlockDiagonalSparse, diag_like

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


class TestEigh:
    """Tests for ``eigh``."""

    @pytest.fixture(params=[42, 78, 36])
    def key(self, request):
        return jax.random.key(request.param)

    @pytest.fixture(params=[5, 10, 30])
    def n(self, request):
        return request.param

    @pytest.fixture(params=[jnp.float32, jnp.float64])
    def dtype(self, request):
        return request.param

    @pytest.fixture(
        params=[
            cola.ops.Dense,
            cola.ops.Diagonal,
            cola.ops.ScalarMul,
            cola.ops.Identity,
        ]
    )
    def op(self, request, n, dtype, key):
        if request.param is Diagonal:
            return Diagonal(jax.random.normal(key, (n,), dtype=dtype))
        elif request.param is cola.ops.ScalarMul:
            return cola.ops.ScalarMul(
                jax.random.normal(key, dtype=dtype), (n, n), dtype=dtype
            )
        elif request.param is cola.ops.Identity:
            return cola.ops.Identity((n, n), dtype=dtype)
        else:
            A = jax.random.normal(key, (n, n), dtype=dtype)
            return cola.ops.Dense(A + A.T)

    @pytest.fixture(params=[Eigh])
    def alg(self, request):
        return request.param()

    def test_eigh(self, op, n, dtype, alg):
        """Test eigh for various operators."""
        result = eigh(op, alg=alg)
        assert isinstance(result, EighResult)
        assert isinstance(result.eigenvalues, jnp.ndarray)
        assert result.eigenvalues.shape == (n,)
        assert result.eigenvalues.dtype == dtype
        assert isinstance(result.eigenvectors, cola.ops.LinearOperator)
        assert result.eigenvectors.isa(cola.Unitary)
        assert result.eigenvectors.shape == (n, n)
        assert result.eigenvectors.dtype == dtype
        if isinstance(op, (cola.ops.Diagonal, cola.ops.ScalarMul, cola.ops.Identity)):
            assert jnp.array_equal(result.eigenvalues, cola.linalg.diag(op))
            assert isinstance(result.eigenvectors, cola.ops.Identity)
        else:
            with jax.default_matmul_precision("highest"):
                op_mat = (
                    result.eigenvectors
                    @ jnp.diag(result.eigenvalues)
                    @ result.eigenvectors.T
                )
            rtol = 1e-2 if dtype == jnp.float32 else 0.0
            assert jnp.allclose(op_mat, op.to_dense(), rtol=rtol)
            if isinstance(alg, (Eigh,)):
                result_jax = jax.numpy.linalg.eigh(op.to_dense())
                assert jnp.allclose(result.eigenvalues, result_jax.eigenvalues)
                assert jnp.allclose(
                    result.eigenvectors.to_dense(), result_jax.eigenvectors
                )

    @pytest.mark.parametrize("grad_rtol", [None, -1.0, 0.0])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_eigh_gradient_degenerate(self, grad_rtol, dtype):
        """Test gradient computation with degenerate eigenvalues.

        Without jitter, gradients contain NaN. With jitter, gradients are finite.
        See https://github.com/jax-ml/jax/issues/669
        """
        n = 4
        A = cola.ops.Dense(jnp.eye(n, dtype=dtype))

        def loss_fn(A_dense):
            A_op = cola.ops.Dense(A_dense)
            result = eigh(A_op, grad_rtol=grad_rtol)
            # Reconstruct the matrix from eigendecomposition to force gradient through eigenvectors
            A_recon = (
                result.eigenvectors
                @ jnp.diag(result.eigenvalues)
                @ result.eigenvectors.T
            )
            return jnp.trace(A_recon)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(A.to_dense())

        if grad_rtol is None or grad_rtol < 0.0:
            assert not jnp.isfinite(grad).all()
        else:
            assert jnp.isfinite(grad).all()


class TestLowerCholesky:
    """Tests for ``lower_cholesky``."""

    @pytest.fixture(params=[42, 78, 36])
    def key(self, request):
        return jax.random.key(request.param)

    @pytest.fixture(params=[None, 1e-3])
    def jitter(self, request):
        return request.param

    @pytest.mark.parametrize("n", [5, 10, 30])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("jitter", [None, 1e-3])
    def test_lower_cholesky_diagonal(self, n, dtype, key, jitter):
        """Test overload where ``A`` is ``Diagonal``."""
        A = Diagonal(jax.random.uniform(key, (n,), dtype=dtype))
        L = lower_cholesky(A, jitter)
        assert isinstance(L, Diagonal)
        assert L.shape == (n, n)
        assert L.dtype == dtype
        L_diag = L.diag
        if jitter is None:
            assert jnp.allclose(L_diag**2, A.diag)
        else:
            assert jnp.allclose(L_diag**2, A.diag + jitter)

    @pytest.mark.parametrize("n", [5, 10, 30])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    @pytest.mark.parametrize("jitter", [None, 1e-3])
    def test_lower_cholesky_dense(self, n, dtype, key, jitter):
        """Test overload where ``A`` is ``Dense``."""
        B = jax.random.normal(key, (n, n), dtype=dtype)
        A_mat = B @ B.T
        A = Dense(A_mat)
        L = lower_cholesky(A, jitter)
        assert isinstance(L, Triangular)
        assert L.shape == (n, n)
        assert L.dtype == dtype
        if jitter is None:
            assert jnp.allclose(L.to_dense(), jnp.linalg.cholesky(A_mat))
        else:
            if dtype == jnp.float64:
                assert jnp.allclose(
                    L.to_dense(),
                    jnp.linalg.cholesky(A_mat + jitter * jnp.eye(n, dtype=dtype)),
                )


class TestAddJitter:
    """Tests for ``_add_jitter`` utility function."""

    @pytest.fixture(params=[42, 78])
    def key(self, request):
        return jax.random.key(request.param)

    @pytest.fixture(params=[4, 10])
    def n(self, request):
        return request.param

    @pytest.fixture(params=[jnp.float32, jnp.float64])
    def dtype(self, request):
        return request.param

    @pytest.mark.parametrize(
        "op_type,jitter_type,out_type",
        [
            (Dense, "scalar", LinearOperator),
            (Dense, "vector", LinearOperator),
            (Diagonal, "scalar", Diagonal),
            (Diagonal, "vector", Diagonal),
            (ScalarMul, "scalar", ScalarMul),
            (ScalarMul, "vector", Diagonal),
            (Identity, "scalar", ScalarMul),
            (Identity, "vector", Diagonal),
        ],
    )
    def test_add_jitter(self, op_type, jitter_type, out_type, n, dtype, key):
        """Test _add_jitter with different operator and jitter types."""
        # Create operator
        device = jax.devices()[0]
        if op_type == Dense:
            A_mat = jax.random.normal(key, (n, n), dtype=dtype)
            A = cola.ops.Dense(A_mat)
        elif op_type == Diagonal:
            diag_vals = jax.random.normal(key, (n,), dtype=dtype)
            A = cola.ops.Diagonal(diag_vals)
        elif op_type == ScalarMul:
            scalar = 2.5
            A = cola.ops.ScalarMul(scalar, (n, n), dtype=dtype, device=device)
        else:  # Identity
            A = cola.ops.Identity((n, n), dtype=dtype)

        # Create and add jitter
        if jitter_type == "scalar":
            jitter = 1e-6
        else:  # vector
            jitter = jax.random.uniform(key, (n,), dtype=dtype) * 1e-6
        A_jittered = _add_jitter(A, jitter)

        # Check output type and basic properties
        assert isinstance(A_jittered, out_type)
        assert A_jittered.shape == A.shape
        assert A_jittered.dtype == A.dtype

        # Check correctness: compare dense representations
        expected = (A + diag_like(A, jitter)).to_dense()
        assert jnp.allclose(A_jittered.to_dense(), expected)
