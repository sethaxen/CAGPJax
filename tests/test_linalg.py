"""Tests for linear algebra functions."""

import jax
import jax.numpy as jnp
import jax.test_util
import lineax as lx
import pytest

from cagpjax.interop import lazify, to_lineax
from cagpjax.linalg import (
    Eigh,
    Lanczos,
    OrthogonalizationMethod,
    apply_operator_left,
    apply_operator_right,
    congruence_transform,
    eigh,
    lower_cholesky,
    orthogonalize,
)
from cagpjax.linalg.eigh import EighResult
from cagpjax.linalg.utils import _add_jitter
from cagpjax.operators import BlockDiagonalSparse, diag_like

jax.config.update("jax_enable_x64", True)


class TestApplyOperatorMatrix:
    """Tests for ``apply_operator_right``/``apply_operator_left``."""

    @pytest.mark.parametrize("A_wrap", [jnp.asarray, lx.MatrixLinearOperator])
    @pytest.mark.parametrize("B_ndim", [1, 2])
    def test_apply_operator_right(
        self, A_wrap, B_ndim, m=4, n=6, dtype=jnp.float64, key=jax.random.key(42)
    ):
        key, subkey = jax.random.split(key)
        A_dense = jax.random.normal(key, (n, m), dtype=dtype)
        B_shape = (m,) if B_ndim == 1 else (m, 3)
        B = jax.random.normal(subkey, B_shape, dtype=dtype)
        A = A_wrap(A_dense)
        result = apply_operator_right(A, B)
        assert jnp.allclose(result, A_dense @ B)

    @pytest.mark.parametrize("A_wrap", [jnp.asarray, lx.MatrixLinearOperator])
    @pytest.mark.parametrize("B_ndim", [1, 2])
    def test_apply_operator_left(
        self, A_wrap, B_ndim, m=4, n=6, dtype=jnp.float64, key=jax.random.key(43)
    ):
        key, subkey = jax.random.split(key)
        A_dense = jax.random.normal(key, (n, m), dtype=dtype)
        B_shape = (n,) if B_ndim == 1 else (3, n)
        B = jax.random.normal(subkey, B_shape, dtype=dtype)
        A = A_wrap(A_dense)
        result = apply_operator_left(B, A)
        assert jnp.allclose(result, B @ A_dense)

    def test_apply_operator_right_diagonal(
        self, n=5, dtype=jnp.float64, key=jax.random.key(44)
    ):
        key, subkey = jax.random.split(key)
        diag = jax.random.normal(key, (n,), dtype=dtype)
        A = lx.DiagonalLinearOperator(diag)
        B = jax.random.normal(subkey, (n, 3), dtype=dtype)
        result = apply_operator_right(A, B)
        assert jnp.allclose(result, jnp.diag(diag) @ B)

    def test_apply_operator_right_never_materializes_large_operator(
        self, n=50_000, dtype=jnp.float64, key=jax.random.key(45)
    ):
        """A non-diagonal operator that would fail if densified must still be
        usable via ``apply_operator_right``, as long as its own ``mv`` avoids
        materialization (guards the contract that motivates this module)."""

        class _RaisingIdentity(lx.AbstractLinearOperator):
            def mv(self, vector):
                return vector

            def as_matrix(self):
                raise AssertionError("as_matrix was called")

            def transpose(self):
                return self

            def in_structure(self):
                return jax.ShapeDtypeStruct((n,), dtype)

            def out_structure(self):
                return jax.ShapeDtypeStruct((n,), dtype)

        lx.is_symmetric.register(_RaisingIdentity, lambda _: True)
        lx.is_diagonal.register(_RaisingIdentity, lambda _: False)

        A = _RaisingIdentity()
        B = jax.random.normal(key, (n, 2), dtype=dtype)
        result = apply_operator_right(A, B)
        assert jnp.allclose(result, B)


class TestCongruenceTransform:
    """Tests for ``congruence_transform``."""

    @pytest.mark.parametrize("A_wrap", [jnp.asarray, lx.MatrixLinearOperator])
    @pytest.mark.parametrize("B_wrap", [jnp.asarray, lx.MatrixLinearOperator])
    @pytest.mark.parametrize("m, n", [(3, 5), (4, 6)])
    def test_congruence_fallback(
        self, A_wrap, B_wrap, m, n, dtype=jnp.float32, key=jax.random.key(42)
    ):
        """Test fallback case where ``A`` and ``B`` can be linear operators or dense arrays."""
        key, subkey = jax.random.split(key)
        A_dense = jax.random.normal(key, (n, m), dtype=dtype)
        B_dense = jax.random.normal(subkey, (n, n), dtype=dtype)
        A = A_wrap(A_dense)
        B = B_wrap(B_dense)
        C = congruence_transform(A, B)
        with jax.default_matmul_precision("highest"):
            C_expected = A_dense.T @ (B_dense @ A_dense)
        if isinstance(A, lx.AbstractLinearOperator):
            assert isinstance(C, lx.AbstractLinearOperator)
            assert jnp.allclose(C.as_matrix(), C_expected)
        else:
            assert isinstance(C, jnp.ndarray)
            assert jnp.allclose(C, C_expected)

    @pytest.mark.parametrize("n", [3, 6])
    def test_congruence_both_diagonal(
        self, n, dtype=jnp.float32, key=jax.random.key(42)
    ):
        """Test overload where both ``A`` and ``B`` are ``Diagonal``."""
        key, subkey = jax.random.split(key)
        A_diag = jax.random.normal(key, (n,), dtype=dtype)
        B_diag = jax.random.normal(subkey, (n,), dtype=dtype)
        A = lx.DiagonalLinearOperator(A_diag)
        B = lx.DiagonalLinearOperator(B_diag)
        C = congruence_transform(A, B)
        assert isinstance(C, lx.DiagonalLinearOperator)
        C_diag = jnp.diagonal(congruence_transform(jnp.diag(A_diag), jnp.diag(B_diag)))
        assert jnp.allclose(lx.diagonal(C), C_diag)

    @pytest.mark.parametrize("n, n_blocks", [(7, 3), (10, 2)])
    def test_congruence_block_diagonal_sparse_diagonal(
        self, n, n_blocks, dtype=jnp.float64, key=jax.random.key(42)
    ):
        """Test overload where ``A`` is ``BlockDiagonalSparse`` and ``B`` is ``Diagonal``."""
        key, subkey = jax.random.split(key)
        A = BlockDiagonalSparse(
            jax.random.normal(subkey, (n,), dtype=dtype), n_blocks=n_blocks
        )
        B = lx.DiagonalLinearOperator(jax.random.normal(subkey, (n,), dtype=dtype))
        C = congruence_transform(A, B)
        assert isinstance(C, lx.DiagonalLinearOperator)
        assert C.out_size() == A.in_size()
        assert C.in_size() == A.in_size()
        assert C.out_structure().dtype == dtype
        assert jnp.allclose(
            C.as_matrix(), congruence_transform(A.as_matrix(), B.as_matrix())
        )

    @pytest.mark.parametrize("n, n_blocks", [(7, 3), (10, 2)])
    def test_congruence_block_diagonal_sparse_lineax_diagonal(
        self, n, n_blocks, dtype=jnp.float64, key=jax.random.key(123)
    ):
        """Fast path for ``BlockDiagonalSparse`` with lineax diagonal ``B``."""
        key, subkey1, subkey2 = jax.random.split(key, 3)
        A = BlockDiagonalSparse(
            jax.random.normal(subkey1, (n,), dtype=dtype), n_blocks=n_blocks
        )
        b_diag = jax.random.normal(subkey2, (n,), dtype=dtype)
        B = lx.DiagonalLinearOperator(b_diag)
        C = congruence_transform(A, B)
        assert isinstance(C, lx.DiagonalLinearOperator)
        expected = congruence_transform(A.as_matrix(), jnp.diag(b_diag))
        assert jnp.allclose(C.as_matrix(), expected)


class TestEigh:
    """Tests for ``eigh``."""

    @staticmethod
    def _matrix(op_or_arr):
        if isinstance(op_or_arr, lx.AbstractLinearOperator):
            return op_or_arr.as_matrix()
        if hasattr(op_or_arr, "to_dense"):
            return op_or_arr.to_dense()
        return op_or_arr

    @staticmethod
    def _shape(op_or_arr):
        if isinstance(op_or_arr, lx.AbstractLinearOperator):
            return (op_or_arr.out_size(), op_or_arr.in_size())
        return op_or_arr.shape

    @staticmethod
    def _dtype(op_or_arr):
        if isinstance(op_or_arr, lx.AbstractLinearOperator):
            return op_or_arr.in_structure().dtype
        return op_or_arr.dtype

    @pytest.fixture(params=[42, 78, 36])
    def key(self, request):
        return jax.random.key(request.param)

    @pytest.fixture(params=[5, 10, 30])
    def n(self, request):
        return request.param

    @pytest.fixture(params=[jnp.float32, jnp.float64])
    def dtype(self, request):
        return request.param

    @pytest.fixture(params=["dense", "diagonal", "identity"])
    def op(self, request, n, dtype, key):
        if request.param == "diagonal":
            return lx.DiagonalLinearOperator(jax.random.normal(key, (n,), dtype=dtype))
        elif request.param == "identity":
            return lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n,), dtype))
        else:
            A = jax.random.normal(key, (n, n), dtype=dtype)
            return lx.MatrixLinearOperator(A + A.T)

    @pytest.fixture(params=[Eigh, Lanczos])
    def alg(self, request):
        return request.param()

    def test_eigh(self, op, n, dtype, alg):
        """Test eigh for various operators."""
        result = eigh(op, alg=alg)
        assert isinstance(result, EighResult)
        assert isinstance(result.eigenvalues, jnp.ndarray)
        assert result.eigenvalues.shape == (n,)
        assert result.eigenvalues.dtype == dtype
        assert isinstance(result.eigenvectors, lx.AbstractLinearOperator)
        assert self._shape(result.eigenvectors) == (n, n)
        assert self._dtype(result.eigenvectors) == dtype
        if lx.is_diagonal(op):
            assert jnp.array_equal(result.eigenvalues, lx.diagonal(op))
            assert jnp.allclose(
                self._matrix(result.eigenvectors), jnp.eye(n, dtype=dtype)
            )
        else:
            with jax.default_matmul_precision("highest"):
                eigenvectors_mat = self._matrix(result.eigenvectors)
                op_mat = (
                    eigenvectors_mat @ jnp.diag(result.eigenvalues) @ eigenvectors_mat.T
                )
            if isinstance(alg, Eigh) or alg.__class__.__name__ == "Eigh":
                rtol = 1e-2 if dtype == jnp.float32 else 1e-3
                assert jnp.allclose(op_mat, self._matrix(op), rtol=rtol)
                result_jax = jax.numpy.linalg.eigh(self._matrix(op))
                assert jnp.allclose(result.eigenvalues, result_jax.eigenvalues)
                assert jnp.allclose(eigenvectors_mat, result_jax.eigenvectors)
            else:  # Lanczos
                result_jax = jax.numpy.linalg.eigh(self._matrix(op))
                assert jnp.allclose(result.eigenvalues, result_jax.eigenvalues)
                eigvecs_mul = eigenvectors_mat.T @ result_jax.eigenvectors
                rtol = 1e-2 if dtype == jnp.float32 else 1e-9
                assert jnp.allclose(
                    jnp.abs(jnp.diag(eigvecs_mul)), jnp.ones(n, dtype=dtype), rtol=rtol
                )

    @pytest.mark.parametrize("key", [None, jax.random.key(42)])
    @pytest.mark.parametrize("v0", [None, jnp.arange(5)])
    @pytest.mark.parametrize("max_iters", [None, 2, 4])
    def test_lanczos_constructor(self, max_iters, v0, key):
        """Test Lanczos constructor."""
        alg = Lanczos(max_iters, v0=v0, key=key)
        assert isinstance(alg, Lanczos)
        assert alg.max_iters == max_iters
        assert alg.v0 is v0
        assert alg.key is key

        with pytest.raises(TypeError):
            Lanczos(max_iters=max_iters, v0=v0, key=key)  # type: ignore

    @pytest.mark.parametrize("n, max_iters", [(10, 4), (20, 10)])
    @pytest.mark.parametrize("v0_type", [None, jnp.array])
    def test_lanczos_eigh_reproducible(self, n, max_iters, v0_type, key, dtype):
        """Test Lanczos eigh basic properties and reproducibility."""
        key, subkey = jax.random.split(key)
        mat = jax.random.normal(subkey, (n, n), dtype=dtype)
        op = lx.MatrixLinearOperator(mat + mat.T)

        if v0_type is None:
            v0 = None
        else:
            v0 = v0_type(jax.random.normal(key, (n,), dtype=dtype))
            key = None

        result = eigh(op, alg=Lanczos(max_iters, v0=v0, key=key))

        # verify basic properties of result
        assert isinstance(result, EighResult)
        assert result.eigenvalues.shape == (max_iters,)
        assert result.eigenvalues.dtype == dtype
        assert isinstance(result.eigenvectors, lx.AbstractLinearOperator)
        assert self._dtype(result.eigenvectors) == dtype
        assert self._shape(result.eigenvectors) == (n, max_iters)

        # verify eigenvectors are orthonormal and actually diagonalize the operator
        eigenvectors_mat = self._matrix(result.eigenvectors)
        rtol = 1e-2 if dtype == jnp.float32 else 1e-6
        assert jnp.allclose(
            eigenvectors_mat.T @ eigenvectors_mat,
            jnp.eye(max_iters, dtype=dtype),
            atol=1e-5,
        )
        with jax.default_matmul_precision("highest"):
            rayleigh = eigenvectors_mat.T @ self._matrix(op) @ eigenvectors_mat
        assert jnp.allclose(
            rayleigh, jnp.diag(result.eigenvalues), rtol=rtol, atol=1e-5
        )

        # verify that result is reproducible
        result2 = eigh(op, alg=Lanczos(max_iters, v0=v0, key=key))
        assert jnp.array_equal(result.eigenvalues, result2.eigenvalues)
        assert jnp.array_equal(
            self._matrix(result.eigenvectors), self._matrix(result2.eigenvectors)
        )

    @pytest.mark.parametrize("alg_class", [Eigh, Lanczos])
    @pytest.mark.parametrize("grad_rtol", [None, 0.0])
    def test_eigh_gradient_degenerate(self, alg_class, grad_rtol, dtype):
        """Test gradient computation with degenerate eigenvalues."""
        n = 4
        A = lx.MatrixLinearOperator(jnp.eye(n, dtype=dtype))
        if alg_class is Lanczos:
            if dtype == jnp.float32:
                pytest.skip("Lanczos gradient currently errors for float32")
        alg = alg_class()

        def loss_fn(A_dense):
            A_op = lx.MatrixLinearOperator(A_dense)
            result = eigh(A_op, alg=alg, grad_rtol=grad_rtol)
            eigenvectors = result.eigenvectors.as_matrix()
            # Reconstruct the matrix from eigendecomposition to force gradient through eigenvectors
            A_recon = eigenvectors @ jnp.diag(result.eigenvalues) @ eigenvectors.T
            return jnp.trace(A_recon)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(A.as_matrix())

        if (alg_class is Eigh) and (grad_rtol is None or grad_rtol < 0.0):
            assert not jnp.isfinite(grad).all()
        else:
            assert jnp.isfinite(grad).all()

    @pytest.mark.parametrize("grad_rtol", [None, 1e-9])
    @pytest.mark.parametrize("dtype", [jnp.float64])
    def test_eigh_gradient_degenerate_zero_grad(self, grad_rtol, dtype):
        """Test computation of almost-zero gradient with degenerate eigenvalues."""
        n = 4
        x = jnp.ones(n, dtype=dtype)

        # a small perturbation makes the gradient almost-zero but not exactly zero
        A_delta = jax.random.normal(jax.random.key(42), (n, n), dtype=dtype)
        A_delta = (A_delta + A_delta.T) * 1e-30

        def loss_fn(A_diag):
            A_op = lx.MatrixLinearOperator(jnp.diag(A_diag) + A_delta)
            result = eigh(A_op, grad_rtol=grad_rtol)
            z = result.eigenvectors.as_matrix().T @ x
            return jnp.sum(jnp.square(z))

        a_diag = jnp.ones(n, dtype=dtype)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(a_diag)
        if grad_rtol is None:
            assert not jnp.isfinite(grad).all()
        else:
            assert jnp.allclose(grad, 0.0, atol=1e-9)

    def test_eigh_errors_on_negative_grad_rtol(self):
        """Test that eigh rejects negative grad_rtol values."""
        A = lx.MatrixLinearOperator(jnp.eye(3, dtype=jnp.float64))
        with pytest.raises(ValueError, match="grad_rtol must be None or non-negative"):
            eigh(A, grad_rtol=-1.0)


class TestLowerCholesky:
    """Tests for ``lower_cholesky``."""

    @staticmethod
    def _matrix(op_or_arr):
        if isinstance(op_or_arr, lx.AbstractLinearOperator):
            return op_or_arr.as_matrix()
        if hasattr(op_or_arr, "to_dense"):
            return op_or_arr.to_dense()
        return op_or_arr

    @staticmethod
    def _diag(op):
        if isinstance(op, lx.AbstractLinearOperator) and lx.is_diagonal(op):
            return lx.diagonal(op)
        if hasattr(op, "diag"):
            return op.diag
        return jnp.diag(TestLowerCholesky._matrix(op))

    @pytest.fixture(params=[42, 78, 36])
    def key(self, request):
        return jax.random.key(request.param)

    @pytest.fixture(params=[None, 1e-3])
    def jitter(self, request):
        return request.param

    @pytest.fixture(params=[5, 10, 30])
    def n(self, request):
        return request.param

    @pytest.fixture(params=[jnp.float32, jnp.float64])
    def dtype(self, request):
        return request.param

    def test_lower_cholesky_diagonal(self, n, dtype, key, jitter):
        """Test overload where ``A`` is ``Diagonal``."""
        A = lx.DiagonalLinearOperator(jax.random.uniform(key, (n,), dtype=dtype))
        L = lower_cholesky(A, jitter)
        L_diag = self._diag(L)
        if jitter is None:
            assert jnp.allclose(L_diag**2, self._diag(A))
        else:
            assert jnp.allclose(L_diag**2, self._diag(A) + jitter)

    def test_lower_cholesky_dense(self, n, dtype, key, jitter):
        """Test overload where ``A`` is ``Dense``."""
        B = jax.random.normal(key, (n, n), dtype=dtype)
        A_mat = B @ B.T
        A = lx.MatrixLinearOperator(A_mat)
        L = lower_cholesky(A, jitter)
        L_mat = self._matrix(L)
        if jitter is None:
            assert jnp.allclose(L_mat, jnp.linalg.cholesky(A_mat))
        else:
            if dtype == jnp.float64:
                assert jnp.allclose(
                    L_mat,
                    jnp.linalg.cholesky(A_mat + jitter * jnp.eye(n, dtype=dtype)),
                )

    def test_lower_cholesky_identity(self, n, dtype, key, jitter):
        """Test overload where ``A`` is ``Identity``."""
        A = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n,), dtype))
        L = lower_cholesky(A, jitter)
        L_mat = self._matrix(L)
        if jitter is None:
            assert jnp.allclose(L_mat, jnp.eye(n, dtype=dtype))
        else:
            assert jnp.allclose(
                L_mat,
                jnp.linalg.cholesky(self._matrix(A) + jitter * jnp.eye(n, dtype=dtype)),
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
        "op_type,jitter_type",
        [
            ("dense", "scalar"),
            ("dense", "vector"),
            ("diagonal", "scalar"),
            ("diagonal", "vector"),
            ("identity", "scalar"),
            ("identity", "vector"),
        ],
    )
    def test_add_jitter(self, op_type, jitter_type, n, dtype, key):
        """Test _add_jitter with different operator and jitter types."""
        # Create operator
        if op_type == "dense":
            A_mat = jax.random.normal(key, (n, n), dtype=dtype)
            A = lx.MatrixLinearOperator(A_mat)
        elif op_type == "diagonal":
            diag_vals = jax.random.normal(key, (n,), dtype=dtype)
            A = lx.DiagonalLinearOperator(diag_vals)
        else:  # Identity
            A = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n,), dtype))

        # Create and add jitter
        if jitter_type == "scalar":
            jitter = 1e-6
        else:  # vector
            jitter = jax.random.uniform(key, (n,), dtype=dtype) * 1e-6
        A_jittered = _add_jitter(A, jitter)

        # Check output type and basic properties
        if isinstance(A_jittered, lx.AbstractLinearOperator):
            assert A_jittered.out_size() == A.out_size()
            assert A_jittered.in_size() == A.in_size()
            assert A_jittered.in_structure().dtype == A.in_structure().dtype

        # Check correctness: compare dense representations
        expected = lazify(to_lineax(A) + diag_like(A, jitter)).as_matrix()
        actual = A_jittered.as_matrix()
        assert jnp.allclose(actual, expected)

    @pytest.mark.parametrize("jitter_type", ["scalar", "vector"])
    def test_add_jitter_lineax_matrix(self, jitter_type, n, dtype, key):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        A = lx.MatrixLinearOperator(matrix)
        if jitter_type == "scalar":
            jitter = 1e-6
            expected = matrix + jnp.eye(n, dtype=dtype) * jitter
        else:
            jitter = jax.random.uniform(key, (n,), dtype=dtype) * 1e-6
            expected = matrix + jnp.diag(jitter)
        A_jittered = _add_jitter(A, jitter)
        assert isinstance(A_jittered, lx.AbstractLinearOperator)
        assert jnp.allclose(A_jittered.as_matrix(), expected)

    @pytest.mark.parametrize("jitter_type", ["scalar", "vector"])
    def test_add_jitter_lineax_diagonal_preserves_structure(
        self, jitter_type, n, dtype, key
    ):
        diag = jax.random.normal(key, (n,), dtype=dtype)
        A = lx.DiagonalLinearOperator(diag)
        if jitter_type == "scalar":
            jitter = 1e-6
            expected_diag = diag + jitter
        else:
            jitter = jax.random.uniform(key, (n,), dtype=dtype) * 1e-6
            expected_diag = diag + jitter
        A_jittered = _add_jitter(A, jitter)
        assert isinstance(A_jittered, lx.DiagonalLinearOperator)
        assert jnp.allclose(lx.diagonal(A_jittered), expected_diag)

    @pytest.mark.parametrize("jitter_type", ["scalar", "vector"])
    def test_add_jitter_lineax_identity_returns_diagonal(
        self, jitter_type, n, dtype, key
    ):
        metadata = jax.ShapeDtypeStruct((n,), dtype)
        A = lx.IdentityLinearOperator(metadata)
        if jitter_type == "scalar":
            jitter = 1e-6
            expected_diag = jnp.ones((n,), dtype=dtype) + jitter
        else:
            jitter = jax.random.uniform(key, (n,), dtype=dtype) * 1e-6
            expected_diag = jnp.ones((n,), dtype=dtype) + jitter
        A_jittered = _add_jitter(A, jitter)
        assert isinstance(A_jittered, lx.DiagonalLinearOperator)
        assert jnp.allclose(lx.diagonal(A_jittered), expected_diag)

    @pytest.mark.parametrize("jitter_type", ["scalar", "vector"])
    def test_add_jitter_lineax_tagged_diagonal_operator(
        self, jitter_type, n, dtype, key
    ):
        diag = jax.random.normal(key, (n,), dtype=dtype)
        base = lx.MatrixLinearOperator(jnp.diag(diag))
        A = lx.TaggedLinearOperator(base, lx.diagonal_tag)
        if jitter_type == "scalar":
            jitter = 1e-6
            expected_diag = diag + jitter
        else:
            jitter = jax.random.uniform(key, (n,), dtype=dtype) * 1e-6
            expected_diag = diag + jitter
        A_jittered = _add_jitter(A, jitter)
        assert isinstance(A_jittered, lx.DiagonalLinearOperator)
        assert jnp.allclose(lx.diagonal(A_jittered), expected_diag)


class TestOrthogonalize:
    """Tests for ``orthogonalize``."""

    @pytest.fixture(
        params=[
            OrthogonalizationMethod.QR,
            OrthogonalizationMethod.CGS,
            OrthogonalizationMethod.MGS,
        ]
    )
    def method(self, request):
        return request.param

    @pytest.mark.parametrize("shape", [(10, 5), (15, 15)])
    @pytest.mark.parametrize("dtype", [jnp.float64])
    @pytest.mark.parametrize("n_reortho", [0, 1])
    def test_orthogonalize_array(
        self, shape, dtype, method, n_reortho, key=jax.random.key(42)
    ):
        """Test orthogonalize with different shapes and dtypes."""
        A = jax.random.normal(key, shape, dtype=dtype)
        Q = orthogonalize(A, method=method, n_reortho=n_reortho)
        assert isinstance(Q, jnp.ndarray)
        assert Q.shape == shape
        assert Q.dtype == dtype
        QT_Q = Q.T @ Q
        # check that the result is orthogonal
        assert jnp.allclose(QT_Q, jnp.eye(shape[1], dtype=dtype))
        # check that Q has the same column space as A
        projector = Q @ Q.T
        assert jnp.allclose(projector @ A, A)

    def test_orthogonalize_errors_on_negative_n_reortho(self):
        """Test that orthogonalize errors on negative n_reortho."""
        with pytest.raises(ValueError):
            orthogonalize(jnp.eye(10), method=OrthogonalizationMethod.QR, n_reortho=-1)

    @pytest.mark.parametrize("n,m,rank", [(10, 5, 3), (15, 15, 10)])
    @pytest.mark.parametrize("dtype", [jnp.float64])
    @pytest.mark.parametrize("n_reortho", [0, 1])
    def test_orthogonalize_rank_deficient(
        self, n, m, rank, dtype, method, n_reortho, key=jax.random.key(42)
    ):
        """Test orthogonalize with different shapes and dtypes."""
        key, subkey = jax.random.split(key)
        A = jax.random.normal(subkey, (n, rank), dtype=dtype)
        B = jax.random.normal(subkey, (rank, m), dtype=dtype)
        C = A @ B
        Q = orthogonalize(C, method, n_reortho=n_reortho)
        assert isinstance(Q, jnp.ndarray)
        QT_Q = Q.T @ Q

        if n_reortho == 0 and method in [
            OrthogonalizationMethod.CGS,
            OrthogonalizationMethod.MGS,
        ]:
            pytest.xfail(
                reason=(
                    "Without reorthogonalization, Gram-Schmidt variants usually produce"
                    " non-orthogonal columns"
                )
            )

        # check that the columns are orthogonal
        assert jnp.allclose(QT_Q - jnp.diag(jnp.diag(QT_Q)), jnp.zeros_like(QT_Q))
        # check that Q's column space is (a superset of) A's column space
        projector = Q @ Q.T
        assert jnp.allclose(projector @ A, A)

    @pytest.mark.parametrize("shape", [(10, 5), (15, 15)])
    @pytest.mark.parametrize("dtype", [jnp.float64])
    @pytest.mark.parametrize("n_reortho", [0, 1])
    def test_orthogonalize_gradient(
        self, shape, dtype, method, n_reortho, key=jax.random.key(42)
    ):
        """Test orthogonalize with different shapes and dtypes."""
        A = jax.random.normal(key, shape, dtype=dtype)
        jax.test_util.check_grads(
            lambda A: jnp.asarray(
                orthogonalize(jnp.asarray(A), method=method, n_reortho=n_reortho)
            ),
            (A,),
            order=1,
        )

    # test operator overloads
    @pytest.mark.parametrize("n", [10])
    @pytest.mark.parametrize("dtype", [jnp.float64])
    @pytest.mark.parametrize(
        "op_type", ["dense", "diagonal", "identity", BlockDiagonalSparse]
    )
    def test_orthogonalize_operator(
        self, n, op_type, dtype, method, key=jax.random.key(42)
    ):
        """Test orthogonalize a LinearOperator."""
        if op_type == "dense":
            op = lx.MatrixLinearOperator(jax.random.normal(key, (n, n), dtype=dtype))
        elif op_type == "diagonal":
            op = lx.DiagonalLinearOperator(jax.random.normal(key, (n,), dtype=dtype))
        elif op_type == "identity":
            op = lx.IdentityLinearOperator(
                jax.ShapeDtypeStruct((n // 2,), dtype),
                jax.ShapeDtypeStruct((n,), dtype),
            )
        elif op_type is BlockDiagonalSparse:
            op = BlockDiagonalSparse(
                jax.random.normal(key, (n,), dtype=dtype), n_blocks=3
            )
        else:
            raise ValueError(f"Unknown operator type: {op_type}")
        Q = orthogonalize(op, method=method)
        if op_type is BlockDiagonalSparse:
            assert isinstance(Q, BlockDiagonalSparse)
            assert Q.out_size() == op.out_size()
            assert Q.in_size() == op.in_size()
            assert jnp.array_equal(Q.nz_values, op.nz_values)
        elif op_type == "diagonal":
            # Diagonal columns have disjoint support, so are already orthogonal
            # regardless of magnitude.
            assert isinstance(Q, lx.AbstractLinearOperator) and lx.is_diagonal(Q)
            assert jnp.array_equal(lx.diagonal(Q), lx.diagonal(op))
        else:
            Q_dense = Q.as_matrix() if isinstance(Q, lx.AbstractLinearOperator) else Q
            op_dense = (
                op.as_matrix() if isinstance(op, lx.AbstractLinearOperator) else op
            )
            assert Q_dense.shape == op_dense.shape
            assert jnp.allclose(
                Q_dense,
                jnp.asarray(orthogonalize(op_dense, method=method)),
            )
