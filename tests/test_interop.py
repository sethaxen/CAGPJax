"""Tests for interop utility functions."""

import cola
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from cagpjax import interop
from cagpjax.interop import ColaLinearOperator

jax.config.update("jax_enable_x64", True)


class CustomMatrixOperator(lx.AbstractLinearOperator):
    matrix: jax.Array

    def __init__(self, matrix: jax.Array):
        self.matrix = matrix

    def mv(self, vector):
        return self.matrix @ vector

    def as_matrix(self):
        return self.matrix

    def transpose(self):
        return CustomMatrixOperator(self.matrix.T)

    def in_structure(self):
        return jax.ShapeDtypeStruct((self.matrix.shape[1],), self.matrix.dtype)

    def out_structure(self):
        return jax.ShapeDtypeStruct((self.matrix.shape[0],), self.matrix.dtype)


@lx.is_symmetric.register(CustomMatrixOperator)
def _is_symmetric_custom(_operator: CustomMatrixOperator) -> bool:
    return False


@lx.is_diagonal.register(CustomMatrixOperator)
def _is_diagonal_custom(_operator: CustomMatrixOperator) -> bool:
    return False


@lx.is_tridiagonal.register(CustomMatrixOperator)
def _is_tridiagonal_custom(_operator: CustomMatrixOperator) -> bool:
    return False


@lx.is_lower_triangular.register(CustomMatrixOperator)
def _is_lower_triangular_custom(_operator: CustomMatrixOperator) -> bool:
    return False


@lx.is_upper_triangular.register(CustomMatrixOperator)
def _is_upper_triangular_custom(_operator: CustomMatrixOperator) -> bool:
    return False


@lx.is_positive_semidefinite.register(CustomMatrixOperator)
def _is_psd_custom(_operator: CustomMatrixOperator) -> bool:
    return False


class TestInteropUtils:
    @pytest.fixture(params=[jnp.float32, jnp.float64])
    def dtype(self, request):
        return request.param

    @pytest.fixture(params=[4, 9])
    def n(self, request):
        return request.param

    def test_lazify_unwraps_cola_lineax_operator(self, n, dtype, key=jax.random.key(0)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        cola_op = cola.lazify(matrix)
        wrapped = ColaLinearOperator(cola_op)
        recovered = interop.lazify(wrapped)
        assert recovered is cola_op

    def test_lazify_preserves_psd_tag(self, n, dtype, key=jax.random.key(1)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        psd_cola = cola.PSD(cola.lazify(matrix @ matrix.T))
        wrapped = lx.TaggedLinearOperator(
            ColaLinearOperator(psd_cola), lx.positive_semidefinite_tag
        )
        recovered = interop.lazify(wrapped)
        assert recovered.isa(cola.PSD)
        np.testing.assert_allclose(cola.densify(recovered), cola.densify(psd_cola))

    def test_lazify_tagged_non_psd_operator(self, n, dtype, key=jax.random.key(11)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        cola_op = cola.lazify(matrix)
        tagged = lx.TaggedLinearOperator(ColaLinearOperator(cola_op), lx.symmetric_tag)
        recovered = interop.lazify(tagged)
        assert not recovered.isa(cola.PSD)
        np.testing.assert_allclose(cola.densify(recovered), matrix)

    def test_lazify_existing_cola_operator_passthrough(
        self, n, dtype, key=jax.random.key(12)
    ):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        cola_op = cola.lazify(matrix)
        recovered = interop.lazify(cola_op)
        assert recovered is cola_op

    def test_lazify_lineax_matrix_operator(self, n, dtype, key=jax.random.key(13)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        op = lx.MatrixLinearOperator(matrix)
        recovered = interop.lazify(op)
        np.testing.assert_allclose(cola.densify(recovered), matrix)

    def test_lazify_generic_lineax_operator(self, n, dtype, key=jax.random.key(14)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        op = lx.AddLinearOperator(
            lx.MatrixLinearOperator(matrix),
            lx.DiagonalLinearOperator(jnp.ones(n, dtype=dtype)),
        )
        recovered = interop.lazify(op)
        np.testing.assert_allclose(cola.densify(recovered), op.as_matrix())

    def test_lazify_custom_lineax_operator(self, n, dtype, key=jax.random.key(20)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        op = CustomMatrixOperator(matrix)
        recovered = interop.lazify(op)
        np.testing.assert_allclose(cola.densify(recovered), matrix)

    def test_lazify_raw_array_fallback(self, n, dtype, key=jax.random.key(15)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        recovered = interop.lazify(matrix)
        np.testing.assert_allclose(cola.densify(recovered), matrix)

    def test_cola_linear_operator_interface(self, n, dtype, key=jax.random.key(18)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        wrapped = ColaLinearOperator(cola.lazify(matrix))
        vector = jax.random.normal(key, (n,), dtype=dtype)
        rtol, atol = (1e-5, 1e-6) if dtype == jnp.float32 else (1e-7, 0)
        np.testing.assert_allclose(
            wrapped.mv(vector), matrix @ vector, rtol=rtol, atol=atol
        )
        np.testing.assert_allclose(wrapped.as_matrix(), matrix, rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            wrapped.transpose().as_matrix(), matrix.T, rtol=rtol, atol=atol
        )

    def test_to_lineax_roundtrip_dense(self, n, dtype, key=jax.random.key(2)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        cola_op = cola.lazify(matrix)
        lineax_op = interop.to_lineax(cola_op)
        back_to_cola = interop.lazify(lineax_op)
        np.testing.assert_allclose(cola.densify(back_to_cola), matrix)
        assert isinstance(lineax_op, ColaLinearOperator)

    def test_to_lineax_diagonal_and_identity(self, n, dtype, key=jax.random.key(3)):
        diag = jax.random.normal(key, (n,), dtype=dtype)
        diag_op = cola.ops.Diagonal(diag)
        id_op = cola.ops.Identity((n, n), dtype)

        lineax_diag = interop.to_lineax(diag_op)
        lineax_id = interop.to_lineax(id_op)

        assert isinstance(lineax_diag, lx.DiagonalLinearOperator)
        assert isinstance(lineax_id, lx.IdentityLinearOperator)
        np.testing.assert_allclose(
            interop.lazify(lineax_diag).to_dense(), jnp.diag(diag)
        )
        np.testing.assert_allclose(interop.lazify(lineax_id).to_dense(), jnp.eye(n))

    def test_to_lineax_passthrough_existing_lineax_operator(
        self, n, dtype, key=jax.random.key(16)
    ):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        op = lx.MatrixLinearOperator(matrix)
        converted = interop.to_lineax(op)
        assert converted is op

    def test_to_lineax_passthrough_custom_lineax_operator(
        self, n, dtype, key=jax.random.key(21)
    ):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        op = CustomMatrixOperator(matrix)
        converted = interop.to_lineax(op)
        assert converted is op

    def test_to_lineax_raw_array_fallback(self, n, dtype, key=jax.random.key(17)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        converted = interop.to_lineax(matrix)
        assert isinstance(converted, lx.MatrixLinearOperator)
        np.testing.assert_allclose(converted.as_matrix(), matrix)

    def test_lineax_psd_predicate_for_cola_wrapper(
        self, n, dtype, key=jax.random.key(19)
    ):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        wrapped_psd = ColaLinearOperator(cola.PSD(cola.lazify(matrix @ matrix.T)))
        wrapped_non_psd = ColaLinearOperator(cola.lazify(matrix))
        assert lx.is_positive_semidefinite(wrapped_psd)
        assert not lx.is_positive_semidefinite(wrapped_non_psd)
