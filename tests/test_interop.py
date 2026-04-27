"""Tests for interop utility functions."""

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from cagpjax import interop

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

    def test_lazify_lineax_matrix_operator(self, n, dtype, key=jax.random.key(13)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        op = lx.MatrixLinearOperator(matrix)
        recovered = interop.lazify(op)
        assert isinstance(recovered, lx.AbstractLinearOperator)
        np.testing.assert_allclose(recovered.as_matrix(), matrix)

    def test_lazify_generic_lineax_operator(self, n, dtype, key=jax.random.key(14)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        op = lx.AddLinearOperator(
            lx.MatrixLinearOperator(matrix),
            lx.DiagonalLinearOperator(jnp.ones(n, dtype=dtype)),
        )
        recovered = interop.lazify(op)
        assert isinstance(recovered, lx.AbstractLinearOperator)
        np.testing.assert_allclose(recovered.as_matrix(), op.as_matrix())

    def test_lazify_custom_lineax_operator(self, n, dtype, key=jax.random.key(20)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        op = CustomMatrixOperator(matrix)
        recovered = interop.lazify(op)
        np.testing.assert_allclose(recovered.as_matrix(), matrix)

    def test_lazify_raw_array_fallback(self, n, dtype, key=jax.random.key(15)):
        matrix = jax.random.normal(key, (n, n), dtype=dtype)
        recovered = interop.lazify(matrix)
        np.testing.assert_allclose(recovered.as_matrix(), matrix)

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
