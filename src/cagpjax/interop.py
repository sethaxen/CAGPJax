"""Interop utilities for normalizing operator-like inputs to Lineax."""

from typing import Any

import jax
import jax.numpy as jnp
import lineax as lx


class CompatLinearOperator(lx.AbstractLinearOperator):
    """Lineax wrapper with legacy operator conveniences."""

    operator: lx.AbstractLinearOperator

    def __init__(self, operator: lx.AbstractLinearOperator):
        self.operator = operator

    @property
    def shape(self) -> tuple[int, int]:
        return (self.out_size(), self.in_size())

    @property
    def dtype(self):
        return self.in_structure().dtype

    @property
    def T(self) -> "CompatLinearOperator":
        return self.transpose()

    def mv(self, vector):
        return self.operator.mv(vector)

    def as_matrix(self):
        return self.operator.as_matrix()

    def to_dense(self):
        return self.as_matrix()

    def transpose(self) -> "CompatLinearOperator":
        return CompatLinearOperator(self.operator.transpose())

    def in_structure(self):
        return self.operator.in_structure()

    def out_structure(self):
        return self.operator.out_structure()

    def __matmul__(self, other):
        if isinstance(other, (CompatLinearOperator, lx.AbstractLinearOperator)):
            return CompatLinearOperator(to_lineax(self) @ to_lineax(other))
        return self.as_matrix() @ other

    def __rmatmul__(self, other):
        return other @ self.as_matrix()


@lx.is_symmetric.register(CompatLinearOperator)
def _(operator: CompatLinearOperator) -> bool:
    return lx.is_symmetric(operator.operator)


@lx.is_positive_semidefinite.register(CompatLinearOperator)
def _(operator: CompatLinearOperator) -> bool:
    return lx.is_positive_semidefinite(operator.operator)


@lx.is_diagonal.register(CompatLinearOperator)
def _(operator: CompatLinearOperator) -> bool:
    return lx.is_diagonal(operator.operator)


@lx.diagonal.register(CompatLinearOperator)
def _(operator: CompatLinearOperator):
    return lx.diagonal(operator.operator)


def lazify(A: Any) -> CompatLinearOperator:
    """Convert operator-like values into a Lineax operator."""
    if isinstance(A, CompatLinearOperator):
        return A
    if isinstance(A, lx.AbstractLinearOperator):
        return CompatLinearOperator(A)
    if hasattr(A, "as_matrix"):
        return CompatLinearOperator(lx.MatrixLinearOperator(jnp.asarray(A.as_matrix())))
    if hasattr(A, "to_dense"):
        return CompatLinearOperator(lx.MatrixLinearOperator(jnp.asarray(A.to_dense())))
    return CompatLinearOperator(lx.MatrixLinearOperator(jnp.asarray(A)))


def to_lineax(A: Any) -> lx.AbstractLinearOperator:
    """Convert arrays/operator-like inputs into Lineax operators."""
    if isinstance(A, CompatLinearOperator):
        return A.operator
    if isinstance(A, lx.AbstractLinearOperator):
        return A
    if hasattr(A, "as_matrix"):
        return lx.MatrixLinearOperator(jnp.asarray(A.as_matrix()))
    if hasattr(A, "to_dense"):
        return lx.MatrixLinearOperator(jnp.asarray(A.to_dense()))
    return lx.MatrixLinearOperator(jnp.asarray(A))
