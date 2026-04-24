"""Interop utilities between cola and lineax operators."""

from typing import Any

import cola
import cola.ops
import jax
import jax.numpy as jnp
import lineax as lx
from cola.ops import LinearOperator
from jaxtyping import Array, Float, PyTree


class ColaLinearOperator(lx.AbstractLinearOperator):
    """Wrap a cola operator with a Lineax-compatible interface."""

    operator: LinearOperator

    def __init__(self, operator: LinearOperator):
        self.operator = operator

    def mv(self, vector: Float[Array, " N"]) -> Float[Array, " M"]:
        return self.operator @ vector

    def as_matrix(self) -> Float[Array, "M N"]:
        return cola.densify(self.operator)

    def transpose(self) -> "ColaLinearOperator":
        return ColaLinearOperator(self.operator.T)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.ShapeDtypeStruct((self.operator.shape[1],), self.operator.dtype)

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.ShapeDtypeStruct((self.operator.shape[0],), self.operator.dtype)


@lx.is_symmetric.register(ColaLinearOperator)
def _(operator: ColaLinearOperator) -> bool:
    return operator.operator.isa(cola.PSD)


@lx.is_positive_semidefinite.register(ColaLinearOperator)
def _(operator: ColaLinearOperator) -> bool:
    return operator.operator.isa(cola.PSD)


@lx.diagonal.register(ColaLinearOperator)
def _(operator: ColaLinearOperator) -> Float[Array, " N"]:
    return jnp.asarray(cola.linalg.diag(operator.operator))


def lazify(A: Any) -> LinearOperator:
    """Convert GPJax/Lineax/array inputs into cola operators."""
    if isinstance(A, LinearOperator):
        return A
    if isinstance(A, ColaLinearOperator):
        return A.operator
    if isinstance(A, lx.TaggedLinearOperator):
        op = lazify(A.operator)
        if lx.positive_semidefinite_tag in A.tags:
            return cola.PSD(op)
        return op
    if isinstance(A, lx.MatrixLinearOperator):
        return cola.ops.Dense(A.matrix)
    if isinstance(A, lx.DiagonalLinearOperator):
        return cola.ops.Diagonal(A.diagonal)
    if isinstance(A, lx.IdentityLinearOperator):
        metadata = jax.eval_shape(A.as_matrix)
        return cola.ops.Identity(metadata.shape, metadata.dtype)
    if isinstance(A, lx.AbstractLinearOperator):
        return cola.lazify(A.as_matrix())
    return cola.lazify(A)


def to_lineax(A: Any) -> lx.AbstractLinearOperator:
    """Convert existing operators into Lineax only where GPJax requires it."""
    if isinstance(A, lx.AbstractLinearOperator):
        return A
    if isinstance(A, cola.ops.Diagonal):
        return lx.DiagonalLinearOperator(A.diag)
    if isinstance(A, cola.ops.Identity):
        metadata = jax.ShapeDtypeStruct((A.shape[1],), A.dtype)
        return lx.IdentityLinearOperator(metadata)
    if isinstance(A, cola.ops.LinearOperator):
        return ColaLinearOperator(A)
    return lx.MatrixLinearOperator(A)
