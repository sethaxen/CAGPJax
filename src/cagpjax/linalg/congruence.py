"""Congruence transformations for linear operators."""

from typing import Any

import cola
import jax.numpy as jnp
from cola.ops import Diagonal, LinearOperator, ScalarMul
from jaxtyping import Array, Float
from typing_extensions import overload

from ..operators import BlockDiagonalSparse


# fallback to plain multiplication
@overload
def congruence_transform(A: Any, B: Any) -> Any:
    return (A @ B) @ A.T


@overload
def congruence_transform(A: Diagonal, B: Diagonal) -> Diagonal:  # pyright: ignore[reportOverlappingOverload]
    return Diagonal(cola.linalg.diag(A) ** 2 * cola.linalg.diag(B))


@overload
def congruence_transform(A: BlockDiagonalSparse, B: Diagonal) -> Diagonal:  # pyright: ignore[reportOverlappingOverload]
    n_used = A.nz_values.size
    diag_values = cola.linalg.diag(B)[:n_used].reshape(A.nz_values.shape)
    diag = jnp.sum(diag_values * jnp.square(A.nz_values), axis=1)
    return Diagonal(diag)


@overload
def congruence_transform(A: BlockDiagonalSparse, B: ScalarMul) -> Diagonal:  # pyright: ignore[reportOverlappingOverload]
    diag = jnp.sum(jnp.square(A.nz_values), axis=1) * B.c
    return Diagonal(diag)


@cola.dispatch
def congruence_transform(A: Any, B: Any) -> Any:
    """Congruence transformation ``A @ B @ A.T``.

    Args:
        A: Linear operator or array to be applied.
        B: Square linear operator or array to be transformed.
    """
    pass
