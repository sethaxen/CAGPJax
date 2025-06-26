"""Congruence transformations for linear operators."""

import math
from typing import Any

import cola
import jax.numpy as jnp
from cola.ops import Diagonal, LinearOperator, ScalarMul
from jaxtyping import Array, Float
from typing_extensions import overload

from ..operators import BlockDiagonalSparse


# fallback to plain multiplication
@overload
def congruence_transform(
    A: LinearOperator | Float[Array, "M N"],
    B: LinearOperator | Float[Array, "N N"],
):
    return (A @ B) @ A.T


@overload
def congruence_transform(A: Diagonal, B: Diagonal) -> Diagonal:  # pyright: ignore[reportOverlappingOverload]
    return Diagonal(cola.linalg.diag(A) ** 2 * cola.linalg.diag(B))


@overload
def congruence_transform(A: BlockDiagonalSparse, B: Diagonal | ScalarMul) -> Diagonal:  # pyright: ignore[reportOverlappingOverload]
    nz_values = B @ A.nz_values**2

    n_blocks, n = A.shape
    block_size = math.ceil(n / n_blocks)
    n_blocks_main = n // block_size
    n_main = n_blocks_main * block_size
    diag = nz_values[:n_main].reshape(n_blocks_main, block_size).sum(axis=1)
    if n > n_main:
        diag = jnp.concatenate([diag, nz_values[n_main:].sum(axis=0, keepdims=True)])

    return Diagonal(diag)


@cola.dispatch
def congruence_transform(A: Any, B: Any) -> Any:
    """Congruence transformation ``A @ B @ A.T``.

    Args:
        A: Linear operator or array to be applied.
        B: Square linear operator or array to be transformed.
    """
    pass
