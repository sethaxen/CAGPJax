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
    return A.T @ (B @ A)


@overload
def congruence_transform(A: Diagonal, B: Diagonal) -> Diagonal:  # pyright: ignore[reportOverlappingOverload]
    return Diagonal(cola.linalg.diag(A) ** 2 * cola.linalg.diag(B))


@overload
def congruence_transform(A: BlockDiagonalSparse, B: Diagonal | ScalarMul) -> Diagonal:  # pyright: ignore[reportOverlappingOverload]
    nz_values = B @ A.nz_values**2

    n_blocks, n = A.shape
    block_size = n // n_blocks
    n_blocks_main = n_blocks if n % n_blocks == 0 else n_blocks - 1
    n_main = n_blocks_main * block_size

    if n_blocks_main > 0:
        diag = nz_values[:n_main].reshape(n_blocks_main, block_size).sum(axis=1)
    else:
        diag = jnp.array([], dtype=nz_values.dtype)

    if n > n_main:
        overhang_sum = nz_values[n_main:].sum(axis=0, keepdims=True)
        diag = jnp.concatenate([diag, overhang_sum])

    return Diagonal(diag)


@cola.dispatch
def congruence_transform(A: Any, B: Any) -> Any:
    """Congruence transformation ``A.T @ B @ A``.

    Args:
        A: Linear operator or array to be applied.
        B: Square linear operator or array to be transformed.
    """
    pass
