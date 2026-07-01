"""Congruence transformations for linear operators."""

from typing import Any

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from ..operators import BlockDiagonalSparse


def _block_diagonal_sparse_congruence_diagonal(
    A: BlockDiagonalSparse, weighted_nz_values: Float[Array, "N"]
) -> Float[Array, "K"]:
    nz_values = weighted_nz_values
    n = A.out_size()
    n_blocks = A.in_size()
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

    return diag


def congruence_transform(A: Any, B: Any) -> Any:
    """Congruence transformation ``A.T @ B @ A``.

    Args:
        A: Linear operator or array to be applied.
        B: Square linear operator or array to be transformed.
    """
    if isinstance(B, lx.AbstractLinearOperator) and not isinstance(
        A, lx.AbstractLinearOperator
    ):
        # TODO: batch multiple vectors instead of materializing
        return A.T @ (B.as_matrix() @ A)
    if (
        isinstance(A, lx.AbstractLinearOperator)
        and isinstance(B, lx.AbstractLinearOperator)
        and lx.is_diagonal(A)
        and lx.is_diagonal(B)
    ):
        return lx.DiagonalLinearOperator(lx.diagonal(A) ** 2 * lx.diagonal(B))
    B_lx = B if isinstance(B, lx.AbstractLinearOperator) else lx.MatrixLinearOperator(B)
    if isinstance(A, BlockDiagonalSparse):
        if lx.is_diagonal(B_lx):
            diagonal_values = lx.diagonal(B_lx)
            diag = _block_diagonal_sparse_congruence_diagonal(
                A, diagonal_values * (A.nz_values**2)
            )
            return lx.DiagonalLinearOperator(diag)
    if isinstance(A, lx.AbstractLinearOperator):
        return A.transpose() @ (B_lx @ A)
    return A.T @ (B @ A)
