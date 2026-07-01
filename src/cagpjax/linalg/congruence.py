"""Congruence transformations for linear operators."""

from typing import Any

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from ..interop import to_lineax
from ..operators import BlockDiagonalSparse
from .operator_matrix import apply_operator_right


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
        return A.T @ apply_operator_right(B, A)
    if (
        isinstance(A, lx.AbstractLinearOperator)
        and isinstance(B, lx.AbstractLinearOperator)
        and lx.is_diagonal(A)
        and lx.is_diagonal(B)
    ):
        return lx.DiagonalLinearOperator(lx.diagonal(A) ** 2 * lx.diagonal(B))
    B_lx = to_lineax(B)
    if isinstance(A, BlockDiagonalSparse):
        if lx.is_diagonal(B_lx):
            diagonal_values = lx.diagonal(B_lx)
            diag = _block_diagonal_sparse_congruence_diagonal(
                A, diagonal_values * (A.nz_values**2)
            )
            return lx.DiagonalLinearOperator(diag)
    if isinstance(A, lx.AbstractLinearOperator):
        A_dense = A.as_matrix()
        return lx.MatrixLinearOperator(A_dense.T @ apply_operator_right(B_lx, A_dense))
    return A.T @ (B @ A)
