"""Congruence transformations for linear operators."""

from typing import Any

import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from ..interop import to_lineax
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
    A_lx = (
        to_lineax(A)
        if (isinstance(A, lx.AbstractLinearOperator) or hasattr(A, "to_dense"))
        else None
    )
    B_lx = (
        to_lineax(B)
        if (isinstance(B, lx.AbstractLinearOperator) or hasattr(B, "to_dense"))
        else None
    )
    if (
        A_lx is not None
        and B_lx is not None
        and lx.is_diagonal(A_lx)
        and lx.is_diagonal(B_lx)
    ):
        return lx.DiagonalLinearOperator(lx.diagonal(A_lx) ** 2 * lx.diagonal(B_lx))
    if isinstance(A, BlockDiagonalSparse):
        B_lx = to_lineax(B)
        if lx.is_diagonal(B_lx):
            diagonal_values = lx.diagonal(B_lx)
            diag = _block_diagonal_sparse_congruence_diagonal(
                A, diagonal_values * (A.nz_values**2)
            )
            return lx.DiagonalLinearOperator(diag)
    if isinstance(A, lx.AbstractLinearOperator):
        B_lx = to_lineax(B)
        return A.transpose() @ (B_lx @ A)
    if isinstance(B, lx.AbstractLinearOperator):
        A_arr = jnp.asarray(A)
        return A_arr.T @ (B.as_matrix() @ A_arr)
    return A.T @ (B @ A)
