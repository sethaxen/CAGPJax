"""Congruence transformations for linear operators."""

from typing import Any

import cola
import jax.numpy as jnp
import lineax as lx
from cola.ops import Diagonal, LinearOperator, ScalarMul
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


def _congruence_block_diagonal_sparse(
    A: BlockDiagonalSparse, B: Diagonal | ScalarMul
) -> Diagonal:
    diag = _block_diagonal_sparse_congruence_diagonal(A, B @ (A.nz_values**2))
    return Diagonal(diag)


def congruence_transform(A: Any, B: Any) -> Any:
    """Congruence transformation ``A.T @ B @ A``.

    Args:
        A: Linear operator or array to be applied.
        B: Square linear operator or array to be transformed.
    """
    if isinstance(A, Diagonal) and isinstance(B, Diagonal):
        return Diagonal(cola.linalg.diag(A) ** 2 * cola.linalg.diag(B))
    if isinstance(A, BlockDiagonalSparse) and isinstance(B, (Diagonal, ScalarMul)):
        return _congruence_block_diagonal_sparse(A, B)
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
