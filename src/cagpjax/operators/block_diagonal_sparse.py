"""Block-diagonal sparse linear operator."""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from .annotations import ScaledOrthogonal


class _BlockDiagonalSparseTranspose(lx.AbstractLinearOperator):
    """Transpose operator for :class:`BlockDiagonalSparse`."""

    parent: "BlockDiagonalSparse"

    def __init__(self, parent: "BlockDiagonalSparse"):
        self.parent = parent

    def mv(self, vector: Float[Array, "N"]) -> Float[Array, "K"]:
        return self.parent._rmatmat(vector[None, :]).squeeze(axis=0)

    def as_matrix(self) -> Float[Array, "K N"]:
        return self.parent.as_matrix().T

    def transpose(self) -> "BlockDiagonalSparse":
        return self.parent

    def in_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.parent.nz_values.shape[0],), dtype=self.parent.nz_values.dtype
        )

    def out_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.parent.n_blocks,), dtype=self.parent.nz_values.dtype
        )


class BlockDiagonalSparse(lx.AbstractLinearOperator):
    """Block-diagonal sparse linear operator.

    This operator represents a block-diagonal matrix structure where the blocks are contiguous, and
    each contains a column vector, so that exactly one value is non-zero in each row.

    Args:
        nz_values: Non-zero values to be distributed across diagonal blocks.
        n_blocks: Number of diagonal blocks in the matrix.

    Examples
    --------
    ```python
    >>> import jax.numpy as jnp
    >>> from cagpjax.operators import BlockDiagonalSparse
    >>>
    >>> # Create a 3x6 block-diagonal matrix with 3 blocks
    >>> nz_values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> op = BlockDiagonalSparse(nz_values, n_blocks=3)
    >>> print(op.shape)
    (6, 3)
    >>>
    >>> # Apply to identity matrices
    >>> op @ jnp.eye(3)
    Array([[1., 0., 0.],
           [2., 0., 0.],
           [0., 3., 0.],
           [0., 4., 0.],
           [0., 0., 5.],
           [0., 0., 6.]], dtype=float32)
    ```
    """

    nz_values: Float[Array, "N"]
    n_blocks: int = eqx.field(static=True)
    annotations: frozenset[type] = eqx.field(static=True)

    def __init__(self, nz_values: Float[Array, "N"], n_blocks: int):
        self.nz_values = nz_values
        self.n_blocks = n_blocks
        self.annotations = frozenset({ScaledOrthogonal})

    def isa(self, annotation: type) -> bool:
        return annotation in self.annotations

    def _matmat(self, X: Float[Array, "K M"]) -> Float[Array, "N M"]:
        n = self.nz_values.shape[0]
        n_blocks = self.n_blocks
        block_size = n // n_blocks
        n_blocks_main = n_blocks if n % n_blocks == 0 else n_blocks - 1
        n_main = n_blocks_main * block_size
        m = X.shape[1]

        # block-wise multiplication for main blocks
        blocks_main = self.nz_values[:n_main].reshape(n_blocks_main, block_size)
        X_main = X[:n_blocks_main, :]
        res_main = (blocks_main[..., None] * X_main[:, None, :]).reshape(n_main, m)

        # handle overhang if any
        if n > n_main:
            n_overhang = n - n_main
            X_overhang = X[n_blocks_main, :]
            block_overhang = self.nz_values[n_main:]
            res_overhang = jnp.outer(block_overhang, X_overhang).reshape(n_overhang, m)
            res = jnp.concatenate([res_main, res_overhang], axis=0)
        else:
            res = res_main

        return res

    def _rmatmat(self, X: Float[Array, "M N"]) -> Float[Array, "M K"]:
        # figure out size of main blocks
        n = self.nz_values.shape[0]
        n_blocks = self.n_blocks
        block_size = n // n_blocks
        n_blocks_main = n_blocks if n % n_blocks == 0 else n_blocks - 1
        n_main = n_blocks_main * block_size
        m = X.shape[0]

        # block-wise multiplication for main blocks
        blocks_main = self.nz_values[:n_main].reshape(n_blocks_main, block_size)
        X_main = X[:, :n_main].reshape(m, n_blocks_main, block_size)
        res_main = jnp.einsum("ik,jik->ji", blocks_main, X_main)

        # handle overhang if any
        if n > n_main:
            n_overhang = n - n_main
            X_overhang = X[:, n_main:].reshape(m, n_overhang)
            block_overhang = self.nz_values[n_main:]
            res_overhang = (X_overhang @ block_overhang)[:, None]
            res = jnp.concatenate([res_main, res_overhang], axis=1)
        else:
            res = res_main

        return res

    def mv(self, vector: Float[Array, "K"]) -> Float[Array, "N"]:
        return self._matmat(vector[:, None]).squeeze(axis=1)

    def as_matrix(self) -> Float[Array, "N K"]:
        n = self.nz_values.shape[0]
        n_blocks = self.n_blocks
        return self._matmat(jnp.eye(n_blocks, dtype=self.nz_values.dtype)).reshape(
            n, n_blocks
        )

    def transpose(self) -> _BlockDiagonalSparseTranspose:
        return _BlockDiagonalSparseTranspose(self)

    def in_structure(self):
        return jax.ShapeDtypeStruct(shape=(self.n_blocks,), dtype=self.nz_values.dtype)

    def out_structure(self):
        return jax.ShapeDtypeStruct(
            shape=(self.nz_values.shape[0],), dtype=self.nz_values.dtype
        )


@lx.is_symmetric.register(BlockDiagonalSparse)
@lx.is_symmetric.register(_BlockDiagonalSparseTranspose)
def _(_operator) -> bool:
    return False


@lx.is_diagonal.register(BlockDiagonalSparse)
@lx.is_diagonal.register(_BlockDiagonalSparseTranspose)
def _(_operator) -> bool:
    return False


@lx.is_tridiagonal.register(BlockDiagonalSparse)
@lx.is_tridiagonal.register(_BlockDiagonalSparseTranspose)
def _(_operator) -> bool:
    return False


@lx.is_lower_triangular.register(BlockDiagonalSparse)
@lx.is_lower_triangular.register(_BlockDiagonalSparseTranspose)
def _(_operator) -> bool:
    return False


@lx.is_upper_triangular.register(BlockDiagonalSparse)
@lx.is_upper_triangular.register(_BlockDiagonalSparseTranspose)
def _(_operator) -> bool:
    return False


@lx.is_positive_semidefinite.register(BlockDiagonalSparse)
@lx.is_positive_semidefinite.register(_BlockDiagonalSparseTranspose)
def _(_operator) -> bool:
    return False
