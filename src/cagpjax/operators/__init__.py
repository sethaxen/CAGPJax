"""Custom linear operators."""

from .block_diagonal_sparse import BlockDiagonalSparse
from .diag_like import diag_like
from .lazy_kernel import LazyKernel

__all__ = [
    "BlockDiagonalSparse",
    "LazyKernel",
    "diag_like",
]
