"""Custom linear operators."""

from .block_diagonal_sparse import BlockDiagonalSparse
from .diag_like import diag_like

__all__ = [
    "BlockDiagonalSparse",
    "diag_like",
]
