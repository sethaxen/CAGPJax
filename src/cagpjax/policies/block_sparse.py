"""Block-sparse policy."""

import warnings

import jax
import jax.numpy as jnp
from cola.ops import LinearOperator
from flax import nnx
from gpjax.parameters import Real
from jaxtyping import Array, Float, PRNGKeyArray
from typing_extensions import override

from ..operators import BlockDiagonalSparse
from .base import AbstractBatchLinearSolverPolicy


class BlockSparsePolicy(AbstractBatchLinearSolverPolicy):
    r"""Block-sparse linear solver policy.
    
    This policy uses a fixed block-diagonal sparse structure to define
    independent learnable actions. The matrix has the following structure:

    $$
    S = \begin{bmatrix}
        s_1 & 0 & \cdots & 0 & 0 \\
        0 & s_2 & \cdots & 0 & 0 \\
        \vdots & \vdots & \ddots & \vdots & \vdots \\
        0 & 0 & \cdots & s_{\text{n_actions}} & 0 \\
        0 & 0 & \cdots & 0 & 0
    \end{bmatrix}.
    $$
    This effectively ignores the last ``n % n_actions`` rows of data.

    These are stacked and stored as a single trainable parameter ``nz_values``.

    !!! note
        The last block is included if necessary to pad the first dimension of the matrix
        to be equal to ``n``. This effectively ignores the last ``n % n_actions`` rows of data.
    """

    def __init__(
        self,
        n: int,
        nz_values: Float[Array, "n_actions block_size"]
        | nnx.Variable[Float[Array, "n_actions block_size"]]
        | None = None,
        n_actions: int | None = None,
        key: PRNGKeyArray | None = None,
        **kwargs,
    ):
        """Initialize the block sparse policy.

        Args:
            n: Number of rows and columns of the full operator.
            nz_values: Non-zero values of the block-diagonal sparse matrix. If not
                provided, random actions are sampled using the ``key``.
            n_actions: Number of actions to use. Required if ``nz_values`` is not provided.
            key: Random key for sampling actions if ``nz_values`` is not provided.
            **kwargs: Additional keyword arguments for ``jax.random.normal`` (e.g. ``dtype``)
        """
        if nz_values is None:
            if n_actions is None:
                raise ValueError(
                    "n_actions must be provided if nz_values is not provided"
                )
            if key is None:
                key = jax.random.PRNGKey(0)
            block_size = n // n_actions
            nz_values = jax.random.normal(key, (n_actions, block_size), **kwargs)
            nz_values /= jnp.sqrt(block_size)

        if not isinstance(nz_values, nnx.Variable):
            nz_values = Real(nz_values)

        self.nz_values: nnx.Variable[Float[Array, "n_actions block_size"]] = nz_values
        self._n: int = n

    @property
    @override
    def n_actions(self) -> int:
        """Number of actions to be used."""
        return self.nz_values.value.shape[0]

    @override
    def to_actions(self, A: LinearOperator) -> LinearOperator:
        """Convert to block diagonal sparse action operators.

        Args:
            A: Linear operator (unused).

        Returns:
            Transposed[BlockDiagonalSparse]: Sparse action structure representing the blocks.
        """
        return BlockDiagonalSparse(self.nz_values.value, self._n).T
