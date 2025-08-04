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
        s_1 & 0 & \cdots & 0 \\
        0 & s_2 & \cdots & 0 \\
        \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & \cdots & s_{\text{n_actions}}
    \end{bmatrix}
    $$

    These are stacked and stored as a single trainable parameter ``nz_values``.
    """

    def __init__(
        self,
        n_actions: int,
        n: int | None = None,
        nz_values: Float[Array, "N"] | nnx.Variable[Float[Array, "N"]] | None = None,
        key: PRNGKeyArray | None = None,
        **kwargs,
    ):
        """Initialize the block sparse policy.

        Args:
            n_actions: Number of actions to use.
            n: Number of rows and columns of the full operator. Must be provided if ``nz_values`` is
                not provided.
            nz_values: Non-zero values of the block-diagonal sparse matrix (shape ``(n,)``). If not
                provided, random actions are sampled using the key if provided.
            key: Random key for sampling actions if ``nz_values`` is not provided.
            **kwargs: Additional keyword arguments for ``jax.random.normal`` (e.g. ``dtype``)
        """
        if nz_values is None:
            if n is None:
                raise ValueError("n must be provided if nz_values is not provided")
            if key is None:
                key = jax.random.PRNGKey(0)
            block_size = n // n_actions
            nz_values = jax.random.normal(key, (n,), **kwargs)
            nz_values /= jnp.sqrt(block_size)
        elif n is not None:
            warnings.warn("n is ignored because nz_values is provided")

        if not isinstance(nz_values, nnx.Variable):
            nz_values = Real(nz_values)

        self.nz_values: nnx.Variable[Float[Array, "N"]] = nz_values
        self._n_actions: int = n_actions

    @property
    @override
    def n_actions(self) -> int:
        """Number of actions to be used."""
        return self._n_actions

    @override
    def to_actions(self, A: LinearOperator) -> LinearOperator:
        """Convert to block diagonal sparse action operators.

        Args:
            A: Linear operator (unused).

        Returns:
            BlockDiagonalSparse: Sparse action structure representing the blocks.
        """
        return BlockDiagonalSparse(self.nz_values.value, self.n_actions)
