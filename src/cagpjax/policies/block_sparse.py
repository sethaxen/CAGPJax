"""Block-sparse policy."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import paramax
from jaxtyping import Array, Float, PRNGKeyArray
from typing_extensions import override

from ..operators import BlockDiagonalSparse
from .base import AbstractBatchLinearSolverPolicy, ActionOperator


def _normalize_by_blocks(
    values: Float[Array, "N"],
    n_actions: int,
) -> Float[Array, "N"]:
    n = values.shape[0]
    block_size = n // n_actions
    n_blocks_main = n_actions if n % n_actions == 0 else n_actions - 1
    n_main = n_blocks_main * block_size

    chunks = []
    if n_main > 0:
        main = values[:n_main].reshape(n_blocks_main, block_size)
        norms = jnp.linalg.vector_norm(main, axis=1, keepdims=True)
        main = main / jnp.where(norms > 0, norms, 1.0)
        chunks.append(main.reshape(n_main))
    if n > n_main:
        overhang = values[n_main:]
        norm = jnp.linalg.vector_norm(overhang)
        overhang = overhang / jnp.where(norm > 0, norm, 1.0)
        chunks.append(overhang)
    if not chunks:
        return values
    return jnp.concatenate(chunks)


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

    Attributes:
        n_actions: Number of actions to use.
        nz_values: Non-zero values of the block-diagonal sparse matrix.
    """

    n_actions: int = eqx.field(static=True)
    nz_values: Float[Array, "N"] | paramax.AbstractUnwrappable[Float[Array, "N"]]

    @classmethod
    def from_random(
        cls,
        key: PRNGKeyArray,
        num_datapoints: int,
        n_actions: int,
        *,
        sampler: Callable[
            [PRNGKeyArray, tuple[int, ...], Any], Float[Array, " N"]
        ] = jax.random.normal,
        dtype: Any = None,
    ) -> "BlockSparsePolicy":
        """Initialize policy from block-normalized random samples.

        Args:
            key: Random key used to sample initial values.
            num_datapoints: Number of rows in the resulting operator.
            n_actions: Number of action columns in the resulting operator.
            sampler: Callable with signature ``(key, shape, dtype) -> values``.
            dtype: Optional dtype forwarded to ``sampler``.
        """
        if num_datapoints < 1:
            raise ValueError("num_datapoints must be at least 1")
        nz_values = sampler(key, (num_datapoints,), dtype)
        nz_values = _normalize_by_blocks(nz_values, n_actions)
        return cls(n_actions=n_actions, nz_values=nz_values)

    @override
    def to_actions(
        self, A: ActionOperator, *, key: PRNGKeyArray | None = None
    ) -> BlockDiagonalSparse:
        """Convert to block diagonal sparse action operators.

        Args:
            A: Linear operator (unused).
            key: Optional random key (unused).

        Returns:
            BlockDiagonalSparse: Sparse action structure representing the blocks.
        """
        return BlockDiagonalSparse(paramax.unwrap(self.nz_values), self.n_actions)
