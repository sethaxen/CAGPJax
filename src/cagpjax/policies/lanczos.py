"""Lanczos-based policies."""

import equinox as eqx
from cola.ops import LinearOperator
from jaxtyping import PRNGKeyArray
from typing_extensions import override

from ..linalg.eigh import Lanczos, eigh
from .base import AbstractBatchLinearSolverPolicy


class LanczosPolicy(AbstractBatchLinearSolverPolicy):
    """Lanczos-based policy for eigenvalue decomposition approximation.

    This policy uses the Lanczos algorithm to compute the top ``n_actions`` eigenvectors
    of the linear operator $A$.

    Attributes:
        n_actions: Number of Lanczos vectors/actions to compute.
        key: Random key for reproducible Lanczos iterations.
    """

    grad_rtol: float | None = eqx.field(static=True, default=0.0)

    def __init__(
        self,
        n_actions: int,
        grad_rtol: float | None = 0.0,
    ):
        """Initialize the Lanczos policy.

        Args:
            n_actions: Number of Lanczos vectors to compute.
            grad_rtol: Specifies the cutoff for similar eigenvalues, used to improve
                gradient computation for (almost-)degenerate matrices.
                If not provided, the default is 0.0.
                If None or negative, all eigenvalues are treated as distinct.
                (see [`cagpjax.linalg.eigh`][] for more details)
        """
        super().__init__(n_actions)
        self.grad_rtol = grad_rtol

    @override
    def to_actions(
        self, A: LinearOperator, *, key: PRNGKeyArray | None = None
    ) -> LinearOperator:
        """Compute action matrix.

        Args:
            A: Symmetric linear operator representing the linear system.
            key: Random key used to initialize the Lanczos run.

        Returns:
            Linear operator containing the Lanczos vectors as columns.
        """
        vecs = eigh(
            A, alg=Lanczos(self.n_actions, key=key), grad_rtol=self.grad_rtol
        ).eigenvectors
        return vecs
