"""Lanczos-based policies."""

import equinox as eqx
from jaxtyping import PRNGKeyArray
from typing_extensions import override

from ..interop import lazify
from ..linalg.eigh import Lanczos, eigh
from .base import AbstractBatchLinearSolverPolicy, ActionOperator


class LanczosPolicy(AbstractBatchLinearSolverPolicy):
    """Lanczos-based policy for eigenvalue decomposition approximation.

    This policy uses the Lanczos algorithm to compute the top ``n_actions`` eigenvectors
    of the linear operator $A$.

    Attributes:
        n_actions: Number of Lanczos vectors/actions to compute.
        grad_rtol: Specifies the cutoff for similar eigenvalues, used to improve
            gradient computation for (almost-)degenerate matrices.
            If not provided, the default is 0.0.
            If None or negative, all eigenvalues are treated as distinct.
            (see [`cagpjax.linalg.eigh`][] for more details)
    """

    n_actions: int = eqx.field(static=True)
    grad_rtol: float | None = eqx.field(static=True, default=0.0)

    @override
    def to_actions(
        self, A: ActionOperator, *, key: PRNGKeyArray | None = None
    ) -> ActionOperator:
        """Compute action matrix.

        Args:
            A: Symmetric linear operator representing the linear system.
            key: Random key used to initialize the Lanczos run.

        Returns:
            Linear operator containing the Lanczos vectors as columns.
        """
        vecs = eigh(
            lazify(A), alg=Lanczos(self.n_actions, key=key), grad_rtol=self.grad_rtol
        ).eigenvectors
        return vecs
