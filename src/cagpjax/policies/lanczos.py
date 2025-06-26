"""Lanczos-based policies."""

import cola
import cola.linalg
from cola.ops import Dense, LinearOperator
from jaxtyping import PRNGKeyArray
from typing_extensions import override

from .base import AbstractBatchLinearSolverPolicy


class LanczosPolicy(AbstractBatchLinearSolverPolicy):
    """Lanczos-based policy for eigenvalue decomposition approximation.

    This policy uses the Lanczos algorithm to compute the top ``n_actions`` eigenvectors
    of the linear operator $A$.

    Attributes:
        n_actions: Number of Lanczos vectors/actions to compute.
        key: Random key for reproducible Lanczos iterations.
    """

    def __init__(self, n_actions: int, key: PRNGKeyArray | None = None):
        """Initialize the Lanczos policy.

        Args:
            n_actions: Number of Lanczos vectors to compute.
            key: Random key for initialization.
        """
        self._n_actions: int = n_actions
        self.key: PRNGKeyArray | None = key

    @property
    @override
    def n_actions(self) -> int:
        return self._n_actions

    @override
    def to_actions(self, A: LinearOperator) -> LinearOperator:
        """Compute action matrix.

        Args:
            A: Symmetric linear operator representing the linear system.

        Returns:
            Linear operator containing the Lanczos vectors as columns.
        """
        vecs = cola.linalg.eig(
            cola.SelfAdjoint(A),
            self.n_actions,
            which="LM",
            alg=cola.linalg.Lanczos(key=self.key),
        )[1]
        if not isinstance(vecs, LinearOperator):
            vecs = Dense(vecs)
        return vecs
