"""Lanczos-based policies."""

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

    key: PRNGKeyArray | None
    grad_rtol: float | None

    def __init__(
        self,
        n_actions: int | None,
        key: PRNGKeyArray | None = None,
        grad_rtol: float | None = 0.0,
    ):
        """Initialize the Lanczos policy.

        Args:
            n_actions: Number of Lanczos vectors to compute.
            key: Random key for initialization.
            grad_rtol: Specifies the cutoff for similar eigenvalues, used to improve
                gradient computation for (almost-)degenerate matrices.
                If not provided, the default is 0.0.
                If None or negative, all eigenvalues are treated as distinct.
                (see [`cagpjax.linalg.eigh`][] for more details)
        """
        self._n_actions: int = n_actions
        self.key = key
        self.grad_rtol = grad_rtol

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
        vecs = eigh(
            A, alg=Lanczos(self.n_actions, key=self.key), grad_rtol=self.grad_rtol
        ).eigenvectors
        return vecs
