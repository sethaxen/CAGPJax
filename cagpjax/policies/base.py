import abc

from cola.ops import LinearOperator
from flax import nnx


class AbstractLinearSolverPolicy(nnx.Module):
    r"""Abstract base class for all linear solver policies.

    Policies define actions used to solve a linear system $A x = b$, where $A$ is a
    square linear operator.
    """

    ...


class AbstractBatchLinearSolverPolicy(AbstractLinearSolverPolicy, abc.ABC):
    """Abstract base class for policies that product action matrices."""

    @property
    @abc.abstractmethod
    def n_actions(self) -> int:
        """Number of actions in this policy."""
        ...

    @abc.abstractmethod
    def to_actions(self, A: LinearOperator) -> LinearOperator:
        r"""Compute all actions used to solve the linear system $Ax=b$.

        For a matrix $A$ with shape ``(n, n)``, the action matrix has shape
        ``(n, n_actions)``.

        Args:
            A: Linear operator representing the linear system.

        Returns:
            Linear operator representing the action matrix.
        """
        ...
