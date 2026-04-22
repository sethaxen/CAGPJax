import abc

import equinox as eqx
from cola.ops import LinearOperator
from jaxtyping import PRNGKeyArray


class AbstractLinearSolverPolicy(eqx.Module):
    r"""Abstract base class for all linear solver policies.

    Policies define actions used to solve a linear system $A x = b$, where $A$ is a
    square linear operator.
    """

    ...


class AbstractBatchLinearSolverPolicy(AbstractLinearSolverPolicy):
    """Abstract base class for policies that produce action matrices."""

    n_actions: int = eqx.field(static=True, init=False)

    def __check_init__(self):
        if self.n_actions < 1:
            raise ValueError("n_actions must be at least 1")

    @abc.abstractmethod
    def to_actions(
        self, A: LinearOperator, *, key: PRNGKeyArray | None = None
    ) -> LinearOperator:
        r"""Compute all actions used to solve the linear system $Ax=b$.

        For a matrix $A$ with shape ``(n, n)``, the action matrix has shape
        ``(n, n_actions)``.

        Args:
            A: Linear operator representing the linear system.
            key: Optional random key used by stochastic policies.

        Returns:
            Linear operator representing the action matrix.
        """
        ...
