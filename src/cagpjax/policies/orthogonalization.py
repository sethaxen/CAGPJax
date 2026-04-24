import cola
import equinox as eqx
import lineax as lx
from cola.ops import LinearOperator
from jaxtyping import PRNGKeyArray
from typing_extensions import override

from ..linalg import OrthogonalizationMethod, orthogonalize
from ..policies.base import AbstractBatchLinearSolverPolicy, ActionOperator


class OrthogonalizationPolicy(AbstractBatchLinearSolverPolicy):
    """Orthogonalization policy.

    This policy orthogonalizes (if necessary) the action operator produced by the base policy.

    Args:
        base_policy: The base policy that produces the action operator to be orthogonalized.
        method: The method to use for orthogonalization.
        n_reortho: The number of times to _re_-orthogonalize each column.
            Reorthogonalizing once is generally sufficient to improve orthogonality
            for Gram-Schmidt variants
            (see e.g. [10.1007/s00211-005-0615-4](https://doi.org/10.1007/s00211-005-0615-4)).
    """

    base_policy: AbstractBatchLinearSolverPolicy
    method: OrthogonalizationMethod = eqx.field(
        static=True, default=OrthogonalizationMethod.QR
    )
    n_reortho: int = eqx.field(static=True, default=0)

    def __post_init__(self):
        self.n_actions = self.base_policy.n_actions

    def __check_init__(self):
        if self.n_reortho < 0:
            raise ValueError("n_reortho must be non-negative")

    @override
    def to_actions(
        self, A: LinearOperator, *, key: PRNGKeyArray | None = None
    ) -> ActionOperator:
        op = self.base_policy.to_actions(A, key=key)
        ortho_actions = orthogonalize(op, method=self.method, n_reortho=self.n_reortho)
        if isinstance(ortho_actions, lx.AbstractLinearOperator):
            return ortho_actions
        return cola.lazify(ortho_actions)
