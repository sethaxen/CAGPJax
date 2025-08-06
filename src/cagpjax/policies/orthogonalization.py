import cola
from cola.ops import LinearOperator
from typing_extensions import override

from ..linalg import OrthogonalizationMethod, orthogonalize
from ..policies.base import AbstractBatchLinearSolverPolicy


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
    method: OrthogonalizationMethod
    n_reortho: int

    def __init__(
        self,
        base_policy: AbstractBatchLinearSolverPolicy,
        method: OrthogonalizationMethod = OrthogonalizationMethod.QR,
        n_reortho: int = 0,
    ):
        self.base_policy = base_policy
        self.method = method
        self.n_reortho = n_reortho

    @property
    @override
    def n_actions(self):
        return self.base_policy.n_actions

    @override
    def to_actions(self, A: LinearOperator) -> LinearOperator:
        op = self.base_policy.to_actions(A)
        return cola.lazify(
            orthogonalize(op, method=self.method, n_reortho=self.n_reortho)
        )
