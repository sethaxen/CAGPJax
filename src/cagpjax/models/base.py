"""Abstract base classes for models."""

import abc

from gpjax.variational_families import AbstractVariationalFamily
from jaxtyping import Array, Float
from typing_extensions import override

from ..distributions import GaussianDistribution


class AbstractComputationAwareGP(AbstractVariationalFamily, abc.ABC):
    """Abstract base class for Computation-Aware Gaussian Processes.

    While CaGPs can be viewed as exact GPs on a data subspace, when the actions
    are learnable, they can also be interpreted as a variational family whose
    variational parameters are the parameters of the actions.

    Note:
        This class technically violates the API for `AbstractComputationAwareGP`, which
        expects that the return type of `predict` is a `gpjax.distributions.GaussianDistribution`,
        not our own `GaussianDistribution`. To use some of GPJax's functionality, you may
        need to convert the return value to a `gpjax.distributions.GaussianDistribution`.
    """

    ...

    @override
    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, test_inputs: Float[Array, "N D"] | None = None
    ) -> GaussianDistribution:
        """Compute the predictive distribution of the GP at the test inputs.

        Args:
            test_inputs: The test inputs at which to make predictions. If not provided,
                predictions are made at the training inputs.

        Returns:
            GaussianDistribution: The predictive distribution of the GP at the
                test inputs.
        """
        raise NotImplementedError
