"""Abstract base classes for models."""

import abc

from gpjax.variational_families import AbstractVariationalFamily


class AbstractComputationallyAwareGP(AbstractVariationalFamily, abc.ABC):
    """Abstract base class for Computationally-Aware Gaussian Processes.

    While CaGPs can be viewed as exact GPs on a data subspace, when the actions
    are learnable, they can also be interpreted as a variational family whose
    variational parameters are the parameters of the actions.
    """

    ...
