"""Computationally-aware Gaussian Process models."""

from dataclasses import dataclass

import cola
import cola.linalg
import jax.numpy as jnp
from cola.linalg import Cholesky
from cola.ops import LinearOperator
from gpjax.distributions import GaussianDistribution
from gpjax.gps import ConjugatePosterior, Dataset
from gpjax.mean_functions import Constant
from jaxtyping import Array, Float
from typing_extensions import override

from ..linalg import congruence_transform
from ..operators import diag_like
from ..policies import AbstractBatchLinearSolverPolicy
from ..typing import ScalarFloat
from .base import AbstractComputationallyAwareGP


class ComputationallyAwareGP(AbstractComputationallyAwareGP):
    """Computationally-aware Gaussian Process model[@wenger2022;@wenger2024].

    This model implements scalable GP inference by using batch linear solver
    policies to project the kernel and data to a lower-dimensional subspace, while
    accounting for the extra uncertainty imposed by observing only this subspace.

    Attributes:
        posterior: The original (exact) posterior.
        policy: The batch linear solver policy.
        jitter: Numerical jitter for stability.

    Notes:
        - Only single-output models are currently supported.
    """

    def __init__(
        self,
        posterior: ConjugatePosterior,
        policy: AbstractBatchLinearSolverPolicy,
        jitter: ScalarFloat = 1e-6,
    ):
        """Initialize the Computationally-Aware GP model.

        Args:
            posterior: GPJax conjugate posterior.
            policy: The batch linear solver policy that defines the subspace into
                which the data is projected.
            jitter: A small positive constant added to the diagonal of a covariance
                matrix when necessary to ensure numerical stability.
        """
        super().__init__(posterior)
        self.policy: AbstractBatchLinearSolverPolicy = policy
        self.jitter: ScalarFloat = jitter
        self._posterior_params: _ProjectedPosteriorParameters | None = None

    @property
    def is_conditioned(self) -> bool:
        """Whether the model has been conditioned on training data."""
        return self._posterior_params is not None

    def condition(self, train_data: Dataset) -> None:
        """Compute and store the projected quantities of the conditioned GP posterior.

        Args:
            train_data: The training data used to fit the GP.
        """
        # Ensure we have supervised training data
        if train_data.X is None or train_data.y is None:
            raise ValueError("Training data must be supervised.")

        # Unpack training data
        x = jnp.atleast_2d(train_data.X)
        y = jnp.atleast_1d(train_data.y).squeeze()

        # Unpack prior and likelihood
        prior = self.posterior.prior
        likelihood = self.posterior.likelihood

        # Mean and covariance of prior-predictive distribution
        mean_prior = prior.mean_function(x).squeeze()
        # Work around GPJax promoting dtype of mean to float64 (See JaxGaussianProcesses/GPJax#523)
        if isinstance(prior.mean_function, Constant):
            mean_prior = mean_prior.astype(prior.mean_function.constant.value.dtype)
        cov_xx = prior.kernel.gram(x)
        obs_cov = diag_like(cov_xx, likelihood.obs_stddev.value**2)
        cov_prior = cov_xx + obs_cov

        # Project quantities to subspace
        proj = self.policy.to_actions(cov_prior).T
        obs_cov_proj = congruence_transform(proj, obs_cov)
        cov_prior_proj = congruence_transform(proj, cov_prior)
        cov_prior_proj = cola.PSD(
            cov_prior_proj + diag_like(cov_prior_proj, self.jitter)
        )
        inv_cov_prior_proj = cola.linalg.inv(cov_prior_proj, alg=Cholesky())

        residual_proj = proj @ (y - mean_prior)
        repr_weights_proj = inv_cov_prior_proj @ residual_proj

        self._posterior_params = _ProjectedPosteriorParameters(
            x=x,
            proj=proj,
            obs_cov_proj=obs_cov_proj,
            cov_prior_proj=cov_prior_proj,
            inv_cov_prior_proj=inv_cov_prior_proj,
            residual_proj=residual_proj,
            repr_weights_proj=repr_weights_proj,
        )

    @override
    def predict(
        self, test_inputs: Float[Array, "N D"] | None = None
    ) -> GaussianDistribution:
        """Compute the predictive distribution of the GP at the test inputs.

        ``condition`` must be called before this method can be used.

        Args:
            test_inputs: The test inputs at which to make predictions. If not provided,
                predictions are made at the training inputs.

        Returns:
            GaussianDistribution: The predictive distribution of the GP at the
                test inputs.
        """
        if not self.is_conditioned:
            raise ValueError("Model is not yet conditioned. Call ``condition`` first.")

        # help out pyright
        assert self._posterior_params is not None

        # Unpack posterior parameters
        x = self._posterior_params.x
        proj = self._posterior_params.proj
        inv_cov_prior_proj = self._posterior_params.inv_cov_prior_proj
        repr_weights_proj = self._posterior_params.repr_weights_proj

        # Predictions at test points
        z = test_inputs if test_inputs is not None else x
        prior = self.posterior.prior
        mean_z = prior.mean_function(z).squeeze()
        # Work around GPJax promoting dtype of mean to float64 (See JaxGaussianProcesses/GPJax#523)
        if isinstance(prior.mean_function, Constant):
            mean_z = mean_z.astype(prior.mean_function.constant.value.dtype)
        cov_zz = prior.kernel.gram(z)
        cov_xz = cov_zz if test_inputs is None else prior.kernel.cross_covariance(x, z)
        cov_xz_proj = proj @ cov_xz

        # Posterior predictive distribution
        mean_pred = jnp.atleast_1d(mean_z + cov_xz_proj.T @ repr_weights_proj)
        cov_pred = cola.PSD(
            cov_zz - congruence_transform(cov_xz_proj.T, inv_cov_prior_proj)
        )

        return GaussianDistribution(mean_pred, cov_pred)


# Technically we need the projected mean and covariance of the prior, projected data, and
# projected likelihood, but these intermediates are more computationally useful.
@dataclass
class _ProjectedPosteriorParameters:
    """Projected quantities for computationally-aware GP inference.

    Args:
        x: N training inputs with D dimensions.
        proj: Projection operator mapping from N-dimensional space
            to M-dimensional subspace.
        obs_cov_proj: Projected covariance of likelihood.
        cov_prior_proj: Projected covariance of prior-predictive distribution.
        inv_cov_prior_proj: Operator representing inverse of ``cov_prior_proj``.
        residual_proj: Projected residuals between observations and prior mean.
        repr_weights_proj: Projected representer weights.
    """

    x: Float[Array, "N D"]
    proj: LinearOperator
    obs_cov_proj: LinearOperator
    cov_prior_proj: LinearOperator
    inv_cov_prior_proj: LinearOperator
    residual_proj: Float[Array, "M"]
    repr_weights_proj: Float[Array, "M"]
