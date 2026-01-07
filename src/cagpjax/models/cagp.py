"""Computation-aware Gaussian Process models."""

from dataclasses import dataclass
from typing import Optional

import cola
import jax.numpy as jnp
from cola.ops import LinearOperator
from flax import nnx
from gpjax.gps import ConjugatePosterior, Dataset
from gpjax.mean_functions import Constant
from jaxtyping import Array, Float
from typing_extensions import override

from ..distributions import GaussianDistribution
from ..linalg import congruence_transform
from ..operators import diag_like
from ..operators.utils import lazify
from ..policies import AbstractBatchLinearSolverPolicy
from ..solvers import AbstractLinearSolver, AbstractLinearSolverMethod, Cholesky
from ..typing import ScalarFloat
from .base import AbstractComputationAwareGP


class ComputationAwareGP(AbstractComputationAwareGP):
    """Computation-aware Gaussian Process model.

    This model implements scalable GP inference by using batch linear solver
    policies to project the kernel and data to a lower-dimensional subspace, while
    accounting for the extra uncertainty imposed by observing only this subspace.

    Attributes:
        posterior: The original (exact) posterior.
        policy: The batch linear solver policy.
        solver_method: The linear solver method to use for solving linear systems
            with positive semi-definite operators.

    Notes:
        - Only single-output models are currently supported.
    """

    posterior: ConjugatePosterior
    policy: AbstractBatchLinearSolverPolicy
    solver_method: AbstractLinearSolverMethod

    def __init__(
        self,
        posterior: ConjugatePosterior,
        policy: AbstractBatchLinearSolverPolicy,
        solver_method: AbstractLinearSolverMethod = Cholesky(1e-6),
    ):
        """Initialize the Computation-Aware GP model.

        Args:
            posterior: GPJax conjugate posterior.
            policy: The batch linear solver policy that defines the subspace into
                which the data is projected.
            solver_method: The linear solver method to use for solving linear systems with
                positive semi-definite operators.
        """
        super().__init__(posterior)
        self.policy = policy
        self.solver_method = solver_method
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
            constant = prior.mean_function.constant[...]
            mean_prior = mean_prior.astype(constant.dtype)
        cov_xx = lazify(prior.kernel.gram(x))
        obs_cov = diag_like(cov_xx, likelihood.obs_stddev[...] ** 2)
        cov_prior = cov_xx + obs_cov

        # Project quantities to subspace
        actions = self.policy.to_actions(cov_prior)
        obs_cov_proj = congruence_transform(actions, obs_cov)
        cov_prior_proj = congruence_transform(actions, cov_prior)
        cov_prior_proj_solver = self.solver_method(cov_prior_proj)

        residual_proj = actions.T @ (y - mean_prior)
        repr_weights_proj = cov_prior_proj_solver.solve(residual_proj)

        self._posterior_params = _ProjectedPosteriorParameters(
            train_data=train_data,
            actions=actions,
            obs_cov_proj=obs_cov_proj,
            cov_prior_proj_solver=cov_prior_proj_solver,
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
        train_data = self._posterior_params.train_data
        assert train_data.X is not None  # help out pyright
        x = jnp.atleast_2d(train_data.X)
        actions = self._posterior_params.actions
        cov_prior_proj_solver = self._posterior_params.cov_prior_proj_solver
        repr_weights_proj = self._posterior_params.repr_weights_proj

        # Predictions at test points
        z = test_inputs if test_inputs is not None else x
        prior = self.posterior.prior
        mean_z = prior.mean_function(z).squeeze()
        # Work around GPJax promoting dtype of mean to float64 (See JaxGaussianProcesses/GPJax#523)
        if isinstance(prior.mean_function, Constant):
            constant = prior.mean_function.constant[...]
            mean_z = mean_z.astype(constant.dtype)
        cov_zz = lazify(prior.kernel.gram(z))
        cov_zx = cov_zz if test_inputs is None else prior.kernel.cross_covariance(z, x)
        cov_zx_proj = cov_zx @ actions

        # Posterior predictive distribution
        mean_pred = jnp.atleast_1d(mean_z + cov_zx_proj @ repr_weights_proj)
        cov_pred = cov_zz - cov_prior_proj_solver.inv_congruence_transform(
            cov_zx_proj.T
        )
        cov_pred = cola.PSD(cov_pred)

        return GaussianDistribution(
            mean_pred, cov_pred, solver_method=self.solver_method
        )

    def prior_kl(self) -> ScalarFloat:
        r"""Compute KL divergence between CaGP posterior and GP prior..

        Calculates $\mathrm{KL}[q(f) || p(f)]$, where $q(f)$ is the CaGP
        posterior approximation and $p(f)$ is the GP prior.

        ``condition`` must be called before this method can be used.

        Returns:
            KL divergence value (scalar).
        """
        if not self.is_conditioned:
            raise ValueError("Model is not yet conditioned. Call ``condition`` first.")

        # help out pyright
        assert self._posterior_params is not None

        # Unpack posterior parameters
        obs_cov_proj = self._posterior_params.obs_cov_proj
        cov_prior_proj_solver = self._posterior_params.cov_prior_proj_solver
        residual_proj = self._posterior_params.residual_proj
        repr_weights_proj = self._posterior_params.repr_weights_proj

        obs_cov_proj_solver = self.solver_method(obs_cov_proj)

        kl = (
            _kl_divergence_from_solvers(
                residual_proj,
                obs_cov_proj_solver,
                jnp.zeros_like(residual_proj),
                cov_prior_proj_solver,
            )
            - 0.5 * congruence_transform(repr_weights_proj.T, obs_cov_proj).squeeze()
        )

        return kl

    def variational_expectation(
        self, data: Optional[Dataset] = None
    ) -> Float[Array, "K"]:
        """Compute the variational expectation.

        Compute the pointwise expected log-likelihood under the variational distribution.

        Note:
            This should be used instead of ``gpjax.objectives.variational_expectation``

        Args:
            data: If provided, a length ``K`` subset of the training data for which the expectation
                  should be computed. If not provided, the expectation is computed for all
                  training data.

        Returns:
            expectation: The pointwise expected log-likelihood under the variational distribution.
        """

        if not self.is_conditioned:
            raise ValueError("Model is not yet conditioned. Call ``condition`` first.")

        # Unpack data
        if data is not None:
            if data.X is None or data.y is None:
                raise ValueError("Data must be supervised.")
            x = jnp.atleast_2d(data.X)
            y = data.y
        else:
            assert self._posterior_params is not None  # help out pyright
            x = None
            y = self._posterior_params.train_data.y

        # Predict and compute expectation
        qpred = self.predict(x)
        mean = qpred.mean
        variance = qpred.variance
        expectation = self.posterior.likelihood.expected_log_likelihood(
            y, mean[:, None], variance[:, None]
        )

        return expectation

    def elbo(self, data: Optional[Dataset] = None) -> ScalarFloat:
        """Compute the evidence lower bound.

        Computes the evidence lower bound (ELBO) under this model's variational distribution.

        Note:
            This should be used instead of ``gpjax.objectives.elbo``

        Args:
            data: If provided, a subset of the training data for which the ELBO should be computed.
                  If not provided, the ELBO is computed for all training data.

        Returns:
            ELBO value (scalar).
        """
        kl = self.prior_kl()
        var_exp = self.variational_expectation(data)
        ntrain = self.posterior.likelihood.num_datapoints
        ntest = data.n if data is not None else ntrain
        return (jnp.sum(var_exp) * ntrain) / ntest - kl


# Technically we need the projected mean and covariance of the prior, projected data, and
# projected likelihood, but these intermediates are more computationally useful.
@dataclass
class _ProjectedPosteriorParameters:
    """Projected quantities for computation-aware GP inference.

    Args:
        train_data: Training data with N inputs with D dimensions.
        actions: Actions operator; transpose of operator projecting from N-dimensional space
            to M-dimensional subspace.
        obs_cov_proj: Projected covariance of likelihood.
        cov_prior_proj_solver: Linear solver for ``cov_prior_proj``.
        residual_proj: Projected residuals between observations and prior mean.
        repr_weights_proj: Projected representer weights.
    """

    train_data: Dataset
    actions: LinearOperator
    obs_cov_proj: LinearOperator
    cov_prior_proj_solver: AbstractLinearSolver
    residual_proj: Float[Array, "M"]
    repr_weights_proj: Float[Array, "M"]


def _kl_divergence_from_solvers(
    mean_q: Float[Array, "N"],
    cov_q_solver: AbstractLinearSolver,
    mean_p: Float[Array, "N"],
    cov_p_solver: AbstractLinearSolver,
) -> ScalarFloat:
    """Compute KL divergence between two Gaussian distributions."""
    n = mean_q.shape[0]
    diff = mean_p - mean_q

    # tr(inv(cov_p) cov_q)
    inner = cov_p_solver.trace_solve(cov_q_solver)

    # (mean_p - mean_q)' inv(cov_p) (mean_p - mean_q)
    mahalanobis = cov_p_solver.inv_quad(diff)

    logdet_ratio = cov_p_solver.logdet() - cov_q_solver.logdet()

    return 0.5 * (inner + logdet_ratio + mahalanobis - n)
