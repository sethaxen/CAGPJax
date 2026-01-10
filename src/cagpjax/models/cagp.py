"""Computation-aware Gaussian Process models."""

from dataclasses import dataclass

import cola
import jax.numpy as jnp
from cola.ops import LinearOperator
from flax import nnx
from gpjax.gps import ConjugatePosterior, Dataset
from gpjax.mean_functions import Constant
from jaxtyping import Array, Float
from typing_extensions import Generic, TypeVar

from ..distributions import GaussianDistribution
from ..linalg import congruence_transform
from ..operators import diag_like
from ..operators.utils import lazify
from ..policies import AbstractBatchLinearSolverPolicy
from ..solvers import AbstractLinearSolver, Cholesky
from ..typing import ScalarFloat

_LinearSolverState = TypeVar("_LinearSolverState")


# Technically we need the projected mean and covariance of the prior, projected data, and
# projected likelihood, but these intermediates are more computationally useful.
@dataclass
class ComputationAwareGPState(Generic[_LinearSolverState]):
    """Projected quantities for computation-aware GP inference.

    Args:
        x: N training inputs with D dimensions.
        actions: Actions operator; transpose of operator projecting from N-dimensional space
            to M-dimensional subspace.
        obs_cov_proj: Projected covariance of likelihood.
        cov_prior_proj_state: Linear solver state for ``cov_prior_proj``.
        residual_proj: Projected residuals between observations and prior mean.
        repr_weights_proj: Projected representer weights.
    """

    x: Float[Array, "N D"]
    actions: LinearOperator
    obs_cov_proj: LinearOperator
    cov_prior_proj_state: _LinearSolverState
    residual_proj: Float[Array, "M"]
    repr_weights_proj: Float[Array, "M"]


class ComputationAwareGP(nnx.Module, Generic[_LinearSolverState]):
    """Computation-aware Gaussian Process model.

    This model implements scalable GP inference by using batch linear solver
    policies to project the kernel and data to a lower-dimensional subspace, while
    accounting for the extra uncertainty imposed by observing only this subspace.

    Attributes:
        posterior: The original (exact) posterior.
        policy: The batch linear solver policy.
        solver: The linear solver method to use for solving linear systems
            with positive semi-definite operators.

    Notes:
        - Only single-output models are currently supported.
    """

    posterior: ConjugatePosterior
    policy: AbstractBatchLinearSolverPolicy
    solver: AbstractLinearSolver[_LinearSolverState]

    def __init__(
        self,
        posterior: ConjugatePosterior,
        policy: AbstractBatchLinearSolverPolicy,
        solver: AbstractLinearSolver[_LinearSolverState] = Cholesky(1e-6),
    ):
        """Initialize the Computation-Aware GP model.

        Args:
            posterior: GPJax conjugate posterior.
            policy: The batch linear solver policy that defines the subspace into
                which the data is projected.
            solver: The linear solver method to use for solving linear systems with
                positive semi-definite operators.
        """
        self.posterior = posterior
        self.policy = policy
        self.solver = solver

    def init(self, train_data: Dataset) -> ComputationAwareGPState[_LinearSolverState]:
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
        cov_prior_proj_state = self.solver.init(cov_prior_proj)

        residual_proj = actions.T @ (y - mean_prior)
        repr_weights_proj = self.solver.solve(cov_prior_proj_state, residual_proj)

        return ComputationAwareGPState(
            x=x,
            actions=actions,
            obs_cov_proj=obs_cov_proj,
            cov_prior_proj_state=cov_prior_proj_state,
            residual_proj=residual_proj,
            repr_weights_proj=repr_weights_proj,
        )

    def predict(
        self,
        state: ComputationAwareGPState[_LinearSolverState],
        test_inputs: Float[Array, "N D"] | None = None,
    ) -> GaussianDistribution:
        """Compute the predictive distribution of the GP at the test inputs.

        Args:
            test_inputs: The test inputs at which to make predictions. If not provided,
                predictions are made at the training inputs.

        Returns:
            GaussianDistribution: The predictive distribution of the GP at the
                test inputs.
        """
        # Unpack posterior parameters
        x = state.x
        actions = state.actions
        cov_prior_proj_state = state.cov_prior_proj_state
        repr_weights_proj = state.repr_weights_proj

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
        cov_pred = cov_zz - self.solver.inv_congruence_transform(
            cov_prior_proj_state, cov_zx_proj.T
        )
        cov_pred = cola.PSD(cov_pred)

        return GaussianDistribution(mean_pred, cov_pred, solver=self.solver)

    def prior_kl(
        self, state: ComputationAwareGPState[_LinearSolverState]
    ) -> ScalarFloat:
        r"""Compute KL divergence between CaGP posterior and GP prior..

        Calculates $\mathrm{KL}[q(f) || p(f)]$, where $q(f)$ is the CaGP
        posterior approximation and $p(f)$ is the GP prior.

        Returns:
            KL divergence value (scalar).
        """
        # Unpack posterior parameters
        obs_cov_proj = state.obs_cov_proj
        cov_prior_proj_state = state.cov_prior_proj_state
        residual_proj = state.residual_proj
        repr_weights_proj = state.repr_weights_proj

        obs_cov_proj_state = self.solver.init(obs_cov_proj)

        kl = (
            _kl_divergence_from_solvers(
                self.solver,
                residual_proj,
                obs_cov_proj_state,
                jnp.zeros_like(residual_proj),
                cov_prior_proj_state,
            )
            - 0.5 * congruence_transform(repr_weights_proj.T, obs_cov_proj).squeeze()
        )

        return kl


def _kl_divergence_from_solvers(
    solver: AbstractLinearSolver[_LinearSolverState],
    mean_q: Float[Array, "N"],
    cov_q_state: _LinearSolverState,
    mean_p: Float[Array, "N"],
    cov_p_state: _LinearSolverState,
) -> ScalarFloat:
    """Compute KL divergence between two Gaussian distributions."""
    n = mean_q.shape[0]
    diff = mean_p - mean_q

    # tr(inv(cov_p) cov_q)
    inner = solver.trace_solve(cov_p_state, cov_q_state)

    # (mean_p - mean_q)' inv(cov_p) (mean_p - mean_q)
    mahalanobis = solver.inv_quad(cov_p_state, diff)

    logdet_ratio = solver.logdet(cov_p_state) - solver.logdet(cov_q_state)

    return 0.5 * (inner + logdet_ratio + mahalanobis - n)
