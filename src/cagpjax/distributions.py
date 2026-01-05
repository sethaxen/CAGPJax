import math

import cola
import jax
import jax.numpy as jnp
from cola.ops import LinearOperator
from jaxtyping import Array, Float, Key
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution

from .solvers import AbstractLinearSolverMethod, Cholesky
from .typing import ScalarFloat


class GaussianDistribution(Distribution):
    """Gaussian distribution with an implicit covariance and customizable linear solver."""

    loc: Float[Array, " N"]
    scale: LinearOperator
    support = constraints.real_vector
    solver_method: AbstractLinearSolverMethod

    def __init__(
        self,
        loc: Float[Array, " N"],
        scale: LinearOperator,
        solver_method: AbstractLinearSolverMethod = Cholesky(1e-6),
        **kwargs,
    ):
        """Initialize the Gaussian distribution.

        Args:
            loc: Mean of the distribution.
            scale: Scale of the distribution.
            solver_method: Method for solving the linear system of equations.
        """
        self.loc = loc
        self.scale = scale
        batch_shape = ()
        event_shape = jnp.shape(self.loc)
        self.solver_method = solver_method
        super().__init__(batch_shape, event_shape, **kwargs)

    @property
    def mean(self) -> Float[Array, " N"]:
        """Mean of the distribution."""
        return self.loc

    @property
    def variance(self) -> Float[Array, " N"]:
        """Marginal variance of the distribution."""
        return cola.diag(self.scale)

    @property
    def stddev(self) -> Float[Array, " N"]:
        """Marginal standard deviation of the distribution."""
        return jnp.sqrt(self.variance)

    def covariance(self) -> LinearOperator:
        """Operator representing the covariance of the distribution."""
        return self.scale

    def log_prob(self, value: Float[Array, " N"]) -> ScalarFloat:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Compute the log probability of the distribution at the given value.
        
        Args:
            value: Value at which to compute the log probability.

        Returns:
            Log probability of the distribution at the given value.
        """
        mu = self.loc
        sigma = self.scale
        n = mu.shape[-1]
        solver = self.solver_method(sigma)
        return (
            n * jnp.log(2 * jnp.pi) + solver.logdet() + solver.inv_quad(value - mu)
        ) / -2

    def sample(
        self,
        key: Key,
        sample_shape: tuple[int, ...] = (),
    ) -> Float[Array, "*sample_shape N"]:
        """Sample from the distribution.
        
        Args:
            key: Random key for sampling.
            sample_shape: Shape of the sample.

        Returns:
            Sample from the distribution.
        """
        mu = self.loc
        sigma = self.scale
        n = mu.shape[-1]
        solver = self.solver_method(sigma)
        z = jax.random.normal(key, (n, math.prod(sample_shape)), dtype=mu.dtype)
        x = solver.unwhiten(z)
        return x.T.reshape(sample_shape + (n,)) + mu
