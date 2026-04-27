"""Pseodo-input linear solver policy."""

import gpjax
import jax.numpy as jnp
import lineax as lx
import paramax
from jaxtyping import Array, Float, PRNGKeyArray

from .base import AbstractBatchLinearSolverPolicy, ActionOperator


class PseudoInputPolicy(AbstractBatchLinearSolverPolicy):
    """Pseudo-input linear solver policy.

    This policy constructs actions from the cross-covariance between the training inputs and
    pseudo-inputs in the same input space. These pseudo-inputs are conceptually similar to
    inducing points and can be marked as trainable.

    Args:
        pseudo_inputs: Pseudo-inputs for the kernel.
        train_inputs: Training inputs or a dataset containing training inputs. These must be the
            same inputs in the same order as the training data used to condition the CaGP model.
        kernel: Kernel for the GP prior. It must be able to take `train_inputs` and `pseudo_inputs`
            as arguments to its `cross_covariance` method.

    Note:
        When training with many pseudo-inputs, it is common for the cross-covariance matrix to
        become poorly conditioned. Performance can be significantly improved by orthogonalizing
        the actions using an [`OrthogonalizationPolicy`][cagpjax.policies.OrthogonalizationPolicy].
    """

    pseudo_inputs: (
        Float[Array, "M D"] | paramax.AbstractUnwrappable[Float[Array, "M D"]]
    )
    train_inputs: paramax.AbstractUnwrappable[Float[Array, "N D"]]
    kernel: gpjax.kernels.AbstractKernel

    def __init__(
        self,
        pseudo_inputs: Float[Array, "M D"]
        | paramax.AbstractUnwrappable[Float[Array, "M D"]],
        train_inputs_or_dataset: Float[Array, "N D"] | gpjax.dataset.Dataset,
        kernel: gpjax.kernels.AbstractKernel,
    ):
        self.pseudo_inputs = pseudo_inputs
        if isinstance(train_inputs_or_dataset, gpjax.dataset.Dataset):
            train_data = train_inputs_or_dataset
            if train_data.X is None:
                raise ValueError("Dataset must contain training inputs.")
            train_inputs = train_data.X
        else:
            train_inputs = train_inputs_or_dataset
        self.train_inputs = paramax.non_trainable(jnp.atleast_2d(train_inputs))
        self.kernel = kernel
        self.n_actions = paramax.unwrap(self.pseudo_inputs).shape[0]

    def __check_init__(self):
        if (
            paramax.unwrap(self.train_inputs).shape[1:]
            != paramax.unwrap(self.pseudo_inputs).shape[1:]
        ):
            raise ValueError(
                "Training inputs and pseudo-inputs must have the same trailing dimensions."
            )

    def to_actions(
        self, A: ActionOperator, *, key: PRNGKeyArray | None = None
    ) -> ActionOperator:
        del A, key
        S = self.kernel.cross_covariance(
            paramax.unwrap(self.train_inputs), paramax.unwrap(self.pseudo_inputs)
        )
        return lx.MatrixLinearOperator(jnp.asarray(S))
