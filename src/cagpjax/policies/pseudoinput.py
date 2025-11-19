"""Pseodo-input linear solver policy."""

import cola
import gpjax
import jax.numpy as jnp
from cola.ops import LinearOperator
from flax import nnx
from gpjax.parameters import Parameter
from jaxtyping import Array, Float

from .base import AbstractBatchLinearSolverPolicy


class PseudoInputPolicy(AbstractBatchLinearSolverPolicy):
    """Pseudo-input linear solver policy.

    This policy constructs actions from the cross-covariance between the training inputs and
    pseudo-inputs in the same input space. These pseudo-inputs are conceptually similar to
    inducing points and can be marked as trainable.

    Args:
        pseudo_inputs: Pseudo-inputs for the kernel. If wrapped as a `gpjax.parameters.Parameter`,
            they will be treated as trainable.
        train_inputs: Training inputs or a dataset containing training inputs. These must be the
            same inputs in the same order as the training data used to condition the CaGP model.
        kernel: Kernel for the GP prior. It must be able to take `train_inputs` and `pseudo_inputs`
            as arguments to its `cross_covariance` method.

    Note:
        When training with many pseudo-inputs, it is common for the cross-covariance matrix to
        become poorly conditioned. Performance can be significantly improved by orthogonalizing
        the actions using an [`OrthogonalizationPolicy`][cagpjax.policies.OrthogonalizationPolicy].
    """

    pseudo_inputs: Float[Array, "M D"] | Parameter[Float[Array, "M D"]]
    train_inputs: Float[Array, "N D"]
    kernel: gpjax.kernels.AbstractKernel

    def __init__(
        self,
        pseudo_inputs: Float[Array, "M D"] | Parameter[Float[Array, "M D"]],
        train_inputs_or_dataset: Float[Array, "N D"] | gpjax.dataset.Dataset,
        kernel: gpjax.kernels.AbstractKernel,
    ):
        if isinstance(train_inputs_or_dataset, gpjax.dataset.Dataset):
            train_data = train_inputs_or_dataset
            if train_data.X is None:
                raise ValueError("Dataset must contain training inputs.")
            train_inputs = train_data.X
        else:
            train_inputs = train_inputs_or_dataset
        self.pseudo_inputs = pseudo_inputs
        self.train_inputs = jnp.atleast_2d(train_inputs)
        self.kernel = kernel

    @property
    def n_actions(self):
        return self.pseudo_inputs.shape[0]

    @property
    def _pseudo_inputs(self) -> Float[Array, "M D"]:
        if isinstance(self.pseudo_inputs, Parameter):
            return self.pseudo_inputs.value
        else:
            return self.pseudo_inputs

    def to_actions(self, A: LinearOperator) -> LinearOperator:
        S = self.kernel.cross_covariance(self.train_inputs, self._pseudo_inputs)
        return cola.lazify(S)
