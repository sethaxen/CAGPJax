from typing import NamedTuple

import cola
import jax
from cola.ops import LinearOperator
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float
from typing_extensions import override

from ..linalg.eigh import Eigh, EighResult, eigh
from ..typing import ScalarFloat
from .base import AbstractLinearSolver


class PseudoInverseState(NamedTuple):
    A: LinearOperator
    eigh_result: EighResult
    eigvals_mask: Bool[Array, "N"]
    eigvals_safe: Float[Array, "N"]
    eigvals_inv: Float[Array, "N"]


class PseudoInverse(AbstractLinearSolver[PseudoInverseState]):
    """
    Solve a linear system using the Moore-Penrose pseudoinverse.

    This solver computes the least-squares solution $x = A^+ b$ for any $A$,
    where $A^+$ is the Moore-Penrose pseudoinverse. This is equivalent to
    the exact solution for non-singular $A$ but generalizes to singular $A$
    and improves stability for almost-singular $A$; note, however, that if the
    rank of $A$ is dependent on hyperparameters being optimized, because the
    pseudoinverse is discontinuous, the optimization problem may be ill-posed.

    Note that if $A$ is (almost-)degenerate (some eigenvalues repeat), then
    the gradient of its solves in JAX may be non-computable or numerically unstable
    (see [jax#669](https://github.com/jax-ml/jax/issues/669)).
    For degenerate operators, it may be necessary to increase `grad_rtol` to improve
    stability of gradients.
    See [`cagpjax.linalg.eigh`][] for more details.

    Attributes:
        rtol: Specifies the cutoff for small eigenvalues.
              Eigenvalues smaller than `rtol * largest_nonzero_eigenvalue` are treated as zero.
              The default is determined based on the floating point precision of the dtype
              of the operator (see [`jax.numpy.linalg.pinv`][]).
        grad_rtol: Specifies the cutoff for similar eigenvalues, used to improve
            gradient computation for (almost-)degenerate matrices.
            If not provided, the default is 0.0.
            If None or negative, all eigenvalues are treated as distinct.
        alg: Algorithm for eigenvalue decomposition passed to [`cagpjax.linalg.eigh`][].
    """

    rtol: ScalarFloat | None
    grad_rtol: float | None
    alg: cola.linalg.Algorithm

    def __init__(
        self,
        rtol: ScalarFloat | None = None,
        grad_rtol: float | None = None,
        alg: cola.linalg.Algorithm = Eigh(),
    ):
        self.rtol = rtol
        self.grad_rtol = grad_rtol
        self.alg = alg

    @override
    def init(self, A: LinearOperator) -> PseudoInverseState:
        n = A.shape[0]
        # select rtol using same heuristic as jax.numpy.linalg.lstsq
        rtol_val = (
            self.rtol if self.rtol is not None else float(jnp.finfo(A.dtype).eps) * n
        )
        eigh_result = eigh(A, alg=self.alg, grad_rtol=self.grad_rtol)
        svdmax = jnp.max(jnp.abs(eigh_result.eigenvalues))
        cutoff = jnp.array(rtol_val * svdmax, dtype=svdmax.dtype)
        eigvals_mask = eigh_result.eigenvalues >= cutoff
        eigvals_safe = jnp.where(eigvals_mask, eigh_result.eigenvalues, 1)
        eigvals_inv = jnp.where(eigvals_mask, jnp.reciprocal(eigvals_safe), 0)
        return PseudoInverseState(
            A, eigh_result, eigvals_mask, eigvals_safe, eigvals_inv
        )

    @override
    def unwhiten(
        self, state: PseudoInverseState, z: Float[Array, "N #K"]
    ) -> Float[Array, "N #K"]:
        eigvals_sqrt = jnp.where(state.eigvals_mask, jnp.sqrt(state.eigvals_safe), 0)
        x = (eigvals_sqrt * z.T).T
        with jax.default_matmul_precision("highest"):
            return state.eigh_result.eigenvectors @ x

    @override
    def solve(
        self, state: PseudoInverseState, b: Float[Array, "N #K"]
    ) -> Float[Array, "N #K"]:
        b_ndim = b.ndim
        b = b if b_ndim == 2 else b[:, None]
        with jax.default_matmul_precision("highest"):
            x = state.eigh_result.eigenvectors.T @ b
        x = x * state.eigvals_inv[:, None]
        with jax.default_matmul_precision("highest"):
            x = state.eigh_result.eigenvectors @ x
        x = x if b_ndim == 2 else x.squeeze(axis=1)
        return x

    @override
    def logdet(self, state: PseudoInverseState) -> ScalarFloat:
        return jnp.sum(jnp.log(state.eigvals_safe))

    @override
    def inv_quad(
        self, state: PseudoInverseState, b: Float[Array, "N #1"]
    ) -> ScalarFloat:
        z = state.eigh_result.eigenvectors.T @ b
        return jnp.dot(jnp.square(z), state.eigvals_inv).squeeze()

    @override
    def inv_congruence_transform(
        self, state: PseudoInverseState, B: LinearOperator | Float[Array, "K N"]
    ) -> LinearOperator | Float[Array, "K K"]:
        eigenvectors = state.eigh_result.eigenvectors
        z = eigenvectors.T @ B
        z = z.T @ cola.ops.Diagonal(state.eigvals_inv) @ z
        return z

    @override
    def trace_solve(
        self, state: PseudoInverseState, state_other: PseudoInverseState
    ) -> ScalarFloat:
        if isinstance(state_other.eigh_result.eigenvectors, cola.ops.Dense):
            vectors_mat = state.eigh_result.eigenvectors.to_dense()
            return jnp.einsum(
                "ij,j,kj,ik",
                vectors_mat,
                state.eigvals_inv,
                vectors_mat,
                state_other.A.to_dense(),
            )
        else:
            W = (
                state_other.eigh_result.eigenvectors.T
                @ state.eigh_result.eigenvectors.to_dense()
            )
            return jnp.einsum(
                "ij,j,ij,i", W, state.eigvals_inv, W, state_other.eigvals_safe
            )
