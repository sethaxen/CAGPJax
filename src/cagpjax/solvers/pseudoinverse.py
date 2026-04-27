from typing import Any, NamedTuple

import equinox as eqx
import jax
import lineax as lx
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float
from typing_extensions import override

from ..interop import lazify, to_lineax
from ..linalg.eigh import Eigh, EighResult, eigh
from ..typing import ScalarFloat
from .base import AbstractLinearSolver, LinearOperatorLike, SupportsDenseOperator


class PseudoInverseState(NamedTuple):
    A: LinearOperatorLike
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
            If None (default), all eigenvalues are treated as distinct.
        alg: Algorithm for eigenvalue decomposition passed to [`cagpjax.linalg.eigh`][].
    """

    rtol: ScalarFloat | None = eqx.field(static=True, default=None)
    grad_rtol: float | None = eqx.field(static=True, default=None)
    alg: Any = eqx.field(static=True, default_factory=Eigh)

    def __check_init__(self):
        if self.rtol is not None and self.rtol < 0:
            raise ValueError("rtol must be non-negative")
        if self.grad_rtol is not None and self.grad_rtol < 0:
            raise ValueError("grad_rtol must be non-negative")

    @staticmethod
    def _to_dense_matrix(A: Any) -> Float[Array, "N M"]:
        if isinstance(A, lx.AbstractLinearOperator):
            return A.as_matrix()
        if hasattr(A, "to_dense"):
            return jnp.asarray(A.to_dense())
        if hasattr(A, "as_matrix"):
            return jnp.asarray(A.as_matrix())
        return jnp.asarray(A)

    @override
    def init(self, A: LinearOperatorLike) -> PseudoInverseState:
        A_lx = to_lineax(A)
        n = A_lx.in_size()
        dtype = A_lx.in_structure().dtype
        # select rtol using same heuristic as jax.numpy.linalg.lstsq
        rtol_val = (
            self.rtol if self.rtol is not None else float(jnp.finfo(dtype).eps) * n
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
        eigenvectors = self._to_dense_matrix(state.eigh_result.eigenvectors)
        with jax.default_matmul_precision("highest"):
            return eigenvectors @ x

    @override
    def solve(
        self, state: PseudoInverseState, b: Float[Array, "N #K"]
    ) -> Float[Array, "N #K"]:
        b_ndim = b.ndim
        b = b if b_ndim == 2 else b[:, None]
        eigenvectors = self._to_dense_matrix(state.eigh_result.eigenvectors)
        with jax.default_matmul_precision("highest"):
            x = eigenvectors.T @ b
        x = x * state.eigvals_inv[:, None]
        with jax.default_matmul_precision("highest"):
            x = eigenvectors @ x
        x = x if b_ndim == 2 else x.squeeze(axis=1)
        return x

    @override
    def logdet(self, state: PseudoInverseState) -> ScalarFloat:
        return jnp.sum(jnp.log(state.eigvals_safe))

    @override
    def inv_quad(
        self, state: PseudoInverseState, b: Float[Array, "N #1"]
    ) -> ScalarFloat:
        eigenvectors = self._to_dense_matrix(state.eigh_result.eigenvectors)
        z = eigenvectors.T @ b
        return jnp.dot(jnp.square(z), state.eigvals_inv).squeeze()

    @override
    def inv_congruence_transform(
        self, state: PseudoInverseState, B: LinearOperatorLike | Float[Array, "K N"]
    ) -> LinearOperatorLike | Float[Array, "K K"]:
        eigenvectors = self._to_dense_matrix(state.eigh_result.eigenvectors)
        B_is_operator = (
            isinstance(B, lx.AbstractLinearOperator)
            or isinstance(B, SupportsDenseOperator)
            or hasattr(B, "as_matrix")
        )
        B_mat = self._to_dense_matrix(B) if B_is_operator else jnp.asarray(B)
        z = eigenvectors.T @ B_mat
        z_weighted = (
            state.eigvals_inv * z if z.ndim == 1 else state.eigvals_inv[:, None] * z
        )
        result = z.T @ z_weighted
        return lazify(result) if B_is_operator else result

    @override
    def trace_solve(
        self, state: PseudoInverseState, state_other: PseudoInverseState
    ) -> ScalarFloat:
        solved = self.solve(state, self._to_dense_matrix(state_other.A))
        return jnp.trace(solved)
