import cola
import jax
from cola.ops import LinearOperator
from jax import numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import Self, override

from ..linalg.eigh import EighResult, eigh
from ..typing import ScalarFloat
from .base import AbstractLinearSolver, AbstractLinearSolverMethod


class PseudoInverse(AbstractLinearSolverMethod):
    """
    Solve a linear system using the Moore-Penrose pseudoinverse.

    This solver computes the least-squares solution $x = A^+ b$ for any $A$,
    where $A^+$ is the Moore-Penrose pseudoinverse. This is equivalent to
    the exact solution for non-singular $A$ but generalizes to singular $A$
    and improves stability for almost-singular $A$; note, however, that if the
    rank of $A$ is dependent on hyperparameters being optimized, because the
    pseudoinverse is discontinuous, the optimization problem may be ill-posed.

    Attributes:
        rtol: Specifies the cutoff for small eigenvalues.
              Eigenvalues smaller than `rtol * largest_nonzero_eigenvalue` are treated as zero.
              The default is determined based on the floating point precision of the dtype
              of the operator (see [`jax.numpy.linalg.pinv`][]).
        alg: Algorithm for eigenvalue decomposition passed to [`cagpjax.linalg.eigh`][].
    """

    alg: cola.linalg.Algorithm
    rtol: ScalarFloat | None

    def __init__(
        self,
        rtol: ScalarFloat | None = None,
        alg: cola.linalg.Algorithm = cola.linalg.Auto(),
    ):
        self.rtol = rtol
        self.alg = alg

    @override
    def __call__(self, A: LinearOperator) -> AbstractLinearSolver:
        return PseudoInverseSolver(A, rtol=self.rtol, alg=self.alg)


class PseudoInverseSolver(AbstractLinearSolver):
    """
    Solve a linear system using the Moore-Penrose pseudoinverse.
    """

    A: LinearOperator
    eigh_result: EighResult
    eigenvalues_safe: Float[Array, "N"]

    def __init__(
        self,
        A: LinearOperator,
        rtol: ScalarFloat | None = None,
        alg: cola.linalg.Algorithm = cola.linalg.Auto(),
    ):
        n = A.shape[0]
        # select rtol using same heuristic as jax.numpy.linalg.lstsq
        if rtol is None:
            rtol = float(jnp.finfo(A.dtype).eps) * n
        self.eigh_result = eigh(A, alg=alg)
        svdmax = jnp.max(jnp.abs(self.eigh_result.eigenvalues))
        cutoff = jnp.array(rtol * svdmax, dtype=svdmax.dtype)
        mask = self.eigh_result.eigenvalues >= cutoff
        self.eigvals_safe = jnp.where(mask, self.eigh_result.eigenvalues, 1)
        self.eigvals_inv = jnp.where(mask, jnp.reciprocal(self.eigvals_safe), 0)
        self.A = A

    @override
    def solve(self, b: Float[Array, "N #K"]) -> Float[Array, "N #K"]:
        # return jnp.linalg.lstsq(self.A.to_dense(), b)[0]
        b_ndim = b.ndim
        b = b if b_ndim == 2 else b[:, None]
        with jax.default_matmul_precision("highest"):
            x = self.eigh_result.eigenvectors.T @ b
        x = x * self.eigvals_inv[:, None]
        with jax.default_matmul_precision("highest"):
            x = self.eigh_result.eigenvectors @ x
        x = x if b_ndim == 2 else x.squeeze(axis=1)
        return x

    @override
    def logdet(self) -> ScalarFloat:
        return jnp.sum(jnp.log(self.eigvals_safe))

    @override
    def inv_quad(self, b: Float[Array, "N #1"]) -> ScalarFloat:
        z = self.eigh_result.eigenvectors.T @ b
        return jnp.dot(jnp.square(z), self.eigvals_inv).squeeze()

    @override
    def inv_congruence_transform(
        self, B: LinearOperator | Float[Array, "N K"]
    ) -> LinearOperator | Float[Array, "K K"]:
        eigenvectors = self.eigh_result.eigenvectors
        z = B @ eigenvectors
        z = z @ cola.ops.Diagonal(self.eigvals_inv) @ z.T
        return z

    @override
    def trace_solve(self, B: Self) -> ScalarFloat:
        if isinstance(B.eigh_result.eigenvectors, cola.ops.Dense):
            vectors_mat = self.eigh_result.eigenvectors.to_dense()
            return jnp.einsum(
                "ij,j,kj,ik",
                vectors_mat,
                self.eigvals_inv,
                vectors_mat,
                B.A.to_dense(),
            )
        else:
            W = B.eigh_result.eigenvectors.T @ self.eigh_result.eigenvectors.to_dense()
            return jnp.einsum(
                "ij,j,ij,i", W, self.eigvals_inv, W, B.eigh_result.eigenvalues
            )
