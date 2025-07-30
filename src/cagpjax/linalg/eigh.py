"""Hermitian eigenvalue decomposition."""

import warnings
from functools import partial

import cola
import jax
import matfree.decomp
from cola.ops import Diagonal, I_like, Identity, LinearOperator, ScalarMul
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing_extensions import NamedTuple


class EighResult(NamedTuple):
    """Result of Hermitian eigenvalue decomposition.

    Attributes:
        eigenvalues: Eigenvalues of the operator.
        eigenvectors: Eigenvectors of the operator.
    """

    eigenvalues: Float[Array, "N"]
    eigenvectors: LinearOperator


class Eigh(cola.linalg.Algorithm):
    """
    Eigh algorithm for eigenvalue decomposition.
    """


class Lanczos(cola.linalg.Algorithm):
    """Lanczos algorithm for approximate partial eigenvalue decomposition.

    Args:
        max_iters: Maximum number of iterations (number of eigenvalues/vectors to compute).
            If `None`, all eigenvalues/eigenvectors are computed.
        v0: Initial vector. If `None`, a random vector is generated using `key`.
        key: Random key for generating a random initial vector if `v0` is
            not provided.
    """

    max_iters: int | None
    v0: Float[Array, "N"] | None
    key: PRNGKeyArray | None

    def __init__(
        self,
        max_iters: int | None = None,
        /,
        *,
        v0: Float[Array, "N"] | None = None,
        key: PRNGKeyArray | None = None,
    ):
        self.max_iters = max_iters
        self.v0 = v0
        self.key = key


def eigh(
    A: LinearOperator,
    alg: cola.linalg.Algorithm = Eigh(),
    grad_rtol: float | None = None,
) -> EighResult:
    """Compute the Hermitian eigenvalue decomposition of a linear operator.

    For some algorithms, the decomposition may be approximate or partial.

    Args:
        A: Hermitian linear operator.
        alg: Algorithm for eigenvalue decomposition.
        grad_rtol: Specifies the cutoff for similar eigenvalues, used to improve
            gradient computation for (almost-)degenerate matrices.
            If not provided, the default is 0.0.
            If None or negative, all eigenvalues are treated as distinct.

    Returns:
        A named tuple of `(eigenvalues, eigenvectors)` where `eigenvectors` is a
            (semi-)orthogonal `LinearOperator`.

    Note:
        Degenerate matrices have repeated eigenvalues.
        The set of eigenvectors that correspond to the same eigenvalue is not unique
        but instead forms a subspace.
        `grad_rtol` only improves stability of gradient-computation if the function
        being differentiated depends only depends on these subspaces and not the
        specific eigenvectors themselves.
    """
    if grad_rtol is None:
        grad_rtol = -1.0
    vals, vecs = _eigh(A, alg, grad_rtol)  # pyright: ignore[reportArgumentType]
    if vecs.shape[-1] == A.shape[-1]:
        vecs = cola.Unitary(vecs)
    else:
        vecs = cola.Stiefel(vecs)
    return EighResult(vals, vecs)


@cola.dispatch(precedence=-2)
def _eigh(A: LinearOperator, alg: cola.linalg.Algorithm, grad_rtol: float):  # pyright: ignore[reportRedeclaration]
    warnings.warn("grad_rtol not supported for cola's eigh algorithms.")
    return cola.linalg.eig(cola.SelfAdjoint(A), A.shape[0], which="SM", alg=alg)


@cola.dispatch(precedence=-1)
def _eigh(A: LinearOperator, alg: Eigh, grad_rtol: float):  # pyright: ignore[reportRedeclaration]
    vals, vecs = _eigh_safe(A.to_dense(), grad_rtol=grad_rtol)
    return vals, cola.lazify(vecs)


@cola.dispatch(precedence=-1)
def _eigh(A: LinearOperator, alg: Lanczos, grad_rtol: float):  # pyright: ignore[reportRedeclaration]
    if alg.v0 is None:
        key = jax.random.key(0) if alg.key is None else alg.key
        v0 = jax.random.normal(key, (A.shape[0],), dtype=A.dtype)
    else:
        v0 = alg.v0

    num_matvecs = alg.max_iters if alg.max_iters is not None else A.shape[0]

    # Set up Lanczos algorithm
    tridiag_sym = matfree.decomp.tridiag_sym(
        num_matvecs, materialize=True, reortho="full"
    )

    # Define matrix-vector product
    def matvec(v):
        return (A @ v).astype(A.dtype)

    # Tridiagonalize the matrix using the Lanczos algorithm
    Q, H, *_ = tridiag_sym(matvec, v0)

    # Diagonalize the tridiagonal matrix
    vals, vecs = _eigh_safe(H, grad_rtol=grad_rtol)
    vecs = Q @ vecs
    return vals, cola.lazify(vecs)


@cola.dispatch
def _eigh(
    A: ScalarMul | Diagonal | Identity, alg: cola.linalg.Algorithm, grad_rtol: float
):
    return cola.linalg.diag(A), I_like(A)


# Eigh with custom vjp to expand its support to (almost-)degenerate matrices.
# jax-ml/jax#669


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _eigh_safe(a: Float[Array, "N N"], grad_rtol: float):
    return jnp.linalg.eigh(a, symmetrize_input=True)


def _eigh_safe_fwd(a: Float[Array, "N N"], grad_rtol: float):
    eigh_result = _eigh_safe(a, grad_rtol)
    return eigh_result, eigh_result


def _eigh_safe_rev(
    grad_rtol: float,
    residual: tuple[Float[Array, "N"], Float[Array, "N N"]],
    grad: tuple[Float[Array, "N"], Float[Array, "N N"]],
):
    grad_eigvals, grad_eigvecs = grad
    eigvals, eigvecs = residual

    dot = partial(jnp.dot, precision=jax.lax.Precision.HIGHEST)
    vt_grad_v = dot(eigvecs.T, grad_eigvecs)
    # the backward part of input-symmetrization skew-symmetrizes this part
    vt_grad_v = 0.5 * (vt_grad_v - vt_grad_v.T)

    with jax.numpy_rank_promotion("allow"):
        w_diff = eigvals[..., None, :] - eigvals[..., None]

        if grad_rtol >= 0.0:
            # If eigenvalues (i,j) are (approximately) equal, then grad_a_v[i,j] is
            # underdetermined unless vt_grad_v[i,j]==0, which implies that the downstream
            # code is only sensitive to the subspace spanned by eigenvectors (i,j) not
            # the eigenvectors themselves.
            w_thresh = (
                grad_rtol
                if grad_rtol == 0.0
                else jnp.abs(eigvals).max(axis=-1) * grad_rtol
            )
            grad_thresh = (
                grad_rtol if grad_rtol == 0.0 else jnp.abs(vt_grad_v).max() * grad_rtol
            )
            inv_mask = (
                (jnp.abs(w_diff) <= w_thresh) & (jnp.abs(vt_grad_v) <= grad_thresh)
            ).astype(eigvals.dtype)
        else:
            inv_mask = jnp.eye(eigvals.shape[-1], dtype=eigvals.dtype)

        Fmat = jnp.reciprocal(w_diff + inv_mask) - inv_mask

    grad_a_v = vt_grad_v * Fmat
    grad_a_v = grad_a_v.at[jnp.diag_indices_from(grad_a_v)].set(grad_eigvals)
    grad_a = dot(eigvecs, dot(grad_a_v, eigvecs.T))

    return (grad_a,)


_eigh_safe.defvjp(_eigh_safe_fwd, _eigh_safe_rev)
