"""Hermitian eigenvalue decomposition."""

from functools import partial

import cola
import jax
from cola.ops import Diagonal, I_like, Identity, LinearOperator, ScalarMul
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing_extensions import NamedTuple, overload

from ..typing import ScalarFloat
from .utils import _add_jitter


class EighResult(NamedTuple):
    """Result of Hermitian eigenvalue decomposition.

    Attributes:
        eigenvalues: Eigenvalues of the operator.
        eigenvectors: Eigenvectors of the operator.
    """

    eigenvalues: Float[Array, "N"]
    eigenvectors: LinearOperator


def eigh(
    A: LinearOperator,
    alg: cola.linalg.Algorithm = cola.linalg.Auto(),
    key: PRNGKeyArray | None = None,
) -> EighResult:
    """Compute the Hermitian eigenvalue decomposition of a linear operator.

    Args:
        A: Hermitian linear operator.
        alg: Algorithm for eigenvalue decomposition (see [`cola.linalg.eig`][]).

    Returns:
        A named tuple of `(eigenvalues, eigenvectors)` where `eigenvectors` is a unitary
            `LinearOperator`.
    """

    vals, vecs = _eigh(A, alg)
    return EighResult(vals, cola.Unitary(vecs))


@overload
def _eigh(A: LinearOperator, alg: cola.linalg.Algorithm):  # pyright: ignore[reportOverlappingOverload]
    A = cola.SelfAdjoint(A)
    return cola.linalg.eig(A, A.shape[0], which="LM", alg=alg)


@overload
def _eigh(A: ScalarMul | Diagonal | Identity, alg: cola.linalg.Algorithm):  # pyright: ignore[reportOverlappingOverload]
    return cola.linalg.diag(A), I_like(A)


@cola.dispatch
def _eigh(A: Any, alg: cola.linalg.Algorithm):
    pass
# Eigh with custom vjp to expand its support to (almost-)degenerate matrices.
# jax-ml/jax#669

@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _eigh_safe(a: Float[Array, "N N"], grad_rtol: Float[Array, ""]):
    return jnp.linalg.eigh(a, symmetrize_input=True)


def _eigh_safe_fwd(a: Float[Array, "N N"], grad_rtol: Float[Array, ""]):
    eigh_result = _eigh_safe(a, grad_rtol)
    return eigh_result, eigh_result


def _eigh_safe_rev(
    grad_rtol: Float[Array, ""],
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
