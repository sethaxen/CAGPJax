"""Hermitian eigenvalue decomposition."""

from functools import partial
from typing import Any

import jax
import lineax as lx
import matfree.decomp
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
    eigenvectors: Any


class Eigh:
    """
    Eigh algorithm for eigenvalue decomposition.
    """


class Lanczos:
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
    A: Any,
    alg: Any = Eigh(),
    grad_rtol: float | None = None,
) -> EighResult:
    """Compute the Hermitian eigenvalue decomposition of a linear operator.

    For some algorithms, the decomposition may be approximate or partial.

    Args:
        A: Hermitian linear operator.
        alg: Algorithm for eigenvalue decomposition.
        grad_rtol: Specifies the cutoff for similar eigenvalues, used to improve
            gradient computation for (almost-)degenerate matrices.
            If None (default), all eigenvalues are treated as distinct.

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
    elif grad_rtol < 0.0:
        raise ValueError("grad_rtol must be None or non-negative.")
    vals, vecs = _eigh(A, alg, grad_rtol)
    return EighResult(vals, vecs)


def _lineax_operator_dense_matrix_dtype(A_lx: lx.AbstractLinearOperator):
    """Dtype that Lineax uses for a dense view of the operator (``as_matrix``).

    ``lx.diagonal`` can disagree (e.g. ``IdentityLinearOperator`` uses
    ``jnp.ones(n)`` in ``diagonal`` but ``result_type`` of ``in_structure`` in
    ``as_matrix``). This matches ``IdentityLinearOperator.as_matrix`` in Lineax.
    https://github.com/patrick-kidger/lineax/blob/eb8cb0d5b44d5cf14532bd26913fe726f0216eb1/lineax/_operator.py#L788-L796
    """
    leaves = jax.tree_util.tree_leaves(A_lx.in_structure())
    with jax.numpy_dtype_promotion("standard"):
        if len(leaves) == 0:
            return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        return jnp.result_type(*leaves)


def _eigh(A: Any, alg: Any, grad_rtol: float):
    A_lx = A if isinstance(A, lx.AbstractLinearOperator) else lx.MatrixLinearOperator(A)

    if lx.is_diagonal(A_lx):
        dtype = _lineax_operator_dense_matrix_dtype(A_lx)
        vals = lx.diagonal(A_lx).astype(dtype)
        vecs = lx.IdentityLinearOperator(A_lx.in_structure())
        return vals, vecs

    if isinstance(alg, Eigh):
        vals, vecs = _eigh_safe(A_lx.as_matrix(), grad_rtol=grad_rtol)
        return vals, lx.MatrixLinearOperator(vecs)

    if isinstance(alg, Lanczos):
        return _eigh_lanczos(A_lx, alg, grad_rtol, A)

    raise ValueError(f"Unsupported algorithm: {alg}")


def _eigh_lanczos(
    A: lx.AbstractLinearOperator, alg: Lanczos, grad_rtol: float, A_original: Any
):
    if alg.v0 is None:
        key = jax.random.key(0) if alg.key is None else alg.key
        v0 = jax.random.normal(key, (A.in_size(),), dtype=A.in_structure().dtype)
    else:
        v0 = alg.v0

    n = A.in_size()
    num_matvecs = n if alg.max_iters is None else min(alg.max_iters, n)

    # Set up Lanczos algorithm
    tridiag_sym = matfree.decomp.tridiag_sym(
        num_matvecs, materialize=True, reortho="full"
    )

    def matvec(v):
        return A.mv(v)

    # Tridiagonalize the matrix using the Lanczos algorithm
    Q, H, *_ = tridiag_sym(matvec, v0)

    # Diagonalize the tridiagonal matrix
    vals, vecs = _eigh_safe(H, grad_rtol=grad_rtol)
    # matfree stacks Lanczos vectors along axis 0, so Q has shape
    # (num_matvecs, n); transpose to get the (n, num_matvecs) eigenvector matrix.
    vecs = Q.T @ vecs
    return vals, lx.MatrixLinearOperator(vecs)


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
            # use 1.0 as reference for rtol check for gradient so that we don't
            # count a very low (e.g. 1e-30) value as being greater than zero
            # when the largest gradient value is also very low.
            inv_mask = (
                (jnp.abs(w_diff) <= w_thresh) & (jnp.abs(vt_grad_v) <= grad_rtol)
            ).astype(eigvals.dtype)
        else:
            inv_mask = jnp.eye(eigvals.shape[-1], dtype=eigvals.dtype)

        Fmat = jnp.reciprocal(w_diff + inv_mask) - inv_mask

    grad_a_v = vt_grad_v * Fmat
    grad_a_v = grad_a_v.at[jnp.diag_indices_from(grad_a_v)].set(grad_eigvals)
    grad_a = dot(eigvecs, dot(grad_a_v, eigvecs.T))

    return (grad_a,)


_eigh_safe.defvjp(_eigh_safe_fwd, _eigh_safe_rev)
