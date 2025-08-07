"""Orthogonalization methods."""

from enum import Enum
from typing import Any

import cola
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import overload

from ..operators.annotations import ScaledOrthogonal
from ..operators.block_diagonal_sparse import BlockDiagonalSparse


class OrthogonalizationMethod(Enum):
    """Methods for orthogonalizing a matrix."""

    QR = "qr"
    """Householder QR decomposition"""
    CGS = "cgs"
    """Classical Gram–Schmidt orthogonalization"""
    MGS = "mgs"
    """Modified Gram–Schmidt orthogonalization"""


def orthogonalize(
    A: Float[Array, "m n"] | cola.ops.LinearOperator,
    /,
    method: OrthogonalizationMethod = OrthogonalizationMethod.QR,
    n_reortho: int = 0,
) -> Float[Array, "m n"] | cola.ops.LinearOperator:
    """
    Orthogonalize the operator using the specified method.

    The columns of the resulting matrix should span a (super-)space of the columns of
    the input matrix and be mutually orthogonal. For column-rank-deficient matrices,
    some methods (e.g. Gram-Schmidt variants) may include columns of norm 0.

    Args:
        A: The operator to orthogonalize.
        method: The method to use for orthogonalization.
        n_reortho: The number of times to _re_-orthogonalize each column.
            Reorthogonalizing once is generally sufficient to improve orthogonality
            for Gram-Schmidt variants
            (see e.g. [10.1007/s00211-005-0615-4](https://doi.org/10.1007/s00211-005-0615-4)).

    Returns:
        The orthogonalized operator. If the input is a LinearOperator, then so is the output.
    """
    return _orthogonalize(A, method, n_reortho)


def _get_return_operator_annotation(method: OrthogonalizationMethod):
    match method:
        case OrthogonalizationMethod.QR:
            return cola.Stiefel
        case OrthogonalizationMethod.CGS:
            return ScaledOrthogonal
        case OrthogonalizationMethod.MGS:
            return ScaledOrthogonal


@overload
def _orthogonalize(
    A: Float[Array, "m n"],
    method: OrthogonalizationMethod,
    n_reortho: int,
) -> Float[Array, "m n"]:
    if n_reortho < 0:
        raise ValueError("n_reortho must be non-negative")
    A = jnp.asarray(A)
    match method:
        case OrthogonalizationMethod.QR:
            return _qr_q(A, n_reortho)
        case OrthogonalizationMethod.CGS:
            return _classical_gram_schmidt(A, n_reortho)
        case OrthogonalizationMethod.MGS:
            return _modified_gram_schmidt(A, n_reortho)


@overload
def _orthogonalize(
    A: cola.ops.LinearOperator,
    method: OrthogonalizationMethod,
    n_reortho: int,
) -> cola.ops.LinearOperator:
    if A.isa(cola.Stiefel) or A.isa(ScaledOrthogonal):
        return A

    Q = cola.lazify(orthogonalize(cola.densify(A), method=method, n_reortho=n_reortho))
    annotation = _get_return_operator_annotation(method)
    return annotation(Q)


@overload
def _orthogonalize(  # pyright: ignore[reportOverlappingOverload]
    A: cola.ops.Identity | cola.ops.Diagonal | cola.ops.ScalarMul,
    method: OrthogonalizationMethod,
    n_reortho: int,
) -> cola.ops.Identity:
    return cola.Unitary(cola.ops.I_like(A))


@cola.dispatch
def _orthogonalize(A: Any, method: OrthogonalizationMethod, n_reortho: int) -> Any:
    pass


def _qr_q(A: Float[Array, "m n"], n_reortho: int) -> Float[Array, "m n"]:
    """Orthogonalization using Householder QR decomposition."""
    Q = A
    for _ in range(n_reortho + 1):
        Q = jnp.linalg.qr(Q)[0]
    return Q


def _classical_gram_schmidt(
    A: Float[Array, "m n"], n_reortho: int
) -> Float[Array, "m n"]:
    """Classical Gram–Schmidt orthogonalization."""

    # NOTE: CGS seems to need an rtol>0 to avoid numerical instability for ill-conditioned matrices
    rtol = float(jnp.finfo(A.dtype).eps * 10)
    n_cols = A.shape[1]

    def body(k, carry):
        Q, is_finalized = carry
        q_k = Q[:, k]

        # (re)orthogonalize against all finalized columns
        for _ in range(n_reortho + 1):
            proj_weights = jnp.where(is_finalized, Q.T @ q_k, 0.0)
            q_k -= Q @ proj_weights

        # finalize current column
        Q = Q.at[:, k].set(_l2_normalize(q_k, rtol=rtol))
        is_finalized = is_finalized.at[k].set(True)

        return Q, is_finalized

    is_finalized = jnp.full((n_cols,), False)
    Q, _ = jax.lax.fori_loop(0, n_cols, body, (A, is_finalized))

    return Q


def _modified_gram_schmidt(
    A: Float[Array, "m n"], n_reortho: int
) -> Float[Array, "m n"]:
    """Modified Gram–Schmidt orthogonalization."""

    n_cols = A.shape[1]

    def body_reortho(v, args):
        q_j, do_reortho = args
        v = jnp.where(do_reortho, v - jnp.vdot(q_j, v) * q_j, v)
        return v, None

    def body(k, carry):
        Q, is_finalized = carry
        q_k = Q[:, k]

        # reorthogonalize against all finalized columns
        for _ in range(n_reortho):
            q_k, _ = jax.lax.scan(body_reortho, q_k, (Q.T, is_finalized))

        # finalize current column
        q_k = _l2_normalize(q_k)
        is_finalized = is_finalized.at[k].set(True)

        # orthogonalize all non-finalized columns against current column
        # (faster than using the nested scan used during *re*orthogonalization)
        proj_weights = jnp.asarray(jnp.where(is_finalized, 0.0, Q.T @ q_k))
        Q -= jnp.outer(q_k, proj_weights)
        Q = Q.at[:, k].set(q_k)

        return Q, is_finalized

    is_finalized = jnp.full((n_cols,), False)
    Q, _ = jax.lax.fori_loop(0, n_cols, body, (A, is_finalized))

    return Q


def _l2_normalize(x: Float[Array, "n"], /, *, rtol: float = 0.0) -> Float[Array, "n"]:
    """Safely L2-normalize a vector.

    This function prevents division by zero when normalizing a vector of zero norm.
    Because the differential of the normalization is undefined at zero norm, we use
    a sub-differential of the normalization which sets the differential of the argument
    to zero.
    """
    r_cutoff = rtol * x.size
    r = jnp.linalg.norm(x)
    return jnp.asarray(jnp.where(r <= r_cutoff, jax.lax.stop_gradient(x), x / r))
