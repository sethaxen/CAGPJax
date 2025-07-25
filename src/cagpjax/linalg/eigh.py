"""Hermitian eigenvalue decomposition."""

from typing import Any

import cola
import jax
from cola.ops import Diagonal, I_like, Identity, LinearOperator, ScalarMul
from jax import numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing_extensions import NamedTuple, overload

from ..operators import diag_like
from ..typing import ScalarFloat


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
    jitter: ScalarFloat | Float[Array, "N"] | None = None,
    key: PRNGKeyArray | None = None,
) -> EighResult:
    """Compute the Hermitian eigenvalue decomposition of a linear operator.

    Args:
        A: Hermitian linear operator.
        alg: Algorithm for eigenvalue decomposition (see [`cola.linalg.eig`][]).
        jitter: Jitter to add to diagonal to stabilize gradients for
            (almost-)degenerate matrices. If scalar, random jitter in `[0, jitter]`
            is generated. If vector, it's added directly.
        key: Random key for jitter generation. Defaults to `key(0)` if not provided.

    Returns:
        A named tuple of `(eigenvalues, eigenvectors)` where `eigenvectors` is a unitary
            `LinearOperator`.
    """
    if jitter is not None:
        if jnp.isscalar(jitter):
            if key is None:
                key = jax.random.key(0)
            n = A.shape[0]
            jitter_values = jax.random.uniform(key, (n,), dtype=A.dtype, maxval=jitter)
        else:
            jitter_values = jnp.asarray(jitter, dtype=A.dtype)
        A = _add_jitter(A, jitter_values)

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


# fallback implementation
@overload
def _add_jitter(A: LinearOperator, jitter: Float[Array, "N"]) -> LinearOperator:
    return A + diag_like(A, jitter)


@overload
def _add_jitter(  # pyright: ignore[reportOverlappingOverload]
    A: ScalarMul | Diagonal | Identity, jitter: Float[Array, "N"]
) -> Diagonal:
    return Diagonal(cola.linalg.diag(A) + jitter)


@cola.dispatch
def _add_jitter(A: Any, jitter: Any) -> Any:
    pass
