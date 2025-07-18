"""Hermitian eigenvalue decomposition."""

from typing import Any

import cola
from cola.ops import Diagonal, I_like, Identity, LinearOperator, ScalarMul
from jaxtyping import Array, Float
from typing_extensions import NamedTuple, overload


class EighResult(NamedTuple):
    """Result of Hermitian eigenvalue decomposition.

    Attributes:
        eigenvalues: Eigenvalues of the operator.
        eigenvectors: Eigenvectors of the operator.
    """

    eigenvalues: Float[Array, "N"]
    eigenvectors: LinearOperator


def eigh(
    A: LinearOperator, alg: cola.linalg.Algorithm = cola.linalg.Auto()
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
