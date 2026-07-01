"""Non-materializing application of a linear operator to a dense matrix.

``lineax``'s own operator composition (``A @ B`` for two
``lineax.AbstractLinearOperator``s) densifies both operands as soon as
``.as_matrix()`` is called on the result, since ``ComposedLinearOperator.as_matrix``
is implemented as ``operator1.as_matrix() @ operator2.as_matrix()``. That is fine
when both operands are small, but is exactly the wrong thing to do when one
operand is a large, lazily-represented operator (e.g. :class:`~cagpjax.operators.LazyKernel`)
being applied to a thin matrix (e.g. a handful of action columns): the whole
point of such operators is to avoid ever forming their dense representation.

:func:`apply_operator_right` and :func:`apply_operator_left` instead apply an
operator to a dense matrix via ``mv``, which every ``lineax.AbstractLinearOperator``
(including composites like ``AddLinearOperator``/``ComposedLinearOperator``, which
delegate to their sub-operators' ``mv``) already implements without materializing.
This works uniformly for any conforming operator, not just operators defined in
this package.
"""

from typing import Any

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


def apply_operator_right(A: Any, B: Float[Array, "n k"]) -> Float[Array, "m k"]:
    """Compute ``A @ B`` without materializing ``A`` when it is a large operator.

    Args:
        A: Linear operator or array, the (potentially large) left operand.
        B: Dense array, the (thin) right operand.
    """
    B = jnp.asarray(B)
    if isinstance(A, lx.AbstractLinearOperator):
        if lx.is_diagonal(A):
            diag = lx.diagonal(A)
            return diag[:, None] * B if B.ndim == 2 else diag * B
        if B.ndim == 1:
            return A.mv(B)
        return jax.vmap(A.mv, in_axes=1, out_axes=1)(B)
    return jnp.asarray(A) @ B


def apply_operator_left(B: Float[Array, "k n"], A: Any) -> Float[Array, "k m"]:
    """Compute ``B @ A`` without materializing ``A`` when it is a large operator.

    Args:
        B: Dense array, the (thin) left operand.
        A: Linear operator or array, the (potentially large) right operand.
    """
    A_T = (
        A.transpose() if isinstance(A, lx.AbstractLinearOperator) else jnp.asarray(A).T
    )
    return apply_operator_right(A_T, jnp.asarray(B).T).T
