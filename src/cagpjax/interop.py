"""Interop utilities for normalizing operator-like inputs to Lineax."""

from typing import Any

import jax.numpy as jnp
import lineax as lx


def to_lineax(A: Any) -> lx.AbstractLinearOperator:
    """Convert an array or lineax operator into a lineax operator."""
    if isinstance(A, lx.AbstractLinearOperator):
        return A
    return lx.MatrixLinearOperator(jnp.asarray(A))
