from typing import Any

import cola.ops
import jax.numpy as jnp
from cola.ops import LinearOperator
from typing_extensions import overload

try:  # support for GPJax v0.12.0
    import gpjax.linalg  # pyright: ignore[reportMissingImports]

    @overload
    def lazify(A: gpjax.linalg.Dense) -> cola.ops.Dense:  # pyright: ignore[reportOverlappingOverload]
        return cola.ops.Dense(A.array)

    @overload
    def lazify(A: gpjax.linalg.Diagonal) -> cola.ops.Diagonal:  # pyright: ignore[reportOverlappingOverload]
        return cola.ops.Diagonal(A.diagonal)

    @overload
    def lazify(A: gpjax.linalg.Identity) -> cola.ops.Identity:  # pyright: ignore[reportOverlappingOverload]
        return cola.ops.Identity(A.shape, A.dtype)

    @overload
    def lazify(A: gpjax.linalg.Triangular) -> cola.ops.Triangular:  # pyright: ignore[reportOverlappingOverload]
        if A.lower:
            return cola.ops.Triangular(jnp.tril(A.array), lower=True)
        else:
            return cola.ops.Triangular(jnp.triu(A.array), lower=False)

except ImportError:
    pass


@overload
def lazify(A: Any) -> cola.ops.LinearOperator:  # pyright: ignore[reportOverlappingOverload]
    return cola.lazify(A)


@cola.dispatch
def lazify(A: Any) -> Any:
    pass
