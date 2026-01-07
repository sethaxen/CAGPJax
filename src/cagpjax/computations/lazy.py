"""Lazy kernel computation."""

import cola
import jax.numpy as jnp
from gpjax.kernels import AbstractKernel
from gpjax.kernels.computations import AbstractKernelComputation, DenseKernelComputation
from jaxtyping import Array, Float
from typing_extensions import override

from ..operators import LazyKernel


class LazyKernelComputation(AbstractKernelComputation):
    """Lazy kernel computation.

    Compute the kernel matrix lazily, so that at most one submatrix of the kernel matrix
    is retained in memory at any time.

    The cross-covariance and Gram matrices are represented by a
    [`LazyKernel`][cagpjax.operators.LazyKernel] operator.

    Args:
        batch_size: The number of rows/cols to materialize at once. If ``None``, the batch
            size is determined based on ``max_memory_mb``.
        max_memory_mb: The maximum number of megabytes of memory to allocate for batching
            the kernel matrix. If ``batch_size`` is provided, this is ignored.
        checkpoint: Whether to checkpoint the computation. `checkpoint=True` is usually
            necessary to prevent all materialized submatrices from being retained in memory
            for gradient computation. However, this generally increases the computation time.

    Note:
        This class technically violates the API for `AbstractKernelComputation`, which
        expects that the return type of `cross_covariance` is an array, not a
        `LinearOperator`. While this class works as expected within this package, it
        should not be be used within GPJax itself.

    Examples
    --------

    We can construct a kernel with a `LazyKernelComputation` to avoid materializing
    the full kernel matrix in memory.

    ```python
    >>> from gpjax.kernels import RBF
    >>> from cagpjax.computations import LazyKernelComputation
    >>>
    >>> # Create a kernel that will be lazily evaluated
    >>> compute_engine = LazyKernelComputation(max_memory_mb=2**10)  # 1GB
    >>> kernel = RBF(compute_engine=compute_engine)
    ```

    If we want to combine multiple kernels (e.g. for a product kernel), then we need to
    set the `compute_engine` attribute of the outermost kernel.

    ```python
    >>> from gpjax.kernels import RBF, Matern32, ProductKernel
    >>> from cagpjax.computations import LazyKernelComputation
    >>>
    >>> # Create a kernel that will be lazily evaluated
    >>> compute_engine = LazyKernelComputation(max_memory_mb=2**10)  # 1GB
    >>> kernel1 = RBF()
    >>> kernel2 = Matern32()
    >>> prod_kernel = kernel1 * kernel2
    >>> prod_kernel.compute_engine = compute_engine
    >>> # We can also explicitly construct the product kernel with a compute engine
    >>> prod_kernel = ProductKernel(kernels=[kernel1, kernel2], compute_engine=compute_engine)
    ```
    """

    batch_size: int | None
    max_memory_mb: int
    checkpoint: bool

    def __init__(
        self,
        *,
        batch_size: int | None = None,
        max_memory_mb: int = 2**10,  # 1GB
        checkpoint: bool = False,
    ):
        """Initialize the lazy kernel computation."""
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.checkpoint = checkpoint

    @override
    def gram(self, kernel: AbstractKernel, x: Float[Array, "N D"]) -> LazyKernel:  # pyright: ignore[reportIncompatibleMethodOverride]
        return cola.PSD(
            LazyKernel(
                kernel,
                x,
                x,
                batch_size=self.batch_size,
                max_memory_mb=self.max_memory_mb,
                checkpoint=self.checkpoint,
            )
        )

    @override
    def cross_covariance(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        kernel: AbstractKernel,
        x1: Float[Array, "N D"],
        x2: Float[Array, "M D"],
    ) -> Float[Array, "N M"] | LazyKernel:
        return LazyKernel(
            kernel,
            x1,
            x2,
            batch_size=self.batch_size,
            max_memory_mb=self.max_memory_mb,
            checkpoint=self.checkpoint,
        )
