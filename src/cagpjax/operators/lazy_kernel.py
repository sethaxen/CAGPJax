"""Lazy kernel operator"""

import jax
import jax.numpy as jnp
from cola.ops import LinearOperator
from gpjax.kernels import AbstractKernel
from gpjax.kernels.computations import DenseKernelComputation
from jaxtyping import Array, Float


class LazyKernel(LinearOperator):
    """A lazy kernel operator that avoids materializing large kernel matrices.

    This class implements a lazy kernel operator that computes rows/cols of a kernel
    matrix in blocks, preventing memory issues with large datasets.

    Args:
        kernel: The kernel function to use for computations.
        x1: First set of input points for kernel evaluation.
        x2: Second set of input points for kernel evaluation.
        max_memory_mb: Maximum number of megabytes of memory to allocate for batching
            the kernel matrix. If ``batch_size`` is provided, this is ignored.
        batch_size: Number of rows/cols to materialize at once. If ``None``,
            the batch size is determined based on ``max_memory_mb``.
        checkpoint: Whether to checkpoint the computation. This is usually necessary to
            prevent all materialized submatrices from being retained in memory for
            gradient computation.
    """

    kernel: AbstractKernel
    x1: Float[Array, "M D"]
    x2: Float[Array, "N D"]
    batch_size: int | None
    max_memory_mb: int
    checkpoint: bool

    def __init__(
        self,
        kernel: AbstractKernel,
        x1: Float[Array, "M D"],
        x2: Float[Array, "N D"],
        /,
        *,
        max_memory_mb: int = 2**10,  # 1GB
        batch_size: int | None = None,
        checkpoint: bool = False,
    ):
        shape = (x1.shape[0], x2.shape[0])
        dtype = kernel(x1[0, ...], x2[0, ...]).dtype
        super().__init__(dtype=dtype, shape=shape)
        self.kernel = kernel
        self.x1 = x1
        self.x2 = x2
        self._compute_engine = DenseKernelComputation()
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
        self.checkpoint = checkpoint

    @property
    def max_elements(self) -> int:
        """Maximum number of elements to store in memory during matmul operations."""
        element_size = self.dtype.itemsize
        return (self.max_memory_mb * 2**20) // element_size

    @property
    def batch_size_row(self) -> int:
        """Maximum number of rows to materialize at once during right mat(-vec)muls."""
        if self.batch_size is not None:
            return self.batch_size
        return max(1, self.max_elements // self.shape[1])

    @property
    def batch_size_col(self) -> int:
        """Maximum number of columns to materialize at once during left mat(-vec)muls."""
        if self.batch_size is not None:
            return self.batch_size
        return max(1, self.max_elements // self.shape[0])

    def _matmat(self, X: Float[Array, "K M"]) -> Float[Array, "N M"]:
        """Compute K(x1,x2) @ X row-wise to control memory usage."""

        def body_row(x1_row: Float[Array, "D"]) -> Float[Array, "M"]:
            k_chunk = self._compute_engine.cross_covariance(
                self.kernel, x1_row[None, ...], self.x2
            )
            return jnp.squeeze(k_chunk @ X, 0)

        body_fun = jax.checkpoint(body_row) if self.checkpoint else body_row  # pyright: ignore[reportPrivateImportUsage]
        res = jax.lax.map(body_fun, self.x1, batch_size=self.batch_size_row)
        return res

    def _rmatmat(self, X: Float[Array, "M K"]) -> Float[Array, "M K"]:
        """Compute X @ K(x1,x2) col-wise to control memory usage."""

        def body_col(x2_row: Float[Array, "D"]) -> Float[Array, "K"]:
            k_chunk = self._compute_engine.cross_covariance(
                self.kernel, self.x1, x2_row[None, ...]
            )
            return jnp.squeeze(X @ k_chunk, -1)

        body_fun = jax.checkpoint(body_col) if self.checkpoint else body_col  # pyright: ignore[reportPrivateImportUsage]
        res = jax.lax.map(body_fun, self.x2, batch_size=self.batch_size_col).T
        return res
