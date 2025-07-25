"""Tests for the linear solvers."""

import cola
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from cagpjax.linalg import congruence_transform
from cagpjax.operators import diag_like
from cagpjax.solvers import Cholesky, PseudoInverse
from cagpjax.solvers.cholesky import CholeskySolver
from cagpjax.solvers.pseudoinverse import PseudoInverseSolver

jax.config.update("jax_enable_x64", True)


class TestSolvers:
    """
    Test the solvers for the linear system of equations Ax = b.
    """

    @pytest.fixture(params=[4, 10])
    def n(self, request) -> int:
        """Size of the square operator A."""
        return request.param

    @pytest.fixture(params=[jnp.float32, jnp.float64])
    def dtype(self, request):
        """Data type of the operator A."""
        return request.param

    @pytest.fixture(params=["nonsingular", "singular", "almost_singular"])
    def op_type(self, request) -> str:
        """Type of the operator (how close to singular it is)."""
        return request.param

    def random_semiorthogonal(self, key, n, dtype) -> Float[Array, "N N"]:
        """Generate a random semiorthogonal matrix."""
        A = jax.random.normal(key, (n, n), dtype=dtype)
        Q, _ = jnp.linalg.qr(A)
        return Q

    @pytest.fixture(params=[23, 98, 30])
    def op(self, request, n, dtype, op_type) -> cola.ops.LinearOperator:
        """Generate a random linear operator operator of the specified type."""
        key = jax.random.key(request.param)
        key, subkey = jax.random.split(key)
        eigenvectors = self.random_semiorthogonal(subkey, n, dtype)

        n_small = 0 if op_type == "nonsingular" else 2
        eigenvalues = jax.random.uniform(key, (n - n_small,), dtype=dtype)
        if op_type == "almost_singular":
            eigmin = jnp.full(
                n_small, jnp.finfo(dtype).eps * jnp.max(eigenvalues) / 10.0
            )
        else:
            eigmin = jnp.zeros(n_small, dtype=dtype)
        eigenvalues = jnp.concatenate([eigenvalues, eigmin])
        A = eigenvectors.T @ jnp.diag(eigenvalues) @ eigenvectors
        A = cola.lazify(A)
        if op_type == "singular":
            return cola.SelfAdjoint(A)
        else:
            return cola.PSD(A)

    @pytest.fixture(params=[PseudoInverse, Cholesky])
    def solver_method(self, request, op_type, dtype):
        """Return a solver method for the linear system of equations op @ x = b."""
        if request.param == PseudoInverse:
            return PseudoInverse(rtol=0.0 if op_type == "nonsingular" else None)
        elif request.param == Cholesky:
            if dtype == jnp.float32:
                pytest.skip("Cholesky is not very numerically stable for float32")
            jitter = 0.0 if op_type == "nonsingular" else 1e-6
            return Cholesky(jitter)

    @pytest.fixture
    def solver(self, solver_method, op):
        """Return a solver for the linear system of equations op @ x = b."""
        return solver_method(op)

    @pytest.fixture(params=[cola.ops.Dense, cola.ops.Diagonal])
    def other_op(
        self, request, n, dtype, key=jax.random.key(78)
    ) -> cola.ops.LinearOperator:
        """Generate another PSD linear operator for the trace solve test."""
        if request.param == cola.ops.Dense:
            A = jax.random.normal(key, (n, n), dtype=dtype)
            A = cola.lazify(A @ A.T)
            return cola.PSD(A)
        elif request.param == cola.ops.Diagonal:
            return cola.PSD(
                cola.ops.Diagonal(jax.random.normal(key, (n,), dtype=dtype) ** 2)
            )

    @pytest.fixture
    def other_solver(self, solver_method, other_op):
        """Return a solver for the linear system of equations other_op @ x = b."""
        if isinstance(solver_method, Cholesky):
            # Because the operator is PSD, we can set jitter to 0.0.
            return Cholesky()(other_op)
        return solver_method(other_op)

    def test_solver_construction(self, solver_method, op):
        """Test the construction of the solver."""
        solver = solver_method(op)
        if isinstance(solver_method, Cholesky):
            assert isinstance(solver, CholeskySolver)
            jitter = solver_method.jitter
            op_actual = op if jitter is None else op + diag_like(op, jitter)
            assert jnp.allclose(
                (solver.lchol @ solver.lchol.T).to_dense(), op_actual.to_dense()
            )
        elif isinstance(solver_method, PseudoInverse):
            assert isinstance(solver, PseudoInverseSolver)
            assert solver.A is op

    @pytest.mark.parametrize("tail_shape", [(), (2,)])
    def test_solve(
        self, tail_shape, solver_method, solver, op, n, dtype, key=jax.random.key(90)
    ):
        """Test the solve method of the solver."""
        b = jax.random.normal(key, (n, *tail_shape), dtype=dtype)
        x = solver.solve(b)
        assert x.shape == b.shape
        assert x.dtype == b.dtype

        if isinstance(solver_method, Cholesky):
            jitter = solver_method.jitter
            op_actual = op if jitter is None else op + diag_like(op, jitter)
        else:
            op_actual = op

        x_lstsq = jnp.linalg.lstsq(op_actual.to_dense(), b)[0]
        # increased rtol for float32 because lstsq and eigh use different algorithms,
        # and this makes more of a difference for float32
        rtol = (1e-3 if dtype == jnp.float32 else 1e-8) * n
        assert jnp.allclose(x, x_lstsq, rtol=rtol)

    @pytest.mark.parametrize("m", [2, 5])
    @pytest.mark.parametrize("B_type", [jnp.ndarray, cola.ops.LinearOperator])
    def test_inv_congruence_transform_consistency(
        self, solver, n, m, dtype, B_type, key=jax.random.key(23)
    ):
        """Test inv_congruence_transform is consistent with solve and congruence_transform."""
        B = jax.random.normal(key, (m, n), dtype=dtype)
        if B_type == cola.ops.LinearOperator:
            B = cola.lazify(B)

        with jax.default_matmul_precision("highest"):
            cong_transform = solver.inv_congruence_transform(B)
            op_mat_inv = solver.solve(jnp.eye(n, dtype=dtype))
            cong_transform_ref = congruence_transform(B, op_mat_inv)
            cong_trans_mat = cola.densify(cong_transform)
            cong_trans_ref_mat = cola.densify(cong_transform_ref)

        assert cong_transform.shape == (m, m)
        assert cong_transform.dtype == dtype
        rtol = 1e-4 if dtype == jnp.float32 else 1e-12
        assert isinstance(cong_transform, B_type)
        assert jnp.allclose(cong_trans_mat, cong_trans_ref_mat, rtol=rtol)

    def test_inv_quad_consistency(self, solver, n, dtype, key=jax.random.key(23)):
        """Test inv_quad is consistent with solve."""
        b = jax.random.normal(key, (n,), dtype=dtype)

        inv_quad = solver.inv_quad(b)
        assert jnp.isscalar(inv_quad)
        assert inv_quad.dtype == dtype
        assert jnp.isclose(inv_quad, jnp.dot(b, solver.solve(b)))

    def test_trace_solve_consistency(self, solver, other_solver, other_op, n, dtype):
        """Test trace_solve is consistent with solve."""
        trace_solve = solver.trace_solve(other_solver)
        assert jnp.isscalar(trace_solve)
        assert trace_solve.dtype == dtype
        trace_solve_solve = jnp.trace(solver.solve(other_op.to_dense()))
        rtol = (1e-4 if dtype == jnp.float32 else 1e-12) * n
        assert jnp.isclose(trace_solve, trace_solve_solve, rtol=rtol)

    @pytest.mark.parametrize("jitter", [None, 1e-6])
    def test_pseudoinverse_gradient_degenerate(self, n, dtype, jitter):
        """Test gradient computation with degenerate operators.

        Without jitter, gradients contain NaN. With jitter, gradients are finite.
        """
        A = cola.ops.Dense(jnp.eye(n, dtype=dtype))

        if dtype == jnp.float32 and jitter is not None:
            # Use larger jitter for float32 due to lower precision
            jitter *= 100

        def loss_fn(A_matrix):
            A_op = cola.ops.Dense(A_matrix)
            solver = PseudoInverse(jitter=jitter)(A_op)
            b = jnp.ones(n, dtype=dtype)
            x = solver.solve(b)
            return jnp.sum(x**2)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(A.to_dense())

        if jitter is None:
            assert not jnp.isfinite(grad).all()
        else:
            assert jnp.isfinite(grad).all()
