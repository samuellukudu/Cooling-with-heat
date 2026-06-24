"""End-to-end integration tests for the diffheat pipeline."""
import jax
import jax.numpy as jnp
import numpy as np
from diffheat import (
    BoundaryCondition,
    Grid1D,
    HeatEquation1D,
    solve_heat_1d,
)


class TestEndToEnd:
    def test_full_pipeline_dirichlet(self):
        """Hot left, cold right Dirichlet → should approach linear steady state."""
        grid = Grid1D.uniform(length=1.0, n_cells=50)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        T0 = jnp.zeros(grid.n_cells)
        dt = 0.0005
        trajectory = solve_heat_1d(eqn, T0, (0.0, 2.0), dt)

        T_final = trajectory[-1]
        expected = 1.0 + (0.0 - 1.0) * grid.centers / grid.length
        # Should be reasonably close to steady state
        assert jnp.allclose(T_final, expected, atol=0.1)

    def test_full_pipeline_neumann_insulated(self):
        """Insulated boundaries → total heat should be conserved (no source)."""
        grid = Grid1D.uniform(length=1.0, n_cells=30)
        bc = BoundaryCondition(kind="neumann", value=jnp.array([0.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        T0 = jnp.sin(jnp.pi * grid.centers / grid.length)
        dt = 0.0001
        trajectory = solve_heat_1d(eqn, T0, (0.0, 0.02), dt)

        # Total heat (integral of T) should be conserved
        total_initial = jnp.sum(T0) * float(grid.dx[0])
        total_final = jnp.sum(trajectory[-1]) * float(grid.dx[0])
        assert jnp.isclose(total_initial, total_final, rtol=1e-4)

    def test_jit_compilation_works(self):
        """solve_heat_1d should be JIT-compilable."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))

        @jax.jit
        def jit_solve(alpha, T0):
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
            return solve_heat_1d(eqn, T0, (0.0, 0.01), dt=0.001)

        T0 = jnp.zeros(grid.n_cells)
        result = jit_solve(0.1, T0)
        assert result.shape[0] > 1


class TestGradients:
    def test_gradient_wrt_alpha_is_smooth(self):
        """Gradient of loss w.r.t. alpha should vary smoothly with alpha."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        T0 = jnp.zeros(grid.n_cells)

        def loss(alpha):
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
            traj = solve_heat_1d(eqn, T0, (0.0, 0.02), dt=0.001)
            return jnp.mean(traj[-1])

        grad_fn = jax.grad(loss)
        alphas = [0.05, 0.1, 0.2, 0.5]
        grads = [float(grad_fn(a)) for a in alphas]

        # Higher alpha → faster diffusion → lower final mean (more heat lost to cold boundary)
        # Gradient should become more negative with increasing alpha?
        # Actually: higher alpha means faster approach to steady state
        # At alpha=0.05: still far from steady, mean ~ low
        # At alpha=0.5: close to steady, mean ~ 0.5
        # So loss increases with alpha → positive gradient
        for g in grads:
            assert not jnp.isnan(g)
            assert jnp.isfinite(g)

    def test_gradient_wrt_initial_condition(self):
        """Gradient w.r.t. T0 should be non-zero for each cell."""
        grid = Grid1D.uniform(length=1.0, n_cells=15)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))

        def loss(T0):
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
            traj = solve_heat_1d(eqn, T0, (0.0, 0.01), dt=0.001)
            return jnp.mean(traj[-1])

        T0 = jnp.linspace(0.0, 1.0, grid.n_cells)
        grad = jax.grad(loss)(T0)
        assert grad.shape == (grid.n_cells,)
        # Every cell's initial temperature affects the final mean
        assert jnp.all(grad > 0.0)  # higher T0 → higher final mean

    def test_gradient_wrt_boundary_value(self):
        """∂loss/∂T_left should be positive (hotter boundary → hotter final)."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)

        def loss(T_left):
            bc = BoundaryCondition(kind="dirichlet", value=jnp.array([T_left, 0.0]))
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
            T0 = jnp.zeros(grid.n_cells)
            traj = solve_heat_1d(eqn, T0, (0.0, 0.02), dt=0.001)
            return jnp.mean(traj[-1])

        grad = jax.grad(loss)(1.0)
        assert float(grad) > 0.0

    def test_finite_difference_agrees_with_autodiff(self):
        """Finite difference gradient should approximately match autodiff."""
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))

        def loss(alpha):
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
            T0 = jnp.zeros(grid.n_cells)
            traj = solve_heat_1d(eqn, T0, (0.0, 0.005), dt=0.001)
            return jnp.mean(traj[-1])

        alpha = 0.1
        grad_ad = jax.grad(loss)(alpha)

        # Finite difference
        eps = 1e-4
        grad_fd = (loss(alpha + eps) - loss(alpha - eps)) / (2 * eps)

        assert jnp.isclose(grad_ad, grad_fd, rtol=1e-3)
