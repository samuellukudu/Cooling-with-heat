# tests/test_solvers.py
import pytest
import jax
import jax.numpy as jnp
from diffheat.mesh import Grid1D, BoundaryCondition
from diffheat.physics import HeatEquation1D
from diffheat.solvers import explicit_euler_step, solve_heat_1d, check_cfl


class TestCheckCFL:
    def test_stable_dt_passes(self):
        grid = Grid1D.uniform(length=1.0, n_cells=50)
        alpha = 0.01
        dt = 0.9 * grid.dx[0] ** 2 / (2 * alpha)  # 90% of CFL limit
        assert check_cfl(grid, alpha, dt) is True

    def test_unstable_dt_fails(self):
        grid = Grid1D.uniform(length=1.0, n_cells=50)
        alpha = 0.01
        dt = 2.0 * grid.dx[0] ** 2 / (2 * alpha)  # 2x CFL limit
        assert check_cfl(grid, alpha, dt) is False


class TestExplicitEulerStep:
    def test_constant_temperature_stays_constant(self):
        """If T is uniform and no source, it should stay uniform."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([0.5, 0.5]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        T0 = jnp.full(grid.n_cells, 0.5)
        dt = 0.001
        T1 = explicit_euler_step(T0, eqn, t=0.0, dt=dt)
        assert jnp.allclose(T1, T0, atol=1e-10)

    def test_cooling_rod_temperature_decreases(self):
        """Hot rod with cold boundaries should cool down."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([0.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        T0 = jnp.ones(grid.n_cells)
        dt = 0.0001
        T1 = explicit_euler_step(T0, eqn, t=0.0, dt=dt)
        # Average temperature should decrease (heat flows to cold boundaries)
        assert jnp.mean(T1) < jnp.mean(T0)

    def test_source_term_applied(self):
        """Constant source should increase temperature."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        bc = BoundaryCondition(kind="neumann", value=jnp.array([0.0, 0.0]))

        def source(x, t):
            return jnp.ones_like(x)  # uniform heating

        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1, source=source)
        T0 = jnp.zeros(grid.n_cells)
        dt = 0.0001
        T1 = explicit_euler_step(T0, eqn, t=0.0, dt=dt)
        assert jnp.all(T1 > T0)


class TestSolveHeat1D:
    def test_returns_correct_shape(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.01)
        T0 = jnp.zeros(grid.n_cells)
        dt = 0.001
        t_span = (0.0, 0.01)
        trajectory = solve_heat_1d(eqn, T0, t_span, dt)

        n_steps = int((t_span[1] - t_span[0]) / dt) + 1
        assert trajectory.shape == (n_steps, grid.n_cells)

    def test_first_frame_is_initial_condition(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.01)
        T0 = jnp.linspace(1.0, 0.0, grid.n_cells)
        dt = 0.001
        t_span = (0.0, 0.01)
        trajectory = solve_heat_1d(eqn, T0, t_span, dt)
        assert jnp.allclose(trajectory[0], T0)

    def test_approaches_steady_state(self):
        """With Dirichlet BCs, the solution should approach the linear steady state."""
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        T_left, T_right = 1.0, 0.0
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([T_left, T_right]))
        alpha = 1.0
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)

        T0 = jnp.zeros(grid.n_cells)
        dt = 0.9 * grid.dx[0] ** 2 / (2 * alpha)  # near CFL limit for speed
        t_span = (0.0, 0.5)  # long enough to approach steady state
        trajectory = solve_heat_1d(eqn, T0, t_span, dt)

        T_final = trajectory[-1]
        expected_steady = T_left + (T_right - T_left) * grid.centers / grid.length
        # Should be close to steady state (within 5%)
        assert jnp.allclose(T_final, expected_steady, atol=0.05)

    def test_gradient_wrt_alpha(self):
        """Verify that gradients flow through the solver w.r.t. alpha."""
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))

        def loss_fn(alpha):
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
            T0 = jnp.zeros(grid.n_cells)
            dt = 0.001
            t_span = (0.0, 0.005)
            trajectory = solve_heat_1d(eqn, T0, t_span, dt)
            return jnp.mean(trajectory[-1])  # mean final temperature

        grad = jax.grad(loss_fn)(0.1)
        assert not jnp.isclose(grad, 0.0)

    def test_gradient_wrt_boundary_value(self):
        """Verify that gradients flow through the solver w.r.t. boundary temperature."""
        grid = Grid1D.uniform(length=1.0, n_cells=10)

        def loss_fn(left_temp):
            bc = BoundaryCondition(kind="dirichlet", value=jnp.array([left_temp, 0.0]))
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
            T0 = jnp.zeros(grid.n_cells)
            dt = 0.001
            t_span = (0.0, 0.005)
            trajectory = solve_heat_1d(eqn, T0, t_span, dt)
            return jnp.mean(trajectory[-1])

        grad = jax.grad(loss_fn)(1.0)
        # Higher left boundary temp -> higher mean final temp -> positive gradient
        assert grad > 0.0
