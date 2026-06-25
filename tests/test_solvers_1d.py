# tests/test_solvers_1d.py
"""Tests for generic 1D time integration solvers."""
import jax
import jax.numpy as jnp
import pytest
from diffheat.mesh import Grid1D, BoundaryCondition
from diffheat.operators import laplacian_1d
from diffheat.mesh.boundary import apply_boundary_conditions_1d


def _make_heat_rhs_1d(alpha, bc):
    """Factory: build a 1D heat equation RHS function."""
    def rhs(T, grid, t, params):
        L_T, b_source = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, bc, T
        )
        return alpha * (L_T + b_source)
    return rhs


class TestExplicitEulerStep1D:
    def test_constant_temperature_stays_constant(self):
        """Uniform temperature with equal Dirichlet BCs should not change."""
        n = 20
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([0.5, 0.5]))
        alpha = 0.1
        rhs_fn = _make_heat_rhs_1d(alpha, bc)
        T0 = jnp.full(n, 0.5)

        from diffheat.solvers import explicit_euler_step_1d
        T1 = explicit_euler_step_1d(T0, rhs_fn, grid, t=0.0, dt=0.001)
        assert jnp.allclose(T1, T0, atol=1e-6)

    def test_cooling_rod_temperature_decreases(self):
        """Hot rod with cold boundaries should cool down."""
        n = 20
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([0.0, 0.0]))
        alpha = 0.1
        rhs_fn = _make_heat_rhs_1d(alpha, bc)
        T0 = jnp.ones(n)

        from diffheat.solvers import explicit_euler_step_1d
        T1 = explicit_euler_step_1d(T0, rhs_fn, grid, t=0.0, dt=0.0001)
        assert jnp.mean(T1) < jnp.mean(T0)

    def test_with_params_dict(self):
        """Params dict should be passed through to rhs_fn."""
        n = 20
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="neumann", value=jnp.array([0.0, 0.0]))

        def rhs_fn(T, grid, t, params):
            alpha = params["alpha"]
            L_T, b_source = apply_boundary_conditions_1d(
                lambda x: laplacian_1d(x, grid), grid, bc, T
            )
            return alpha * (L_T + b_source)

        T0 = jnp.exp(-((grid.centers - 0.5) ** 2) / 0.05)

        from diffheat.solvers import explicit_euler_step_1d
        T1_slow = explicit_euler_step_1d(
            T0, rhs_fn, grid, t=0.0, dt=0.001, params={"alpha": 0.01}
        )
        T1_fast = explicit_euler_step_1d(
            T0, rhs_fn, grid, t=0.0, dt=0.001, params={"alpha": 0.1}
        )
        # Higher alpha = more diffusion = faster change away from initial
        assert jnp.sum(jnp.abs(T1_fast - T0)) > jnp.sum(jnp.abs(T1_slow - T0))


class TestSolve1D:
    def test_returns_correct_shape(self):
        n = 15
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        alpha = 0.01
        rhs_fn = _make_heat_rhs_1d(alpha, bc)
        T0 = jnp.zeros(n)
        dt = 0.001
        t_span = (0.0, 0.01)

        from diffheat.solvers import solve_1d
        trajectory = solve_1d(rhs_fn, T0, grid, t_span, dt)

        n_steps = int((t_span[1] - t_span[0]) / dt) + 1
        assert trajectory.shape == (n_steps, n)

    def test_first_frame_is_initial_condition(self):
        n = 15
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        alpha = 0.01
        rhs_fn = _make_heat_rhs_1d(alpha, bc)
        T0 = jnp.linspace(1.0, 0.0, n)

        from diffheat.solvers import solve_1d
        trajectory = solve_1d(rhs_fn, T0, grid, (0.0, 0.01), dt=0.001)
        assert jnp.allclose(trajectory[0], T0)

    def test_gradient_wrt_alpha(self):
        """Gradients flow through solve_1d w.r.t. alpha."""
        n = 15
        grid = Grid1D.uniform(length=1.0, n_cells=n)

        def loss_fn(alpha):
            bc = BoundaryCondition(
                kind="dirichlet", value=jnp.array([1.0, 0.0])
            )
            rhs_fn = _make_heat_rhs_1d(alpha, bc)
            T0 = jnp.zeros(n)
            from diffheat.solvers import solve_1d
            traj = solve_1d(rhs_fn, T0, grid, (0.0, 0.005), dt=0.001)
            return jnp.mean(traj[-1])

        grad = jax.grad(loss_fn)(0.1)
        assert not jnp.isclose(grad, 0.0)

    def test_raises_on_too_short_t_span(self):
        n = 10
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        rhs_fn = _make_heat_rhs_1d(0.01, bc)
        T0 = jnp.zeros(n)

        from diffheat.solvers import solve_1d
        with pytest.raises(ValueError):
            solve_1d(rhs_fn, T0, grid, (0.0, 0.0001), dt=0.001)


class TestBackwardCompatibility:
    """Verify existing heat-specific functions still work as before."""

    def test_explicit_euler_step_still_works(self):
        from diffheat.physics import HeatEquation1D
        from diffheat.solvers import explicit_euler_step

        n = 20
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)
        T0 = jnp.zeros(n)

        T1 = explicit_euler_step(T0, eqn, t=0.0, dt=0.0001)
        assert T1.shape == (n,)
        # Temperature should increase (heat flowing in from left Dirichlet=1)
        assert jnp.mean(T1) > jnp.mean(T0)

    def test_solve_heat_1d_still_works(self):
        from diffheat.physics import HeatEquation1D
        from diffheat.solvers import solve_heat_1d

        n = 15
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.01)
        T0 = jnp.zeros(n)

        traj = solve_heat_1d(eqn, T0, (0.0, 0.01), dt=0.001)
        n_steps = int(0.01 / 0.001) + 1
        assert traj.shape == (n_steps, n)

    def test_solve_heat_1d_gradient(self):
        """Gradients flow through solve_heat_1d (backward compat)."""
        from diffheat.physics import HeatEquation1D
        from diffheat.solvers import solve_heat_1d

        n = 15
        grid = Grid1D.uniform(length=1.0, n_cells=n)

        def loss_fn(alpha):
            bc = BoundaryCondition(
                kind="dirichlet", value=jnp.array([1.0, 0.0])
            )
            eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
            T0 = jnp.zeros(n)
            traj = solve_heat_1d(eqn, T0, (0.0, 0.005), dt=0.001)
            return jnp.mean(traj[-1])

        grad = jax.grad(loss_fn)(0.1)
        assert not jnp.isclose(grad, 0.0)

    def test_explicit_euler_step_matches_old_behavior(self):
        """New wrapper should match old matrix-based implementation output."""
        from diffheat.physics import HeatEquation1D, apply_boundary_conditions
        from diffheat.solvers import explicit_euler_step
        from diffheat.operators import make_laplacian

        n = 20
        grid = Grid1D.uniform(length=1.0, n_cells=n)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        eqn = HeatEquation1D(grid=grid, bc=bc, alpha=0.1)

        # Old matrix-based approach for comparison
        L = make_laplacian(grid)
        L_mod, b_source_old = apply_boundary_conditions(L, grid, bc)
        T_old = jnp.zeros(n)
        dT_dt_old = eqn.alpha * (L_mod @ T_old + b_source_old)
        T_old_next = T_old + 0.0001 * dT_dt_old

        # New wrapper approach
        T_new = explicit_euler_step(jnp.zeros(n), eqn, t=0.0, dt=0.0001)

        # Both should produce the same result for the first step
        # (allow small float32 differences from field vs matrix approach)
        assert jnp.allclose(T_new, T_old_next, atol=1e-4)
