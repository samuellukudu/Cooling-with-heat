# tests/test_solvers_2d.py
"""Tests for 2D time integration solvers."""
import jax
import jax.numpy as jnp
import pytest
from diffheat.mesh import Grid2D, BoundaryCondition2D
from diffheat.operators import laplacian_2d
from diffheat.mesh.boundary import apply_boundary_conditions_2d


def _make_heat_rhs(alpha, bc):
    """Factory: build a heat equation RHS function for a given diffusivity and BCs."""
    def rhs(T, grid, t, params):
        L_T_mod, b_source = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, bc, T
        )
        return alpha * (L_T_mod + b_source)
    return rhs


class TestCheckCFL2D:
    def test_stable_dt_passes(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=50, ny=50)
        from diffheat.solvers import check_cfl_2d
        alpha = 0.01
        dx_min = float(jnp.min(grid.dx))
        dy_min = float(jnp.min(grid.dy))
        cfl_limit = min(dx_min**2, dy_min**2) / (4 * alpha)
        dt = 0.9 * cfl_limit
        assert check_cfl_2d(grid, alpha, dt) is True

    def test_unstable_dt_fails(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=50, ny=50)
        from diffheat.solvers import check_cfl_2d
        alpha = 0.01
        dx_min = float(jnp.min(grid.dx))
        dy_min = float(jnp.min(grid.dy))
        cfl_limit = min(dx_min**2, dy_min**2) / (4 * alpha)
        dt = 2.0 * cfl_limit
        assert check_cfl_2d(grid, alpha, dt) is False


class TestExplicitEulerStep2D:
    def test_constant_temperature_stays_constant(self):
        """Uniform temperature with equal Dirichlet BCs should not change."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 0.5},
            right={"kind": "dirichlet", "value": 0.5},
            bottom={"kind": "dirichlet", "value": 0.5},
            top={"kind": "dirichlet", "value": 0.5},
        )
        alpha = 0.1
        rhs_fn = _make_heat_rhs(alpha, bc)
        T0 = jnp.full((20, 20), 0.5)

        from diffheat.solvers import explicit_euler_step_2d
        T1 = explicit_euler_step_2d(T0, rhs_fn, grid, t=0.0, dt=0.001)
        assert jnp.allclose(T1, T0, atol=1e-10)

    def test_cooling_plate_temperature_decreases(self):
        """Hot plate with cold boundaries should cool down."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.0},
            top={"kind": "dirichlet", "value": 0.0},
        )
        alpha = 0.1
        rhs_fn = _make_heat_rhs(alpha, bc)
        T0 = jnp.ones((20, 20))

        from diffheat.solvers import explicit_euler_step_2d
        T1 = explicit_euler_step_2d(T0, rhs_fn, grid, t=0.0, dt=0.0001)
        assert jnp.mean(T1) < jnp.mean(T0)

    def test_with_params_dict(self):
        """Params dict should be passed through to rhs_fn."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=10)
        bc = BoundaryCondition2D(
            left={"kind": "neumann", "value": 0.0},
            right={"kind": "neumann", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )

        def rhs_fn(T, grid, t, params):
            alpha = params["alpha"]
            L_T_mod, b_source = apply_boundary_conditions_2d(
                lambda x: laplacian_2d(x, grid), grid, bc, T
            )
            return alpha * (L_T_mod + b_source)

        # Non-uniform initial condition so diffusion actually occurs
        X, Y = grid.X, grid.Y
        T0 = jnp.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.05)

        from diffheat.solvers import explicit_euler_step_2d
        T1_slow = explicit_euler_step_2d(T0, rhs_fn, grid, t=0.0, dt=0.001, params={"alpha": 0.01})
        T1_fast = explicit_euler_step_2d(T0, rhs_fn, grid, t=0.0, dt=0.001, params={"alpha": 0.1})
        # Higher alpha = more diffusion = faster change away from initial
        assert jnp.sum(jnp.abs(T1_fast - T0)) > jnp.sum(jnp.abs(T1_slow - T0))


class TestSolve2D:
    def test_returns_correct_shape(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=15)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        alpha = 0.01
        rhs_fn = _make_heat_rhs(alpha, bc)
        T0 = jnp.zeros((10, 15))
        dt = 0.001
        t_span = (0.0, 0.01)

        from diffheat.solvers import solve_2d
        trajectory = solve_2d(rhs_fn, T0, grid, t_span, dt)

        n_steps = int((t_span[1] - t_span[0]) / dt) + 1
        assert trajectory.shape == (n_steps, 10, 15)

    def test_first_frame_is_initial_condition(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=10)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        alpha = 0.01
        rhs_fn = _make_heat_rhs(alpha, bc)
        T0 = jnp.linspace(1.0, 0.0, 100).reshape(10, 10)

        from diffheat.solvers import solve_2d
        trajectory = solve_2d(rhs_fn, T0, grid, (0.0, 0.01), dt=0.001)
        assert jnp.allclose(trajectory[0], T0)

    def test_gradient_wrt_alpha(self):
        """Gradients flow through solve_2d w.r.t. alpha."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=10)

        def loss_fn(alpha):
            bc = BoundaryCondition2D(
                left={"kind": "dirichlet", "value": 1.0},
                right={"kind": "dirichlet", "value": 0.0},
                bottom={"kind": "neumann", "value": 0.0},
                top={"kind": "neumann", "value": 0.0},
            )
            rhs_fn = _make_heat_rhs(alpha, bc)
            T0 = jnp.zeros((10, 10))
            from diffheat.solvers import solve_2d
            traj = solve_2d(rhs_fn, T0, grid, (0.0, 0.005), dt=0.001)
            return jnp.mean(traj[-1])

        grad = jax.grad(loss_fn)(0.1)
        assert not jnp.isclose(grad, 0.0)
