# tests/test_solvers_3d.py
"""Tests for 3D time integration solvers."""
import jax
import jax.numpy as jnp
import pytest
from diffheat.mesh import Grid3D, BoundaryCondition3D
from diffheat.operators.laplacian import laplacian_3d
from diffheat.mesh.boundary import apply_boundary_conditions_3d


def _make_heat_rhs_3d(alpha, bc):
    """Factory: build a 3D heat equation RHS for a given diffusivity and BCs."""
    def rhs(T, grid, t, params):
        L_T_mod, b_source = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, bc, T
        )
        return alpha * (L_T_mod + b_source)
    return rhs


def _all_dirichlet(v):
    d = {"kind": "dirichlet", "value": v}
    return BoundaryCondition3D(left=d, right=d, bottom=d, top=d, front=d, back=d)


def _all_neumann(v=0.0):
    n = {"kind": "neumann", "value": v}
    return BoundaryCondition3D(left=n, right=n, bottom=n, top=n, front=n, back=n)


class TestCheckCFL3D:
    def test_stable_dt_passes(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=20, nz=20)
        from diffheat.solvers import check_cfl_3d
        alpha = 0.01
        h = float(jnp.min(grid.dx))
        cfl_limit = h ** 2 / (6 * alpha)
        dt = 0.9 * cfl_limit
        assert check_cfl_3d(grid, alpha, dt) is True

    def test_unstable_dt_fails(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=20, nz=20)
        from diffheat.solvers import check_cfl_3d
        alpha = 0.01
        h = float(jnp.min(grid.dx))
        cfl_limit = h ** 2 / (6 * alpha)
        dt = 2.0 * cfl_limit
        assert check_cfl_3d(grid, alpha, dt) is False

    def test_anisotropic_grid_uses_smallest_cell(self):
        """Should use the tightest (smallest) spacing."""
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=10, ny=10, nz=10)
        from diffheat.solvers import check_cfl_3d
        alpha = 0.01
        # dx = 0.1, dy = 0.2, dz = 0.3  → limit = 0.01 / (6*0.01)
        h_min = float(jnp.min(grid.dx))  # 0.1
        cfl_limit = h_min ** 2 / (6 * alpha)
        assert check_cfl_3d(grid, alpha, 0.9 * cfl_limit) is True
        assert check_cfl_3d(grid, alpha, 2.0 * cfl_limit) is False


class TestExplicitEulerStep3D:
    def test_constant_temperature_stays_constant(self):
        """Uniform temperature with equal Dirichlet BCs should not change."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        bc = _all_dirichlet(0.5)
        alpha = 0.1
        rhs_fn = _make_heat_rhs_3d(alpha, bc)
        T0 = jnp.full((8, 8, 8), 0.5)

        from diffheat.solvers import explicit_euler_step_3d
        T1 = explicit_euler_step_3d(T0, rhs_fn, grid, t=0.0, dt=1e-4)
        assert jnp.allclose(T1, T0, atol=1e-9)

    def test_hot_box_with_cold_walls_cools(self):
        """Hot interior with cold Dirichlet walls should lose heat."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        bc = _all_dirichlet(0.0)
        alpha = 0.1
        rhs_fn = _make_heat_rhs_3d(alpha, bc)
        T0 = jnp.ones((8, 8, 8))

        from diffheat.solvers import explicit_euler_step_3d
        T1 = explicit_euler_step_3d(T0, rhs_fn, grid, t=0.0, dt=1e-4)
        assert jnp.mean(T1) < jnp.mean(T0)

    def test_with_params_dict(self):
        """Params dict should be passed through to rhs_fn."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=6, ny=6, nz=6)
        bc = _all_neumann(0.0)

        def rhs_fn(T, grid, t, params):
            alpha = params["alpha"]
            L_T, b = apply_boundary_conditions_3d(
                lambda x: laplacian_3d(x, grid), grid, bc, T
            )
            return alpha * (L_T + b)

        X, Y, Z = grid.X, grid.Y, grid.Z
        T0 = jnp.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2) / 0.05)

        from diffheat.solvers import explicit_euler_step_3d
        T1_slow = explicit_euler_step_3d(T0, rhs_fn, grid, t=0.0, dt=1e-4, params={"alpha": 0.01})
        T1_fast = explicit_euler_step_3d(T0, rhs_fn, grid, t=0.0, dt=1e-4, params={"alpha": 0.1})
        # Higher alpha → more diffusion → bigger change per step
        assert jnp.sum(jnp.abs(T1_fast - T0)) > jnp.sum(jnp.abs(T1_slow - T0))


class TestSolve3D:
    def test_returns_correct_shape(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=6, ny=8, nz=5)
        bc = BoundaryCondition3D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
            front={"kind": "neumann", "value": 0.0},
            back={"kind": "neumann", "value": 0.0},
        )
        alpha = 0.01
        rhs_fn = _make_heat_rhs_3d(alpha, bc)
        T0 = jnp.zeros((6, 8, 5))
        dt = 1e-4
        t_span = (0.0, 5e-4)  # 5 steps

        from diffheat.solvers import solve_3d
        trajectory = solve_3d(rhs_fn, T0, grid, t_span, dt)

        n_steps = int((t_span[1] - t_span[0]) / dt) + 1  # 5 + 1 = 6 frames
        assert trajectory.shape == (n_steps, 6, 8, 5)

    def test_first_frame_is_initial_condition(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=6, ny=6, nz=6)
        bc = _all_dirichlet(1.0)
        rhs_fn = _make_heat_rhs_3d(0.01, bc)
        T0 = grid.X + grid.Y + grid.Z

        from diffheat.solvers import solve_3d
        traj = solve_3d(rhs_fn, T0, grid, (0.0, 5e-4), dt=1e-4)
        assert jnp.allclose(traj[0], T0)

    def test_save_every_reduces_frames(self):
        """save_every=5 on 10 steps should yield 2 saved frames + IC = 3 total."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=4, ny=4, nz=4)
        bc = _all_neumann()
        rhs_fn = _make_heat_rhs_3d(0.01, bc)
        T0 = jnp.ones((4, 4, 4))

        from diffheat.solvers import solve_3d
        traj = solve_3d(rhs_fn, T0, grid, (0.0, 1e-3), dt=1e-4, save_every=5)
        # n_steps = 10, save_every = 5 → n_outer = 2 → 2 + 1 = 3 frames
        assert traj.shape == (3, 4, 4, 4)
        assert jnp.allclose(traj[0], T0)

    def test_gradient_wrt_alpha(self):
        """Gradients must flow through solve_3d w.r.t. alpha."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=5, ny=5, nz=5)

        def loss_fn(alpha):
            bc = _all_dirichlet(1.0)
            rhs_fn = _make_heat_rhs_3d(alpha, bc)
            T0 = jnp.zeros((5, 5, 5))
            from diffheat.solvers import solve_3d
            traj = solve_3d(rhs_fn, T0, grid, (0.0, 2e-4), dt=1e-4)
            return jnp.mean(traj[-1])

        grad = jax.grad(loss_fn)(0.1)
        assert not jnp.isclose(grad, 0.0)

    def test_temperature_bounded(self):
        """With Dirichlet BCs in [0, 1], temperature should stay in [0, 1]."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        bc = _all_dirichlet(0.0)
        rhs_fn = _make_heat_rhs_3d(0.01, bc)
        T0 = jnp.ones((8, 8, 8))

        from diffheat.solvers import solve_3d
        traj = solve_3d(rhs_fn, T0, grid, (0.0, 5e-4), dt=1e-4)
        assert float(jnp.min(traj)) >= -1e-10
        assert float(jnp.max(traj)) <= 1.0 + 1e-10
