# tests/test_wave.py
"""Tests for wave equation solver (leapfrog scheme)."""
import jax
import jax.numpy as jnp
import pytest

from diffheat import (
    BoundaryCondition,
    BoundaryCondition2D,
    BoundaryCondition3D,
    Grid1D,
    Grid2D,
    Grid3D,
    WaveEquation1D,
    WaveEquation2D,
    WaveEquation3D,
    check_cfl_wave_1d,
    check_cfl_wave_2d,
    check_cfl_wave_3d,
    solve_wave_1d,
    solve_wave_2d,
    solve_wave_3d,
)
from diffheat.operators import gradient_1d, gradient_2d, gradient_3d
from diffheat.utils import array


# ---------------------------------------------------------------------------
# CFL checks
# ---------------------------------------------------------------------------

class TestCFLWave1D:
    def test_stable(self):
        grid = Grid1D.uniform(length=1.0, n_cells=100)
        assert check_cfl_wave_1d(grid, c=1.0, dt=0.005)

    def test_unstable(self):
        grid = Grid1D.uniform(length=1.0, n_cells=100)
        assert not check_cfl_wave_1d(grid, c=1.0, dt=0.02)


class TestCFLWave2D:
    def test_stable(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=50, ny=50)
        assert check_cfl_wave_2d(grid, c=1.0, dt=0.01)

    def test_unstable(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=50, ny=50)
        # c*dt = 1*0.03 = 0.03, dx/sqrt(2) = 0.02/1.414 = 0.0141 — too large
        assert not check_cfl_wave_2d(grid, c=1.0, dt=0.03)


class TestCFLWave3D:
    def test_stable(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=20, nz=20)
        assert check_cfl_wave_3d(grid, c=1.0, dt=0.02)

    def test_unstable(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=20, nz=20)
        assert not check_cfl_wave_3d(grid, c=1.0, dt=0.1)


# ---------------------------------------------------------------------------
# 1D wave equation — standing wave solution
# ---------------------------------------------------------------------------

class TestWave1D:
    def test_standing_wave_conserves_energy(self):
        """A standing wave should approximately conserve energy."""
        L = 1.0
        n_cells = 100
        grid = Grid1D.uniform(length=L, n_cells=n_cells)
        c = 1.0

        # Standing wave: u(x,0) = sin(pi*x), v(x,0) = 0
        x = grid.centers
        u0 = jnp.sin(jnp.pi * x / L)
        v0 = jnp.zeros(n_cells)

        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([0.0, 0.0]))
        eqn = WaveEquation1D(grid=grid, bc=bc, c=c)

        dt = 0.8 * float(jnp.min(grid.dx)) / c  # 80% of CFL
        t_end = 0.2
        traj = solve_wave_1d(eqn, u0, v0, (0.0, t_end), dt)

        # Energy at first and last frame: E = 0.5 * ∫(v² + c²|∇u|²) dx
        def energy(u_frame, u_prev_frame, dt_val):
            """Approximate velocity v = (u_next - u_prev) / (2*dt) for energy."""
            v_approx = (u_frame - u_prev_frame) / dt_val
            g = gradient_1d(u_frame, grid)
            return 0.5 * jnp.sum((v_approx ** 2 + c ** 2 * g ** 2) * grid.dx)

        # Use first two frames to estimate energy at t=dt
        e_mid = energy(traj[1], traj[0], dt)
        # Use last two frames
        e_end = energy(traj[-1], traj[-2], dt)

        # Energy should be roughly conserved (within 2% for short simulation)
        assert jnp.abs(e_mid - e_end) / e_mid < 0.02

    def test_trajectory_shape(self):
        grid = Grid1D.uniform(length=1.0, n_cells=50)
        u0 = jnp.sin(jnp.pi * grid.centers)
        v0 = jnp.zeros(50)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([0.0, 0.0]))
        eqn = WaveEquation1D(grid=grid, bc=bc, c=1.0)

        dt = 0.005
        n_steps = 20
        t_end = n_steps * dt
        traj = solve_wave_1d(eqn, u0, v0, (0.0, t_end), dt)
        assert traj.shape == (n_steps + 1, 50)
        assert jnp.allclose(traj[0], u0)

    def test_zero_velocity_starts_from_rest(self):
        """With v0=0, the first step should approximately preserve the shape."""
        grid = Grid1D.uniform(length=1.0, n_cells=100)
        u0 = jnp.sin(jnp.pi * grid.centers)
        v0 = jnp.zeros(100)
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([0.0, 0.0]))
        eqn = WaveEquation1D(grid=grid, bc=bc, c=1.0)

        dt = 0.001
        traj = solve_wave_1d(eqn, u0, v0, (0.0, dt), dt)

        # u^1 should be close to u^0 (since v0=0, change is O(dt²))
        diff = jnp.max(jnp.abs(traj[1] - u0))
        assert diff < 1e-4


# ---------------------------------------------------------------------------
# 2D wave equation
# ---------------------------------------------------------------------------

class TestWave2D:
    def test_trajectory_shape(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=30, ny=30)
        u0 = jnp.sin(jnp.pi * grid.X.T) * jnp.sin(jnp.pi * grid.Y.T)
        v0 = jnp.zeros((30, 30))
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.0},
            top={"kind": "dirichlet", "value": 0.0},
        )
        eqn = WaveEquation2D(grid=grid, bc=bc, c=1.0)

        dt = 0.005
        n_steps = 10
        t_end = n_steps * dt
        traj = solve_wave_2d(eqn, u0, v0, (0.0, t_end), dt)
        assert traj.shape == (n_steps + 1, 30, 30)
        assert jnp.allclose(traj[0], u0)

    def test_solution_stays_smooth(self):
        """Solution should remain bounded and smooth."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        # Localized initial pulse
        Xc, Yc = grid.X.T, grid.Y.T
        u0 = jnp.exp(-((Xc - 0.5) ** 2 + (Yc - 0.5) ** 2) / 0.01)
        v0 = jnp.zeros((20, 20))
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.0},
            top={"kind": "dirichlet", "value": 0.0},
        )
        eqn = WaveEquation2D(grid=grid, bc=bc, c=1.0)

        dt = 0.002
        traj = solve_wave_2d(eqn, u0, v0, (0.0, 0.02), dt)
        # Solution should stay bounded (no blow-up)
        assert jnp.max(jnp.abs(traj)) < 10.0
        # No NaNs
        assert not jnp.any(jnp.isnan(traj))

    def test_differentiable(self):
        """Gradient of final state w.r.t. wave speed should be computable."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=16, ny=16)
        u0 = jnp.sin(jnp.pi * grid.X.T) * jnp.sin(jnp.pi * grid.Y.T)
        v0 = jnp.zeros((16, 16))
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.0},
            top={"kind": "dirichlet", "value": 0.0},
        )

        def loss(c_val):
            eqn = WaveEquation2D(grid=grid, bc=bc, c=c_val)
            traj = solve_wave_2d(eqn, u0, v0, (0.0, 0.01), 0.005)
            return jnp.mean(traj[-1] ** 2)

        grad = jax.grad(loss)(1.0)
        assert jnp.isfinite(grad)


# ---------------------------------------------------------------------------
# 3D wave equation
# ---------------------------------------------------------------------------

class TestWave3D:
    def test_trajectory_shape(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        u0 = jnp.ones((8, 8, 8)) * 0.1
        v0 = jnp.zeros((8, 8, 8))
        bc = BoundaryCondition3D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.0},
            top={"kind": "dirichlet", "value": 0.0},
            front={"kind": "dirichlet", "value": 0.0},
            back={"kind": "dirichlet", "value": 0.0},
        )
        eqn = WaveEquation3D(grid=grid, bc=bc, c=1.0)

        dt = 0.01
        n_steps = 5
        t_end = n_steps * dt
        traj = solve_wave_3d(eqn, u0, v0, (0.0, t_end), dt)
        assert traj.shape == (n_steps + 1, 8, 8, 8)
        assert jnp.allclose(traj[0], u0)

    def test_save_every(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=6, ny=6, nz=6)
        u0 = jnp.ones((6, 6, 6)) * 0.1
        v0 = jnp.zeros((6, 6, 6))
        bc = BoundaryCondition3D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.0},
            top={"kind": "dirichlet", "value": 0.0},
            front={"kind": "dirichlet", "value": 0.0},
            back={"kind": "dirichlet", "value": 0.0},
        )
        eqn = WaveEquation3D(grid=grid, bc=bc, c=1.0)

        dt = 0.01
        traj = solve_wave_3d(eqn, u0, v0, (0.0, 0.04), dt, save_every=2)
        # 4 steps total, save_every=2 => 2 outer steps + initial = 3 frames
        assert traj.shape[0] == 3

    def test_no_nans(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        u0 = jnp.sin(jnp.pi * grid.X) * jnp.sin(jnp.pi * grid.Y) * jnp.sin(jnp.pi * grid.Z)
        v0 = jnp.zeros((8, 8, 8))
        bc = BoundaryCondition3D(
            left={"kind": "dirichlet", "value": 0.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.0},
            top={"kind": "dirichlet", "value": 0.0},
            front={"kind": "dirichlet", "value": 0.0},
            back={"kind": "dirichlet", "value": 0.0},
        )
        eqn = WaveEquation3D(grid=grid, bc=bc, c=1.0)

        dt = 0.005
        traj = solve_wave_3d(eqn, u0, v0, (0.0, 0.02), dt)
        assert not jnp.any(jnp.isnan(traj))
