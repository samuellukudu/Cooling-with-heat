# tests/test_advection_diffusion.py
"""Tests for advection-diffusion solvers and CFL conditions."""
import jax.numpy as jnp
import pytest
from diffheat.mesh import Grid1D, Grid2D, Grid3D


class TestCFLAdvectionDiffusion1D:
    @pytest.fixture
    def grid(self):
        return Grid1D.uniform(length=1.0, n_cells=50)

    def test_stable_dt_passes(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_1d
        alpha = 0.01
        u_max = 1.0
        dx_min = float(jnp.min(grid.dx))
        dt_diff = dx_min**2 / (2 * alpha)
        dt_adv = dx_min / u_max
        dt_max = min(dt_diff, dt_adv)
        assert check_cfl_advection_diffusion_1d(grid, alpha, u_max, 0.9 * dt_max)

    def test_unstable_diffusive_fails(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_1d
        alpha = 0.01
        u_max = 0.0  # pure diffusion
        dx_min = float(jnp.min(grid.dx))
        dt_limit = dx_min**2 / (2 * alpha)
        assert not check_cfl_advection_diffusion_1d(grid, alpha, u_max, 2.0 * dt_limit)

    def test_unstable_advective_fails(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_1d
        alpha = 0.0  # pure advection
        u_max = 2.0
        dx_min = float(jnp.min(grid.dx))
        dt_limit = dx_min / u_max
        if dt_limit > 0:
            assert not check_cfl_advection_diffusion_1d(grid, alpha, u_max, 2.0 * dt_limit)

    def test_zero_velocity_matches_heat_cfl(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_1d, check_cfl
        alpha = 0.01
        dt = 0.001
        assert check_cfl_advection_diffusion_1d(grid, alpha, 0.0, dt) == check_cfl(grid, alpha, dt)


class TestCFLAdvectionDiffusion2D:
    @pytest.fixture
    def grid(self):
        return Grid2D.uniform(Lx=1.0, Ly=1.0, nx=40, ny=40)

    def test_stable_dt_passes(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_2d
        alpha = 0.01
        u_x_max = 1.0
        u_y_max = 0.5
        dx_min = float(jnp.min(grid.dx))
        dy_min = float(jnp.min(grid.dy))
        dt_diff = min(dx_min**2, dy_min**2) / (4 * alpha)
        dt_adv = 1.0 / (u_x_max / dx_min + u_y_max / dy_min)
        dt_max = min(dt_diff, dt_adv)
        assert check_cfl_advection_diffusion_2d(grid, alpha, u_x_max, u_y_max, 0.9 * dt_max)

    def test_unstable_dt_fails(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_2d
        alpha = 0.01
        u_x_max = 10.0
        u_y_max = 10.0
        assert not check_cfl_advection_diffusion_2d(grid, alpha, u_x_max, u_y_max, 0.1)

    def test_zero_velocity_matches_heat_cfl(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_2d, check_cfl_2d
        alpha = 0.01
        dt = 0.001
        assert check_cfl_advection_diffusion_2d(grid, alpha, 0.0, 0.0, dt) == check_cfl_2d(grid, alpha, dt)


class TestCFLAdvectionDiffusion3D:
    @pytest.fixture
    def grid(self):
        return Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=20, ny=20, nz=20)

    def test_stable_dt_passes(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_3d
        alpha = 0.01
        u_max = 1.0
        dx_min = float(jnp.min(grid.dx))
        dy_min = float(jnp.min(grid.dy))
        dz_min = float(jnp.min(grid.dz))
        dt_diff = min(dx_min**2, dy_min**2, dz_min**2) / (6 * alpha)
        dt_adv = 1.0 / (u_max / dx_min + u_max / dy_min + u_max / dz_min)
        dt_max = min(dt_diff, dt_adv)
        assert check_cfl_advection_diffusion_3d(grid, alpha, u_max, u_max, u_max, 0.9 * dt_max)

    def test_zero_velocity_matches_heat_cfl(self, grid):
        from diffheat.solvers.stability import check_cfl_advection_diffusion_3d, check_cfl_3d
        alpha = 0.01
        dt = 0.0001
        assert check_cfl_advection_diffusion_3d(grid, alpha, 0.0, 0.0, 0.0, dt) == check_cfl_3d(grid, alpha, dt)
