# tests/test_grid3d.py
"""Tests for the 3D uniform grid."""
import jax.numpy as jnp
import pytest
from diffheat.mesh import Grid3D


class TestGrid3D:
    def test_uniform_creates_correct_dimensions(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=10, ny=20, nz=30)
        assert grid.nx == 10
        assert grid.ny == 20
        assert grid.nz == 30

    def test_uniform_x_has_nx_plus_one_points(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        assert grid.x.shape == (9,)

    def test_uniform_y_has_ny_plus_one_points(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=12, nz=8)
        assert grid.y.shape == (13,)

    def test_uniform_z_has_nz_plus_one_points(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=16)
        assert grid.z.shape == (17,)

    def test_uniform_centers_have_correct_shapes(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=5, ny=7, nz=9)
        assert grid.x_centers.shape == (5,)
        assert grid.y_centers.shape == (7,)
        assert grid.z_centers.shape == (9,)

    def test_uniform_dx_dy_dz_have_correct_shapes(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=5, ny=7, nz=9)
        assert grid.dx.shape == (5,)
        assert grid.dy.shape == (7,)
        assert grid.dz.shape == (9,)

    def test_uniform_x_starts_at_zero(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=10, ny=10, nz=10)
        assert float(grid.x[0]) == pytest.approx(0.0)
        assert float(grid.y[0]) == pytest.approx(0.0)
        assert float(grid.z[0]) == pytest.approx(0.0)

    def test_uniform_x_ends_at_L(self):
        grid = Grid3D.uniform(Lx=2.0, Ly=3.0, Lz=4.0, nx=10, ny=10, nz=10)
        assert float(grid.x[-1]) == pytest.approx(2.0)
        assert float(grid.y[-1]) == pytest.approx(3.0)
        assert float(grid.z[-1]) == pytest.approx(4.0)

    def test_meshgrid_shapes(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=5, ny=7, nz=9)
        assert grid.X.shape == (5, 7, 9)
        assert grid.Y.shape == (5, 7, 9)
        assert grid.Z.shape == (5, 7, 9)

    def test_meshgrid_x_values(self):
        """X[:, 0, 0] should equal x_centers."""
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=5, ny=7, nz=9)
        assert jnp.allclose(grid.X[:, 0, 0], grid.x_centers)

    def test_meshgrid_y_values(self):
        """Y[0, :, 0] should equal y_centers."""
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=5, ny=7, nz=9)
        assert jnp.allclose(grid.Y[0, :, 0], grid.y_centers)

    def test_meshgrid_z_values(self):
        """Z[0, 0, :] should equal z_centers."""
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=5, ny=7, nz=9)
        assert jnp.allclose(grid.Z[0, 0, :], grid.z_centers)

    def test_frozen_dataclass(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=5, ny=5, nz=5)
        with pytest.raises(Exception):
            grid.nx = 10  # type: ignore[misc]

    def test_validation_rejects_zero_length(self):
        with pytest.raises(ValueError, match="Lx"):
            Grid3D.uniform(Lx=0.0, Ly=1.0, Lz=1.0, nx=5, ny=5, nz=5)
        with pytest.raises(ValueError, match="Ly"):
            Grid3D.uniform(Lx=1.0, Ly=-1.0, Lz=1.0, nx=5, ny=5, nz=5)
        with pytest.raises(ValueError, match="Lz"):
            Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=0.0, nx=5, ny=5, nz=5)

    def test_validation_rejects_too_few_cells(self):
        with pytest.raises(ValueError, match="nx"):
            Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=1, ny=5, nz=5)
        with pytest.raises(ValueError, match="ny"):
            Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=5, ny=1, nz=5)
        with pytest.raises(ValueError, match="nz"):
            Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=5, ny=5, nz=1)

    def test_uniform_dx_is_constant(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=10, ny=20, nz=30)
        assert jnp.allclose(grid.dx, grid.dx[0])
        assert float(grid.dx[0]) == pytest.approx(1.0 / 10)
        assert float(grid.dy[0]) == pytest.approx(2.0 / 20)
        assert float(grid.dz[0]) == pytest.approx(3.0 / 30)
