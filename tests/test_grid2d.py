# tests/test_grid2d.py
import pytest
import jax.numpy as jnp
from diffheat.mesh import Grid2D


class TestGrid2D:
    def test_uniform_creates_correct_dimensions(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=2.0, nx=10, ny=20)
        assert grid.nx == 10
        assert grid.ny == 20
        assert grid.Lx == 1.0
        assert grid.Ly == 2.0

    def test_uniform_x_has_nx_plus_one_points(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=20)
        assert grid.x.shape == (11,)  # nx + 1

    def test_uniform_y_has_ny_plus_one_points(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=20)
        assert grid.y.shape == (21,)  # ny + 1

    def test_uniform_centers_have_correct_shapes(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=20)
        assert grid.x_centers.shape == (10,)
        assert grid.y_centers.shape == (20,)

    def test_uniform_dx_dy_have_correct_shapes(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=2.0, nx=10, ny=20)
        assert grid.dx.shape == (10,)
        assert grid.dy.shape == (20,)
        assert jnp.allclose(grid.dx, 0.1)
        assert jnp.allclose(grid.dy, 0.1)

    def test_uniform_x_starts_at_zero(self):
        grid = Grid2D.uniform(Lx=3.0, Ly=2.0, nx=5, ny=4)
        assert jnp.isclose(grid.x[0], 0.0)
        assert jnp.isclose(grid.x[-1], 3.0)

    def test_uniform_y_starts_at_zero(self):
        grid = Grid2D.uniform(Lx=3.0, Ly=2.0, nx=5, ny=4)
        assert jnp.isclose(grid.y[0], 0.0)
        assert jnp.isclose(grid.y[-1], 2.0)

    def test_meshgrid_shapes(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=2.0, nx=8, ny=12)
        assert grid.X.shape == (12, 8)  # (ny, nx) — imshow convention
        assert grid.Y.shape == (12, 8)

    def test_meshgrid_values(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=4, ny=4)
        # X[j, i] = x_centers[i] for all j
        for j in range(4):
            assert jnp.allclose(grid.X[j, :], grid.x_centers)
        # Y[j, i] = y_centers[j] for all i
        for i in range(4):
            assert jnp.allclose(grid.Y[:, i], grid.y_centers)

    def test_frozen_dataclass(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=10)
        with pytest.raises(Exception):
            grid.nx = 20

    def test_validation_rejects_zero_length(self):
        with pytest.raises(ValueError):
            Grid2D.uniform(Lx=0.0, Ly=1.0, nx=10, ny=10)

    def test_validation_rejects_too_few_cells(self):
        with pytest.raises(ValueError):
            Grid2D.uniform(Lx=1.0, Ly=1.0, nx=1, ny=10)
