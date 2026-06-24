# tests/test_mesh.py
import pytest
import jax.numpy as jnp
from diffheat.mesh import Grid1D, BoundaryCondition


class TestGrid1D:
    def test_uniform_creates_correct_n_cells(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        assert grid.n_cells == 10
        assert grid.length == 1.0

    def test_uniform_x_has_n_plus_one_points(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        assert grid.x.shape == (11,)  # n_cells + 1

    def test_uniform_centers_has_n_points(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        assert grid.centers.shape == (10,)  # n_cells

    def test_uniform_dx_has_n_points(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        assert grid.dx.shape == (10,)
        assert jnp.allclose(grid.dx, 0.1)

    def test_uniform_x_starts_at_zero(self):
        grid = Grid1D.uniform(length=2.0, n_cells=5)
        assert jnp.isclose(grid.x[0], 0.0)
        assert jnp.isclose(grid.x[-1], 2.0)

    def test_uniform_centers_are_midpoints(self):
        grid = Grid1D.uniform(length=1.0, n_cells=4)
        dx = 1.0 / 4
        expected_centers = jnp.array([dx/2, dx + dx/2, 2*dx + dx/2, 3*dx + dx/2])
        assert jnp.allclose(grid.centers, expected_centers)

    def test_frozen_dataclass(self):
        grid = Grid1D.uniform(length=1.0, n_cells=10)
        with pytest.raises(Exception):
            grid.length = 2.0


class TestBoundaryCondition:
    def test_dirichlet_creation(self):
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        assert bc.kind == "dirichlet"
        assert bc.value.shape == (2,)
        assert jnp.isclose(bc.value[0], 1.0)
        assert jnp.isclose(bc.value[1], 0.0)

    def test_neumann_creation(self):
        bc = BoundaryCondition(kind="neumann", value=jnp.array([0.0, -1.0]))
        assert bc.kind == "neumann"
        assert bc.value.shape == (2,)

    def test_frozen_dataclass(self):
        bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
        with pytest.raises(Exception):
            bc.kind = "neumann"
