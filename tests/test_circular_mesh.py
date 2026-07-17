# tests/test_circular_mesh.py
"""Tests for the PolarGrid mesh class."""
import jax.numpy as jnp
import pytest

from diffheat import PolarGrid


class TestPolarGrid:
    def test_creation(self):
        grid = PolarGrid.uniform(R=1.0, nr=20, ntheta=32)
        assert grid.nr == 20
        assert grid.ntheta == 32
        assert grid.R == 1.0

    def test_shapes(self):
        grid = PolarGrid.uniform(R=2.0, nr=30, ntheta=64)
        assert grid.r.shape == (31,)          # nr + 1
        assert grid.theta.shape == (65,)      # ntheta + 1
        assert grid.r_centers.shape == (30,)
        assert grid.theta_centers.shape == (64,)
        assert grid.dr.shape == (30,)
        assert grid.X.shape == (30, 64)
        assert grid.Y.shape == (30, 64)
        assert grid.R_mesh.shape == (30, 64)
        assert grid.THETA_mesh.shape == (30, 64)
        assert grid.area.shape == (30, 64)

    def test_cartesian_coordinates(self):
        """X² + Y² should equal R_mesh² on the grid."""
        grid = PolarGrid.uniform(R=1.5, nr=30, ntheta=64)
        r2 = grid.X ** 2 + grid.Y ** 2
        assert jnp.allclose(r2, grid.R_mesh ** 2, atol=1e-6)

    def test_radial_bounds(self):
        """All cell centres should be within (0, R)."""
        grid = PolarGrid.uniform(R=1.0, nr=40, ntheta=16)
        assert float(jnp.min(grid.R_mesh)) > 0.0
        assert float(jnp.max(grid.R_mesh)) < 1.0

    def test_angular_range(self):
        """Theta should cover [-π, π]."""
        grid = PolarGrid.uniform(R=1.0, nr=10, ntheta=32)
        assert float(jnp.min(grid.THETA_mesh)) >= -jnp.pi
        assert float(jnp.max(grid.THETA_mesh)) <= jnp.pi

    def test_area_sum_approx_disc_area(self):
        """Sum of cell areas should approximate πR²."""
        grid = PolarGrid.uniform(R=2.0, nr=100, ntheta=200)
        computed_area = float(jnp.sum(grid.area))
        expected = jnp.pi * 4.0
        assert abs(computed_area - expected) / expected < 0.01

    def test_dtheta(self):
        grid = PolarGrid.uniform(R=1.0, nr=10, ntheta=64)
        assert abs(grid.dtheta - 2.0 * jnp.pi / 64) < 1e-10

    def test_validation(self):
        with pytest.raises(ValueError):
            PolarGrid.uniform(R=0.0, nr=10, ntheta=16)
        with pytest.raises(ValueError):
            PolarGrid.uniform(R=1.0, nr=1, ntheta=16)
        with pytest.raises(ValueError):
            PolarGrid.uniform(R=1.0, nr=10, ntheta=2)
