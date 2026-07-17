# tests/test_eigen.py
"""Tests for Rayleigh Quotient and eigenvalue calculation."""
import jax
import jax.numpy as jnp
import pytest

from diffheat import (
    Grid1D,
    Grid2D,
    Grid3D,
    rayleigh_quotient_1d,
    rayleigh_quotient_2d,
    rayleigh_quotient_3d,
    rayleigh_upper_bounds_1d,
    rayleigh_upper_bounds_2d,
    rayleigh_upper_bounds_3d,
    find_first_eigenvalue_1d,
    find_first_eigenvalue_2d,
    find_first_eigenvalue_3d,
)


class TestRayleighQuotient:
    def test_rayleigh_quotient_1d_analytic(self):
        """Test R(v) matches analytical eigenvalue for v = sin(pi * x)."""
        grid = Grid1D.uniform(length=1.0, n_cells=100)
        # Analytic first eigenfunction: v(x) = sin(pi * x)
        v = jnp.sin(jnp.pi * grid.centers)
        
        rq = rayleigh_quotient_1d(v, grid)
        expected = jnp.pi**2
        assert jnp.isclose(rq, expected, rtol=1e-3)

    def test_rayleigh_quotient_2d_analytic(self):
        """Test R(v) matches analytical eigenvalue for v = sin(pi*x)*sin(pi*y)."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=40, ny=40)
        # X.T and Y.T have shape (nx, ny)
        v = jnp.sin(jnp.pi * grid.X.T) * jnp.sin(jnp.pi * grid.Y.T)
        
        rq = rayleigh_quotient_2d(v, grid)
        expected = 2.0 * (jnp.pi**2)
        assert jnp.isclose(rq, expected, rtol=1e-3)


class TestFirstEigenvalue:
    def test_find_first_eigenvalue_1d(self):
        """1D smallest eigenvalue should converge to exact discrete value."""
        grid = Grid1D.uniform(length=1.0, n_cells=50)
        val, func = find_first_eigenvalue_1d(grid, max_iter=15)
        
        # Exact discrete eigenvalue
        dx = 1.0 / 50
        expected = (2.0 - 2.0 * jnp.cos(jnp.pi / 50)) / (dx * dx)
        
        assert jnp.isclose(val, expected, rtol=1e-4)
        assert func.shape == (50,)
        assert jnp.all(func > 0.0)

    def test_find_first_eigenvalue_2d(self):
        """2D smallest eigenvalue on unit square should converge to exact discrete value."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        val, func = find_first_eigenvalue_2d(grid, max_iter=15)
        
        # Exact discrete eigenvalue
        dx = 1.0 / 20
        dy = 1.0 / 20
        expected = (2.0 - 2.0 * jnp.cos(jnp.pi / 20)) / (dx * dx) + (2.0 - 2.0 * jnp.cos(jnp.pi / 20)) / (dy * dy)
        
        assert jnp.isclose(val, expected, rtol=1e-4)
        assert func.shape == (20, 20)
        assert jnp.all(func > 0.0)

    def test_find_first_eigenvalue_3d(self):
        """3D smallest eigenvalue on unit cube should converge to exact discrete value."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=10, ny=10, nz=10)
        val, func = find_first_eigenvalue_3d(grid, max_iter=10)
        
        # Exact discrete eigenvalue
        h = 1.0 / 10
        expected = 3.0 * (2.0 - 2.0 * jnp.cos(jnp.pi / 10)) / (h * h)
        
        assert jnp.isclose(val, expected, rtol=1e-4)
        assert func.shape == (10, 10, 10)
        assert jnp.all(func > 0.0)


    def test_gradient_flow_through_eigenvalue(self):
        """Verify that gradients can be taken through the eigenvalue solver."""
        # We parameterize the domain length Lx and compute d(lambda_1) / d(Lx)
        def loss_fn(Lx):
            grid = Grid2D.uniform(Lx=Lx, Ly=1.0, nx=10, ny=10)
            val, _ = find_first_eigenvalue_2d(grid, max_iter=5)
            return val

        # Analytical: lambda_1 = pi^2 / Lx^2 + pi^2 / Ly^2
        # d(lambda_1)/d(Lx) = -2 * pi^2 / Lx^3
        # For Lx = 1.0, gradient = -2 * pi^2 = -19.7392
        grad = jax.grad(loss_fn)(1.0)
        assert jnp.isclose(grad, -2.0 * (jnp.pi**2), rtol=1e-2)


class TestRayleighUpperBounds:
    def test_upper_bound_1d(self):
        """Rayleigh quotient of exact eigenfunction gives exact eigenvalue."""
        grid = Grid1D.uniform(length=1.0, n_cells=64)

        def trial_sine(grid):
            return jnp.sin(jnp.pi * grid.centers)

        def trial_sine2(grid):
            return jnp.sin(2.0 * jnp.pi * grid.centers)

        rqs = rayleigh_upper_bounds_1d(grid, [trial_sine, trial_sine2])
        assert rqs.shape == (2,)

        # sin(pi*x) should give R ≈ π² (the exact eigenvalue)
        assert jnp.isclose(rqs[0], jnp.pi ** 2, rtol=1e-3)
        # sin(2*pi*x) should give R ≈ 4π² > π²
        assert float(rqs[1]) > float(rqs[0])

    def test_upper_bound_2d(self):
        """Rayleigh quotient upper bounds on 2D trial functions."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=32, ny=32)

        def trial_11(grid):
            """First eigenfunction: sin(pi*x)*sin(pi*y) — λ = 2π²."""
            return jnp.sin(jnp.pi * grid.X.T) * jnp.sin(jnp.pi * grid.Y.T)

        def trial_21(grid):
            """Higher mode: sin(2*pi*x)*sin(pi*y) — λ = 5π²."""
            return jnp.sin(2.0 * jnp.pi * grid.X.T) * jnp.sin(jnp.pi * grid.Y.T)

        rqs = rayleigh_upper_bounds_2d(grid, [trial_11, trial_21])
        assert rqs.shape == (2,)

        # First trial should give ≈ 2π²
        assert jnp.isclose(rqs[0], 2.0 * jnp.pi ** 2, rtol=1e-3)
        # Second trial should be higher
        assert float(rqs[1]) > float(rqs[0])

    def test_upper_bound_3d(self):
        """Rayleigh quotient upper bounds on 3D trial functions."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=16, ny=16, nz=16)

        def trial_111(grid):
            """First eigenfunction on cube: λ = 3π²."""
            return (
                jnp.sin(jnp.pi * grid.X)
                * jnp.sin(jnp.pi * grid.Y)
                * jnp.sin(jnp.pi * grid.Z)
            )

        def trial_211(grid):
            """Higher mode: λ = 6π²."""
            return (
                jnp.sin(2.0 * jnp.pi * grid.X)
                * jnp.sin(jnp.pi * grid.Y)
                * jnp.sin(jnp.pi * grid.Z)
            )

        rqs = rayleigh_upper_bounds_3d(grid, [trial_111, trial_211])
        assert rqs.shape == (2,)
        assert jnp.isclose(rqs[0], 3.0 * jnp.pi ** 2, rtol=1e-2)
        assert float(rqs[1]) > float(rqs[0])
