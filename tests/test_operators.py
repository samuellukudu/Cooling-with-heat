# tests/test_operators.py
"""Tests for discrete differential operators."""
import jax.numpy as jnp
from diffheat.mesh import Grid2D
from diffheat.operators import laplacian_2d


class TestLaplacian2D:
    def test_returns_correct_shape(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=15)
        T = jnp.ones((10, 15))
        result = laplacian_2d(T, grid)
        assert result.shape == (10, 15)

    def test_constant_field_is_zero_in_interior(self):
        """Laplacian of a constant field should be zero everywhere."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        T = jnp.full((20, 20), 2.5)
        result = laplacian_2d(T, grid)
        # All interior cells (not boundaries) should be ~0
        assert jnp.allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_field_is_zero_in_interior(self):
        """Laplacian of a linear field T(x,y) = a*x + b*y + c should be zero."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        a, b, c = 2.0, -3.0, 1.0
        T = a * grid.X + b * grid.Y + c  # (ny, nx) meshgrid shapes
        T = T.T  # transpose to (nx, ny) for our field convention
        result = laplacian_2d(T, grid)
        assert jnp.allclose(result[2:-2, 2:-2], 0.0, atol=1e-3)

    def test_known_quadratic(self):
        """Laplacian of T = x^2 should be 2 everywhere."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        T = grid.X ** 2  # (ny, nx)
        T = T.T  # (nx, ny)
        result = laplacian_2d(T, grid)
        # Interior should be ~2
        assert jnp.allclose(result[1:-1, 1:-1], 2.0, atol=0.1)

    def test_symmetry(self):
        """Laplacian should be symmetric: swapping x for y with appropriate field."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=15, ny=15)
        # T(x, y) = sin(pi*x)*sin(pi*y) is symmetric
        T = jnp.sin(jnp.pi * grid.X) * jnp.sin(jnp.pi * grid.Y)
        T = T.T  # (nx, ny)
        result = laplacian_2d(T, grid)
        # Result should be symmetric: L[i, j] == L[j, i] for square grid
        assert jnp.allclose(result, result.T, atol=1e-10)


class TestGradient2D:
    def test_gradient_x_shape(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=15)
        from diffheat.operators import gradient_x
        T = jnp.ones((10, 15))
        result = gradient_x(T, grid)
        assert result.shape == (10, 15)

    def test_gradient_y_shape(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=15)
        from diffheat.operators import gradient_y
        T = jnp.ones((10, 15))
        result = gradient_y(T, grid)
        assert result.shape == (10, 15)

    def test_gradient_2d_returns_tuple(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=15)
        from diffheat.operators import gradient_2d
        T = jnp.ones((10, 15))
        gx, gy = gradient_2d(T, grid)
        assert gx.shape == (10, 15)
        assert gy.shape == (10, 15)

    def test_gradient_x_of_linear_x(self):
        """dT/dx of T = a*x should be a everywhere."""
        grid = Grid2D.uniform(Lx=2.0, Ly=1.0, nx=30, ny=15)
        from diffheat.operators import gradient_x
        a = 3.0
        T = a * grid.X.T  # (nx, ny) from (ny, nx) meshgrid
        result = gradient_x(T, grid)
        assert jnp.allclose(result[1:-1, 1:-1], a, atol=1e-2)

    def test_gradient_y_of_linear_y(self):
        """dT/dy of T = b*y should be b everywhere."""
        grid = Grid2D.uniform(Lx=1.0, Ly=2.0, nx=15, ny=30)
        from diffheat.operators import gradient_y
        b = -2.0
        T = b * grid.Y.T  # (nx, ny)
        result = gradient_y(T, grid)
        assert jnp.allclose(result[1:-1, 1:-1], b, atol=1e-2)

    def test_gradient_constant_is_zero(self):
        """Gradient of constant field should be zero (interior)."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        from diffheat.operators import gradient_2d
        T = jnp.full((20, 20), 5.0)
        gx, gy = gradient_2d(T, grid)
        assert jnp.allclose(gx[1:-1, 1:-1], 0.0, atol=1e-10)
        assert jnp.allclose(gy[1:-1, 1:-1], 0.0, atol=1e-10)
