# tests/test_operators.py
"""Tests for discrete differential operators."""
import jax.numpy as jnp
from diffheat.mesh import Grid1D, Grid2D
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


class TestDivergence2D:
    def test_returns_correct_shape(self):
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=10, ny=15)
        from diffheat.operators import divergence_2d
        ux = jnp.ones((10, 15))
        uy = jnp.zeros((10, 15))
        result = divergence_2d(ux, uy, grid)
        assert result.shape == (10, 15)

    def test_divergence_of_uniform_flow_is_zero(self):
        """div (constant, constant) = 0."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=20, ny=20)
        from diffheat.operators import divergence_2d
        ux = jnp.full((20, 20), 2.0)
        uy = jnp.full((20, 20), -1.0)
        result = divergence_2d(ux, uy, grid)
        assert jnp.allclose(result[1:-1, 1:-1], 0.0, atol=1e-10)

    def test_divergence_of_expanding_flow(self):
        """div(x, 0) = 1."""
        grid = Grid2D.uniform(Lx=2.0, Ly=1.0, nx=30, ny=15)
        from diffheat.operators import divergence_2d
        ux = grid.X.T  # (nx, ny), ux = x
        uy = jnp.zeros((30, 15))
        result = divergence_2d(ux, uy, grid)
        assert jnp.allclose(result[1:-1, 1:-1], 1.0, atol=1e-2)

    def test_divergence_is_linear(self):
        """div(a*u1 + b*u2) = a*div(u1) + b*div(u2)."""
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=15, ny=15)
        from diffheat.operators import divergence_2d
        ux1 = grid.X.T
        uy1 = jnp.zeros((15, 15))
        ux2 = jnp.zeros((15, 15))
        uy2 = grid.Y.T
        div1 = divergence_2d(ux1, uy1, grid)
        div2 = divergence_2d(ux2, uy2, grid)
        div_combined = divergence_2d(2.0 * ux1 - 3.0 * uy2,
                                     2.0 * uy1 - 3.0 * uy2, grid)
        expected = 2.0 * div1 - 3.0 * div2
        assert jnp.allclose(div_combined[1:-1, 1:-1], expected[1:-1, 1:-1], atol=1e-2)


class TestLaplacian1D:
    def test_returns_correct_shape(self):
        grid = Grid1D.uniform(length=1.0, n_cells=20)
        from diffheat.operators import laplacian_1d
        from diffheat.mesh import Grid1D as G1
        T = jnp.ones(20)
        result = laplacian_1d(T, grid)
        assert result.shape == (20,)

    def test_constant_field_is_zero_in_interior(self):
        """Laplacian of a constant field should be zero away from boundaries."""
        grid = Grid1D.uniform(length=1.0, n_cells=30)
        from diffheat.operators import laplacian_1d
        T = jnp.full(30, 2.5)
        result = laplacian_1d(T, grid)
        # Interior cells (away from periodic wrap-around) should be ~0
        assert jnp.allclose(result[2:-2], 0.0, atol=1e-6)

    def test_linear_field_is_zero_in_interior(self):
        """d^2(ax+b)/dx^2 = 0 away from boundaries."""
        grid = Grid1D.uniform(length=1.0, n_cells=40)
        from diffheat.operators import laplacian_1d
        a, b = 2.0, 1.0
        T = a * grid.centers + b
        result = laplacian_1d(T, grid)
        assert jnp.allclose(result[3:-3], 0.0, atol=1e-3)

    def test_known_quadratic(self):
        """d^2(x^2)/dx^2 = 2 everywhere."""
        grid = Grid1D.uniform(length=2.0, n_cells=50)
        from diffheat.operators import laplacian_1d
        T = grid.centers ** 2
        result = laplacian_1d(T, grid)
        # Interior should be ~2
        assert jnp.allclose(result[3:-3], 2.0, atol=1e-3)

    def test_uniform_grid_scaling(self):
        """Laplacian entries should scale as 1/dx^2."""
        grid_coarse = Grid1D.uniform(length=1.0, n_cells=10)
        grid_fine = Grid1D.uniform(length=1.0, n_cells=20)
        from diffheat.operators import laplacian_1d
        T = grid_coarse.centers ** 2
        Lc = laplacian_1d(T, grid_coarse)
        T = grid_fine.centers ** 2
        Lf = laplacian_1d(T, grid_fine)
        # Both should be ~2 in interior regardless of resolution
        assert jnp.allclose(Lc[3:-3], 2.0, atol=1e-3)
        assert jnp.allclose(Lf[3:-3], 2.0, atol=1e-3)


class TestGradient1D:
    def test_returns_correct_shape(self):
        grid = Grid1D.uniform(length=1.0, n_cells=25)
        from diffheat.operators import gradient_1d
        T = jnp.ones(25)
        result = gradient_1d(T, grid)
        assert result.shape == (25,)

    def test_constant_field_is_zero_in_interior(self):
        """Gradient of constant field should be zero."""
        grid = Grid1D.uniform(length=1.0, n_cells=30)
        from diffheat.operators import gradient_1d
        T = jnp.full(30, 5.0)
        result = gradient_1d(T, grid)
        assert jnp.allclose(result[2:-2], 0.0, atol=1e-6)

    def test_gradient_of_linear(self):
        """d(ax+b)/dx = a."""
        grid = Grid1D.uniform(length=1.0, n_cells=40)
        from diffheat.operators import gradient_1d
        a, b = 3.0, -1.0
        T = a * grid.centers + b
        result = gradient_1d(T, grid)
        assert jnp.allclose(result[2:-2], a, atol=1e-3)

    def test_uniform_grid_scaling(self):
        """Gradient should be independent of resolution for smooth fields."""
        grid_coarse = Grid1D.uniform(length=1.0, n_cells=10)
        grid_fine = Grid1D.uniform(length=1.0, n_cells=20)
        from diffheat.operators import gradient_1d
        a = 3.0
        Tc = a * grid_coarse.centers
        Tf = a * grid_fine.centers
        gc = gradient_1d(Tc, grid_coarse)
        gf = gradient_1d(Tf, grid_fine)
        assert jnp.allclose(gc[2:-2], a, atol=1e-3)
        assert jnp.allclose(gf[2:-2], a, atol=1e-3)
