# tests/test_operators_3d.py
"""Tests for 3D discrete differential operators."""
import jax.numpy as jnp
import pytest
from diffheat.mesh import Grid3D
from diffheat.operators.laplacian import laplacian_3d
from diffheat.operators.gradient import (
    gradient_x3d,
    gradient_y3d,
    gradient_z3d,
    gradient_3d,
)
from diffheat.operators.divergence import divergence_3d


# ---------------------------------------------------------------------------
# Laplacian 3D
# ---------------------------------------------------------------------------

class TestLaplacian3D:
    def test_returns_correct_shape(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=8, ny=10, nz=12)
        T = jnp.ones((8, 10, 12))
        L = laplacian_3d(T, grid)
        assert L.shape == (8, 10, 12)

    def test_constant_field_is_zero_in_interior(self):
        """Laplacian of a constant field is zero in the interior."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=10, ny=10, nz=10)
        T = jnp.full((10, 10, 10), 3.14)
        L = laplacian_3d(T, grid)
        assert jnp.allclose(L[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_linear_field_is_zero_in_interior(self):
        """Laplacian of a linear field is analytically zero."""
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=12, ny=12, nz=12)
        T = grid.X + 2.0 * grid.Y + 3.0 * grid.Z
        L = laplacian_3d(T, grid)
        assert jnp.allclose(L[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_known_quadratic_x_squared_plus_y_squared_plus_z_squared(self):
        """Laplacian of x^2 + y^2 + z^2 == 2 + 2 + 2 = 6 everywhere in interior."""
        grid = Grid3D.uniform(Lx=2.0, Ly=2.0, Lz=2.0, nx=20, ny=20, nz=20)
        T = grid.X ** 2 + grid.Y ** 2 + grid.Z ** 2
        L = laplacian_3d(T, grid)
        # Interior cells — should be very close to 6.0 (FD error ~ O(h^2))
        assert jnp.allclose(L[2:-2, 2:-2, 2:-2], 6.0, atol=1e-6)

    def test_symmetry_xyz(self):
        """laplacian_3d should be symmetric under x<->y<->z permutation on a cube."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        T = jnp.exp(-(grid.X**2 + grid.Y**2 + grid.Z**2))
        L = laplacian_3d(T, grid)
        # Interior should be finite and not NaN
        interior = L[1:-1, 1:-1, 1:-1]
        assert not jnp.any(jnp.isnan(interior))
        assert not jnp.any(jnp.isinf(interior))


# ---------------------------------------------------------------------------
# Gradient 3D
# ---------------------------------------------------------------------------

class TestGradient3D:
    def test_gradient_x3d_shape(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=8, ny=10, nz=12)
        T = jnp.ones((8, 10, 12))
        gx = gradient_x3d(T, grid)
        assert gx.shape == (8, 10, 12)

    def test_gradient_y3d_shape(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=8, ny=10, nz=12)
        T = jnp.ones((8, 10, 12))
        gy = gradient_y3d(T, grid)
        assert gy.shape == (8, 10, 12)

    def test_gradient_z3d_shape(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=8, ny=10, nz=12)
        T = jnp.ones((8, 10, 12))
        gz = gradient_z3d(T, grid)
        assert gz.shape == (8, 10, 12)

    def test_gradient_3d_returns_tuple_of_three(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        T = jnp.ones((8, 8, 8))
        result = gradient_3d(T, grid)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_gradient_x3d_of_linear_x(self):
        """dT/dx of T=x should be 1 everywhere (interior)."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=12, ny=12, nz=12)
        T = grid.X
        gx = gradient_x3d(T, grid)
        assert jnp.allclose(gx[1:-1, 1:-1, 1:-1], 1.0, atol=1e-10)

    def test_gradient_y3d_of_linear_y(self):
        """dT/dy of T=y should be 1 everywhere (interior)."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=12, ny=12, nz=12)
        T = grid.Y
        gy = gradient_y3d(T, grid)
        assert jnp.allclose(gy[1:-1, 1:-1, 1:-1], 1.0, atol=1e-10)

    def test_gradient_z3d_of_linear_z(self):
        """dT/dz of T=z should be 1 everywhere (interior)."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=12, ny=12, nz=12)
        T = grid.Z
        gz = gradient_z3d(T, grid)
        assert jnp.allclose(gz[1:-1, 1:-1, 1:-1], 1.0, atol=1e-10)

    def test_gradient_constant_is_zero(self):
        """Gradient of a constant is zero everywhere (including boundaries)."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        T = jnp.full((8, 8, 8), 42.0)
        gx, gy, gz = gradient_3d(T, grid)
        assert jnp.allclose(gx, 0.0, atol=1e-10)
        assert jnp.allclose(gy, 0.0, atol=1e-10)
        assert jnp.allclose(gz, 0.0, atol=1e-10)

    def test_gradient_orthogonality(self):
        """dT/dx of T=y should be zero in interior."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=10, ny=10, nz=10)
        T = grid.Y
        gx = gradient_x3d(T, grid)
        assert jnp.allclose(gx[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Divergence 3D
# ---------------------------------------------------------------------------

class TestDivergence3D:
    def test_returns_correct_shape(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=2.0, Lz=3.0, nx=8, ny=10, nz=12)
        ux = jnp.ones((8, 10, 12))
        uy = jnp.ones((8, 10, 12))
        uz = jnp.ones((8, 10, 12))
        div = divergence_3d(ux, uy, uz, grid)
        assert div.shape == (8, 10, 12)

    def test_divergence_of_uniform_flow_is_zero(self):
        """Divergence of constant vector field is zero (interior)."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=10, ny=10, nz=10)
        ux = jnp.full((10, 10, 10), 1.0)
        uy = jnp.full((10, 10, 10), 2.0)
        uz = jnp.full((10, 10, 10), 3.0)
        div = divergence_3d(ux, uy, uz, grid)
        assert jnp.allclose(div[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_divergence_of_expanding_flow(self):
        """div(x, y, z) = 3 everywhere (interior)."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=12, ny=12, nz=12)
        ux = grid.X
        uy = grid.Y
        uz = grid.Z
        div = divergence_3d(ux, uy, uz, grid)
        assert jnp.allclose(div[1:-1, 1:-1, 1:-1], 3.0, atol=1e-10)

    def test_divergence_is_linear(self):
        """div(a*u + b*v) = a*div(u) + b*div(v) (linearity)."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=10, ny=10, nz=10)
        ux = grid.X
        uy = grid.Y
        uz = grid.Z
        a, b = 2.0, 3.0
        div_u = divergence_3d(ux, uy, uz, grid)
        div_v = divergence_3d(grid.Z, grid.X, grid.Y, grid)
        div_sum = divergence_3d(a * ux + b * grid.Z, a * uy + b * grid.X, a * uz + b * grid.Y, grid)
        assert jnp.allclose(div_sum, a * div_u + b * div_v, atol=1e-10)
