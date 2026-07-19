"""Tests for advection operators."""
import jax
import jax.numpy as jnp
import pytest


class TestAdvection1D:
    @pytest.fixture
    def grid_1d(self):
        """100 cells, length 10.0, dx = 0.1."""
        from diffheat.mesh import Grid1D
        return Grid1D.uniform(length=10.0, n_cells=100)

    def test_zero_velocity_returns_zero(self, grid_1d):
        """advection_1d with u=0 everywhere should return zeros."""
        from diffheat.operators.advection import advection_1d

        T = jnp.sin(2 * jnp.pi * grid_1d.centers / grid_1d.length)
        u = jnp.zeros_like(T)
        result = advection_1d(T, u, grid_1d.dx)
        assert result.shape == T.shape
        assert jnp.allclose(result, 0.0)

    def test_constant_temperature_returns_zero(self, grid_1d):
        """Advection of constant field should be zero regardless of velocity."""
        from diffheat.operators.advection import advection_1d

        T = jnp.full(grid_1d.n_cells, 5.0)
        u = jnp.full(grid_1d.n_cells, 2.0)
        result = advection_1d(T, u, grid_1d.dx)
        # interior should be zero; boundaries may differ due to roll wrap-around
        assert jnp.allclose(result[1:-1], 0.0, atol=1e-10)

    def test_positive_velocity_uses_backward_difference(self, grid_1d):
        """With u > 0 everywhere, advection uses backward difference."""
        from diffheat.operators.advection import advection_1d

        # Linear T = a*x => dT/dx = a everywhere
        a = 3.0
        T = a * grid_1d.centers
        # With u = 2.0 everywhere: advection = -u * dT/dx = -2.0 * a
        u = jnp.full(grid_1d.n_cells, 2.0)
        result = advection_1d(T, u, grid_1d.dx)
        expected = -2.0 * a
        # First cell uses forward difference (one-sided), skip it
        assert jnp.allclose(result[1:], expected, atol=0.1)

    def test_negative_velocity_uses_forward_difference(self, grid_1d):
        """With u < 0 everywhere, advection uses forward difference."""
        from diffheat.operators.advection import advection_1d

        a = 3.0
        T = a * grid_1d.centers
        u = jnp.full(grid_1d.n_cells, -2.0)
        result = advection_1d(T, u, grid_1d.dx)
        expected = 2.0 * a  # -(-2.0) * a = 2.0 * a
        # Last cell uses backward difference (one-sided), skip it
        assert jnp.allclose(result[:-1], expected, atol=0.1)

    def test_gaussian_translation(self, grid_1d):
        """A Gaussian pulse advected at constant u should translate."""
        from diffheat.operators.advection import advection_1d

        dx = float(jnp.mean(grid_1d.dx))
        x = grid_1d.centers
        sigma = 0.5
        x0 = 5.0
        # Gaussian centered at x0
        T = jnp.exp(-((x - x0) ** 2) / (2 * sigma**2))
        u = jnp.full_like(T, 1.0)  # uniform flow to the right

        # After one explicit Euler step: T_new = T - dt * u * dT/dx
        dt = 0.01
        adv = advection_1d(T, u, grid_1d.dx)
        T_new = T + dt * adv

        # Exact: Gaussian centered at x0 + u*dt
        T_exact = jnp.exp(-((x - (x0 + 1.0 * dt)) ** 2) / (2 * sigma**2))
        # Check that peak moved right (interior only, skip boundaries)
        assert jnp.max(T_new[10:-10]) > jnp.max(T[10:-10]) * 0.9
        # Peak should be near new position
        peak_idx = jnp.argmax(T_new[10:-10]) + 10
        expected_peak_idx = jnp.argmax(T_exact)
        assert abs(peak_idx - expected_peak_idx) <= 3  # within 3 cells

    def test_is_jax_differentiable(self, grid_1d):
        """jax.grad should work through advection_1d."""
        from diffheat.operators.advection import advection_1d

        T = jnp.sin(grid_1d.centers)
        u = jnp.ones_like(T)
        dx = float(jnp.mean(grid_1d.dx))

        # Use sum of squares to avoid telescoping cancellation
        def sum_sq_adv(T):
            adv = advection_1d(T, u, dx)
            return jnp.sum(adv ** 2)

        grad = jax.grad(sum_sq_adv)(T)
        assert grad.shape == T.shape
        assert not jnp.allclose(grad, 0.0)  # gradient should be non-zero


class TestAdvection2D:
    @pytest.fixture
    def grid_2d(self):
        """40x30 grid on [0,2]x[0,1]."""
        from diffheat.mesh import Grid2D
        return Grid2D.uniform(Lx=2.0, Ly=1.0, nx=40, ny=30)

    def test_returns_correct_shape(self, grid_2d):
        from diffheat.operators.advection import advection_2d
        T = jnp.ones((grid_2d.nx, grid_2d.ny))
        u_x = jnp.zeros_like(T)
        u_y = jnp.zeros_like(T)
        result = advection_2d(T, u_x, u_y, grid_2d.dx, grid_2d.dy)
        assert result.shape == (grid_2d.nx, grid_2d.ny)

    def test_zero_velocity_returns_zero(self, grid_2d):
        from diffheat.operators.advection import advection_2d
        T = jnp.sin(jnp.pi * jnp.arange(grid_2d.nx)[:, None] / grid_2d.nx)
        u_x = jnp.zeros_like(T)
        u_y = jnp.zeros_like(T)
        result = advection_2d(T, u_x, u_y, grid_2d.dx, grid_2d.dy)
        assert jnp.allclose(result, 0.0)

    def test_uniform_x_flow(self, grid_2d):
        """With u_x > 0, u_y = 0: result should match 1D advection row-by-row."""
        from diffheat.operators.advection import advection_2d

        # T varies only in x: T = a * x
        a = 2.0
        T = a * grid_2d.x_centers[:, None] * jnp.ones(grid_2d.ny)[None, :]
        u_x = jnp.full_like(T, 1.5)
        u_y = jnp.zeros_like(T)
        dx = float(jnp.mean(grid_2d.dx))

        result = advection_2d(T, u_x, u_y, grid_2d.dx, grid_2d.dy)
        expected = -1.5 * a  # -u_x * dT/dx
        # interior x, all y
        assert jnp.allclose(result[1:-1, :], expected, atol=0.15)

    def test_uniform_y_flow(self, grid_2d):
        """With u_x = 0, u_y > 0: advection acts only in y."""
        from diffheat.operators.advection import advection_2d

        # T varies only in y: T = b * y
        b = 2.0
        T = b * jnp.ones(grid_2d.nx)[:, None] * grid_2d.y_centers[None, :]
        u_x = jnp.zeros_like(T)
        u_y = jnp.full_like(T, 3.0)
        dy = float(jnp.mean(grid_2d.dy))

        result = advection_2d(T, u_x, u_y, grid_2d.dx, grid_2d.dy)
        expected = -3.0 * b
        # all x, interior y
        assert jnp.allclose(result[:, 1:-1], expected, atol=0.15)

    def test_is_jax_differentiable(self, grid_2d):
        from diffheat.operators.advection import advection_2d

        # Use a full 2D field and sum of squares to avoid telescoping cancellation
        X = grid_2d.x_centers[None, :]  # (1, nx), broadcast
        Y = grid_2d.y_centers[:, None]  # (ny, 1), broadcast
        T = jnp.sin(jnp.pi * X.T / grid_2d.Lx) * jnp.cos(jnp.pi * Y.T / grid_2d.Ly)
        u_x = jnp.ones_like(T)
        u_y = jnp.zeros_like(T)

        def sum_sq_adv(T):
            adv = advection_2d(T, u_x, u_y, grid_2d.dx, grid_2d.dy)
            return jnp.sum(adv ** 2)

        grad = jax.grad(sum_sq_adv)(T)
        assert grad.shape == T.shape
        assert not jnp.allclose(grad, 0.0)
