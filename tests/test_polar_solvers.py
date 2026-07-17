# tests/test_polar_solvers.py
"""Tests for analytical disc and cylinder solvers."""
import jax.numpy as jnp

from diffheat import (
    solve_heat_disc_analytical,
    solve_steady_state_disc,
    solve_steady_state_cylinder_3d,
)


class TestHeatDiscAnalytical:
    def test_monotonic_decay(self):
        """Temperature should decay monotonically (heat equation without source)."""
        def u0(r, theta):
            # Smooth bump centred off-origin
            return jnp.exp(-((r * jnp.cos(theta) - 0.3) ** 2
                           + (r * jnp.sin(theta) - 0.2) ** 2) / 0.05)

        traj, times, grid = solve_heat_disc_analytical(
            u0, R=1.0, t_span=(0.0, 0.1), n_times=5,
            m_max=6, n_max=6, nr=40, ntheta=64,
        )
        assert traj.shape == (5, 40, 64)

        max_vals = [float(jnp.max(traj[i])) for i in range(5)]
        for i in range(len(max_vals) - 1):
            assert max_vals[i] >= max_vals[i + 1] - 1e-10

    def test_boundary_zero(self):
        """Solution must be ~0 at r=R for all time (homogeneous Dirichlet BC)."""
        def u0(r, theta):
            return jnp.exp(-r ** 2 / 0.1)

        traj, times, grid = solve_heat_disc_analytical(
            u0, R=1.0, t_span=(0.0, 0.05), n_times=3,
            m_max=6, n_max=6, nr=30, ntheta=32,
        )
        # Last radial cell should be near zero
        for i in range(3):
            boundary_vals = traj[i, -1, :]
            assert float(jnp.max(jnp.abs(boundary_vals))) < 0.05

    def test_returns_grid(self):
        """Should return a valid PolarGrid."""
        def u0(r, theta):
            return 1.0 - r ** 2  # paraboloid, ~0 at r=1

        traj, times, grid = solve_heat_disc_analytical(
            u0, R=1.0, t_span=(0.0, 0.02), n_times=2,
            m_max=4, n_max=4, nr=20, ntheta=16,
        )
        assert grid.nr == 20
        assert grid.ntheta == 16


class TestSteadyStateDisc:
    def test_constant_boundary(self):
        """If g(θ) = const, u_E should equal that constant everywhere."""
        def g(theta):
            return 100.0 * jnp.ones_like(theta)

        uE, grid = solve_steady_state_disc(g, R=1.0, m_max=10, nr=30, ntheta=64)
        assert jnp.allclose(uE, 100.0, atol=1e-10)

    def test_mean_value_at_centre(self):
        """Centre value must equal mean of boundary data (Mean Value Property)."""
        def g(theta):
            # Asymmetric boundary data
            return 50.0 + 30.0 * jnp.cos(theta) + 20.0 * jnp.sin(2.0 * theta)

        uE, grid = solve_steady_state_disc(g, R=1.0, m_max=10, nr=50, ntheta=128)

        # Centre is at r ≈ dr/2 (first radial cell)
        u_centre = float(uE[0, :].mean())
        # Analytical mean: only the constant term survives
        g_mean = 50.0
        assert abs(u_centre - g_mean) < 0.03

    def test_cos_theta_boundary(self):
        """g(θ) = cos(θ) → u_E(r,θ) = r * cos(θ)."""
        def g(theta):
            return jnp.cos(theta)

        uE, grid = solve_steady_state_disc(g, R=1.0, m_max=10, nr=40, ntheta=64)

        # uE should equal r*cos(θ) = X coordinate
        expected = grid.X  # X = r*cos(θ) for R=1
        assert jnp.allclose(uE, expected, atol=0.02)

    def test_shape(self):
        def g(theta):
            return jnp.sin(theta)

        uE, grid = solve_steady_state_disc(g, R=2.0, m_max=8, nr=30, ntheta=48)
        assert uE.shape == (30, 48)


class TestSteadyStateCylinder3D:
    def test_zero_boundary_gives_zero(self):
        """If g(θ,z) = 0, solution should be zero everywhere."""
        def g(theta, z):
            return jnp.zeros((len(theta), len(z)))

        uE, r_vals, z_vals = solve_steady_state_cylinder_3d(
            g, a=1.0, L=2.0, m_max=4, n_max=4, nr=20, nz=20,
        )
        assert jnp.allclose(uE, 0.0, atol=1e-10)

    def test_output_shapes(self):
        def g(theta, z):
            return jnp.ones((len(theta), len(z)))

        uE, r_vals, z_vals = solve_steady_state_cylinder_3d(
            g, a=1.0, L=3.0, m_max=3, n_max=3, nr=30, nz=25,
        )
        assert uE.shape == (30, 25)
        assert r_vals.shape == (30,)
        assert z_vals.shape == (25,)

    def test_ends_zero(self):
        """Solution at z≈0 and z≈L should be near zero (homogeneous Dirichlet at ends)."""
        def g(theta, z):
            return jnp.sin(jnp.pi * z / 2.0)[jnp.newaxis, :] * jnp.ones((len(theta), 1))

        uE, r_vals, z_vals = solve_steady_state_cylinder_3d(
            g, a=1.0, L=2.0, m_max=4, n_max=4, nr=20, nz=20,
        )
        # Values near z=0
        assert float(jnp.max(jnp.abs(uE[:, 0]))) < 0.1
        # Values near z=L
        assert float(jnp.max(jnp.abs(uE[:, -1]))) < 0.1
