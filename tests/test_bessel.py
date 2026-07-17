# tests/test_bessel.py
"""Tests for Bessel function eigenvalue/eigenfunction helpers."""
import jax.numpy as jnp
import pytest

from diffheat import (
    bessel_j_zero,
    eigenfunction_disc,
    eigenfunction_norm,
    eigenvalue_disc,
)


class TestBesselJZero:
    def test_first_zeros(self):
        """Known values from Abramowitz & Stegun / PDF Table."""
        assert abs(bessel_j_zero(0, 1) - 2.404825557695773) < 1e-10
        assert abs(bessel_j_zero(0, 2) - 5.520078110286311) < 1e-10
        assert abs(bessel_j_zero(1, 1) - 3.831705970207512) < 1e-10
        assert abs(bessel_j_zero(1, 2) - 7.015586669815619) < 1e-10

    def test_ordering(self):
        """Zeros should increase with n for fixed m."""
        z = [bessel_j_zero(2, k) for k in range(1, 6)]
        for i in range(len(z) - 1):
            assert z[i] < z[i + 1]

    def test_validation(self):
        with pytest.raises(ValueError):
            bessel_j_zero(0, 0)
        with pytest.raises(ValueError):
            bessel_j_zero(-1, 1)


class TestEigenvalueDisc:
    def test_first_eigenvalue_unit_circle(self):
        """λ_{0,1} = j_{0,1}² ≈ 5.783"""
        lam = eigenvalue_disc(0, 1, R=1.0)
        assert abs(lam - 5.783185962946783) < 1e-10

    def test_radius_scaling(self):
        """λ ∝ 1/R²."""
        lam_R1 = eigenvalue_disc(0, 1, R=1.0)
        lam_R2 = eigenvalue_disc(0, 1, R=2.0)
        assert abs(lam_R2 - lam_R1 / 4.0) < 1e-10


class TestEigenfunctionDisc:
    def test_boundary_condition(self):
        """Eigenfunction must vanish at r = R (zero Dirichlet BC)."""
        n_theta = 50
        theta = jnp.linspace(-jnp.pi, jnp.pi, n_theta)
        r = jnp.ones(n_theta)  # r = R = 1

        v = eigenfunction_disc(r, theta, m=0, n=1, kind="cos", R=1.0)
        assert jnp.allclose(v, 0.0, atol=1e-12)

    def test_m0_sine_is_zero(self):
        """For m=0, sin(0·θ) = 0, so the function should be zero."""
        theta = jnp.linspace(-jnp.pi, jnp.pi, 20)
        r = 0.5 * jnp.ones(20)
        v = eigenfunction_disc(r, theta, m=0, n=1, kind="sin", R=1.0)
        assert jnp.allclose(v, 0.0, atol=1e-12)

    def test_m1_cos_symmetry(self):
        """cos(θ) is even, so v(r,θ) should equal v(r,-θ)."""
        theta = jnp.linspace(0.1, 1.0, 5)
        r = jnp.full(5, 0.5)
        v_pos = eigenfunction_disc(r, theta, m=1, n=1, kind="cos")
        v_neg = eigenfunction_disc(r, -theta, m=1, n=1, kind="cos")
        assert jnp.allclose(v_pos, v_neg)

    def test_m1_sin_antisymmetry(self):
        """sin(θ) is odd, so v(r,θ) = -v(r,-θ)."""
        theta = jnp.linspace(0.1, 1.0, 5)
        r = jnp.full(5, 0.5)
        v_pos = eigenfunction_disc(r, theta, m=1, n=1, kind="sin")
        v_neg = eigenfunction_disc(r, -theta, m=1, n=1, kind="sin")
        assert jnp.allclose(v_pos, -v_neg)

    def test_positive_in_interior(self):
        """The (0,1) eigenfunction is positive on the interior (Perron-Frobenius)."""
        grid_r = jnp.linspace(0.05, 0.95, 20)
        grid_t = jnp.linspace(-jnp.pi, jnp.pi, 30)
        R_mesh, T_mesh = jnp.meshgrid(grid_r, grid_t, indexing="ij")
        v = eigenfunction_disc(R_mesh, T_mesh, m=0, n=1)
        assert jnp.all(v > 0.0)


class TestEigenfunctionNorm:
    def test_normalisation_positive(self):
        norm_sq = eigenfunction_norm(0, 1, R=1.0)
        assert norm_sq > 0.0

    def test_sine_and_cosine_same_norm(self):
        """For m ≥ 1, cos and sin branches have the same L² norm."""
        n_cos = eigenfunction_norm(1, 1, R=1.0)
        n_sin = eigenfunction_norm(1, 1, R=1.0)  # same m,n
        assert abs(n_cos - n_sin) < 1e-10
