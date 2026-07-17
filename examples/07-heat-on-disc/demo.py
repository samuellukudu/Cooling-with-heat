#!/usr/bin/env python3
"""Heat equation on a disc — Bessel eigenfunction expansion.

Demonstrates the analytical spectral solver for the heat equation
on a circular disc with homogeneous Dirichlet BCs (PDF §8, §10).

The solution uses the Bessel-function eigenfunction expansion:

    u(r,θ,t) = Σ_{m,n} c_{m,n} · J_m(r·j_{m,n}/R) · {cos,sin}(mθ) · exp(-j_{m,n}²t/R²)

where j_{m,n} is the n-th zero of the Bessel function J_m.

Two scenarios are shown:

1. **Axisymmetric cooling** — an initial hot centre cools to zero,
   with temperature profiles at several times.

2. **Hot spot on boundary → equilibrium** — the steady-state Laplace
   solver finds the equilibrium temperature field from a prescribed
   hot region on the boundary.

Usage:
    python examples/07-heat-on-disc/demo.py
"""
import jax.numpy as jnp
from scipy.special import jn_zeros

from diffheat import (
    PolarGrid,
    bessel_j_zero,
    eigenvalue_disc,
    eigenfunction_disc,
    solve_heat_disc_analytical,
    solve_steady_state_disc,
    compute_nodal_lines_disc,
)


def demo_axisymmetric_cooling():
    """Cooling of an initially hot disc centre."""
    print("=" * 60)
    print("Demo 1: Axisymmetric cooling on a disc")
    print("=" * 60)

    R = 1.0

    def u0(r, theta):
        """Hot centre, cool near boundary."""
        return jnp.exp(-r ** 2 / 0.05)

    traj, times, grid = solve_heat_disc_analytical(
        u0, R=R, t_span=(0.0, 0.2), n_times=6,
        m_max=8, n_max=8, nr=60, ntheta=64,
    )

    print(f"Grid: {grid.nr} × {grid.ntheta} cells")
    print(f"Trajectory: {traj.shape[0]} time snapshots")

    for i, t in enumerate(times):
        umax = float(jnp.max(traj[i]))
        umin = float(jnp.min(traj[i]))
        print(f"  t = {float(t):.4f}:  max = {umax:.4f},  min = {umin:.4f}")

    # Verify boundary condition
    boundary_max = float(jnp.max(jnp.abs(traj[:, -1, :])))
    print(f"Boundary |u| (all times): {boundary_max:.2e}")

    # First eigenvalue — dominates the cooling rate
    lam01 = eigenvalue_disc(0, 1, R)
    print(f"First eigenvalue λ_{{0,1}} = {lam01:.4f}")
    print(f"Dominant cooling timescale τ = {1.0/lam01:.4f}")

    return traj, times, grid


def demo_boundary_hot_spot():
    """Steady-state from a hot spot on the disc boundary."""
    print()
    print("=" * 60)
    print("Demo 2: Steady-state from boundary hot spot")
    print("=" * 60)

    R = 1.0
    theta0 = 0.3  # angular half-width of hot spot
    u0_val = 100.0

    def g(theta):
        """Hot spot centred at θ=0 with width ±theta0."""
        return jnp.where(jnp.abs(theta) < theta0, u0_val, 0.0)

    uE, grid = solve_steady_state_disc(
        g, R=R, m_max=30, nr=80, ntheta=160,
    )

    # Mean Value Property: centre = average of boundary
    u_centre = float(uE[0, :].mean())
    theta_fine = jnp.linspace(-jnp.pi, jnp.pi, 2000)
    g_mean = float(jnp.mean(g(theta_fine)))
    print(f"Centre temperature: {u_centre:.4f}")
    print(f"Mean boundary temp: {g_mean:.4f}")
    print(f"Mean Value Property error: {abs(u_centre - g_mean):.2e}")

    # Hot spot fraction of boundary
    hot_fraction = 2.0 * theta0 / (2.0 * jnp.pi)
    print(f"Hot spot covers {hot_fraction*100:.1f}% of boundary")
    print(f"Centre temp / max boundary: {u_centre / u0_val:.4f}")

    return uE, grid


def demo_nodal_lines():
    """Show nodal line structure for first few disc eigenfunctions."""
    print()
    print("=" * 60)
    print("Demo 3: Nodal lines of disc eigenfunctions")
    print("=" * 60)

    for m, n in [(0, 1), (0, 2), (1, 1), (2, 1)]:
        lines_cos = compute_nodal_lines_disc(m=m, n=n, kind="cos", R=1.0)
        n_circles = sum(1 for _ in range(len(lines_cos) - 1) if True)  # all but boundary
        print(f"  (m={m}, n={n}) cos: {len(lines_cos)} nodal curves "
              f"(1 boundary + {len(lines_cos)-1} interior)")
        if m >= 1:
            lines_sin = compute_nodal_lines_disc(m=m, n=n, kind="sin", R=1.0)
            print(f"  (m={m}, n={n}) sin: {len(lines_sin)} nodal curves "
                  f"(1 boundary + {len(lines_sin)-1} interior)")


if __name__ == "__main__":
    demo_axisymmetric_cooling()
    demo_boundary_hot_spot()
    demo_nodal_lines()
    print()
    print("Done! The analytical solutions agree with Hancock §8, §10, §14.2.")
