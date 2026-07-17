#!/usr/bin/env python3
"""2D wave equation — Standing wave on a square membrane.

Demonstrates the leapfrog (Störmer-Verlet) solver for the wave equation:

    u_tt = c^2 * (u_xx + u_yy)

on a unit square with zero-Dirichlet boundaries.  Two initial conditions
are shown:

1. A pure eigenmode  sin(pi*x) * sin(pi*y)  which oscillates sinusoidally
   in time at frequency  omega = c * pi * sqrt(2).

2. A Gaussian pulse that spreads into circular wavefronts, reflects off
   the walls, and produces a complex interference pattern.

Usage:
    python examples/06-wave-equation/demo.py
"""
import jax.numpy as jnp

from diffheat import (
    BoundaryCondition2D,
    Grid2D,
    WaveEquation2D,
    check_cfl_wave_2d,
    solve_wave_2d,
    solve_steady_state_2d,
    HeatEquation2D,
)
from diffheat.utils import array


def demo_standing_mode():
    """Pure eigenmode — should oscillate without changing shape."""
    print("=" * 60)
    print("Demo 1: Standing eigenmode  sin(pi*x) * sin(pi*y)")
    print("=" * 60)

    Lx = Ly = 1.0
    nx = ny = 60
    c = 1.0

    grid = Grid2D.uniform(Lx=Lx, Ly=Ly, nx=nx, ny=ny)

    # Initial displacement: first eigenmode
    u0 = jnp.sin(jnp.pi * grid.X.T) * jnp.sin(jnp.pi * grid.Y.T)
    v0 = jnp.zeros((nx, ny))

    # Clamped edges
    bc = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 0.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "dirichlet", "value": 0.0},
        top={"kind": "dirichlet", "value": 0.0},
    )
    eqn = WaveEquation2D(grid=grid, bc=bc, c=c)

    # CFL for wave: c*dt <= min(dx,dy) / sqrt(2)
    dx = Lx / nx
    dt = 0.4 * dx / c  # well within CFL
    print(f"dx = {dx:.4f}, dt = {dt:.4f}")

    stable = check_cfl_wave_2d(grid, c, dt)
    print(f"CFL stable: {stable}")

    # One full period of the fundamental mode
    omega = c * jnp.pi * jnp.sqrt(2.0)
    T_period = 2.0 * jnp.pi / omega
    t_end = float(T_period)

    print(f"Fundamental period = {T_period:.4f}")
    print(f"Simulating to t = {t_end:.4f} (one period)...")

    traj = solve_wave_2d(eqn, u0, v0, (0.0, t_end), dt)

    n_steps = int(t_end / dt)
    print(f"Trajectory shape: {traj.shape}  ({n_steps} steps + initial)")

    # After one period the solution should return to u0 (modulo sign for
    # a pure mode — actually it should be -u0 after half period and back
    # to +u0 after full period).
    diff = jnp.max(jnp.abs(traj[-1] - u0))
    print(f"Max |u(T) - u(0)| after one period: {diff:.6f}")

    # Compute a simple energy proxy
    def energy(frame, prev_frame):
        v = (frame - prev_frame) / dt
        from diffheat.operators import gradient_2d
        gx, gy = gradient_2d(frame, grid)
        dA = grid.dx[:, None] * grid.dy[None, :]
        return 0.5 * jnp.sum((v ** 2 + c ** 2 * (gx ** 2 + gy ** 2)) * dA)

    e_init = energy(traj[1], traj[0])
    e_final = energy(traj[-1], traj[-2])
    print(f"Energy drift: {(float(e_final - e_init) / float(e_init)) * 100:.3f}%")

    return traj, grid


def demo_gaussian_pulse():
    """Gaussian pulse — wave propagation and reflection."""
    print()
    print("=" * 60)
    print("Demo 2: Gaussian pulse on a square drum")
    print("=" * 60)

    Lx = Ly = 2.0
    nx = ny = 80
    c = 1.0

    grid = Grid2D.uniform(Lx=Lx, Ly=Ly, nx=nx, ny=ny)

    Xc, Yc = grid.X.T, grid.Y.T
    # Localised pulse slightly off-centre for more interesting patterns
    u0 = jnp.exp(-((Xc - 1.0) ** 2 + (Yc - 1.3) ** 2) / 0.02)
    v0 = jnp.zeros((nx, ny))

    bc = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 0.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "dirichlet", "value": 0.0},
        top={"kind": "dirichlet", "value": 0.0},
    )
    eqn = WaveEquation2D(grid=grid, bc=bc, c=c)

    dx = Lx / nx
    dt = 0.4 * dx / c
    print(f"dx = {dx:.4f}, dt = {dt:.4f}")

    t_end = 2.0  # long enough to see reflections
    n_steps = int(t_end / dt)
    print(f"Simulating {n_steps} steps to t = {t_end:.2f}...")

    traj = solve_wave_2d(eqn, u0, v0, (0.0, t_end), dt)

    print(f"Trajectory shape: {traj.shape}")

    # Check solution stays bounded
    umax = float(jnp.max(jnp.abs(traj)))
    print(f"Max |u| across trajectory: {umax:.4f}")
    print(f"Any NaNs: {bool(jnp.any(jnp.isnan(traj)))}")

    return traj, grid


if __name__ == "__main__":
    demo_standing_mode()
    demo_gaussian_pulse()
    print()
    print("Done! Use diffheat.viz.run_viewer_2d(traj, grid) to visualise.")
