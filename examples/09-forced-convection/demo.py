#!/usr/bin/env python3
"""Demo: 2D forced convection — channel flow over a heated component.

A 2D channel with uniform inlet flow from the left at T_cold. A heated chip
sits on the bottom wall at center. The thermal plume bends in the flow direction,
illustrating the transition from diffusion-dominated to advection-dominated
heat transfer.

Controls:
    - Adjust flow speed to see the Peclet number effect
    - Watch the thermal plume bend downstream

Run:
    python examples/09-forced-convection/demo.py
"""
import jax.numpy as jnp

from diffheat import (
    AdvectionDiffusion2D,
    BoundaryCondition2D,
    Grid2D,
    check_cfl_advection_diffusion_2d,
    get_device,
    solve_advection_diffusion_2d,
)
from diffheat.viz import run_viewer_2d


def main():
    print(f"Running on: {get_device()}")
    print("=" * 50)

    # --- Channel geometry ---
    Lx, Ly = 4.0, 1.0  # 4:1 aspect ratio channel
    nx, ny = 120, 30
    grid = Grid2D.uniform(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    dx = float(grid.dx[0])
    dy = float(grid.dy[0])
    print(f"Grid: {nx}*{ny} cells, dx = {dx:.4f}, dy = {dy:.4f}")

    # --- Material ---
    alpha = 0.01  # thermal diffusivity

    # --- Flow ---
    inlet_velocity = 1.0  # m/s, uniform inlet from left
    peclet = inlet_velocity * Lx / alpha
    print(f"Inlet velocity: {inlet_velocity} m/s")
    print(f"Peclet number (Pe = UL/alpha): {peclet:.1f}")

    # --- Boundary conditions ---
    bc = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 0.0},     # cold inlet
        right={"kind": "dirichlet", "value": 0.0},     # cold outlet (convective)
        bottom={"kind": "neumann", "value": 0.0},      # insulated bottom
        top={"kind": "neumann", "value": 0.0},         # insulated top
    )

    # --- Velocity field: uniform horizontal flow ---
    def channel_flow(X, Y, t):
        """Uniform horizontal flow through the channel."""
        u_x = inlet_velocity * jnp.ones_like(X)
        u_y = jnp.zeros_like(Y)
        return u_x, u_y

    # --- Heated chip source at bottom center ---
    # Gaussian heat source centered at (Lx/2, 0)
    chip_x0 = Lx / 2
    chip_y0 = 0.0
    chip_width = 0.1

    def chip_source(X, Y, t):
        """Gaussian heat source — heated component on bottom wall."""
        r2 = ((X - chip_x0) ** 2 + (Y - chip_y0) ** 2) / (2 * chip_width ** 2)
        return 500.0 * jnp.exp(-r2)

    eqn = AdvectionDiffusion2D(
        grid=grid,
        bc=bc,
        alpha=alpha,
        velocity=channel_flow,
        source=chip_source,
    )

    # --- Initial condition ---
    T0 = jnp.zeros((nx, ny))

    # --- Time parameters ---
    t_end = 2.0
    dt = 0.002

    # CFL check
    u_x_max = inlet_velocity
    u_y_max = 0.0
    stable = check_cfl_advection_diffusion_2d(grid, alpha, u_x_max, u_y_max, dt)
    print(f"dt: {dt:.4f} s (stable: {stable})")

    # --- Solve ---
    print(f"Solving from t=0 to t={t_end}...")
    trajectory = solve_advection_diffusion_2d(eqn, T0, (0.0, t_end), dt)

    n_steps = len(trajectory)
    print(f"Done. {n_steps} timesteps computed.")
    print(f"Initial max T: {float(jnp.max(trajectory[0])):.2f}°C")
    print(f"Final max T:   {float(jnp.max(trajectory[-1])):.2f}°C")
    print(f"Pe = {peclet:.1f} ({'advection-dominated' if peclet > 10 else 'diffusion-dominated' if peclet < 1 else 'mixed'})")

    # --- Visualize ---
    print("\nLaunching viewer...")
    run_viewer_2d(trajectory, grid, dt)


if __name__ == "__main__":
    main()
