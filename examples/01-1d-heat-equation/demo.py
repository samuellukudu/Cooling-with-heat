#!/usr/bin/env python3
"""Demo: 1D heat equation — hot left boundary, cold right boundary.

A uniform rod initially at 0°C, with the left end held at 1°C and the
right end at 0°C. Over time, the temperature profile approaches the
linear steady-state solution.

Run:
    python examples/01-1d-heat-equation/demo.py
"""
import jax.numpy as jnp

from diffheat import (
    BoundaryCondition,
    Grid1D,
    HeatEquation1D,
    get_device,
    solve_heat_1d,
)
from diffheat.viz import run_viewer


def main():
    print(f"Running on: {get_device()}")
    print("=" * 50)

    # --- Grid ---
    length = 1.0  # 1 meter rod
    n_cells = 100
    grid = Grid1D.uniform(length=length, n_cells=n_cells)
    print(f"Grid: {grid.n_cells} cells, dx = {float(grid.dx[0]):.4f} m")

    # --- Boundary conditions ---
    T_left = 1.0  # hot left end
    T_right = 0.0  # cold right end
    bc = BoundaryCondition(kind="dirichlet", value=jnp.array([T_left, T_right]))

    # --- Material ---
    alpha = 0.01  # thermal diffusivity (m²/s), roughly like some polymers

    # --- Initial condition ---
    T0 = jnp.zeros(grid.n_cells)  # rod starts at 0°C everywhere

    # --- Time parameters ---
    t_end = 2.0  # simulate for 2 seconds
    dt = 0.001  # time step

    # CFL check
    dx = float(grid.dx[0])
    cfl_limit = dx**2 / (2 * alpha)
    print(f"CFL limit: {cfl_limit:.4f} s")
    print(f"dt: {dt:.4f} s (stable: {dt <= cfl_limit})")

    # --- Solve ---
    eqn = HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
    print(f"Solving from t=0 to t={t_end}...")
    trajectory = solve_heat_1d(eqn, T0, (0.0, t_end), dt)

    n_steps = len(trajectory)
    print(f"Done. {n_steps} timesteps computed.")
    print(f"Initial mean T: {jnp.mean(trajectory[0]):.4f}")
    print(f"Final mean T:   {jnp.mean(trajectory[-1]):.4f}")
    print(f"Steady-state expected mean: {(T_left + T_right) / 2:.4f}")

    # --- Visualize ---
    print("\nLaunching viewer...")
    run_viewer(trajectory, grid, dt)


if __name__ == "__main__":
    main()
