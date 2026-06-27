#!/usr/bin/env python3
"""Demo: 2D heat equation — hot left edge, cold right edge, insulated top/bottom.

A square plate initially at 0°C, with the left edge held at 100°C and the
right edge at 0°C. Top and bottom are insulated. Over time, the temperature
profile approaches the linear steady-state solution (horizontal gradient).

Run:
    python examples/02-2d-heat-equation/demo.py
"""
import jax.numpy as jnp

from diffheat import (
    BoundaryCondition2D,
    Grid2D,
    get_device,
    solve_2d,
    check_cfl_2d,
)
from diffheat.operators import laplacian_2d
from diffheat.mesh.boundary import apply_boundary_conditions_2d
from diffheat.viz import run_viewer_2d


def main():
    print(f"Running on: {get_device()}")
    print("=" * 50)

    # --- Grid ---
    Lx, Ly = 1.0, 1.0  # 1 m × 1 m plate
    nx, ny = 64, 64
    grid = Grid2D.uniform(Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    print(f"Grid: {grid.nx}×{grid.ny} cells, dx = {float(grid.dx[0]):.4f}, dy = {float(grid.dy[0]):.4f}")

    # --- Boundary conditions ---
    bc = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 100.0},   # hot left
        right={"kind": "dirichlet", "value": 0.0},     # cold right
        bottom={"kind": "neumann", "value": 0.0},      # insulated bottom
        top={"kind": "neumann", "value": 0.0},         # insulated top
    )

    # --- Material ---
    alpha = 0.01

    # --- RHS function: dT/dt = alpha * laplacian(T) with BCs ---
    def heat_rhs(T, grid, t, params):
        alpha = params["alpha"]
        L_T_mod, b_source = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, bc, T
        )
        return alpha * (L_T_mod + b_source)

    # --- Initial condition ---
    T0 = jnp.zeros((nx, ny))

    # --- Time parameters ---
    t_end = 5.0
    dt = 0.001

    # CFL check
    dx_min = float(jnp.min(grid.dx))
    dy_min = float(jnp.min(grid.dy))
    cfl_limit = min(dx_min**2, dy_min**2) / (4 * alpha)
    print(f"CFL limit: {cfl_limit:.6f} s")
    print(f"dt: {dt:.4f} s (stable: {check_cfl_2d(grid, alpha, dt)})")

    # --- Solve ---
    print(f"Solving from t=0 to t={t_end}...")
    params = {"alpha": alpha}
    trajectory = solve_2d(heat_rhs, T0, grid, (0.0, t_end), dt, params=params)

    n_steps = len(trajectory)
    print(f"Done. {n_steps} timesteps computed.")
    print(f"Initial mean T: {jnp.mean(trajectory[0]):.2f}°C")
    print(f"Final mean T:   {jnp.mean(trajectory[-1]):.2f}°C")
    print(f"Steady-state expected mean: {(100.0 + 0.0) / 2:.1f}°C")

    # --- Visualize ---
    print("\nLaunching viewer...")
    run_viewer_2d(trajectory, grid, dt)


if __name__ == "__main__":
    main()
