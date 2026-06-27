#!/usr/bin/env python3
"""Demo: 3D heat equation with a hot central core.

A 3D domain initially at 0°C has a hot sphere in the center. All outer boundary
faces are held at 0°C (cooling). Over time, heat diffuses from the core to the
boundaries, smoothing out the temperature field.

Run:
    python examples/03-3d-heat-equation/demo.py
"""
import jax.numpy as jnp

from diffheat import (
    BoundaryCondition3D,
    Grid3D,
    get_device,
    solve_3d,
    check_cfl_3d,
)
from diffheat.operators import laplacian_3d
from diffheat.mesh.boundary import apply_boundary_conditions_3d
from diffheat.viz import run_viewer_3d


def main():
    print(f"Running on: {get_device()}")
    print("=" * 50)

    # --- Grid ---
    Lx, Ly, Lz = 1.0, 1.0, 1.0  # 1m x 1m x 1m cube
    nx, ny, nz = 32, 32, 32     # 32x32x32 cells
    grid = Grid3D.uniform(Lx=Lx, Ly=Ly, Lz=Lz, nx=nx, ny=ny, nz=nz)
    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz} cells")
    print(f"dx = {float(grid.dx[0]):.4f}, dy = {float(grid.dy[0]):.4f}, dz = {float(grid.dz[0]):.4f}")

    # --- Boundary conditions (all faces held at 0°C) ---
    cold = {"kind": "dirichlet", "value": 0.0}
    bc = BoundaryCondition3D(
        left=cold, right=cold,
        bottom=cold, top=cold,
        front=cold, back=cold,
    )

    # --- Material ---
    alpha = 0.05

    # --- RHS function: dT/dt = alpha * laplacian(T) with BCs ---
    def heat_rhs(T, grid, t, params):
        alpha = params["alpha"]
        L_T_mod, b_source = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, bc, T
        )
        return alpha * (L_T_mod + b_source)

    # --- Initial condition (Hot core in the center) ---
    X, Y, Z = grid.X, grid.Y, grid.Z
    distance_squared = (X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2
    # Radius = 0.2 m, max temperature = 100°C
    T0 = jnp.where(distance_squared <= 0.2**2, 100.0, 0.0)

    # --- Time parameters ---
    t_end = 2.0
    dt = 0.001
    save_every = 5  # Save every 5 steps to conserve memory

    # CFL check
    cfl_stable = check_cfl_3d(grid, alpha, dt)
    h_min = min(float(grid.dx[0]), float(grid.dy[0]), float(grid.dz[0]))
    cfl_limit = h_min**2 / (6 * alpha)
    print(f"CFL limit: {cfl_limit:.6f} s")
    print(f"dt: {dt:.4f} s (stable: {cfl_stable})")

    # --- Solve ---
    print(f"Solving from t=0 to t={t_end}...")
    params = {"alpha": alpha}
    trajectory = solve_3d(heat_rhs, T0, grid, (0.0, t_end), dt, params=params, save_every=save_every)

    n_steps = len(trajectory)
    print(f"Done. {n_steps} frames stored.")
    print(f"Initial max T: {jnp.max(trajectory[0]):.2f}°C")
    print(f"Final max T:   {jnp.max(trajectory[-1]):.2f}°C")

    # --- Visualize ---
    print("\nLaunching 3D slice viewer...")
    run_viewer_3d(trajectory, grid, dt * save_every)


if __name__ == "__main__":
    main()
