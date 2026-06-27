#!/usr/bin/env python3
"""Demo: 3D Differentiable Forced Convection Cooling.

Simulates laminar fluid flow in a channel cooling a heated electronic component.
Governing PDE:
    dT/dt + u·∇T = α ∇²T + Q(x,y,z)

Where:
    - u is a prescribed laminar parabolic velocity field.
    - Q is the constant volumetric heat source of the component.
    - Boundaries: Dirichlet at inlet (x=0), Neumann (outflow/insulated) elsewhere.

Optimizing / Auto-diff:
    Computes the gradient of the maximum system temperature with respect to the
    maximum inlet velocity (U_max) using JAX's reverse-mode differentiation.

Run:
    python examples/04-3d-forced-convection/demo.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from diffheat import (
    BoundaryCondition3D,
    Grid3D,
    get_device,
    solve_3d,
    check_cfl_3d,
    laplacian_3d,
    apply_boundary_conditions_3d,
    gradient_3d,
)
from diffheat.viz import run_viewer_3d


def run_simulation(U_max: float, grid: Grid3D, dt: float, t_end: float, save_every: int):
    """Run convection simulation and return the final temperature field."""
    # --- Velocity Field (Laminar channel profile) ---
    # Parabolic in y and z, zero at walls
    # u_x(x, y, z) = U_max * (16 * y * (Ly - y) * z * (Lz - z) / (Ly^2 * Lz^2))
    # Normalized so the peak velocity is U_max
    Ly, Lz = grid.Ly, grid.Lz
    Y, Z = grid.Y, grid.Z
    ux = U_max * 16.0 * (Y * (Ly - Y)) * (Z * (Lz - Z)) / (Ly**2 * Lz**2)
    uy = jnp.zeros_like(ux)
    uz = jnp.zeros_like(ux)

    # --- Heat Source Q (Heated block at bottom center) ---
    # Centered at x=1.0, y=0.5, z=0.1
    X = grid.X
    Q = jnp.where(
        (X >= 0.8) & (X <= 1.2) &
        (Y >= 0.4) & (Y <= 0.6) &
        (Z >= 0.0) & (Z <= 0.2),
        250.0,  # Heat generation rate
        0.0
    )

    # --- Boundary Conditions ---
    # Inlet (x=0): 20.0 °C
    # Outlet (x=Lx): Neumann 0.0 (outflow convective)
    # Others: Neumann 0.0 (insulated walls)
    bc = BoundaryCondition3D(
        left={"kind": "dirichlet", "value": 20.0},
        right={"kind": "neumann", "value": 0.0},
        bottom={"kind": "neumann", "value": 0.0},
        top={"kind": "neumann", "value": 0.0},
        front={"kind": "neumann", "value": 0.0},
        back={"kind": "neumann", "value": 0.0},
    )

    alpha = 0.02  # Thermal diffusivity

    def rhs_fn(T, grid, t, params):
        # 1. Diffusion term
        L_T, b = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, bc, T
        )
        diffusion = alpha * (L_T + b)

        # 2. Advection term: u·∇T = ux * dT/dx + uy * dT/dy + uz * dT/dz
        dT_dx, dT_dy, dT_dz = gradient_3d(T, grid)
        advection = ux * dT_dx + uy * dT_dy + uz * dT_dz

        # dT/dt = diffusion - advection + Q
        return diffusion - advection + Q

    # Initial condition: uniform inlet temperature
    T0 = jnp.full((grid.nx, grid.ny, grid.nz), 20.0)

    # Solve trajectory
    trajectory = solve_3d(
        rhs_fn, T0, grid, (0.0, t_end), dt, save_every=save_every
    )
    return trajectory


def main():
    print(f"Running on: {get_device()}")
    print("=" * 50)

    # --- Grid ---
    Lx, Ly, Lz = 2.0, 1.0, 1.0  # Elongated channel
    nx, ny, nz = 32, 16, 16
    grid = Grid3D.uniform(Lx=Lx, Ly=Ly, Lz=Lz, nx=nx, ny=ny, nz=nz)
    print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz} cells")

    # Time parameters
    dt = 0.002
    t_end = 1.0
    save_every = 5

    # Base velocity
    U_max_val = 2.0
    
    # Check CFL for diffusion part (advection also adds stability limits, but check diffusion as baseline)
    alpha = 0.02
    cfl_stable = check_cfl_3d(grid, alpha, dt)
    print(f"CFL diffusion check: dt={dt} (stable={cfl_stable})")

    # --- Step 1: Run and Visualize ---
    print(f"\nSimulating channel cooling with U_max = {U_max_val} m/s...")
    trajectory = run_simulation(U_max_val, grid, dt, t_end, save_every)
    
    print("Done.")
    print(f"Initial max T: {float(trajectory[0].max()):.2f} °C")
    print(f"Final max T:   {float(trajectory[-1].max()):.2f} °C")

    # --- Step 2: Differentiate with JAX ---
    print("\nComputing sensitivity of average final temperature w.r.t U_max...")

    def loss_fn(u_max):
        # We run for a shorter time and smaller grid to keep gradient calculation fast
        grid_small = Grid3D.uniform(Lx=2.0, Ly=1.0, Lz=1.0, nx=16, ny=8, nz=8)
        traj = run_simulation(u_max, grid_small, dt=0.005, t_end=0.05, save_every=1)
        # Target: minimize the average temperature of the final state
        return jnp.mean(traj[-1])

    # Compute loss value and gradient
    val, grad = jax.value_and_grad(loss_fn)(U_max_val)
    print(f"Loss value (mean T): {val:.4f} °C")
    print(f"d(mean T) / d(U_max): {grad:.4f} °C/(m/s)")
    print("Notice the negative gradient: increasing velocity decreases the system temperature!")

    # --- Step 3: Launch Viewer ---
    print("\nLaunching 3D slice viewer...")
    run_viewer_3d(trajectory, grid, dt * save_every)


if __name__ == "__main__":
    main()
