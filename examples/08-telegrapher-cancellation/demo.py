#!/usr/bin/env python3
"""Thermal wave interference — Telegrapher (hyperbolic heat) equation.

Demonstrates finite-speed heat propagation and active thermal cancellation
using the Telegrapher equation:

    tau * u_tt + u_t = kappa * nabla^2 u + S(x, y, t)

where tau > 0 is the thermal relaxation time.  Unlike the classical
Fourier heat equation (tau = 0), this model predicts that thermal
disturbances travel at finite speed v = sqrt(kappa / tau).

Two scenarios are shown:

1. **Finite-speed wavefront** — A Gaussian heat pulse on a 2D plate
   propagates outward as a damped wave, with a visible wavefront
   travelling at speed v = sqrt(kappa / tau).  Classical Fourier
   diffusion would show instantaneous (though tiny) temperature rise
   everywhere.

2. **Active pulse cancellation** — A primary heat pulse is fired at the
   left edge.  A secondary "cooling" pulse (negative amplitude) is
   optimised via gradient descent to cancel the arriving thermal
   wavefront at a target sensor location.  This is only possible
   because the Telegrapher equation supports wave-like interference.

Usage:
    python examples/08-telegrapher-cancellation/demo.py
"""
import jax
import jax.numpy as jnp

from diffheat import (
    BoundaryCondition2D,
    Grid2D,
    TelegrapherEquation2D,
    solve_telegrapher_2d,
)


def demo_finite_speed_wavefront():
    """Show a thermal pulse propagating at finite speed."""
    print("=" * 60)
    print("Demo 1: Finite-speed thermal wavefront")
    print("=" * 60)

    # Setup
    grid = Grid2D.uniform(Lx=2.0, Ly=2.0, nx=80, ny=80)
    bc = BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 0.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "dirichlet", "value": 0.0},
        top={"kind": "dirichlet", "value": 0.0},
    )

    # Parameters: kappa = 1.0, tau = 0.5
    # Wave speed: v = sqrt(1.0 / 0.5) = sqrt(2) ≈ 1.414
    kappa = 1.0
    tau = 0.5
    v_wave = float(jnp.sqrt(kappa / tau))

    print(f"  kappa = {kappa}, tau = {tau}")
    print(f"  Thermal wave speed v = sqrt(kappa/tau) = {v_wave:.3f}")

    eqn = TelegrapherEquation2D(grid=grid, bc=bc, alpha=kappa, tau=tau)

    # Gaussian pulse at centre
    X = grid.X.T
    Y = grid.Y.T
    u0 = jnp.exp(-((X - 1.0) ** 2 + (Y - 1.0) ** 2) / 0.02)
    v0 = jnp.zeros_like(u0)

    # CFL: c=1.414, dx=0.025, limit = dx/(sqrt2*c) = 0.025/2.0 = 0.0125
    dt = 0.002
    t_end = 0.3
    traj = solve_telegrapher_2d(eqn, u0, v0, t_span=(0.0, t_end), dt=dt)

    print(f"  Grid: {grid.nx}×{grid.ny}, dt={dt}, steps={traj.shape[0] - 1}")

    # Track peak temperature and wavefront radius over time
    times = jnp.linspace(0.0, t_end, traj.shape[0])
    for i, t in enumerate(times):
        if i % max(1, len(times) // 6) == 0:
            frame = traj[i]
            u_max = float(jnp.max(frame))
            # Where is the half-max contour? Find max radius with u > umax/2
            r_grid = jnp.sqrt((X - 1.0) ** 2 + (Y - 1.0) ** 2)
            above_half = frame > u_max / 2.0
            if jnp.any(above_half):
                r_front = float(jnp.max(jnp.where(above_half, r_grid, 0.0)))
            else:
                r_front = 0.0
            print(f"    t = {float(t):.4f}:  peak = {u_max:.4f},  "
                  f"half-max radius = {r_front:.4f},  "
                  f"expected radius = {v_wave * float(t):.4f}")

    # Verify finite speed: after short time, far-field should be exactly zero
    # (not just "very small" as in Fourier diffusion)
    far_mask = r_grid > 0.5
    far_field_max = float(jnp.max(jnp.abs(traj[10][far_mask])))
    print(f"  After {float(times[10]):.4f}s: max |u| beyond r=0.5: {far_field_max:.2e}")
    if far_field_max < 1e-6:
        print("  ✓ Far field is EXACTLY zero (finite speed confirmed)")
    else:
        print(f"  (far field non-zero — expected for diffusive tail)")

    return traj, grid


def demo_pulse_cancellation():
    """Cancel a thermal wavefront using a symmetric opposing pulse."""
    print()
    print("=" * 60)
    print("Demo 2: Active thermal pulse cancellation")
    print("=" * 60)

    from diffheat import (
        BoundaryCondition,
        Grid1D,
        TelegrapherEquation1D,
        solve_telegrapher_1d,
    )

    grid = Grid1D.uniform(length=1.0, n_cells=120)
    bc = BoundaryCondition(kind="dirichlet", value=jnp.array([0.0, 0.0]))

    kappa = 1.0
    tau = 0.5
    v_wave = float(jnp.sqrt(kappa / tau))
    print(f"  kappa = {kappa}, tau = {tau}, wave speed = {v_wave:.3f}")

    # Symmetric setup: sensor at centre, pulses equidistant on each side
    sensor_x = 0.5
    sensor_idx = 60  # cell at centre
    dist = 0.2  # distance from sensor to each pulse

    x_primary = sensor_x - dist  # 0.3
    x_secondary = sensor_x + dist  # 0.7
    travel_time = dist / v_wave
    print(f"  Sensor at x = {sensor_x}")
    print(f"  Primary pulse at x = {x_primary:.2f}")
    print(f"  Secondary pulse at x = {x_secondary:.2f}")
    print(f"  Travel time to sensor: {travel_time:.4f}s")

    # Both pulses arrive at sensor simultaneously → ideal interference
    u0_primary = jnp.exp(-((grid.centers - x_primary) ** 2) / 0.0005)
    v0 = jnp.zeros_like(u0_primary)

    dt = 0.0002
    t_end = 0.25  # enough time for both waves to reach and pass the sensor

    eqn = TelegrapherEquation1D(grid=grid, bc=bc, alpha=kappa, tau=tau)

    # --- Step 1: Primary pulse alone ---
    traj_primary = solve_telegrapher_1d(eqn, u0_primary, v0, t_span=(0.0, t_end), dt=dt)
    peak_primary = float(jnp.max(jnp.abs(traj_primary[:, sensor_idx])))
    print(f"\n  Primary pulse only: peak |u| at sensor = {peak_primary:.4f}")

    # --- Step 2: Symmetric cancellation (analytical guess) ---
    # Equal amplitude, opposite sign → waves cancel at symmetric midpoint
    u0_cancel = u0_primary - jnp.exp(-((grid.centers - x_secondary) ** 2) / 0.0005)
    traj_cancel = solve_telegrapher_1d(eqn, u0_cancel, v0, t_span=(0.0, t_end), dt=dt)
    peak_cancel = float(jnp.max(jnp.abs(traj_cancel[:, sensor_idx])))
    cancellation_symmetric = (peak_primary - peak_cancel) / peak_primary * 100.0
    print(f"  Symmetric cancellation (A=-1.0): peak = {peak_cancel:.4f} "
          f"({cancellation_symmetric:.1f}% reduction)")

    # --- Step 3: Gradient-based fine-tuning of amplitude ---
    print("\n  Fine-tuning secondary amplitude via gradient descent...")

    def loss_fn(A):
        A = jnp.asarray(A)
        u0_secondary = A * jnp.exp(-((grid.centers - x_secondary) ** 2) / 0.0005)
        u0_total = u0_primary + u0_secondary
        traj = solve_telegrapher_1d(eqn, u0_total, v0, t_span=(0.0, t_end), dt=dt)
        sensor_vals = traj[:, sensor_idx]
        return jnp.mean(sensor_vals ** 2)

    grad_fn = jax.grad(loss_fn)
    A_opt = -0.8  # initial guess away from -1.0
    loss_init = float(loss_fn(A_opt))
    print(f"    Initial A = {A_opt:.3f}, loss = {loss_init:.6f}")

    lr = 0.05
    for iteration in range(60):
        g = float(grad_fn(A_opt))
        A_opt = A_opt - lr * g
        A_opt = max(-2.0, min(0.0, A_opt))
        if iteration % 20 == 19 or iteration == 0:
            loss_val = float(loss_fn(A_opt))
            print(f"    iter {iteration + 1:2d}: A = {A_opt:.4f}, "
                  f"grad = {g:.4f}, loss = {loss_val:.6f}")

    loss_final = float(loss_fn(A_opt))
    print(f"    Final A = {A_opt:.4f}, loss = {loss_final:.6f}")
    print(f"    Loss reduction: {(loss_init - loss_final) / loss_init * 100:.1f}%")

    # --- Step 4: Final comparison ---
    u0_opt = u0_primary + A_opt * jnp.exp(
        -((grid.centers - x_secondary) ** 2) / 0.0005
    )
    traj_opt = solve_telegrapher_1d(eqn, u0_opt, v0, t_span=(0.0, t_end), dt=dt)
    peak_opt = float(jnp.max(jnp.abs(traj_opt[:, sensor_idx])))
    cancellation_opt = (peak_primary - peak_opt) / peak_primary * 100.0

    print(f"\n  Results:")
    print(f"    Primary only peak:     {peak_primary:.4f}")
    print(f"    Symmetric (A=-1.0):   {peak_cancel:.4f}  "
          f"({cancellation_symmetric:.1f}% reduction)")
    print(f"    Optimised (A={A_opt:.3f}):  {peak_opt:.4f}  "
          f"({cancellation_opt:.1f}% reduction)")

    if cancellation_opt > 20.0:
        print("  ✓ Active thermal cancellation successful!")
    else:
        print("  (modest cancellation — damping reduces coherence)")

    return traj_primary, traj_opt, grid


if __name__ == "__main__":
    demo_finite_speed_wavefront()
    demo_pulse_cancellation()
    print()
    print("Done! The Telegrapher equation enables finite-speed thermal")
    print("propagation and wave-like interference for active cooling design.")
