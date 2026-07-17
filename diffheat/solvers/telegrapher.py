# diffheat/solvers/telegrapher.py
"""Telegrapher (hyperbolic heat) equation solvers for 1D, 2D, and 3D.

Solves the Telegrapher equation:

    tau * u_tt + u_t = kappa * nabla^2 u + S

using a second-order central-difference time-integration scheme.  The
scheme reduces to the leapfrog wave solver when the damping term u_t is
small (large tau) and to the explicit Euler heat solver when the
inertial term tau*u_tt is small (small tau).

The time-stepping formula for a step from t^n to t^{n+1}:

    u^{n+1} = [2*tau / (tau + dt/2)] * u^n
            + [(dt/2 - tau) / (tau + dt/2)] * u^{n-1}
            + [dt^2 / (tau + dt/2)] * (kappa * nabla^2 u^n + S^n)

The first step is bootstrapped using a second-order Taylor expansion
from the initial conditions (u^0, v^0 = du/dt|_0):

    u^1 = u^0 + dt * v^0 + (dt^2 / (2*tau)) * (kappa * nabla^2 u^0 + S^0 - v^0)

Reference:
    https://mathoverflow.net/questions/343438/
    (Green's function for 3D relativistic heat equation)
"""
import logging

import jax
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D
from ..mesh.grid3d import Grid3D
from ..physics.telegrapher import (
    TelegrapherEquation1D,
    TelegrapherEquation2D,
    TelegrapherEquation3D,
)
from .stability import (
    check_cfl_telegrapher_1d,
    check_cfl_telegrapher_2d,
    check_cfl_telegrapher_3d,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1D Telegrapher Equation
# ---------------------------------------------------------------------------


def solve_telegrapher_1d(
    eqn: TelegrapherEquation1D,
    u0: jnp.ndarray,
    v0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Solve the 1D Telegrapher equation with jax.lax.scan.

    tau * u_tt + u_t = kappa * u_xx + S(x, t)

    Second-order accurate central-difference time stepping.

    Args:
        eqn: Telegrapher equation problem definition.
        u0: (n_cells,) initial temperature field at t=0.
        v0: (n_cells,) initial rate of change du/dt at t=0.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.

    Returns:
        (n_steps+1, n_cells) temperature trajectory. First frame is u0.

    Raises:
        UserWarning: if dt violates the CFL stability condition.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)

    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    # CFL check
    try:
        if not check_cfl_telegrapher_1d(eqn.grid, eqn.alpha, eqn.tau, dt):
            dx_min = float(jnp.min(eqn.grid.dx))
            alpha_val = float(jnp.max(jnp.asarray(eqn.alpha)))
            tau_val = float(jnp.max(jnp.asarray(eqn.tau)))
            c = jnp.sqrt(alpha_val / tau_val)
            cfl_limit = dx_min / c
            _logger.warning(
                f"dt={dt:.2e} exceeds 1D Telegrapher CFL limit {cfl_limit:.2e}. "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    from ..mesh.boundary import apply_boundary_conditions_1d
    from ..operators import laplacian_1d

    def laplacian_rhs(u, grid, t):
        """Compute kappa * laplacian(u) + source, with BCs applied."""
        L_u, b_source = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, eqn.bc, u
        )
        rhs = eqn.alpha * (L_u + b_source)
        if eqn.source is not None:
            rhs = rhs + eqn.source(grid.centers, t)
        return rhs

    # Precompute coefficients
    tau_arr = jnp.asarray(eqn.tau)
    dt_arr = jnp.asarray(dt)
    denom = tau_arr + dt_arr / 2.0
    coeff_a = 2.0 * tau_arr / denom
    coeff_b = (dt_arr / 2.0 - tau_arr) / denom
    coeff_c = dt_arr ** 2 / denom

    # First step: Taylor expansion
    # u_tt(0) = (1/tau) * (kappa * laplacian(u0) + S0 - v0)
    t0_val = t0
    lap_u0 = laplacian_rhs(u0, eqn.grid, t0_val)
    u_tt0 = (lap_u0 - v0) / tau_arr
    u_curr = u0 + dt * v0 + 0.5 * dt ** 2 * u_tt0
    u_prev = u0

    def telegrapher_step(carry, step_idx):
        u_prev, u_curr = carry
        t_current = t0 + (step_idx + 1) * dt
        lap_u = laplacian_rhs(u_curr, eqn.grid, t_current)
        u_next = coeff_a * u_curr + coeff_b * u_prev + coeff_c * lap_u
        return (u_curr, u_next), u_next

    n_remaining = n_steps - 1
    if n_remaining > 0:
        (_, _), traj = jax.lax.scan(
            telegrapher_step, (u_prev, u_curr), jnp.arange(n_remaining)
        )
        trajectory = jnp.concatenate(
            [u0[jnp.newaxis, :], u_curr[jnp.newaxis, :], traj], axis=0
        )
    else:
        trajectory = jnp.stack([u0, u_curr], axis=0)

    return trajectory


# ---------------------------------------------------------------------------
# 2D Telegrapher Equation
# ---------------------------------------------------------------------------


def solve_telegrapher_2d(
    eqn: TelegrapherEquation2D,
    u0: jnp.ndarray,
    v0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Solve the 2D Telegrapher equation with jax.lax.scan.

    tau * u_tt + u_t = kappa * (u_xx + u_yy) + S(x, y, t)

    Second-order accurate central-difference time stepping.

    Args:
        eqn: Telegrapher equation problem definition.
        u0: (nx, ny) initial temperature field at t=0.
        v0: (nx, ny) initial rate of change du/dt at t=0.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.

    Returns:
        (n_steps+1, nx, ny) temperature trajectory. First frame is u0.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)

    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    # CFL check
    try:
        if not check_cfl_telegrapher_2d(eqn.grid, eqn.alpha, eqn.tau, dt):
            dx_min = float(jnp.min(eqn.grid.dx))
            dy_min = float(jnp.min(eqn.grid.dy))
            alpha_val = float(jnp.max(jnp.asarray(eqn.alpha)))
            tau_val = float(jnp.max(jnp.asarray(eqn.tau)))
            c = jnp.sqrt(alpha_val / tau_val)
            cfl_limit = min(dx_min, dy_min) / (jnp.sqrt(2.0) * c)
            _logger.warning(
                f"dt={dt:.2e} exceeds 2D Telegrapher CFL limit {cfl_limit:.2e}. "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    from ..mesh.boundary import apply_boundary_conditions_2d
    from ..operators import laplacian_2d

    def laplacian_rhs(u, grid, t):
        """Compute kappa * laplacian(u) + source, with BCs applied."""
        L_u, b_source = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, eqn.bc, u
        )
        rhs = eqn.alpha * (L_u + b_source)
        if eqn.source is not None:
            rhs = rhs + eqn.source(grid.X.T, grid.Y.T, t)
        return rhs

    # Precompute coefficients
    tau_arr = jnp.asarray(eqn.tau)
    dt_arr = jnp.asarray(dt)
    denom = tau_arr + dt_arr / 2.0
    coeff_a = 2.0 * tau_arr / denom
    coeff_b = (dt_arr / 2.0 - tau_arr) / denom
    coeff_c = dt_arr ** 2 / denom

    # First step: Taylor expansion
    t0_val = t0
    lap_u0 = laplacian_rhs(u0, eqn.grid, t0_val)
    u_tt0 = (lap_u0 - v0) / tau_arr
    u_curr = u0 + dt * v0 + 0.5 * dt ** 2 * u_tt0
    u_prev = u0

    def telegrapher_step(carry, step_idx):
        u_prev, u_curr = carry
        t_current = t0 + (step_idx + 1) * dt
        lap_u = laplacian_rhs(u_curr, eqn.grid, t_current)
        u_next = coeff_a * u_curr + coeff_b * u_prev + coeff_c * lap_u
        return (u_curr, u_next), u_next

    n_remaining = n_steps - 1
    if n_remaining > 0:
        (_, _), traj = jax.lax.scan(
            telegrapher_step, (u_prev, u_curr), jnp.arange(n_remaining)
        )
        trajectory = jnp.concatenate(
            [u0[jnp.newaxis, :, :], u_curr[jnp.newaxis, :, :], traj], axis=0
        )
    else:
        trajectory = jnp.stack([u0, u_curr], axis=0)

    return trajectory


# ---------------------------------------------------------------------------
# 3D Telegrapher Equation
# ---------------------------------------------------------------------------


def solve_telegrapher_3d(
    eqn: TelegrapherEquation3D,
    u0: jnp.ndarray,
    v0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
    save_every: int = 1,
) -> jnp.ndarray:
    """Solve the 3D Telegrapher equation with jax.lax.scan.

    tau * u_tt + u_t = kappa * nabla^2 u + S(x, y, z, t)

    Second-order accurate central-difference time stepping.

    **Memory note:** A 3D trajectory can be large.  Use ``save_every`` to
    reduce memory usage by storing only every N-th frame.

    Args:
        eqn: Telegrapher equation problem definition.
        u0: (nx, ny, nz) initial temperature field at t=0.
        v0: (nx, ny, nz) initial rate of change du/dt at t=0.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.
        save_every: Save a frame every this many steps (default 1 = save all).

    Returns:
        (n_saved+1, nx, ny, nz) temperature trajectory. First frame is u0.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)

    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")
    if save_every < 1:
        raise ValueError(f"save_every must be >= 1, got {save_every}")

    # CFL check
    try:
        if not check_cfl_telegrapher_3d(eqn.grid, eqn.alpha, eqn.tau, dt):
            dx_min = float(jnp.min(eqn.grid.dx))
            dy_min = float(jnp.min(eqn.grid.dy))
            dz_min = float(jnp.min(eqn.grid.dz))
            alpha_val = float(jnp.max(jnp.asarray(eqn.alpha)))
            tau_val = float(jnp.max(jnp.asarray(eqn.tau)))
            c = jnp.sqrt(alpha_val / tau_val)
            cfl_limit = min(dx_min, dy_min, dz_min) / (jnp.sqrt(3.0) * c)
            _logger.warning(
                f"dt={dt:.2e} exceeds 3D Telegrapher CFL limit {cfl_limit:.2e}. "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    from ..mesh.boundary import apply_boundary_conditions_3d
    from ..operators import laplacian_3d

    def laplacian_rhs(u, grid, t):
        """Compute kappa * laplacian(u) + source, with BCs applied."""
        L_u, b_source = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, eqn.bc, u
        )
        rhs = eqn.alpha * (L_u + b_source)
        if eqn.source is not None:
            rhs = rhs + eqn.source(grid.X, grid.Y, grid.Z, t)
        return rhs

    # Precompute coefficients
    tau_arr = jnp.asarray(eqn.tau)
    dt_arr = jnp.asarray(dt)
    denom = tau_arr + dt_arr / 2.0
    coeff_a = 2.0 * tau_arr / denom
    coeff_b = (dt_arr / 2.0 - tau_arr) / denom
    coeff_c = dt_arr ** 2 / denom

    # First step: Taylor expansion
    t0_val = t0
    lap_u0 = laplacian_rhs(u0, eqn.grid, t0_val)
    u_tt0 = (lap_u0 - v0) / tau_arr
    u_curr = u0 + dt * v0 + 0.5 * dt ** 2 * u_tt0
    u_prev = u0

    # Inner telegrapher step for save_every sub-stepping
    def inner_telegrapher(carry, _):
        u_prev, u_curr, t = carry
        lap_u = laplacian_rhs(u_curr, eqn.grid, t)
        u_next = coeff_a * u_curr + coeff_b * u_prev + coeff_c * lap_u
        return (u_curr, u_next, t + dt), u_next

    n_outer = (n_steps - 1) // save_every

    def outer_step(carry, outer_idx):
        u_prev, u_curr = carry
        t_outer = t0 + (outer_idx * save_every + 1) * dt
        (u_prev_out, u_curr_out, _), _ = jax.lax.scan(
            inner_telegrapher, (u_prev, u_curr, t_outer), jnp.arange(save_every)
        )
        return (u_prev_out, u_curr_out), u_curr_out

    if n_outer > 0:
        (_, _), traj = jax.lax.scan(
            outer_step, (u_prev, u_curr), jnp.arange(n_outer)
        )
        trajectory = jnp.concatenate(
            [u0[jnp.newaxis, :, :, :], u_curr[jnp.newaxis, :, :, :], traj], axis=0
        )
    else:
        trajectory = jnp.stack([u0, u_curr], axis=0)

    return trajectory
