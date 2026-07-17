# diffheat/solvers/wave.py
"""Leapfrog (Störmer-Verlet) wave equation solvers for 1D, 2D, and 3D.

Solves the wave equation:

    u_tt = c^2 * nabla^2 u + S

using the second-order explicit leapfrog time-integration scheme.

For homogeneous Dirichlet or Neumann BCs, the leapfrog scheme is
symplectic and conserves a discrete energy exactly (to machine
precision for linear problems).

Reference:
    Hancock, MIT 18.303 Fall 2006, "Heat & Wave Equations in 2D/3D", §2-4.
"""
import logging

import jax
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D
from ..mesh.grid3d import Grid3D
from ..physics.wave import WaveEquation1D, WaveEquation2D, WaveEquation3D
from .stability import check_cfl_wave_1d, check_cfl_wave_2d, check_cfl_wave_3d

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1D Wave Equation
# ---------------------------------------------------------------------------

def solve_wave_1d(
    eqn: WaveEquation1D,
    u0: jnp.ndarray,
    v0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Solve the 1D wave equation using the leapfrog scheme with jax.lax.scan.

    u_tt = c^2 * u_xx + S(x, t)

    The scheme is second-order accurate, symplectic, and energy-conserving.

    Args:
        eqn: Wave equation problem definition.
        u0: (n_cells,) initial displacement field at t=0.
        v0: (n_cells,) initial velocity field du/dt at t=0.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.

    Returns:
        (n_steps+1, n_cells) displacement trajectory. First frame is u0.

    Raises:
        UserWarning: if dt violates the CFL stability condition.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)

    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    # CFL check
    try:
        if not check_cfl_wave_1d(eqn.grid, eqn.c, dt):
            dx_min = float(jnp.min(eqn.grid.dx))
            c_max = float(jnp.max(jnp.asarray(eqn.c)))
            cfl_limit = dx_min / c_max
            _logger.warning(
                f"dt={dt:.2e} exceeds 1D wave CFL limit {cfl_limit:.2e}. "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    from ..mesh.boundary import apply_boundary_conditions_1d
    from ..operators import laplacian_1d

    def laplacian_rhs(u, grid, t):
        """Compute c^2 * laplacian(u) + source, with BCs applied."""
        L_u, b_source = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, eqn.bc, u
        )
        rhs = eqn.c ** 2 * (L_u + b_source)
        if eqn.source is not None:
            rhs = rhs + eqn.source(grid.centers, t)
        return rhs

    # First step: Taylor expansion u^1 = u^0 + dt*v^0 + (dt^2/2)*c^2*laplacian(u^0)
    t0_val = t0
    lap_u0 = laplacian_rhs(u0, eqn.grid, t0_val)
    u_curr = u0 + dt * v0 + 0.5 * dt ** 2 * lap_u0
    u_prev = u0

    def leapfrog_step(carry, step_idx):
        u_prev, u_curr = carry
        t_current = t0 + (step_idx + 1) * dt  # step_idx 0 corresponds to step 1→2
        lap_u = laplacian_rhs(u_curr, eqn.grid, t_current)
        u_next = 2.0 * u_curr - u_prev + dt ** 2 * lap_u
        return (u_curr, u_next), u_next

    n_remaining = n_steps - 1
    if n_remaining > 0:
        (_, _), traj = jax.lax.scan(
            leapfrog_step, (u_prev, u_curr), jnp.arange(n_remaining)
        )
        trajectory = jnp.concatenate(
            [u0[jnp.newaxis, :], u_curr[jnp.newaxis, :], traj], axis=0
        )
    else:
        # Single-step simulation
        trajectory = jnp.stack([u0, u_curr], axis=0)

    return trajectory


# ---------------------------------------------------------------------------
# 2D Wave Equation
# ---------------------------------------------------------------------------

def solve_wave_2d(
    eqn: WaveEquation2D,
    u0: jnp.ndarray,
    v0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Solve the 2D wave equation using the leapfrog scheme with jax.lax.scan.

    u_tt = c^2 * (u_xx + u_yy) + S(x, y, t)

    Second-order accurate, symplectic, and energy-conserving.

    Args:
        eqn: Wave equation problem definition.
        u0: (nx, ny) initial displacement field at t=0.
        v0: (nx, ny) initial velocity field du/dt at t=0.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.

    Returns:
        (n_steps+1, nx, ny) displacement trajectory. First frame is u0.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)

    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    # CFL check
    try:
        if not check_cfl_wave_2d(eqn.grid, eqn.c, dt):
            dx_min = float(jnp.min(eqn.grid.dx))
            dy_min = float(jnp.min(eqn.grid.dy))
            c_max = float(jnp.max(jnp.asarray(eqn.c)))
            cfl_limit = min(dx_min, dy_min) / (jnp.sqrt(2.0) * c_max)
            _logger.warning(
                f"dt={dt:.2e} exceeds 2D wave CFL limit {cfl_limit:.2e}. "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    from ..mesh.boundary import apply_boundary_conditions_2d
    from ..operators import laplacian_2d

    def laplacian_rhs(u, grid, t):
        """Compute c^2 * laplacian(u) + source, with BCs applied."""
        L_u, b_source = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, eqn.bc, u
        )
        rhs = eqn.c ** 2 * (L_u + b_source)
        if eqn.source is not None:
            rhs = rhs + eqn.source(grid.X.T, grid.Y.T, t)
        return rhs

    # First step: Taylor expansion
    t0_val = t0
    lap_u0 = laplacian_rhs(u0, eqn.grid, t0_val)
    u_curr = u0 + dt * v0 + 0.5 * dt ** 2 * lap_u0
    u_prev = u0

    def leapfrog_step(carry, step_idx):
        u_prev, u_curr = carry
        t_current = t0 + (step_idx + 1) * dt
        lap_u = laplacian_rhs(u_curr, eqn.grid, t_current)
        u_next = 2.0 * u_curr - u_prev + dt ** 2 * lap_u
        return (u_curr, u_next), u_next

    n_remaining = n_steps - 1
    if n_remaining > 0:
        (_, _), traj = jax.lax.scan(
            leapfrog_step, (u_prev, u_curr), jnp.arange(n_remaining)
        )
        trajectory = jnp.concatenate(
            [u0[jnp.newaxis, :, :], u_curr[jnp.newaxis, :, :], traj], axis=0
        )
    else:
        trajectory = jnp.stack([u0, u_curr], axis=0)

    return trajectory


# ---------------------------------------------------------------------------
# 3D Wave Equation
# ---------------------------------------------------------------------------

def solve_wave_3d(
    eqn: WaveEquation3D,
    u0: jnp.ndarray,
    v0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
    save_every: int = 1,
) -> jnp.ndarray:
    """Solve the 3D wave equation using the leapfrog scheme with jax.lax.scan.

    u_tt = c^2 * nabla^2 u + S(x, y, z, t)

    Second-order accurate, symplectic, and energy-conserving.

    **Memory note:** A 3D trajectory can be large.  Use ``save_every`` to
    reduce memory usage by storing only every N-th frame.

    Args:
        eqn: Wave equation problem definition.
        u0: (nx, ny, nz) initial displacement field at t=0.
        v0: (nx, ny, nz) initial velocity field du/dt at t=0.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.
        save_every: Save a frame every this many steps (default 1 = save all).

    Returns:
        (n_saved+1, nx, ny, nz) displacement trajectory. First frame is u0.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)

    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")
    if save_every < 1:
        raise ValueError(f"save_every must be >= 1, got {save_every}")

    # CFL check
    try:
        if not check_cfl_wave_3d(eqn.grid, eqn.c, dt):
            dx_min = float(jnp.min(eqn.grid.dx))
            dy_min = float(jnp.min(eqn.grid.dy))
            dz_min = float(jnp.min(eqn.grid.dz))
            c_max = float(jnp.max(jnp.asarray(eqn.c)))
            cfl_limit = min(dx_min, dy_min, dz_min) / (jnp.sqrt(3.0) * c_max)
            _logger.warning(
                f"dt={dt:.2e} exceeds 3D wave CFL limit {cfl_limit:.2e}. "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    from ..mesh.boundary import apply_boundary_conditions_3d
    from ..operators import laplacian_3d

    def laplacian_rhs(u, grid, t):
        """Compute c^2 * laplacian(u) + source, with BCs applied."""
        L_u, b_source = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, eqn.bc, u
        )
        rhs = eqn.c ** 2 * (L_u + b_source)
        if eqn.source is not None:
            rhs = rhs + eqn.source(grid.X, grid.Y, grid.Z, t)
        return rhs

    # First step: Taylor expansion
    t0_val = t0
    lap_u0 = laplacian_rhs(u0, eqn.grid, t0_val)
    u_curr = u0 + dt * v0 + 0.5 * dt ** 2 * lap_u0
    u_prev = u0

    # Inner leapfrog step for save_every sub-stepping
    def inner_leapfrog(carry, _):
        u_prev, u_curr, t = carry
        lap_u = laplacian_rhs(u_curr, eqn.grid, t)
        u_next = 2.0 * u_curr - u_prev + dt ** 2 * lap_u
        return (u_curr, u_next, t + dt), u_next

    n_outer = (n_steps - 1) // save_every

    def outer_step(carry, outer_idx):
        u_prev, u_curr = carry
        t_outer = t0 + (outer_idx * save_every + 1) * dt
        (u_prev_out, u_curr_out, _), _ = jax.lax.scan(
            inner_leapfrog, (u_prev, u_curr, t_outer), jnp.arange(save_every)
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
