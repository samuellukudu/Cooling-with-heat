# diffheat/solvers.py
"""Time integration solvers for the 1D heat equation."""
import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .mesh import Grid1D
from .physics import HeatEquation1D, apply_boundary_conditions, make_laplacian

_logger = logging.getLogger(__name__)


class Trajectory(NamedTuple):
    """Result of a heat equation solve.

    Attributes:
        temperature: (n_steps+1, N) array -- temperature at each timestep.
        t: (n_steps+1,) array -- time values.
        grid: The Grid1D used.
    """
    temperature: jnp.ndarray
    t: jnp.ndarray
    grid: Grid1D


def check_cfl(grid: Grid1D, alpha: float | jnp.ndarray, dt: float) -> bool:
    """Check if dt satisfies the CFL stability condition for explicit Euler.

    dt <= dx^2 / (2 * alpha)

    Args:
        grid: The spatial grid.
        alpha: Thermal diffusivity (scalar or field).
        dt: Time step size.

    Returns:
        True if stable, False otherwise.
    """
    alpha_max = float(jnp.max(jnp.asarray(alpha)))
    dx_min = float(jnp.min(grid.dx))
    cfl_limit = dx_min ** 2 / (2 * alpha_max)
    return bool(dt <= cfl_limit)


def explicit_euler_step(
    T: jnp.ndarray,
    eqn: HeatEquation1D,
    t: float,
    dt: float,
) -> jnp.ndarray:
    """Single explicit Euler time step for the 1D heat equation.

    T^{n+1} = T^n + dt * [alpha * L @ T^n + alpha * b_source + S(x, t)]

    Args:
        T: (N,) temperature at current timestep.
        eqn: Heat equation definition.
        t: Current time (for source term evaluation).
        dt: Time step size.

    Returns:
        (N,) temperature at next timestep.
    """
    L = make_laplacian(eqn.grid)
    L_mod, b_source = apply_boundary_conditions(L, eqn.grid, eqn.bc)

    dT_dt = eqn.alpha * (L_mod @ T + b_source)

    if eqn.source is not None:
        dT_dt = dT_dt + eqn.source(eqn.grid.centers, t)

    return T + dt * dT_dt


def solve_heat_1d(
    eqn: HeatEquation1D,
    T0: jnp.ndarray,
    t_span: tuple[float, float],
    dt: float,
) -> jnp.ndarray:
    """Solve the 1D heat equation using explicit Euler with jax.lax.scan.

    The entire solve is JIT-compiled and differentiable. Gradients flow
    through the full trajectory.

    Args:
        eqn: Heat equation problem definition.
        T0: (N,) initial temperature field.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.

    Returns:
        (n_steps+1, N) temperature trajectory. First row is T0.

    Raises:
        UserWarning: if dt violates the CFL stability condition.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)

    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    # CFL check — skip during gradient tracing to avoid concretization errors
    try:
        if not check_cfl(eqn.grid, eqn.alpha, dt):
            cfl_limit = float(jnp.min(eqn.grid.dx)) ** 2 / (2 * float(jnp.max(jnp.asarray(eqn.alpha))))
            _logger.warning(
                f"dt={dt:.2e} exceeds CFL limit {cfl_limit:.2e}. "
                f"Solution may be unstable."
            )
    except jax.errors.ConcretizationTypeError:
        pass

    # Pre-compute time array
    t = t0 + dt * jnp.arange(n_steps + 1)

    def step_fn(T, step_idx):
        t_current = t0 + step_idx * dt
        T_next = explicit_euler_step(T, eqn, t_current, dt)
        return T_next, T_next

    # scan over n_steps, prepend T0
    _, T_traj = jax.lax.scan(step_fn, T0, jnp.arange(n_steps))
    trajectory = jnp.concatenate([T0[jnp.newaxis, :], T_traj], axis=0)

    return trajectory
