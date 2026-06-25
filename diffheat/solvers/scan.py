# diffheat/solvers/scan.py
"""Scan-based trajectory solvers for 1D and 2D."""
import logging

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D
from ..physics.heat1d import HeatEquation1D
from .explicit import explicit_euler_step, explicit_euler_step_2d
from .stability import check_cfl

_logger = logging.getLogger(__name__)


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


def solve_2d(
    rhs_fn: Callable,
    initial_state: jnp.ndarray,
    grid: Grid2D,
    t_span: tuple[float, float],
    dt: float,
    params: Optional[dict] = None,
) -> jnp.ndarray:
    """Solve a 2D PDE using explicit Euler with jax.lax.scan.

    The entire solve is JIT-compiled and differentiable.

    .. note::

        This solver is equation-agnostic and does not perform automatic CFL
        checks.  For diffusive systems use ``check_cfl_2d(grid, alpha, dt)``
        before calling this function.

    Args:
        rhs_fn: Right-hand side function.
            Signature: rhs_fn(state, grid, t, params) -> dstate_dt
        initial_state: (nx, ny) initial field.
        grid: The 2D grid.
        t_span: (t_start, t_end) simulation time range.
        dt: Time step size.
        params: Optional dict of parameters passed to rhs_fn.

    Returns:
        (n_steps+1, nx, ny) field trajectory. First frame is initial_state.
    """
    t0, t_end = t_span
    n_steps = int((t_end - t0) / dt)

    if n_steps < 1:
        raise ValueError(f"t_span too short for dt={dt}: {t_span}")

    def step_fn(state, step_idx):
        t_current = t0 + step_idx * dt
        state_next = explicit_euler_step_2d(state, rhs_fn, grid, t_current, dt, params)
        return state_next, state_next

    _, traj = jax.lax.scan(step_fn, initial_state, jnp.arange(n_steps))
    trajectory = jnp.concatenate([initial_state[jnp.newaxis, :, :], traj], axis=0)

    return trajectory
