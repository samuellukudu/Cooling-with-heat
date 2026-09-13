"""Scan-based trajectory solvers for 1D/2D/3D (DESIGN §4).

Generic explicit-Euler rollouts over :mod:`harness.mesh` grids. The whole
solve is one ``jax.lax.scan`` — jittable and end-to-end differentiable.
Equation-specific wrappers (``solve_heat_1d/2d/3d``) live in
:mod:`harness.physics.heat1d` etc.; this module holds only the
equation-agnostic scan machinery plus the dim-dispatched entry point.

Trajectory contract (the 4D rule): time-first ``(n_steps+1, *spatial)``
with frame 0 the initial condition. ``save_every`` (3D default path)
stores every N-th frame plus the initial condition.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from .time import n_steps_for, save_every_outer_steps

_logger = logging.getLogger(__name__)


def _explicit_step(state, rhs_fn, grid, t, dt, params):
    return state + dt * rhs_fn(state, grid, t, params)


def solve_1d(
    rhs_fn: Callable,
    initial_state: jnp.ndarray,
    grid,
    t_span: tuple[float, float],
    dt: float,
    params: Optional[dict] = None,
) -> jnp.ndarray:
    """Solve a 1D PDE (explicit Euler in scan).

    Args:
        rhs_fn: ``rhs_fn(state, grid, t, params) -> dstate_dt``.
        initial_state: ``(n_cells,)`` field.
        grid: :class:`harness.mesh.Grid1D`.
        t_span: ``(t0, t1)``. dt: step size.

    Returns:
        ``(n_steps+1, n_cells)`` trajectory, frame 0 = initial.
    """
    t0, _ = t_span
    n_steps = n_steps_for(t_span, dt)

    def step_fn(state, k):
        t = t0 + k * dt
        nxt = _explicit_step(state, rhs_fn, grid, t, dt, params)
        return nxt, nxt

    _, traj = jax.lax.scan(step_fn, initial_state, jnp.arange(n_steps))
    return jnp.concatenate([initial_state[jnp.newaxis, :], traj], axis=0)


def solve_2d(
    rhs_fn: Callable,
    initial_state: jnp.ndarray,
    grid,
    t_span: tuple[float, float],
    dt: float,
    params: Optional[dict] = None,
) -> jnp.ndarray:
    """Solve a 2D PDE (explicit Euler in scan).

    Returns ``(n_steps+1, nx, ny)`` trajectory.
    """
    t0, _ = t_span
    n_steps = n_steps_for(t_span, dt)

    def step_fn(state, k):
        t = t0 + k * dt
        nxt = _explicit_step(state, rhs_fn, grid, t, dt, params)
        return nxt, nxt

    _, traj = jax.lax.scan(step_fn, initial_state, jnp.arange(n_steps))
    return jnp.concatenate([initial_state[jnp.newaxis, :, :], traj], axis=0)


def solve_3d(
    rhs_fn: Callable,
    initial_state: jnp.ndarray,
    grid,
    t_span: tuple[float, float],
    dt: float,
    params: Optional[dict] = None,
    save_every: int = 1,
) -> jnp.ndarray:
    """Solve a 3D PDE (explicit Euler in scan, ``save_every`` aware).

    Returns ``(n_saved+1, nx, ny, nz)`` with ``n_saved = n_steps//save_every``.
    """
    t0, _ = t_span
    n_steps = n_steps_for(t_span, dt)
    n_outer = save_every_outer_steps(n_steps, save_every)

    def outer_step(state, outer_idx):
        def inner_step(s, inner_idx):
            t = t0 + (outer_idx * save_every + inner_idx) * dt
            return _explicit_step(s, rhs_fn, grid, t, dt, params), None

        state_out, _ = jax.lax.scan(inner_step, state, jnp.arange(save_every))
        return state_out, state_out

    _, traj = jax.lax.scan(outer_step, initial_state, jnp.arange(n_outer))
    return jnp.concatenate(
        [initial_state[jnp.newaxis, :, :, :], traj], axis=0
    )


def solve_dim(
    spatial_dim: int,
    rhs_fn: Callable,
    initial_state: jnp.ndarray,
    grid,
    t_span: tuple[float, float],
    dt: float,
    params: Optional[dict] = None,
    save_every: int = 1,
) -> jnp.ndarray:
    """Dim-dispatched rollout (``spatial_dim`` ∈ {1, 2, 3})."""
    if spatial_dim == 1:
        return solve_1d(rhs_fn, initial_state, grid, t_span, dt, params)
    if spatial_dim == 2:
        return solve_2d(rhs_fn, initial_state, grid, t_span, dt, params)
    if spatial_dim == 3:
        return solve_3d(rhs_fn, initial_state, grid, t_span, dt, params,
                        save_every=save_every)
    raise ValueError(f"spatial_dim must be 1/2/3, got {spatial_dim!r}")


__all__ = ["solve_1d", "solve_2d", "solve_3d", "solve_dim"]
