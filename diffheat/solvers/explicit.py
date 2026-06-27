# diffheat/solvers/explicit.py
"""Explicit Euler time-stepping for 1D, 2D, and 3D."""
import jax.numpy as jnp

from typing import Callable, Optional

from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D
from ..mesh.grid3d import Grid3D
from ..operators.laplacian import make_laplacian
from ..physics.heat1d import HeatEquation1D, apply_boundary_conditions


def explicit_euler_step_1d(
    state: jnp.ndarray,
    rhs_fn: Callable[[jnp.ndarray, Grid1D, float, Optional[dict]], jnp.ndarray],
    grid: Grid1D,
    t: float,
    dt: float,
    params: Optional[dict] = None,
) -> jnp.ndarray:
    """Single explicit Euler time step for an arbitrary 1D system.

    state^{n+1} = state^n + dt * rhs_fn(state^n, grid, t, params)

    Args:
        state: (N,) field at current timestep.
        rhs_fn: Right-hand side function.
            Signature: rhs_fn(state, grid, t, params) -> dstate_dt
        grid: The 1D grid.
        t: Current time.
        dt: Time step size.
        params: Optional dict of parameters passed to rhs_fn.

    Returns:
        (N,) field at next timestep.
    """
    dstate_dt = rhs_fn(state, grid, t, params)
    return state + dt * dstate_dt


def explicit_euler_step(
    T: jnp.ndarray,
    eqn: HeatEquation1D,
    t: float,
    dt: float,
) -> jnp.ndarray:
    """Single explicit Euler time step for the 1D heat equation.

    Convenience wrapper around ``explicit_euler_step_1d`` that constructs
    the heat-equation RHS from ``eqn``.  For non-heat-equation systems,
    use ``explicit_euler_step_1d`` directly with a custom ``rhs_fn``.
    """
    from ..mesh.boundary import apply_boundary_conditions_1d
    from ..operators.laplacian import laplacian_1d

    def rhs_fn(T, grid, t, params):
        L_T, b_source = apply_boundary_conditions_1d(
            lambda x: laplacian_1d(x, grid), grid, eqn.bc, T
        )
        dT_dt = eqn.alpha * (L_T + b_source)
        if eqn.source is not None:
            dT_dt = dT_dt + eqn.source(grid.centers, t)
        return dT_dt

    return explicit_euler_step_1d(T, rhs_fn, eqn.grid, t, dt)


def explicit_euler_step_2d(
    state: jnp.ndarray,
    rhs_fn: Callable[[jnp.ndarray, Grid2D, float, Optional[dict]], jnp.ndarray],
    grid: Grid2D,
    t: float,
    dt: float,
    params: Optional[dict] = None,
) -> jnp.ndarray:
    """Single explicit Euler time step for an arbitrary 2D system.

    state^{n+1} = state^n + dt * rhs_fn(state^n, grid, t, params)

    Args:
        state: (nx, ny) field at current timestep.
        rhs_fn: Right-hand side function.
            Signature: rhs_fn(state, grid, t, params) -> dstate_dt
        grid: The 2D grid.
        t: Current time.
        dt: Time step size.
        params: Optional dict of parameters passed to rhs_fn.

    Returns:
        (nx, ny) field at next timestep.
    """
    dstate_dt = rhs_fn(state, grid, t, params)
    return state + dt * dstate_dt


def explicit_euler_step_3d(
    state: jnp.ndarray,
    rhs_fn: Callable[[jnp.ndarray, Grid3D, float, Optional[dict]], jnp.ndarray],
    grid: Grid3D,
    t: float,
    dt: float,
    params: Optional[dict] = None,
) -> jnp.ndarray:
    """Single explicit Euler time step for an arbitrary 3D system.

    state^{n+1} = state^n + dt * rhs_fn(state^n, grid, t, params)

    Args:
        state: (nx, ny, nz) field at current timestep.
        rhs_fn: Right-hand side function.
            Signature: rhs_fn(state, grid, t, params) -> dstate_dt
        grid: The 3D grid.
        t: Current time.
        dt: Time step size.
        params: Optional dict of parameters passed to rhs_fn.

    Returns:
        (nx, ny, nz) field at next timestep.
    """
    dstate_dt = rhs_fn(state, grid, t, params)
    return state + dt * dstate_dt
