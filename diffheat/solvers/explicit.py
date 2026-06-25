# diffheat/solvers/explicit.py
"""Explicit Euler time-stepping for 1D and 2D."""
import jax.numpy as jnp

from typing import Callable, Optional

from ..mesh.grid2d import Grid2D
from ..operators.laplacian import make_laplacian
from ..physics.heat1d import HeatEquation1D, apply_boundary_conditions


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
