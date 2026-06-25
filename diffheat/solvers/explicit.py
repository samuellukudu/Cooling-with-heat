# diffheat/solvers/explicit.py
"""Explicit Euler time-stepping for 1D and 2D."""
import jax.numpy as jnp

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
