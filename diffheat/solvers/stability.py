# diffheat/solvers/stability.py
"""CFL stability conditions for explicit time integration."""
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D


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


def check_cfl_2d(grid: Grid2D, alpha: float | jnp.ndarray, dt: float) -> bool:
    """Check if dt satisfies the 2D CFL stability condition for explicit Euler.

    dt <= min(dx^2, dy^2) / (4 * alpha)

    Args:
        grid: The 2D spatial grid.
        alpha: Thermal diffusivity (scalar or field).
        dt: Time step size.

    Returns:
        True if stable, False otherwise.
    """
    alpha_max = float(jnp.max(jnp.asarray(alpha)))
    dx_min = float(jnp.min(grid.dx))
    dy_min = float(jnp.min(grid.dy))
    cfl_limit = min(dx_min * dx_min, dy_min * dy_min) / (4.0 * alpha_max)
    return bool(dt <= cfl_limit)
