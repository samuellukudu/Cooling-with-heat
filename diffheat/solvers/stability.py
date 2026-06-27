# diffheat/solvers/stability.py
"""CFL stability conditions for explicit time integration."""
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D
from ..mesh.grid3d import Grid3D


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


def check_cfl_3d(grid: Grid3D, alpha: float | jnp.ndarray, dt: float) -> bool:
    """Check if dt satisfies the 3D CFL stability condition for explicit Euler.

    From the 3D von Neumann stability analysis for the explicit diffusion equation:

        dt <= min(dx^2, dy^2, dz^2) / (6 * alpha)

    The factor of 6 comes from three spatial dimensions each contributing 2.

    Args:
        grid: The 3D spatial grid.
        alpha: Thermal diffusivity (scalar or field).
        dt: Time step size.

    Returns:
        True if stable, False otherwise.
    """
    alpha_max = float(jnp.max(jnp.asarray(alpha)))
    dx_min = float(jnp.min(grid.dx))
    dy_min = float(jnp.min(grid.dy))
    dz_min = float(jnp.min(grid.dz))
    cfl_limit = min(dx_min ** 2, dy_min ** 2, dz_min ** 2) / (6.0 * alpha_max)
    return bool(dt <= cfl_limit)
