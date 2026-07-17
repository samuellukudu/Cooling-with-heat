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


# ---------------------------------------------------------------------------
# Wave equation CFL conditions
# ---------------------------------------------------------------------------

def check_cfl_wave_1d(grid: Grid1D, c: float | jnp.ndarray, dt: float) -> bool:
    """Check if dt satisfies the 1D wave equation CFL condition.

    For the leapfrog scheme on the 1D wave equation:

        c * dt <= dx_min

    Args:
        grid: The 1D spatial grid.
        c: Wave speed (scalar or field).
        dt: Time step size.

    Returns:
        True if stable, False otherwise.
    """
    c_max = float(jnp.max(jnp.asarray(c)))
    dx_min = float(jnp.min(grid.dx))
    return bool(c_max * dt <= dx_min)


def check_cfl_wave_2d(grid: Grid2D, c: float | jnp.ndarray, dt: float) -> bool:
    """Check if dt satisfies the 2D wave equation CFL condition.

    For the leapfrog scheme on the 2D wave equation:

        c * dt <= min(dx, dy) / sqrt(2)

    The sqrt(2) factor comes from the two spatial dimensions.

    Args:
        grid: The 2D spatial grid.
        c: Wave speed (scalar or field).
        dt: Time step size.

    Returns:
        True if stable, False otherwise.
    """
    c_max = float(jnp.max(jnp.asarray(c)))
    dx_min = float(jnp.min(grid.dx))
    dy_min = float(jnp.min(grid.dy))
    cfl_limit = min(dx_min, dy_min) / (jnp.sqrt(2.0) * c_max)
    return bool(dt <= cfl_limit)


def check_cfl_wave_3d(grid: Grid3D, c: float | jnp.ndarray, dt: float) -> bool:
    """Check if dt satisfies the 3D wave equation CFL condition.

    For the leapfrog scheme on the 3D wave equation:

        c * dt <= min(dx, dy, dz) / sqrt(3)

    The sqrt(3) factor comes from the three spatial dimensions.

    Args:
        grid: The 3D spatial grid.
        c: Wave speed (scalar or field).
        dt: Time step size.

    Returns:
        True if stable, False otherwise.
    """
    c_max = float(jnp.max(jnp.asarray(c)))
    dx_min = float(jnp.min(grid.dx))
    dy_min = float(jnp.min(grid.dy))
    dz_min = float(jnp.min(grid.dz))
    cfl_limit = min(dx_min, dy_min, dz_min) / (jnp.sqrt(3.0) * c_max)
    return bool(dt <= cfl_limit)


# ---------------------------------------------------------------------------
# Telegrapher equation CFL conditions
# ---------------------------------------------------------------------------

def check_cfl_telegrapher_1d(
    grid: Grid1D, alpha: float | jnp.ndarray, tau: float | jnp.ndarray, dt: float
) -> bool:
    """Check if dt satisfies the 1D Telegrapher equation CFL condition.

    Equivalent to wave CFL with speed c = sqrt(alpha / tau).
    """
    c = jnp.sqrt(jnp.asarray(alpha) / jnp.asarray(tau))
    return check_cfl_wave_1d(grid, c, dt)


def check_cfl_telegrapher_2d(
    grid: Grid2D, alpha: float | jnp.ndarray, tau: float | jnp.ndarray, dt: float
) -> bool:
    """Check if dt satisfies the 2D Telegrapher equation CFL condition.

    Equivalent to wave CFL with speed c = sqrt(alpha / tau).
    """
    c = jnp.sqrt(jnp.asarray(alpha) / jnp.asarray(tau))
    return check_cfl_wave_2d(grid, c, dt)


def check_cfl_telegrapher_3d(
    grid: Grid3D, alpha: float | jnp.ndarray, tau: float | jnp.ndarray, dt: float
) -> bool:
    """Check if dt satisfies the 3D Telegrapher equation CFL condition.

    Equivalent to wave CFL with speed c = sqrt(alpha / tau).
    """
    c = jnp.sqrt(jnp.asarray(alpha) / jnp.asarray(tau))
    return check_cfl_wave_3d(grid, c, dt)

