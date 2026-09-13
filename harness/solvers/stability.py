"""CFL stability conditions for explicit time integration (DESIGN §4.2).

Mirrored from the frozen ``diffheat`` stability module without importing
it, but over :mod:`harness.mesh` grids. All helpers are plain-Python
``bool``/``float`` (fail loudly before tracing, never inside ``jit``).
"""

from __future__ import annotations

import jax.numpy as jnp


def _alpha_max(alpha) -> float:
    return float(jnp.max(jnp.asarray(alpha)))


def max_timestep_1d(grid, alpha) -> float:
    """``dt ≤ dx²/(2α)``."""
    a = _alpha_max(alpha)
    if a <= 0:
        return float("inf")
    dx_min = float(jnp.min(grid.dx))
    return dx_min * dx_min / (2.0 * a)


def check_cfl_1d(grid, alpha, dt: float) -> bool:
    """True when ``dt`` satisfies the 1D diffusion CFL."""
    return bool(float(dt) <= max_timestep_1d(grid, alpha))


def max_timestep_2d(grid, alpha) -> float:
    """``dt ≤ min(dx²,dy²)/(4α)``."""
    a = _alpha_max(alpha)
    if a <= 0:
        return float("inf")
    dx_min = float(jnp.min(grid.dx))
    dy_min = float(jnp.min(grid.dy))
    return min(dx_min * dx_min, dy_min * dy_min) / (4.0 * a)


def check_cfl_2d(grid, alpha, dt: float) -> bool:
    """True when ``dt`` satisfies the 2D diffusion CFL."""
    return bool(float(dt) <= max_timestep_2d(grid, alpha))


def max_timestep_3d(grid, alpha) -> float:
    """``dt ≤ min(dx²,dy²,dz²)/(6α)``."""
    a = _alpha_max(alpha)
    if a <= 0:
        return float("inf")
    dx_min = float(jnp.min(grid.dx))
    dy_min = float(jnp.min(grid.dy))
    dz_min = float(jnp.min(grid.dz))
    return min(dx_min**2, dy_min**2, dz_min**2) / (6.0 * a)


def check_cfl_3d(grid, alpha, dt: float) -> bool:
    """True when ``dt`` satisfies the 3D diffusion CFL."""
    return bool(float(dt) <= max_timestep_3d(grid, alpha))


def max_timestep_advection_diffusion_2d(grid, alpha, u_max: float) -> float:
    """Combined 2D advection-diffusion limit.

    ``min(min(dx²,dy²)/(4α), dx/u_max)`` — the advective branch uses the
    x-spacing (channel-flow convention of ``forced_conv``).
    """
    a = _alpha_max(alpha)
    dx = float(jnp.min(grid.dx))
    dy = float(jnp.min(grid.dy))
    dt_diff = min(dx * dx, dy * dy) / (4.0 * a) if a > 0 else float("inf")
    dt_adv = dx / float(u_max) if float(u_max) > 0 else float("inf")
    return min(dt_diff, dt_adv)


def check_cfl(grid, alpha, dt: float, *, spatial_dim: int, _grid=None) -> bool:
    """Dim-dispatched CFL check (``spatial_dim`` ∈ {1, 2, 3})."""
    if _grid is None:
        raise ValueError("check_cfl needs _grid=<Grid> (dim-dispatched)")
    if spatial_dim == 1:
        return check_cfl_1d(_grid, alpha, dt)
    if spatial_dim == 2:
        return check_cfl_2d(_grid, alpha, dt)
    if spatial_dim == 3:
        return check_cfl_3d(_grid, alpha, dt)
    raise ValueError(f"spatial_dim must be 1/2/3, got {spatial_dim!r}")


__all__ = [
    "max_timestep_1d",
    "check_cfl_1d",
    "max_timestep_2d",
    "check_cfl_2d",
    "max_timestep_3d",
    "check_cfl_3d",
    "max_timestep_advection_diffusion_2d",
    "check_cfl",
]
