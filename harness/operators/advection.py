"""Advection operators, first-order upwind (DESIGN §4).

``-(u·∇T)`` with per-direction upwinding. Mirrored from the frozen
``diffheat`` advection module without importing it. Takes harness
``Grid`` objects (``grid.dx`` etc.) so 1D/2D/3D share one calling
convention; scalar spacings also accepted for the 1D path.
"""

from __future__ import annotations

import jax.numpy as jnp


def _as_1d_spacing(dx, grid_dx=None):
    if dx is None and grid_dx is not None:
        return grid_dx
    return jnp.asarray(dx) if not hasattr(dx, "shape") else dx


def advection_1d(T: jnp.ndarray, u: jnp.ndarray, grid_or_dx) -> jnp.ndarray:
    """``-u·dT/dx`` upwind. ``T, u`` shape ``(N,)``."""
    dx = grid_or_dx.dx if hasattr(grid_or_dx, "dx") else grid_or_dx
    Tf = jnp.roll(T, -1)
    Tb = jnp.roll(T, 1)
    fwd = (Tf - T) / dx
    bwd = (T - Tb) / dx
    return -u * jnp.where(u > 0, bwd, fwd)


def advection_2d(
    T: jnp.ndarray,
    u_x: jnp.ndarray,
    u_y: jnp.ndarray,
    grid,
) -> jnp.ndarray:
    """``-(u_x·dT/dx + u_y·dT/dy)`` upwind. ``T`` shape ``(nx, ny)``."""
    dx = grid.dx[:, jnp.newaxis]
    dy = grid.dy[jnp.newaxis, :]
    adv_x = -u_x * jnp.where(
        u_x > 0,
        (T - jnp.roll(T, 1, axis=0)) / dx,
        (jnp.roll(T, -1, axis=0) - T) / dx,
    )
    adv_y = -u_y * jnp.where(
        u_y > 0,
        (T - jnp.roll(T, 1, axis=1)) / dy,
        (jnp.roll(T, -1, axis=1) - T) / dy,
    )
    return adv_x + adv_y


def advection_3d(
    T: jnp.ndarray,
    u_x: jnp.ndarray,
    u_y: jnp.ndarray,
    u_z: jnp.ndarray,
    grid,
) -> jnp.ndarray:
    """``-(u·∇T)`` upwind. ``T`` shape ``(nx, ny, nz)``."""
    dx = grid.dx[:, jnp.newaxis, jnp.newaxis]
    dy = grid.dy[jnp.newaxis, :, jnp.newaxis]
    dz = grid.dz[jnp.newaxis, jnp.newaxis, :]
    adv_x = -u_x * jnp.where(
        u_x > 0,
        (T - jnp.roll(T, 1, axis=0)) / dx,
        (jnp.roll(T, -1, axis=0) - T) / dx,
    )
    adv_y = -u_y * jnp.where(
        u_y > 0,
        (T - jnp.roll(T, 1, axis=1)) / dy,
        (jnp.roll(T, -1, axis=1) - T) / dy,
    )
    adv_z = -u_z * jnp.where(
        u_z > 0,
        (T - jnp.roll(T, 1, axis=2)) / dz,
        (jnp.roll(T, -1, axis=2) - T) / dz,
    )
    return adv_x + adv_y + adv_z


__all__ = ["advection_1d", "advection_2d", "advection_3d"]
