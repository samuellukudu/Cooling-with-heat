"""Divergence operators for vector fields (DESIGN §4)."""

from __future__ import annotations

import jax.numpy as jnp


def divergence_2d(u_x: jnp.ndarray, u_y: jnp.ndarray, grid) -> jnp.ndarray:
    """``∇·u = du_x/dx + du_y/dy`` centered. Fields shape ``(nx, ny)``."""
    dx = grid.dx[:, jnp.newaxis]
    dy = grid.dy[jnp.newaxis, :]
    dux = (jnp.roll(u_x, -1, axis=0) - jnp.roll(u_x, 1, axis=0)) / (2.0 * dx)
    duy = (jnp.roll(u_y, -1, axis=1) - jnp.roll(u_y, 1, axis=1)) / (2.0 * dy)
    return dux + duy


def divergence_3d(
    u_x: jnp.ndarray, u_y: jnp.ndarray, u_z: jnp.ndarray, grid
) -> jnp.ndarray:
    """``∇·u`` centered. Fields shape ``(nx, ny, nz)``."""
    dx = grid.dx[:, jnp.newaxis, jnp.newaxis]
    dy = grid.dy[jnp.newaxis, :, jnp.newaxis]
    dz = grid.dz[jnp.newaxis, jnp.newaxis, :]
    dux = (jnp.roll(u_x, -1, axis=0) - jnp.roll(u_x, 1, axis=0)) / (2.0 * dx)
    duy = (jnp.roll(u_y, -1, axis=1) - jnp.roll(u_y, 1, axis=1)) / (2.0 * dy)
    duz = (jnp.roll(u_z, -1, axis=2) - jnp.roll(u_z, 1, axis=2)) / (2.0 * dz)
    return dux + duy + duz


__all__ = ["divergence_2d", "divergence_3d"]
