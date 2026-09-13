"""Gradient operators for 1D/2D/3D scalar fields (DESIGN §4).

Centered interior, one-sided at domain faces. Mirrored from the frozen
``diffheat`` gradient module without importing it.
"""

from __future__ import annotations

import jax.numpy as jnp


def gradient_1d(T: jnp.ndarray, grid) -> jnp.ndarray:
    """1D gradient dT/dx. ``T`` shape ``(n_cells,)``."""
    dx = grid.dx
    g = (jnp.roll(T, -1) - jnp.roll(T, 1)) / (2.0 * dx)
    g = g.at[0].set((T[1] - T[0]) / dx[0])
    g = g.at[-1].set((T[-1] - T[-2]) / dx[-1])
    return g


def gradient_x(T: jnp.ndarray, grid) -> jnp.ndarray:
    """dT/dx on a 2D grid. ``T`` shape ``(nx, ny)``."""
    dx = grid.dx[:, jnp.newaxis]
    gx = (jnp.roll(T, -1, axis=0) - jnp.roll(T, 1, axis=0)) / (2.0 * dx)
    gx = gx.at[0, :].set((T[1, :] - T[0, :]) / dx[0, 0])
    gx = gx.at[-1, :].set((T[-1, :] - T[-2, :]) / dx[-1, 0])
    return gx


def gradient_y(T: jnp.ndarray, grid) -> jnp.ndarray:
    """dT/dy on a 2D grid. ``T`` shape ``(nx, ny)``."""
    dy = grid.dy[jnp.newaxis, :]
    gy = (jnp.roll(T, -1, axis=1) - jnp.roll(T, 1, axis=1)) / (2.0 * dy)
    gy = gy.at[:, 0].set((T[:, 1] - T[:, 0]) / dy[0, 0])
    gy = gy.at[:, -1].set((T[:, -1] - T[:, -2]) / dy[0, -1])
    return gy


def gradient_2d(T: jnp.ndarray, grid) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Full 2D gradient ``(dT/dx, dT/dy)``."""
    return gradient_x(T, grid), gradient_y(T, grid)


def gradient_x3d(T: jnp.ndarray, grid) -> jnp.ndarray:
    """dT/dx on a 3D grid. ``T`` shape ``(nx, ny, nz)``."""
    dx = grid.dx[:, jnp.newaxis, jnp.newaxis]
    gx = (jnp.roll(T, -1, axis=0) - jnp.roll(T, 1, axis=0)) / (2.0 * dx)
    gx = gx.at[0, :, :].set((T[1, :, :] - T[0, :, :]) / dx[0, 0, 0])
    gx = gx.at[-1, :, :].set((T[-1, :, :] - T[-2, :, :]) / dx[-1, 0, 0])
    return gx


def gradient_y3d(T: jnp.ndarray, grid) -> jnp.ndarray:
    """dT/dy on a 3D grid."""
    dy = grid.dy[jnp.newaxis, :, jnp.newaxis]
    gy = (jnp.roll(T, -1, axis=1) - jnp.roll(T, 1, axis=1)) / (2.0 * dy)
    gy = gy.at[:, 0, :].set((T[:, 1, :] - T[:, 0, :]) / dy[0, 0, 0])
    gy = gy.at[:, -1, :].set((T[:, -1, :] - T[:, -2, :]) / dy[0, -1, 0])
    return gy


def gradient_z3d(T: jnp.ndarray, grid) -> jnp.ndarray:
    """dT/dz on a 3D grid."""
    dz = grid.dz[jnp.newaxis, jnp.newaxis, :]
    gz = (jnp.roll(T, -1, axis=2) - jnp.roll(T, 1, axis=2)) / (2.0 * dz)
    gz = gz.at[:, :, 0].set((T[:, :, 1] - T[:, :, 0]) / dz[0, 0, 0])
    gz = gz.at[:, :, -1].set((T[:, :, -1] - T[:, :, -2]) / dz[0, 0, -1])
    return gz


def gradient_3d(T, grid) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full 3D gradient ``(dT/dx, dT/dy, dT/dz)``."""
    return gradient_x3d(T, grid), gradient_y3d(T, grid), gradient_z3d(T, grid)


__all__ = [
    "gradient_1d",
    "gradient_x",
    "gradient_y",
    "gradient_2d",
    "gradient_x3d",
    "gradient_y3d",
    "gradient_z3d",
    "gradient_3d",
]
