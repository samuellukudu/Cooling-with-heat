"""Discrete Laplacians for 1D/2D/3D (DESIGN §4).

Centered finite differences via ``jnp.roll`` (implicitly periodic —
correct with ``harness.mesh`` boundary helpers). Mirrored from the
frozen ``diffheat`` operators without importing them.
"""

from __future__ import annotations

import jax.numpy as jnp


def laplacian_1d(T: jnp.ndarray, grid) -> jnp.ndarray:
    """1D Laplacian d²T/dx². ``T`` shape ``(n_cells,)``."""
    dx2 = grid.dx * grid.dx
    return (jnp.roll(T, -1) + jnp.roll(T, 1) - 2.0 * T) / dx2


def laplacian_2d(T: jnp.ndarray, grid) -> jnp.ndarray:
    """2D Laplacian. ``T`` shape ``(nx, ny)`` (axis 0 = x)."""
    dx2 = grid.dx * grid.dx
    dy2 = grid.dy * grid.dy
    d2x = (jnp.roll(T, -1, axis=0) + jnp.roll(T, 1, axis=0) - 2.0 * T) / dx2[:, jnp.newaxis]
    d2y = (jnp.roll(T, -1, axis=1) + jnp.roll(T, 1, axis=1) - 2.0 * T) / dy2[jnp.newaxis, :]
    return d2x + d2y


def laplacian_3d(T: jnp.ndarray, grid) -> jnp.ndarray:
    """3D Laplacian. ``T`` shape ``(nx, ny, nz)``."""
    dx2 = grid.dx * grid.dx
    dy2 = grid.dy * grid.dy
    dz2 = grid.dz * grid.dz
    d2x = (jnp.roll(T, -1, axis=0) + jnp.roll(T, 1, axis=0) - 2.0 * T) / dx2[:, jnp.newaxis, jnp.newaxis]
    d2y = (jnp.roll(T, -1, axis=1) + jnp.roll(T, 1, axis=1) - 2.0 * T) / dy2[jnp.newaxis, :, jnp.newaxis]
    d2z = (jnp.roll(T, -1, axis=2) + jnp.roll(T, 1, axis=2) - 2.0 * T) / dz2[jnp.newaxis, jnp.newaxis, :]
    return d2x + d2y + d2z


__all__ = ["laplacian_1d", "laplacian_2d", "laplacian_3d"]
