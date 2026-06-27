# diffheat/operators/divergence.py
"""Divergence operators for 2D and 3D vector fields."""
import jax.numpy as jnp

from ..mesh.grid2d import Grid2D
from ..mesh.grid3d import Grid3D


def divergence_2d(ux: jnp.ndarray, uy: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """Compute divergence = dux/dx + duy/dy.

    Uses centered finite differences at interior cells and one-sided differences
    at boundary cells.

    Args:
        ux: (nx, ny) x-component of the vector field at cell centers.
        uy: (nx, ny) y-component of the vector field at cell centers.
        grid: The 2D grid.

    Returns:
        (nx, ny) divergence at cell centers.
    """
    dx = grid.dx[:, jnp.newaxis]  # (nx, 1)
    dy = grid.dy[jnp.newaxis, :]  # (1, ny)

    dux_dx = (jnp.roll(ux, -1, axis=0) - jnp.roll(ux, 1, axis=0)) / (2.0 * dx)
    dux_dx = dux_dx.at[0, :].set((ux[1, :] - ux[0, :]) / dx[0, 0])
    dux_dx = dux_dx.at[-1, :].set((ux[-1, :] - ux[-2, :]) / dx[-1, 0])

    duy_dy = (jnp.roll(uy, -1, axis=1) - jnp.roll(uy, 1, axis=1)) / (2.0 * dy)
    duy_dy = duy_dy.at[:, 0].set((uy[:, 1] - uy[:, 0]) / dy[0, 0])
    duy_dy = duy_dy.at[:, -1].set((uy[:, -1] - uy[:, -2]) / dy[0, -1])

    return dux_dx + duy_dy


def divergence_3d(
    ux: jnp.ndarray,
    uy: jnp.ndarray,
    uz: jnp.ndarray,
    grid: Grid3D,
) -> jnp.ndarray:
    """Compute divergence = dux/dx + duy/dy + duz/dz on a 3D grid.

    Uses centered finite differences at interior cells and one-sided differences
    at boundary slices (6 faces total).

    Args:
        ux: (nx, ny, nz) x-component of the vector field at cell centers.
        uy: (nx, ny, nz) y-component of the vector field at cell centers.
        uz: (nx, ny, nz) z-component of the vector field at cell centers.
        grid: The 3D grid.

    Returns:
        (nx, ny, nz) divergence at cell centers.
    """
    dx = grid.dx[:, jnp.newaxis, jnp.newaxis]  # (nx, 1, 1)
    dy = grid.dy[jnp.newaxis, :, jnp.newaxis]  # (1, ny, 1)
    dz = grid.dz[jnp.newaxis, jnp.newaxis, :]  # (1, 1, nz)

    # dux/dx  (axis 0)
    dux_dx = (jnp.roll(ux, -1, axis=0) - jnp.roll(ux, 1, axis=0)) / (2.0 * dx)
    dux_dx = dux_dx.at[0, :, :].set((ux[1, :, :] - ux[0, :, :]) / dx[0, 0, 0])
    dux_dx = dux_dx.at[-1, :, :].set((ux[-1, :, :] - ux[-2, :, :]) / dx[-1, 0, 0])

    # duy/dy  (axis 1)
    duy_dy = (jnp.roll(uy, -1, axis=1) - jnp.roll(uy, 1, axis=1)) / (2.0 * dy)
    duy_dy = duy_dy.at[:, 0, :].set((uy[:, 1, :] - uy[:, 0, :]) / dy[0, 0, 0])
    duy_dy = duy_dy.at[:, -1, :].set((uy[:, -1, :] - uy[:, -2, :]) / dy[0, -1, 0])

    # duz/dz  (axis 2)
    duz_dz = (jnp.roll(uz, -1, axis=2) - jnp.roll(uz, 1, axis=2)) / (2.0 * dz)
    duz_dz = duz_dz.at[:, :, 0].set((uz[:, :, 1] - uz[:, :, 0]) / dz[0, 0, 0])
    duz_dz = duz_dz.at[:, :, -1].set((uz[:, :, -1] - uz[:, :, -2]) / dz[0, 0, -1])

    return dux_dx + duy_dy + duz_dz
