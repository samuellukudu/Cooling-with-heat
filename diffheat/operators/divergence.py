# diffheat/operators/divergence.py
"""Divergence operator for 2D vector fields."""
import jax.numpy as jnp

from ..mesh.grid2d import Grid2D


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
