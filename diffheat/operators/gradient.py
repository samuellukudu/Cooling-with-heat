# diffheat/operators/gradient.py
"""Gradient operators for 1D and 2D scalar fields."""
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D


def gradient_1d(T: jnp.ndarray, grid: Grid1D) -> jnp.ndarray:
    """Compute the 1D gradient dT/dx using centered finite differences.

    Uses one-sided differences at boundary cells (forward at left, backward at right)
    and centered differences in the interior.

    Args:
        T: (N,) field at cell centers.
        grid: The 1D grid.

    Returns:
        (N,) gradient at cell centers.
    """
    dx = grid.dx
    dT_dx = (jnp.roll(T, -1) - jnp.roll(T, 1)) / (2.0 * dx)
    dT_dx = dT_dx.at[0].set((T[1] - T[0]) / dx[0])
    dT_dx = dT_dx.at[-1].set((T[-1] - T[-2]) / dx[-1])
    return dT_dx


def gradient_x(T: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """Compute dT/dx using centered finite differences.

    Uses (T[i+1,j] - T[i-1,j]) / (2*dx) at interior cells.
    Boundary cells use one-sided differences (forward at left, backward at right).

    Args:
        T: (nx, ny) field at cell centers.
        grid: The 2D grid.

    Returns:
        (nx, ny) x-gradient at cell centers.
    """
    dx = grid.dx[:, jnp.newaxis]  # (nx, 1)
    # Centered difference interior, one-sided at boundaries
    gx = (jnp.roll(T, -1, axis=0) - jnp.roll(T, 1, axis=0)) / (2.0 * dx)
    gx = gx.at[0, :].set((T[1, :] - T[0, :]) / dx[0, 0])
    gx = gx.at[-1, :].set((T[-1, :] - T[-2, :]) / dx[-1, 0])
    return gx


def gradient_y(T: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """Compute dT/dy using centered finite differences.

    Uses (T[i,j+1] - T[i,j-1]) / (2*dy) at interior cells.
    Boundary cells use one-sided differences.

    Args:
        T: (nx, ny) field at cell centers.
        grid: The 2D grid.

    Returns:
        (nx, ny) y-gradient at cell centers.
    """
    dy = grid.dy[jnp.newaxis, :]  # (1, ny)
    gy = (jnp.roll(T, -1, axis=1) - jnp.roll(T, 1, axis=1)) / (2.0 * dy)
    gy = gy.at[:, 0].set((T[:, 1] - T[:, 0]) / dy[0, 0])
    gy = gy.at[:, -1].set((T[:, -1] - T[:, -2]) / dy[0, -1])
    return gy



def gradient_2d(T: jnp.ndarray, grid: Grid2D) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the full 2D gradient (dT/dx, dT/dy).

    Args:
        T: (nx, ny) field at cell centers.
        grid: The 2D grid.

    Returns:
        (dT_dx, dT_dy) each with shape (nx, ny).
    """
    return gradient_x(T, grid), gradient_y(T, grid)
