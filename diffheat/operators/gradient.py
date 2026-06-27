# diffheat/operators/gradient.py
"""Gradient operators for 1D, 2D, and 3D scalar fields."""
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D
from ..mesh.grid3d import Grid3D


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


# ---------------------------------------------------------------------------
# 3D gradient operators
# ---------------------------------------------------------------------------

def gradient_x3d(T: jnp.ndarray, grid: Grid3D) -> jnp.ndarray:
    """Compute dT/dx on a 3D grid using centered finite differences.

    Interior: ``(T[i+1,j,k] - T[i-1,j,k]) / (2*dx)``.
    Boundary slices (i=0 and i=nx-1) use one-sided differences.

    Args:
        T: (nx, ny, nz) field at cell centers.
        grid: The 3D grid.

    Returns:
        (nx, ny, nz) x-gradient at cell centers.
    """
    dx = grid.dx[:, jnp.newaxis, jnp.newaxis]  # (nx, 1, 1)
    gx = (jnp.roll(T, -1, axis=0) - jnp.roll(T, 1, axis=0)) / (2.0 * dx)
    gx = gx.at[0, :, :].set((T[1, :, :] - T[0, :, :]) / dx[0, 0, 0])
    gx = gx.at[-1, :, :].set((T[-1, :, :] - T[-2, :, :]) / dx[-1, 0, 0])
    return gx


def gradient_y3d(T: jnp.ndarray, grid: Grid3D) -> jnp.ndarray:
    """Compute dT/dy on a 3D grid using centered finite differences.

    Interior: ``(T[i,j+1,k] - T[i,j-1,k]) / (2*dy)``.
    Boundary slices (j=0 and j=ny-1) use one-sided differences.

    Args:
        T: (nx, ny, nz) field at cell centers.
        grid: The 3D grid.

    Returns:
        (nx, ny, nz) y-gradient at cell centers.
    """
    dy = grid.dy[jnp.newaxis, :, jnp.newaxis]  # (1, ny, 1)
    gy = (jnp.roll(T, -1, axis=1) - jnp.roll(T, 1, axis=1)) / (2.0 * dy)
    gy = gy.at[:, 0, :].set((T[:, 1, :] - T[:, 0, :]) / dy[0, 0, 0])
    gy = gy.at[:, -1, :].set((T[:, -1, :] - T[:, -2, :]) / dy[0, -1, 0])
    return gy


def gradient_z3d(T: jnp.ndarray, grid: Grid3D) -> jnp.ndarray:
    """Compute dT/dz on a 3D grid using centered finite differences.

    Interior: ``(T[i,j,k+1] - T[i,j,k-1]) / (2*dz)``.
    Boundary slices (k=0 and k=nz-1) use one-sided differences.

    Args:
        T: (nx, ny, nz) field at cell centers.
        grid: The 3D grid.

    Returns:
        (nx, ny, nz) z-gradient at cell centers.
    """
    dz = grid.dz[jnp.newaxis, jnp.newaxis, :]  # (1, 1, nz)
    gz = (jnp.roll(T, -1, axis=2) - jnp.roll(T, 1, axis=2)) / (2.0 * dz)
    gz = gz.at[:, :, 0].set((T[:, :, 1] - T[:, :, 0]) / dz[0, 0, 0])
    gz = gz.at[:, :, -1].set((T[:, :, -1] - T[:, :, -2]) / dz[0, 0, -1])
    return gz


def gradient_3d(
    T: jnp.ndarray, grid: Grid3D
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the full 3D gradient (dT/dx, dT/dy, dT/dz).

    Args:
        T: (nx, ny, nz) field at cell centers.
        grid: The 3D grid.

    Returns:
        (dT_dx, dT_dy, dT_dz) each with shape (nx, ny, nz).
    """
    return gradient_x3d(T, grid), gradient_y3d(T, grid), gradient_z3d(T, grid)
