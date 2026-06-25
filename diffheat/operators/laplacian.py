# diffheat/operators/laplacian.py
"""Discrete Laplacian operators for 1D and 2D."""
import jax.numpy as jnp

from ..mesh.grid1d import Grid1D
from ..utils import array


def make_laplacian(grid: Grid1D) -> jnp.ndarray:
    """Build the (N, N) tridiagonal Laplacian matrix using centered finite differences.

    Interior: [1, -2, 1] / dx^2
    Boundaries: left unmodified (apply_boundary_conditions handles them).

    Args:
        grid: The 1D grid.

    Returns:
        (N, N) Laplacian matrix.
    """
    n = grid.n_cells
    dx = grid.dx

    # Diagonal: -2 / dx^2
    diag = array(-2.0 / (dx * dx))

    # Off-diagonals: 1 / dx^2
    # NOTE: this formula is valid only for uniform grids.
    # Non-uniform grids require a different finite-difference stencil.
    dx2_left = dx[:-1] * dx[:-1]
    dx2_right = dx[1:] * dx[1:]
    off_diag = array(2.0 / (dx2_left + dx2_right))

    L = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)
    return L
