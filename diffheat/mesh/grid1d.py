# diffheat/mesh/grid1d.py
"""1D uniform grid for finite difference discretization."""
from dataclasses import dataclass

import jax.numpy as jnp

from ..utils import array


@dataclass(frozen=True)
class Grid1D:
    """Uniform 1D grid for finite difference discretization.

    Attributes:
        x: (N+1,) cell interface positions [x0, x1, ..., xN]
        centers: (N,) cell center positions
        dx: (N,) cell widths (uniform = length / n_cells for each cell)
        length: total domain length
        n_cells: number of cells N
    """
    x: jnp.ndarray
    centers: jnp.ndarray
    dx: jnp.ndarray
    length: float
    n_cells: int

    @classmethod
    def uniform(cls, length: float, n_cells: int) -> "Grid1D":
        """Create a uniformly spaced 1D grid.

        Args:
            length: Domain length (0 to length).
            n_cells: Number of interior cells.

        Returns:
            Grid1D with uniform spacing.
        """
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if n_cells < 2:
            raise ValueError(f"n_cells must be at least 2, got {n_cells}")

        cell_width = length / n_cells
        x = array(jnp.linspace(0.0, length, n_cells + 1))          # interfaces
        centers = array(0.5 * (x[:-1] + x[1:]))                     # cell centers
        dx = array(jnp.full(n_cells, cell_width))                   # cell widths

        return cls(x=x, centers=centers, dx=dx, length=length, n_cells=n_cells)
