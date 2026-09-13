"""1D uniform grid for finite-difference discretization (DESIGN §4).

Mirrored from the frozen ``diffheat`` 1D mesh without importing it
(import-hygiene contract: ``harness`` never imports ``diffheat``).
Cell-centred finite volumes; fields live at cell centers.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Grid1D:
    """Uniform 1D grid.

    Attributes:
        x: (n_cells+1,) cell interface positions [0..length].
        centers: (n_cells,) cell center positions.
        dx: (n_cells,) cell widths (uniform = length / n_cells).
        length: total domain length.
        n_cells: number of cells N.
    """

    x: jnp.ndarray
    centers: jnp.ndarray
    dx: jnp.ndarray
    length: float
    n_cells: int

    @classmethod
    def uniform(cls, length: float, n_cells: int) -> "Grid1D":
        """Create a uniformly spaced 1D grid."""
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if n_cells < 2:
            raise ValueError(f"n_cells must be at least 2, got {n_cells}")
        cell_width = float(length) / int(n_cells)
        n = int(n_cells)
        x = jnp.linspace(0.0, float(length), n + 1, dtype=jnp.float64)
        centers = 0.5 * (x[:-1] + x[1:])
        dx = jnp.full((n,), cell_width, dtype=jnp.float64)
        return cls(x=x, centers=centers, dx=dx,
                   length=float(length), n_cells=n)

    @property
    def dx_min(self) -> float:
        return float(jnp.min(self.dx))

    def shape(self) -> tuple[int, ...]:
        """Field shape for this grid: ``(n_cells,)``."""
        return (self.n_cells,)


__all__ = ["Grid1D"]
