# diffheat/physics/wave.py
"""1D, 2D, and 3D wave equation problem definitions.

The wave equation in d spatial dimensions:

    u_tt = c^2 * nabla^2 u + S(x, t)

where c is the wave speed and S is an optional source term.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition, BoundaryCondition2D, BoundaryCondition3D
from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D
from ..mesh.grid3d import Grid3D
from ..utils import array


@dataclass(frozen=True)
class WaveEquation1D:
    """Complete 1D wave equation problem definition.

    u_tt = c^2 * u_xx + S(x, t)

    Args:
        grid: 1D spatial grid.
        bc: Boundary conditions (Dirichlet or Neumann).
        c: Wave speed. Scalar or (n_cells,) field.
        source: Optional source term S(x, t). Called as source(x_coords, t).
    """
    grid: Grid1D
    bc: BoundaryCondition
    c: float | jnp.ndarray
    source: Optional[Callable[[jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.c, (int, float)):
            if self.c <= 0:
                raise ValueError(f"c must be positive, got {self.c}")
            object.__setattr__(self, "c", array(float(self.c)))


@dataclass(frozen=True)
class WaveEquation2D:
    """Complete 2D wave equation problem definition.

    u_tt = c^2 * (u_xx + u_yy) + S(x, y, t)

    Args:
        grid: 2D spatial grid.
        bc: Boundary conditions.
        c: Wave speed. Scalar or (nx, ny) field.
        source: Optional source term S(x, y, t).
                Called as source(X, Y, t) where X, Y have shape (nx, ny).
    """
    grid: Grid2D
    bc: BoundaryCondition2D
    c: float | jnp.ndarray
    source: Optional[Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.c, (int, float)):
            if self.c <= 0:
                raise ValueError(f"c must be positive, got {self.c}")
            object.__setattr__(self, "c", array(float(self.c)))


@dataclass(frozen=True)
class WaveEquation3D:
    """Complete 3D wave equation problem definition.

    u_tt = c^2 * nabla^2 u + S(x, y, z, t)

    Args:
        grid: 3D spatial grid.
        bc: Boundary conditions.
        c: Wave speed. Scalar or (nx, ny, nz) field.
        source: Optional source term S(x, y, z, t).
                Called as source(X, Y, Z, t) where X, Y, Z have shape
                (nx, ny, nz).
    """
    grid: Grid3D
    bc: BoundaryCondition3D
    c: float | jnp.ndarray
    source: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.c, (int, float)):
            if self.c <= 0:
                raise ValueError(f"c must be positive, got {self.c}")
            object.__setattr__(self, "c", array(float(self.c)))
