# diffheat/physics/heat2d.py
"""2D heat equation problem definition."""
from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition2D
from ..mesh.grid2d import Grid2D
from ..utils import array


@dataclass(frozen=True)
class HeatEquation2D:
    """Complete 2D heat equation problem definition.

    dT/dt = alpha * nabla^2 T + S(x, y, t)

    Args:
        grid: 2D spatial grid.
        bc: Boundary conditions.
        alpha: Thermal diffusivity. Scalar or (nx, ny) field.
        source: Optional source term S(x, y, t). Called as source(X, Y, t).
                X and Y have shape (ny, nx) matching grid.X and grid.Y, or (nx, ny) if transposed.
                Typically, grid.X.T and grid.Y.T have shape (nx, ny) matching T.
    """
    grid: Grid2D
    bc: BoundaryCondition2D
    alpha: float | jnp.ndarray
    source: Optional[Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            # Convert to array for JAX tracing
            object.__setattr__(self, "alpha", array(float(self.alpha)))
