# diffheat/physics/heat3d.py
"""3D heat equation problem definition."""
from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition3D
from ..mesh.grid3d import Grid3D
from ..utils import array


@dataclass(frozen=True)
class HeatEquation3D:
    """Complete 3D heat equation problem definition.

    dT/dt = alpha * nabla^2 T + S(x, y, z, t)

    Args:
        grid: 3D spatial grid.
        bc: Boundary conditions.
        alpha: Thermal diffusivity. Scalar or (nx, ny, nz) field.
        source: Optional source term S(x, y, z, t). Called as source(X, Y, Z, t).
                X, Y, and Z have shape (nx, ny, nz) matching grid.X, grid.Y, grid.Z.
    """
    grid: Grid3D
    bc: BoundaryCondition3D
    alpha: float | jnp.ndarray
    source: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            # Convert to array for JAX tracing
            object.__setattr__(self, "alpha", array(float(self.alpha)))
