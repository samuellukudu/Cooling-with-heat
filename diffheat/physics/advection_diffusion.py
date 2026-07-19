# diffheat/physics/advection_diffusion.py
"""Advection-diffusion equation problem definitions."""
from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition, BoundaryCondition2D, BoundaryCondition3D
from ..mesh.grid1d import Grid1D
from ..mesh.grid2d import Grid2D
from ..mesh.grid3d import Grid3D
from ..utils import array


@dataclass(frozen=True)
class AdvectionDiffusion1D:
    """Complete 1D advection-diffusion equation problem definition.

    dT/dt = alpha * d^2T/dx^2 - u * dT/dx + S(x, t)

    Args:
        grid: 1D spatial grid.
        bc: Boundary conditions.
        alpha: Thermal diffusivity. Scalar or (N,) field.
        velocity: Velocity field u(x, t). Called as velocity(x_centers, t).
                  Returns (N,) velocity array.
        source: Optional source term S(x, t).
    """
    grid: Grid1D
    bc: BoundaryCondition
    alpha: float | jnp.ndarray
    velocity: Callable[[jnp.ndarray, float], jnp.ndarray]
    source: Optional[Callable[[jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            object.__setattr__(self, "alpha", array(float(self.alpha)))


@dataclass(frozen=True)
class AdvectionDiffusion2D:
    """Complete 2D advection-diffusion equation problem definition.

    dT/dt = alpha * nabla^2 T - u_x*dT/dx - u_y*dT/dy + S(x, y, t)

    Args:
        grid: 2D spatial grid.
        bc: Boundary conditions.
        alpha: Thermal diffusivity. Scalar or (nx, ny) field.
        velocity: Velocity field (u_x, u_y). Called as velocity(X, Y, t)
                  where X, Y are (nx, ny) arrays matching T.
                  Returns ((nx, ny), (nx, ny)) tuple.
        source: Optional source term S(x, y, t).
    """
    grid: Grid2D
    bc: BoundaryCondition2D
    alpha: float | jnp.ndarray
    velocity: Callable[[jnp.ndarray, jnp.ndarray, float], tuple[jnp.ndarray, jnp.ndarray]]
    source: Optional[Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            object.__setattr__(self, "alpha", array(float(self.alpha)))


@dataclass(frozen=True)
class AdvectionDiffusion3D:
    """Complete 3D advection-diffusion equation problem definition.

    dT/dt = alpha * nabla^2 T - u_x*dT/dx - u_y*dT/dy - u_z*dT/dz + S(x, y, z, t)

    Args:
        grid: 3D spatial grid.
        bc: Boundary conditions.
        alpha: Thermal diffusivity. Scalar or (nx, ny, nz) field.
        velocity: Velocity field (u_x, u_y, u_z). Called as velocity(X, Y, Z, t)
                  where X, Y, Z are (nx, ny, nz) arrays matching T.
                  Returns ((nx, ny, nz), (nx, ny, nz), (nx, ny, nz)) tuple.
        source: Optional source term S(x, y, z, t).
    """
    grid: Grid3D
    bc: BoundaryCondition3D
    alpha: float | jnp.ndarray
    velocity: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray, float],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ]
    source: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            object.__setattr__(self, "alpha", array(float(self.alpha)))
