# diffheat/physics/telegrapher.py
"""1D, 2D, and 3D Telegrapher (hyperbolic heat) equation problem definitions.

The Telegrapher equation (also known as the relativistic heat equation):

    tau * u_tt + u_t = kappa * nabla^2 u + S

where:
    - kappa is the thermal diffusivity (``alpha``)
    - tau > 0 is the thermal relaxation time
    - When tau -> 0, this reduces to the classical Fourier heat equation
    - When tau is large, heat propagates as a damped wave with speed
      v = sqrt(kappa / tau)

Reference:
    The finite propagation speed models heat carried by phonons/electrons
    at their physical speed of sound, correcting the non-physical
    infinite-speed paradox of the parabolic Fourier equation.
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
class TelegrapherEquation1D:
    """Complete 1D Telegrapher equation problem definition.

    tau * u_tt + u_t = kappa * u_xx + S(x, t)

    Args:
        grid: 1D spatial grid.
        bc: Boundary conditions (Dirichlet or Neumann).
        alpha: Thermal diffusivity kappa. Scalar or (n_cells,) field.
        tau: Thermal relaxation time. Positive scalar or (n_cells,) field.
        source: Optional source term S(x, t). Called as source(centers, t).
    """
    grid: Grid1D
    bc: BoundaryCondition
    alpha: float | jnp.ndarray
    tau: float | jnp.ndarray
    source: Optional[Callable[[jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            object.__setattr__(self, "alpha", array(float(self.alpha)))
        if isinstance(self.tau, (int, float)):
            if self.tau <= 0:
                raise ValueError(f"tau must be positive, got {self.tau}")
            object.__setattr__(self, "tau", array(float(self.tau)))


@dataclass(frozen=True)
class TelegrapherEquation2D:
    """Complete 2D Telegrapher equation problem definition.

    tau * u_tt + u_t = kappa * (u_xx + u_yy) + S(x, y, t)

    Args:
        grid: 2D spatial grid.
        bc: Boundary conditions.
        alpha: Thermal diffusivity kappa. Scalar or (nx, ny) field.
        tau: Thermal relaxation time. Positive scalar or (nx, ny) field.
        source: Optional source term S(x, y, t).
                Called as source(X, Y, t) where X, Y have shape (nx, ny).
    """
    grid: Grid2D
    bc: BoundaryCondition2D
    alpha: float | jnp.ndarray
    tau: float | jnp.ndarray
    source: Optional[Callable[[jnp.ndarray, jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            object.__setattr__(self, "alpha", array(float(self.alpha)))
        if isinstance(self.tau, (int, float)):
            if self.tau <= 0:
                raise ValueError(f"tau must be positive, got {self.tau}")
            object.__setattr__(self, "tau", array(float(self.tau)))


@dataclass(frozen=True)
class TelegrapherEquation3D:
    """Complete 3D Telegrapher equation problem definition.

    tau * u_tt + u_t = kappa * nabla^2 u + S(x, y, z, t)

    Args:
        grid: 3D spatial grid.
        bc: Boundary conditions.
        alpha: Thermal diffusivity kappa. Scalar or (nx, ny, nz) field.
        tau: Thermal relaxation time. Positive scalar or (nx, ny, nz) field.
        source: Optional source term S(x, y, z, t).
                Called as source(X, Y, Z, t) where X, Y, Z have shape
                (nx, ny, nz).
    """
    grid: Grid3D
    bc: BoundaryCondition3D
    alpha: float | jnp.ndarray
    tau: float | jnp.ndarray
    source: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            object.__setattr__(self, "alpha", array(float(self.alpha)))
        if isinstance(self.tau, (int, float)):
            if self.tau <= 0:
                raise ValueError(f"tau must be positive, got {self.tau}")
            object.__setattr__(self, "tau", array(float(self.tau)))
