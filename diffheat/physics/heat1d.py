# diffheat/physics/heat1d.py
"""1D heat equation problem definition."""
from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from ..mesh.boundary import BoundaryCondition
from ..mesh.grid1d import Grid1D
from ..utils import array


def apply_boundary_conditions(
    L: jnp.ndarray,
    grid: Grid1D,
    bc: BoundaryCondition,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Modify the Laplacian and create a boundary source vector for the given BCs.

    Uses ghost-cell method:
      Dirichlet: T_ghost = 2*T_boundary - T_boundary_cell
      Neumann:   T_ghost = T_boundary_cell - dx * dT/dx

    Args:
        L: (N, N) raw Laplacian matrix from make_laplacian.
        grid: The 1D grid.
        bc: Boundary conditions.

    Returns:
        (L_modified, b_source) where dT/dt = alpha * (L_modified @ T + b_source) + S.
    """
    n = grid.n_cells
    dx = grid.dx
    L_mod = L  # JAX arrays are immutable, no .copy() needed
    b_source = jnp.zeros(n, dtype=L.dtype)

    # --- Left boundary (cell index 0) ---
    if bc.kind == "dirichlet":
        L_mod = L_mod.at[0, 0].set(-3.0 / (dx[0] * dx[0]))
        L_mod = L_mod.at[0, 1].set(1.0 / (dx[0] * dx[0]))
        b_source = b_source.at[0].set(2.0 * bc.value[0] / (dx[0] * dx[0]))
    elif bc.kind == "neumann":
        L_mod = L_mod.at[0, 0].set(-1.0 / (dx[0] * dx[0]))
        L_mod = L_mod.at[0, 1].set(1.0 / (dx[0] * dx[0]))
        b_source = b_source.at[0].set(-bc.value[0] / dx[0])

    # --- Right boundary (cell index n-1) ---
    if bc.kind == "dirichlet":
        L_mod = L_mod.at[n - 1, n - 1].set(-3.0 / (dx[n - 1] * dx[n - 1]))
        L_mod = L_mod.at[n - 1, n - 2].set(1.0 / (dx[n - 1] * dx[n - 1]))
        b_source = b_source.at[n - 1].set(
            2.0 * bc.value[1] / (dx[n - 1] * dx[n - 1])
        )
    elif bc.kind == "neumann":
        L_mod = L_mod.at[n - 1, n - 1].set(-1.0 / (dx[n - 1] * dx[n - 1]))
        L_mod = L_mod.at[n - 1, n - 2].set(1.0 / (dx[n - 1] * dx[n - 1]))
        b_source = b_source.at[n - 1].set(bc.value[1] / dx[n - 1])

    return L_mod, b_source


@dataclass(frozen=True)
class HeatEquation1D:
    """Complete 1D heat equation problem definition.

    dT/dt = alpha * d^2T/dx^2 + S(x, t)

    Args:
        grid: 1D spatial grid.
        bc: Boundary conditions.
        alpha: Thermal diffusivity. Scalar or (N,) field.
        source: Optional source term S(x, t). Called as source(x_coords, time).
    """
    grid: Grid1D
    bc: BoundaryCondition
    alpha: float | jnp.ndarray
    source: Optional[Callable[[jnp.ndarray, float], jnp.ndarray]] = None

    def __post_init__(self):
        if isinstance(self.alpha, (int, float)):
            if self.alpha <= 0:
                raise ValueError(f"alpha must be positive, got {self.alpha}")
            # Convert to array for JAX tracing
            object.__setattr__(self, "alpha", array(float(self.alpha)))
