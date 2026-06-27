# diffheat/mesh/boundary.py
"""Boundary condition definitions for 1D and 2D heat equations."""
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from ..utils import array
from .grid1d import Grid1D


@dataclass(frozen=True)
class BoundaryCondition:
    """Boundary condition for the 1D heat equation.

    Args:
        kind: "dirichlet" (fixed temperature) or "neumann" (fixed flux/gradient).
        value: (2,) array — [left_value, right_value].
               For Dirichlet: prescribed temperature at each boundary.
               For Neumann: prescribed dT/dx at each boundary (positive = into domain).
    """
    kind: str
    value: jnp.ndarray

    def __post_init__(self):
        if self.kind not in ("dirichlet", "neumann"):
            raise ValueError(f"Unknown boundary kind: {self.kind}")
        if self.value.shape != (2,):
            raise ValueError(f"Boundary value must have shape (2,), got {self.value.shape}")


@dataclass(frozen=True)
class BoundaryCondition2D:
    r"""Boundary conditions for the 2D heat equation.

    Each edge is a dict with keys:
        kind: "dirichlet" (fixed temperature) or "neumann" (fixed flux/gradient).
        value: prescribed temperature (Dirichlet) or dT/dn (Neumann, positive = into domain).

    **Neumann sign convention:** ``value`` is the inward normal derivative dT/dn,
    i.e., positive flux INTO the domain.  On the right (x = Lx) and top (y = Ly)
    edges this is opposite to the +x / +y coordinate direction.  This is the same
    "into domain" convention used by the 1D ``BoundaryCondition``.

    Args:
        left: dict with "kind" and "value" for the left boundary (x = 0).
        right: dict with "kind" and "value" for the right boundary (x = Lx).
        bottom: dict with "kind" and "value" for the bottom boundary (y = 0).
        top: dict with "kind" and "value" for the top boundary (y = Ly).
    """
    left: dict
    right: dict
    bottom: dict
    top: dict

    def __post_init__(self):
        for edge_name in ("left", "right", "bottom", "top"):
            edge = getattr(self, edge_name)
            if edge["kind"] not in ("dirichlet", "neumann"):
                raise ValueError(
                    f"Unknown boundary kind for {edge_name}: {edge['kind']}"
                )


def apply_boundary_conditions_2d(
    operator_fn: Callable[[jnp.ndarray], jnp.ndarray],
    grid: "Grid2D",
    bc: BoundaryCondition2D,
    T: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply 2D boundary conditions by correcting the wrap-around operator output.

    The raw operators (e.g. ``laplacian_2d``) use ``jnp.roll`` which implicitly
    imposes periodic boundary conditions.  This function replaces the incorrect
    (periodic) finite-difference stencils at the domain edges with the correct
    ghost-cell stencils for the prescribed Dirichlet or Neumann conditions.

    Args:
        operator_fn: Callable ``T -> operator(T)`` that computes the raw discrete
                     operator (e.g. ``lambda T: laplacian_2d(T, grid)``).
        grid: The 2D grid.
        bc: Boundary conditions for the four edges.
        T: (nx, ny) temperature field at cell centers.

    Returns:
        (L_T, b_source) where ``dT/dt = alpha * (L_T + b_source) + S``.
        Both have shape ``(nx, ny)``.
    """
    nx, ny = grid.nx, grid.ny
    dx = grid.dx  # (nx,)
    dy = grid.dy  # (ny,)

    L_T = operator_fn(T)
    b_source = jnp.zeros((nx, ny), dtype=L_T.dtype)

    # --- Left boundary (i = 0) ---
    if bc.left["kind"] == "dirichlet":
        T_boundary = bc.left["value"]
        incorrect_x = (
            T[nx - 1, :] + T[1, :] - 2.0 * T[0, :]
        ) / (dx[0] * dx[0])
        correct_x = (
            T[1, :] - 3.0 * T[0, :]
        ) / (dx[0] * dx[0])
        L_T = L_T.at[0, :].add(correct_x - incorrect_x)
        b_source = b_source.at[0, :].add(
            2.0 * T_boundary / (dx[0] * dx[0])
        )
    elif bc.left["kind"] == "neumann":
        dT_dn = bc.left["value"]
        incorrect_x = (
            T[nx - 1, :] + T[1, :] - 2.0 * T[0, :]
        ) / (dx[0] * dx[0])
        correct_x = (
            T[1, :] - T[0, :]
        ) / (dx[0] * dx[0])
        L_T = L_T.at[0, :].add(correct_x - incorrect_x)
        b_source = b_source.at[0, :].add(-dT_dn / dx[0])

    # --- Right boundary (i = nx - 1) ---
    if bc.right["kind"] == "dirichlet":
        T_boundary = bc.right["value"]
        incorrect_x = (
            T[0, :] + T[nx - 2, :] - 2.0 * T[nx - 1, :]
        ) / (dx[nx - 1] * dx[nx - 1])
        correct_x = (
            T[nx - 2, :] - 3.0 * T[nx - 1, :]
        ) / (dx[nx - 1] * dx[nx - 1])
        L_T = L_T.at[nx - 1, :].add(correct_x - incorrect_x)
        b_source = b_source.at[nx - 1, :].add(
            2.0 * T_boundary / (dx[nx - 1] * dx[nx - 1])
        )
    elif bc.right["kind"] == "neumann":
        dT_dn = bc.right["value"]
        incorrect_x = (
            T[0, :] + T[nx - 2, :] - 2.0 * T[nx - 1, :]
        ) / (dx[nx - 1] * dx[nx - 1])
        correct_x = (
            T[nx - 2, :] - T[nx - 1, :]
        ) / (dx[nx - 1] * dx[nx - 1])
        L_T = L_T.at[nx - 1, :].add(correct_x - incorrect_x)
        b_source = b_source.at[nx - 1, :].add(-dT_dn / dx[nx - 1])

    # --- Bottom boundary (j = 0) ---
    if bc.bottom["kind"] == "dirichlet":
        T_boundary = bc.bottom["value"]
        incorrect_y = (
            T[:, ny - 1] + T[:, 1] - 2.0 * T[:, 0]
        ) / (dy[0] * dy[0])
        correct_y = (
            T[:, 1] - 3.0 * T[:, 0]
        ) / (dy[0] * dy[0])
        L_T = L_T.at[:, 0].add(correct_y - incorrect_y)
        b_source = b_source.at[:, 0].add(
            2.0 * T_boundary / (dy[0] * dy[0])
        )
    elif bc.bottom["kind"] == "neumann":
        dT_dn = bc.bottom["value"]
        incorrect_y = (
            T[:, ny - 1] + T[:, 1] - 2.0 * T[:, 0]
        ) / (dy[0] * dy[0])
        correct_y = (
            T[:, 1] - T[:, 0]
        ) / (dy[0] * dy[0])
        L_T = L_T.at[:, 0].add(correct_y - incorrect_y)
        b_source = b_source.at[:, 0].add(-dT_dn / dy[0])

    # --- Top boundary (j = ny - 1) ---
    if bc.top["kind"] == "dirichlet":
        T_boundary = bc.top["value"]
        incorrect_y = (
            T[:, 0] + T[:, ny - 2] - 2.0 * T[:, ny - 1]
        ) / (dy[ny - 1] * dy[ny - 1])
        correct_y = (
            T[:, ny - 2] - 3.0 * T[:, ny - 1]
        ) / (dy[ny - 1] * dy[ny - 1])
        L_T = L_T.at[:, ny - 1].add(correct_y - incorrect_y)
        b_source = b_source.at[:, ny - 1].add(
            2.0 * T_boundary / (dy[ny - 1] * dy[ny - 1])
        )
    elif bc.top["kind"] == "neumann":
        dT_dn = bc.top["value"]
        incorrect_y = (
            T[:, 0] + T[:, ny - 2] - 2.0 * T[:, ny - 1]
        ) / (dy[ny - 1] * dy[ny - 1])
        correct_y = (
            T[:, ny - 2] - T[:, ny - 1]
        ) / (dy[ny - 1] * dy[ny - 1])
        L_T = L_T.at[:, ny - 1].add(correct_y - incorrect_y)
        b_source = b_source.at[:, ny - 1].add(-dT_dn / dy[ny - 1])

    return L_T, b_source


def apply_boundary_conditions_1d(
    operator_fn: Callable[[jnp.ndarray], jnp.ndarray],
    grid: Grid1D,
    bc: "BoundaryCondition",
    T: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply 1D boundary conditions by correcting periodic operator output.

    Same ghost-cell method as the 2D version.  The raw operator (e.g.
    ``laplacian_1d``) uses ``jnp.roll`` which imposes periodic BCs.
    This function corrects the edge cells for the prescribed Dirichlet
    or Neumann conditions.

    Args:
        operator_fn: Callable ``T -> operator(T)`` (e.g.
                     ``lambda T: laplacian_1d(T, grid)``).
        grid: The 1D grid.
        bc: Boundary conditions.
        T: (N,) field at cell centers.

    Returns:
        (L_T, b_source) where ``dT/dt = alpha * (L_T + b_source) + S``.
        Both have shape ``(N,)``.
    """
    n = grid.n_cells
    dx = grid.dx

    L_T = operator_fn(T)
    b_source = jnp.zeros(n, dtype=L_T.dtype)

    # --- Left boundary (i = 0) ---
    if bc.kind == "dirichlet":
        T_boundary = bc.value[0]
        incorrect = (T[n - 1] + T[1] - 2.0 * T[0]) / (dx[0] * dx[0])
        correct = (T[1] - 3.0 * T[0]) / (dx[0] * dx[0])
        L_T = L_T.at[0].add(correct - incorrect)
        b_source = b_source.at[0].add(2.0 * T_boundary / (dx[0] * dx[0]))
    elif bc.kind == "neumann":
        dT_dn = bc.value[0]
        incorrect = (T[n - 1] + T[1] - 2.0 * T[0]) / (dx[0] * dx[0])
        correct = (T[1] - T[0]) / (dx[0] * dx[0])
        L_T = L_T.at[0].add(correct - incorrect)
        b_source = b_source.at[0].add(-dT_dn / dx[0])

    # --- Right boundary (i = n - 1) ---
    if bc.kind == "dirichlet":
        T_boundary = bc.value[1]
        incorrect = (T[0] + T[n - 2] - 2.0 * T[n - 1]) / (dx[n - 1] * dx[n - 1])
        correct = (T[n - 2] - 3.0 * T[n - 1]) / (dx[n - 1] * dx[n - 1])
        L_T = L_T.at[n - 1].add(correct - incorrect)
        b_source = b_source.at[n - 1].add(
            2.0 * T_boundary / (dx[n - 1] * dx[n - 1])
        )
    elif bc.kind == "neumann":
        dT_dn = bc.value[1]
        incorrect = (T[0] + T[n - 2] - 2.0 * T[n - 1]) / (dx[n - 1] * dx[n - 1])
        correct = (T[n - 2] - T[n - 1]) / (dx[n - 1] * dx[n - 1])
        L_T = L_T.at[n - 1].add(correct - incorrect)
        b_source = b_source.at[n - 1].add(-dT_dn / dx[n - 1])

    return L_T, b_source
