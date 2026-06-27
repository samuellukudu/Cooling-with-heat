# diffheat/mesh/boundary.py
"""Boundary condition definitions for 1D, 2D, and 3D heat equations."""
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


# ---------------------------------------------------------------------------
# 3D boundary conditions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundaryCondition3D:
    r"""Boundary conditions for the 3D heat equation.

    Each of the six faces is a dict with keys:
        kind: ``"dirichlet"`` (fixed temperature) or ``"neumann"`` (fixed flux).
        value: Prescribed temperature (Dirichlet) or ``dT/dn`` (Neumann).

    **Neumann sign convention:** ``value`` is the inward normal derivative
    ``dT/dn``, i.e., positive flux *into* the domain.  On the positive-side
    faces (right, top, back) this is opposite to the +x/+y/+z direction.

    Faces:
        left:   x = 0
        right:  x = Lx
        bottom: y = 0
        top:    y = Ly
        front:  z = 0
        back:   z = Lz
    """
    left: dict
    right: dict
    bottom: dict
    top: dict
    front: dict
    back: dict

    def __post_init__(self):
        for face in ("left", "right", "bottom", "top", "front", "back"):
            edge = getattr(self, face)
            if edge["kind"] not in ("dirichlet", "neumann"):
                raise ValueError(
                    f"Unknown boundary kind for {face}: {edge['kind']}"
                )


def apply_boundary_conditions_3d(
    operator_fn: Callable[[jnp.ndarray], jnp.ndarray],
    grid: "Grid3D",
    bc: BoundaryCondition3D,
    T: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply 3D boundary conditions by correcting the wrap-around operator output.

    The raw ``laplacian_3d`` uses ``jnp.roll`` which implicitly imposes periodic
    BCs.  This function replaces the incorrect stencils at the six domain faces
    with ghost-cell stencils for Dirichlet or Neumann conditions.

    Args:
        operator_fn: Callable ``T -> operator(T)`` (e.g.
                     ``lambda T: laplacian_3d(T, grid)``).
        grid: The 3D grid.
        bc: Boundary conditions for the six faces.
        T: (nx, ny, nz) temperature field at cell centers.

    Returns:
        ``(L_T, b_source)`` where ``dT/dt = alpha * (L_T + b_source) + S``.
        Both have shape ``(nx, ny, nz)``.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx = grid.dx  # (nx,)
    dy = grid.dy  # (ny,)
    dz = grid.dz  # (nz,)

    L_T = operator_fn(T)
    b_source = jnp.zeros((nx, ny, nz), dtype=L_T.dtype)

    # ------------------------------------------------------------------
    # X-direction faces  (axis 0)
    # ------------------------------------------------------------------

    # --- Left face (i = 0) ---
    if bc.left["kind"] == "dirichlet":
        T_bc = bc.left["value"]
        incorrect = (T[nx - 1, :, :] + T[1, :, :] - 2.0 * T[0, :, :]) / (dx[0] ** 2)
        correct = (T[1, :, :] - 3.0 * T[0, :, :]) / (dx[0] ** 2)
        L_T = L_T.at[0, :, :].add(correct - incorrect)
        b_source = b_source.at[0, :, :].add(2.0 * T_bc / (dx[0] ** 2))
    elif bc.left["kind"] == "neumann":
        dT_dn = bc.left["value"]
        incorrect = (T[nx - 1, :, :] + T[1, :, :] - 2.0 * T[0, :, :]) / (dx[0] ** 2)
        correct = (T[1, :, :] - T[0, :, :]) / (dx[0] ** 2)
        L_T = L_T.at[0, :, :].add(correct - incorrect)
        b_source = b_source.at[0, :, :].add(-dT_dn / dx[0])

    # --- Right face (i = nx - 1) ---
    if bc.right["kind"] == "dirichlet":
        T_bc = bc.right["value"]
        incorrect = (T[0, :, :] + T[nx - 2, :, :] - 2.0 * T[nx - 1, :, :]) / (dx[nx - 1] ** 2)
        correct = (T[nx - 2, :, :] - 3.0 * T[nx - 1, :, :]) / (dx[nx - 1] ** 2)
        L_T = L_T.at[nx - 1, :, :].add(correct - incorrect)
        b_source = b_source.at[nx - 1, :, :].add(2.0 * T_bc / (dx[nx - 1] ** 2))
    elif bc.right["kind"] == "neumann":
        dT_dn = bc.right["value"]
        incorrect = (T[0, :, :] + T[nx - 2, :, :] - 2.0 * T[nx - 1, :, :]) / (dx[nx - 1] ** 2)
        correct = (T[nx - 2, :, :] - T[nx - 1, :, :]) / (dx[nx - 1] ** 2)
        L_T = L_T.at[nx - 1, :, :].add(correct - incorrect)
        b_source = b_source.at[nx - 1, :, :].add(-dT_dn / dx[nx - 1])

    # ------------------------------------------------------------------
    # Y-direction faces  (axis 1)
    # ------------------------------------------------------------------

    # --- Bottom face (j = 0) ---
    if bc.bottom["kind"] == "dirichlet":
        T_bc = bc.bottom["value"]
        incorrect = (T[:, ny - 1, :] + T[:, 1, :] - 2.0 * T[:, 0, :]) / (dy[0] ** 2)
        correct = (T[:, 1, :] - 3.0 * T[:, 0, :]) / (dy[0] ** 2)
        L_T = L_T.at[:, 0, :].add(correct - incorrect)
        b_source = b_source.at[:, 0, :].add(2.0 * T_bc / (dy[0] ** 2))
    elif bc.bottom["kind"] == "neumann":
        dT_dn = bc.bottom["value"]
        incorrect = (T[:, ny - 1, :] + T[:, 1, :] - 2.0 * T[:, 0, :]) / (dy[0] ** 2)
        correct = (T[:, 1, :] - T[:, 0, :]) / (dy[0] ** 2)
        L_T = L_T.at[:, 0, :].add(correct - incorrect)
        b_source = b_source.at[:, 0, :].add(-dT_dn / dy[0])

    # --- Top face (j = ny - 1) ---
    if bc.top["kind"] == "dirichlet":
        T_bc = bc.top["value"]
        incorrect = (T[:, 0, :] + T[:, ny - 2, :] - 2.0 * T[:, ny - 1, :]) / (dy[ny - 1] ** 2)
        correct = (T[:, ny - 2, :] - 3.0 * T[:, ny - 1, :]) / (dy[ny - 1] ** 2)
        L_T = L_T.at[:, ny - 1, :].add(correct - incorrect)
        b_source = b_source.at[:, ny - 1, :].add(2.0 * T_bc / (dy[ny - 1] ** 2))
    elif bc.top["kind"] == "neumann":
        dT_dn = bc.top["value"]
        incorrect = (T[:, 0, :] + T[:, ny - 2, :] - 2.0 * T[:, ny - 1, :]) / (dy[ny - 1] ** 2)
        correct = (T[:, ny - 2, :] - T[:, ny - 1, :]) / (dy[ny - 1] ** 2)
        L_T = L_T.at[:, ny - 1, :].add(correct - incorrect)
        b_source = b_source.at[:, ny - 1, :].add(-dT_dn / dy[ny - 1])

    # ------------------------------------------------------------------
    # Z-direction faces  (axis 2)
    # ------------------------------------------------------------------

    # --- Front face (k = 0) ---
    if bc.front["kind"] == "dirichlet":
        T_bc = bc.front["value"]
        incorrect = (T[:, :, nz - 1] + T[:, :, 1] - 2.0 * T[:, :, 0]) / (dz[0] ** 2)
        correct = (T[:, :, 1] - 3.0 * T[:, :, 0]) / (dz[0] ** 2)
        L_T = L_T.at[:, :, 0].add(correct - incorrect)
        b_source = b_source.at[:, :, 0].add(2.0 * T_bc / (dz[0] ** 2))
    elif bc.front["kind"] == "neumann":
        dT_dn = bc.front["value"]
        incorrect = (T[:, :, nz - 1] + T[:, :, 1] - 2.0 * T[:, :, 0]) / (dz[0] ** 2)
        correct = (T[:, :, 1] - T[:, :, 0]) / (dz[0] ** 2)
        L_T = L_T.at[:, :, 0].add(correct - incorrect)
        b_source = b_source.at[:, :, 0].add(-dT_dn / dz[0])

    # --- Back face (k = nz - 1) ---
    if bc.back["kind"] == "dirichlet":
        T_bc = bc.back["value"]
        incorrect = (T[:, :, 0] + T[:, :, nz - 2] - 2.0 * T[:, :, nz - 1]) / (dz[nz - 1] ** 2)
        correct = (T[:, :, nz - 2] - 3.0 * T[:, :, nz - 1]) / (dz[nz - 1] ** 2)
        L_T = L_T.at[:, :, nz - 1].add(correct - incorrect)
        b_source = b_source.at[:, :, nz - 1].add(2.0 * T_bc / (dz[nz - 1] ** 2))
    elif bc.back["kind"] == "neumann":
        dT_dn = bc.back["value"]
        incorrect = (T[:, :, 0] + T[:, :, nz - 2] - 2.0 * T[:, :, nz - 1]) / (dz[nz - 1] ** 2)
        correct = (T[:, :, nz - 2] - T[:, :, nz - 1]) / (dz[nz - 1] ** 2)
        L_T = L_T.at[:, :, nz - 1].add(correct - incorrect)
        b_source = b_source.at[:, :, nz - 1].add(-dT_dn / dz[nz - 1])

    return L_T, b_source
