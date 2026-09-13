"""Boundary conditions for 1D/2D/3D (DESIGN §4).

Ghost-cell method mirrored from the frozen ``diffheat`` boundary module
without importing it. Raw operators use ``jnp.roll`` (implicitly periodic);
these helpers replace the incorrect wrap-around stencils at domain faces
with Dirichlet/Neumann ghost-cell stencils.

Neumann sign convention (shared with diffheat): ``value`` is the inward
normal derivative dT/dn, i.e. positive flux INTO the domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp


@dataclass(frozen=True)
class BoundaryCondition:
    """1D boundary condition (both ends share one kind).

    Args:
        kind: "dirichlet" or "neumann".
        value: (2,) array — [left_value, right_value].
    """

    kind: str
    value: jnp.ndarray

    def __post_init__(self):
        if self.kind not in ("dirichlet", "neumann"):
            raise ValueError(f"Unknown boundary kind: {self.kind}")
        if self.value.shape != (2,):
            raise ValueError(
                f"Boundary value must have shape (2,), got {self.value.shape}"
            )


@dataclass(frozen=True)
class BoundaryCondition2D:
    """2D boundary conditions (one dict per edge).

    Each edge dict: {"kind": "dirichlet"|"neumann", "value": float}.
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


@dataclass(frozen=True)
class BoundaryCondition3D:
    """3D boundary conditions (one dict per face)."""

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


def apply_boundary_conditions_1d(
    operator_fn: Callable[[jnp.ndarray], jnp.ndarray],
    grid,
    bc: "BoundaryCondition",
    T: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Correct periodic 1D operator output for Dirichlet/Neumann BCs."""
    n = grid.n_cells
    dx = grid.dx
    L_T = operator_fn(T)
    b_source = jnp.zeros(n, dtype=L_T.dtype)
    if bc.kind == "dirichlet":
        T_b = bc.value[0]
        incorrect = (T[n - 1] + T[1] - 2.0 * T[0]) / (dx[0] * dx[0])
        correct = (T[1] - 3.0 * T[0]) / (dx[0] * dx[0])
        L_T = L_T.at[0].add(correct - incorrect)
        b_source = b_source.at[0].add(2.0 * T_b / (dx[0] * dx[0]))
        T_b = bc.value[1]
        incorrect = (T[0] + T[n - 2] - 2.0 * T[n - 1]) / (dx[n - 1] * dx[n - 1])
        correct = (T[n - 2] - 3.0 * T[n - 1]) / (dx[n - 1] * dx[n - 1])
        L_T = L_T.at[n - 1].add(correct - incorrect)
        b_source = b_source.at[n - 1].add(2.0 * T_b / (dx[n - 1] * dx[n - 1]))
    else:
        dT_dn = bc.value[0]
        incorrect = (T[n - 1] + T[1] - 2.0 * T[0]) / (dx[0] * dx[0])
        correct = (T[1] - T[0]) / (dx[0] * dx[0])
        L_T = L_T.at[0].add(correct - incorrect)
        b_source = b_source.at[0].add(-dT_dn / dx[0])
        dT_dn = bc.value[1]
        incorrect = (T[0] + T[n - 2] - 2.0 * T[n - 1]) / (dx[n - 1] * dx[n - 1])
        correct = (T[n - 2] - T[n - 1]) / (dx[n - 1] * dx[n - 1])
        L_T = L_T.at[n - 1].add(correct - incorrect)
        b_source = b_source.at[n - 1].add(-dT_dn / dx[n - 1])
    return L_T, b_source


def apply_boundary_conditions_2d(
    operator_fn: Callable[[jnp.ndarray], jnp.ndarray],
    grid,
    bc: BoundaryCondition2D,
    T: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Correct periodic 2D operator output. Fields have shape ``(nx, ny)``."""
    nx, ny = grid.nx, grid.ny
    dx = grid.dx
    dy = grid.dy
    L_T = operator_fn(T)
    b_source = jnp.zeros((nx, ny), dtype=L_T.dtype)

    if bc.left["kind"] == "dirichlet":
        T_b = bc.left["value"]
        incorrect = (T[nx - 1, :] + T[1, :] - 2.0 * T[0, :]) / (dx[0] ** 2)
        correct = (T[1, :] - 3.0 * T[0, :]) / (dx[0] ** 2)
        L_T = L_T.at[0, :].add(correct - incorrect)
        b_source = b_source.at[0, :].add(2.0 * T_b / (dx[0] ** 2))
    else:
        dT_dn = bc.left["value"]
        incorrect = (T[nx - 1, :] + T[1, :] - 2.0 * T[0, :]) / (dx[0] ** 2)
        correct = (T[1, :] - T[0, :]) / (dx[0] ** 2)
        L_T = L_T.at[0, :].add(correct - incorrect)
        b_source = b_source.at[0, :].add(-dT_dn / dx[0])

    if bc.right["kind"] == "dirichlet":
        T_b = bc.right["value"]
        incorrect = (T[0, :] + T[nx - 2, :] - 2.0 * T[nx - 1, :]) / (dx[nx - 1] ** 2)
        correct = (T[nx - 2, :] - 3.0 * T[nx - 1, :]) / (dx[nx - 1] ** 2)
        L_T = L_T.at[nx - 1, :].add(correct - incorrect)
        b_source = b_source.at[nx - 1, :].add(2.0 * T_b / (dx[nx - 1] ** 2))
    else:
        dT_dn = bc.right["value"]
        incorrect = (T[0, :] + T[nx - 2, :] - 2.0 * T[nx - 1, :]) / (dx[nx - 1] ** 2)
        correct = (T[nx - 2, :] - T[nx - 1, :]) / (dx[nx - 1] ** 2)
        L_T = L_T.at[nx - 1, :].add(correct - incorrect)
        b_source = b_source.at[nx - 1, :].add(-dT_dn / dx[nx - 1])

    if bc.bottom["kind"] == "dirichlet":
        T_b = bc.bottom["value"]
        incorrect = (T[:, ny - 1] + T[:, 1] - 2.0 * T[:, 0]) / (dy[0] ** 2)
        correct = (T[:, 1] - 3.0 * T[:, 0]) / (dy[0] ** 2)
        L_T = L_T.at[:, 0].add(correct - incorrect)
        b_source = b_source.at[:, 0].add(2.0 * T_b / (dy[0] ** 2))
    else:
        dT_dn = bc.bottom["value"]
        incorrect = (T[:, ny - 1] + T[:, 1] - 2.0 * T[:, 0]) / (dy[0] ** 2)
        correct = (T[:, 1] - T[:, 0]) / (dy[0] ** 2)
        L_T = L_T.at[:, 0].add(correct - incorrect)
        b_source = b_source.at[:, 0].add(-dT_dn / dy[0])

    if bc.top["kind"] == "dirichlet":
        T_b = bc.top["value"]
        incorrect = (T[:, 0] + T[:, ny - 2] - 2.0 * T[:, ny - 1]) / (dy[ny - 1] ** 2)
        correct = (T[:, ny - 2] - 3.0 * T[:, ny - 1]) / (dy[ny - 1] ** 2)
        L_T = L_T.at[:, ny - 1].add(correct - incorrect)
        b_source = b_source.at[:, ny - 1].add(2.0 * T_b / (dy[ny - 1] ** 2))
    else:
        dT_dn = bc.top["value"]
        incorrect = (T[:, 0] + T[:, ny - 2] - 2.0 * T[:, ny - 1]) / (dy[ny - 1] ** 2)
        correct = (T[:, ny - 2] - T[:, ny - 1]) / (dy[ny - 1] ** 2)
        L_T = L_T.at[:, ny - 1].add(correct - incorrect)
        b_source = b_source.at[:, ny - 1].add(-dT_dn / dy[ny - 1])

    return L_T, b_source


def apply_boundary_conditions_3d(
    operator_fn: Callable[[jnp.ndarray], jnp.ndarray],
    grid,
    bc: BoundaryCondition3D,
    T: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Correct periodic 3D operator output. Fields have shape ``(nx, ny, nz)``."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    L_T = operator_fn(T)
    b_source = jnp.zeros((nx, ny, nz), dtype=L_T.dtype)

    if bc.left["kind"] == "dirichlet":
        T_b = bc.left["value"]
        incorrect = (T[nx - 1, :, :] + T[1, :, :] - 2.0 * T[0, :, :]) / (dx[0] ** 2)
        correct = (T[1, :, :] - 3.0 * T[0, :, :]) / (dx[0] ** 2)
        L_T = L_T.at[0, :, :].add(correct - incorrect)
        b_source = b_source.at[0, :, :].add(2.0 * T_b / (dx[0] ** 2))
    else:
        dT_dn = bc.left["value"]
        incorrect = (T[nx - 1, :, :] + T[1, :, :] - 2.0 * T[0, :, :]) / (dx[0] ** 2)
        correct = (T[1, :, :] - T[0, :, :]) / (dx[0] ** 2)
        L_T = L_T.at[0, :, :].add(correct - incorrect)
        b_source = b_source.at[0, :, :].add(-dT_dn / dx[0])

    if bc.right["kind"] == "dirichlet":
        T_b = bc.right["value"]
        incorrect = (T[0, :, :] + T[nx - 2, :, :] - 2.0 * T[nx - 1, :, :]) / (dx[nx - 1] ** 2)
        correct = (T[nx - 2, :, :] - 3.0 * T[nx - 1, :, :]) / (dx[nx - 1] ** 2)
        L_T = L_T.at[nx - 1, :, :].add(correct - incorrect)
        b_source = b_source.at[nx - 1, :, :].add(2.0 * T_b / (dx[nx - 1] ** 2))
    else:
        dT_dn = bc.right["value"]
        incorrect = (T[0, :, :] + T[nx - 2, :, :] - 2.0 * T[nx - 1, :, :]) / (dx[nx - 1] ** 2)
        correct = (T[nx - 2, :, :] - T[nx - 1, :, :]) / (dx[nx - 1] ** 2)
        L_T = L_T.at[nx - 1, :, :].add(correct - incorrect)
        b_source = b_source.at[nx - 1, :, :].add(-dT_dn / dx[nx - 1])

    if bc.bottom["kind"] == "dirichlet":
        T_b = bc.bottom["value"]
        incorrect = (T[:, ny - 1, :] + T[:, 1, :] - 2.0 * T[:, 0, :]) / (dy[0] ** 2)
        correct = (T[:, 1, :] - 3.0 * T[:, 0, :]) / (dy[0] ** 2)
        L_T = L_T.at[:, 0, :].add(correct - incorrect)
        b_source = b_source.at[:, 0, :].add(2.0 * T_b / (dy[0] ** 2))
    else:
        dT_dn = bc.bottom["value"]
        incorrect = (T[:, ny - 1, :] + T[:, 1, :] - 2.0 * T[:, 0, :]) / (dy[0] ** 2)
        correct = (T[:, 1, :] - T[:, 0, :]) / (dy[0] ** 2)
        L_T = L_T.at[:, 0, :].add(correct - incorrect)
        b_source = b_source.at[:, 0, :].add(-dT_dn / dy[0])

    if bc.top["kind"] == "dirichlet":
        T_b = bc.top["value"]
        incorrect = (T[:, 0, :] + T[:, ny - 2, :] - 2.0 * T[:, ny - 1, :]) / (dy[ny - 1] ** 2)
        correct = (T[:, ny - 2, :] - 3.0 * T[:, ny - 1, :]) / (dy[ny - 1] ** 2)
        L_T = L_T.at[:, ny - 1, :].add(correct - incorrect)
        b_source = b_source.at[:, ny - 1, :].add(2.0 * T_b / (dy[ny - 1] ** 2))
    else:
        dT_dn = bc.top["value"]
        incorrect = (T[:, 0, :] + T[:, ny - 2, :] - 2.0 * T[:, ny - 1, :]) / (dy[ny - 1] ** 2)
        correct = (T[:, ny - 2, :] - T[:, ny - 1, :]) / (dy[ny - 1] ** 2)
        L_T = L_T.at[:, ny - 1, :].add(correct - incorrect)
        b_source = b_source.at[:, ny - 1, :].add(-dT_dn / dy[ny - 1])

    if bc.front["kind"] == "dirichlet":
        T_b = bc.front["value"]
        incorrect = (T[:, :, nz - 1] + T[:, :, 1] - 2.0 * T[:, :, 0]) / (dz[0] ** 2)
        correct = (T[:, :, 1] - 3.0 * T[:, :, 0]) / (dz[0] ** 2)
        L_T = L_T.at[:, :, 0].add(correct - incorrect)
        b_source = b_source.at[:, :, 0].add(2.0 * T_b / (dz[0] ** 2))
    else:
        dT_dn = bc.front["value"]
        incorrect = (T[:, :, nz - 1] + T[:, :, 1] - 2.0 * T[:, :, 0]) / (dz[0] ** 2)
        correct = (T[:, :, 1] - T[:, :, 0]) / (dz[0] ** 2)
        L_T = L_T.at[:, :, 0].add(correct - incorrect)
        b_source = b_source.at[:, :, 0].add(-dT_dn / dz[0])

    if bc.back["kind"] == "dirichlet":
        T_b = bc.back["value"]
        incorrect = (T[:, :, 0] + T[:, :, nz - 2] - 2.0 * T[:, :, nz - 1]) / (dz[nz - 1] ** 2)
        correct = (T[:, :, nz - 2] - 3.0 * T[:, :, nz - 1]) / (dz[nz - 1] ** 2)
        L_T = L_T.at[:, :, nz - 1].add(correct - incorrect)
        b_source = b_source.at[:, :, nz - 1].add(2.0 * T_b / (dz[nz - 1] ** 2))
    else:
        dT_dn = bc.back["value"]
        incorrect = (T[:, :, 0] + T[:, :, nz - 2] - 2.0 * T[:, :, nz - 1]) / (dz[nz - 1] ** 2)
        correct = (T[:, :, nz - 2] - T[:, :, nz - 1]) / (dz[nz - 1] ** 2)
        L_T = L_T.at[:, :, nz - 1].add(correct - incorrect)
        b_source = b_source.at[:, :, nz - 1].add(-dT_dn / dz[nz - 1])

    return L_T, b_source


__all__ = [
    "BoundaryCondition",
    "BoundaryCondition2D",
    "BoundaryCondition3D",
    "apply_boundary_conditions_1d",
    "apply_boundary_conditions_2d",
    "apply_boundary_conditions_3d",
]
