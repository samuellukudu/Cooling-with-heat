"""Cloak 2-D physics — heat equation with spatially-varying diffusivity.

Mirrors the frozen ``diffheat`` 2-D heat hot path
(``laplacian_2d`` + ghost-cell BC correction + ``solve_heat_2d`` explicit
Euler in scan) without importing it. Parity pinned by
``tests/harness/test_diffheat_trials.py::test_cloak_parity_vs_diffheat``.

Model (trial T-A4, ``docs/applications.md`` §4):
unit square ``[0,L]²``, left Dirichlet hot (1), right Dirichlet cold (0),
top/bottom insulated. Diffusivity field ``kappa(x,y)`` built from two ring
parameters over a static radius map::

    kappa = inner  (r < r_in)
            ring   (r_in <= r < r_out)
            bg     (else)

RHS mirrors diffheat exactly: ``dT/dt = kappa*(L_T + b_source)`` (field
multiply, not the full ``∇·(κ∇T)`` divergence form — documented gap; the
full form is a pull-driven H3 extension).

Whole rollout is jittable/differentiable in ``(kappa_inner, kappa_ring)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .forced_conv import apply_bc_2d, laplacian_2d

DEFAULT_L = 1.0
DEFAULT_BG = 0.01
DEFAULT_R_IN = 0.20
DEFAULT_R_OUT = 0.40
DEFAULT_T_HOT = 1.0
DEFAULT_T_COLD = 0.0

CLOAK_METRIC_KEYS = (
    "interior_grad_norm",
    "far_field_mismatch",
    "kappa_ring",
    "kappa_inner",
)


def max_timestep_cloak(dx: float, dy: float, kappa_max: float) -> float:
    """2-D diffusion limit with field max (mirrors ``check_cfl_2d``)."""
    k = float(kappa_max)
    if k <= 0:
        return float("inf")
    return min(dx * dx, dy * dy) / (4.0 * k)


def check_timestep_cloak(dx: float, dy: float, kappa_max: float, dt: float) -> bool:
    return bool(dt <= max_timestep_cloak(dx, dy, kappa_max))


def make_grids_cloak(L: float, nx: int, ny: int):
    dx = L / nx
    dy = L / ny
    xc = (jnp.arange(nx, dtype=jnp.float64) + 0.5) * dx
    yc = (jnp.arange(ny, dtype=jnp.float64) + 0.5) * dy
    X, Y = jnp.meshgrid(xc, yc, indexing="ij")
    R = jnp.sqrt((X - L / 2.0) ** 2 + (Y - L / 2.0) ** 2)
    return X, Y, R, float(dx), float(dy)


def kappa_field_from_rings(R, *, kappa_inner, kappa_ring, bg: float = DEFAULT_BG,
                           r_in: float = DEFAULT_R_IN, r_out: float = DEFAULT_R_OUT):
    """Build ``(nx, ny)`` diffusivity field (tracer-safe in ring params)."""
    return jnp.where(R < r_in, kappa_inner, jnp.where(R < r_out, kappa_ring, bg))


def simulate_cloak(
    *,
    kappa_inner=0.1,
    kappa_ring=1.0,
    bg: float = DEFAULT_BG,
    L: float = DEFAULT_L,
    nx: int = 16,
    ny: int = 16,
    dt: float = 0.001,
    n_steps: int = 50,
    r_in: float = DEFAULT_R_IN,
    r_out: float = DEFAULT_R_OUT,
    t_hot: float = DEFAULT_T_HOT,
    t_cold: float = DEFAULT_T_COLD,
    collect_trace: bool = False,
):
    """Roll out cloak transient (explicit Euler in scan, grad-safe)."""
    nx, ny, n_steps = int(nx), int(ny), int(n_steps)
    X, Y, R, dx, dy = make_grids_cloak(float(L), nx, ny)
    kappa = kappa_field_from_rings(R, kappa_inner=kappa_inner, kappa_ring=kappa_ring,
                                   bg=float(bg), r_in=float(r_in), r_out=float(r_out))
    T_init = jnp.zeros((nx, ny), dtype=jnp.float64)

    def step_fn(T, _):
        L_raw = laplacian_2d(T, dx, dy)
        L_T, b = apply_bc_2d(
            L_raw, T, dx, dy,
            bc_left=("dirichlet", float(t_hot)),
            bc_right=("dirichlet", float(t_cold)),
            bc_bottom=("neumann", 0.0),
            bc_top=("neumann", 0.0),
        )
        return T + dt * kappa * (L_T + b), None

    try:
        kmax = float(jnp.max(kappa))
        if not check_timestep_cloak(dx, dy, kmax, float(dt)):
            import logging as _logging  # noqa: WPS433
            _logging.getLogger(__name__).warning(
                "dt=%g exceeds cloak CFL (dx=%g kmax=%g)", float(dt), dx, kmax)
    except Exception:
        pass

    T_final, _ = jax.lax.scan(step_fn, T_init, None, length=n_steps)
    out: dict = {"final": T_final, "kappa": kappa, "X": X, "Y": Y, "R": R, "dx": dx, "dy": dy}
    if collect_trace:
        def step_emit(T, _):
            L_raw = laplacian_2d(T, dx, dy)
            L_T, b = apply_bc_2d(
                L_raw, T, dx, dy,
                bc_left=("dirichlet", float(t_hot)),
                bc_right=("dirichlet", float(t_cold)),
                bc_bottom=("neumann", 0.0),
                bc_top=("neumann", 0.0),
            )
            Tn = T + dt * kappa * (L_T + b)
            return Tn, Tn
        _, traj = jax.lax.scan(step_emit, T_init, None, length=n_steps)
        out["trajectory"] = jnp.concatenate([T_init[jnp.newaxis], traj], axis=0)
    return out


def cloak_metrics(T_final, kappa_field, X, Y, R, dx: float, dy: float, L: float,
                  kappa_inner, kappa_ring,
                  r_in: float = DEFAULT_R_IN, r_out: float = DEFAULT_R_OUT):
    """Cloak objectives from the final field (all JAX, grad-safe).

    - ``interior_grad_norm``: mean |grad T| inside ``r_in`` (want ~0).
    - ``far_field_mismatch``: mean (T - T_linear)^2 outside ``r_out``
      with ``T_linear = 1 - x/L`` (want ~0).
    """
    dTdx = (jnp.roll(T_final, -1, axis=0) - jnp.roll(T_final, 1, axis=0)) / (2.0 * dx)
    dTdy = (jnp.roll(T_final, -1, axis=1) - jnp.roll(T_final, 1, axis=1)) / (2.0 * dy)
    grad_norm = jnp.sqrt(dTdx * dTdx + dTdy * dTdy)
    interior = R < r_in
    exterior = R >= r_out
    safe_in = jnp.maximum(jnp.sum(interior), 1)
    safe_out = jnp.maximum(jnp.sum(exterior), 1)
    interior_grad = jnp.sum(jnp.where(interior, grad_norm, 0.0)) / safe_in
    T_linear = 1.0 - X / L
    mismatch = jnp.sum(jnp.where(exterior, (T_final - T_linear) ** 2, 0.0)) / safe_out
    return {
        "interior_grad_norm": interior_grad,
        "far_field_mismatch": mismatch,
        "kappa_ring": kappa_ring,
        "kappa_inner": kappa_inner,
    }


__all__ = [
    "CLOAK_METRIC_KEYS",
    "check_timestep_cloak",
    "cloak_metrics",
    "kappa_field_from_rings",
    "make_grids_cloak",
    "max_timestep_cloak",
    "simulate_cloak",
]
