"""Telegrapher 1-D physics — hyperbolic heat mirror (T-E08 hardening).

Mirrors the frozen ``diffheat`` 1-D telegrapher hot path
(``laplacian_1d`` + 1-D ghost-cell BC correction +
``solve_telegrapher_1d`` leapfrog-style coefficients and Taylor bootstrap)
without importing it. Parity pinned by
``tests/harness/test_telegrapher_trial.py`` (< 1e-4; same dtype note as
the T-A5/T-A4 parity tests).

Equation: ``tau·u_tt + u_t = alpha·u_xx`` (+ Gaussian pulse initial
conditions; zero Dirichlet ends). The scheme ::

    u^{n+1} = a·u^n + b·u^{n-1} + c·(alpha·∇²u^n),   a/b/c from (tau, dt)

with ``u^1`` from the second-order Taylor bootstrap. Whole rollout is one
``jax.lax.scan`` — jittable and differentiable in the initial-condition
amplitudes (the cancellation discovery axis).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

TELEGRAPHER_METRIC_KEYS = (
    "sensor_mse",
    "sensor_peak",
    "cancellation_pct",
    "secondary_amplitude",
)

DEFAULT_L = 1.0
DEFAULT_ALPHA = 1.0
DEFAULT_TAU = 0.5


def laplacian_1d(u: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Centered 3-point Laplacian via roll (periodic; BC-corrected below)."""
    return (jnp.roll(u, -1) + jnp.roll(u, 1) - 2.0 * u) / (dx * dx)


def apply_bc_1d(L_raw: jnp.ndarray, u: jnp.ndarray, dx: float,
                left: tuple[str, float] = ("dirichlet", 0.0),
                right: tuple[str, float] = ("dirichlet", 0.0)):
    """Ghost-cell BC correction (mirrors diffheat 1-D boundary)."""
    n = u.shape[0]
    L_T, b = L_raw, jnp.zeros_like(u)
    for edge, idx in ((left, 0), (right, n - 1)):
        kind, val = edge
        nbr = u[1] if idx == 0 else u[n - 2]
        wrap = u[n - 1] if idx == 0 else u[0]
        incorrect = (wrap + nbr - 2.0 * u[idx]) / (dx * dx)
        if kind == "dirichlet":
            correct = (nbr - 3.0 * u[idx]) / (dx * dx)
            L_T = L_T.at[idx].add(correct - incorrect)
            b = b.at[idx].add(2.0 * val / (dx * dx))
        else:
            correct = (nbr - u[idx]) / (dx * dx)
            L_T = L_T.at[idx].add(correct - incorrect)
            b = b.at[idx].add(-val / dx)
    return L_T, b


def max_timestep_1d(dx: float, alpha: float, tau: float) -> float:
    """Wave CFL ``dt ≤ dx/c``, ``c = sqrt(alpha/tau)`` (mirrors diffheat)."""
    c = (float(alpha) / float(tau)) ** 0.5
    return float(dx) / c


def check_timestep_1d(dx: float, alpha: float, tau: float, dt: float) -> bool:
    return bool(dt <= max_timestep_1d(dx, alpha, tau))


def gaussian(x: jnp.ndarray, center: float, width2: float = 0.0005) -> jnp.ndarray:
    return jnp.exp(-((x - center) ** 2) / width2)


def simulate_telegrapher_1d(
    *,
    u0,
    alpha: float = DEFAULT_ALPHA,
    tau: float = DEFAULT_TAU,
    L: float = DEFAULT_L,
    n_cells: int = 120,
    dt: float = 0.001,
    n_steps: int = 250,
    collect_trace: bool = False,
):
    """Roll out ``tau·u_tt + u_t = alpha·u_xx`` (``u0`` may carry tracers)."""
    n_cells, n_steps = int(n_cells), int(n_steps)
    dx = float(L) / n_cells
    xc = (jnp.arange(n_cells, dtype=jnp.float64) + 0.5) * dx
    u0 = jnp.asarray(u0, dtype=jnp.float64)
    v0 = jnp.zeros_like(u0)

    try:
        if not check_timestep_1d(dx, float(alpha), float(tau), float(dt)):
            import logging as _logging  # noqa: WPS433
            _logging.getLogger(__name__).warning("dt=%g exceeds telegrapher CFL", float(dt))
    except Exception:
        pass

    def rhs(u, t):
        del t
        L_raw = laplacian_1d(u, dx)
        L_T, b = apply_bc_1d(L_raw, u, dx)
        return alpha * (L_T + b)

    denom = tau + dt / 2.0
    a, b = 2.0 * tau / denom, (dt / 2.0 - tau) / denom
    c = dt ** 2 / denom
    u_tt0 = (rhs(u0, 0.0) - v0) / tau
    u_curr = u0 + dt * v0 + 0.5 * dt ** 2 * u_tt0
    u_prev = u0

    def step_fn(carry, idx):
        up, uc = carry
        u_next = a * uc + b * up + c * rhs(uc, (idx + 1) * dt)
        return (uc, u_next), (u_next if collect_trace else None)

    (_, u_final), traj = jax.lax.scan(step_fn, (u_prev, u_curr), jnp.arange(n_steps - 1))
    out: dict = {"xc": xc, "dx": dx, "final": u_final}
    if collect_trace:
        out["trajectory"] = jnp.concatenate([u0[jnp.newaxis], u_curr[jnp.newaxis], traj], axis=0)
    return out


__all__ = [
    "TELEGRAPHER_METRIC_KEYS",
    "apply_bc_1d",
    "check_timestep_1d",
    "gaussian",
    "laplacian_1d",
    "max_timestep_1d",
    "simulate_telegrapher_1d",
]
