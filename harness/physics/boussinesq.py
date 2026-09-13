"""Boussinesq cavity — resolved 2-D Rayleigh–Bénard solver (T-A1 extension).

The correlation model (``natural_conv.py``) cannot answer resolved-flow
questions (plume structure, onset dynamics, geometry beyond a gap). This
module is the nondimensional Boussinesq system in the unit square on a
MAC grid (u on x-faces, v on y-faces, p/T at centers) with Chorin-style
artificial compressibility (fully explicit — no Poisson solve, so the
whole rollout is one ``jax.lax.scan``)::

    u_t + (u·∇)u = −∇p + Pr·∇²u
    v_t + (u·∇)v = −∇p + Pr·∇²v + Ra·Pr·T
    p_t + c²·∇·u = 0,            mean(p) pinned to 0
    T_t + (u·∇)T = ∇²T

Hot bottom (T=1) / cold top (T=0) Dirichlet, insulated sides, no-slip
walls. Advection is first-order upwind (stable past cell-Pe 2 at the
cost of numerical diffusion — the validation bands own that); diffusion
is 5-point central; wall ghosts are mirror/antisymmetric exact.

Units: L=1, α=1 (time in diffusion units), ΔT=1. The env maps physical
(fluid, ΔT, L) to (Ra, Pr) so both T-A1 levels share one design
language. ``c²``/``dt`` follow :func:`cfl_params` (acoustic + diffusive
limits with safety factors); ``Ra`` may be a tracer (all ``jnp`` ops),
``c²``/``dt``/``N`` are static.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

BOUSSINESQ_METRIC_KEYS = (
    "Nu",
    "Nu_bot",
    "Nu_top",
    "u_max",
    "Ra",
    "delta_T_K",
    "in_validated",
)

#: Validated laminar envelope (N ≥ 24, Ra ≤ 3e5, air-like Pr): steady,
#: bot/top-converged, literature-banded. Outside it the solver still
#: runs (explicit scheme, no hidden switches) but metrics are unpinned.
RA_VALID_HI = 3e5
N_VALID_LO = 24

#: Bulk-viscosity (grad-div) coefficient: ``divb = DIVB_COEF·c·dx`` damps
#: acoustic/divergent modes in ~1 transit time while barely touching the
#: physical (div≈0) flow. Standing acoustic modes driven by buoyancy
#: spin-up survive ``DIVB_COEF = 0.1`` at Ra ≳ 3e4 (observed: frozen
#: checkerboard, Nu stuck at 1); 0.3 kills them with margin inside the
#: explicit bulk limit ``DIVB·π²·(c·dt/dx) < 2`` at acoustic safety 0.4.
DIVB_COEF = 0.3


def cfl_params(Ra_hi: float, Pr: float, dx: float, safety: float = 0.4) -> tuple[float, float]:
    """Static timestep for the worst case (highest Ra) + reference sound speed.

    Returns ``(c2_hi, dt)`` with ``c = 3·sqrt(Ra_hi·Pr)`` (free-fall scale
    with margin) and ``dt = safety·min(dx/c, 0.2·dx²/max(Pr, 1))``. The
    timestep is static (scan length); the sound speed used in a rollout
    is matched per-evaluation, ``c² = 9·Ra·Pr`` (tracer-safe) — running
    low-Ra cases at the high-Ra ``c²`` makes acoustic transients
    violently overshoot (observed: 4-order pressure pile-up → NaN).
    """
    import math  # noqa: WPS433
    c = 3.0 * math.sqrt(float(Ra_hi) * float(Pr))
    dt = safety * min(dx / c, 0.2 * dx * dx / max(float(Pr), 1.0))
    return c * c, dt


def _y_upwind(u, vf, ug, dx):
    """dudy at interior x-faces, upwind in vf with antisymmetric ghosts."""
    um = ug[1:-1, 1:-1]          # == u[1:-1, :]
    ulo = ug[1:-1, :-2][:, : um.shape[1]]  # j-1 (ghost at j=0)
    uhi = ug[1:-1, 2:]           # j+1 (ghost at j=n-1)
    bwd = (um - ulo) / dx
    fwd = (uhi - um) / dx
    return jnp.where(vf > 0, bwd, fwd)


def _x_upwind_v(v, uf, vg, dx):
    """dvdx at interior y-faces, upwind in uf with antisymmetric ghosts."""
    vm = vg[1:-1, 1:-1]          # == v[:, 1:-1]
    vlo = vg[:-2, 1:-1]          # i-1 (ghost at i=0)
    vhi = vg[2:, 1:-1]           # i+1 (ghost at i=n-1)
    bwd = (vm - vlo) / dx
    fwd = (vhi - vm) / dx
    return jnp.where(uf > 0, bwd, fwd)


def step_fields(u, v, p, T, *, Ra, Pr, c2, dx, dt):
    """One explicit MAC step. All arrays concrete-shaped; Ra may be a tracer."""
    n = T.shape[0]
    # -- center velocities --
    uc = 0.5 * (u[:-1, :] + u[1:, :])          # (n, n)
    vc = 0.5 * (v[:, :-1] + v[:, 1:])          # (n, n)
    # -- divergence + pressure (mean-pinned) --
    div = (u[1:, :] - u[:-1, :]) / dx + (v[:, 1:] - v[:, :-1]) / dx
    p_new = p - dt * c2 * div
    p_new = p_new - jnp.mean(p_new)
    # -- bulk-viscosity damping of divergent modes (grad-div) --
    c_sound = jnp.sqrt(c2)
    divb = DIVB_COEF * c_sound * dx
    # grad(div) at interior faces (unused wall faces stay out):
    gdiv_u = (div[1:, :] - div[:-1, :]) / dx          # (n-1, n)
    gdiv_v = (div[:, 1:] - div[:, :-1]) / dx          # (n, n-1)
    # -- padded T (ghost Dirichlet top/bottom, mirror Neumann sides) --
    Tg_bot = 2.0 * 1.0 - T[:, 0]
    Tg_top = 2.0 * 0.0 - T[:, -1]
    P = jnp.zeros((n + 2, n + 2), dtype=T.dtype)
    P = P.at[1:-1, 1:-1].set(T)
    P = P.at[1:-1, 0].set(Tg_bot)
    P = P.at[1:-1, -1].set(Tg_top)
    P = P.at[0, :].set(P[1, :])
    P = P.at[-1, :].set(P[-2, :])
    Cc = P[1:-1, 1:-1]
    lap_T = (P[2:, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] + P[1:-1, :-2]
             - 4.0 * Cc) / (dx * dx)
    dTdx = jnp.where(uc > 0, (Cc - P[:-2, 1:-1]) / dx,
                     (P[2:, 1:-1] - Cc) / dx)
    dTdy = jnp.where(vc > 0, (Cc - P[1:-1, :-2]) / dx,
                     (P[1:-1, 2:] - Cc) / dx)
    T_new = T + dt * (lap_T - uc * dTdx - vc * dTdy)
    # -- u update (interior x-faces; wall faces stay 0) --
    # v at u-faces: plain 4-average (all referenced v-faces exist).
    vf = 0.25 * (v[:-1, :-1] + v[:-1, 1:] + v[1:, :-1] + v[1:, 1:])
    ug = jnp.concatenate([-u[:, 0:1], u, -u[:, -1:]], axis=1)  # y-ghosts
    lap_u = ((u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / (dx * dx)
             + (ug[1:-1, 2:] - 2.0 * u[1:-1, :] + ug[1:-1, :-2]) / (dx * dx))
    dudx = jnp.where(u[1:-1, :] > 0,
                     (u[1:-1, :] - u[:-2, :]) / dx,
                     (u[2:, :] - u[1:-1, :]) / dx)
    dudy = _y_upwind(u, vf, ug, dx)
    gradp_u = (p[1:, :] - p[:-1, :]) / dx
    u_new = u.at[1:-1, :].set(
        u[1:-1, :] + dt * (-(u[1:-1, :] * dudx + vf * dudy) - gradp_u
                           + Pr * lap_u + divb * gdiv_u))
    # -- v update (interior y-faces; wall faces stay 0) --
    uf = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
    vg = jnp.concatenate([-v[0:1, :], v, -v[-1:, :]], axis=0)  # x-ghosts
    lap_v = ((vg[2:, 1:-1] - 2.0 * v[:, 1:-1] + vg[:-2, 1:-1]) / (dx * dx)
             + (v[:, 2:] - 2.0 * v[:, 1:-1] + v[:, :-2]) / (dx * dx))
    dvdx = _x_upwind_v(v, uf, vg, dx)
    dvdy = jnp.where(v[:, 1:-1] > 0,
                     (v[:, 1:-1] - v[:, :-2]) / dx,
                     (v[:, 2:] - v[:, 1:-1]) / dx)
    gradp_v = (p[:, 1:] - p[:, :-1]) / dx
    Tv = 0.5 * (T[:, :-1] + T[:, 1:])
    v_new = v.at[:, 1:-1].set(
        v[:, 1:-1] + dt * (-(uf * dvdx + v[:, 1:-1] * dvdy) - gradp_v
                           + Pr * lap_v + Ra * Pr * Tv + divb * gdiv_v))
    return u_new, v_new, p_new, T_new


def wall_nusselt(T, dx):
    """Bottom/top Nusselt from ghost-face gradients (ΔT=1, L=1, k=1)."""
    nu_bot = jnp.mean(2.0 * (1.0 - T[:, 0]) / dx)
    nu_top = jnp.mean(2.0 * (T[:, -1] - 0.0) / dx)
    return nu_bot, nu_top


def simulate_boussinesq(
    *,
    Ra=1e4,
    Pr=0.707,
    n_cells: int = 24,
    c2: float | None = None,
    dt: float | None = None,
    t_end: float = 0.5,
    Ra_hi: float = 3e5,
    collect_trace: bool = False,
):
    """Roll out the RB cavity (explicit MAC in scan; Ra may be a tracer).

    ``dt`` defaults from :func:`cfl_params` at ``(Ra_hi, Pr)`` (static —
    scan length); ``c2`` defaults to the per-evaluation matched value
    ``9·Ra·Pr`` (tracer-safe — see :func:`cfl_params` for why the
    high-Ra ``c²`` must NOT be reused at low Ra). Returns final fields
    + ``Nu`` metrics.
    """
    n = int(n_cells)
    dx = 1.0 / n
    if dt is None:
        _, dt = cfl_params(Ra_hi, float(Pr), dx)
    else:
        dt = float(dt)
    c2 = 9.0 * Ra * float(Pr) if c2 is None else c2
    n_steps = int(round(float(t_end) / float(dt)))
    u = jnp.zeros((n + 1, n))
    v = jnp.zeros((n, n + 1))
    p = jnp.zeros((n, n))
    yc = (jnp.arange(n, dtype=jnp.float64) + 0.5) * dx
    xc = (jnp.arange(n, dtype=jnp.float64) + 0.5) * dx
    X, Y = jnp.meshgrid(xc, yc, indexing="ij")
    # Deterministic asymmetric seed (no RNG — reproducible): breaks the
    # conduction symmetry so convection onsets promptly above Ra_c.
    T = (jnp.tile((1.0 - yc)[None, :], (n, 1))
         + 1e-3 * jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y))

    def step_fn(carry, _):
        u_, v_, p_, T_ = carry
        # No float() on c2/dt here: c2 may be a tracer (per-eval matched
        # sound speed); step_fields is pure jnp and takes either.
        return step_fields(u_, v_, p_, T_, Ra=Ra, Pr=float(Pr),
                           c2=c2, dx=dx, dt=dt), None

    (u_f, v_f, p_f, T_f), _ = jax.lax.scan(step_fn, (u, v, p, T),
                                           None, length=n_steps)
    nu_bot, nu_top = wall_nusselt(T_f, dx)
    umax = jnp.maximum(jnp.max(jnp.abs(u_f)), jnp.max(jnp.abs(v_f)))
    nu = 0.5 * (nu_bot + nu_top)
    out: dict = {"Nu": nu, "Nu_bot": nu_bot, "Nu_top": nu_top,
                 "u_max": umax, "Ra": Ra, "u": u_f, "v": v_f,
                 "p": p_f, "T": T_f, "dx": dx, "dt": dt, "c2": c2,
                 "n_steps": n_steps}
    if collect_trace:
        def step_emit(carry, _):
            carry_next, _ = step_fn(carry, None)
            return carry_next, carry_next[3]
        (_, _, _, _), traj = jax.lax.scan(step_emit, (u, v, p, T),
                                          None, length=n_steps)
        out["trajectory"] = traj
    return out


__all__ = [
    "BOUSSINESQ_METRIC_KEYS",
    "cfl_params",
    "simulate_boussinesq",
    "step_fields",
    "wall_nusselt",
]
