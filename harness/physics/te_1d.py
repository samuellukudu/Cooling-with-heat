"""Segmented thermoelectric leg — 1-D two-field PDE (T-A2 extension).

The lumped pair model (``thermoelectric.py``) cannot answer spatially
resolved questions (segment grading, interface placement). This module
is the 1-D series leg: uniform current density ``J = I/A`` (series —
no elliptic solve needed), segmented Seebeck/conductivity fields, and
the heat equation with Joule + Thomson sources::

    rho_c·∂T/∂t = ∂/∂x(k·∂T/∂x) + J²/σ(x) − T·J·dS/dx

Boundary total-energy fluxes carry the Peltier term
(``−k·∇T + S·T·J``), so ``Qc``/``Qh``/``W_in`` close the books.
Per-leg ``S, σ, k`` derive from the pair rows at the reference
geometry (``LEG_A = 1 mm²``, ``LEG_L = 1 mm``, pair = 2 legs series/
parallel): ``S = S_pair/2``, ``σ = 2L/(RA)``, ``k = KL/(2A)`` —
plausible module-scale values, documented, no new tables. ``rho_c`` is
a fixed default (steady metrics don't depend on it). Pair metrics
(``Qc``, ``W_in``) are 2× the simulated leg; ``COP``/``ZT`` are
scale-invariant.

Uniform-leg limit reproduces the lumped model (parity gate, V3-style).
Whole rollout is one ``jax.lax.scan`` — jittable, differentiable in
current and segment fraction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .thermoelectric import TE_PAIRS, resolve_te_pair

TE1D_METRIC_KEYS = (
    "Qc_W",
    "COP",
    "delta_T_K",
    "ZT_cold",
    "I_opt_cold_A",
    "out_of_window",
)

LEG_A_M2 = 1e-6
LEG_L_M = 1e-3
RHO_C_J_M3_K = 1.2e6  # fixed (steady metrics independent of it)


def leg_sigma_k(pair_name: str) -> tuple[float, float, float]:
    """Per-leg (S, sigma, k) from a pair row at the reference geometry.

    Pair = 2 legs (electrically series, thermally parallel): per-leg
    ``S = S_pair/2``, ``sigma = 2·L/(R·A)``, ``k = K·L/(2·A)``.
    """
    p = resolve_te_pair(pair_name)
    S = p["S_V_K"] / 2.0
    sigma = 2.0 * LEG_L_M / (p["R_ohm"] * LEG_A_M2)
    k = p["K_W_K"] * LEG_L_M / (2.0 * LEG_A_M2)
    return S, sigma, k


def max_timestep(dx: float, k_max: float) -> float:
    """Explicit diffusion limit ``dt ≤ dx²/(2α)``."""
    alpha = k_max / RHO_C_J_M3_K
    return dx * dx / (2.0 * alpha)


def check_timestep(dx: float, k_max: float, dt: float) -> bool:
    return bool(dt <= max_timestep(dx, k_max))


def simulate_te_1d(
    *,
    Tc_C: float = 18.0,
    Th_C: float = 35.0,
    current_A=3.0,
    pair_cold: str = "Bi2Te3",
    pair_hot: str = "Bi2Te3",
    seg_frac: float = 0.5,
    n_cells: int = 16,
    dt: float = 0.001,
    n_steps: int = 3000,
    collect_trace: bool = False,
):
    """Roll out the segmented leg (explicit Euler in scan, grad-safe).

    Segment A (cold fraction ``seg_frac``) uses ``pair_cold``, the rest
    ``pair_hot``. ``current_A``/``seg_frac`` may be tracers.
    """
    n_cells, n_steps = int(n_cells), int(n_steps)
    dx = LEG_L_M / n_cells
    xc = (jnp.arange(n_cells, dtype=jnp.float64) + 0.5) * dx
    S_a, sig_a, k_a = leg_sigma_k(pair_cold)
    S_b, sig_b, k_b = leg_sigma_k(pair_hot)
    cold_mask = xc < seg_frac * LEG_L_M
    S = jnp.where(cold_mask, S_a, S_b)
    sig = jnp.where(cold_mask, sig_a, sig_b)
    kap = jnp.where(cold_mask, k_a, k_b)

    try:
        if not check_timestep(dx, float(jnp.max(kap)), float(dt)):
            import logging as _logging  # noqa: WPS433
            _logging.getLogger(__name__).warning("dt=%g exceeds TE1D CFL", float(dt))
    except Exception:
        pass

    J = current_A / LEG_A_M2
    Tc, Th = Tc_C + 273.15, Th_C + 273.15
    T_init = jnp.linspace(Tc, Th, n_cells)

    def rhs(T):
        # Conduction with ghost Dirichlet ends (half-cell faces).
        Tg0 = 2.0 * Tc - T[0]
        Tg1 = 2.0 * Th - T[-1]
        Tp = jnp.concatenate([jnp.reshape(Tg0, (1,)), T, jnp.reshape(Tg1, (1,))])
        # Face conductivities (harmonic mean — exact for layered media).
        kf = 2.0 * kap[:-1] * kap[1:] / (kap[:-1] + kap[1:])
        q_r = -jnp.concatenate([kf * (T[1:] - T[:-1]) / dx,
                                jnp.reshape(kap[-1] * (Tg1 - T[-1]) / dx, (1,))])
        q_l = -jnp.concatenate([jnp.reshape(kap[0] * (T[0] - Tg0) / dx, (1,)),
                                kf * (T[1:] - T[:-1]) / dx])
        div = (q_l - q_r) / dx
        joule = J * J / sig
        # dS/dx one-sided at the Dirichlet ends (central differences would
        # wrap the segment jump to the boundaries as spurious sources).
        dSdx = jnp.concatenate([
            jnp.reshape((S[1] - S[0]) / dx, (1,)),
            (S[2:] - S[:-2]) / (2.0 * dx),
            jnp.reshape((S[-1] - S[-2]) / dx, (1,))])
        thomson = -T * J * dSdx
        return (div + joule + thomson) / RHO_C_J_M3_K

    def step_fn(T, _):
        return T + dt * rhs(T), None

    T_final, _ = jax.lax.scan(step_fn, T_init, None, length=n_steps)

    # Boundary total-energy fluxes (conduction + Peltier), outward +.
    Tg0 = 2.0 * Tc - T_final[0]
    Tg1 = 2.0 * Th - T_final[-1]
    qc_dens = -kap[0] * (T_final[0] - Tg0) / dx + S[0] * T_final[0] * J
    qh_dens = -kap[-1] * (Tg1 - T_final[-1]) / dx + S[-1] * T_final[-1] * J
    Qc = qc_dens * LEG_A_M2
    Qh = qh_dens * LEG_A_M2
    # Voltage drop: E = J/σ + S·∇T integrated. End gradients match the
    # ghost-cell flux stencil (half-cell faces).
    dTdx = jnp.concatenate([
        jnp.reshape(2.0 * (T_final[0] - Tc) / dx, (1,)),
        (T_final[2:] - T_final[:-2]) / (2.0 * dx),
        jnp.reshape(2.0 * (Th - T_final[-1]) / dx, (1,))])
    E = J / sig + S * dTdx
    Vdrop = jnp.sum(E) * dx
    W_in = current_A * Vdrop
    # Pair metrics = 2 legs (series/parallel); COP is scale-invariant.
    Qc, Qh, W_in = 2.0 * Qc, 2.0 * Qh, 2.0 * W_in
    useful = Qc > 0.0
    cop = jnp.where(useful & (W_in > 1e-12), Qc / jnp.maximum(W_in, 1e-12), 0.0)
    Qc_out = jnp.where(useful, Qc, 0.0)
    T_avg = (Tc + Th) / 2.0
    zt_c = S_a * S_a * T_avg * sig_a / k_a
    # Pair-level I_opt for guidance: S_pair·Tc/R_pair.
    i_opt = (2.0 * S_a) * Tc / (2.0 * LEG_L_M / (sig_a * LEG_A_M2))
    oow = jnp.where((Th_C > resolve_te_pair(pair_cold)["Th_max_C"])
                    | (Th_C > resolve_te_pair(pair_hot)["Th_max_C"]), 1.0, 0.0)
    out: dict = {"Qc_W": Qc_out, "COP": cop,
                 "delta_T_K": jnp.asarray(Th_C - Tc_C), "ZT_cold": zt_c,
                 "I_opt_cold_A": i_opt, "out_of_window": oow,
                 "Qh_W": Qh, "W_in_W": W_in, "T_final": T_final}
    if collect_trace:
        def step_emit(T, _):
            Tn = T + dt * rhs(T)
            return Tn, Tn
        _, traj = jax.lax.scan(step_emit, T_init, None, length=n_steps)
        out["trajectory"] = jnp.concatenate([T_init[jnp.newaxis], traj], axis=0)
    return out


__all__ = [
    "LEG_A_M2",
    "LEG_L_M",
    "RHO_C_J_M3_K",
    "TE1D_METRIC_KEYS",
    "check_timestep",
    "leg_sigma_k",
    "max_timestep",
    "simulate_te_1d",
]
