"""Absorption-chiller cycle physics — lumped single-effect LiBr/H2O-class model.

Trial T-A3 (``docs/applications.md`` §3 — the scenario flagged most likely
to return inside the ML pipeline as bed-level modeling behind the
surrogate). This is a *lumped* model, not a PDE: Generator → Condenser →
Evaporator → Absorber at steady point conditions, exactly the model class
the applications note says needs ODE support. The harness poses it as a
static Problem; dynamics/schedules are a pull-driven extension.

Model (ideal 3-temperature chiller, Novikov form)::

    COP_ideal = ((T_gen - T_cond) / T_gen) * (T_evap / (T_cond - T_evap))

(engine × refrigerator Carnot factors, Kelvin), derated by the pair
effectiveness ``eps`` (real irreversibilities), with a linear
concentration-spread capacity::

    dx = clip(a*(t_gen - t_cond) - b*(t_cond - t_evap), 0, dx_max)
    Q_evap = dx * h_fg(T_evap)          [J per kg solution]
    COP    = eps * COP_ideal   (0 when dx == 0 — no circulation, no cooling)
    Q_gen  = Q_evap / COP
    Q_gen_total = Q_gen + ua_loss*(t_gen - t_amb)*cycle_time
    SCP    = Q_evap / cycle_time        [W per kg solution]

``h_fg`` reuses :mod:`harness.physics.thermo` (V1-parity correlations).
``ua_loss`` (default 0 — idealized textbook cycle) is an optional
structural knob for heat-loss-aware experiments; ``t_amb_c`` likewise.

Calibration anchor (V-ABS gate): single-effect LiBr/H2O at
t_evap=7 °C / t_cond=30 °C / t_gen=85 °C gives COP ≈ 0.70
(``eps=0.37`` → ideal 1.87 × 0.37 = 0.69). Textbook band ±15 %.

Everything is ``jnp``-only and tracer-safe in ``t_gen_c`` (grad flows;
``clip``/``where`` give subgradients at the dx=0 kink — the honest
band-bottom degeneracy the H2.2 experiments also surfaced).
"""

from __future__ import annotations

import jax.numpy as jnp

from .thermo import water_h_fg_j_kg

ABSORPTION_METRIC_KEYS = (
    "COP",
    "SCP_W_kg",
    "Q_evap_J_kg",
    "Q_gen_J_kg",
    "delta_x",
    "COP_ideal",
    "effectiveness",
)

# Working-pair table (cf. anchor table for adsorption): effectiveness plus
# the linear-spread coefficients. NH3-H2O trades peak efficiency for a
# wider low-grade band (higher SCP than LiBr at 60 °C regeneration —
# the T-A3 ranking signal, mirroring the H2.3 13X finding).
WORKING_PAIRS: dict[str, dict[str, float]] = {
    "LiBr-H2O": {"eps": 0.37, "a": 0.00145, "b": 0.00085, "dx_max": 0.12,
                 "notes": 0.0},  # solar/HVAC standard
    "NH3-H2O": {"eps": 0.30, "a": 0.00200, "b": 0.00060, "dx_max": 0.15,
                "notes": 0.0},  # low-grade tolerant
    "LiCl-H2O": {"eps": 0.40, "a": 0.00120, "b": 0.00090, "dx_max": 0.10,
                 "notes": 0.0},  # high-eff, narrow band
}

DEFAULT_PAIR = "LiBr-H2O"
T_AMB_C = 25.0


def resolve_pair(pair: str = DEFAULT_PAIR) -> dict[str, float]:
    """Case-insensitive pair lookup (``LiBr``, ``libr-h2o`` also match)."""
    key = str(pair).strip()
    for name, params in WORKING_PAIRS.items():
        if key.lower() == name.lower() or key.lower().replace("_", "-") == name.lower():
            return {"name": name, **params}
    short = key.lower().replace("-h2o", "").replace("h2o", "")
    for name, params in WORKING_PAIRS.items():
        if name.lower().startswith(short) and short:
            return {"name": name, **params}
    raise KeyError(f"unknown working pair {pair!r}; available: {sorted(WORKING_PAIRS)}")


def cop_ideal_3t(t_evap_c, t_cond_c, t_gen_c):
    """Ideal 3-temperature COP (JAX scalars, grad-safe; 0 outside the physical wedge)."""
    Te = t_evap_c + 273.15
    Tc = t_cond_c + 273.15
    Tg = t_gen_c + 273.15
    engine = (Tg - Tc) / Tg
    fridge = Te / jnp.maximum(Tc - Te, 1e-6)
    ideal = engine * fridge
    valid = jnp.logical_and(Tg > Tc, Tc > Te)
    return jnp.where(valid, jnp.maximum(ideal, 0.0), 0.0)


def concentration_spread(pair_params, t_evap_c, t_cond_c, t_gen_c):
    """Linear spread clipped to ``[0, dx_max]`` (0 = degenerate band bottom)."""
    raw = pair_params["a"] * (t_gen_c - t_cond_c) - pair_params["b"] * (t_cond_c - t_evap_c)
    return jnp.clip(raw, 0.0, pair_params["dx_max"])


def simulate_absorption(
    *,
    t_evap_c: float = 7.0,
    t_cond_c: float = 30.0,
    t_gen_c=85.0,
    pair: str = DEFAULT_PAIR,
    cycle_time_s: float = 600.0,
    ua_loss_W_K_per_kg: float = 0.0,
    t_amb_c: float = T_AMB_C,
):
    """Point-cycle evaluation (all outputs JAX scalars; ``t_gen_c`` may be a tracer)."""
    params = resolve_pair(pair)
    cop_id = cop_ideal_3t(t_evap_c, t_cond_c, t_gen_c)
    dx = concentration_spread(params, t_evap_c, t_cond_c, t_gen_c)
    h_fg = water_h_fg_j_kg(t_evap_c + 273.15)
    q_evap = dx * h_fg
    has_flow = dx > 0.0
    cop_base = jnp.where(has_flow, params["eps"] * cop_id, 0.0)
    safe_cop = jnp.where(cop_base > 1e-12, cop_base, 1.0)
    q_gen = jnp.where(has_flow, q_evap / safe_cop, 0.0)
    q_loss = ua_loss_W_K_per_kg * jnp.maximum(t_gen_c - t_amb_c, 0.0) * cycle_time_s
    q_total = q_gen + q_loss
    cop = jnp.where(q_total > 0.0, q_evap / jnp.maximum(q_total, 1e-12), 0.0)
    scp = q_evap / cycle_time_s
    return {
        "COP": cop,
        "SCP_W_kg": scp,
        "Q_evap_J_kg": q_evap,
        "Q_gen_J_kg": q_total,
        "delta_x": dx,
        "COP_ideal": cop_id,
        "effectiveness": jnp.asarray(params["eps"]),
    }


__all__ = [
    "ABSORPTION_METRIC_KEYS",
    "DEFAULT_PAIR",
    "T_AMB_C",
    "WORKING_PAIRS",
    "concentration_spread",
    "cop_ideal_3t",
    "resolve_pair",
    "simulate_absorption",
]
