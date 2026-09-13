"""Thermoelectric-cooler physics — lumped single-pair Peltier model (T-A2).

Trial T-A2 (``docs/applications.md`` §2): the coupled electro-thermal PDE
system (heat + Joule + Thomson, Seebeck-sourced potential) is reduced to
its standard lumped pair model — the experiment the harness can pose
today without a two-field PDE solver (which stays a pull-driven H3 head)::

    Qc   = S·Tc·I − ½·I²·R − K·ΔT        [W per pair]
    W_in = S·I·ΔT + I²·R
    COP  = Qc / W_in          (0 when Qc ≤ 0 — no useful cooling)
    ZT   = S²·T_avg / (R·K)

``S/R/K`` are per-pair module values (contacts baked in — which is why
ΔTmax reads like a packaged module, not a materials ZT; the contact
derate is documented, not hidden). Closed forms the tests pin:

- max-Qc current ``I_opt = S·Tc/R``,
- ``Qcmax = S²·Tc²/(2R) − K·ΔT``,
- ``ΔTmax = S²·Tc²/(2·K·R)`` (Qc = 0 crossing).

Validity windows (``Th_max_C``) are first-class: a pair driven past its
hot-side rating reports ``out_of_window = 1`` (mirroring the
``harness/rank.py`` honesty pattern) — validity-driven selection across
scenarios is part of the materials discovery, not an error.

Everything is ``jnp``-only and tracer-safe in the drive current ``I``.
"""

from __future__ import annotations

import jax.numpy as jnp

TE_METRIC_KEYS = (
    "Qc_W",
    "COP",
    "delta_T_K",
    "ZT_pair",
    "I_opt_A",
    "out_of_window",
)

# Representative module rows (per-pair S/R/K with contacts; NOT datasheet
# values — provenance is explicit so a future contact-resistance head can
# refine them without changing the contract).
TE_PAIRS: dict[str, dict[str, float]] = {
    "Bi2Te3": {"S_V_K": 400e-6, "R_ohm": 0.030, "K_W_K": 0.0035,
               "Th_max_C": 40.0, "notes": 0.0},   # consumer module, room-temp best
    "PbTe": {"S_V_K": 480e-6, "R_ohm": 0.045, "K_W_K": 0.0040,
             "Th_max_C": 300.0, "notes": 0.0},    # mid-temp
    "SiGe": {"S_V_K": 600e-6, "R_ohm": 0.080, "K_W_K": 0.0030,
             "Th_max_C": 600.0, "notes": 0.0},    # high-temp
}

DEFAULT_TE_PAIR = "Bi2Te3"


def resolve_te_pair(pair: str = DEFAULT_TE_PAIR) -> dict[str, float]:
    """Case-insensitive pair lookup (``bi2te3``, ``Bi2Te3`` both match)."""
    key = str(pair).strip().lower().replace("_", "")
    for name, params in TE_PAIRS.items():
        if key == name.lower().replace("_", ""):
            return {"name": name, **params}
    raise KeyError(f"unknown thermoelectric pair {pair!r}; available: {sorted(TE_PAIRS)}")


def simulate_te(
    *,
    Tc_C: float = 18.0,
    Th_C: float = 35.0,
    current_A=3.0,
    pair: str = DEFAULT_TE_PAIR,
):
    """Lumped pair evaluation (JAX scalars; ``current_A`` may be a tracer)."""
    params = resolve_te_pair(pair)
    S, R, K = params["S_V_K"], params["R_ohm"], params["K_W_K"]
    Tc = Tc_C + 273.15
    dT = Th_C - Tc_C
    I = current_A
    qc = S * Tc * I - 0.5 * I * I * R - K * dT
    w_in = S * I * dT + I * I * R
    useful = qc > 0.0
    cop = jnp.where(jnp.logical_and(useful, w_in > 1e-12), qc / jnp.maximum(w_in, 1e-12), 0.0)
    qc_out = jnp.where(useful, qc, 0.0)
    T_avg = (Tc + Th_C + 273.15) / 2.0
    zt = S * S * T_avg / (R * K)
    i_opt = S * Tc / R
    oow = jnp.where(Th_C > params["Th_max_C"], 1.0, 0.0)
    return {
        "Qc_W": qc_out,
        "COP": cop,
        "delta_T_K": jnp.asarray(dT),
        "ZT_pair": zt,
        "I_opt_A": i_opt,
        "out_of_window": oow,
    }


def qcmax_closed_form(*, Tc_C: float, Th_C: float, pair: str = DEFAULT_TE_PAIR) -> float:
    """Analytic max-Qc (for the V-gate pin; plain floats)."""
    import math  # noqa: WPS433
    p = resolve_te_pair(pair)
    S, R, K = p["S_V_K"], p["R_ohm"], p["K_W_K"]
    Tc = Tc_C + 273.15
    return S * S * Tc * Tc / (2.0 * R) - K * (Th_C - Tc_C)


__all__ = [
    "DEFAULT_TE_PAIR",
    "TE_METRIC_KEYS",
    "TE_PAIRS",
    "qcmax_closed_form",
    "resolve_te_pair",
    "simulate_te",
]
