"""Natural-convection physics — Nusselt-correlation model (T-A1).

Trial T-A1 (``docs/applications.md`` §1): the Boussinesq PDE system
(energy + momentum + continuity) is posed as its engineering-correlation
reduction — the experiment the harness can run today without a CFD
solver (resolved vorticity-streamfunction simulation stays a pull-driven
H3 head; the adapter contract already carries it — see ``adapters.py``)::

    Ra_L = group·ΔT·L³,   group = g·β/(ν·α)     [1/(K·m³), per fluid]
    Nu   = max(1, 0.069·Ra^(1/3)·Pr^0.074)      Globe–Dropkin form
    q    = Nu·k·ΔT/L                             [W/m²]

Honesty model (mirrors the ``harness/rank.py`` flag pattern):

- ``Ra < 1708`` → pure conduction (``Nu = 1``, the linear onset kink —
  subgradients exist, which is part of the gradient test);
- ``in_range = 1`` iff ``3e5 ≤ Ra ≤ 7e9`` (the Globe–Dropkin validity
  band); outside it the correlation is an extrapolation and flagged;
- fluid properties are representative @300 K film values — film-temperature
  evaluation is the documented H3 gap (same class as the §8.1 transport
  honesty note).

Everything is ``jnp``-only and tracer-safe in ``delta_T_K``.
"""

from __future__ import annotations

import jax.numpy as jnp

NATCONV_METRIC_KEYS = (
    "Nu",
    "Ra",
    "heat_flux_W_m2",
    "delta_T_K",
    "in_range",
)

RA_ONSET = 1708.0
RA_LO, RA_HI = 3e5, 7e9

# Representative fluid rows @300 K film (group, Pr, k, alpha). Provenance
# is explicit; per-temperature property evaluation is the H3 head.
# ``alpha`` serves the resolved Boussinesq level (diffusion-time scale).
FLUIDS: dict[str, dict[str, float]] = {
    "air": {"ra_group": 9.15e7, "Pr": 0.707, "k_W_m_K": 0.0263,
            "alpha_m2_s": 22.5e-6, "notes": 0.0},
    "water": {"ra_group": 2.37e10, "Pr": 5.83, "k_W_m_K": 0.613,
              "alpha_m2_s": 0.143e-6, "notes": 0.0},
    "oil": {"ra_group": 2.0e8, "Pr": 100.0, "k_W_m_K": 0.145,
            "alpha_m2_s": 0.07e-6, "notes": 0.0},
}

DEFAULT_FLUID = "air"


def resolve_fluid(fluid: str = DEFAULT_FLUID) -> dict[str, float]:
    key = str(fluid).strip().lower()
    for name, params in FLUIDS.items():
        if key == name.lower():
            return {"name": name, **params}
    raise KeyError(f"unknown fluid {fluid!r}; available: {sorted(FLUIDS)}")


def nusselt(Ra, Pr):
    """Globe–Dropkin with conduction floor (tracer-safe; kink at onset)."""
    corr = 0.069 * jnp.power(jnp.maximum(Ra, 1e-12), 1.0 / 3.0) * jnp.power(Pr, 0.074)
    return jnp.maximum(1.0, corr)


def regime_of(Ra: float) -> str:
    """Human-readable regime (plain Python — NOT a metric, keep out of Objective)."""
    Ra = float(Ra)
    if Ra < RA_ONSET:
        return "conduction"
    if Ra < RA_LO:
        return "transitional (extrapolated)"
    if Ra <= RA_HI:
        return "convection (correlated)"
    return "beyond-range (extrapolated)"


def simulate_natconv(
    *,
    delta_T_K=10.0,
    L_m: float = 0.1,
    fluid: str = DEFAULT_FLUID,
):
    """Correlation evaluation (JAX scalars; ``delta_T_K`` may be a tracer)."""
    params = resolve_fluid(fluid)
    Ra = params["ra_group"] * delta_T_K * L_m ** 3
    Nu = nusselt(Ra, params["Pr"])
    q = Nu * params["k_W_m_K"] * delta_T_K / L_m
    in_range = jnp.where(jnp.logical_and(Ra >= RA_LO, Ra <= RA_HI), 1.0, 0.0)
    return {
        "Nu": Nu,
        "Ra": Ra,
        "heat_flux_W_m2": q,
        "delta_T_K": delta_T_K,
        "in_range": in_range,
    }


__all__ = [
    "DEFAULT_FLUID",
    "FLUIDS",
    "NATCONV_METRIC_KEYS",
    "RA_HI",
    "RA_LO",
    "RA_ONSET",
    "nusselt",
    "regime_of",
    "resolve_fluid",
    "simulate_natconv",
]
