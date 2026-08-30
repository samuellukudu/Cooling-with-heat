"""Water properties and Dubinin–Astakhov uptake — exact JAX mirror of
``Materials/cooling_physics.py``.

Validation V1 (``../DESIGN.md`` §9) pins this module to the canonical NumPy
implementation with max relative error < 1e-12 on a dense temperature grid.
The expressions below therefore mirror the canonical operation order exactly
— do not "simplify" them.

Correlations (identical to the canonical module):
- Saturation pressure: Alduchov–Eskridge (1996) Magnus form below 100 °C,
  NIST WebBook Antoine above; error < 0.3 % vs IAPWS-IF97.
- Latent heat: Watson (1943) with the water-specific exponent 0.321,
  anchored to 2 256 400 J/kg at 100 °C; error < 0.5 % over 0–150 °C.
- Equilibrium uptake: Dubinin–Astakhov with an independent characteristic
  energy and the Polanyi potential ``A = R·T·ln(P_sat/P)`` clamped at zero
  (``q = q_sat`` when ``P ≥ P_sat``).

CoolProp is a *reference*, not a hot path (root commit e6ca51f): parity
against it runs in tests via the optional ``cool`` extra.
"""

from __future__ import annotations

import jax.numpy as jnp

GAS_CONSTANT = 8.314  # J/(mol·K)
CP_ADSORBENT = 1000.0  # J/(kg·K)
CP_LIQUID = 4184.0  # J/(kg·K)


def water_sat_pressure_pa(t_k):
    """Saturation vapour pressure of water [Pa], two-branch (see module doc)."""
    t_c = t_k - 273.15
    magnus = 611.2 * jnp.exp(17.502 * t_c / (240.97 + t_c))
    log10_p_mmhg = 8.10765 - 1750.286 / (235.0 + t_c)
    antoine = (10.0 ** log10_p_mmhg) * 133.322
    return jnp.where(t_c < 100.0, magnus, antoine)


def water_h_fg_j_kg(t_k):
    """Latent heat of vaporisation of water [J/kg] (Watson, exponent 0.321)."""
    h_fg_ref = 2_256_400.0  # J/kg at 373.15 K (IAPWS-IF97)
    t_crit = 647.1  # K
    t_ref = 373.15  # K
    t_clamped = jnp.minimum(t_k, t_crit - 1.0)
    return h_fg_ref * ((t_crit - t_clamped) / (t_crit - t_ref)) ** 0.321


def da_uptake(t_k, p_pa, q_sat, e_char_j_mol, n_heterogeneity):
    """Dubinin–Astakhov equilibrium uptake [kg/kg].

    ``q = q_sat`` when ``p_pa ≥ P_sat(t)``: the Polanyi potential is clamped
    at zero via the ratio floor, which keeps the branch branchless (and the
    gradients clean) while returning exactly the canonical value.
    """
    p_sat = water_sat_pressure_pa(t_k)
    ratio = jnp.maximum(p_sat / p_pa, 1.0)
    potential = GAS_CONSTANT * t_k * jnp.log(ratio)
    return q_sat * jnp.exp(-((potential / e_char_j_mol) ** n_heterogeneity))
