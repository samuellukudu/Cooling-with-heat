"""Equilibrium adsorption cycle — exact JAX mirror of
``Materials/cooling_physics.simulate_adsorption_cycle`` (Env-0 oracle).

Every output is a JAX scalar with gradients flowing through it; the numbers
are pinned to the canonical NumPy implementation by validation V1
(``../DESIGN.md`` §9, rel. err. < 1e-12). The dynamic bed model must
degenerate to this function in its equilibrium limit (V3).

The canonical function carries the physical documentation (D–A uptake with
independent characteristic energy, HX thermal-mass factor, SCP convention);
only the branch handling differs here: divisions are guarded with
``jnp.where`` so the function stays differentiable and ``jit``-safe.
"""

from __future__ import annotations

import jax.numpy as jnp

from .thermo import (
    CP_ADSORBENT,
    CP_LIQUID,
    da_uptake,
    water_h_fg_j_kg,
    water_sat_pressure_pa,
)

# Metric keys of the cycle model — mirrored verbatim from the canonical
# dict in cooling_physics.simulate_adsorption_cycle (DESIGN §5.1).
METRIC_KEYS = (
    "COP",
    "SCP_W_kg",
    "delta_q",
    "q_ads",
    "q_des",
    "P_evap_kPa",
    "P_cond_kPa",
    "h_fg_MJ_kg",
)


def simulate_cycle(
    q_sat,
    q_st,
    t_evap_c,
    t_cond_c,
    t_des_c,
    cycle_time_sec,
    e_char_j_mol=4500.0,
    n_heterogeneity=1.8,
    hx_mass_factor=1.35,
):
    """Differentiable single-bed equilibrium cycle (water refrigerant).

    Parameters mirror ``cooling_physics.simulate_adsorption_cycle`` exactly:
    ``q_sat`` [kg/kg], ``q_st`` [J/kg], temperatures [°C], ``cycle_time_sec``
    = half-cycle time [s], ``e_char_j_mol`` [J/mol], ``n_heterogeneity``
    [–], ``hx_mass_factor`` [–]. Returns a dict of JAX scalars.
    """
    t_evap = t_evap_c + 273.15
    t_cond = t_cond_c + 273.15
    t_des = t_des_c + 273.15
    t_ads = t_cond

    p_evap = water_sat_pressure_pa(t_evap)
    p_cond = water_sat_pressure_pa(t_cond)
    h_fg = water_h_fg_j_kg(t_evap)

    q_ads = da_uptake(t_ads, p_evap, q_sat, e_char_j_mol, n_heterogeneity)
    q_des = da_uptake(t_des, p_cond, q_sat, e_char_j_mol, n_heterogeneity)
    delta_q = jnp.maximum(0.0, q_ads - q_des)

    q_cool = delta_q * h_fg
    q_sensible = (CP_ADSORBENT + q_ads * CP_LIQUID) * (t_des - t_ads)
    q_in = hx_mass_factor * (q_sensible + delta_q * q_st)

    safe_q_in = jnp.where(q_in > 0.0, q_in, 1.0)
    cop = jnp.where(q_in > 0.0, q_cool / safe_q_in, 0.0)
    safe_cycle = jnp.where(cycle_time_sec > 0.0, cycle_time_sec, 1.0)
    scp = jnp.where(cycle_time_sec > 0.0, q_cool / safe_cycle, 0.0)

    return {
        "COP": cop,
        "SCP_W_kg": scp,
        "delta_q": delta_q,
        "q_ads": q_ads,
        "q_des": q_des,
        "P_evap_kPa": p_evap / 1000.0,
        "P_cond_kPa": p_cond / 1000.0,
        "h_fg_MJ_kg": h_fg / 1e6,
    }
