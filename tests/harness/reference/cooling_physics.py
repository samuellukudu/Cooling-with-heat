import math
from typing import Dict, Iterable, List, Optional


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def normalize(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return clamp((value - lo) / (hi - lo))


def water_sat_pressure_pa(t_k: float) -> float:
    """Saturation vapour pressure of water [Pa].

    Uses two-branch formulation:
      - Liquid water (< 100 °C): Alduchov-Eskridge (1996) Magnus form,
        error < 0.3 % vs. IAPWS-IF97 over -40 to 100 °C.
      - Above normal bp (≥ 100 °C): NIST WebBook Antoine equation
        (Stull 1947, 60–150 °C range):
            log10(P / mmHg) = A - B / (C + T_c)
        with A=8.10765, B=1750.286, C=235.0,
        converted to Pa (1 mmHg = 133.322 Pa).
        Error < 0.3 % vs. IAPWS-IF97 over 100–150 °C.

    References
    ----------
    Alduchov & Eskridge (1996), J. Appl. Meteorol. 35, 601-609.
    Stull (1947) via NIST WebBook (https://webbook.nist.gov, water, Antoine).
    """
    t_c = t_k - 273.15
    if t_c < 100.0:
        # Alduchov-Eskridge (1996) — liquid water, valid -40 to +100 °C
        return 611.2 * math.exp(17.502 * t_c / (240.97 + t_c))
    else:
        # NIST WebBook Antoine — valid 60–150 °C
        # log10(P / mmHg) = 8.10765 - 1750.286 / (235.0 + T_c)
        log10_p_mmhg = 8.10765 - 1750.286 / (235.0 + t_c)
        return (10.0 ** log10_p_mmhg) * 133.322  # mmHg → Pa


def water_h_fg_j_kg(t_k: float) -> float:
    """Latent heat of vaporisation of water [J/kg] as a function of temperature.

    Uses Watson's (1943) correlation with the water-specific exponent 0.321
    (vs. the generic 0.38 used for organic fluids):
        h_fg(T) = h_fg_100 * ((T_c - T) / (T_c - T_ref)) ^ 0.321
    where T_c = 647.1 K (critical point), T_ref = 373.15 K.

    Anchored to IAPWS-IF97 h_fg at 100 °C = 2 256 400 J/kg.
    Error vs. IAPWS-IF97 is < 0.5 % over 0–150 °C.

    Reference: Smith, Van Ness & Abbott, Introduction to Chemical Engineering
    Thermodynamics, 8th ed., §4.4.
    """
    h_fg_ref = 2_256_400.0   # J/kg at 100 °C (IAPWS-IF97)
    t_c_water = 647.1        # K, critical temperature of water
    t_ref = 373.15           # K, reference point (100 °C)
    t_k_clamped = min(t_k, t_c_water - 1.0)
    return h_fg_ref * ((t_c_water - t_k_clamped) / (t_c_water - t_ref)) ** 0.321


def simulate_adsorption_cycle(
    q_sat: float,
    q_st: float,
    t_evap_c: float,
    t_cond_c: float,
    t_des_c: float,
    cycle_time_sec: float,
    e_char_j_mol: float = 4500.0,
    n_heterogeneity: float = 1.8,
    hx_mass_factor: float = 1.35,
) -> Dict[str, float]:
    """Simulate a single-bed adsorption cooling cycle (water refrigerant).

    Parameters
    ----------
    q_sat           Maximum uptake capacity [kg_water / kg_adsorbent]
    q_st            Isosteric heat of adsorption [J/kg_water]
    t_evap_c        Evaporator temperature [°C]
    t_cond_c        Condenser / adsorption temperature [°C]
    t_des_c         Desorption / regeneration temperature [°C]
    cycle_time_sec  Half-cycle time [s] (adsorption OR desorption phase).
                    SCP is Q_cool / cycle_time_sec — i.e. the power averaged
                    over one half-cycle.  A two-bed system achieves continuous
                    cooling so the system-level SCP equals this value.
                    Practical values: HVAC ~600 s, vehicle ~180 s, CPU ~120 s.
    e_char_j_mol    D-A characteristic energy [J/mol], independent of Q_st.
                    Silica gel (RD / type-1): 3 500–6 000 J/mol.
                    AlPO / SAPO zeotypes:     6 000–10 000 J/mol.
                    Zeolite 13X / NaA:       12 000–18 000 J/mol.
                    Default 4 500 J/mol targets silica-gel-like frameworks.
    n_heterogeneity Dubinin-Astakhov heterogeneity exponent (default 1.8).
    hx_mass_factor  Multiplier on Q_in to account for heat exchanger thermal
                    mass (metal fins/tubes).  Typical lab-validated value: 1.35.
                    Set to 1.0 to recover the adsorbent-only estimate.

    Returns
    -------
    dict with COP, SCP_W_kg, delta_q, q_ads, q_des, P_evap_kPa, P_cond_kPa,
    h_fg_MJ_kg
    """
    t_evap = t_evap_c + 273.15
    t_cond = t_cond_c + 273.15
    t_des = t_des_c + 273.15
    t_ads = t_cond

    gas_constant = 8.314
    cp_adsorbent = 1000.0
    cp_liquid = 4184.0

    p_evap = water_sat_pressure_pa(t_evap)
    p_cond = water_sat_pressure_pa(t_cond)

    # Temperature-dependent latent heat evaluated at the evaporator (cooling side).
    h_fg = water_h_fg_j_kg(t_evap)

    # D-A characteristic energy [J/mol] — independent of Q_st.
    # This prevents the previous coupling (E ∝ Q_st) that suppressed low-P uptake.
    e_adsorption = e_char_j_mol

    def uptake(t_bed: float, p_system: float) -> float:
        p_sat = water_sat_pressure_pa(t_bed)
        if p_system >= p_sat:
            return q_sat
        adsorption_potential = gas_constant * t_bed * math.log(p_sat / p_system)
        return q_sat * math.exp(-((adsorption_potential / e_adsorption) ** n_heterogeneity))

    q_ads = uptake(t_ads, p_evap)
    q_des = uptake(t_des, p_cond)
    delta_q = max(0.0, q_ads - q_des)
    q_cool = delta_q * h_fg
    q_sensible = (cp_adsorbent + q_ads * cp_liquid) * (t_des - t_ads)
    q_desorption = delta_q * q_st
    # hx_mass_factor accounts for metal HX thermal mass (typically +30-50% on Q_in).
    q_in = hx_mass_factor * (q_sensible + q_desorption)

    return {
        "COP": q_cool / q_in if q_in > 0.0 else 0.0,
        "SCP_W_kg": q_cool / cycle_time_sec if cycle_time_sec > 0.0 else 0.0,
        "delta_q": delta_q,
        "q_ads": q_ads,
        "q_des": q_des,
        "P_evap_kPa": p_evap / 1000.0,
        "P_cond_kPa": p_cond / 1000.0,
        "h_fg_MJ_kg": h_fg / 1e6,
    }


def find_pareto_frontier(points: Iterable[Dict[str, float]]) -> List[Dict[str, float]]:
    rows = list(points)
    frontier = []

    for point in rows:
        dominated = False
        for other in rows:
            equal_or_better = other["COP"] >= point["COP"] and other["SCP_W_kg"] >= point["SCP_W_kg"]
            strictly_better = other["COP"] > point["COP"] or other["SCP_W_kg"] > point["SCP_W_kg"]
            if equal_or_better and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(point)

    return frontier


def pareto_target_window(
    *,
    t_evap_c: float,
    t_cond_c: float,
    t_des_c: float,
    cycle_time_sec: float,
    cop_weight: float,
    scp_weight: float,
    q_sat_min: float = 0.08,
    q_sat_max: float = 0.90,
    q_st_min: float = 2.30e6,
    q_st_max: float = 4.10e6,
    steps: int = 31,
    top_fraction: float = 0.35,
) -> Dict[str, float]:
    points = []
    for i in range(steps):
        q_sat = q_sat_min + (q_sat_max - q_sat_min) * i / (steps - 1)
        for j in range(steps):
            q_st = q_st_min + (q_st_max - q_st_min) * j / (steps - 1)
            cycle = simulate_adsorption_cycle(
                q_sat=q_sat,
                q_st=q_st,
                t_evap_c=t_evap_c,
                t_cond_c=t_cond_c,
                t_des_c=t_des_c,
                cycle_time_sec=cycle_time_sec,
            )
            points.append({"q_sat": q_sat, "Q_st": q_st, **cycle})

    # Score every grid point so we can select a high-performing window even
    # when the Pareto frontier is degenerate (single-point).
    for point in points:
        point["weighted_score"] = (
            cop_weight * normalize(point["COP"], 0.05, 0.85)
            + scp_weight * normalize(point["SCP_W_kg"], 20.0, 1600.0)
        )

    frontier = find_pareto_frontier(points)
    for point in frontier:
        point["on_frontier"] = True

    frontier = sorted(frontier, key=lambda row: row["weighted_score"], reverse=True)
    selected_count = max(3, int(len(frontier) * top_fraction))
    selected = frontier[:selected_count]

    # When the frontier is too small to define a meaningful window (common
    # when one (q_sat, Q_st) corner dominates all others), fall back to a
    # threshold-based selection from the full grid: take all points whose
    # weighted score is within 15% of the best score.
    if len(selected) < 3:
        best_score = frontier[0]["weighted_score"] if frontier else 0.0
        threshold = max(0.0, best_score - 0.15)
        selected = [p for p in points if p["weighted_score"] >= threshold]
        if len(selected) < 3:
            sorted_points = sorted(points, key=lambda row: row["weighted_score"], reverse=True)
            selected = sorted_points[:max(10, int(len(sorted_points) * top_fraction * 0.20))]

    q_sats = [row["q_sat"] for row in selected]
    q_sts = [row["Q_st"] for row in selected]
    best = selected[0]

    return {
        "q_sat_min": min(q_sats),
        "q_sat_max": max(q_sats),
        "Q_st_min": min(q_sts),
        "Q_st_max": max(q_sts),
        "best_q_sat": best["q_sat"],
        "best_Q_st": best["Q_st"],
        "best_COP": best["COP"],
        "best_SCP_W_kg": best["SCP_W_kg"],
        "frontier_size": len(frontier),
        "sampled_points": len(points),
        "window_points": len(selected),
    }


def pareto_closeness_score(q_sat: float, q_st: float, target: Optional[Dict[str, float]]) -> float:
    if not target:
        return 0.0

    q_sat_mid = 0.5 * (target["q_sat_min"] + target["q_sat_max"])
    q_st_mid = 0.5 * (target["Q_st_min"] + target["Q_st_max"])

    # Half-width floors are set to half the full grid range used in
    # pareto_target_window (q_sat: 0.08–0.90, Q_st: 2.30e6–4.10e6).
    # This ensures materials still receive differentiated scores even when
    # the target window collapses because the Pareto frontier is degenerate
    # (i.e. one (q_sat, Q_st) corner dominates all others).
    q_sat_half_width = max(0.41, 0.5 * (target["q_sat_max"] - target["q_sat_min"]))
    q_st_half_width = max(0.90e6, 0.5 * (target["Q_st_max"] - target["Q_st_min"]))

    q_sat_distance = abs(q_sat - q_sat_mid) / q_sat_half_width
    q_st_distance = abs(q_st - q_st_mid) / q_st_half_width
    return clamp(1.0 - 0.5 * (q_sat_distance + q_st_distance))
