"""Differentiable (PyTorch) mirror of ``cooling_physics.simulate_adsorption_cycle``.

This is the single differentiable implementation of the cycle model. It
replicates the canonical NumPy physics exactly — Magnus/Antoine saturation
pressure (two branches), Watson latent heat, Dubinin-Astakhov uptake with an
independent characteristic energy, and the heat-exchanger mass factor — so
any gradient computed here corresponds to the same numbers the screeners use.

A consistency check against ``cooling_physics`` runs in ``__main__``; keep it
passing whenever either model changes.

Note: for the current ML roadmap this module stays PyTorch-only as a
gradient-verification tool. A JAX port happens only if guided design starts.
"""

import torch

from cooling_physics import simulate_adsorption_cycle

GAS_CONSTANT = 8.314          # J/(mol*K)
CP_ADSORBENT = 1000.0         # J/(kg*K)
CP_LIQUID = 4184.0            # J/(kg*K)


def water_sat_pressure_pa(t_k: torch.Tensor) -> torch.Tensor:
    """Two-branch saturation pressure of water [Pa] (see cooling_physics)."""
    t_c = t_k - 273.15
    magnus = 611.2 * torch.exp(17.502 * t_c / (240.97 + t_c))       # < 100 C
    log10_p_mmhg = 8.10765 - 1750.286 / (235.0 + t_c)               # >= 100 C
    antoine = (10.0 ** log10_p_mmhg) * 133.322
    return torch.where(t_c < 100.0, magnus, antoine)


def water_h_fg_j_kg(t_k: torch.Tensor) -> torch.Tensor:
    """Watson correlation for the latent heat of water [J/kg]."""
    h_fg_ref = 2_256_400.0   # at 373.15 K
    t_crit = 647.1           # K
    t_ref = 373.15           # K
    t_clamped = torch.clamp(t_k, max=t_crit - 1.0)
    return h_fg_ref * ((t_crit - t_clamped) / (t_crit - t_ref)) ** 0.321


def differentiable_hvac_cycle(
    q_sat: torch.Tensor,
    Q_st: torch.Tensor,
    t_evap_c: float = 7.0,
    t_cond_c: float = 35.0,
    t_des_c: float = 80.0,
    cycle_time_sec: float = 600.0,
    e_char_j_mol: float = 4500.0,
    n_heterogeneity: float = 1.8,
    hx_mass_factor: float = 1.35,
) -> dict:
    """Fully differentiable adsorption cycle; mirrors cooling_physics.

    The uptake branch ``P_system >= P_sat`` needs no conditional: clamping the
    pressure ratio at 1.0 makes the D-A formula return exactly ``q_sat`` there
    while keeping gradients clean (log(1) = 0).
    """
    t_evap = torch.tensor(t_evap_c + 273.15, dtype=q_sat.dtype)
    t_cond = torch.tensor(t_cond_c + 273.15, dtype=q_sat.dtype)
    t_des = torch.tensor(t_des_c + 273.15, dtype=q_sat.dtype)
    t_ads = t_cond

    p_evap = water_sat_pressure_pa(torch.tensor(t_evap_c + 273.15, dtype=q_sat.dtype))
    p_cond = water_sat_pressure_pa(torch.tensor(t_cond_c + 273.15, dtype=q_sat.dtype))
    h_fg = water_h_fg_j_kg(torch.tensor(t_evap_c + 273.15, dtype=q_sat.dtype))

    def uptake(t_bed: torch.Tensor, p_system: torch.Tensor) -> torch.Tensor:
        p_sat = water_sat_pressure_pa(t_bed)
        ratio = torch.clamp(p_sat / p_system, min=1.0)
        adsorption_potential = GAS_CONSTANT * t_bed * torch.log(ratio)
        return q_sat * torch.exp(-((adsorption_potential / e_char_j_mol) ** n_heterogeneity))

    q_ads = uptake(t_ads, p_evap)
    q_des = uptake(t_des, p_cond)
    delta_q = torch.clamp(q_ads - q_des, min=0.0)

    q_cool = delta_q * h_fg
    q_sensible = (CP_ADSORBENT + q_ads * CP_LIQUID) * (t_des - t_ads)
    q_in = hx_mass_factor * (q_sensible + delta_q * Q_st)

    cop = q_cool / q_in
    scp = q_cool / cycle_time_sec if cycle_time_sec > 0 else torch.zeros(())
    return {"COP": cop, "SCP_W_kg": scp, "delta_q": delta_q}


# --- Verification and gradient demo ---
if __name__ == "__main__":
    dtype = torch.float64  # match Python-float precision for the cross-check

    probe_points = [
        (0.45, 2.8e6),
        (0.20, 3.6e6),
        (0.70, 2.4e6),
        (0.35, 3.1e6),
        (0.55, 4.0e6),
    ]
    print("--- CONSISTENCY CHECK vs cooling_physics (canonical NumPy) ---")
    worst = 0.0
    for q, qst in probe_points:
        ref = simulate_adsorption_cycle(
            q_sat=q, q_st=qst,
            t_evap_c=7.0, t_cond_c=35.0, t_des_c=80.0,
            cycle_time_sec=600.0,
        )
        got = differentiable_hvac_cycle(
            torch.tensor(q, dtype=dtype), torch.tensor(qst, dtype=dtype),
        )
        cop_err = abs(got["COP"].item() - ref["COP"]) / max(ref["COP"], 1e-12)
        scp_err = abs(got["SCP_W_kg"].item() - ref["SCP_W_kg"]) / max(ref["SCP_W_kg"], 1e-12)
        worst = max(worst, cop_err, scp_err)
        print(f"  q_sat={q:.2f} Q_st={qst/1e6:.2f} MJ/kg | "
              f"COP torch={got['COP'].item():.9f} numpy={ref['COP']:.9f} "
              f"(rel err {cop_err:.2e})")

    assert worst < 1e-12, f"Torch/NumPy models diverged (worst rel err {worst:.2e})"
    print("  PASS — differentiable model matches canonical physics exactly.\n")

    # Gradient demonstration at the nominal point.
    q_sat_pred = torch.tensor(0.45, dtype=dtype, requires_grad=True)
    Q_st_pred = torch.tensor(2.8e6, dtype=dtype, requires_grad=True)

    result = differentiable_hvac_cycle(q_sat_pred, Q_st_pred)
    result["COP"].backward()

    print("--- ANALYTICAL SYSTEM GRADIENTS (d COP / d material property) ---")
    print(f"COP: {result['COP'].item():.4f}")
    print(f"d(COP)/d(q_sat): {q_sat_pred.grad.item():+.4f}")
    print(f"d(COP)/d(Q_st) : {Q_st_pred.grad.item():+.4e}")
