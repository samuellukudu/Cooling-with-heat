"""V3 — the oracle-limit keystone (DESIGN §9/§12 H1.3).

The dynamic bed, run in its equilibrium limit (kinetic time τ_kin = 1/k_LDF
≈ 0.2 s against 600 s phases, strong wall coupling h = 2000 W/m²K so the
bed tracks the fluid, 2 mm bed ≪ the cycle diffusion length), must
reproduce the frozen equilibrium oracle ``Cycle0D`` within 2 % on COP and
SCP.

What the residual gap *is*: the oracle treats the isosteric swing as
instantaneous and books the desorption sensible heat at the frozen
adsorption-phase uptake (``q_ads·c_pl·ΔT``); the dynamic bed follows the
coupled trajectory ``∫ q*(T, P_cond) dT``. The difference —
``c_pl·(∫q* dT − q_ads·ΔT)`` — is a documented convention gap of ~1 % for
this configuration, inside the 2 % gate.

"k_LDF → ∞" is taken as τ_kin ≪ cycle time (the split scheme's two-way
coupling bound caps k_LDF·Δt ≲ 0.25 — see physics/bed1d.py); convergence
of the metrics in k_LDF is asserted directly (k = 5 vs k = 2.5 s⁻¹), along
with convergence in dt.
"""

import numpy as np
import pytest

from harness.envs import Bed1D  # noqa: F401  (ensures env registrations run)
from harness.envs.cycle0d import Cycle0D
from harness.physics.bed1d import simulate_bed

MATERIAL = "anchor:Silica gel RD"
PROFILE = "cpu"  # t_evap 18 °C, t_cond 35 °C, t_des 75 °C
T_SWITCH_S = 600.0  # long half-cycle: the oracle's regime
BED = dict(
    q_sat_kg_kg=0.35,
    q_st_j_kg=2.5e6,
    e_char_j_mol=4500.0,
    n_da=1.8,
    rho_s_kg_m3=600.0,
    c_s_j_kg_k=1000.0,
    c_pl_j_kg_k=4184.0,
    k_eff_w_m_k=0.3,
    h_wall_w_m2_k=2000.0,
    L_m=0.002,
    n_cells=16,
    hx_mass_factor=1.0,  # bare bed: oracle correspondence is bed-only accounting
    t_evap_c=18.0,
    t_cond_c=35.0,
    t_f_ads_c=35.0,
    t_f_des_c=75.0,
)


def _bed_metrics(k_ldf, dt_s, n_cycles=4):
    out = simulate_bed(
        **BED,
        k_ldf_s_1=k_ldf,
        t_ads_s=T_SWITCH_S,
        t_des_s=T_SWITCH_S,
        dt_s=dt_s,
        n_cycles=n_cycles,
    )
    return {k: float(v) for k, v in out["summary"].items()}


def test_v3_oracle_limit():
    # hx = 1 on both sides: bare-bed vs bare-oracle accounting (hx is
    # rig-level metal inventory, not part of the bed↔oracle correspondence).
    oracle = Cycle0D(MATERIAL, PROFILE).evaluate(
        {"cycle_time_s": T_SWITCH_S, "hx_mass_factor": 1.0})
    bed = _bed_metrics(k_ldf=5.0, dt_s=0.015)

    print(f"\nV3: oracle COP {oracle['COP']:.4f} SCP {oracle['SCP_W_kg']:.1f} | "
          f"bed COP {bed['COP']:.4f} SCP {bed['SCP_W_kg']:.1f}")
    for key in ("COP", "SCP_W_kg"):
        gap = abs(bed[key] - oracle[key]) / oracle[key]
        assert gap < 0.02, f"{key}: bed {bed[key]:.4f} vs oracle {oracle[key]:.4f} (gap {gap:.2%})"


def test_v3_converges_in_k_ldf():
    """The equilibrium limit is reached: halving k_LDF (τ_kin 0.2 → 0.4 s,
    both ≪ the 600 s phase) moves the metrics by well under the gate."""
    fast = _bed_metrics(k_ldf=5.0, dt_s=0.015, n_cycles=2)
    slow = _bed_metrics(k_ldf=2.5, dt_s=0.015, n_cycles=2)
    for key in ("COP", "SCP_W_kg"):
        drift = abs(fast[key] - slow[key]) / fast[key]
        assert drift < 0.005, f"{key}: k=5 {fast[key]:.4f} vs k=2.5 {slow[key]:.4f} (drift {drift:.2%})"


def test_v3_converges_in_dt():
    """Numerics convergence: halving dt moves the metrics negligibly."""
    coarse = _bed_metrics(k_ldf=5.0, dt_s=0.015, n_cycles=2)
    fine = _bed_metrics(k_ldf=5.0, dt_s=0.0075, n_cycles=2)
    for key in ("COP", "SCP_W_kg"):
        drift = abs(coarse[key] - fine[key]) / fine[key]
        assert drift < 0.005, f"{key}: dt=0.015 {coarse[key]:.4f} vs dt=0.0075 {fine[key]:.4f} (drift {drift:.2%})"
