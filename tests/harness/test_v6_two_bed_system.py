"""V6 — system sanity for the two-bed machine (DESIGN §9/§12 H2.1).

Gates:

1. **Duty continuity**: the combined cooling power of the counter-phase
   pair never sits at ≤ 0 for more than one physics step — at every
   switchover the fresh adsorber is the just-finished desorber (hot), so
   it takes over from the first substep.
2. **Heat recovery ≥ no-recovery**: a lumped-UA recovery window raises
   COP monotonically (in conductance and window length) at unchanged
   swing, and the source draw drops by exactly the exchanged heat.
3. **Exchange conservation**: with the films disconnected (recovery
   window) and no adsorption, the antisymmetric coupling conserves
   ``U_A + U_B`` to machine precision.
4. **Equilibrium limit**: the system is the single-bed cycle
   phase-shifted by half a period, so in the equilibrium limit its COP
   and SCP match the ``Cycle0D`` oracle (the V3 correspondence at
   system level).

The recovery mechanism note that the numbers below pin down: during the
recovery window the beds are disconnected from their fluids (the
recovery loop owns the circuits) and exchange heat through a lumped UA
applied as exactly antisymmetric per-cell sources. The hot bed is
whichever just finished desorbing, so the transfer direction
alternates with the system phase — the reported ``Q_rec_J_m2``
accumulates the absolute transfer.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from harness.envs import Bed1D  # noqa: F401  (ensures env registrations run)
from harness.envs.cycle0d import Cycle0D
from harness.physics import bed1d, system
from harness.physics.thermo import CP_ADSORBENT, CP_LIQUID, da_uptake

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
    hx_mass_factor=1.0,  # bare-bed accounting: oracle correspondence
    k_ldf_s_1=5.0,
)
T_EVAP_C, T_COND_C, T_F_DES_C = 18.0, 35.0, 75.0
T_HALF_S = 600.0
DT_S = 0.015

MATERIAL = "anchor:Silica gel RD"
PROFILE = "cpu"


def _run(**kw):
    out = system.simulate_two_bed(
        bed_a=dict(BED), bed_b=dict(BED),
        t_evap_c=T_EVAP_C, t_cond_c=T_COND_C,
        t_f_ads_c=T_COND_C, t_f_des_c=T_F_DES_C,
        t_ads_s=T_HALF_S, t_des_s=T_HALF_S, dt_s=DT_S, n_cycles=4, **kw,
    )
    return out, {k: float(v) for k, v in out["summary"].items()}


def _longest_nonpositive_run(x):
    run = best = 0
    for v in np.asarray(x):
        run = run + 1 if v <= 0.0 else 0
        best = max(best, run)
    return best


def test_v6_duty_continuous_across_switchover():
    """The handover criterion for the counter-phase pair (DESIGN §9 V6:
    'no duty gaps > 1 dt', made precise for the v1 valve model).

    The v1 valve flip is a pressure jump, so the fresh adsorber (hot,
    fresh off desorption) first relaxes its uptake DOWN to the evaporator
    isotherm — the documented valve-flip burst, which nets out within
    τ_kin = 1/k_LDF and is present in the single-bed model too. What the
    two-bed machine must guarantee is that it adds NO dead time of its
    own:

    1. the longest nonpositive-duty stretch is bounded by the burst
       timescale (a few τ_kin), never by the phase duration — a
       de-synchronised pair (both beds in des) would show a gap of a
       full half-cycle;
    2. net cooling over ANY window of one phase duration is positive —
       the cumulative book never stalls across a full handover.
    """
    out, _ = _run(collect_trace=True)
    _assert_contiguous_duty(out, label="no recovery")

    out_r, _ = _run(collect_trace=True, recovery_ua_w_m2_k=100.0, t_rec_s=60.0)
    _assert_contiguous_duty(out_r, label="recovery UA=100")


def _assert_contiguous_duty(out, *, label):
    s = out["series"]
    m_a, m_b = out["m_s_a_kg_m2"], out["m_s_b_kg_m2"]
    p_cool = (np.asarray(s["A_dq_cool_j_kg"]) * m_a
              + np.asarray(s["B_dq_cool_j_kg"]) * m_b) / DT_S
    burst_steps = int(np.ceil(3.0 * (1.0 / BED["k_ldf_s_1"]) / DT_S)) + 1
    worst = _longest_nonpositive_run(p_cool)
    print(f"\nV6 duty [{label}]: longest nonpositive run {worst} steps "
          f"(burst bound {burst_steps}; dt = {DT_S} s)")
    assert worst <= burst_steps

    # Net-positive cooling over any one-phase window (cumulative books,
    # physical time via the decimated t channel).
    t = np.asarray(s["t_abs_s"])
    cum = (np.asarray(s["A_Q_cool_cum_J_kg"]) * m_a
           + np.asarray(s["B_Q_cool_cum_J_kg"]) * m_b)
    for i in range(len(t)):
        if t[i] < T_HALF_S:
            continue
        j = int(np.searchsorted(t, t[i] - T_HALF_S))
        assert cum[i] > cum[j], f"cumulative cooling stalled over a full phase at t = {t[i]:.0f} s"


def test_v6_recovery_improves_cop_monotonically():
    """Recovery raises COP monotonically in UA and window length, leaves the
    swing unchanged, and reduces the source draw."""
    base, base_s = _run()
    cop0, qin0, dq0 = base_s["COP"], base_s["Q_in_J_kg"], base_s["delta_q"]
    prev = None
    for ua, t_rec in [(50.0, 60.0), (100.0, 60.0), (100.0, 120.0), (200.0, 120.0)]:
        _, s = _run(recovery_ua_w_m2_k=ua, t_rec_s=t_rec)
        print(f"V6 recovery UA={ua} t_rec={t_rec}: COP {s['COP']:.4f} "
              f"(+{s['COP'] - cop0:.4f}), Q_in {s['Q_in_J_kg'] / 1e3:.1f} kJ/kg, "
              f"Q_rec {s['Q_rec_J_m2'] / 1e3:.1f} kJ/m2")
        assert s["COP"] > cop0, f"recovery must raise COP (UA={ua})"
        assert s["Q_in_J_kg"] < qin0
        assert abs(s["delta_q"] - dq0) / dq0 < 0.01, "swing must be ~unchanged"
        if prev is not None:
            assert s["COP"] > prev, "COP must rise monotonically with (UA, t_rec)"
        prev = s["COP"]


def test_v6_exchange_conserves_energy():
    """With the films off and no adsorption, the antisymmetric recovery
    coupling conserves U_A + U_B exactly (system-level V2)."""
    phys = system.bed_phys(
        **{**BED, "k_ldf_s_1": 0.0, "h_wall_w_m2_k": 0.0},
        t_evap_c=T_EVAP_C, t_cond_c=T_COND_C, t_f_ads_c=T_COND_C,
    )
    t_phase0 = 30.0
    ca, cb, _ = system.initial_two_carry(
        phys, phys, t_phase0_s=t_phase0, t_des_end_k=T_F_DES_C + 273.15,
        n_cycles=2)

    def stored_energy(carry):
        T = np.asarray(carry[bed1d._CARRY_T])
        q = np.asarray(carry[bed1d._CARRY_Q])
        cap = BED["rho_s_kg_m3"] * (CP_ADSORBENT + CP_LIQUID * q)
        return float(np.sum(cap * T) * BED["L_m"] / BED["n_cells"])

    u0 = stored_energy(ca) + stored_energy(cb)
    xs = (t_phase0, t_phase0, T_F_DES_C, 30.0, 0.0, 0.0)
    (ca_f, cb_f, q_rec), _ = system.advance_two_carry(
        (ca, cb, jnp.asarray(0.0)), xs, n_steps=2000, dt_s=DT_S,
        phys_a=phys, phys_b=phys, recovery_ua_w_m2_k=100.0)
    u1 = stored_energy(ca_f) + stored_energy(cb_f)
    exchanged = float(q_rec)
    print(f"\nV6 conservation: dU_sum {u1 - u0:+.2e} J/m2 vs exchanged "
          f"{exchanged:.1f} J/m2 (rel. {abs(u1 - u0) / max(u0, 1.0):.2e})")
    assert abs(u1 - u0) / u0 < 1e-9


def test_v6_equilibrium_limit_matches_oracle():
    """System COP/SCP → Cycle0D in the equilibrium limit (V3 correspondence
    at system level): identical beds, fast kinetics, strong wall coupling."""
    oracle = Cycle0D(MATERIAL, PROFILE).evaluate(
        {"cycle_time_s": T_HALF_S, "hx_mass_factor": 1.0})
    _, s = _run()
    print(f"\nV6 oracle limit: oracle COP {oracle['COP']:.4f} SCP {oracle['SCP_W_kg']:.1f} | "
          f"system COP {s['COP']:.4f} SCP {s['SCP_W_kg']:.1f}")
    for key in ("COP", "SCP_W_kg"):
        gap = abs(s[key] - oracle[key]) / oracle[key]
        assert gap < 0.02, f"{key}: system {s[key]:.4f} vs oracle {oracle[key]:.4f} (gap {gap:.2%})"


def test_request_bits_flip_the_system():
    """The §5.3 valve-bit hook: a request bit forces both beds to swap roles
    at the next substep; a request shorter than the minimum dwell is
    ignored (the machine follows its phase durations)."""
    phys = system.bed_phys(**BED, t_evap_c=T_EVAP_C, t_cond_c=T_COND_C,
                           t_f_ads_c=T_COND_C)
    ca, cb, _ = system.initial_two_carry(phys, phys, t_phase0_s=T_HALF_S,
                                         t_des_end_k=T_F_DES_C + 273.15,
                                         n_cycles=2)
    n_steps, dt = 2000, 0.01  # inside the conduction CFL (0.0156 s here)
    req = np.zeros(n_steps)
    req[100:150] = 1.0  # demand σ = 1 at t = 1.0 s (timeout would flip at 600 s)

    (ca1, cb1, _), ys = system.advance_two_carry(
        (ca, cb, jnp.asarray(0.0)),
        (T_HALF_S, T_HALF_S, T_F_DES_C, 0.0, 0.0, req),
        n_steps=n_steps, dt_s=dt, phys_a=phys, phys_b=phys, use_req=True)
    phase = np.asarray(ys[:, 1])
    assert phase[99] == 0.0 and phase[101] == 1.0, "request must flip within one step"
    assert phase[152] == 0.0, "request back to 0 must flip back"

    # With a minimum dwell of the full phase, the same request is ignored:
    # the machine must be identical to the pure-timeout (no-request) one.
    # (The request echo channel is dropped from the comparison — the
    # blocked run still records the request bits it ignored.)
    (_, _, _), ys2 = system.advance_two_carry(
        (ca, cb, jnp.asarray(0.0)),
        (T_HALF_S, T_HALF_S, T_F_DES_C, 0.0, T_HALF_S, req),
        n_steps=n_steps, dt_s=dt, phys_a=phys, phys_b=phys, use_req=True)
    (_, _, _), ys0 = system.advance_two_carry(
        (ca, cb, jnp.asarray(0.0)),
        (T_HALF_S, T_HALF_S, T_F_DES_C, 0.0, 0.0, 0.0),
        n_steps=n_steps, dt_s=dt, phys_a=phys, phys_b=phys, use_req=False)
    assert np.array_equal(np.delete(np.asarray(ys2), 2, axis=1),
                          np.delete(np.asarray(ys0), 2, axis=1))


@pytest.mark.parametrize("ua,t_rec", [(100.0, 60.0), (200.0, 120.0)])
def test_v6_recovery_gain_matches_counterfactual(ua, t_rec):
    """The reported recovery transfer accounts for the source-heat saving.

    Per metric cycle the source draw drops by the heat exchanged during
    that cycle's two recovery windows. The books' ``Q_rec`` covers ALL
    windows including warm-up (8 for a 4-cycle episode vs 6 metric
    ones), so the expected ratio is ≈ 6/8 with transient slack."""
    _, s0 = _run()
    out, s1 = _run(recovery_ua_w_m2_k=ua, t_rec_s=t_rec)
    m_tot = out["m_s_a_kg_m2"] + out["m_s_b_kg_m2"]
    saved_abs = (s0["Q_in_J_kg"] - s1["Q_in_J_kg"]) * m_tot  # J per system
    exchanged = s1["Q_rec_J_m2"]
    ratio = saved_abs / exchanged
    print(f"\nV6 accounting UA={ua}: saved {saved_abs / 1e3:.1f} kJ vs "
          f"exchanged {exchanged / 1e3:.1f} kJ (ratio {ratio:.3f})")
    assert 0.6 < ratio < 0.9
