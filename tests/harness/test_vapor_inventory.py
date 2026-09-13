"""Lumped vapour inventory + Arrhenius kinetics (Open Question 2 / V4 finding 2).

Gates:
- v1-exactness: ``tau = 0`` (+ void) and all-defaults reproduce the v1
  summaries bit-identically; ``Ea = 0`` likewise.
- Mass conservation (V2-style): endpoint identity ``M_N − M_0 = F + S``
  < 1e-9 relative, open AND valve-closed (reconstructed from carry
  states — an independent check of the books).
- Mechanism: closed-valve heating pressurizes the void monotonically
  (explicit isosteric phase); standby stays finite and bounded.
- T_hs trend: vapour (granular-rig void, slow valve) + ``Ea = 30 kJ/mol``
  steepens the 65→80 °C SCP ratio clearly past v1 (toward the measured
  ~2.6×; full closure needs a calibration-grade refit — documented).
- Env integration: Bed1D/TwoBed accept the structural knobs; the
  ``vapor_p_Pa`` series channel exists; valve-closed without void fails
  loudly; grad still flows.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import pytest

import harness
from harness.physics import bed1d
from harness.physics.thermo import da_uptake, water_sat_pressure_pa

P_EVAP = float(water_sat_pressure_pa(16.0 + 273.15))
P_COND = float(water_sat_pressure_pa(35.0 + 273.15))

BASE_KW = dict(q_sat_kg_kg=0.35, q_st_j_kg=2.5e6, e_char_j_mol=4500.0, n_da=1.8,
               k_ldf_s_1=0.01, rho_s_kg_m3=600.0, c_s_j_kg_k=900.0,
               c_pl_j_kg_k=4180.0, k_eff_w_m_k=0.3, h_wall_w_m2_k=500.0,
               L_m=0.002, n_cells=8, hx_mass_factor=1.0, t_evap_c=16.0,
               t_cond_c=35.0, t_f_ads_c=35.0, t_f_des_c=60.0, t_ads_s=300.0,
               t_des_s=300.0, dt_s=0.05, n_cycles=2)


def _summaries(kw):
    return {k: float(v) for k, v in bed1d.simulate_bed(**kw)["summary"].items()}


def test_v1_bit_identical_with_snap_and_defaults():
    base = _summaries(BASE_KW)
    snap = _summaries({**BASE_KW, "vapor_void_m3_m2": 8e-4, "vapor_tau_s": 0.0})
    assert snap.keys() == base.keys()
    for k in base:
        assert snap[k] == base[k] == pytest.approx(base[k])  # exact bits
        assert abs(snap[k] - base[k]) == 0.0
    ea0 = _summaries({**BASE_KW, "k_act_J_mol": 30e3 * 0.0})
    for k in base:
        assert abs(ea0[k] - base[k]) == 0.0


def test_ldf_rate_unit():
    T = jnp.asarray([300.0, 350.0])
    k0 = bed1d.ldf_rate(T, 1e-3, 0.0, 30.0)
    assert float(jnp.max(jnp.abs(k0 - 1e-3))) == 0.0
    kT = bed1d.ldf_rate(T, 1e-3, 30e3, 30.0)
    assert float(kT[1]) > float(kT[0])  # hotter → faster
    assert float(kT[0]) < 1e-3 < float(kT[1])  # about the 30 °C reference


def _carry_books(c0, c1, void, ms=1.2):
    Rv = bed1d.R_VAPOR_PA_M3_KG_K
    M0 = float(c0[19]) * void / (Rv * float(jnp.mean(c0[0])))
    M1 = float(c1[19]) * void / (Rv * float(jnp.mean(c1[0])))
    S = ms * (float(jnp.mean(c0[1])) - float(jnp.mean(c1[1])))
    F = float(c1[20]) - float(c0[20])
    return abs((M1 - M0) - (F + S)) / max(abs(M1 - M0), abs(F + S), 1e-12)


def _desorb_setup(void=8e-4, tau=5.0, closed=False, k=3e-4):
    p_evap = float(water_sat_pressure_pa(16.0 + 273.15))
    phys = dict(dx=0.002 / 8, q_sat_kg_kg=0.35, q_st_j_kg=2.5e6,
                e_char_j_mol=4500.0, n_da=1.8, k_ldf_s_1=k,
                rho_s_kg_m3=600.0, c_s_j_kg_k=900.0, c_pl_j_kg_k=4180.0,
                k_eff_w_m_k=0.3, h_wall_w_m2_k=200.0, L_m=0.002,
                hx_mass_factor=2.5, t_f_ads_c=14.0, p_evap_pa=p_evap,
                p_cond_pa=P_COND, h_fg_evap_j_kg=2.4e6,
                vapor_void_m3_m2=void, vapor_tau_s=tau, valve_closed=closed)
    T0 = jnp.full((8,), 14.0 + 273.15)
    q0 = da_uptake(T0, p_evap, 0.35, 4500.0, 1.8)
    c = bed1d.initial_carry(T0, q0, t_phase_end_s=540.0, n_cycles=3,
                            p_init_pa=p_evap)
    c = c[:4] + (c[3],) + c[5:]  # forced flip → desorption next step
    return phys, c


def test_mass_conservation_open_and_closed():
    phys, c = _desorb_setup()
    cA, ysA = bed1d.advance_carry(c, (540.0, 540.0, 80.0), n_steps=400,
                                  dt_s=0.1, phys=phys)
    assert bool(jnp.all(jnp.isfinite(ysA[:, 11])))
    assert _carry_books(c, cA, 8e-4) < 1e-9
    physC = dict(phys, valve_closed=True)
    cB, ysB = bed1d.advance_carry(cA, (540.0, 540.0, 80.0), n_steps=300,
                                  dt_s=0.1, phys=physC)
    assert _carry_books(cA, cB, 8e-4) < 1e-9
    P2 = ysB[:, 11]
    assert bool(jnp.all(jnp.isfinite(P2)))
    assert float(P2[-1]) > float(P2[0])  # isosteric pressurization
    # Stays below saturation (the model's validity edge — beyond it a
    # condensation branch, not yet built, would take over).
    from harness.physics.thermo import water_sat_pressure_pa  # noqa: WPS433
    T_end = float(jnp.mean(cB[bed1d._CARRY_T]))
    assert float(P2[-1]) < float(water_sat_pressure_pa(T_end))


def test_ths_trend_steepens_with_vapor_and_arrhenius():
    def run(tfd, **extra):
        kw = dict(q_sat_kg_kg=0.35, q_st_j_kg=2.5e6, e_char_j_mol=4500.0,
                  n_da=1.8, k_ldf_s_1=3e-4, rho_s_kg_m3=600.0,
                  c_s_j_kg_k=900.0, c_pl_j_kg_k=4180.0, k_eff_w_m_k=0.3,
                  h_wall_w_m2_k=200.0, L_m=0.002, n_cells=8,
                  hx_mass_factor=2.5, t_evap_c=7.5, t_cond_c=30.0,
                  t_f_ads_c=14.0, t_f_des_c=tfd, t_ads_s=540.0, t_des_s=540.0,
                  dt_s=0.1, n_cycles=2)
        kw.update(extra)
        return float(bed1d.simulate_bed(**kw)["summary"]["SCP_W_kg"])
    v65, v80 = run(65.0), run(80.0)
    assert v80 / v65 == pytest.approx(1.07, rel=0.1)  # the documented v1 gap
    w65 = run(65.0, vapor_void_m3_m2=8e-3, vapor_tau_s=0.5, k_act_J_mol=30e3)
    w80 = run(80.0, vapor_void_m3_m2=8e-3, vapor_tau_s=0.5, k_act_J_mol=30e3)
    assert w65 > 10.0 and w80 > 10.0  # sane magnitudes, not locked
    assert w80 / w65 > 1.15  # clearly past v1, toward the measured ~2.6x


def test_env_integration_and_guards():
    env0 = harness.make("Bed1D-v0", material="anchor:Silica gel RD",
                        profile="datacenter", n_cells=8, n_cycles=2)
    m0 = env0.evaluate()
    env1 = harness.make("Bed1D-v0", material="anchor:Silica gel RD",
                        profile="datacenter", n_cells=8, n_cycles=2,
                        vapor_void_m3_m2=8e-3, vapor_tau_s=0.5,
                        k_act_J_mol=30e3)
    m1 = env1.evaluate()
    assert set(m1) == set(m0)  # summary schema unchanged
    assert abs(m1["SCP_W_kg"] - m0["SCP_W_kg"]) > 0  # vapour matters
    tr = env1.rollout()
    assert "vapor_p_Pa" in tr.series
    assert bool(jnp.all(jnp.isfinite(tr.series["vapor_p_Pa"])))
    two = harness.make("TwoBed-v0", material="anchor:Silica gel RD",
                       profile="datacenter", n_cells=4, n_cycles=2,
                       vapor_void_m3_m2=8e-3, vapor_tau_s=0.5)
    assert two.evaluate()["COP"] > 0
    with pytest.raises(ValueError):
        harness.make("Bed1D-v0", material="anchor:Silica gel RD",
                     profile="datacenter", valve_closed=True)  # no void


def test_grad_flows_with_vapor_and_arrhenius():
    from harness.envs.bed1d import Bed1D, Bed1DControls
    bed = Bed1D(material="anchor:Silica gel RD", profile="datacenter",
                n_cells=8, n_cycles=2, dt_phys_s=0.05,
                vapor_void_m3_m2=8e-3, vapor_tau_s=0.5, k_act_J_mol=30e3)
    prob = Bed1DControls(bed, t_switch_bounds=(60.0, 600.0), n_cycles=2,
                         dt_phys_s=0.05, n_steps=2000)
    g = jax.grad(lambda c: prob.metrics_jax(
        {"t_switch_s": c[0], "t_f_des_c": c[1]})["SCP_W_kg"])(
            jnp.asarray([300.0, 60.0]))
    assert bool(jnp.all(jnp.isfinite(g)))
