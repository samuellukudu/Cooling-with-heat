"""Absorption-chiller trial on the harness (T-A3, docs/applications.md §3).

Gates:
- V-ABS: LiBr-H2O at the textbook point (t_evap=7 / t_cond=30 / t_gen=85 °C)
  gives COP ≈ 0.70 within ±15 %.
- Discovery signals: COP rises monotonically in t_gen (fixed pair/profile);
  at low-grade datacenter regeneration (60 °C) NH3-H2O out-SCPs LiBr-H2O
  (materials axis — the T-A3 analog of the H2.3 13X ranking flip).
- Search improves COP over the naive profile-default t_gen; grad flows.
- Registry + Problem-protocol + adapter-protocol compliance.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import pytest

import harness
from harness.envs.base import Objective, validate_problem
from harness.registry import REGISTRIES


def test_absorption_registered():
    assert "AbsorptionCycle-v0" in REGISTRIES["envs"].names()
    assert "absorption" in REGISTRIES["models"].names()


def test_absorption_protocol_and_metrics():
    prob = harness.make("AbsorptionCycle-v0", material="LiBr-H2O", profile="human")
    validate_problem(prob)
    m = prob.evaluate()
    assert set(m) >= {"COP", "SCP_W_kg", "Q_evap_J_kg", "Q_gen_J_kg", "delta_x"}
    assert all(v == v and abs(v) != float("inf") for v in m.values()), m


def test_unknown_design_key_raises():
    prob = harness.make("AbsorptionCycle-v0", material="LiBr-H2O", profile="human")
    with pytest.raises(KeyError):
        prob.evaluate({"eps": 0.5})  # pair property, not design
    with pytest.raises(KeyError):
        prob.evaluate({"t_cond_c": 30.0})  # scenario, not design


def test_anchor_material_falls_back_with_provenance():
    prob = harness.make("AbsorptionCycle-v0", material="anchor:Silica gel RD", profile="human")
    assert prob.pair_name == "LiBr-H2O"
    assert prob.pair_provenance == "anchor-fallback"
    assert prob.evaluate()["COP"] > 0


def test_unknown_pair_raises():
    from harness.physics.absorption import resolve_pair
    with pytest.raises(KeyError):
        resolve_pair("Nope-H2O")


def test_v_abs_standard_point():
    # Textbook single-effect LiBr/H2O band: COP ≈ 0.70 at 7/30/85 °C
    # (raw temperatures — profile setpoints differ slightly; see below).
    from harness.physics.absorption import simulate_absorption
    m = {k: float(v) for k, v in simulate_absorption(
        t_evap_c=7.0, t_cond_c=30.0, t_gen_c=85.0,
        pair="LiBr-H2O", cycle_time_s=600.0).items()}
    assert m["COP"] == pytest.approx(0.70, rel=0.15)
    assert m["SCP_W_kg"] > 0 and m["delta_x"] > 0
    # Via the human profile (10/35 °C, 80 °C default) the same pair is
    # positive and close to the band.
    prob = harness.make("AbsorptionCycle-v0", material="LiBr-H2O", profile="human")
    mh = prob.evaluate({"t_gen_c": 85.0})
    assert mh["COP"] == pytest.approx(0.70, rel=0.25)


def test_cop_monotone_in_t_gen():
    prob = harness.make("AbsorptionCycle-v0", material="LiBr-H2O", profile="human")
    lo = prob.evaluate({"t_gen_c": 70.0})["COP"]
    hi = prob.evaluate({"t_gen_c": 90.0})["COP"]
    assert hi > lo  # the design discovery signal


def test_degenerate_band_bottom():
    # Below the driving threshold the spread clips to 0: honest no-cooling.
    prob = harness.make("AbsorptionCycle-v0", material="LiBr-H2O", profile="datacenter")
    m = prob.evaluate({"t_gen_c": 60.0})
    assert m["delta_x"] >= 0.0
    if m["delta_x"] == 0.0:
        assert m["COP"] == 0.0 and m["SCP_W_kg"] == 0.0


def test_pair_ranking_flips_with_regeneration_temperature():
    # Materials axis: at low-grade 60 °C regeneration NH3-H2O (wide band)
    # delivers more specific cooling than LiBr-H2O (peaky) on datacenter.
    libr = harness.make("AbsorptionCycle-v0", material="LiBr-H2O", profile="datacenter")
    nh3 = harness.make("AbsorptionCycle-v0", material="NH3-H2O", profile="datacenter")
    design = {"t_gen_c": 60.0}
    assert nh3.evaluate(design)["SCP_W_kg"] > libr.evaluate(design)["SCP_W_kg"]
    # …while at high-grade human regeneration LiBr wins on COP.
    libr_h = harness.make("AbsorptionCycle-v0", material="LiBr-H2O", profile="human")
    nh3_h = harness.make("AbsorptionCycle-v0", material="NH3-H2O", profile="human")
    design_h = {"t_gen_c": 90.0}
    assert libr_h.evaluate(design_h)["COP"] > nh3_h.evaluate(design_h)["COP"]


def test_search_improves_cop_over_naive():
    prob = harness.make("AbsorptionCycle-v0", material="LiBr-H2O", profile="human")
    obj = Objective.single("COP")
    from harness.envs.base import objective_value
    naive = float(objective_value(obj, prob.evaluate()))
    res = harness.optimize(prob, obj, backend="search", method="tpe", seed=0, budget=30)
    assert res.best_objective >= naive
    assert res.best_design["t_gen_c"] >= prob.design_space.defaults["t_gen_c"] - 1e-9
    assert res.history


def test_grad_flows_in_t_gen():
    prob = harness.make("AbsorptionCycle-v0", material="LiBr-H2O", profile="human")
    g = jax.grad(lambda t: prob.metrics_jax({"t_gen_c": t})["COP"])(80.0)
    assert float(g) == float(g) and float(g) > 0


def test_adapter_model_satisfies_protocol():
    from harness.physics.adapters import SimulatorAdapter
    inst = REGISTRIES["models"].resolve("absorption")()
    assert isinstance(inst, SimulatorAdapter)
    state, fluxes = inst.step({}, {"t_gen_c": 85.0}, 600.0)
    assert "COP" in inst.metrics(state)
