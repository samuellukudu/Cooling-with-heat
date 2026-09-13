"""Natural-convection trial on the harness (T-A1, docs/applications.md §1).

Gates:
- Correlation pin: Nu(1e9, 0.707) ≈ 67.2 (Globe–Dropkin) ±3%.
- Onset: Nu == 1 below Ra = 1708 (conduction floor with kink).
- Flags: in_range reads the [3e5, 7e9] validity band honestly.
- Materials: water ≫ air in flux at fixed ΔT (fluid shortlist signal).
- Discovery: search maximizes flux over ΔT and beats the 10 K default;
  grad flows with the right sign.
"""

import jax

jax.config.update("jax_enable_x64", True)

import pytest

import harness
from harness.envs.base import Objective, objective_value
from harness.registry import REGISTRIES


def test_natconv_registered():
    assert "NaturalConv-v0" in REGISTRIES["envs"].names()
    assert "natural_conv" in REGISTRIES["models"].names()


def test_natconv_protocol_and_metrics():
    prob = harness.make("NaturalConv-v0", material="air", profile="human")
    from harness.envs.base import validate_problem
    validate_problem(prob)
    m = prob.evaluate()
    assert set(m) >= {"Nu", "Ra", "heat_flux_W_m2", "delta_T_K", "in_range"}
    assert all(v == v and abs(v) != float("inf") for v in m.values()), m


def test_unknown_design_key_raises():
    prob = harness.make("NaturalConv-v0", material="air", profile="human")
    with pytest.raises(KeyError):
        prob.evaluate({"L_m": 0.2})  # structural, not design
    with pytest.raises(KeyError):
        prob.evaluate({"Pr": 1.0})  # fluid property, not design


def test_anchor_falls_back_with_provenance():
    prob = harness.make("NaturalConv-v0", material="anchor:Silica gel RD", profile="human")
    assert prob.fluid_name == "air"
    assert prob.fluid_provenance == "anchor-fallback"


def test_unknown_fluid_raises():
    from harness.physics.natural_conv import resolve_fluid
    with pytest.raises(KeyError):
        resolve_fluid("unobtainium")


def test_correlation_pin():
    from harness.physics.natural_conv import nusselt
    assert float(nusselt(1e9, 0.707)) == pytest.approx(67.2, rel=0.03)


def test_conduction_onset_and_range_flags():
    prob = harness.make("NaturalConv-v0", material="air", profile="human")
    deep_sub = prob.evaluate({"delta_T_K": 0.01})  # Ra ≈ 915 — below onset
    assert deep_sub["Nu"] == pytest.approx(1.0)
    assert deep_sub["in_range"] == 0.0
    assert prob.regime({"delta_T_K": 0.01}) == "conduction"
    correlated = prob.evaluate({"delta_T_K": 10.0})  # Ra ≈ 9.2e5 — in band
    assert correlated["Nu"] > 1.0
    assert correlated["in_range"] == 1.0
    assert prob.regime() == "convection (correlated)"


def test_water_beats_air_at_fixed_delta_T():
    air = harness.make("NaturalConv-v0", material="air", profile="human")
    water = harness.make("NaturalConv-v0", material="water", profile="human")
    design = {"delta_T_K": 10.0}
    assert water.evaluate(design)["heat_flux_W_m2"] > 10 * air.evaluate(design)["heat_flux_W_m2"]


def test_search_maximizes_flux_over_delta_T():
    prob = harness.make("NaturalConv-v0", material="air", profile="human")
    obj = Objective.single("heat_flux_W_m2")
    naive = float(objective_value(obj, prob.evaluate()))
    res = harness.optimize(prob, obj, backend="search", method="tpe", seed=0, budget=30)
    assert res.best_objective > naive
    assert res.best_design["delta_T_K"] > prob.design_space.defaults["delta_T_K"]


def test_grad_flows_with_right_sign():
    prob = harness.make("NaturalConv-v0", material="air", profile="human")
    g = float(jax.grad(lambda d: prob.metrics_jax({"delta_T_K": d})["heat_flux_W_m2"])(10.0))
    assert g == g and g > 0  # more driving ΔT → more flux


def test_adapter_model_satisfies_protocol():
    from harness.physics.adapters import SimulatorAdapter
    inst = REGISTRIES["models"].resolve("natural_conv")()
    assert isinstance(inst, SimulatorAdapter)
    state, _ = inst.step({}, {"delta_T_K": 10.0}, 1.0)
    assert "Nu" in inst.metrics(state)
