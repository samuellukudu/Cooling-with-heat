"""Thermoelectric-cooler trial on the harness (T-A2, docs/applications.md §2).

Gates:
- Closed-form pins: Qc(I_opt) matches ``qcmax_closed_form`` <1%;
  I_opt == S·Tc/R exactly (same equation — implementation check).
- Discovery: search recovers I_opt from the off-optimum 2 A default;
  grad-vs-FD gate on Qc(I).
- Materials: with per-pair optimized current, Bi2Te3 wins cpu (room-temp);
  at vehicle rejection temps Bi2Te3 is out_of_window while PbTe pumps
  (validity-driven selection).
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

import harness
from harness.envs.base import Objective, objective_value
from harness.registry import REGISTRIES


def test_te_registered():
    assert "Thermoelectric-v0" in REGISTRIES["envs"].names()
    assert "thermoelectric" in REGISTRIES["models"].names()


def test_te_protocol_and_metrics():
    prob = harness.make("Thermoelectric-v0", material="Bi2Te3", profile="cpu")
    from harness.envs.base import validate_problem
    validate_problem(prob)
    m = prob.evaluate()
    assert set(m) >= {"Qc_W", "COP", "delta_T_K", "ZT_pair", "I_opt_A", "out_of_window"}
    assert all(v == v and abs(v) != float("inf") for v in m.values()), m


def test_unknown_design_key_raises():
    prob = harness.make("Thermoelectric-v0", material="Bi2Te3", profile="cpu")
    with pytest.raises(KeyError):
        prob.evaluate({"S_V_K": 1.0})  # pair property, not design
    with pytest.raises(KeyError):
        prob.evaluate({"Th_C": 40.0})  # scenario, not design


def test_anchor_falls_back_with_provenance():
    prob = harness.make("Thermoelectric-v0", material="anchor:Silica gel RD", profile="cpu")
    assert prob.pair_name == "Bi2Te3"
    assert prob.pair_provenance == "anchor-fallback"


def test_unknown_pair_raises():
    from harness.physics.thermoelectric import resolve_te_pair
    with pytest.raises(KeyError):
        resolve_te_pair("NopeTe")


def test_closed_form_pin():
    from harness.physics.thermoelectric import qcmax_closed_form, resolve_te_pair
    prob = harness.make("Thermoelectric-v0", material="Bi2Te3", profile="cpu")
    m = prob.evaluate()
    p = resolve_te_pair("Bi2Te3")
    assert m["I_opt_A"] == pytest.approx(p["S_V_K"] * (18.0 + 273.15) / p["R_ohm"])
    m_opt = prob.evaluate({"current_A": m["I_opt_A"]})
    assert m_opt["Qc_W"] == pytest.approx(
        qcmax_closed_form(Tc_C=18.0, Th_C=35.0, pair="Bi2Te3"), rel=1e-2)


def test_grad_vs_fd_on_current():
    prob = harness.make("Thermoelectric-v0", material="Bi2Te3", profile="cpu")
    f = lambda i: prob.metrics_jax({"current_A": i})["Qc_W"]
    g_ad = float(jax.grad(f)(2.0))
    h = 1e-4
    g_fd = (float(f(2.0 + h)) - float(f(2.0 - h))) / (2 * h)
    assert g_ad == pytest.approx(g_fd, rel=1e-4)
    assert g_ad > 0  # below I_opt — more current helps (discovery direction)


def test_search_recovers_i_opt():
    prob = harness.make("Thermoelectric-v0", material="Bi2Te3", profile="cpu")
    obj = Objective.single("Qc_W")
    naive = float(objective_value(obj, prob.evaluate()))
    res = harness.optimize(prob, obj, backend="search", method="tpe", seed=0, budget=30)
    assert res.best_objective > naive
    assert res.best_design["current_A"] == pytest.approx(prob.evaluate()["I_opt_A"], rel=0.1)


def _best_qc(pair: str, profile: str) -> float:
    prob = harness.make("Thermoelectric-v0", material=pair, profile=profile)
    res = harness.optimize(prob, Objective.single("Qc_W"), backend="search",
                           method="tpe", seed=0, budget=20)
    return res.best_objective


def test_bi2te3_wins_room_temp_with_optimized_current():
    assert _best_qc("Bi2Te3", "cpu") > _best_qc("PbTe", "cpu")


def test_validity_flip_at_vehicle_temps():
    bi = harness.make("Thermoelectric-v0", material="Bi2Te3", profile="vehicle")
    pb = harness.make("Thermoelectric-v0", material="PbTe", profile="vehicle")
    assert bi.evaluate()["out_of_window"] == 1.0
    assert pb.evaluate()["out_of_window"] == 0.0
    assert pb.evaluate()["Qc_W"] > 0  # the valid pick still pumps


def test_adapter_model_satisfies_protocol():
    from harness.physics.adapters import SimulatorAdapter
    inst = REGISTRIES["models"].resolve("thermoelectric")()
    assert isinstance(inst, SimulatorAdapter)
    state, _ = inst.step({}, {"current_A": 3.0}, 1.0)
    assert "Qc_W" in inst.metrics(state)
