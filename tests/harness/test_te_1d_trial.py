"""Segmented-leg TE trial (T-A2 extension).

Gates:
- Uniform-leg parity vs the lumped pair model < 2 % (V3-style).
- Energy books (Qh − Qc − W_in) < 1 % (V2-style).
- Grading discovery: search over (current, seg_frac) on a Bi2Te3-cold /
  PbTe-hot leg beats the best uniform-Bi2Te3 leg clearly.
- Grad flows in both design axes; protocol/registry/adapter compliance.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

import harness
from harness.envs.base import Objective, objective_value, validate_problem
from harness.registry import REGISTRIES


def test_te1d_registered():
    assert "Thermoelectric1D-v0" in REGISTRIES["envs"].names()
    assert "te_1d" in REGISTRIES["models"].names()


def test_te1d_protocol_and_metrics():
    prob = harness.make("Thermoelectric1D-v0", material="Bi2Te3", profile="cpu")
    validate_problem(prob)
    m = prob.evaluate()
    assert set(m) >= {"Qc_W", "COP", "delta_T_K", "ZT_cold", "I_opt_cold_A",
                      "out_of_window"}
    assert all(v == v and abs(v) != float("inf") for v in m.values()), m
    assert prob.pair_hot == "Bi2Te3"  # uniform default (parity config)


def test_unknown_design_key_raises():
    prob = harness.make("Thermoelectric1D-v0", material="Bi2Te3", profile="cpu")
    with pytest.raises(KeyError):
        prob.evaluate({"pair_cold": "PbTe"})
    with pytest.raises(KeyError):
        prob.evaluate({"Th_C": 40.0})


def test_uniform_parity_vs_lumped():
    from harness.physics.te_1d import simulate_te_1d
    from harness.physics.thermoelectric import simulate_te
    kw = dict(Tc_C=18.0, Th_C=35.0, current_A=3.0)
    pde = {k: float(simulate_te_1d(
        **kw, pair_cold="Bi2Te3", pair_hot="Bi2Te3")[k])
        for k in ("Qc_W", "COP")}
    lump = {k: float(v) for k, v in simulate_te(
        **kw, pair="Bi2Te3").items() if k in ("Qc_W", "COP")}
    # Residual = spatial discretization + Peltier evaluation point.
    assert pde["Qc_W"] == pytest.approx(lump["Qc_W"], rel=0.02)
    assert pde["COP"] == pytest.approx(lump["COP"], rel=0.05)


def test_energy_books_close():
    from harness.physics.te_1d import simulate_te_1d
    p = simulate_te_1d(Tc_C=18.0, Th_C=35.0, current_A=4.0,
                       pair_cold="Bi2Te3", pair_hot="PbTe", seg_frac=0.5)
    Qc, Qh, W = float(p["Qc_W"]), float(p["Qh_W"]), float(p["W_in_W"])
    assert abs(Qh - Qc - W) / max(abs(W), 1e-12) < 0.01


def test_grading_beats_uniform():
    uni = harness.make("Thermoelectric1D-v0", material="Bi2Te3", profile="cpu")
    obj = Objective.single("Qc_W")
    best_uni = max(float(objective_value(
        obj, uni.evaluate({"current_A": I}))) for I in (2.0, 3.0, 4.0, 5.0))
    graded = harness.make("Thermoelectric1D-v0", material="Bi2Te3",
                          material_b="PbTe", profile="cpu")
    res = harness.optimize(graded, obj, backend="search", method="tpe",
                           seed=0, budget=25)
    assert res.best_objective > best_uni * 1.05
    assert res.best_objective > float(objective_value(obj, graded.evaluate()))


def test_grad_flows_in_both_axes():
    prob = harness.make("Thermoelectric1D-v0", material="Bi2Te3",
                        material_b="PbTe", profile="cpu")
    f = lambda d: prob.metrics_jax(d)["Qc_W"]
    g = jax.grad(lambda c: f({"current_A": c[0], "seg_frac": c[1]}))(
        jnp.asarray([3.0, 0.5]))
    assert all(v == v and abs(v) != float("inf") for v in g)


def test_adapter_model_satisfies_protocol():
    from harness.physics.adapters import SimulatorAdapter
    inst = REGISTRIES["models"].resolve("te_1d")()
    assert isinstance(inst, SimulatorAdapter)
