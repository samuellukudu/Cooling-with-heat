"""Resolved-cavity trial on the harness (T-A1 extension).

Gates:
- Onset: Nu == 1 below Ra_c (conduction, velocities ~0).
- Pins (laminar envelope, bot/top-converged): Nu(1e4) and Nu(1e5, N=32)
  against the Globe–Dropkin values the correlation level ships.
- Regime: umax stays subsonic vs the matched sound speed (Mach < 0.5).
- Cross-level consistency: resolved Nu vs ``NaturalConv-v0`` ±30 %.
- Discovery: search maximizes flux over ΔT and beats the default;
  grad dNu/dRa is finite and positive; protocol/registry/adapter hold.
"""

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

import harness
from harness.envs.base import Objective, objective_value, validate_problem
from harness.registry import REGISTRIES


def test_boussinesq_registered():
    assert "Boussinesq-v0" in REGISTRIES["envs"].names()
    assert "boussinesq" in REGISTRIES["models"].names()


def test_boussinesq_protocol_and_metrics():
    prob = harness.make("Boussinesq-v0", material="air", profile="human",
                        L_m=0.03, n_cells=16, t_end=0.3)
    validate_problem(prob)
    m = prob.evaluate()
    assert set(m) >= {"Nu", "Nu_bot", "Nu_top", "u_max", "Ra", "delta_T_K",
                      "in_validated"}
    assert all(v == v and abs(v) != float("inf") for v in m.values()), m


def test_unknown_design_key_raises():
    prob = harness.make("Boussinesq-v0", material="air", profile="human")
    with pytest.raises(KeyError):
        prob.evaluate({"Ra": 1e4})  # derived, not design
    with pytest.raises(KeyError):
        prob.evaluate({"L_m": 0.1})  # structural, not design


def test_conduction_onset():
    from harness.physics import boussinesq as bq
    m = bq.simulate_boussinesq(Ra=1000.0, Pr=0.707, n_cells=16, t_end=0.3)
    assert float(m["Nu"]) == pytest.approx(1.0, rel=5e-3)
    assert float(m["u_max"]) < 0.5


def test_nu_pins_and_wall_agreement():
    from harness.physics import boussinesq as bq
    from harness.physics.natural_conv import nusselt
    m4 = bq.simulate_boussinesq(Ra=1e4, Pr=0.707, n_cells=24, t_end=0.6)
    assert float(m4["Nu"]) == pytest.approx(float(nusselt(1e4, 0.707)), rel=0.15)
    m5 = bq.simulate_boussinesq(Ra=1e5, Pr=0.707, n_cells=32, t_end=0.6)
    nu5 = float(m5["Nu"])
    assert nu5 == pytest.approx(float(nusselt(1e5, 0.707)), rel=0.30)
    bot, top = float(m5["Nu_bot"]), float(m5["Nu_top"])
    assert abs(bot - top) / max(nu5, 1e-12) < 0.02  # resolved books
    c = 3.0 * math.sqrt(1e5 * 0.707)
    assert float(m5["u_max"]) < 0.5 * c  # incompressible regime


def test_cross_level_consistency():
    res = harness.make("Boussinesq-v0", material="air", profile="human",
                       L_m=0.03, n_cells=24, t_end=0.6)
    cor = harness.make("NaturalConv-v0", material="air", profile="human",
                       L_m=0.03)
    design = {"delta_T_K": 10.0}
    assert res.evaluate(design)["Nu"] == pytest.approx(
        cor.evaluate(design)["Nu"], rel=0.30)


def test_search_maximizes_nu_over_delta_T():
    prob = harness.make("Boussinesq-v0", material="air", profile="human",
                        L_m=0.02, n_cells=16, t_end=0.3)
    # The resolved env reports Nu (no flux metric): maximize it over the
    # driving ΔT — the discovery is the operating point, found by search.
    obj = Objective.single("Nu")
    naive = float(objective_value(obj, prob.evaluate()))
    res = harness.optimize(prob, obj, backend="search", method="tpe",
                           seed=0, budget=15)
    assert res.best_objective >= naive
    assert res.best_design["delta_T_K"] > prob.design_space.defaults["delta_T_K"]


def test_grad_flows_with_right_sign():
    prob = harness.make("Boussinesq-v0", material="air", profile="human",
                        L_m=0.03, n_cells=16, t_end=0.3)
    g = float(jax.grad(lambda d: prob.metrics_jax({"delta_T_K": d})["Nu"])(10.0))
    assert g == g and g > 0  # hotter driving → stronger convection


def test_adapter_model_satisfies_protocol():
    from harness.physics.adapters import SimulatorAdapter
    inst = REGISTRIES["models"].resolve("boussinesq")()
    assert isinstance(inst, SimulatorAdapter)
