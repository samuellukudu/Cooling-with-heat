"""H3 SimulatorAdapter protocol — ModelSpec, MockSimulatorAdapter, env wiring (DESIGN §7.2)."""

import pytest

import harness
from harness.physics.adapters import MockSimulatorAdapter, ModelSpec, SimulatorAdapter
from harness.envs.base import Objective, validate_problem
from harness.registry import REGISTRIES


def test_model_spec_and_protocol():
    spec = ModelSpec(name="test", state_vars=("T", "q"), units={"T": "K"}, dt_min_s=1e-3, dt_max_s=1.0)
    assert spec.name == "test"
    assert spec.units["T"] == "K"
    # Protocol is runtime_checkable
    mock = MockSimulatorAdapter()
    assert isinstance(mock, SimulatorAdapter)
    assert hasattr(mock, "spec")
    assert hasattr(mock, "step")
    assert hasattr(mock, "metrics")
    assert isinstance(mock.spec, ModelSpec)


def test_mock_adapter_step_and_metrics_match_cycle0d():
    # Mock via adapter should be close to Cycle0D oracle for same inputs
    from harness.physics import simulate_cycle

    mock = MockSimulatorAdapter(t_evap_c=16.0, t_cond_c=35.0, cycle_time_s=300.0)
    control = {"q_sat_kg_kg": 0.35, "Q_st_j_kg": 2.5e6, "e_char_j_mol": 4500.0, "n_da": 1.8, "t_des_c": 60.0}
    state, fluxes = mock.step({}, control, dt_s=300.0)
    assert "COP" in state and "SCP_W_kg" in state
    assert fluxes.q_cool_w_m2 != 0.0 or fluxes.q_in_w_m2 != 0.0
    metrics = mock.metrics(state)
    assert set(metrics) >= {"COP", "SCP_W_kg", "delta_q", "q_ads", "q_des"}
    # Compare with direct simulate_cycle
    direct = simulate_cycle(0.35, 2.5e6, 16.0, 35.0, 60.0, 300.0, 4500.0, 1.8, 1.35)
    assert metrics["COP"] == pytest.approx(float(direct["COP"]), rel=1e-6)
    assert metrics["SCP_W_kg"] == pytest.approx(float(direct["SCP_W_kg"]), rel=1e-6)


def test_registry_wiring_models_and_env():
    # SimulatorAdapter registry is harness.models
    assert "mock_cycle0d" in REGISTRIES["models"].names()
    factory = REGISTRIES["models"].resolve("mock_cycle0d")
    inst = factory()
    assert isinstance(inst, MockSimulatorAdapter)
    # Env registry for adapter demo
    assert "MockCycle-v0" in REGISTRIES["envs"].names()
    # harness.make round-trip satisfies Problem protocol
    prob = harness.make("MockCycle-v0", material="anchor:Silica gel RD", profile="datacenter")
    validate_problem(prob)
    metrics = prob.evaluate()
    assert "COP" in metrics and "SCP_W_kg" in metrics
    # Metrics should be finite and positive for this anchor/profile
    assert metrics["COP"] > 0.0
    assert metrics["SCP_W_kg"] > 0.0


def test_adapter_cycle_comparable_to_cycle0d():
    cyc = harness.make("Cycle0D-v0", material="anchor:Silica gel RD", profile="datacenter", hx_mass_factor=1.35)
    adap = harness.make("MockCycle-v0", material="anchor:Silica gel RD", profile="datacenter", hx_mass_factor=1.35)
    m_cyc = cyc.evaluate()
    m_ad = adap.evaluate()
    # Mock reproduces oracle within tolerance (it delegates to same simulate_cycle)
    assert m_ad["COP"] == pytest.approx(m_cyc["COP"], rel=1e-6)
    assert m_ad["SCP_W_kg"] == pytest.approx(m_cyc["SCP_W_kg"], rel=1e-9)
    # Design space mutation flows through adapter
    m_cyc2 = adap.evaluate({"q_sat_kg_kg": 0.45})
    assert m_cyc2["delta_q"] != m_ad["delta_q"]


def test_adapter_problem_optimizable_by_search():
    prob = harness.make("MockCycle-v0", material="anchor:Silica gel RD", profile="datacenter")
    obj = Objective.single("COP")
    res = harness.optimize(prob, obj, backend="search", method="tpe", seed=0, budget=30)
    assert res.best_objective >= float(Objective.single("COP").weights["COP"] * prob.evaluate()["COP"]) - 0.2
    assert res.best_metrics["COP"] > 0
    assert res.history

def test_gymnasium_registration_for_mockcycle():
    import gymnasium as gym
    # Check that gym registry contains our entry (or at least make via harmness works)
    # Gymnasium's registry may have been populated at import
    # We test that making via harness still yields gym Env via envs.adapter
    from harness.envs.adapter import AdapterCycleGym  # type: ignore
    env = AdapterCycleGym(material="anchor:Silica gel RD", profile="datacenter")
    obs, info = env.reset()
    assert obs.shape == (1,)
    action = env.action_space.sample()
    obs2, rew, term, trunc, info = env.step(action)
    assert term is True
    assert "metrics" in info
