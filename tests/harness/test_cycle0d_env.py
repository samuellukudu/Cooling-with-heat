"""Cycle0D-v0 env behaviour: defaults from material × profile, evaluation
matches canonical physics, design overrides, rollout schema (DESIGN §5.1)."""

import numpy as np
import pytest

import cooling_physics
import harness
from harness.envs.base import EpisodeTrace


@pytest.fixture()
def env():
    return harness.make("Cycle0D-v0", material="anchor:Silica gel RD", profile="datacenter")


def test_defaults_come_from_material_and_profile(env):
    assert env.design_space.defaults["q_sat_kg_kg"] == 0.35
    assert env.design_space.defaults["Q_st_j_kg"] == pytest.approx(2.5e6)
    assert env.design_space.defaults["e_char_j_mol"] == 4500.0
    assert env.design_space.defaults["n_da"] == 1.8
    assert env.design_space.defaults["cycle_time_s"] == 300.0
    assert env.design_space.defaults["hx_mass_factor"] == 1.35


def test_evaluate_matches_canonical_physics(env):
    got = env.evaluate()
    ref = cooling_physics.simulate_adsorption_cycle(
        q_sat=0.35, q_st=2.5e6, t_evap_c=16.0, t_cond_c=35.0, t_des_c=60.0,
        cycle_time_sec=300.0, e_char_j_mol=4500.0, n_heterogeneity=1.8, hx_mass_factor=1.35,
    )
    for key, ref_value in ref.items():
        np.testing.assert_allclose(got[key], ref_value, rtol=1e-12, atol=1e-15)


def test_metric_keys_match_evaluate(env):
    assert set(env.spec.metric_keys) == set(env.evaluate())


def test_design_override_and_unknown_key(env):
    base = env.evaluate()["COP"]
    changed = env.evaluate({"q_sat_kg_kg": 0.9})["COP"]
    assert changed > base  # more capacity ⇒ more cooling at this operating point
    with pytest.raises(KeyError, match="unknown design keys"):
        env.evaluate({"not_a_key": 1.0})


def test_rollout_is_static_trace(env):
    trace = env.rollout()
    assert isinstance(trace, EpisodeTrace)
    assert trace.series == {}  # static problem: no time series (DESIGN §4.1)
    assert trace.summary == env.evaluate()


def test_design_space_bounds_are_finite_and_contain_defaults(env):
    lo, hi = env.design_space.bounds_for(env.design_space.keys)
    for key, lo_v, hi_v in zip(env.design_space.keys, lo, hi):
        d = env.design_space.defaults[key]
        assert np.isfinite(lo_v) and np.isfinite(hi_v) and lo_v <= d <= hi_v


def test_instance_passthrough():
    profile = harness.get_profile("human")
    material = harness.get_material("anchor:Zeolite 13X (NaX)")
    env = harness.make("Cycle0D-v0", material=material, profile=profile)
    assert env.material is material
    assert env.profile is profile
    assert env.design_space.defaults["cycle_time_s"] == 600.0
