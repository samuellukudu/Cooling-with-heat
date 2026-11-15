"""H2.4 acceptance: the stochastic-source schedule bandit (ROADMAP item 2).

- scenario sampler: deterministic under a seed, sinusoid inside the band,
  bursts produce real dips;
- TwoBedScenario: gym contract (obs in [0,1], one-step episode, finite
  reward, schema-ordered metrics), same-seed reproducibility, cross-seed
  variation;
- V8 honesty (DESIGN §6, spec'd until now): real PPO on deterministic
  Cycle0D-v0 must NOT beat search — RL is justified only under stochastic
  boundaries;
- smoke: the full rl_h24 pipeline (floors → PPO → policy-vs-floor report)
  runs CI-sized and produces a JSON report + saved policy.

Numerics sized for CI: horizon 600 s at dt 50 ms (12 000 physics steps,
~70 ms per rollout) after the H2.2 precedent.
"""

import json

import numpy as np
import pytest

from harness import optimize
from harness.envs import TwoBedScenario
from harness.envs.base import Objective, objective_value
from harness.envs.cycle0d import CANONICAL_NORMALIZATION
from harness.envs.cycle0d import build_cycle0d
from harness.envs.two_bed import SCENARIO_DESC_KEYS, sample_source_scenario
from harness.rl_h24 import (FAST_DESIGN, build_floors, build_problem,
                            report, train_policy)


@pytest.fixture(scope="module")
def scenario():
    return build_problem(horizon_s=600.0, dt_phys_s=0.05)


def test_scenario_sampler_deterministic_and_in_band():
    rng = np.random.default_rng(0)
    src1, d1 = sample_source_scenario(rng, 60.0, 85.0, burst_prob=0.0)
    rng = np.random.default_rng(0)
    src2, d2 = sample_source_scenario(rng, 60.0, 85.0, burst_prob=0.0)
    t = np.linspace(0.0, 7200.0, 500)
    assert d1 == d2
    assert [src1(x) for x in t] == [src2(x) for x in t]
    vals = np.asarray([src1(x) for x in t])
    assert vals.min() >= 60.0 - 1e-9 and vals.max() <= 85.0 + 1e-9


def test_scenario_bursts_dip_below_the_sinusoid():
    rng = np.random.default_rng(3)
    src, desc = sample_source_scenario(rng, 60.0, 85.0,
                                       period_range=(1800.0, 1800.0),
                                       burst_prob=1.0)
    assert desc["burst_depth_c"] > 0.0
    # Inside the dip window (first `burst_frac` of each period) the source
    # sits exactly `burst_depth` below the plain sinusoid. The *global*
    # minimum may still be the sinusoid trough — dips do not necessarily
    # coincide with it.
    t_dip = 0.02 * desc["period_s"]
    no_burst = desc["base_c"] + desc["amp_c"] * np.sin(
        2.0 * np.pi * t_dip / desc["period_s"])
    assert src(t_dip) == pytest.approx(no_burst - desc["burst_depth_c"])


def test_scenario_reset_reproducible_cross_seed_varies(scenario):
    design = {"t_set_c": 80.0, "t_thresh_c": 65.0, "t_half_s": 174.0,
              "t_rec_s": 30.0, "t_dwell_s": 300.0}
    m1 = scenario.evaluate_seed(design, 7)
    m2 = scenario.evaluate_seed(design, 7)
    assert m1 == m2  # seeded scenario → identical duty envelope
    m3 = scenario.evaluate_seed(design, 8)
    assert (m1["COP"], m1["SCP_W_kg"]) != (m3["COP"], m3["SCP_W_kg"])


def test_scenario_gym_contract(scenario):
    obs, info = scenario.reset(seed=1)
    assert obs.shape == (len(SCENARIO_DESC_KEYS),)
    assert scenario.observation_space.contains(obs)
    assert "scenario" in info and info["seed"] == 1
    a = np.asarray([80.0, 65.0, 174.0, 30.0, 300.0], dtype=np.float64)
    obs2, reward, terminated, truncated, info2 = scenario.step(a)
    assert terminated and not truncated  # one-step contextual bandit
    assert np.isfinite(reward)
    assert tuple(info2["metrics"]) == scenario.spec.metric_keys  # schema order
    assert scenario.action_space.contains(np.asarray(a, dtype=np.float32))


def test_v8_rl_must_not_beat_search_on_cycle0d():
    """DESIGN §6 V8: RL on a deterministic static problem must not beat
    search — the honesty posture, spec'd at DESIGN.md:564, now a test."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    prob = build_cycle0d(profile="cpu")
    obj = Objective(weights={"COP": 1.0},
                    normalize={"COP": CANONICAL_NORMALIZATION["COP"]})
    res_search = optimize(prob, obj, backend="search", method="cmaes",
                          budget=80, seed=0)
    res_rl = optimize(prob, obj, backend="rl", total_timesteps=4096,
                      seed=0, n_eval_episodes=3)
    print(f"\nV8: search {res_search.best_objective:.4f}  "
          f"rl {res_rl.best_objective:.4f}")
    assert res_rl.best_objective <= res_search.best_objective + 1e-9, (
        "V8 violated: RL beat search on a deterministic static problem — "
        "the honesty floor posture is broken")


def test_h24_smoke_pipeline(scenario, tmp_path):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from harness.backends import evaluate_policy

    seeds = [0, 1]
    floors = build_floors(scenario, seeds, floor_budget=4, seed_base=2000)
    res, model, vn = train_policy(scenario, total_timesteps=256, seed=0,
                                  save_dir=tmp_path / "pol", n_steps=64)
    rows = evaluate_policy(scenario, model, scenario.objective, seeds,
                           vec_normalize=vn)
    assert len(rows) == 2
    for r in rows:
        assert np.isfinite(r["objective"])
        assert set(r["design"]) == set(scenario.POLICY_KEYS)
    rep = report(scenario, floors, rows,
                 extra={"total_timesteps": 256, "seconds": 0.0})
    assert set(rep["gates"]) == {"gate_floor", "gate_margin"}
    assert all(isinstance(v, bool) for v in rep["gates"].values())
    assert len(rep["rows"]) == 2
    json.dumps(rep)  # the report is the JSON artifact
    assert (tmp_path / "pol" / "policy.zip").exists()
