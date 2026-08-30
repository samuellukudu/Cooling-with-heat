"""TwoBed-v0 env gates (DESIGN §12 H2.1): Gymnasium checker, metric schema
stability, rollout/step agreement, recovery gain, composite beds, gradient
flow.

Test instances pass ``dt_phys_s`` explicitly (and short episodes) so the
file stays fast; the full default construction is exercised by the V6
test and the registries tests.
"""

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from harness import make
from harness.envs import (
    TWO_BED_METRIC_KEYS,
    TWO_BED_SCHEMA_VERSION,
    TwoBed,
    TwoBedGymEnv,
)
from harness.envs.base import validate_problem
from harness.envs.two_bed import DESIGN_KEYS
from harness.registry import REGISTRIES

FAST = dict(n_cycles=2, dt_ctrl_s=20.0, dt_phys_s=0.01)


def test_gym_check_env():
    check_env(TwoBedGymEnv(profile="cpu", **FAST), skip_render_check=True)


def test_registered_in_both_registries():
    assert "TwoBed-v0" in REGISTRIES["envs"].names()
    problem = make("TwoBed-v0", profile="cpu")
    assert isinstance(problem, TwoBed)
    gym_env = gymnasium.make("TwoBed-v0")  # guarded gymnasium registration
    obs, _ = gym_env.reset(seed=0)
    assert obs.dtype == np.float32 and obs.shape == (22,)


def test_problem_protocol():
    problem = TwoBed(profile="cpu")
    validate_problem(problem)
    assert problem.spec.name == "TwoBed-v0"
    assert problem.spec.kind == "dynamic"
    assert problem.spec.schema_version == TWO_BED_SCHEMA_VERSION
    assert problem.spec.metric_keys == TWO_BED_METRIC_KEYS
    assert set(DESIGN_KEYS) <= set(problem.design_space.defaults)


def test_stepwise_episode_metrics_schema():
    env = TwoBed(profile="cpu", **FAST)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (22,) and obs.dtype == np.float32
    terminated, steps = False, 0
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(
            np.array([120.0, 75.0, 0.0], dtype=np.float32))
        steps += 1
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
    assert terminated and not truncated
    metrics = info["metrics"]
    assert tuple(metrics) == TWO_BED_METRIC_KEYS  # schema stability
    assert all(np.isfinite(v) for v in metrics.values())
    assert steps > 0


def test_rollout_matches_stepwise():
    """The jitted whole-episode rollout and the gym-style stepwise path
    must produce the same episode metrics (same dt, same controls)."""
    env = TwoBed(profile="cpu", **FAST)
    rollout = env.rollout()

    env.reset(seed=0)
    terminated, metrics = False, None
    while not terminated:
        _, _, terminated, _, info = env.step(np.array([120.0, 75.0, 0.0]))
        metrics = info.get("metrics", metrics)

    for key in TWO_BED_METRIC_KEYS:
        assert rollout.summary[key] == pytest.approx(metrics[key], rel=1e-6), key


def test_composite_beds_differ():
    """Two different materials must be usable per bed (the composite-bed
    experiment basis) and must move the metrics."""
    sym = TwoBed(profile="cpu", **FAST).evaluate()
    comp = TwoBed("anchor:Silica gel RD", "anchor:Zeolite 13X (NaX)",
                  profile="cpu", **FAST).evaluate()
    assert comp["delta_q"] != sym["delta_q"]
    assert all(np.isfinite(v) for v in comp.values())


def test_recovery_gain_reported():
    """``recovery_gain`` = COP with recovery − COP without, computed against
    the identical no-recovery machine.

    The complete-swing regime (fast kinetics, phases long against the
    swing timescales) is the regime where recovery pays: with the default
    slow kinetics and short phases the film-off window delays the swings
    and costs more throughput than the exchanged heat saves — a real
    trade-off, asserted separately by the V6 recovery gate.
    """
    design = {"A_k_ldf_s_1": 5.0, "A_h_wall_w_m2_k": 2000.0,
              "B_k_ldf_s_1": 5.0, "B_h_wall_w_m2_k": 2000.0,
              "recovery_ua_w_m2_k": 100.0}
    controls = {"t_ads_s": 600.0, "t_des_s": 600.0, "t_rec_s": 60.0}
    env = TwoBed(profile="cpu", design=design, n_cycles=2,
                 dt_ctrl_s=20.0, dt_phys_s=0.01)
    metrics = env.evaluate(controls=controls)
    assert metrics["Q_rec_J_m2"] > 0.0
    assert metrics["recovery_gain"] > 0.0
    norec = TwoBed(profile="cpu", design=design, n_cycles=2,
                   dt_ctrl_s=20.0, dt_phys_s=0.01).evaluate(
        controls={"t_ads_s": 600.0, "t_des_s": 600.0})
    assert metrics["COP"] == pytest.approx(norec["COP"] + metrics["recovery_gain"],
                                           rel=1e-9)


def test_grad_flows_through_episode():
    """Episode metrics are differentiable in the (per-bed) design."""
    env = TwoBed(profile="cpu", counterfactual=False, **FAST)

    def scp(q_sat):
        return env.metrics_jax({"A_q_sat_kg_kg": q_sat})["SCP_W_kg"]

    g = jax.jit(jax.grad(scp))(jnp.float64(0.35))
    assert np.isfinite(float(g))
