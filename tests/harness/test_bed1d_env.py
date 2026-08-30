"""H1.2 gates (DESIGN §12): Bed1D-v0 env — Gymnasium checker, metric
schema stability, rollout/step agreement, control sensitivity, gradient
flow.

Test instances pass ``dt_phys_s`` explicitly (and short episodes) so the
file stays fast; production use auto-computes the worst-case dt from the
design bounds. The full default construction is exercised by the V3 test
and the registries tests.
"""

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from harness import make
from harness.envs import BED1D_SCHEMA_VERSION, BED_METRIC_KEYS, Bed1D, Bed1DGymEnv
from harness.envs.bed1d import DESIGN_KEYS
from harness.registry import REGISTRIES

# dt below the default design's conduction CFL (0.0156 s at L=2mm, N=16,
# k_eff=0.3) — fine for the fixed designs used here.
FAST = dict(n_cycles=1, dt_ctrl_s=20.0, dt_phys_s=0.01)


@pytest.fixture(scope="module")
def fast_gym_env():
    return Bed1DGymEnv(profile="cpu", **FAST)


def test_gym_check_env(fast_gym_env):
    check_env(fast_gym_env, skip_render_check=True)


def test_registered_in_both_registries():
    assert "Bed1D-v0" in REGISTRIES["envs"].names()
    problem = make("Bed1D-v0")
    assert isinstance(problem, Bed1D)
    gym_env = gymnasium.make("Bed1D-v0")  # guarded gymnasium registration
    obs, _ = gym_env.reset(seed=0)
    assert obs.dtype == np.float32


def test_problem_protocol():
    problem = Bed1D(profile="cpu")
    from harness.envs.base import validate_problem

    validate_problem(problem)
    assert problem.spec.name == "Bed1D-v0"
    assert problem.spec.kind == "dynamic"
    assert problem.spec.schema_version == BED1D_SCHEMA_VERSION
    assert problem.spec.metric_keys == BED_METRIC_KEYS
    assert set(DESIGN_KEYS) <= set(problem.design_space.defaults)


def test_stepwise_episode_metrics_schema():
    env = Bed1D(profile="cpu", n_cycles=2, dt_ctrl_s=20.0, dt_phys_s=0.01)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (10,) and obs.dtype == np.float32
    terminated, truncated, steps = False, False, 0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(
            np.array([75.0, 120.0], dtype=np.float32)
        )
        steps += 1
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
    assert terminated and not truncated
    metrics = info["metrics"]
    assert tuple(metrics) == BED_METRIC_KEYS  # schema stability (§12 gate)
    assert all(np.isfinite(v) for v in metrics.values())
    assert steps > 0


def test_rollout_matches_stepwise():
    """The jitted whole-episode rollout and the gym-style stepwise path
    must produce the same episode metrics (same dt, same controls)."""
    env = Bed1D(profile="cpu", n_cycles=2, dt_ctrl_s=20.0, dt_phys_s=0.01)
    rollout = env.rollout()

    env.reset(seed=0)
    terminated, metrics = False, None
    while not terminated:
        _, _, terminated, _, info = env.step(np.array([75.0, 120.0]))
        metrics = info.get("metrics", metrics)

    for key in BED_METRIC_KEYS:
        assert rollout.summary[key] == pytest.approx(metrics[key], rel=1e-6), key


def test_action_changes_episode():
    """T_f,des and t_switch are real controls: they must move the metrics.

    Fast kinetics + strong wall coupling (k_ldf=5, h=2000) so swings
    complete within the phases; n_cycles=2 gives one post-warm-up metric
    cycle (metrics cover the last n_cycles−1 cycles)."""
    design = {"k_ldf_s_1": 5.0, "h_wall_w_m2_k": 2000.0}
    base = Bed1D(profile="cpu", n_cycles=2, dt_ctrl_s=20.0, dt_phys_s=0.01,
                 design=design).evaluate()
    hot = Bed1D(profile="cpu", n_cycles=2, dt_ctrl_s=20.0, dt_phys_s=0.01,
                design=design).evaluate(controls={"t_f_des_c": 85.0})
    assert hot["delta_q"] > base["delta_q"]  # hotter regeneration desorbs further

    slow = Bed1D(profile="cpu", n_cycles=2, dt_ctrl_s=20.0, dt_phys_s=0.01,
                 design=design).evaluate(controls={"t_ads_s": 300.0, "t_des_s": 300.0})
    fast = Bed1D(profile="cpu", n_cycles=2, dt_ctrl_s=20.0, dt_phys_s=0.01,
                 design=design).evaluate(controls={"t_ads_s": 60.0, "t_des_s": 60.0})
    assert slow["SCP_W_kg"] < fast["SCP_W_kg"]  # completed swings: frequency wins


def test_grad_flows_through_episode():
    """V5 prerequisite: episode metrics are differentiable in the design.

    The sign assertion (SCP grows with q_sat) holds only when swings
    complete within the phases — under slow default kinetics the transient
    burden (valve-flip bursts, adsorbed-phase heat capacity) can reverse
    even that, so the sign is checked on the fast-kinetics design.
    """
    env = Bed1D(profile="cpu", n_cycles=2, dt_ctrl_s=20.0, dt_phys_s=0.01,
                design={"k_ldf_s_1": 5.0, "h_wall_w_m2_k": 2000.0})

    def scp(q_sat, e_char):
        return env.metrics_jax({"q_sat_kg_kg": q_sat, "e_char_j_mol": e_char})["SCP_W_kg"]

    grad_fn = jax.jit(jax.grad(scp, argnums=(0, 1)))
    g_q, g_e = grad_fn(jnp.float64(0.35), jnp.float64(4500.0))
    assert np.isfinite(float(g_q)) and np.isfinite(float(g_e))
    assert float(g_q) > 0.0


def test_discrete_action_mode():
    env = Bed1D(profile="cpu", action_mode="discrete", **FAST)
    obs, _ = env.reset(seed=0)
    assert env.action_space.n == 2
    obs, reward, terminated, truncated, info = env.step(1)
    assert info["phase"] == "des"  # connect_des flips within one control step
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["phase"] == "ads"
