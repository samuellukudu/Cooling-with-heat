"""V7 (DESIGN §9): grad and search agree on the Cycle0D optimum and
reproduce the legacy screen; OptimizeResult schema; determinism; the rl
stub refuses politely.

Reference optimum: a 31×31 grid over (q_sat, Q_st) — exactly the legacy
screen grid in ``cooling_physics.pareto_target_window`` — evaluated with the
harness oracle. V1 (rel. err. < 1e-12 vs canonical physics) justifies using
the fast vmapped oracle for the 961-point grid. To mirror the screen, the
env's design space is restricted to the screen's two-parameter sweep
subspace (everything else fixed at material/profile defaults).

We do NOT compare designs against ``pareto_target_window``'s
``best_q_sat``: when the Pareto frontier is degenerate (single point — the
silica-gel/datacenter case), its fallback selects the *first* grid point
above a score threshold, which is not the grid argmax. Instead we assert
the harness optimum scores ≥ the window's reported best score, which holds
regardless of that quirk.
"""

import json

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import cooling_physics
import harness
from harness.envs.base import DesignSpace, Objective
from harness.envs.cycle0d import CANONICAL_NORMALIZATION, Q_SAT_BOUNDS, Q_ST_BOUNDS, build_cycle0d
from harness.physics import simulate_cycle

MATERIAL = "anchor:Silica gel RD"
GRID_STEPS = 31
Q_LO, Q_HI = Q_SAT_BOUNDS
QST_LO, QST_HI = Q_ST_BOUNDS
GRID_STEP_Q = (Q_HI - Q_LO) / (GRID_STEPS - 1)
GRID_STEP_QST = (QST_HI - QST_LO) / (GRID_STEPS - 1)


def _screen_subspace_env(profile_name):
    """Cycle0D restricted to the legacy screen's (q_sat, Q_st) sweep."""
    env = build_cycle0d(MATERIAL, profile_name)
    env.design_space = DesignSpace(
        keys=("q_sat_kg_kg", "Q_st_j_kg"),
        defaults=dict(env.design_space.defaults),
        bounds={"q_sat_kg_kg": Q_SAT_BOUNDS, "Q_st_j_kg": Q_ST_BOUNDS},
    )
    return env


def _weighted_score(profile, cop, scp):
    (cop_lo, cop_hi) = CANONICAL_NORMALIZATION["COP"]
    (scp_lo, scp_hi) = CANONICAL_NORMALIZATION["SCP_W_kg"]
    return profile.cop_weight * np.clip((cop - cop_lo) / (cop_hi - cop_lo), 0.0, 1.0) + profile.scp_weight * np.clip(
        (scp - scp_lo) / (scp_hi - scp_lo), 0.0, 1.0
    )


def _grid_argmax(profile_name):
    """(score, q_sat, Q_st) of the legacy 31×31 screen grid argmax."""
    profile = harness.get_profile(profile_name)
    env = build_cycle0d(MATERIAL, profile_name)
    d = env.design_space.defaults
    qs = jnp.linspace(Q_LO, Q_HI, GRID_STEPS)
    qsts = jnp.linspace(QST_LO, QST_HI, GRID_STEPS)
    q_grid, qst_grid = jnp.meshgrid(qs, qsts, indexing="ij")
    sim = jax.jit(
        jax.vmap(
            lambda q, qst: simulate_cycle(
                q, qst, profile.t_evap_c, profile.t_cond_c, profile.t_des_c,
                d["cycle_time_s"], d["e_char_j_mol"], d["n_da"], d["hx_mass_factor"],
            )
        )
    )
    out = sim(q_grid.ravel(), qst_grid.ravel())
    scores = _weighted_score(profile, np.asarray(out["COP"]), np.asarray(out["SCP_W_kg"]))
    i = int(np.argmax(scores))
    return float(scores[i]), float(q_grid.ravel()[i]), float(qst_grid.ravel()[i])


def _objective(profile):
    return Objective(
        weights={"COP": profile.cop_weight, "SCP_W_kg": profile.scp_weight},
        normalize=dict(CANONICAL_NORMALIZATION),
    )


@pytest.mark.parametrize("profile_name", ["cpu", "human", "vehicle", "datacenter"])
def test_v7_grad_reproduces_screen_optimum(profile_name):
    env = _screen_subspace_env(profile_name)
    score_ref, q_ref, qst_ref = _grid_argmax(profile_name)

    result = harness.optimize(env, _objective(env.profile), backend="grad", seed=0, n_starts=2, n_steps=300)

    # Continuous optimum must be at least the grid maximum (the grid is a
    # subset of the box) and within one grid cell of it.
    assert result.best_objective >= score_ref - 1e-9
    assert abs(result.best_design["q_sat_kg_kg"] - q_ref) <= GRID_STEP_Q + 1e-9
    assert abs(result.best_design["Q_st_j_kg"] - qst_ref) <= GRID_STEP_QST + 1e-9

    # Consistency with the legacy tool's reported best point (score-level).
    p = env.profile
    window = cooling_physics.pareto_target_window(
        t_evap_c=p.t_evap_c, t_cond_c=p.t_cond_c, t_des_c=p.t_des_c,
        cycle_time_sec=p.cycle_time_s, cop_weight=p.cop_weight, scp_weight=p.scp_weight,
    )
    legacy_score = _weighted_score(p, window["best_COP"], window["best_SCP_W_kg"])
    assert result.best_objective >= legacy_score - 1e-9


def test_v7_cmaes_agrees_with_grad():
    env = _screen_subspace_env("datacenter")
    score_ref, q_ref, qst_ref = _grid_argmax("datacenter")
    objective = _objective(env.profile)

    grad = harness.optimize(env, objective, backend="grad", seed=0, n_starts=2, n_steps=300)
    cmaes = harness.optimize(env, objective, backend="search", method="cmaes", seed=0, budget=1200)

    assert cmaes.best_objective >= score_ref - 1e-9
    assert abs(cmaes.best_design["q_sat_kg_kg"] - q_ref) <= GRID_STEP_Q + 1e-9
    assert abs(cmaes.best_design["Q_st_j_kg"] - qst_ref) <= GRID_STEP_QST + 1e-9
    # Cross-backend agreement: same optimum within two grid cells.
    assert abs(cmaes.best_design["q_sat_kg_kg"] - grad.best_design["q_sat_kg_kg"]) <= 2 * GRID_STEP_Q + 1e-9
    assert abs(cmaes.best_design["Q_st_j_kg"] - grad.best_design["Q_st_j_kg"]) <= 2 * GRID_STEP_QST + 1e-9
    assert abs(cmaes.best_objective - grad.best_objective) <= 0.01


def test_v7_tpe_smoke():
    env = _screen_subspace_env("datacenter")
    score_ref, _, _ = _grid_argmax("datacenter")
    result = harness.optimize(env, _objective(env.profile), backend="search", method="tpe", seed=0, budget=120)
    assert result.best_objective >= score_ref - 0.05


def test_grad_is_deterministic_under_seed():
    env = _screen_subspace_env("datacenter")
    objective = _objective(env.profile)
    first = harness.optimize(env, objective, backend="grad", seed=0, n_starts=1, n_steps=150)
    second = harness.optimize(env, objective, backend="grad", seed=0, n_starts=1, n_steps=150)
    assert first.best_design == second.best_design
    assert first.best_objective == second.best_objective


def test_rl_stub_refuses():
    env = build_cycle0d(MATERIAL, "datacenter")
    with pytest.raises(NotImplementedError, match="rl backend ships with the dynamic envs"):
        harness.optimize(env, _objective(env.profile), backend="rl")


def test_invalid_problem_gives_actionable_error():
    class NotAProblem:
        pass

    with pytest.raises(TypeError, match="does not satisfy the harness Problem protocol"):
        harness.optimize(NotAProblem(), Objective.single("COP"))


def test_result_schema_and_report_round_trip(tmp_path):
    env = build_cycle0d(MATERIAL, "datacenter")
    result = harness.optimize(env, _objective(env.profile), backend="grad", seed=0, n_starts=1, n_steps=100)

    assert result.schema_version == 1
    payload = result.to_dict()
    for key in ("schema_version", "problem", "backend", "objective", "best_design", "best_metrics", "best_objective", "n_evals"):
        assert key in payload
    # Objective weights must be preserved in the snapshot for comparability.
    assert payload["objective"]["weights"] == {"COP": 0.35, "SCP_W_kg": 0.30}

    text = harness.report.summary(result)
    assert "COP" in text and "Cycle0D-v0" in text

    out = harness.report.to_jsonl([result], tmp_path / "run.jsonl")
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["best_design"] == result.best_design


def test_objective_value_math_on_floats():
    objective = Objective(
        weights={"m": 2.0},
        normalize={"m": (0.0, 10.0)},
        constraints={"m": (">=", 5.0)},
        penalty_scale=10.0,
    )
    # 0.5 normalized × 2 = 1.0, no violation.
    assert harness.objective_value(objective, {"m": 5.0}) == pytest.approx(1.0)
    # Clamp at the top: 2.0; no violation.
    assert harness.objective_value(objective, {"m": 25.0}) == pytest.approx(2.0)
    # Violation below the constraint floor: 2×0.2 − 10×(5.0 − 1.0) = −39.8.
    assert harness.objective_value(objective, {"m": 1.0}) == pytest.approx(-39.8)


def test_unknown_search_method():
    env = build_cycle0d(MATERIAL, "datacenter")
    with pytest.raises(ValueError, match="unknown search method"):
        harness.optimize(env, Objective.single("COP"), backend="search", method="nonsense")
