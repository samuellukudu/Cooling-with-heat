"""H2.2 gate (DESIGN §12): schedule optimization beats the nominal fixed
schedule under a time-varying heat source.

Scenario: silica-gel RD two-bed machine on a coolant-loop source swinging
60–85 °C (the cpu profile's fluid band) on a compressed 30-min period,
2-period horizon. The nominal is the profile's fixed schedule (75 °C
setpoint capped by availability, cycle_time_s phases, no gating, no
recovery). CMA-ES optimizes the five schedule parameters
(``t_set_c, t_thresh_c, t_half_s, t_rec_s, t_dwell_s``) against the
profile-weighted normalized objective (the legacy screen's COP/SCP
weights and canonical normalization ranges).

Numerics are sized for CI: N = 8 cells, dt = 50 ms (inside the conduction
CFL of 62.5 ms and the k·dt split bound), 72 000 physics steps per
episode, ~0.4 s per evaluation, CMA-ES budget 80.

Findings this test pins down (2026-08):

- The optimized schedule (source-following setpoint ~80 °C, ~174 s
  cycles, a small recovery window) beats the nominal by ≈ 7 % on the
  profile-weighted objective — COP +17 %, SCP +9 %.
- At the optimum the request-gating is actually *blocked* (minimum
  dwell ≫ phase length): on a band where every source level gives a
  positive swing, the win comes from following the source and cycling
  faster, not from gating. Gating matters when parts of the source band
  are unusable — on the datacenter 45–70 °C loop with silica gel RD the
  band bottom is degenerate (zero swing below ~60 °C regeneration) and
  the v1 machine (no valve-closed standby) cannot coast out of it; that
  materials marginality is the H2.3 sweep's finding, and the standby
  capability is the lumped-vapour-inventory extension (Open Question 2).
"""

import numpy as np
import pytest

from harness import optimize
from harness.envs import TwoBed, TwoBedSchedule
from harness.envs.base import Objective, objective_value
from harness.envs.cycle0d import CANONICAL_NORMALIZATION
from harness.registry import REGISTRIES

# cpu profile fluid band 60–85 °C, compressed 30-min load period.
SOURCE = lambda t: 72.5 + 12.5 * np.sin(2.0 * np.pi * t / 1800.0)  # noqa: E731
FAST_DESIGN = {
    "A_k_ldf_s_1": 4.0, "A_h_wall_w_m2_k": 2000.0, "A_hx_mass_factor": 1.0,
    "B_k_ldf_s_1": 4.0, "B_h_wall_w_m2_k": 2000.0, "B_hx_mass_factor": 1.0,
}


@pytest.fixture(scope="module")
def schedule_problem():
    bed = TwoBed("anchor:Silica gel RD", profile="cpu", design=FAST_DESIGN,
                 n_cells=8, dt_phys_s=0.05, counterfactual=False)
    return TwoBedSchedule(bed, source_schedule=SOURCE, horizon_s=3600.0,
                          dt_phys_s=0.05)


@pytest.fixture(scope="module")
def objective(schedule_problem):
    prof = schedule_problem.bed.profile
    w = prof.cop_weight + prof.scp_weight
    return Objective(
        weights={"COP": prof.cop_weight / w, "SCP_W_kg": prof.scp_weight / w},
        normalize=CANONICAL_NORMALIZATION,
    )


def test_registered_and_nominal_is_the_fixed_schedule(schedule_problem):
    assert "TwoBedSchedule-v0" in REGISTRIES["envs"].names()
    nom = schedule_problem.nominal_controls()
    prof = schedule_problem.bed.profile
    assert nom["t_set_c"] == prof.t_des_c
    assert nom["t_half_s"] == prof.cycle_time_s
    assert nom["t_rec_s"] == 0.0
    # Blocking dwell: the nominal never acts on the request bits.
    assert nom["t_dwell_s"] >= 2.0 * schedule_problem.t_half_bounds[1]


def test_schedule_problem_consistency(schedule_problem):
    """The jitted metrics path and the eager rollout agree; the metric
    schema is stable."""
    prob = schedule_problem
    metrics = prob.evaluate()
    trace = prob.rollout()
    assert tuple(metrics) == prob.spec.metric_keys
    for key in ("COP", "SCP_W_kg", "Q_cool_J_kg", "Q_in_J_kg"):
        assert trace.summary[key] == pytest.approx(metrics[key], rel=1e-9), key


def test_schedule_optimization_beats_nominal(schedule_problem, objective):
    prob = schedule_problem
    nominal = prob.evaluate()
    nominal_obj = float(objective_value(objective, nominal))
    assert nominal["Q_cool_J_kg"] > 0.0, "the nominal must be viable on this band"

    res = optimize(prob, objective, backend="search", method="cmaes",
                   budget=80, seed=0)

    print(f"\nH2.2 nominal: obj {nominal_obj:.4f}  COP {nominal['COP']:.4f} "
          f"SCP {nominal['SCP_W_kg']:.1f}")
    print(f"H2.2 optimized: obj {res.best_objective:.4f}  "
          f"COP {res.best_metrics['COP']:.4f} SCP {res.best_metrics['SCP_W_kg']:.1f}")
    print(f"H2.2 policy: { {k: round(v, 1) for k, v in res.best_design.items()} }")

    assert res.best_objective > 1.02 * nominal_obj, (
        f"optimized schedule {res.best_objective:.4f} must clearly beat the "
        f"nominal fixed schedule {nominal_obj:.4f}")
    assert res.best_metrics["Q_cool_J_kg"] > 0.0
    assert all(np.isfinite(v) for v in res.best_metrics.values())
