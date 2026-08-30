"""V5 — control-gradient gates (DESIGN §10/§12 H1.5).

- **V5 gate**: ``jax.grad`` vs central finite differences on the smooth
  (soft-switched) control problem — max relative error < 1e-3 at 10
  probe points spanning the control box.
- **Open Question 3 evidence (hard-switch no-go)**: at
  ``soft_switch_frac = 0`` the episode metric is a staircase in the
  switch time (flips quantize to physics substeps), so the finite
  difference inside one stair is exactly zero while ``jax.grad`` returns
  the trivial ``−SCP/t_switch`` denominator term — nonzero and actively
  misleading (gradient ascent runs the switch time to a bound). Soft
  switching restores real gradients; the go/no-go decision is recorded
  in DESIGN §13.3.

Numerics: the demo design (anchor RD isotherm, k_LDF = 0.01 s⁻¹,
h = 500 W/m²K, N = 8, dt = 100 ms) resolves the τ_kin ≈ 100 s dynamics
with ~1e3 headroom; the SCP optimum sits at t_switch ≈ 225 s, inside
the §5.2 action window.
"""

import numpy as np
import pytest

from harness.control import demo_problem, gradient_check, switch_time_sweep

# The episode metric is smooth in the switch time between valve-event
# regroupings (which sit where k·t_switch crosses a physics-substep edge,
# jumping ~0.5 % of Q as the exact-exponential relaxation re-partitions).
# With the short test horizon (n_steps = 10000 -> at most 10 flips at
# t_switch = 100 s) and probe fractions at 1/11 of a substep, every flip
# keeps a >= 1/11-cell margin, so a +-5e-4 s FD window never crosses a
# regrouping and the finite difference sees one smooth branch.
PROBES = [
    (100.0091, 65.0), (100.0091, 75.0), (150.0091, 68.0), (200.0091, 75.0),
    (225.0091, 62.0), (250.0091, 70.0), (300.0091, 65.0), (350.0091, 75.0),
    (400.0091, 72.0), (450.0091, 80.0),
]
V5_NUMERICS = dict(t_switch_bounds=(60.0, 600.0), soft_switch=True,
                   n_steps=10000)


def test_v5_soft_switch_gradients_match_fd():
    """The V5 gate: gradients of the smooth problem are the true ones."""
    problem = demo_problem(**V5_NUMERICS)
    rows = gradient_check(problem, PROBES, h_switch=5e-4, h_t_f=0.05)
    worst = max(rows, key=lambda r: r["rel_err"])
    for r in rows:
        assert r["rel_err"] < 1e-3, r
    # both control directions carry signal at the optimum-ish probes
    any_switch = max(abs(r["grad_ad"][0]) for r in rows)
    any_t_f = max(abs(r["grad_ad"][1]) for r in rows)
    assert any_switch > 0.1 and any_t_f > 0.1


def test_hard_switch_gradient_is_blind_to_the_numerator():
    """Open Question 3, no-go side: with the physical hard valve the
    episode is a staircase in the switch time. jax.grad returns only the
    smooth denominator term −SCP/t_switch (the numerator's true response
    is invisible), so gradient ascent runs the switch time to a bound."""
    from harness.backends import GradientBackend
    from harness.envs.base import Objective

    problem = demo_problem(t_switch_bounds=(60.0, 600.0), soft_switch=False,
                           n_steps=10000)
    t_switch, t_f_des = 225.0091, 70.0
    scp = float(problem.evaluate({"t_switch_s": t_switch,
                                  "t_f_des_c": t_f_des})["SCP_W_kg"])
    rows = gradient_check(problem, [(t_switch, t_f_des)],
                          h_switch=5e-4, h_t_f=0.05)
    r = rows[0]
    # the AD gradient is exactly the blind denominator term
    assert r["grad_ad"][0] == pytest.approx(-scp / t_switch, rel=1e-6)
    # ...and the tiny-h FD sees the same (the numerator is frozen between
    # substeps): the gradient is "correct" for the staircase and useless
    # for control — SCP actually peaks at ~225 s, yet its gradient points
    # down everywhere right of the peak.
    assert r["grad_fd"][0] == pytest.approx(-scp / t_switch, rel=0.05)

    # optimization-level consequence: ascent collapses to the lower bound
    result = GradientBackend().solve(
        problem, Objective.single("SCP_W_kg"),
        n_starts=1, n_steps=20, step_size=0.05, seed=0)
    assert result.best_design["t_switch_s"] <= 61.0, result.best_design


def test_scp_sweep_has_interior_optimum_and_cop_tradeoff():
    """The headline-physics sanity behind the notebook plot: SCP peaks at
    an intermediate switch time; COP rises with switch time."""
    problem = demo_problem(t_switch_bounds=(60.0, 1200.0), soft_switch=True)
    rows = switch_time_sweep(problem, [60.0, 100.0, 150.0, 225.0, 340.0,
                                       500.0, 750.0, 1200.0])
    scp = [r["SCP_W_kg"] for r in rows]
    cop = [r["COP"] for r in rows]
    peak = int(np.argmax(scp))
    assert 0 < peak < len(scp) - 1, f"SCP optimum at the edge: {scp}"
    assert cop == sorted(cop), f"COP not monotone in switch time: {cop}"


def test_grad_backend_climbs_near_the_sweep_optimum():
    """Open Question 3, go side (part 1): with soft switching the grad
    backend's gradients are real (V5 gate) and Adam climbs from the
    default controls to within a few percent of the brute-force 2-D grid
    optimum. The residual gap (fixed-lr Adam drifts along the ridge
    instead of settling) is the recorded finding behind the §6 hybrid
    recommendation: search on switch times, grad on smooth controls
    (DESIGN §13.3)."""
    from harness.backends import GradientBackend
    from harness.envs.base import Objective

    problem = demo_problem(t_switch_bounds=(60.0, 500.0), soft_switch=True,
                           n_steps=20008)
    start = problem.evaluate()
    grid = [problem.evaluate({"t_switch_s": t, "t_f_des_c": f})
            for t in (60.0, 100.0, 150.0, 200.0, 250.0, 300.0)
            for f in (75.0, 85.0)]
    best = max(m["SCP_W_kg"] for m in grid)

    result = GradientBackend().solve(
        problem, Objective.single("SCP_W_kg"),
        n_starts=1, n_steps=100, step_size=0.005, seed=0)
    scp_opt = result.best_metrics["SCP_W_kg"]
    # a large genuine improvement over the default controls, and close to
    # the brute-force optimum
    assert scp_opt > 1.25 * start["SCP_W_kg"], (scp_opt, start)
    assert scp_opt >= 0.93 * best, (scp_opt, best)


def test_search_backend_finds_the_global_basin():
    """Open Question 3, go side (part 2): the §6 fallback — search on the
    switch times — finds the global basin that local grad ascent misses.
    This comparison is the data behind the hybrid recommendation."""
    from harness.backends import SearchBackend
    from harness.envs.base import Objective

    problem = demo_problem(t_switch_bounds=(60.0, 500.0), soft_switch=True,
                           n_steps=20008)
    grid = [problem.evaluate({"t_switch_s": t, "t_f_des_c": f})
            | {"t_switch_s": t, "t_f_des_c": f}
            for t in (60.0, 100.0, 150.0, 225.0, 340.0, 450.0)
            for f in (75.0, 85.0)]
    best = max(grid, key=lambda m: m["SCP_W_kg"])

    result = SearchBackend().solve(problem, Objective.single("SCP_W_kg"),
                                   method="cmaes", budget=80, seed=0)
    assert result.best_objective >= 0.99 * best["SCP_W_kg"], \
        (result.best_objective, best)
    assert abs(result.best_design["t_switch_s"] - best["t_switch_s"]) < 60.0
    assert abs(result.best_design["t_f_des_c"] - best["t_f_des_c"]) < 3.0
