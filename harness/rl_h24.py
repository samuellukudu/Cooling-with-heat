"""H2.4: RL on the stochastic-source schedule bandit — the honest report.

Pipeline (ROADMAP ladder item 2, DESIGN §6 honesty posture):

1. build :class:`harness.envs.two_bed.TwoBedScenario` — the H2.2 machine
   (silica gel RD, cpu profile, FAST_DESIGN numerics) under *randomized*
   source scenarios (sinusoid + IT-load bursts inside the 60–85 °C band);
2. **floors** (DESIGN §6 mandatory): the nominal fixed schedule, and the
   best *fixed* schedule — one design CMA-ES-optimized on a reference
   scenario, then applied unchanged to every eval seed. The per-seed
   CMA-ES optimum is reported too, labeled as an *oracle* (it re-optimizes
   per scenario and is therefore not a floor, it is the upper bound);
3. train PPO via the ``rl`` backend (policy + VecNormalize saved to
   ``data_cache/rl/h24/``);
4. evaluate the policy on the *same seeds* (``evaluate_policy`` — identical
   duty envelopes, ROADMAP evaluation protocol);
5. report per-seed rows + means and the two gates, pass or fail as-is:
   ``gate_floor``    — policy mean beats the best fixed schedule;
   ``gate_margin``   — policy mean beats nominal by the H2.2 margin (≈7 %).

CLI::

    python -m harness.rl_h24 --total-timesteps 20000 --seeds 16
    python -m harness.rl_h24 --smoke          # CI-sized everything
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .envs import TwoBed, TwoBedScenario
from .envs.base import objective_value

# The H2.2 gate machine (tests/harness/test_h2_2_schedule.py): fast numerics
# so one episode rollout costs ~0.4 s at the default horizon.
FAST_DESIGN = {
    "A_k_ldf_s_1": 4.0, "A_h_wall_w_m2_k": 2000.0, "A_hx_mass_factor": 1.0,
    "B_k_ldf_s_1": 4.0, "B_h_wall_w_m2_k": 2000.0, "B_hx_mass_factor": 1.0,
}
REFERENCE_SEED = 12345  # the scenario the fixed schedule is optimized on


def build_problem(*, horizon_s: float = 3600.0, dt_phys_s: float = 0.05,
                  burst_prob: float = 0.35) -> TwoBedScenario:
    bed = TwoBed("anchor:Silica gel RD", profile="cpu", design=FAST_DESIGN,
                 n_cells=8, dt_phys_s=dt_phys_s, counterfactual=False)
    return TwoBedScenario(bed, horizon_s=horizon_s, dt_phys_s=dt_phys_s,
                          burst_prob=burst_prob)


def build_floors(problem: TwoBedScenario, seeds: "list[int]", *,
                 floor_budget: int, seed_base: int = 1000) -> dict:
    """Nominal + fixed-schedule floor + per-seed oracle (for the report).

    The fixed schedule is CMA-ES-optimized once on the reference scenario and
    then applied unchanged to every eval seed — the DESIGN §6 "best fixed
    schedule" floor. The oracle re-optimizes per seed (upper bound, labeled).
    """
    from . import optimize

    objective = problem.objective
    # --- nominal (the profile's fixed schedule), per seed -------------------
    nominal = {}
    for s in seeds:
        metrics = problem.evaluate_seed(None, s)
        nominal[s] = {"objective": float(objective_value(objective, metrics)),
                      "metrics": metrics}
    # --- best FIXED schedule: optimize once on the reference scenario -------
    problem.freeze_scenario(REFERENCE_SEED)
    res_fixed = optimize(problem, objective, backend="search", method="cmaes",
                         budget=floor_budget, seed=seed_base)
    fixed_design = dict(res_fixed.best_design)
    fixed = {}
    for s in seeds:
        metrics = problem.evaluate_seed(fixed_design, s)
        fixed[s] = {"objective": float(objective_value(objective, metrics)),
                    "metrics": metrics}
    # --- oracle: re-optimize per seed (upper bound, NOT a floor) ------------
    oracle = {}
    for i, s in enumerate(seeds):
        problem.freeze_scenario(s)
        res = optimize(problem, objective, backend="search", method="cmaes",
                       budget=floor_budget, seed=seed_base + 1 + i)
        oracle[s] = {"objective": float(res.best_objective),
                     "metrics": dict(res.best_metrics),
                     "design": dict(res.best_design)}
    return {"nominal": nominal, "fixed": fixed, "fixed_design": fixed_design,
            "reference_seed": REFERENCE_SEED, "floor_budget": floor_budget,
            "oracle": oracle}


def train_policy(problem: TwoBedScenario, *, total_timesteps: int, seed: int,
                 save_dir: Path, n_steps: int = 64):
    """PPO via the rl backend; returns ``(result, model, vec_normalize)``."""
    from . import optimize
    from .backends import load_policy

    res = optimize(problem, problem.objective, backend="rl",
                   total_timesteps=total_timesteps, seed=seed,
                   n_eval_episodes=3, n_steps=n_steps, save_dir=str(save_dir))
    model, vn = load_policy(save_dir)
    return res, model, vn


def report(problem: TwoBedScenario, floors: dict, policy_rows: "list[dict]",
           *, extra: "dict | None" = None) -> dict:
    """Assemble the honest H2.4 report (per-seed rows + means + gates)."""
    seeds = [r["seed"] for r in policy_rows]
    rows = []
    for r in policy_rows:
        s = r["seed"]
        rows.append({
            "seed": s,
            "nominal": floors["nominal"][s]["objective"],
            "fixed": floors["fixed"][s]["objective"],
            "oracle": floors["oracle"][s]["objective"],
            "policy": r["objective"],
        })
    mean = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    means = {k: mean(k) for k in ("nominal", "fixed", "oracle", "policy")}
    gates = {
        # ROADMAP: beats the best fixed schedule (the mandatory floor).
        "gate_floor": bool(means["policy"] > means["fixed"]),
        # ROADMAP: beats the H2.2 margin (≈7 % over the nominal) honestly.
        "gate_margin": bool(means["policy"] > 1.07 * means["nominal"]),
    }
    return {"rows": rows, "means": means, "gates": gates,
            "fixed_design": floors["fixed_design"],
            "reference_seed": floors["reference_seed"],
            "floor_budget": floors["floor_budget"],
            "gate_note": ("oracle re-optimizes per scenario and is an upper "
                          "bound, not a floor (DESIGN §6 honesty posture)"),
            **(extra or {})}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="H2.4: PPO on the stochastic-source schedule bandit.")
    ap.add_argument("--total-timesteps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--floor-budget", type=int, default=24)
    ap.add_argument("--horizon-s", type=float, default=3600.0)
    ap.add_argument("--dt-phys-s", type=float, default=0.05)
    ap.add_argument("--ppo-n-steps", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data_cache/rl/h24_report.json"))
    ap.add_argument("--save-dir", type=Path, default=Path("data_cache/rl/h24"))
    ap.add_argument("--smoke", action="store_true",
                    help="CI-sized: 2 seeds, tiny budgets, 600 s horizon")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = min(args.seeds, 2)
        args.floor_budget = min(args.floor_budget, 4)
        args.total_timesteps = min(args.total_timesteps, 256)
        args.horizon_s = min(args.horizon_s, 600.0)

    t0 = time.time()
    seeds = list(range(args.seeds))
    problem = build_problem(horizon_s=args.horizon_s, dt_phys_s=args.dt_phys_s)
    print(f"H2.4: horizon {args.horizon_s:g} s, dt {args.dt_phys_s:g} s → "
          f"{problem.n_steps} physics steps/episode; {len(seeds)} seeds")

    print("building floors (nominal / fixed / oracle) …", flush=True)
    floors = build_floors(problem, seeds, floor_budget=args.floor_budget,
                          seed_base=1000 + args.seed)
    print(f"training PPO ({args.total_timesteps} timesteps) …", flush=True)
    res, model, vn = train_policy(problem, total_timesteps=args.total_timesteps,
                                  seed=args.seed, save_dir=args.save_dir,
                                  n_steps=args.ppo_n_steps)
    policy_rows = evaluate_seeds(problem, model, vn, seeds)
    rep = report(problem, floors, policy_rows,
                 extra={"total_timesteps": args.total_timesteps,
                        "seconds": time.time() - t0,
                        "rl_best_objective": float(res.best_objective)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))

    print(f"\n{'seed':>6}{'nominal':>10}{'fixed':>10}{'oracle':>10}{'policy':>10}")
    for r in rep["rows"]:
        print(f"{r['seed']:>6}{r['nominal']:>10.4f}{r['fixed']:>10.4f}"
              f"{r['oracle']:>10.4f}{r['policy']:>10.4f}")
    m = rep["means"]
    print(f"{'MEAN':>6}{m['nominal']:>10.4f}{m['fixed']:>10.4f}"
          f"{m['oracle']:>10.4f}{m['policy']:>10.4f}")
    for k, v in rep["gates"].items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print(f"report: {args.out}  policy: {args.save_dir}")
    if not all(rep["gates"].values()):
        print("(negative results are results — DESIGN §6 honesty posture)")


def evaluate_seeds(problem: TwoBedScenario, model, vn, seeds: "list[int]") -> "list[dict]":
    from .backends import evaluate_policy

    return evaluate_policy(problem, model, problem.objective, seeds,
                           vec_normalize=vn)


if __name__ == "__main__":
    main()
