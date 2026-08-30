"""Optimization backends — ``grad``, ``search``, ``rl`` (DESIGN §6).

One protocol, one result schema, three methods. The honesty rule in one
line: match the backend to the problem class — differentiable smooth
objectives take ``grad``; mixed/cheap spaces take ``search``; only schedule
control under time-varying or stochastic profiles justifies ``rl``.

Both numerical backends optimize in **unit-box coordinates**
(``x01 ∈ [0,1]^d`` mapped affinely onto the declared design bounds): this
makes a single ``step_size`` meaningful across parameters spanning orders of
magnitude (e.g. ``q_sat`` ~ 0.5 vs ``Q_st`` ~ 3e6) and lets CMA-ES use one
sigma for all axes. Results are always reported in real units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .envs.base import Objective, objective_value, validate_problem
from .registry import REGISTRIES

OPTIMIZE_RESULT_SCHEMA_VERSION = 1


@dataclass
class OptimizeResult:
    """Uniform result schema across backends (DESIGN §6)."""

    problem: str
    backend: str
    objective: dict[str, Any]
    best_design: dict[str, float]
    best_metrics: dict[str, float]
    best_objective: float
    n_evals: int
    history: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = OPTIMIZE_RESULT_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "problem": self.problem,
            "backend": self.backend,
            "objective": self.objective,
            "best_design": self.best_design,
            "best_metrics": self.best_metrics,
            "best_objective": self.best_objective,
            "n_evals": self.n_evals,
            "extra": self.extra,
        }


def _objective_snapshot(objective: Objective) -> dict[str, Any]:
    return objective.snapshot()


class GradientBackend:
    """Projected gradient ascent through the jitted rollout (optax Adam).

    Requires a problem whose ``metrics_jax`` is differentiable. Multi-start:
    start 0 from the design defaults, the rest from uniform draws in the
    unit box — deterministic under ``seed``.
    """

    name = "grad"

    def solve(
        self,
        problem,
        objective: Objective,
        *,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        budget: int | None = None,
        seed: int = 0,
        n_starts: int = 3,
        n_steps: int = 400,
        step_size: float = 0.05,
    ) -> OptimizeResult:
        validate_problem(problem)
        del budget  # budget = n_starts × n_steps for this backend
        try:
            import optax
        except ImportError as exc:  # pragma: no cover
            raise ImportError("the grad backend needs optax (pip install optax)") from exc

        keys = problem.design_space.keys
        lo, hi = problem.design_space.bounds_for(keys)
        if bounds:
            lo = [float(bounds.get(k, (lo[i], hi[i]))[0]) for i, k in enumerate(keys)]
            hi = [float(bounds.get(k, (lo[i], hi[i]))[1]) for i, k in enumerate(keys)]
        lo_arr, hi_arr = np.asarray(lo), np.asarray(hi)
        span_arr = hi_arr - lo_arr
        lo_j, span_j = jnp.asarray(lo_arr), jnp.asarray(span_arr)
        x0_unit = np.clip((np.asarray([problem.design_space.defaults[k] for k in keys]) - lo_arr) / span_arr, 0.0, 1.0)

        def score_unit(x01):
            real = lo_j + x01 * span_j
            design = {k: real[i] for i, k in enumerate(keys)}
            return objective_value(objective, problem.metrics_jax(design))

        value_and_grad = jax.jit(jax.value_and_grad(score_unit))

        def unit_to_design(x01: np.ndarray) -> dict[str, float]:
            return {k: float(v) for k, v in zip(keys, lo_arr + np.clip(x01, 0.0, 1.0) * span_arr)}

        rng = np.random.default_rng(seed)
        best_obj, best_x = -np.inf, None
        history: list[dict[str, Any]] = []
        n_evals = 0
        for start in range(max(1, n_starts)):
            x = jnp.asarray(x0_unit if start == 0 else rng.uniform(0.0, 1.0, size=len(keys)))
            optimizer = optax.adam(step_size)
            opt_state = optimizer.init(x)
            for _ in range(int(n_steps)):
                value, grad = value_and_grad(x)
                n_evals += 1
                value_f = float(value)
                history.append({"objective": value_f, **unit_to_design(np.asarray(x))})
                if value_f > best_obj:
                    best_obj, best_x = value_f, np.array(x)
                updates, opt_state = optimizer.update(grad, opt_state, x)
                # optax transforms minimize: their updates are meant to be
                # ADDED for descent, so subtract them to ascend the objective.
                x = jnp.clip(x - updates, 0.0, 1.0)

        best_design = unit_to_design(best_x)
        best_metrics = problem.evaluate(best_design)
        return OptimizeResult(
            problem=problem.spec.name,
            backend=self.name,
            objective=_objective_snapshot(objective),
            best_design=best_design,
            best_metrics=best_metrics,
            best_objective=float(objective_value(objective, best_metrics)),
            n_evals=n_evals,
            history=history,
            extra={"n_starts": n_starts, "n_steps": n_steps, "step_size": step_size, "coordinates": "unit-box"},
        )


class SearchBackend:
    """Derivative-free search: CMA-ES (``cma``) or TPE (``optuna``).

    Right problem class: mixed/structured design spaces and cheap
    evaluations. Constraints are applied as soft penalties by the shared
    objective function.
    """

    name = "search"

    def solve(
        self,
        problem,
        objective: Objective,
        *,
        method: str = "cmaes",
        bounds: Mapping[str, tuple[float, float]] | None = None,
        budget: int | None = None,
        seed: int = 0,
        sigma0: float | None = None,
    ) -> OptimizeResult:
        validate_problem(problem)
        keys = problem.design_space.keys
        lo, hi = problem.design_space.bounds_for(keys)
        if bounds:
            lo = [float(bounds.get(k, (lo[i], hi[i]))[0]) for i, k in enumerate(keys)]
            hi = [float(bounds.get(k, (lo[i], hi[i]))[1]) for i, k in enumerate(keys)]
        lo_arr, hi_arr = np.asarray(lo), np.asarray(hi)
        span_arr = hi_arr - lo_arr
        x0_unit = np.clip((np.asarray([problem.design_space.defaults[k] for k in keys]) - lo_arr) / span_arr, 0.01, 0.99)

        def unit_to_design(x01: np.ndarray) -> dict[str, float]:
            return {k: float(v) for k, v in zip(keys, lo_arr + np.clip(x01, 0.0, 1.0) * span_arr)}

        history: list[dict[str, Any]] = []
        n_evals = 0

        def score(x01: np.ndarray) -> float:
            nonlocal n_evals
            n_evals += 1
            metrics = problem.evaluate(unit_to_design(np.asarray(x01)))
            value = float(objective_value(objective, metrics))
            if len(history) < 2000:
                history.append({"objective": value, **unit_to_design(np.asarray(x01))})
            return value

        method = method.lower()
        if method == "cmaes":
            import cma

            budget = int(budget) if budget else 300 * len(keys)
            sigma = float(sigma0) if sigma0 else 0.25
            es = cma.CMAEvolutionStrategy(
                list(x0_unit),
                sigma,
                {"bounds": [[0.0] * len(keys), [1.0] * len(keys)], "seed": int(seed), "maxfevals": budget, "verbose": -9},
            )
            es.optimize(lambda x01: -score(x01))
            best_design = unit_to_design(np.asarray(es.result.xbest))
            best_objective = -float(es.result.fbest)
        elif method == "tpe":
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            budget = int(budget) if budget else 200

            def trial_objective(trial: "optuna.Trial") -> float:
                x01 = np.asarray([trial.suggest_float(k, 0.0, 1.0) for k in keys])
                return score(x01)

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(seed)))
            study.optimize(trial_objective, n_trials=budget)
            best_design = unit_to_design(np.asarray([study.best_params[k] for k in keys]))
            best_objective = float(study.best_value)
        else:
            raise ValueError(f"unknown search method {method!r}; use 'cmaes' or 'tpe'")

        best_metrics = problem.evaluate(best_design)
        return OptimizeResult(
            problem=problem.spec.name,
            backend=f"{self.name}:{method}",
            objective=_objective_snapshot(objective),
            best_design=best_design,
            best_metrics=best_metrics,
            best_objective=float(objective_value(objective, best_metrics)),
            n_evals=n_evals,
            history=history,
            extra={"method": method, "budget": budget, "coordinates": "unit-box"},
        )


class RLBackend:
    """PPO over the gym-compatible API (behind the ``rl`` extra).

    Deliberately not implemented at H0: it ships with the dynamic envs
    (DESIGN §12, H2) and is justified only for schedule control under
    time-varying or stochastic profiles.
    """

    name = "rl"

    def solve(self, problem, objective, **kwargs) -> OptimizeResult:
        raise NotImplementedError(
            "The rl backend ships with the dynamic envs (harness/DESIGN.md §12, H2) "
            "and requires the 'rl' extra (stable-baselines3). It is justified only "
            "for schedule control under time-varying or stochastic profiles; use "
            "'grad' or 'search' otherwise."
        )


def optimize(
    problem,
    objective: Objective,
    *,
    backend: "str | Any" = "grad",
    seed: int = 0,
    **kwargs,
) -> OptimizeResult:
    """Run a registered backend on a problem (DESIGN §3 top-level API)."""
    validate_problem(problem)
    solver_factory = backend if callable(backend) and not isinstance(backend, str) else REGISTRIES["backends"].resolve(str(backend))
    solver = solver_factory()
    if not hasattr(solver, "solve"):
        raise TypeError(f"backend {backend!r} has no solve() method")
    return solver.solve(problem, objective, seed=seed, **kwargs)


REGISTRIES["backends"].register("grad", GradientBackend)
REGISTRIES["backends"].register("search", SearchBackend)
REGISTRIES["backends"].register("rl", RLBackend)
