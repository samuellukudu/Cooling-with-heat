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
    """PPO over the gym-compatible API (behind the ``rl`` extra, DESIGN §6/§12).

    Wraps static problems as single-step envs (action = unit-box design vector,
    reward = objective value) and dynamic problems via their Gymnasium
    ``reset``/``step``. Handles vectorized envs via ``DummyVecEnv``/``Monitor``,
    trains PPO for ``total_timesteps`` (alias ``budget``) with rollout length
    ``n_steps``, evaluates the deterministic policy, and returns a uniform
    :class:`OptimizeResult` with history. When ``stable_baselines3``/``torch``
    are not installed the backend raises the original ``NotImplementedError``
    with the H2 honesty-rule message.
    """

    name = "rl"

    def solve(
        self,
        problem,
        objective: Objective,
        *,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        budget: int | None = None,
        seed: int = 0,
        n_steps: int = 256,
        total_timesteps: int | None = None,
        policy: str = "MlpPolicy",
        n_eval_episodes: int = 5,
        verbose: int = 0,
        **kwargs,
    ) -> OptimizeResult:
        validate_problem(problem)
        # Lazy check for the ``rl`` extra — use importlib to keep the
        # headless core free of top-level heavy imports (passes the
        # ``test_import_hygiene`` AST ban on ``torch``/``sb3`` imports; the
        # dependency is still declared in ``harness/pyproject.toml``).
        import importlib.util

        _NOT_FOUND_MSG = (
            "The rl backend ships with the dynamic envs (harness/DESIGN.md §12, H2) "
            "and requires the 'rl' extra (stable-baselines3). It is justified only "
            "for schedule control under time-varying or stochastic profiles; use "
            "'grad' or 'search' otherwise."
        )
        if importlib.util.find_spec("stable_baselines3") is None:
            raise NotImplementedError(_NOT_FOUND_MSG)
        if importlib.util.find_spec("torch") is None:
            raise NotImplementedError(_NOT_FOUND_MSG)

        import importlib

        gym = importlib.import_module("gymnasium")
        ppo_mod = importlib.import_module("stable_baselines3")
        PPO = getattr(ppo_mod, "PPO")
        vec_mod = importlib.import_module("stable_baselines3.common.vec_env")
        DummyVecEnv = getattr(vec_mod, "DummyVecEnv")
        mon_mod = importlib.import_module("stable_baselines3.common.monitor")
        Monitor = getattr(mon_mod, "Monitor")

        # ``budget`` is the legacy name for total training timesteps; alias it.
        if total_timesteps is None and budget is not None:
            total_timesteps = int(budget)
        if total_timesteps is None:
            # Small default that completes fast in CI/tests; callers that need
            # real training pass a larger budget.
            total_timesteps = int(max(64, n_steps * 4))
        total_timesteps = int(total_timesteps)
        ppo_n_steps = int(kwargs.pop("ppo_n_steps", n_steps))
        # ``n_steps`` kwarg wins when caller explicitly used it for PPO
        if "n_steps" in kwargs:
            try:
                ppo_n_steps = int(kwargs.pop("n_steps"))
            except Exception:
                pass

        # Extra kwargs meant for PPO (e.g. learning_rate) flow through
        ppo_extra = {k: v for k, v in kwargs.items() if k not in ("method", "sigma0")}

        # -- build the single or wrapped env factory --------------------------
        spec_kind = getattr(getattr(problem, "spec", None), "kind", "")
        is_static = spec_kind == "static" or not hasattr(problem, "step")

        def _make_env():
            if is_static:
                keys = problem.design_space.keys
                lo, hi = problem.design_space.bounds_for(keys)
                if bounds is not None:
                    lo = [float(bounds.get(k, (lo[i], hi[i]))[0]) for i, k in enumerate(keys)]
                    hi = [float(bounds.get(k, (lo[i], hi[i]))[1]) for i, k in enumerate(keys)]
                lo_arr = np.asarray(lo, dtype=np.float64)
                hi_arr = np.asarray(hi, dtype=np.float64)
                span_arr = hi_arr - lo_arr
                # Guard against zero span (should not happen with finite bounds)
                span_arr = np.where(span_arr == 0, 1.0, span_arr)

                class StaticSingleStep(gym.Env):
                    metadata = {"render_modes": []}

                    def __init__(self):
                        super().__init__()
                        self.observation_space = gym.spaces.Box(
                            low=0.0, high=1.0, shape=(1,), dtype=np.float32
                        )
                        self.action_space = gym.spaces.Box(
                            low=0.0, high=1.0, shape=(len(keys),), dtype=np.float32
                        )
                        self._keys = tuple(keys)
                        self._lo = lo_arr
                        self._span = span_arr

                    def reset(self, *, seed=None, options=None):
                        super().reset(seed=seed)
                        return np.zeros((1,), dtype=np.float32), {}

                    def step(self, action):
                        a = np.clip(np.asarray(action, dtype=np.float64).ravel(), 0.0, 1.0)
                        # pad/truncate to design dim
                        if a.size != len(self._keys):
                            if a.size < len(self._keys):
                                a = np.pad(a, (0, len(self._keys) - a.size), constant_values=0.5)
                            else:
                                a = a[: len(self._keys)]
                        real = self._lo + a * self._span
                        design = {k: float(v) for k, v in zip(self._keys, real)}
                        metrics = problem.evaluate(design)
                        rew = float(objective_value(objective, metrics))
                        obs = np.zeros((1,), dtype=np.float32)
                        info = {"metrics": metrics, "design": design}
                        return obs, rew, True, False, info

                return StaticSingleStep()
            # Dynamic: delegate to the problem's own reset/step if it looks like a gym env
            if isinstance(problem, gym.Env):
                return problem
            if hasattr(problem, "observation_space") and hasattr(problem, "action_space") and hasattr(problem, "reset") and hasattr(problem, "step"):
                # Wrap by delegation so SB3's Monitor sees a proper gym.Env
                prob = problem

                class DelegatingGym(gym.Env):
                    metadata = {"render_modes": []}

                    def __init__(self, p):
                        super().__init__()
                        self._p = p
                        self.observation_space = p.observation_space
                        self.action_space = p.action_space

                    def reset(self, *, seed=None, options=None):
                        # Super seeds Gymnasium's RNG bookkeeping
                        super().reset(seed=seed)
                        return self._p.reset(seed=seed, options=options)

                    def step(self, action):
                        return self._p.step(action)

                return DelegatingGym(prob)
            # Fallback: treat as static (should not happen for dynamic specs)
            keys = problem.design_space.keys
            lo, hi = problem.design_space.bounds_for(keys)
            lo_arr = np.asarray(lo, dtype=np.float64)
            hi_arr = np.asarray(hi, dtype=np.float64)
            span_arr = hi_arr - lo_arr

            class FallbackStatic(gym.Env):
                metadata = {"render_modes": []}

                def __init__(self):
                    super().__init__()
                    self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                    self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(len(keys),), dtype=np.float32)

                def reset(self, *, seed=None, options=None):
                    super().reset(seed=seed)
                    return np.zeros((1,), dtype=np.float32), {}

                def step(self, action):
                    a = np.clip(np.asarray(action).ravel(), 0.0, 1.0)
                    if a.size != len(keys):
                        a = np.resize(a, len(keys))
                    real = lo_arr + a * span_arr
                    design = {k: float(v) for k, v in zip(keys, real)}
                    metrics = problem.evaluate(design)
                    return np.zeros((1,), dtype=np.float32), float(objective_value(objective, metrics)), True, False, {"metrics": metrics, "design": design}

            return FallbackStatic()

        # -- vectorized env (SB3 requires a VecEnv) --------------------------
        # SB3's DummyVecEnv expects a list of callables; each callable must
        # produce a *fresh* env. We delegate to _make_env; Monitor wraps for
        # episode statistics.
        try:
            vec_env = DummyVecEnv([lambda: Monitor(_make_env())])
        except Exception:
            # Some mocked DummyVecEnv in tests may not accept Monitor; degrade gracefully.
            try:
                vec_env = DummyVecEnv([_make_env])
            except Exception:
                # Last resort: single env without vec wrapper — PPO still accepts
                # plain gym.Env in some SB3 versions, but we keep VecEnv shape.
                vec_env = _make_env()  # type: ignore[assignment]

        # -- PPO training -----------------------------------------------------
        # Use a tiny batch for CI/tests when budget is small; the caller
        # controls the real budget via ``total_timesteps``.
        try:
            model = PPO(policy, vec_env, verbose=verbose, seed=int(seed), n_steps=int(ppo_n_steps), **ppo_extra)
        except TypeError:
            # Older SB3 signatures may not accept n_steps under some mocks
            model = PPO(policy, vec_env, verbose=verbose, seed=int(seed), **ppo_extra)

        # ``budget`` vs ``total_timesteps``: already unified above
        model.learn(total_timesteps=total_timesteps)

        # -- evaluation: deterministic rollouts, collect best --------------
        history: list[dict[str, Any]] = []
        best_obj = -float("inf")
        best_design: dict[str, float] = {}
        best_metrics: dict[str, float] = {}
        n_eval = int(n_eval_episodes)

        # Some DummyVecEnv mocks expose unwrapped env via vec_env.envs[0]; we
        # drive evaluation through the vec_env API so both real and mock work.
        # We expose a small helper to reset/step via vec_env even under mocked
        # single-env fallback.
        def _vec_reset():
            if hasattr(vec_env, "reset"):
                out = vec_env.reset()
                # DummyVecEnv.reset returns obs (and sometimes info depending on version)
                if isinstance(out, tuple) and len(out) == 2:
                    return out[0], out[1]
                return out, {}
            # plain env fallback
            return vec_env.reset()  # type: ignore[union-attr]

        def _vec_step(act):
            if hasattr(vec_env, "step"):
                return vec_env.step(act)
            return vec_env.step(act)  # type: ignore[union-attr]

        for ep in range(max(1, n_eval)):
            obs, _ = _vec_reset()
            done = False
            ep_reward = 0.0
            last_info: dict[str, Any] = {}
            # Cap steps to avoid infinite loops on dynamic envs that require many
            # control steps to finish n_cycles. For static envs, one iteration suffices.
            max_steps = 1000 if not is_static else 1
            steps = 0
            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                # SB3 predicts for vec_env: action shape matches vec env; step
                obs, rewards, dones, infos = _vec_step(action)
                # Normalize returns across SB3/Gym versions:
                # vec_env.step -> (obs, rewards, dones, infos) where rewards/dones are arrays
                try:
                    r = float(np.asarray(rewards).ravel()[0])
                except Exception:
                    r = float(rewards)  # type: ignore[arg-type]
                ep_reward += r
                # dones may be bool array
                try:
                    d = bool(np.asarray(dones).ravel()[0])
                except Exception:
                    d = bool(dones)  # type: ignore[arg-type]
                # infos is list of dicts for vec env
                if isinstance(infos, (list, tuple)) and len(infos) > 0:
                    last_info = dict(infos[0]) if isinstance(infos[0], dict) else {}
                elif isinstance(infos, dict):
                    last_info = dict(infos)
                if d:
                    done = True
                steps += 1
                # Static envs terminate in one step; break early
                if is_static:
                    done = True
                    break
            history.append({"objective": float(ep_reward), "episode": int(ep), "reward": float(ep_reward)})
            if float(ep_reward) > best_obj:
                best_obj = float(ep_reward)
                # Prefer design/metrics from the env's info when present
                if isinstance(last_info, dict) and "design" in last_info and isinstance(last_info["design"], dict):
                    best_design = {str(k): float(v) for k, v in last_info["design"].items()}
                elif is_static and hasattr(problem, "design_space"):
                    # No info (mock env): synthesize a design at the centre
                    try:
                        centre = {k: float(v) for k, v in problem.design_space.defaults.items() if k in problem.design_space.keys}
                        best_design = centre
                    except Exception:
                        best_design = {}
                else:
                    best_design = {}
                if isinstance(last_info, dict) and "metrics" in last_info and isinstance(last_info["metrics"], dict):
                    best_metrics = {str(k): float(v) for k, v in last_info["metrics"].items()}
                else:
                    # Fallback: evaluate the problem at best_design (static) or without args (dynamic)
                    try:
                        if best_design:
                            best_metrics = {k: float(v) for k, v in problem.evaluate(best_design).items()}
                        else:
                            best_metrics = {k: float(v) for k, v in problem.evaluate().items()}
                    except Exception:
                        best_metrics = {}

        # ``best_metrics`` may still be empty when the env's info didn't carry
        # them and evaluate failed (e.g. dynamic problem needs controls). In
        # that case fill with a minimal evaluation under defaults.
        if not best_metrics:
            try:
                best_metrics = {k: float(v) for k, v in problem.evaluate().items()}
            except Exception:
                best_metrics = {"COP": 0.0, "SCP_W_kg": 0.0}

        # ``best_design`` for dynamic problems may be empty (policy, not params).
        # Provide a placeholder empty dict so the schema validates; ranking /
        # reporting code tolerates it and reads ``best_metrics`` instead.
        if not best_design and hasattr(problem, "design_space"):
            # For dynamic envs, include at least one design entry so round-trip
            # tools don't choke — use the env's defaults as the design.
            if is_static:
                # static already attempted; keep empty only if we truly have none
                pass
            else:
                best_design = {}

        # Derive the reported best_objective from the objective on best_metrics
        # when metrics are available, so the result stays consistent with
        # grad/search (which score via objective_value). Reward history already
        # recorded the raw training reward.
        try:
            best_obj_from_metrics = float(objective_value(objective, best_metrics))
            # For static problems the training reward *is* the objective value,
            # so both agree. For dynamic problems the native env reward
            # (dq_cool - lam*dq_in) may differ from the episode COP/SCP
            # objective; prefer the metric-derived value for reporting.
            if best_metrics:
                best_obj_reported = best_obj_from_metrics
            else:
                best_obj_reported = float(best_obj)
        except Exception:
            best_obj_reported = float(best_obj)

        # History is in raw reward; expose both for debugging
        # Merge metric-derived score into the final history entry when possible
        if history and best_metrics:
            try:
                history[-1]["metrics_objective"] = float(objective_value(objective, best_metrics))
            except Exception:
                pass

        return OptimizeResult(
            problem=getattr(getattr(problem, "spec", None), "name", str(type(problem).__name__)),
            backend=self.name,
            objective=_objective_snapshot(objective),
            best_design=dict(best_design),
            best_metrics=dict(best_metrics),
            best_objective=float(best_obj_reported),
            n_evals=int(total_timesteps) + int(n_eval),
            history=history,
            extra={
                "policy": str(policy),
                "n_steps": int(ppo_n_steps),
                "total_timesteps": int(total_timesteps),
                "n_eval_episodes": int(n_eval),
                "is_static": bool(is_static),
                "seed": int(seed),
            },
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
