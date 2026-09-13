"""Qt-free run core for the simulation launcher.

Registry-driven: the env catalog comes from ``harness.REGISTRIES["envs"]``
plus ``inspect.signature`` of each registered factory — no hard-coded
per-env tables. ``execute`` turns a ``RunSpec`` into metrics/trace/sweep
payloads and is importable without Qt, so tests and the "generate Python"
button exercise exactly the code path the GUI runs.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

# Structural/kwargs parameter names every env factory shares and the launcher
# treats specially (registry pickers instead of free fields).
SPECIAL_PARAMS = ("material", "material_b", "profile", "bed")


# -- catalog -----------------------------------------------------------------


def env_names() -> tuple[str, ...]:
    import harness  # noqa: PLC0415

    return harness.REGISTRIES["envs"].names()


def _resolve_return_class(factory) -> type | None:
    """``build_x(**kwargs) -> X`` factories: resolve the class from the
    return annotation (a string under ``from __future__ import annotations``)."""
    import sys

    ann = inspect.signature(factory).return_annotation
    if ann is inspect.Parameter.empty or not isinstance(ann, str):
        return ann if isinstance(ann, type) else None
    module = sys.modules.get(getattr(factory, "__module__", ""), None)
    return getattr(module, ann, None)


def env_factory_params(env_name: str) -> dict[str, dict[str, Any]]:
    """``{param: {"default": ..., "kind": "float|int|bool|str|any"}}`` for one env.

    ``material`` / ``profile`` / ``material_b`` are included so the form can
    show them, flagged by name. ``**kwargs``-style factories are resolved
    through their return-annotation class ``__init__`` — no hard-coded
    per-env tables.
    """
    import harness  # noqa: PLC0415

    factory = harness.REGISTRIES["envs"].resolve(env_name)
    sig = inspect.signature(factory)
    out: dict[str, dict[str, Any]] = {}
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        default = None if param.default is inspect.Parameter.empty else param.default
        out[name] = {"default": default, "kind": _kind_of(default)}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        cls = _resolve_return_class(factory)
        if cls is not None:
            for name, param in inspect.signature(cls.__init__).parameters.items():
                if name in ("self",) or name in out:
                    continue
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                default = (None if param.default is inspect.Parameter.empty
                           else param.default)
                out[name] = {"default": default, "kind": _kind_of(default)}
    return out


def _kind_of(default: Any) -> str:
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int) and not isinstance(default, bool):
        return "int"
    if isinstance(default, float):
        return "float"
    if default is None:
        return "any"
    return "str"


def backend_params() -> dict[str, dict[str, dict[str, Any]]]:
    """Per-backend tunable kwargs, from the backend ``solve`` signatures."""
    import harness  # noqa: PLC0415

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name in harness.REGISTRIES["backends"].names():
        solver = harness.REGISTRIES["backends"].resolve(name)()
        params: dict[str, dict[str, Any]] = {}
        for pname, param in inspect.signature(solver.solve).parameters.items():
            if pname in ("self", "problem", "objective", "seed", "bounds"):
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            params[pname] = {"default": None if param.default is inspect.Parameter.empty
                             else param.default, "kind": _kind_of(param.default)}
        out[name] = params
    return out


# -- spec --------------------------------------------------------------------


@dataclass
class RunSpec:
    """Everything one launcher run needs (plain data — serializable)."""

    env_name: str = "Cycle0D-v0"
    material_ref: str = "anchor:Silica gel RD"
    profile_ref: str = "datacenter"
    material_b_ref: str = ""
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    design_overrides: dict[str, float] = field(default_factory=dict)
    mode: str = "evaluate"  # evaluate | optimize
    backend: str = "grad"  # grad | search | rl
    cop_weight: float = 1.0
    scp_weight: float = 0.0
    seed: int = 0
    collect_trace: bool = True
    # backend kwargs (budget / n_starts / n_steps / method / sigma0 / ...)
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    # sweep ("none" | "t_switch" | "material")
    sweep_axis: str = "none"
    sweep_min: float = 60.0
    sweep_max: float = 600.0
    sweep_steps: int = 6


@dataclass
class RunResult:
    metrics: dict[str, float]
    summary_text: str
    result: Any = None       # OptimizeResult | None
    trace: Any = None        # EpisodeTrace | None
    sweep_df: Any = None     # pandas DataFrame | None
    sweep_axis: str | None = None
    elapsed_s: float = 0.0
    python_code: str = ""


Log = Callable[[str], None]
Cancel = Callable[[], bool]


# -- execution ---------------------------------------------------------------


def build_problem(spec: RunSpec, log: Log | None = None):
    """``harness.make`` the configured problem (registry resolution inside)."""
    import harness  # noqa: PLC0415

    params = env_factory_params(spec.env_name)
    kwargs: dict[str, Any] = {}
    if "material" in params:
        kwargs["material"] = spec.material_ref or None
    if "profile" in params:
        kwargs["profile"] = spec.profile_ref or None
    if "material_b" in params and spec.material_b_ref:
        kwargs["material_b"] = spec.material_b_ref
    for k, v in spec.env_kwargs.items():
        if k in params and k not in kwargs:
            kwargs[k] = v
    problem = harness.make(spec.env_name, **kwargs)
    if spec.design_overrides and getattr(problem, "design_space", None) is not None:
        # overrides replace the defaults — the transparent optimize start point
        unknown = set(spec.design_overrides) - set(problem.design_space.defaults)
        if unknown:
            raise KeyError(f"unknown design keys {sorted(unknown)}; "
                           f"valid: {sorted(problem.design_space.defaults)}")
        for k, v in spec.design_overrides.items():
            problem.design_space.defaults[k] = float(v)
        if log:
            log(f"design defaults overridden: {spec.design_overrides}")
    if log:
        log(f"make {spec.env_name} material={spec.material_ref!r} "
            f"profile={spec.profile_ref!r} kwargs={kwargs}")
    return problem


def execute(spec: RunSpec, *, log: Log | None = None,
            is_cancelled: Cancel | None = lambda: False) -> RunResult:
    """Run one spec: sweep, evaluate, or optimize. Heavy imports happen here."""
    import time

    import numpy as np  # noqa: F401,PLC0415

    t0 = time.time()
    emit = log or (lambda _s: None)

    if spec.sweep_axis != "none":
        payload = _execute_sweep(spec, emit, is_cancelled)
    elif spec.mode == "optimize":
        payload = _execute_optimize(spec, emit)
    else:
        payload = _execute_evaluate(spec, emit)
    payload.elapsed_s = time.time() - t0
    payload.python_code = python_script(spec)
    return payload


def _execute_evaluate(spec: RunSpec, emit: Log) -> RunResult:
    problem = build_problem(spec, emit)
    design = spec.design_overrides or None
    emit("evaluating single design point …")
    metrics = problem.evaluate(design) if design else problem.evaluate()
    trace = None
    if spec.collect_trace:
        trace = safe_rollout(problem, design)
    summary = "\n".join(f"{k:>16s}  {v:.6g}" for k, v in metrics.items())
    return RunResult(metrics=dict(metrics), summary_text=summary, trace=trace)


def _execute_optimize(spec: RunSpec, emit: Log) -> RunResult:
    import harness  # noqa: PLC0415
    from harness.envs.base import Objective  # noqa: PLC0415

    problem = build_problem(spec, emit)
    objective = Objective(weights={"COP": float(spec.cop_weight),
                                   "SCP_W_kg": float(spec.scp_weight)})
    emit(f"optimize backend={spec.backend!r} seed={spec.seed} "
         f"kwargs={spec.backend_kwargs}")
    result = harness.optimize(problem, objective, backend=spec.backend,
                              seed=int(spec.seed), **spec.backend_kwargs)
    emit(f"done: best_objective={result.best_objective:.6g} "
         f"n_evals={result.n_evals}")
    trace = None
    if spec.collect_trace:
        trace = safe_rollout(problem, getattr(result, "best_design", None))
    try:
        import harness.report as report  # noqa: PLC0415

        summary = report.summary(result)
    except Exception:  # noqa: BLE001
        summary = (f"backend: {result.backend}\n"
                   f"best_objective: {result.best_objective:.6g}  "
                   f"n_evals: {result.n_evals}\n"
                   f"best_design: {result.best_design}\n"
                   f"best_metrics: {result.best_metrics}")
    return RunResult(metrics=dict(result.best_metrics), summary_text=summary,
                     result=result, trace=trace)


def _execute_sweep(spec: RunSpec, emit: Log,
                   is_cancelled: Cancel) -> RunResult:
    import pandas as pd  # noqa: PLC0415

    if spec.sweep_axis == "material":
        return _sweep_materials(spec, emit)
    return _sweep_t_switch(spec, emit, pd, is_cancelled)


def _sweep_t_switch(spec: RunSpec, emit: Log, pd: Any,
                    is_cancelled: Cancel) -> RunResult:
    from harness.envs.bed1d import Bed1D, Bed1DControls  # noqa: PLC0415

    grid = np_linspace(spec.sweep_min, spec.sweep_max, max(2, int(spec.sweep_steps)))
    if is_cancelled():
        raise RuntimeError("cancelled before start")
    emit(f"t_switch sweep grid: {len(grid)} points "
         f"[{grid[0]:.0f} … {grid[-1]:.0f}] s")
    kwargs = dict(spec.env_kwargs)
    bed_kwargs = {k: v for k, v in kwargs.items()
                  if k in ("n_cells", "n_cycles", "dt_phys_s", "soft_switch", "dt_ctrl_s")}
    bed = Bed1D(material=spec.material_ref, profile=spec.profile_ref, **bed_kwargs)
    prob = Bed1DControls(bed, t_switch_bounds=(float(grid[0]), float(grid[-1])),
                         n_cycles=int(bed_kwargs.get("n_cycles", 2)),
                         dt_phys_s=bed_kwargs.get("dt_phys_s"))
    rows = []
    for i, t in enumerate(grid):
        if is_cancelled():
            emit(f"cancelled at {i}/{len(grid)} points")
            break
        m = prob.evaluate({"t_switch_s": float(t)})
        rows.append({"t_switch_s": float(t), **{k: float(v) for k, v in m.items()
                                                if isinstance(v, (int, float))}})
        emit(f"  t_switch {t:.0f}s → COP {m['COP']:.4g} SCP {m['SCP_W_kg']:.4g}")
    df = pd.DataFrame(rows)
    best_idx = int(df["SCP_W_kg"].idxmax())
    best_row = df.loc[best_idx].to_dict()
    metrics = {k: float(v) for k, v in best_row.items()
               if isinstance(v, (int, float))}
    emit(f"sweep done: best SCP {best_row['SCP_W_kg']:.4g} "
         f"@ t_switch {best_row['t_switch_s']:.0f}s")
    return RunResult(metrics=metrics, summary_text=df.to_string(index=False),
                     sweep_df=df, sweep_axis="t_switch")


def _sweep_materials(spec: RunSpec, emit: Log) -> RunResult:
    import pandas as pd  # noqa: PLC0415
    from harness import rank as rank_mod  # noqa: PLC0415
    from harness.materials import get_material  # noqa: PLC0415

    mats = [r.strip() for r in spec.material_ref.split(",") if r.strip()]
    valid = []
    for ref in mats:
        try:
            valid.append(get_material(ref))
        except Exception as exc:  # noqa: BLE001
            emit(f"skipping unknown material {ref!r}: {exc}")
    if not valid:
        raise ValueError(f"no valid materials in {spec.material_ref!r} "
                         "(comma-separated refs, e.g. 'anchor:Silica gel RD, anchor:Zeolite 13X (NaX)')")
    emit(f"material sweep over {len(valid)} materials, "
         f"profile={spec.profile_ref!r}")
    df = rank_mod.sweep_materials(valid, profiles=[spec.profile_ref])
    best = float(df["score"].max()) if "score" in df.columns and len(df) else 0.0
    emit(f"sweep done: {len(df)} rows, best score {best:.4g}")
    return RunResult(metrics={"n_materials": float(len(df)), "best_score": best},
                     summary_text=df.to_string(index=False),
                     sweep_df=df, sweep_axis="material")


def safe_rollout(problem: Any, design: Any | None) -> Any | None:
    """Best-effort rollout with trace; static problems degrade to None."""
    attempts: list[dict[str, Any]] = []
    if design is not None:
        attempts.append({"design": design, "collect_trace": True})
        attempts.append({"design": design})
    attempts.append({"collect_trace": True})
    attempts.append({})
    for kwargs in attempts:
        try:
            return problem.rollout(**kwargs)
        except TypeError as exc:
            if "collect_trace" in str(exc) and "collect_trace" in kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k != "collect_trace"}
                try:
                    return problem.rollout(**kwargs)
                except Exception:  # noqa: BLE001
                    continue
            continue
        except Exception:  # noqa: BLE001
            continue
    return None


def np_linspace(lo: float, hi: float, n: int) -> list[float]:
    import numpy as np  # noqa: PLC0415

    return np.linspace(float(lo), float(hi), int(n)).tolist()


def python_script(spec: RunSpec) -> str:
    """Reproducible script for the current configuration."""
    kwargs: list[tuple[str, str]] = [
        ("material", repr(spec.material_ref)),
        ("profile", repr(spec.profile_ref)),
    ]
    if spec.material_b_ref:
        kwargs.append(("material_b", repr(spec.material_b_ref)))
    kwargs += [(k, repr(v)) for k, v in spec.env_kwargs.items()]
    if spec.design_overrides:
        kwargs.append(("design", repr(spec.design_overrides)))
    arg_lines = [f"    {spec.env_name!r},"] + [f"    {k}={v}," for k, v in kwargs]
    lines = ["import harness"]
    if spec.mode == "optimize":
        lines.append("from harness.envs.base import Objective")
    lines += ["", "env = harness.make("] + arg_lines + [")"]
    if spec.mode == "optimize":
        lines += [
            f"objective = Objective(weights={{'COP': {spec.cop_weight}, "
            f"'SCP_W_kg': {spec.scp_weight}}})",
            f"result = harness.optimize(env, objective, backend={spec.backend!r}, "
            f"seed={spec.seed})",
            "print(result.best_metrics)",
        ]
    else:
        lines.append("print(env.evaluate())")
    return "\n".join(lines) + "\n"
