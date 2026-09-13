"""Worker that runs a harness experiment off the Qt event loop.

The GUI thread must never import `jax`/`harness.physics` — all that happens
here in a QThread. Signals stream progress back to the main window.

QProcess fallback note:
    If JAX + Qt thread contention ever surfaces (e.g. GUI stutter during the
    first JIT compile), this worker can be promoted to a QProcess-backed
    executor: serialize ExperimentGraph to JSON, spawn `python -m harness.gui._process_worker`,
    stream stdout/stderr back via QProcess signals, and deserialize ExecutorResult.
    The current QThread design is retained for low latency on small graphs; the
    process boundary is only needed for heavy Bed1D searches (budget > 500) or
    when running on a GUI thread that must stay at 60 fps. See harness/gui/README.md
    for the migration note.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from .model import ExperimentGraph


@dataclass
class ExecutorResult:
    """Payload emitted on success."""

    result: Any  # OptimizeResult
    trace: Any   # EpisodeTrace | None
    metrics: dict[str, float]  # fallback when no optimizer wired
    summary_text: str
    python_code: str
    elapsed_s: float
    sweep_df: Any = None  # DataFrame for sweep (if Sweep node present)
    sweep_axis: str | None = None
    sweep_grid: Any = None


class JobWorker(QObject):
    """QObject that lives in a QThread and executes one ExperimentGraph."""

    started = pyqtSignal(str)          # job label
    progressed = pyqtSignal(dict)      # {"n_evals", "best_objective", ...} or history tick
    finished = pyqtSignal(object)      # ExecutorResult
    failed = pyqtSignal(str)           # error string
    log = pyqtSignal(str)              # free-form log line

    def __init__(self, parent: Any | None = None) -> None:
        super().__init__(parent)
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    @pyqtSlot(object)
    def run(self, graph: ExperimentGraph) -> None:
        t0 = time.time()
        self._cancel_requested = False
        try:
            self.started.emit(f"Running {graph.name} ...")
            # Validate before touching JAX
            errs = graph.validate()
            # Non-blocking warnings: filter only hard errors
            hard = [e for e in errs if e["code"] not in ("schedule_on_static",)]
            if hard:
                msg = "Validation failed:\n" + "\n".join(f"  - {e['message']}" for e in hard)
                self.failed.emit(msg)
                return

            # Heavy imports inside worker only
            import jax  # noqa: WPS433
            jax.config.update("jax_enable_x64", True)
            import harness  # noqa: WPS433
            from harness.envs.base import Objective  # noqa: WPS433

            mat_ref = graph.material_ref()
            prof_ref = graph.profile_ref()
            phys = graph.physics_node()
            if phys is None:
                self.failed.emit("Graph needs a Physics block.")
                return
            kind = phys.data.get("kind", "Cycle0D-v0")
            phys_kwargs = graph.physics_kwargs()
            obj_kw = graph.objective_kwargs()
            opt_kw = graph.optimizer_kwargs()
            backend_name = graph.backend_name()

            self.log.emit(f"make {kind} material={mat_ref!r} profile={prof_ref!r} kwargs={phys_kwargs}")
            # harness.make resolves registry strings internally
            # Bed1D/TwoBed need design dict vs structural kwargs split — handle generically

            # -- Sweep branch (G3): if Sweep node present, run grid sweep instead of optimize/evaluate --
            if graph.has_sweep():
                payload = self._run_sweep(graph, t0)
                if payload is not None:
                    self.finished.emit(payload)
                    return
                # fallback to normal path if sweep failed (error already emitted)

            mat_b_ref = graph.material_b_ref()
            problem = self._make_problem(kind, mat_ref, prof_ref, phys_kwargs, material_b=mat_b_ref)
            self.log.emit(f"problem spec: {problem.spec.name} kind={problem.spec.kind}")

            objective = Objective(
                weights=obj_kw["weights"],
                normalize=obj_kw["normalize"],
                constraints=obj_kw["constraints"],
                penalty_scale=obj_kw["penalty_scale"],
            )
            self.log.emit(f"objective: {objective.weights} normalize={objective.normalize}")

            # Decide whether we have an optimizer wire
            opt_node = graph.optimizer_node()
            if opt_node is None:
                # No optimizer: just evaluate
                self.log.emit("No Optimizer block — evaluating single design point …")
                metrics = problem.evaluate()
                trace = None
                if graph.collect_trace:
                    trace = self._safe_rollout(problem, None)
                # Fabricate a minimal result-like summary
                summary = "\n".join(f"{k:>16s}  {v:.6g}" for k, v in metrics.items())
                elapsed = time.time() - t0
                payload = ExecutorResult(
                    result=None, trace=trace, metrics=metrics,
                    summary_text=summary, python_code=graph.to_python(), elapsed_s=elapsed,
                )
                self.finished.emit(payload)
                return

            # Optimizer path
            self.log.emit(f"optimize backend={backend_name!r} seed={graph.seed} opt_kw={opt_kw}")
            # Extract budget etc. for harness.optimize dispatch
            solve_kwargs: dict[str, Any] = {}
            if backend_name == "search":
                if "budget" in opt_kw:
                    solve_kwargs["budget"] = int(opt_kw["budget"])
                if opt_kw.get("method") in ("cmaes", "tpe"):
                    solve_kwargs["method"] = str(opt_kw["method"])
                if opt_kw.get("sigma0") is not None:
                    solve_kwargs["sigma0"] = float(opt_kw["sigma0"])
            elif backend_name == "grad":
                for k in ("n_starts", "n_steps", "step_size"):
                    if k in opt_kw and opt_kw[k] is not None:
                        solve_kwargs[k] = opt_kw[k]
                # allow budget as alias for n_starts*n_steps? not needed
            else:
                if "budget" in opt_kw:
                    solve_kwargs["budget"] = int(opt_kw["budget"])

            result = harness.optimize(problem, objective, backend=backend_name, seed=int(graph.seed), **solve_kwargs)

            self.log.emit(f"optimize done: best_objective={result.best_objective:.6g} n_evals={result.n_evals}")
            # Trace from the best design
            trace = None
            if graph.collect_trace:
                # prefer rollout at best_design; fallback to default
                trace = self._safe_rollout(problem, getattr(result, "best_design", None))
                if trace is None:
                    trace = self._safe_rollout(problem, None)

            # Build textual summary via harness.report if available
            try:
                import harness.report as report  # noqa: WPS433

                summary_text = report.summary(result)
            except Exception:  # noqa: BLE001
                summary_text = (
                    f"problem: {result.problem} backend: {result.backend}\n"
                    f"best_objective: {result.best_objective:.6g}  n_evals: {result.n_evals}\n"
                    f"best_design: {result.best_design}\nbest_metrics: {result.best_metrics}"
                )

            # emit progressive history if available (for future streaming)
            if getattr(result, "history", None):
                try:
                    self.progressed.emit({"history_len": len(result.history),
                                          "best_objective": float(result.best_objective)})
                except Exception:  # noqa: BLE001
                    pass

            elapsed = time.time() - t0
            payload = ExecutorResult(
                result=result, trace=trace, metrics=dict(result.best_metrics),
                summary_text=summary_text, python_code=graph.to_python(), elapsed_s=elapsed,
            )
            self.finished.emit(payload)

        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}\n{tb}")

    def _run_sweep(self, graph: ExperimentGraph, t0: float) -> Any | None:
        """Execute Sweep node: grid sweep over t_switch (Bed1DControls) or material (rank)."""
        try:
            import pandas as pd  # noqa: WPS433
            import numpy as np  # noqa: WPS433
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"Sweep needs pandas/numpy: {e}")
            return None
        sweep = graph.sweep_node()
        if sweep is None:
            return None
        axis = sweep.data.get("axis", "t_switch")
        raw_grid = sweep.data.get("grid")
        budget = int(sweep.data.get("budget", 8))
        # normalize grid for t_switch axis
        if axis == "t_switch":
            if isinstance(raw_grid, (list, tuple)) and len(raw_grid) >= 2:
                try:
                    grid = [float(v) for v in raw_grid]  # type: ignore
                except Exception:
                    grid = [float(raw_grid[0]), float(raw_grid[-1])]  # type: ignore
                    grid = np.linspace(min(grid), max(grid), min(budget, 8)).tolist()
            else:
                gmin = float(sweep.data.get("grid_min", 60.0))
                gmax = float(sweep.data.get("grid_max", 600.0))
                steps = int(sweep.data.get("steps", min(budget, 8)))
                grid = np.linspace(gmin, gmax, steps).tolist()
            # cap grid size to budget (and to 12 for performance) to avoid heavy run
            if len(grid) > budget:
                grid = np.linspace(min(grid), max(grid), budget).tolist()
            if len(grid) > 12:
                # downsample for heavy guard, but keep test small (8 points typical)
                grid = np.linspace(min(grid), max(grid), 12).tolist()
            self.log.emit(f"Sweep t_switch grid {len(grid)} points: {grid[:3]}…{grid[-3:]}")
            # Build Bed1DControls problem honoring graph's material/profile/physics (n_cells, n_cycles)
            try:
                phys = graph.physics_node()
                phys_kind = phys.data.get("kind", "Bed1D-v0") if phys else "Bed1D-v0"
                # Prefer Bed1D even if Cycle0D wired; t_switch sweep only meaningful on dynamic
                # If phys is Cycle0D, still create Bed1DControls with demo design for sweep
                from harness.envs.bed1d import Bed1D, Bed1DControls  # noqa: WPS433
                from harness.materials import get_material  # noqa: WPS433
                from harness.profiles import get_profile  # noqa: WPS433
                mat_ref = graph.material_ref() or "anchor:Silica gel RD"
                prof_ref = graph.profile_ref() or "datacenter"
                # validate material/prof exist, fallback
                try:
                    get_material(mat_ref)
                except Exception:
                    mat_ref = "anchor:Silica gel RD"
                try:
                    get_profile(prof_ref)
                except Exception:
                    prof_ref = "datacenter"
                phys_kwargs = graph.physics_kwargs()
                n_cells = int(phys_kwargs.get("n_cells", 8))
                n_cycles = int(phys_kwargs.get("n_cycles", 2))
                dt_phys = phys_kwargs.get("dt_phys_s", None)
                if dt_phys is not None:
                    dt_phys = float(dt_phys)
                soft_switch = bool(phys_kwargs.get("soft_switch", False))
                # keep n_cells small for sweep speed as per spec (8)
                if n_cells > 16:
                    n_cells = 8
                bed = Bed1D(material=mat_ref, profile=prof_ref, n_cells=n_cells, n_cycles=n_cycles,
                            dt_phys_s=dt_phys, soft_switch=soft_switch)
                # Bed1DControls horizon fixed to upper bound — use same dt as Bed1D for consistency
                hi = float(max(grid))
                dt_for_ctrl = float(dt_phys) if dt_phys is not None else 0.1
                n_steps = int(round(n_cycles * 2.0 * hi / dt_for_ctrl)) + 8
                n_steps = min(n_steps, 20008)
                prob = Bed1DControls(bed, t_switch_bounds=(float(min(grid)), hi), n_cycles=n_cycles, dt_phys_s=dt_for_ctrl, n_steps=n_steps)
                # Prefer harness.control.switch_time_sweep if available (reuses demo design nuances)
                try:
                    import harness.control as ctrl  # noqa: WPS433
                    rows = ctrl.switch_time_sweep(prob, grid)
                except Exception:
                    rows = []
                    for t in grid:
                        m = prob.evaluate({"t_switch_s": float(t)})
                        rows.append({"t_switch_s": float(t), "COP": m["COP"], "SCP_W_kg": m["SCP_W_kg"], "delta_q": m["delta_q"]})
                df = pd.DataFrame(rows)
                self.log.emit(f"Sweep t_switch done: {len(df)} rows best SCP {df['SCP_W_kg'].max():.1f}")
                # summary and best metrics
                best_idx = int(df["SCP_W_kg"].idxmax())
                best_row = df.loc[best_idx].to_dict()
                metrics = {k: float(v) for k, v in best_row.items() if isinstance(v, (int, float))}
                summary_text = df.to_string(index=False)
                # best trace?
                trace = None
                if graph.collect_trace:
                    try:
                        trace = prob.rollout(design={"t_switch_s": float(best_row["t_switch_s"])})
                    except Exception:
                        trace = None
                elapsed = time.time() - t0
                payload = ExecutorResult(
                    result=None, trace=trace, metrics=metrics,
                    summary_text=summary_text, python_code=graph.to_python(), elapsed_s=elapsed,
                    sweep_df=df, sweep_axis="t_switch", sweep_grid=grid,
                )
                return payload
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                self.failed.emit(f"Sweep t_switch failed: {exc}\n{tb}")
                return None
        else:  # material axis
            # grid is list of material refs
            mat_grid = sweep.data.get("grid") or sweep.data.get("material_grid") or []
            if isinstance(mat_grid, str):
                mat_grid = [s.strip() for s in mat_grid.split(",") if s.strip()]
            if not isinstance(mat_grid, (list, tuple)) or len(mat_grid) == 0:
                self.failed.emit("Sweep material axis needs non-empty grid of material refs (e.g. anchor:Silica gel RD, anchor:zeolite 13X)")
                return None
            # cap to budget
            grid = list(mat_grid)[:budget] if budget else list(mat_grid)
            self.log.emit(f"Sweep material grid {len(grid)}: {grid}")
            try:
                import harness.rank as rank  # noqa: WPS433
                from harness.materials import get_material  # noqa: WPS433
                # validate materials
                valid = []
                for ref in grid:
                    try:
                        m = get_material(ref)
                        valid.append(m)
                    except Exception as e:
                        self.log.emit(f"Skipping unknown material {ref!r}: {e}")
                if not valid:
                    self.failed.emit(f"No valid materials in grid {grid}")
                    return None
                prof_ref = graph.profile_ref() or "datacenter"
                profiles = [prof_ref]
                df = rank.sweep_materials(valid, profiles=profiles)
                self.log.emit(f"Sweep material done: {len(df)} rows")
                metrics = {"n_materials": len(df), "best_score": float(df["score"].max()) if "score" in df.columns else 0.0}
                summary_text = df.to_string(index=False)
                trace = None
                elapsed = time.time() - t0
                payload = ExecutorResult(
                    result=None, trace=trace, metrics=metrics,
                    summary_text=summary_text, python_code=graph.to_python(), elapsed_s=elapsed,
                    sweep_df=df, sweep_axis="material", sweep_grid=grid,
                )
                return payload
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                self.failed.emit(f"Sweep material failed: {exc}\n{tb}")
                return None

    def _make_problem(self, kind: str, mat_ref: Any, prof_ref: Any, phys_kwargs: dict[str, Any], material_b: Any | None = None) -> Any:
        import harness  # noqa: WPS433

        # Known structural keys per env (others go to design)
        structural_keys = {
            "Cycle0D-v0": {"hx_mass_factor"},
            "Bed1D-v0": {"n_cells", "n_cycles", "dt_phys_s", "soft_switch", "dt_ctrl_s", "lam", "action_mode", "counterfactual"},
            "Bed1DControls-v0": {"n_cells", "n_cycles", "dt_phys_s", "soft_switch", "dt_ctrl_s", "lam"},
            "TwoBed-v0": {"n_cells", "n_cycles", "dt_phys_s", "dt_ctrl_s", "lam", "counterfactual"},
            "TwoBedSchedule-v0": {"n_cells", "n_cycles", "dt_phys_s", "dt_ctrl_s", "lam", "horizon_s"},
            "ForcedConv-v0": {"nx", "ny", "dt", "n_steps", "alpha", "Lx", "Ly", "source_amplitude"},
            "Cloak2D-v0": {"nx", "ny", "dt", "n_steps", "L", "r_in", "r_out", "bg"},
            "AbsorptionCycle-v0": {"cycle_time_s", "ua_loss_W_K_per_kg", "t_amb_c"},
            "Thermoelectric-v0": set(),
            "NaturalConv-v0": {"L_m"},
            "Telegrapher-v0": {"n_cells", "dt", "t_end", "alpha", "tau"},
            "Thermoelectric1D-v0": {"n_cells", "dt", "n_steps"},
            "Boussinesq-v0": {"L_m", "n_cells", "t_end"},
        }
        allowed_struct = structural_keys.get(kind, set())
        # Split
        struct = {k: v for k, v in phys_kwargs.items() if k in allowed_struct}
        design = {k: v for k, v in phys_kwargs.items() if k not in allowed_struct}
        # Filter design to only valid keys for this env (avoid KeyError on unknown)
        try:
            if kind == "Bed1D-v0":
                from harness.envs.bed1d import DESIGN_KEYS as BED_KEYS  # noqa: WPS433

                valid = set(BED_KEYS)
                design = {k: v for k, v in design.items() if k in valid}
            elif kind in ("TwoBed-v0", "TwoBedSchedule-v0"):
                from harness.envs.two_bed import DESIGN_KEYS as TWOBED_KEYS  # noqa: WPS433

                # DESIGN_KEYS for TwoBed is prefixed; we already expanded L_m etc
                # keep only valid
                valid = set(TWOBED_KEYS)
                design = {k: v for k, v in design.items() if k in valid}
            elif kind == "Cycle0D-v0":
                from harness.envs.cycle0d import DESIGN_KEYS as CYCLE_KEYS  # noqa: WPS433

                valid = set(CYCLE_KEYS)
                # Cycle0D design via hx is handled as direct, not design, so drop
                design = {k: v for k, v in design.items() if k in valid}
                # for Cycle0D, hx is structural, so design should be empty
                if design:
                    # log dropped
                    self.log.emit(f"Cycle0D dropping design keys {list(design)} (use hx_mass_factor directly)")
                    design = {}
        except Exception:
            pass
        # TwoBed design keys are per-bed (A_/B_); expand generic L_m etc to both beds
        # Drop runtime controls that are not part of make() (t_rec_s is a per-step action)
        if kind in ("TwoBed-v0", "TwoBedSchedule-v0") and design:
            # t_rec_s, dt_ctrl_s are not make() design keys — they are controls; drop them for construction
            design = {k: v for k, v in design.items() if k not in ("t_rec_s", "dt_ctrl_s")}
            expanded: dict[str, Any] = {}
            for k, v in design.items():
                if k.startswith("A_") or k.startswith("B_") or k == "recovery_ua_w_m2_k":
                    expanded[k] = v
                elif k in ("L_m", "k_eff_w_m_k", "h_wall_w_m2_k", "hx_mass_factor", "k_ldf_s_1", "q_sat_kg_kg", "Q_st_j_kg", "e_char_j_mol", "n_da"):
                    # apply to both beds
                    expanded[f"A_{k}"] = v
                    expanded[f"B_{k}"] = v
                else:
                    expanded[k] = v
            design = expanded
        # For Cycle0D, direct kwargs is correct; for Bed1D/TwoBed design goes via `design` param
        # Composite support (TwoBed pair, TE1D grading): forward material_b
        make_kwargs: dict[str, Any] = {}
        if kind in ("TwoBed-v0", "TwoBedSchedule-v0", "Thermoelectric1D-v0") and material_b:
            make_kwargs["material_b"] = material_b
        try:
            if design and kind in ("Bed1D-v0", "TwoBed-v0", "TwoBedSchedule-v0", "Bed1DControls-v0"):
                return harness.make(kind, material=mat_ref, profile=prof_ref, design=design, **struct, **make_kwargs)
            return harness.make(kind, material=mat_ref, profile=prof_ref, **phys_kwargs, **make_kwargs)
        except TypeError as e:
            # Fallback: try design split if direct failed
            if "unexpected keyword" in str(e) and design:
                try:
                    return harness.make(kind, material=mat_ref, profile=prof_ref, design=design, **struct, **make_kwargs)
                except Exception:
                    pass
            # Last try: strip unknown keys
            self.log.emit(f"_make_problem fallback after {e}: struct={struct} design={design}")
            try:
                return harness.make(kind, material=mat_ref, profile=prof_ref, **struct, **make_kwargs)
            except Exception as e2:
                raise e2 from e

    def _safe_rollout(self, problem: Any, design: Any | None) -> Any | None:
        """Try rollout with collect_trace, with and without design, tolerating static problems."""
        for kwargs in (
            {"design": design, "collect_trace": True} if design is not None else {"collect_trace": True},
            {"design": design} if design is not None else {},
            {"collect_trace": True},
            {},
        ):
            # filter kwargs that cause TypeError by probing signature
            try:
                # for Cycle0D, collect_trace is invalid; for others it's valid
                # attempt with design if provided
                if design is not None and "design" not in kwargs:
                    continue
                # actually dispatch
                if design is not None:
                    try:
                        if "collect_trace" in kwargs:
                            return problem.rollout(design=design, collect_trace=True)  # type: ignore[call-arg]
                        return problem.rollout(design=design)  # type: ignore[call-arg]
                    except TypeError as e:
                        # maybe collect_trace not accepted
                        if "collect_trace" in str(e):
                            try:
                                return problem.rollout(design=design)  # type: ignore[call-arg]
                            except Exception:
                                continue
                        continue
                else:
                    try:
                        if "collect_trace" in kwargs:
                            return problem.rollout(collect_trace=True)  # type: ignore[call-arg]
                        return problem.rollout()  # type: ignore[call-arg]
                    except TypeError as e:
                        if "collect_trace" in str(e):
                            try:
                                return problem.rollout()  # type: ignore[call-arg]
                            except Exception:
                                continue
                        continue
            except Exception as e:  # noqa: BLE001
                # log and try next
                try:
                    self.log.emit(f"_safe_rollout attempt {kwargs} failed: {e}")
                except Exception:
                    pass
                continue
        return None
