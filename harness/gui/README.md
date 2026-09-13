# harness GUI — PyQt6 Desktop Workbench (Simulink × Wolfram)

> **Status:** G1 skeleton + G2 scopes + **G3/G4 Sweep & Compare polish** — desktop-only PyQt6. The headless `harness` package stays pure; this `harness.gui` subpackage is UI-only and is never imported by `harness.physics`/`envs`/`backends`.

## Quick start

```bash
# from the repo root (uv workspace) — root now re-exports harness[gui]
uv sync --extra gui              # was failing before fix: now defines gui = ["harness[gui]"] in root pyproject.toml
# alternatives (all equivalent):
uv pip install -e "harness[gui]"           # workspace member directly
uv pip install -e ".[gui]"                 # via root re-export
pip install -e "harness[gui]"              # if using pip without uv
pip install -e ".[gui]"                    # harness[gui] = PyQt6 + matplotlib

# launch
uv run python -m harness.gui     # or python -m harness.gui inside activated venv
uv run harness-gui               # console script harness/pyproject.toml:22
harness-gui                      # after pip install, .venv/bin/harness-gui
```

Headless test (CI, no display):

```bash
QT_QPA_PLATFORM=offscreen python -m harness.gui --help  # not yet CLI, but import check
QT_QPA_PLATFORM=offscreen pytest -k gui
```

## What it is

- **Simulink axis:** `QGraphicsView` node canvas. Blocks = `Material` / `Profile` / `Physics` (`Cycle0D`, `Bed1D`, `TwoBed`, `TwoBedSchedule`) / `Objective` / `Optimizer` (`search` CMA-ES/TPE, `grad` Adam) / `Scope` (`Metrics`, `History`, `Trace`, `Ranking`, `Calibration`, `Compare`, `Sweep`) + **Sweep** meta-block. Typed ports, bezier wires, inspector on the right.
- **Wolfram axis:** `Ctrl+K` command palette — e.g. `optimize Bed1D silica gel RD for SCP under datacenter` → patch is previewed then applied to the canvas. All registry names (`anchor:Silica gel RD`, `datacenter`, etc.) are auto-completed.
- **Tracer-safe execution:** `JobWorker` lives in a `QThread`; only the worker imports `jax`/`harness.physics`. The UI thread never blocks. `ExperimentGraph` is toolkit-agnostic JSON (`*.harness.json`, `schemaVersion=2`) and can generate a reproducible `harness.make`/`optimize`/`switch_time_sweep` Python script (`Generate Python`). Sweep node triggers `harness.control.switch_time_sweep` or `harness.rank.sweep_materials` via the worker.

## Layout

```
Palette (left)  |  Canvas (center)  |  Inspector (right)
                |                   |
                |  Scopes (right/bottom tabs): Metrics  History  Trace  Log  Python  Ranking  Calibration  Compare  Sweep
```

- **Palette:** searchable block list; double-click or drag to add. Filter highlights compatible drop targets. New: **Sweep** (`t_switch` and `materials` variants) under *Design & Objective*, and **Scope → Compare** / **Scope → Sweep** under *Scopes*.
- **Canvas:** pan (middle-drag / space-drag), zoom (wheel), rubber-band select, `Delete` removes block, right-drag a port to wire, double-click a node to inspect. Sweep nodes are violet accent, with `in: problem/objective` → `out: result`.
- **Inspector:** contextual `QFormLayout` per block type (material/profile registries, physics `n_cells`/`n_cycles`/`soft_switch`/`hx_mass_factor`, objective weights/normalize, optimizer `budget`/`method`/`n_starts`, **Sweep** `axis/min/max/steps/budget` grid editor with live preview, **Scope** `Compare`/`Sweep` hints).
- **Toolbar:** `New` / `Open…` / `Save` / `Save As…` / `▶ Run` / `■ Stop` / `⌘ Command…` / `Generate Python` / `Presets ▾` / `seed` / `Collect trace`.

## Validation & honesty

- Live validation bar (`✓ valid` / `⚠ N issue(s)`). Hard errors block `Run` (e.g. missing Physics, `rl` on `Cycle0D`, bad port). Soft warnings (e.g. `Schedule` wired to `Cycle0D`) show as info.
- `transport_provenance="default"` and `out_of_window` flags surface in Metrics/Ranking — same honesty as `harness/rank.py` and `DESIGN.md`.

## Experiment lifecycle

1. **Single run:** palette or `Ctrl+K` → inspector tweaks → `▶ Run` → Metrics/History/Trace/Log/Python tabs update. Worker streams `log` lines and final `ExecutorResult` (metrics + `EpisodeTrace`).
2. **Sweep (G3):** Add **Sweep** block (axis `t_switch` or `material`, grid `min/max/steps` + `budget`) → wire `Physics problem` → `Sweep problem` → `Scope result` (or leave unwired — presence of Sweep alone triggers sweep mode). **▶ Run** now dispatches through `harness.control.switch_time_sweep` (t_switch, n_cells=8, n_cycles=2) or `harness.rank.sweep_materials` (material) in the worker; `Sweep` tab shows table + Matplotlib curve (SCP/COP vs t_switch, reuses History logic) and `Generate Python` emits the sweep script.
3. **Compare (G4):** Run twice (change a parameter between runs) → **Experiment → Compare — last two results diff…** (or `Compare` scope tab). `CompareScope` shows side-by-side `harness.report.summary`-style metrics table with Δ and Δ% columns and color-coded deltas; `Scope → Compare` block is a placeholder for future wiring.
4. **Ranking:** `Scope → Ranking` (palette or `run_ranking` hook) produces sortable table from `harness.rank.sweep_materials`.
5. **Persistence:** `*.harness.json` (JSON, **`schemaVersion=2`** with migration from v1). Load restores identical `harness.make`/`optimize`/`sweep` call (same seed ⇒ same trajectory). `Generate Python` is the reproducibility anchor.

## Architecture (single process, two thread domains)

```
UI thread (QApplication)  ──signal──▶  Worker QThread (Executor)
  MainWindow                         harness.make / optimize / rollout / switch_time_sweep / rank
  CanvasScene (ExperimentGraph) ◀──   + JAX jitted scan
  ScopeTabs (matplotlib)              ──finished──▶ ScopeTabs.set_result / set_sweep_dataframe / set_compare
```

`ExperimentGraph` (`harness/gui/model.py`) is UI-agnostic; the `QGraphicsScene` is a projection. `lint-imports` enforces `gui never imported by core`.

### QProcess fallback note (G3)

The worker currently runs in a `QThread` so the canvas stays at 60 fps while the first JIT compile (~10 s for Bed1D) happens off the UI thread. If JAX + Qt thread contention ever surfaces (GUI stutter), `JobWorker` can be promoted to a `QProcess` boundary without changing the signal contract: serialize `ExperimentGraph` to `*.harness.json`, spawn `python -m harness.gui._process_worker` via `QProcess`, stream `progressed`/`log` over stdout, and deserialize `ExecutorResult` (including `sweep_df` as Parquet/CSV) on `finished`. The `executor.py` module header documents the migration path.

## Files

```
harness/gui/
  __init__.py  app.py  __main__.py
  model.py      # ExperimentGraph (NODE_TYPES incl. Sweep, SCOPE_KINDS incl. Compare/Sweep, SCHEMA_VERSION=2, sweep_kwargs, to_python sweep)
  executor.py   # JobWorker (QThread, tracer-safe, Sweep branch via switch_time_sweep / rank, QProcess fallback comment, sweep_df in ExecutorResult)
  scene.py      # CanvasScene/CanvasView/NodeItem/EdgeItem (dark-lab theme, Sweep violet accent)
  palette.py    # PaletteDock (BLOCKS incl. Sweep t_switch / materials + Scope Compare/Sweep, filter)
  inspector.py  # InspectorDock (Sweep grid editor axis/min/max/steps/budget, Compare hint, Scope kind selector)
  command_palette.py  # Wolfram-style QLineEdit + QCompleter + patch parser
  scopes.py     # Metrics/History/Trace/Text/DataFrame/SweepScope/CompareScope (Compare diff table, Sweep curve reusing History logic), ScopeTabs (Metrics/History/Trace/Log/Python/Ranking/Calibration/Compare/Sweep)
  qss/dark.qss
```

## Presets

- **Cycle0D demo** — `Material anchor:Silica gel RD` → `Profile datacenter` → `Cycle0D-v0` → `Objective COP` → `Optimizer search:cmaes budget 200` → `Metrics`. Fastest, V7-reproducible.
- **Bed1D demo** — same but `Physics Bed1D-v0 n_cells 16` → `Objective SCP_W_kg` → `Trace` extra. Dynamic trace appears only with `Collect trace` checked (first JIT compile ≈ 10 s).

## Sweep & Compare walkthrough (G3/G4)

```python
# t_switch sweep (small, fast): n_cells=8, 3–8 points, budget caps grid size
# Via GUI: palette → Sweep (t_switch) → inspector sets 60→600 s, 6 steps → Run
# Via Python (what Generate Python emits):
import harness.control as ctrl
import pandas as pd
from harness.envs.bed1d import Bed1D, Bed1DControls
bed = Bed1D(material="anchor:Silica gel RD", profile="datacenter", n_cells=8, n_cycles=2)
prob = Bed1DControls(bed, n_cycles=2, dt_phys_s=0.1, n_steps=20008)
rows = ctrl.switch_time_sweep(prob, [80, 180, 300, 480])
df = pd.DataFrame(rows)  # t_switch_s, COP, SCP_W_kg, delta_q
# SweepScope plots t_switch vs SCP (blue, left) & COP (amber, right)

# material sweep: palette → Sweep (materials) with grid ["anchor:Silica gel RD", ...]
import harness.rank as rank
df = rank.sweep_materials(["anchor:Silica gel RD", "anchor:zeolite 13X"], profiles=["datacenter"])
# Compare: run twice, then Compare tab shows Δ and Δ% per metric
```

## Known limits (G3/G4)

- Bed1D/TwoBed `search` with large budgets is heavy (each eval is a full `jax.lax.scan` rollout). For demo keep `budget ≤ 30` or use `Evaluate` (remove Optimizer) or **Sweep** (budget 6–8, n_cells=8, grid 60–600 s) for instant curves. Sweep caps at 12 points for safety.
- **Tracer-safe:** Sweep `t_switch` uses a fixed horizon (`n_steps=20008`, dt=0.1 s) sized for the max switch time, so every grid point shares the same JIT shape — differentiable and free of recompiles.
- **QProcess fallback:** still `QThread` today; the `QProcess` isolation note in `executor.py:1` explains the upgrade path (serialize graph → `QProcess` → deserialize `ExecutorResult`) if GUI stutter appears. No `ProcessPoolExecutor` change needed for now.

## Dev

```bash
QT_QPA_PLATFORM=offscreen /home/samu2505/ENTERPRISE/Cooling-with-heat/.venv/bin/python -m pytest tests/harness_gui -q
/home/samu2505/ENTERPRISE/Cooling-with-heat/.venv/bin/lint-imports  # 3 kept, 0 broken (gui never imported by core)
# offscreen headless run of Sweep/Compare polish:
QT_QPA_PLATFORM=offscreen pytest tests/harness_gui/test_sweep_compare.py -q
```

### JSON versioning (G3)

- `schemaVersion` bumped **1 → 2** to carry Sweep nodes (`axis`/`grid`/`budget`). `ExperimentGraph.from_dict` migrates v1 files automatically (injects Sweep defaults, bumps `schemaVersion` to 2); new files save with `schemaVersion=2`.
- `harness/gui/model.py:SCHEMA_VERSION` is the source of truth; `app.py:open_graph` logs migration and forces `schemaVersion=2` on save.

See `harness/DESIGN.md` §3-§6 and `/home/samu2505/.opencode/plan/harness-gui-simulink-wolfram-plan.md` for the full spec.
