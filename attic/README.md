# Attic

Parked code: out of the packages, out of the test suite, out of import-linter
scope — but kept in the repository because it was finished, tested work that
may be revived.

> **2026-09 update:** the workbench was replaced by two focused apps — the
> simulation launcher (`harness/gui`, `harness-gui`) and the dataset explorer
> (`adsorbent-ml/gui/data_app.py`), which reuse the dark theme, scopes, and
> the corrected worker-thread pattern from here. The node canvas itself lives
> on only in this attic; restore instructions below remain valid if the
> composer UX is ever wanted again.

## gui/ — PyQt6 workbench (parked 2026-09)

The "Simulink × Wolfram" node-canvas desktop app (~5,000 lines: scene/model
graph editor, inspector, command palette, Metrics/History/Trace/Ranking/
Calibration/Compare/Sweep scopes, background QThread executor, dark QSS).
All physics blocks it offers are the cooling envs (Cycle0D, Bed1D, TwoBed,
TwoBedSchedule), so it still matches the simplified project — it was parked
purely to slim the library while the focus is PINN + RL development, where a
GUI is not on the critical path.

Status when parked: G1 skeleton + G2 scopes + G3/G4 Sweep & Compare polish;
34 offscreen Qt tests green (`gui_tests/`).

### Restore

1. `git mv attic/gui harness/gui && git mv attic/gui_tests tests/harness_gui`
2. `harness/pyproject.toml`: re-add `gui = ["PyQt6>=6.7", "matplotlib>=3.8"]`
   extra, `pytest-qt>=4.4` to `dev`, and
   `[project.scripts] harness-gui = "harness.gui.app:run"`.
3. `uv sync --extra gui && uv run pytest tests/harness_gui`.

Note: `attic/gui/command_palette.py` references the removed trial envs
(Heat1D/…); prune those palette entries when restoring.
