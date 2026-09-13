"""PyQt6 desktop workbench for the harness — Simulink × Wolfram palette.

This package is UI-only; it never participates in the headless
import-linter contracts (`harness.physics` / `harness.envs` must not import it).
All harness execution is funneled through `harness.gui.executor.JobWorker`
which runs in a QThread so the canvas stays responsive.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
