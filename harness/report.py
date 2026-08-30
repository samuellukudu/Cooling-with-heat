"""Result rendering: text summaries and JSONL dumps (DESIGN §2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .backends import OptimizeResult


def summary(result: OptimizeResult) -> str:
    """Human-readable one-problem summary: objective, design, metrics."""
    lines = [
        f"problem    : {result.problem}",
        f"backend    : {result.backend}  (evals: {result.n_evals})",
        f"objective  : {result.best_objective:.6f}   {result.objective}",
    ]
    lines.append("design     :")
    for key in sorted(result.best_design):
        lines.append(f"    {key:<20s} {result.best_design[key]:>14.6g}")
    lines.append("metrics    :")
    for key in sorted(result.best_metrics):
        lines.append(f"    {key:<20s} {result.best_metrics[key]:>14.6g}")
    return "\n".join(lines)


def to_json(result: OptimizeResult) -> dict:
    return result.to_dict()


def to_jsonl(results: Iterable[OptimizeResult], path: str | Path) -> Path:
    """Append one JSON object per line (schema: OptimizeResult.to_dict)."""
    path = Path(path)
    with open(path, "a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    return path
