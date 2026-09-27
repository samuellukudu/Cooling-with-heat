"""Standardized build manifests for the data-curation pipeline.

Every exporter / builder writes one ``manifest.json`` next to its outputs so
downstream code, docs, and releases can rely on a common shape instead of
per-directory ad-hoc JSON:

    {
      "stage": "nist_isodb water-subset v1",
      "schema_version": 1,
      "created_unix": 1789000000.0,
      "code_hash": "abc1234 (dirty?) or unknown",
      "args": {...},      # CLI args that produced this build (optional)
      "counts": {...},    # stage-specific counts (optional)
      ...stage extras, flattened at top level for backward compat...
    }

Backward compatible: readers tolerate the legacy manifests (which lack
``schema_version`` / ``created_unix`` / ``code_hash``). Writers always emit
the standard fields.

Pure stdlib. Never imports harness backends, RL, or training code
(separation rule — see tests/adsorbent_ml/test_curation.py).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

SCHEMA_VERSION = 1


def code_hash() -> str:
    """Short git SHA of the working tree, ``"unknown"`` when unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        sha = (out.stdout or "").strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def write_manifest(out_dir: str | Path, stage: str, extra: dict | None = None,
                   args: dict | None = None) -> Path:
    """Write ``manifest.json`` into ``out_dir``. Returns the manifest path.

    ``extra`` holds stage-specific counts/provenance; standard fields win on
    key collisions (except ``stage``, which is always the argument).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = dict(extra or {})
    manifest.update({
        "stage": stage,
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "code_hash": code_hash(),
    })
    if args is not None:
        manifest["args"] = args
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def read_manifest(path: str | Path) -> dict:
    """Read a manifest, tolerating legacy files missing standard fields."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    return raw


def collect_manifests(cache_dir: str | Path) -> dict[str, dict]:
    """``{relative_key: manifest}`` for every ``manifest.json`` under cache.

    Mirrors ``datasets.status()`` discovery (``*/manifest.json`` +
    ``*/*/manifest.json``); unreadable files surface as ``{"error": ...}``.
    """
    cache = Path(cache_dir)
    out: dict[str, dict] = {}
    paths = sorted(cache.glob("*/manifest.json")) + sorted(cache.glob("*/*/manifest.json"))
    for mp in paths:
        # "fits/manifest.json" -> "fits/manifest"; nested caches stay distinct.
        key = str(mp.relative_to(cache))[:-len(".json")]
        try:
            out[key] = read_manifest(mp)
        except Exception as exc:  # noqa: BLE001 — surfaced, never raised
            out[key] = {"error": str(exc)}
    return out


__all__ = ["SCHEMA_VERSION", "code_hash", "collect_manifests",
           "read_manifest", "write_manifest"]
