"""Named dataset loaders for deep-learning development (thin façade).

One import gives DL code every organized dataset with provenance — no
scattering of ``data_cache`` paths through models/training code:

    from datasets import datasets

    X, Y, prov = datasets.material_table()      # features + D-A labels per material
    fits, prov  = datasets.isotherm_fits()      # raw per-isotherm D-A fits
    conds, prov = datasets.pinn_conditions()    # real-condition rows for the bed PINN
    ok, bad     = datasets.stability_shortlist()
    status      = datasets.status()             # what is on disk, from manifests

Every loader is read-only over ``data_cache/``; build the caches with the
per-source exporters first (see ``ACQUISITION.md``). Missing caches raise
``FileNotFoundError`` loudly with the builder command — never silently empty.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "data_cache"

# Library + thin-CLI wiring (train_baseline precedent): subpackages are flat
# module dirs, not installed packages.
for _sub in ("data", "features", "eval", "models"):
    _p = str(REPO_ROOT / "adsorbent-ml" / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

TARGETS = ["log_q_sat", "q_sat_logstd", "log_Q_st", "log_E", "n_da"]


def _require(path: Path, build_cmd: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — build it first: {build_cmd} "
            "(see adsorbent-ml/data/ACQUISITION.md)")
    return path


def _prov(**kwargs: Any) -> dict:
    return kwargs


def isotherm_fits() -> tuple[pd.DataFrame, dict]:
    """Per-isotherm Dubinin–Astakhov fits (the N2 label source)."""
    path = _require(CACHE / "fits" / "da_params.csv",
                    "python adsorbent-ml/data/fit_da.py")
    df = pd.read_csv(path)
    return df, _prov(path=str(path), n_rows=len(df))


def material_table() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(X, Y, provenance)`` — the N2 supervised table.

    X: feature matrix (pores incl. IZA framework data, family, topology,
    composition). Y: one row per material with ``name``, ``family`` and the
    D–A targets (``TARGETS``). Coverage stats come back in provenance.
    """
    import descriptors
    import labels as labels_mod

    n2 = CACHE / "n2"
    labels_path = _require(n2 / "labels.csv",
                           "python adsorbent-ml/data/labels.py")
    matched_path = _require(n2 / "matched.parquet",
                            "python adsorbent-ml/data/match_structures.py")
    labels = pd.read_csv(labels_path)
    matched = pd.read_parquet(matched_path)
    X, feature_names, coverage = descriptors.build_feature_matrix(labels, matched)
    y_cols = [c for c in ["name", "family"] + TARGETS if c in labels.columns]
    Y = labels[y_cols].copy()
    return X, Y, _prov(labels=str(labels_path), matched=str(matched_path),
                       feature_names=feature_names, coverage=coverage,
                       targets=[c for c in TARGETS if c in Y.columns])


def pinn_conditions() -> tuple[list[dict], dict]:
    """Real-condition rows (fitted + anchor materials) for the bed PINN."""
    from corpus import load_real_conditions

    return load_real_conditions()


def stability_shortlist(*, min_decomp_c: float = 150.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(pass, fail)`` feasibility tables from the MOFSimplify build."""
    import stability as stability_mod

    path = _require(CACHE / "stability" / "mofsimplify_ssd_tsd.csv",
                    "python adsorbent-ml/data/stability_export.py")
    recs = stability_mod.load_stability_csv(path)
    ok, bad = stability_mod.feasibility_filter(recs, min_decomp_c=min_decomp_c)
    cols = ["name", "water_stable", "solvent_removal_stable",
            "thermal_decomp_c", "source"]
    ok_df = pd.DataFrame([{c: getattr(r, c) for c in cols} for r in ok])
    bad_df = pd.DataFrame([{c: getattr(r, c) for c in cols} for r in bad])
    return ok_df, bad_df


def structures_inventory() -> dict[str, dict]:
    """Counts + directories of every CIF cache on disk (per source)."""
    sources = {
        "core_mof": CACHE / "core_mof" / "structures",
        "qmof": CACHE / "qmof" / "relaxed_structures",
        "iza": CACHE / "iza" / "frameworks",
        "mp": CACHE / "mp" / "structures",
    }
    out = {}
    for name, d in sources.items():
        n = sum(1 for _ in d.glob("*.cif")) if d.exists() else 0
        out[name] = {"dir": str(d), "n_cif": n}
    for optimade_dir in sorted((CACHE / "optimade").glob("*")) if (CACHE / "optimade").exists() else []:
        manifest = optimade_dir / "manifest.json"
        if manifest.exists():
            m = json.loads(manifest.read_text())
            out[f"optimade/{optimade_dir.name}"] = {
                "dir": str(optimade_dir), "n_entries": m.get("n_entries"),
                "provider": m.get("provider")}
    return out


def status() -> dict[str, Any]:
    """What is on disk: manifest summaries for every built dataset."""
    out: dict[str, Any] = {"data_cache": str(CACHE)}
    manifest_paths = sorted(CACHE.glob("*/*/manifest.json")) + sorted(CACHE.glob("*/manifest.json"))
    for mp in manifest_paths:
        key = str(mp.relative_to(CACHE).with_suffix(""))
        try:
            out[key] = json.loads(mp.read_text())
        except Exception as exc:
            out[key] = {"error": str(exc)}
    out["structures"] = structures_inventory()
    return out


class _Datasets:
    """Namespace so callers can pin one import: ``from datasets import datasets``."""

    CACHE = CACHE
    isotherm_fits = staticmethod(isotherm_fits)
    material_table = staticmethod(material_table)
    pinn_conditions = staticmethod(pinn_conditions)
    stability_shortlist = staticmethod(stability_shortlist)
    structures_inventory = staticmethod(structures_inventory)
    status = staticmethod(status)
    TARGETS = TARGETS


datasets = _Datasets()

__all__ = ["datasets", "TARGETS", "isotherm_fits", "material_table",
           "pinn_conditions", "stability_shortlist", "structures_inventory",
           "status"]
