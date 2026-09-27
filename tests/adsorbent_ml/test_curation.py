"""Tests for the data-curation + documentation pipeline.

Covers the standardized manifests (manifest.py), the validation gates
(validate.py — fits/labels/matched/stability/refrigerants), the generated
dataset cards (cards.py — including refrigerants + stability), the new
datasets façade helpers, and the data↔RL separation rule (no RL/backend
imports in the data layer).
"""

import json

import pandas as pd
import pytest


def _fits_df():
    return pd.DataFrame([
        {"material_id": "m1", "name": "MOF-A", "q_sat_kg_kg": 0.4,
         "q_st_j_kg": 2.5e6, "e_char_j_mol": 5000.0, "n_da": 1.8,
         "fit_flag": "ok", "fit_rmse": 0.02},
        {"material_id": "m2", "name": "Zeo-B", "q_sat_kg_kg": 0.3,
         "q_st_j_kg": None, "e_char_j_mol": 8000.0, "n_da": 2.0,
         "fit_flag": "ok", "fit_rmse": 0.03},
        {"material_id": "m3", "name": "MIL-X", "q_sat_kg_kg": None,
         "q_st_j_kg": None, "e_char_j_mol": None, "n_da": None,
         "fit_flag": "S_shaped", "fit_rmse": 0.20},
    ])


def test_manifest_roundtrip(tmp_path):
    from manifest import collect_manifests, read_manifest, write_manifest

    p = write_manifest(tmp_path / "fits", "test-stage v1",
                       extra={"counts": {"n": 3}}, args={"t-min": 280.0})
    assert p.exists()
    m = read_manifest(p)
    assert m["stage"] == "test-stage v1"
    assert m["schema_version"] == 1
    assert m["counts"] == {"n": 3}
    assert "created_unix" in m and "code_hash" in m
    collected = collect_manifests(tmp_path)
    assert "fits/manifest" in collected
    assert collected["fits/manifest"]["stage"] == "test-stage v1"


def test_validate_fits_synthetic():
    from validate import validate_fits

    rep = validate_fits(_fits_df())
    assert rep.ok
    assert rep.stats["n_usable"] == 2
    assert rep.stats["by_flag"]["ok"] == 2

    bad = _fits_df().drop(columns=["n_da"])
    rep2 = validate_fits(bad)
    assert not rep2.ok and any("missing column" in i for i in rep2.issues)

    empty = _fits_df().iloc[0:0]
    assert not validate_fits(empty).ok


def test_validate_labels_minimal():
    from validate import validate_labels

    df = pd.DataFrame([
        {"material_id": "m1", "name": "MOF-A", "family": "mof",
         "q_sat_kg_kg": 0.4, "q_st_j_kg": 2.5e6,
         "log_q_sat": -0.9, "log_Q_st": 14.7, "log_E": 8.5},
        {"material_id": "anchor:x", "name": "Silica gel RD", "family": "silica",
         "q_sat_kg_kg": 0.35, "q_st_j_kg": 2.5e6,
         "log_q_sat": -1.0, "log_Q_st": 14.7, "log_E": 8.4},
    ])
    rep = validate_labels(df)
    assert rep.ok
    assert rep.stats["n_with_Q_st"] == 2

    dup = pd.concat([df, df.iloc[0:1]], ignore_index=True)
    assert any("duplicate" in i for i in validate_labels(dup).issues)


def test_validate_matched_rates():
    from validate import validate_matched

    md = pd.DataFrame({"name": ["a", "b", "c"],
                       "match_kind": ["csd", "iza", "none"],
                       "refcode": ["X", "", ""],
                       "ambiguous": [False, False, False]})
    rep = validate_matched(md)
    assert rep.stats["n_unmatched"] == 1
    assert rep.stats["match_rate"] == pytest.approx(2 / 3)


def test_validate_stability_and_gate_counts():
    from stability import StabilityRecord
    from validate import validate_stability

    recs = [StabilityRecord("A", True, 300.0),
            StabilityRecord("B", False, 500.0),
            StabilityRecord("C", None, 500.0)]
    rep = validate_stability(recs)
    assert rep.ok  # schema-valid; unknown-water only affects the gate split
    assert rep.stats["n_rows"] == 3
    assert rep.stats["n_pass_default_gate"] == 1  # only A passes fail-closed


def test_validate_refrigerants_live():
    from validate import validate_refrigerants

    rep = validate_refrigerants()
    assert rep.ok, rep.issues
    assert set(rep.stats["fluids"]) == {"water", "methanol", "ethanol", "ammonia"}
    assert rep.stats["windows"]["water"] == (1.0, 100.0)
    assert rep.stats["windows"]["ammonia"] == (-60.0, 60.0)


def test_cards_write_all_tmp(tmp_path):
    from cards import write_dataset_cards

    written = write_dataset_cards(tmp_path)
    for slug in ("isodb", "fits", "labels", "structures",
                 "stability", "refrigerants", "pinn_corpus", "README"):
        assert slug in written
        assert written[slug].exists()

    refr = (tmp_path / "refrigerants.md").read_text()
    assert "methanol" in refr and "28 kPa" in refr
    assert "Antoine window" in refr or "Antoine" in refr
    assert "Never imported by `harness.physics`" in refr

    stab = (tmp_path / "stability.md").read_text()
    assert "MOFSimplify" in stab
    assert "feasibility_filter" in stab
    assert "unknown" in stab.lower()

    readme = (tmp_path / "README.md").read_text()
    assert "Lineage" in readme
    for slug in ("stability", "refrigerants"):
        assert slug in readme


def test_datasets_facade_new_helpers():
    from datasets import datasets

    refr, prov = datasets.refrigerants_table()
    assert set(refr["fluid"].tolist()) == {"water", "methanol", "ethanol", "ammonia"}
    assert prov["fluids"] == ["ammonia", "ethanol", "methanol", "water"]

    reports = datasets.validate_all()
    assert set(reports) == {"fits", "labels", "matched", "stability", "refrigerants"}
    for rep in reports.values():
        assert {"name", "ok", "issues", "stats"} <= set(rep)

    snap = datasets.release_snapshot()
    assert snap["schema_version"] == 1
    assert "gates_ok" in snap and "stats" in snap


def test_data_layer_independence():
    """Data layer must not import RL/backend stacks (separation rule).

    Allowed harness contact: trusted physics + material rows
    (harness.physics.*, harness.materials) used by corpus/datasets probes.
    Forbidden: envs, backends, RL (sb3/torch/gymnasium), optimizers.
    """
    import re
    from pathlib import Path

    data_dir = Path(__file__).resolve().parents[2] / "adsorbent-ml" / "data"
    forbidden = ["harness.backends", "harness.envs", "harness.report",
                 "stable_baselines3", "stable-baselines3",
                 "import gymnasium", "from gymnasium",
                 "import torch", "from torch",
                 "import optuna", "from optuna",
                 "import cma", "from cma"]
    # New pipeline modules must stay the cleanest: only the trusted-physics
    # reference (harness.physics) is allowed — never envs/backends/RL.
    strict_files = {"manifest.py": set(), "validate.py": {"harness.physics"},
                    "cards.py": set()}
    offenders: list[str] = []
    for py in sorted(data_dir.glob("*.py")):
        text = py.read_text(encoding="utf-8")
        for pat in forbidden:
            if pat in text:
                offenders.append(f"{py.name}: {pat}")
        allowed = strict_files.get(py.name)
        if allowed is not None:
            for m in re.finditer(r"^\s*(?:import|from)\s+(\S+)", text, re.M):
                mod = m.group(1)
                if mod.startswith("harness") and not any(
                        mod.startswith(a) for a in allowed):
                    offenders.append(f"{py.name}: harness import ({mod})")
    assert not offenders, f"data↔RL separation violated: {offenders}"


def test_release_snapshot_writes_file(tmp_path):
    from datasets import datasets

    snap = datasets.release_snapshot(tmp_path)
    assert (tmp_path / "release.json").exists()
    raw = json.loads((tmp_path / "release.json").read_text())
    assert raw["schema_version"] == 1
