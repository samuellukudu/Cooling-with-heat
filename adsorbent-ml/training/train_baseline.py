"""N2-v1 training CLI: labels → match → features → grouped CV → bundle.

Thin CLI (fit_da precedent): parse → call libraries → write. Artifacts land
in ``data_cache/n2/``: ``labels.csv``, ``matched.parquet``,
``unmatched_names.csv``, ``baseline_bundle.pkl``, ``baseline_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # harness oracle + registry
for _sub in ("data", "models", "features", "eval"):
    sys.path.insert(0, str(REPO_ROOT / "adsorbent-ml" / _sub))

from descriptors import build_feature_matrix  # noqa: E402
from labels import build_labels, write_labels  # noqa: E402
from match_structures import match_materials, write_match  # noqa: E402
from ongari import load_ongari_map  # noqa: E402
from qmof import load_qmof_table  # noqa: E402
from rank_eval import system_rank_eval  # noqa: E402
from tabular_baseline import (BaselineConfig, cross_validate, fit_full,  # noqa: E402
                              predict, save_bundle)


def _material_doi_sets() -> dict[str, set[str]]:
    """ISODB hashkey → DOI set from the exported water isotherms."""
    import pandas as pd  # noqa: PLC0415

    w = pd.read_parquet(REPO_ROOT / "data_cache" / "isodb" / "water_isotherms.parquet",
                        columns=["adsorbent_hashkey", "doi"])
    return {hk: set(g["doi"].dropna().astype(str))
            for hk, g in w.groupby("adsorbent_hashkey")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the N2-v1 tabular baseline.")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data_cache" / "n2")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--splits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    labels, lab_manifest = build_labels()
    write_labels(labels, lab_manifest, args.out)
    try:
        ongari_map = load_ongari_map()
    except FileNotFoundError:
        print("[warn] data_cache/ongari tables absent — csd rung skipped "
              "(run ongari.py --fetch)")
        ongari_map = None
    qmof_table = load_qmof_table()
    matched, match_report = match_materials(
        labels[["name", "material_id"]],
        anchor_names=labels[labels["anchor"]]["name"].tolist(),
        ongari_map=ongari_map, qmof_table=qmof_table,
        material_dois=_material_doi_sets())
    from match_structures import pore_consistency  # noqa: E402

    matched = pore_consistency(matched, labels)
    bad = set(matched[matched["pore_consistent"] == False]["name"])  # noqa: E712
    bad -= set(labels[labels["anchor"]]["name"])  # anchors are gold: never gated
    train_labels = labels[~labels["name"].isin(bad)].copy()
    match_report["pore_gated_out"] = sorted(bad)
    write_match(matched, match_report, args.out)
    X, feature_names, coverage = build_feature_matrix(train_labels, matched)

    cfg = BaselineConfig(max_iter=args.max_iter, n_splits=args.splits,
                         random_state=args.seed)
    oof, cv_report = cross_validate(X, train_labels, feature_names, cfg)

    # System-level: OOF predictions (natural scale) vs true params ranking.
    oof_params = pd.DataFrame({
        "q_sat_kg_kg": np.exp(oof["log_q_sat"]),
        "q_st_j_kg": np.exp(oof["log_Q_st"]),
        "e_char_j_mol": np.exp(oof["log_E"]),
        "n_da": oof["n_da"]}, index=train_labels.index)
    true_params = train_labels[["q_sat_kg_kg", "q_st_j_kg",
                                "e_char_j_mol", "n_da"]].rename_axis(index="idx")
    has_qst = train_labels["q_st_j_kg"].notna() & oof_params["q_st_j_kg"].notna()
    rank_report = system_rank_eval(true_params[has_qst], oof_params[has_qst])

    bundle = fit_full(X, train_labels, feature_names, cfg)
    save_bundle(bundle, args.out / "baseline_bundle.pkl")
    report = {"labels": lab_manifest, "match": match_report, "coverage": coverage,
              "cv": cv_report, "rank": rank_report,
              "n_materials": len(labels), "n_train": len(train_labels),
              "n_features": len(feature_names), "feature_names": feature_names,
              "seconds": time.time() - t0}
    (args.out / "baseline_report.json").write_text(json.dumps(report, indent=2,
                                                              default=float))
    print(f"materials={len(labels)} train={len(train_labels)} "
          f"gated={len(match_report['pore_gated_out'])} feats={len(feature_names)} "
          f"match={match_report['by_kind']}")
    for target, m in cv_report.items():
        print(f"  {target}: OOF spearman={m['spearman']:.3f} rmse={m['rmse']:.3e} n={m['n']}")
    for prof, m in rank_report.items():
        print(f"  rank@{prof}: spearman_cop={m['spearman_cop']:.3f} "
              + " ".join(f"{k}={v:.2f}" for k, v in m.items()
                         if k.startswith("top")))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
