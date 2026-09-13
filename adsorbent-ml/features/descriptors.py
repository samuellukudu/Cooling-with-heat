"""Feature matrix builder: pores + family + composition → one numeric table.

Explicit engineering features (GeoField lesson), three blocks:

- **Pore block** (CoRE properties on the exemplar match): LCD/PLD [Å],
  gravimetric ASA/AV, open-metal-site flag — the geometry the D–A energy
  actually depends on. IZA-matched zeolites take LCD/PLD from the IZA-SC
  framework table (experimental ground truth, ``pore_iza=1``); other
  IZA/non-CoRE matches fall back to QMOF DFT geometry (``pore_qmof=1``);
  anything else stays NaN + ``pore_has_data=0``.
- **Family block**: one-hot over the v1 families (fixed column order,
  persisted with the model bundle for inference).
- **Composition block** (Magpie-lite from ``composition.py``): unparseable
  formulas degrade to NaN, never crash the build (``formula_prov=failed``).

Missingness is structural (unmatched zeolite, formula-free MOF) — the
baseline model (HistGradientBoosting) handles NaN natively, and the
coverage report quantifies every gap instead of hiding it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from composition import (COMPOSITION_COLUMNS, featurize_composition,
                         formula_for, parse_formula)

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PROPS = REPO_ROOT / "data_cache" / "core_mof" / "properties.parquet"
IZA_FRAMEWORK_DATA = REPO_ROOT / "data_cache" / "iza" / "framework_data.csv"

PORE_COLUMNS = ["LCD", "PLD", "ASA_m2_g", "AV_cm3_g"]
FAMILIES = ("zeolite", "silica", "carbon", "mof", "cof", "composite", "other")


def build_feature_matrix(labels: pd.DataFrame, matched: pd.DataFrame, *,
                         core_props: Path = CORE_PROPS,
                         iza_pores: Path | None = IZA_FRAMEWORK_DATA,
                         topologies: list[str] | None = None,
                         top_n_topologies: int = 8
                         ) -> tuple[pd.DataFrame, list[str], dict]:
    """Returns ``(X, feature_names, coverage)`` aligned to ``labels`` rows.

    Formula precedence: parsed-from-name > QMOF (refcode-joined, per-material)
    > class table. Pores: CoRE exemplar first, IZA framework table for
    zeolite matches (``pore_iza=1``), QMOF DFT geometry as flagged fallback
    (``pore_qmof=1``). Topologies: top-N one-hots + ``topo_other``; pass the
    training ``topologies`` list at inference for stable columns.
    """
    core = pd.read_parquet(core_props,
                           columns=["filename"] + PORE_COLUMNS + ["Has_OMS"])
    core_by_file = core.set_index("filename")
    iza_by_code: dict[str, dict] = {}
    if iza_pores is not None and Path(iza_pores).exists():
        iza = pd.read_csv(iza_pores).set_index("code")
        iza_by_code = {str(k): v for k, v in iza.to_dict("index").items()}
    m = matched.set_index("name")
    if topologies is None:
        top = (m["qmof_topology"].replace("", np.nan).dropna().value_counts()
               .head(top_n_topologies).index.tolist() if "qmof_topology" in m.columns
               else [])
        topologies = top
    topo_cols = [f"topo_{t}" for t in topologies] + ["topo_other", "has_topology"]

    fam_oh = [f"fam_{f}" for f in FAMILIES]
    feature_names = (["pore_LCD", "pore_PLD", "pore_ASA", "pore_AV", "pore_OMS",
                      "pore_has_data", "pore_ambiguous", "pore_qmof", "pore_iza",
                      "qmof_density"]
                     + fam_oh + topo_cols + list(COMPOSITION_COLUMNS)
                     + ["has_formula", "is_parsed", "is_qmof_formula"])
    rows: list[dict] = []
    n_pore = n_qmof_pore = n_iza_pore = n_form = n_qmof_form = 0
    for _, lab in labels.iterrows():
        name = str(lab["name"])
        rec = {"name": name}
        mm = m.loc[name] if name in m.index else None
        mm = mm.iloc[0] if isinstance(mm, pd.DataFrame) else mm
        pore = {c: np.nan for c in
                ["pore_LCD", "pore_PLD", "pore_ASA", "pore_AV", "pore_OMS"]}
        pore.update(pore_has_data=0, pore_ambiguous=0, pore_qmof=0,
                    pore_iza=0, qmof_density=np.nan)
        if mm is not None and mm.get("core_exemplar", ""):
            if mm["core_exemplar"] in core_by_file.index:
                c = core_by_file.loc[mm["core_exemplar"]]
                c = c.iloc[0] if isinstance(c, pd.DataFrame) else c
                pore.update(pore_LCD=float(c["LCD"]), pore_PLD=float(c["PLD"]),
                            pore_ASA=float(c["ASA_m2_g"]), pore_AV=float(c["AV_cm3_g"]),
                            pore_OMS=int(bool(c["Has_OMS"])),
                            pore_has_data=1,
                            pore_ambiguous=int(bool(mm.get("ambiguous", False))))
                n_pore += 1
        if mm is not None and not pore["pore_has_data"] and str(
                mm.get("iza_code", "") or "") in iza_by_code:
            # IZA-SC framework table: LCD = largest included sphere; PLD =
            # largest free sphere = max of the three directional diffuse PLDs.
            row = iza_by_code[str(mm["iza_code"])]
            pore.update(pore_LCD=float(row["lcd_included_a"]),
                        pore_PLD=float(max(row["pld_diffuse_a"],
                                           row["pld_diffuse_b"],
                                           row["pld_diffuse_c"])),
                        pore_has_data=1, pore_iza=1,
                        pore_ambiguous=int(bool(mm.get("ambiguous", False))))
            n_iza_pore += 1
        if mm is not None and not pore["pore_has_data"]:
            try:
                pld, lcd = float(mm.get("qmof_pld", float("nan"))), float(
                    mm.get("qmof_lcd", float("nan")))
            except (TypeError, ValueError):
                pld, lcd = float("nan"), float("nan")
            if np.isfinite(pld) and np.isfinite(lcd):
                pore.update(pore_LCD=lcd, pore_PLD=pld, pore_has_data=1,
                            pore_qmof=1,
                            pore_ambiguous=int(bool(mm.get("ambiguous", False))))
                n_qmof_pore += 1
        if mm is not None:
            try:
                pore["qmof_density"] = float(mm.get("qmof_density", float("nan")))
            except (TypeError, ValueError):
                pass
        rec.update(pore)
        fam = str(lab["family"])
        for f in FAMILIES:
            rec[f"fam_{f}"] = int(fam == f)
        topo = str(mm.get("qmof_topology", "") or "") if mm is not None else ""
        for t in topologies:
            rec[f"topo_{t}"] = int(topo == t)
        rec["topo_other"] = int(bool(topo) and topo not in topologies)
        rec["has_topology"] = int(bool(topo))
        formula, prov = formula_for(name, fam)
        if prov != "parsed" and mm is not None and str(
                mm.get("qmof_formula", "") or ""):
            formula, prov = str(mm["qmof_formula"]), "qmof"
        comp = {c: np.nan for c in COMPOSITION_COLUMNS}
        has_f, is_p, is_q = 0, 0, 0
        if formula is not None:
            try:
                comp.update(featurize_composition(parse_formula(formula)))
                has_f, is_p, is_q = 1, int(prov == "parsed"), int(prov == "qmof")
                n_form += 1
                n_qmof_form += is_q
            except ValueError:
                prov = "failed"
        rec.update(comp, has_formula=has_f, is_parsed=is_p,
                   is_qmof_formula=is_q, formula_prov=prov)
        rows.append(rec)
    X = pd.DataFrame(rows)
    coverage = {"n_rows": len(X), "n_with_pores": n_pore,
                "n_with_iza_pores": n_iza_pore,
                "n_with_qmof_pores": n_qmof_pore, "n_with_formula": n_form,
                "n_with_qmof_formula": n_qmof_form,
                "topologies": topologies,
                "formula_prov": X["formula_prov"].value_counts().to_dict()}
    return X[["name"] + feature_names].copy(), feature_names, coverage


__all__ = ["FAMILIES", "PORE_COLUMNS", "build_feature_matrix"]
