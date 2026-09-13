"""N2 label assembly: per-material training table from fits + anchors.

Companion to ROADMAP Stage 1 (tabular baseline). One row per material:

- usable ISODB fits (``fit_flag == ok``, physical ``q_sat``, mass-normalized)
  aggregated by median across isotherms (spread kept as ``q_sat_logstd`` —
  an honest disagreement proxy, not a modeled uncertainty);
- the 13 commercial anchors appended as gold rows (``anchor=True``).

Targets follow the log-space-heads lesson (ROADMAP GeoField reference):
``log_q_sat``, ``log_Q_st`` (multi-T subset only — single-T rows carry NO
isosteric heat and must never be imputed), ``log_E``, ``n_da`` linear.

Families (``family_of``) are v1 keyword rules — coarse but leak-safe for
GroupKFold (zeolite/silica/carbon/MOF/COF/composite/other). Refined
node/topology families arrive with the CoRE/MOFid join (Stage 2).

Writes ``data_cache/n2/labels.csv`` + manifest. Pure pandas/numpy.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FITS_CSV = REPO_ROOT / "data_cache" / "fits" / "da_params.csv"
ANCHORS_CSV = REPO_ROOT / "adsorbent-ml" / "data" / "anchors.csv"
OUT_DIR = REPO_ROOT / "data_cache" / "n2"

# Ordered keyword rules: first hit wins. Zeolite codes before generic words
# (e.g. "MFI" must beat any "silica" mention in "Silicalite MFI/Na").
FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("zeolite", ("13x", "nax", "naa", "4a", "5a", "zsm", "mfi", "chabazite",
                 "zeolite", "fau", "lta", "hisiv")),
    ("composite", ("sws", "licl", "cacl2", "composite", "salt")),
    ("silica", ("silica", "silicalite", "xtrusorb", "sio2", "cab-")),
    ("carbon", ("carbon", "charcoal", "soot", "cnt", "mwcnt", "sorbonorit",
                "bpl", "ax-21", "blucher", "wva", "npc", "pcgf", "pcf",
                "filtrasorb", "calgon")),
    ("mof", ("mof", "mil-", "uio", "zif", "hkust", "cubtc", "cpo-27", "dmof",
             "dut-", "calf", "sim-1", "cau-", "nenu", "taf-", "juc-", "tetzb",
             "cu-tdpat", "sc3", "tb-fum")),
    ("cof", ("cof-",)),
    ("composite", ("sws", "licl", "cacl2", "composite", "salt")),
)
ANCHOR_CLASS_TO_FAMILY = {
    "silica gel": "silica", "zeolite": "zeolite",
    "aluminophosphate": "zeolite", "MOF": "mof", "zeosil": "silica",
    "carbon": "carbon", "composite salt": "composite",
}


def family_of(name: str) -> str:
    """Coarse chemistry family for leak-safe splits (v1 keyword rules)."""
    n = name.lower()
    for family, keywords in FAMILY_RULES:
        if any(k in n for k in keywords):
            return family
    return "other"


def _median_logspread(s: pd.Series) -> tuple[float, float, float]:
    vals = np.asarray(s.dropna(), dtype=float)
    vals = vals[vals > 0]
    med = float(np.median(vals))
    lstd = float(np.std(np.log(vals))) if len(vals) > 1 else 0.0
    return med, lstd, float(len(vals))


def build_labels(*, fits_csv: Path = FITS_CSV,
                 anchors_csv: Path = ANCHORS_CSV) -> tuple[pd.DataFrame, dict]:
    """Assemble the per-material N2 table. Returns ``(labels, manifest)``."""
    fits = pd.read_csv(fits_csv, low_memory=False)
    use = fits[(fits["fit_flag"] == "ok")
               & (fits["q_sat_physical"] == True)  # noqa: E712
               & (fits["q_sat_kg_kg"].notna())].copy()
    rows = []
    for mid, g in use.groupby("material_id"):
        q_med, q_lstd, n_iso = _median_logspread(g["q_sat_kg_kg"])
        e_med, _, _ = _median_logspread(g["e_char_j_mol"])
        n_med = float(np.median(g["n_da"].dropna()))
        qst = g["q_st_j_kg"].dropna()
        rows.append({
            "material_id": mid, "name": str(g["name"].iloc[0]),
            "source": "isodb", "anchor": False,
            "family": family_of(str(g["name"].iloc[0])),
            "q_sat_kg_kg": q_med, "q_sat_logstd": q_lstd,
            "n_isotherms": int(n_iso),
            "q_st_j_kg": float(qst.iloc[0]) if len(qst) else np.nan,
            "e_char_j_mol": e_med, "n_da": n_med,
        })
    anchors = pd.read_csv(anchors_csv)
    for _, r in anchors.iterrows():
        rows.append({
            "material_id": f"anchor:{r['name']}", "name": str(r["name"]),
            "source": "anchor", "anchor": True,
            "family": ANCHOR_CLASS_TO_FAMILY.get(str(r["class"]).strip(), "other"),
            "q_sat_kg_kg": float(r["q_sat_kg_kg"]), "q_sat_logstd": 0.0,
            "n_isotherms": 0,
            "q_st_j_kg": float(r["q_st_MJ_kg"]) * 1e6,
            "e_char_j_mol": float(r["e_char_J_mol"]), "n_da": float(r["n_da"]),
        })
    labels = pd.DataFrame(rows)
    labels["log_q_sat"] = np.log(labels["q_sat_kg_kg"])
    labels["log_E"] = np.log(labels["e_char_j_mol"])
    labels["log_Q_st"] = np.log(labels["q_st_j_kg"].where(labels["q_st_j_kg"] > 0))
    manifest = {"n_materials": len(labels),
                "n_isodb": int((~labels["anchor"]).sum()),
                "n_anchors": int(labels["anchor"].sum()),
                "n_with_Q_st": int(labels["q_st_j_kg"].notna().sum()),
                "families": labels["family"].value_counts().to_dict(),
                "provenance": f"fits:{fits_csv} + anchors:{anchors_csv} (median-aggregated)",
                "created_unix": time.time()}
    return labels, manifest


def write_labels(labels: pd.DataFrame, manifest: dict,
                 out_dir: Path = OUT_DIR) -> Path:
    """Write ``labels.csv`` + ``manifest.json``. Returns the table path."""
    import json  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_dir / "labels.csv", index=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir / "labels.csv"


__all__ = ["ANCHOR_CLASS_TO_FAMILY", "FAMILY_RULES", "build_labels",
           "family_of", "write_labels"]
