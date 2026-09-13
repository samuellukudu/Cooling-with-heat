"""QMOF join table: per-refcode formulas, topologies, DOIs, pore geometry.

``data_cache/qmof/properties/qmof.csv`` (20,372 rows) carries CSD-style
``name`` values (``ABACUF01_FSR``), real compositions (``info.formula``),
MOFid topologies, publication DOIs and DFT-geometry pore data. This module
reduces it to one row per CSD refcode:

- ``formula``: first non-null composition (variants share chemistry);
- ``topology``: modal MOFid topology (the Stage-2 family-split axis);
- ``doi``: lowercased DOI set (Ongari-protocol bridge key);
- ``pld/lcd/density``: median DFT-geometry values (pore fallback where no
  CoRE exemplar exists — flagged ``pore_qmof`` downstream);
- ``n_variants``: row count per refcode (disorder/polymorph spread signal).

Formulas that fail ``parse_formula`` (brackets, adducts) yield None —
curate, don't crash.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
QMOF_CSV = REPO_ROOT / "data_cache" / "qmof" / "properties" / "qmof.csv"

REFCODE_RE = re.compile(r"^([A-Z]{6}\d{0,2})")
USECOLS = ["name", "info.formula", "info.mofid.topology", "info.doi",
           "info.pld", "info.lcd", "info.density"]


def refcode_of(name: str) -> str | None:
    """Leading CSD refcode from a QMOF/CoRE-style name (None if absent)."""
    m = REFCODE_RE.match(str(name).upper())
    return m.group(1) if m else None


def load_qmof_table(*, qmof_csv: Path = QMOF_CSV) -> pd.DataFrame:
    """One row per refcode. Adds ``formula_dict`` (parsed or None)."""
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
    from composition import parse_formula  # noqa: PLC0415

    q = pd.read_csv(qmof_csv, usecols=USECOLS, low_memory=False)
    q["refcode"] = q["name"].apply(refcode_of)
    q = q[q["refcode"].notna()].copy()
    q["doi_norm"] = q["info.doi"].str.lower()

    def _parse(s) -> dict | None:
        try:
            return parse_formula(str(s)) if pd.notna(s) else None
        except ValueError:
            return None

    q["formula_dict"] = q["info.formula"].apply(_parse)
    rows = []
    for ref, g in q.groupby("refcode"):
        topo = g["info.mofid.topology"].dropna()
        forms = [f for f in g["formula_dict"] if f]
        rows.append({
            "refcode": ref,
            "formula_dict": forms[0] if forms else None,
            "formula_str": str(g.loc[g["formula_dict"].notna(),
                                     "info.formula"].iloc[0]) if forms else "",
            "topology": str(topo.mode().iloc[0]) if len(topo) else "",
            "dois": sorted(set(g["doi_norm"].dropna())),
            "pld": float(g["info.pld"].median()),
            "lcd": float(g["info.lcd"].median()),
            "density": float(g["info.density"].median()),
            "n_variants": int(len(g)),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


__all__ = ["load_qmof_table", "refcode_of"]
