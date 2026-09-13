"""Name→structure matching: ISODB material names to CoRE / IZA / anchors.

The known hard problem (ACQUISITION §1): ISODB names are non-standard
("Cu-BTC", "MOF-199", "MIL-53(Al)"). v1 strategy, cheapest first:

1. **IZA rules** (ordered regex → 3-letter code → ``data_cache/iza`` CIF):
   zeolite trade names have deterministic framework codes.
2. **CoRE substring hits** (self-verifying): normalized-name synonyms are
   kept ONLY if they hit ≥ 1 CoRE filename; hit counts and ambiguity are
   recorded, never hidden. Substrings < 4 chars are rejected (the UKEBOD
   lesson: "BOD" matches CSD refcodes that are not CuBTC).
3. **Anchor join** on normalized names (the 13 gold rows).

Writes ``matched.parquet`` + ``unmatched_names.csv`` (the filenames
ACQUISITION §1 promises). The Coudert-lab SI mapping + MOFid/MOFkey joins
are the Stage-2 upgrade, explicitly out of v1.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PROPS = REPO_ROOT / "data_cache" / "core_mof" / "properties.parquet"
IZA_DIR = REPO_ROOT / "data_cache" / "iza" / "frameworks"
OUT_DIR = REPO_ROOT / "data_cache" / "n2"

# Ordered (regex, IZA code, confidence). Trade names → framework codes.
IZA_RULES: tuple[tuple[str, str, str], ...] = (
    (r"13x|nax\b|zeolite\s*y\b|\by\b.*zeolite", "FAU", "high"),
    (r"\b4a\b|naa\b|zeolite\s*a\b|\b5a\b", "LTA", "high"),
    (r"zsm-?5|mfi|silicalite|hisiv", "MFI", "high"),
    (r"chabazite|\bcha\b|sap-?o?-?34|alpo-?18", "CHA", "medium"),
)

# Normalized-name → CoRE filename substring. Candidates are VERIFIED against
# the table at match time (≥1 hit required); misses stay unmatched, loudly.
CORE_SYNONYMS: tuple[tuple[str, str], ...] = (
    ("zif-8", "ZIF8"), ("ed-zif-8", "ZIF8"),
    ("uio-66", "UIO66"),
    ("mil-100", "MIL100"), ("mil-101", "MIL101"), ("mil-53", "MIL53"),
    ("mil-160", "MIL160"),
    ("cau-10", "CAU10"), ("cau-1", "CAU1"),
    ("mof-177", "MOF177"), ("dut-4", "DUT4"), ("calf-25", "CALF25"),
    ("sim-1", "SIM1"), ("cof-1", "COF1"),
    ("hkust", "HKUST"), ("mof-199", "HKUST"),
)
MIN_SUBSTR_LEN = 4
AMBIGUOUS_ABOVE = 5  # hits above this are flagged ambiguous (still recorded)
FUZZY_CUTOFF = 0.85  # difflib ratio for csd-fuzzy (always ambiguous + gated)

REFCODE_RE = re.compile(r"^([A-Z]{6}\d{0,2})")


def refcode_of_filename(filename: str) -> str | None:
    """Leading CSD refcode of a CoRE-style filename (None for DOI-derived)."""
    tok = re.split(r"[_\-.]", str(filename))[0].upper()
    m = REFCODE_RE.fullmatch(tok)
    return m.group(1) if m else None


def normalize(name: str) -> str:
    """Lowercase alphanumeric skeleton: 'MIL-100(Cr)-EG' -> 'mil100creg'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_iza(name: str) -> tuple[str, str] | None:
    """(code, confidence) or None. First rule wins."""
    n = name.lower()
    for pattern, code, conf in IZA_RULES:
        if re.search(pattern, n):
            return code, conf
    return None


def match_core(name: str, filenames: pd.Series) -> tuple[str, int, str] | None:
    """(substring, n_hits, exemplar_filename) for the first verified synonym.

    The exemplar is the first hitting filename (pore-descriptor join key);
    multi-hit matches are flagged ambiguous downstream — the exemplar is a
    stated approximation, not a resolved identity.
    """
    norm = normalize(name)
    for key, substr in CORE_SYNONYMS:
        if len(substr) < MIN_SUBSTR_LEN:
            continue
        if key in norm or norm in key or substr.lower() in norm:
            mask = filenames.str.upper().apply(
                lambda s: substr in re.sub(r"[^A-Z0-9]", "", s))
            hits = filenames[mask.fillna(False)]
            if len(hits) >= 1:
                return substr, len(hits), str(hits.iloc[0])
    return None


def match_materials(names: pd.DataFrame, *,
                    core_props: Path = CORE_PROPS,
                    anchor_names: list[str] | None = None,
                    ongari_map: pd.DataFrame | None = None,
                    qmof_table: pd.DataFrame | None = None,
                    material_dois: dict[str, set[str]] | None = None,
                    ) -> tuple[pd.DataFrame, dict]:
    """Match a table with ``name`` (+ optional ``material_id``) columns.

    Precedence (first hit wins): ``csd`` (Ongari name→refcode) → ``doi``
    (shared-publication bridge, Ongari one-to-one protocol) → ``core``
    (legacy verified substring) → ``iza`` (trade-name rules). Optional
    inputs enable the upper rungs; without them the v1 behavior is kept.

    New columns: ``refcode`` (+``refcode_source``), ``doi_evidence``,
    ``qmof_formula``/``qmof_topology``/``qmof_density``/``qmof_pld``/
    ``qmof_lcd`` (refcode-joined), ``core_exemplar`` (refcode-joined first).
    """
    core = pd.read_parquet(core_props)
    keep = ["filename"] + (["DOI_public"] if "DOI_public" in core.columns else [])
    core = core[keep]
    core["refcode"] = core["filename"].apply(refcode_of_filename)
    core_by_ref: dict[str, str] = {}
    for ref, g in core.dropna(subset=["refcode"]).groupby("refcode"):
        core_by_ref[ref] = str(sorted(g["filename"])[0])
    core_doi: dict[str, set[str]] = {}
    if "DOI_public" in core.columns:
        for _, r in core.dropna(subset=["DOI_public", "refcode"]).iterrows():
            core_doi.setdefault(str(r["DOI_public"]).lower(), set()).add(r["refcode"])

    qmof_by_ref: dict[str, dict] = {}
    qmof_doi: dict[str, set[str]] = {}
    if qmof_table is not None:
        for _, r in qmof_table.iterrows():
            qmof_by_ref[str(r["refcode"])] = r.to_dict()
            for d in (r["dois"] if isinstance(r["dois"], list) else []):
                qmof_doi.setdefault(str(d).lower(), set()).add(str(r["refcode"]))

    ong: dict[str, list[tuple[str, str, str]]] = {}
    if ongari_map is not None:
        for _, r in ongari_map.iterrows():
            ong.setdefault(normalize(str(r["adsorbent_name"])),
                           []).append((str(r["refcode"]), str(r["source"]),
                                       str(r["adsorbent_name"])))
    anchors = {normalize(a) for a in (anchor_names or [])}
    has_ids = "material_id" in names.columns
    id_of = dict(zip(names["name"].astype(str),
                     names["material_id"].astype(str))) if has_ids else {}

    def _qmof_cols(ref: str) -> dict:
        q = qmof_by_ref.get(ref, {})
        return {"qmof_formula": str(q.get("formula_str", "") or ""),
                "qmof_topology": str(q.get("topology", "") or ""),
                "qmof_density": float(q.get("density", float("nan"))),
                "qmof_pld": float(q.get("pld", float("nan"))),
                "qmof_lcd": float(q.get("lcd", float("nan")))}

    rows = []
    for name in names["name"].astype(str):
        rec: dict = {"name": name, "match_kind": "none", "iza_code": "",
                     "iza_conf": "", "core_substr": "", "core_hits": 0,
                     "core_exemplar": "", "refcode": "", "refcode_source": "",
                     "csd_name": "", "fuzzy_score": 0.0, "doi_evidence": "",
                     "ambiguous": False,
                     "anchor_hit": normalize(name) in anchors,
                     **{k: ("" if "formula" in k or "topology" in k else float("nan"))
                        for k in ("qmof_formula", "qmof_topology")},
                     **{"qmof_density": float("nan"), "qmof_pld": float("nan"),
                        "qmof_lcd": float("nan")}}
        done = False
        norm = normalize(name)
        refs = sorted({r for r, _, _ in ong.get(norm, [])})
        if refs:
            srcs = sorted({s for r, s, _ in ong.get(norm, []) if r in refs})
            shown = sorted({o for r, _, o in ong.get(norm, []) if r in refs})
            rec.update(match_kind="csd", refcode=";".join(refs),
                       refcode_source="+".join(srcs), csd_name=";".join(shown),
                       ambiguous=len(refs) > 1)
            done = True
        if not done and material_dois:
            dois = {d.lower() for d in material_dois.get(id_of.get(name, ""), set())}
            shared = [d for d in dois if d in core_doi or d in qmof_doi]
            rset = sorted({r for d in shared
                           for r in core_doi.get(d, set()) | qmof_doi.get(d, set())})
            if rset:
                rec.update(match_kind="doi", refcode=";".join(rset),
                           refcode_source="doi-bridge",
                           doi_evidence=";".join(sorted(shared)),
                           ambiguous=not (len(shared) == 1 and len(rset) == 1))
                done = True
        if not done:
            hit = match_core(name, core["filename"])
            if hit:
                rec.update(match_kind="core", core_substr=hit[0],
                           core_hits=hit[1], core_exemplar=hit[2],
                           ambiguous=hit[1] > AMBIGUOUS_ABOVE)
                done = True
            elif ong:
                import difflib  # noqa: PLC0415

                cand = difflib.get_close_matches(norm, list(ong), n=1,
                                                 cutoff=FUZZY_CUTOFF)
                if cand:
                    frefs = sorted({r for r, _, _ in ong[cand[0]]})
                    score = difflib.SequenceMatcher(None, norm, cand[0]).ratio()
                    shown = sorted({o for r, _, o in ong[cand[0]] if r in frefs})
                    rec.update(match_kind="csd-fuzzy", refcode=";".join(frefs),
                               refcode_source="ongari-fuzzy", csd_name=";".join(shown),
                               fuzzy_score=round(float(score), 4),
                               ambiguous=True)
                    done = True
        if not done:
            iza = match_iza(name)
            if iza:
                rec.update(match_kind="iza", iza_code=iza[0], iza_conf=iza[1])
                done = True
        if rec["match_kind"] == "core" and not rec["refcode"]:
            derived = refcode_of_filename(rec["core_exemplar"])
            if derived:
                rec.update(refcode=derived, refcode_source="core-refcode")
        first_ref = (rec["refcode"].split(";")[0] if rec["refcode"] else "")
        if first_ref:
            if first_ref in core_by_ref:
                rec["core_exemplar"] = core_by_ref[first_ref]
            rec.update(_qmof_cols(first_ref))
        rows.append(rec)
    matched = pd.DataFrame(rows)
    kinds = matched["match_kind"].value_counts().to_dict()
    report = {"n_names": len(matched), "by_kind": kinds,
              "n_anchor_hits": int(matched["anchor_hit"].sum()),
              "n_ambiguous": int(matched["ambiguous"].sum()),
              "n_with_refcode": int((matched["refcode"] != "").sum()),
              "n_with_qmof": int((matched["qmof_formula"] != "").sum()),
              "provenance": f"ongari/csd-refcode + doi-bridge + substring + IZA vs {core_props}",
              "created_unix": time.time()}
    return matched, report


def write_match(matched: pd.DataFrame, report: dict,
                out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    """``matched.parquet`` + ``unmatched_names.csv`` (ACQUISITION §1 names)."""
    import json  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    mp = out_dir / "matched.parquet"
    up = out_dir / "unmatched_names.csv"
    matched.to_parquet(mp, index=False)
    matched[matched["match_kind"] == "none"][["name"]].to_csv(up, index=False)
    (out_dir / "match_report.json").write_text(json.dumps(report, indent=2))
    return mp, up


def pore_consistency(matched: pd.DataFrame, labels: pd.DataFrame, *,
                     core_props: Path = CORE_PROPS,
                     margin: float = 1.5) -> pd.DataFrame:
    """Flag ``pore_consistent`` per row (Ongari-style verifier, water edition).

    Fitted ``q_sat`` bounds the micropore volume directly (kg/kg water ≡
    cm³/g at ρ=1 g/cm³); a match whose geometric ``AV`` is far SMALLER
    cannot be the measured sample (wrong structure, collapsed pores, or
    binder-diluted pellet). Consistent iff ``q_sat ≤ AV × margin``; rows
    without both sides get None (no evidence either way — never failed
    without evidence).
    """
    core = pd.read_parquet(core_props, columns=["filename", "AV_cm3_g"])
    av = dict(zip(core["filename"], core["AV_cm3_g"]))
    qs = dict(zip(labels["name"].astype(str), labels["q_sat_kg_kg"]))
    out = matched.copy()
    flags, details = [], []
    for _, r in out.iterrows():
        q, v = qs.get(str(r["name"])), av.get(str(r.get("core_exemplar", "")))
        if q is None or v is None or not np.isfinite(v) or v <= 0:
            flags.append(None)
            details.append("no-evidence")
        elif float(q) <= float(v) * margin:
            flags.append(True)
            details.append(f"qvol={float(q):.3f}<=AV={float(v):.3f}x{margin}")
        else:
            flags.append(False)
            details.append(f"qvol={float(q):.3f}>AV={float(v):.3f}x{margin}")
    out["pore_consistent"] = flags
    out["pore_check"] = details
    return out


__all__ = ["AMBIGUOUS_ABOVE", "CORE_SYNONYMS", "FUZZY_CUTOFF", "IZA_RULES",
           "match_core", "match_iza", "match_materials", "normalize",
           "pore_consistency", "refcode_of_filename", "write_match"]
