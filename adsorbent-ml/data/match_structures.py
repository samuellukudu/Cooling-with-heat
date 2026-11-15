"""Name→structure matching: ISODB material names to CoRE / IZA / OPTIMADE.

The known hard problem (ACQUISITION §1): ISODB names are non-standard
("Cu-BTC", "MOF-199", "MIL-53(Al)"). Strategy, strongest evidence first:

1. **CSD refcodes** (Ongari name→refcode map) → **DOI bridge**
   (ISODB-publication ↔ CoRE/QMOF DOIs, one-to-one) → **CoRE verified
   substring** (self-verifying: normalized synonyms kept ONLY if they hit
   ≥ 1 CoRE filename; < 4 chars rejected — the UKEBOD lesson) →
   **csd-fuzzy** (difflib ≥ 0.85, always ambiguous) → **IZA trade-name
   rules** (13X→FAU, …).
2. **OPTIMADE formula rung** (the matching round, ROADMAP item 3): the
   still-unmatched names whose names/families resolve to a composition are
   matched against the cached OPTIMADE corpora (``data_cache/optimade/*/``
   — pyrene-MOFs, curated COFs) by H-free composition cosine with exact
   metal sets (``optimade_export.formula_similarity``). Composition identity
   is *weaker* evidence than a refcode (same stoichiometry ≠ same
   framework), so hits record their score, and are flagged ambiguous unless
   the best composition is unique AND near-exact. Pore descriptors for
   these rows stay NaN (no CoRE exemplar) — the CIFs are the future
   descriptor source.

Writes ``matched.parquet`` + ``unmatched_names.csv`` + ``match_report.json``
(ACQUISITION §1 names). Run all rungs: ``python adsorbent-ml/data/match_structures.py``.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_PROPS = REPO_ROOT / "data_cache" / "core_mof" / "properties.parquet"
IZA_DIR = REPO_ROOT / "data_cache" / "iza" / "frameworks"
OPTIMADE_ROOT = REPO_ROOT / "data_cache" / "optimade"
QMOF_CSV = REPO_ROOT / "data_cache" / "qmof" / "properties" / "qmof.csv"
OUT_DIR = REPO_ROOT / "data_cache" / "n2"
OPTIMADE_SCORE_MIN = 0.90  # composition-cosine acceptance (metals must match exactly)

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


def load_optimade_index(root: Path = OPTIMADE_ROOT) -> list[dict]:
    """Element-count index over the cached OPTIMADE corpora.

    One entry per structure line: ``{db, id, formula, counts, cif}`` — the
    candidate pool for the composition rung. Pure local read of the cached
    ``structures.jsonl`` files; no HTTP.
    """
    import json  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
    from composition import parse_formula  # noqa: PLC0415

    out: list[dict] = []
    for jf in sorted(root.glob("*/structures.jsonl")):
        db = jf.parent.name
        cifs = jf.parent / "cifs"
        with open(jf, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                attrs = e.get("attributes") or {}
                formula = attrs.get("chemical_formula_descriptive") or ""
                if not formula:
                    continue
                try:
                    counts = parse_formula(formula)
                except Exception:
                    continue  # solvent adducts etc. fail loud upstream; skip here
                if not counts:
                    continue
                eid = str(e.get("id", ""))
                out.append({"db": db, "id": eid, "formula": formula,
                            "counts": counts, "cif": cifs / f"{eid}.cif"})
    return out


def load_qmof_index(path: "Path | None" = None) -> list[dict]:
    """Element-count index over the cached QMOF table (20k DFT rows).

    Same shape as ``load_optimade_index`` (``{db, id, ref, formula, counts,
    cif=None}``) built from ``info.formula`` — the candidate pool the
    ``qmoftab-formula`` rung scores against. Metal-bearing rows only (the
    rung abstains on metal-free targets, so metal-free candidates can never
    win a match; dropping them is speed, not evidence). No CIF path: pore
    descriptors stay NaN for this rung, same as optimade-formula hits
    without a downloaded CIF.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
    from composition import parse_formula  # noqa: PLC0415
    from optimade_export import METALS  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    p = path or QMOF_CSV
    df = pd.read_csv(p, low_memory=False, usecols=["qmof_id", "name",
                                                   "info.formula"])
    out: list[dict] = []
    for _, r in df.iterrows():
        formula = r["info.formula"]
        if not isinstance(formula, str) or not formula:
            continue
        try:
            counts = parse_formula(formula)
        except Exception:
            continue
        if not counts or not any(e in METALS for e in counts):
            continue
        out.append({"db": "qmoftab", "id": str(r["qmof_id"]),
                    "ref": str(r["name"]), "formula": formula,
                    "counts": counts, "cif": None})
    return out


def match_formula_index(name: str, family: str, index: list[dict], *,
                        source: str) -> dict | None:
    """Composition-exact candidate from a formula index, or None.

    ``formula_for(name, family)`` resolves the target composition (parsed
    formula-names like ``C9H11ClN2O4Cd`` or the class table for standard
    materials); ``formula_similarity`` scores H-free cosine with exact
    metal sets. Score and provenance always travel with the hit, under
    ``{source}_*`` keys.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from composition import formula_for, parse_formula  # noqa: PLC0415
    from optimade_export import METALS, formula_similarity  # noqa: PLC0415

    formula, _src = formula_for(name, family)
    if formula is None:
        return None
    try:
        target = parse_formula(formula)
    except Exception:
        return None
    if not target:
        return None
    # Metal-free compositions (carbons, SiO2) have no discriminative power
    # here: {C:1} scores ~0.98 against ANY carbon-rich COF. The metals
    # define the framework — without one, the rung abstains honestly. The
    # carbons/silicas among the unmatched need different corpora entirely.
    if not any(e in METALS for e in target):
        return None
    scored = []
    for cand in index:
        s = formula_similarity(target, cand["counts"])
        if s >= OPTIMADE_SCORE_MIN:
            scored.append((s, cand))
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    best_s, best = scored[0]
    n_ties = sum(1 for s, _ in scored if abs(s - best_s) < 1e-6)
    return {f"{source}_db": best.get("db", source),
            f"{source}_id": best["id"],
            f"{source}_formula": best["formula"],
            f"{source}_score": round(float(best_s), 4),
            f"{source}_ref": best.get("ref", ""),
            f"{source}_cif": str(best.get("cif") or ""),
            "ambiguous": bool(n_ties > 1 or best_s < 0.999)}


def match_optimade_formula(name: str, family: str, index: list[dict]) -> dict | None:
    """Composition-exact candidate from the OPTIMADE corpora, or None."""
    return match_formula_index(name, family, index, source="optimade")


def match_qmoftab_formula(name: str, family: str, index: list[dict]) -> dict | None:
    """Composition-exact candidate from the QMOF table (20k DFT rows), or None."""
    return match_formula_index(name, family, index, source="qmoftab")


def match_materials(names: pd.DataFrame, *,
                    core_props: Path = CORE_PROPS,
                    anchor_names: list[str] | None = None,
                    ongari_map: pd.DataFrame | None = None,
                    qmof_table: pd.DataFrame | None = None,
                    material_dois: dict[str, set[str]] | None = None,
                    optimade_index: "list[dict] | None" = None,
                    qmof_index: "list[dict] | None" = None,
                    ) -> tuple[pd.DataFrame, dict]:
    """Match a table with ``name`` (+ optional ``material_id``/``family``) columns.

    Precedence (first hit wins): ``csd`` (Ongari name→refcode) → ``doi``
    (shared-publication bridge, Ongari one-to-one protocol) → ``core``
    (legacy verified substring) → ``csd-fuzzy`` → ``iza`` (trade-name
    rules) → ``optimade-formula`` (composition cosine over the cached
    OPTIMADE corpora) → ``qmoftab-formula`` (same scoring over the cached
    QMOF table; weakest evidence, scored + flagged). Optional inputs
    enable the rungs; without them the v1 behavior is kept.

    New columns: ``refcode`` (+``refcode_source``), ``doi_evidence``,
    ``qmof_formula``/``qmof_topology``/``qmof_density``/``qmof_pld``/
    ``qmof_lcd`` (refcode-joined), ``core_exemplar`` (refcode-joined first),
    ``optimade_db``/``optimade_id``/``optimade_formula``/``optimade_score``/
    ``optimade_cif`` (composition rung),
    ``qmoftab_id``/``qmoftab_ref``/``qmoftab_formula``/``qmoftab_score``/
    ``qmoftab_cif`` (QMOF-table composition rung).
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
    fam_of = (dict(zip(names["name"].astype(str), names["family"].astype(str)))
              if "family" in names.columns else {})

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
                     "optimade_db": "", "optimade_id": "",
                     "optimade_formula": "", "optimade_score": float("nan"),
                     "optimade_cif": "",
                     "qmoftab_db": "", "qmoftab_id": "", "qmoftab_ref": "",
                     "qmoftab_formula": "", "qmoftab_score": float("nan"),
                     "qmoftab_cif": "",
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
        if not done and (optimade_index or qmof_index):
            # the two composition rungs carry equal evidential weight, so the
            # higher score wins (round-2 fix: precedence alone handed DUT-4 a
            # 0.985 OPTIMADE candidate while the QMOF table holds a 1.0 exact).
            fam = fam_of.get(name, "other")
            o = (match_optimade_formula(name, fam, optimade_index)
                 if optimade_index else None)
            q = (match_qmoftab_formula(name, fam, qmof_index)
                 if qmof_index else None)
            if o and q:
                hit, kind = ((o, "optimade-formula")
                             if o["optimade_score"] >= q["qmoftab_score"]
                             else (q, "qmoftab-formula"))
            else:
                hit, kind = ((o, "optimade-formula") if o
                             else (q, "qmoftab-formula") if q
                             else (None, None))
            if hit:
                rec.update(match_kind=kind, **hit)
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
              "n_with_optimade": int(
                  (matched["match_kind"] == "optimade-formula").sum()),
              "n_with_qmoftab": int(
                  (matched["match_kind"] == "qmoftab-formula").sum()),
              "provenance": ("ongari/csd-refcode + doi-bridge + substring + "
                             "fuzzy + IZA + optimade-formula + qmoftab-formula "
                             f"vs {core_props}"),
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
           "OPTIMADE_SCORE_MIN", "load_optimade_index", "load_qmof_index",
           "match_core", "match_materials", "match_optimade_formula",
           "match_qmoftab_formula", "match_iza", "normalize",
           "pore_consistency", "refcode_of_filename", "write_match"]


def main() -> None:
    """CLI runner — makes validate.py's "run match_structures.py" hint real.

    Replicates the train_baseline call (all rungs incl. the OPTIMADE formula
    round) and writes the audit artifacts without touching training.
    """
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description="Match material names to structures (all rungs).")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--no-optimade", action="store_true",
                    help="skip the OPTIMADE composition rung")
    ap.add_argument("--no-qmoftab", action="store_true",
                    help="skip the QMOF-table composition rung")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
    from labels import build_labels  # noqa: PLC0415
    from ongari import load_ongari_map  # noqa: PLC0415
    from qmof import load_qmof_table  # noqa: PLC0415

    labels, _manifest = build_labels()
    try:
        om = load_ongari_map()
    except FileNotFoundError:
        print("[warn] data_cache/ongari tables absent — csd rung skipped "
              "(run ongari.py --fetch)")
        om = None
    qt = load_qmof_table()
    w = pd.read_parquet(REPO_ROOT / "data_cache" / "isodb" / "water_isotherms.parquet",
                        columns=["adsorbent_hashkey", "doi"])
    dois = {hk: set(g["doi"].dropna().astype(str))
            for hk, g in w.groupby("adsorbent_hashkey")}
    idx = None if args.no_optimade else load_optimade_index()
    n_opt = f"{len(idx)} OPTIMADE candidates" if idx else "no optimade index"
    qidx = None
    if not args.no_qmoftab:
        try:
            qidx = load_qmof_index()
        except FileNotFoundError:
            print("[warn] data_cache/qmof table absent — qmoftab rung skipped")
    n_q = f"+{len(qidx)} QMOF candidates" if qidx else "no qmof index"
    matched, rep = match_materials(
        labels[["name", "material_id", "family"]],
        anchor_names=labels[labels["anchor"]]["name"].tolist(),
        ongari_map=om, qmof_table=qt, material_dois=dois,
        optimade_index=idx, qmof_index=qidx)
    matched = pore_consistency(matched, labels)
    write_match(matched, rep, args.out)
    n_hit = len(matched) - int((matched["match_kind"] == "none").sum())
    print(f"matched {n_hit}/{len(matched)} ({n_hit / len(matched):.0%})  "
          f"by_kind={rep['by_kind']}  [{n_opt} {n_q}]")
    print(f"artifacts → {args.out}/ (matched.parquet, unmatched_names.csv, "
          "match_report.json)")


if __name__ == "__main__":
    main()
