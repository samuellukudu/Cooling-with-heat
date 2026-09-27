"""Dataset-card generation for the data-curation pipeline.

One markdown card per dataset under ``docs/datasets/`` — generated from live
``data_cache/`` manifests + small file probes, never hand-copied numbers:

    python adsorbent-ml/data/cards.py                  # writes docs/datasets/
    python adsorbent-ml/data/cards.py --out /tmp/cards # preview elsewhere

Cards: ``isodb`` (experimental isotherms), ``fits`` (D-A parameters),
``labels`` (N2 material table), ``structures`` (CoRE/QMOF/IZA/OPTIMADE/MP),
``stability`` (MOFSimplify feasibility gate), ``refrigerants`` (working
fluids), ``pinn_corpus`` (synthetic + real-condition PINN inputs), plus a
``README`` index with lineage and validation summary.

Separation rule: stdlib + pandas only for probes. No harness envs/backends,
no RL, no training imports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "data_cache"
DEFAULT_OUT = REPO_ROOT / "docs" / "datasets"


def _read_json(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text())
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _manifest(sub: str) -> dict | None:
    return _read_json(CACHE / sub / "manifest.json")


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.4g}"
    if isinstance(v, bool):
        return str(v)
    return str(v)


def _kv_table(d: dict) -> str:
    lines = ["| key | value |", "|---|---|"]
    for k, v in d.items():
        lines.append(f"| `{k}` | {_fmt(v)} |")
    return "\n".join(lines)


def _validation_block() -> str:
    """One-line-per-gate summary; failures name the build command."""
    try:
        from validate import validate_all  # noqa: PLC0415
    except ImportError:
        try:
            from data.validate import validate_all  # noqa: PLC0415
        except ImportError:
            return "_validation unavailable (import failed)._"
    try:
        reports = validate_all(cache_dir=CACHE)
    except Exception as exc:  # noqa: BLE001
        return f"_validation failed to run: {exc}_"
    lines = ["| gate | ok | issues |", "|---|---|---|"]
    for name, rep in reports.items():
        flag = "✅" if rep.ok else "❌"
        first = rep.issues[0] if rep.issues else "—"
        lines.append(f"| `{name}` | {flag} | {first} |")
    return "\n".join(lines)


def card_isodb() -> str:
    m = _manifest("isodb") or {}
    counts = {k: m.get(k) for k in
              ("total", "pure_water", "mixed_water", "unique_adsorbents",
               "rows_in_temp_window") if k in m}
    return "\n".join([
        "# ISODB — experimental water isotherms",
        "",
        "**Role:** L1 experimental adsorption labels — the raw material every",
        "fit, label, and match downstream consumes.",
        "",
        "## Source / license / access",
        "",
        "- NIST ISODB bulk mirror: `git clone https://github.com/NIST-ISODB/isodb-library`",
        "  → `data_cache/isodb/mirror/` (one JSON per isotherm + `DOI_mapping.csv`).",
        "- Targeted refreshes via the JSON/XML/CSV REST APIs at <https://adsorption.nist.gov>.",
        "- License: source records carry their own citations (see `doi` column);",
        "  the export preserves provenance per row.",
        "",
        "## Build",
        "",
        "```bash",
        "python adsorbent-ml/data/nist_isodb.py   # → water_isotherms.parquet + adsorbent_index.csv + manifest.json",
        "```",
        "",
        "## Counts (from `data_cache/isodb/manifest.json`)",
        "",
        _kv_table(counts) if counts else "_not built yet._",
        "",
        "## Schema (`water_isotherms.parquet`)",
        "",
        "| column | notes |",
        "|---|---|",
        "| `isotherm_id, doi, material_name, matched_refcode?` | provenance |",
        "| `T_K, pressure[], uptake[], uptake_unit, original_unit` | originals preserved |",
        "| `temperature` window | 280–380 K application window |",
        "",
        "Unit normalization to kg/kg vs Pa happens in `fit_da.py`, not here.",
        "",
        "## Gotchas",
        "",
        "- Filename casing varies (`isotherm2.json` / `Isotherm4.json`); some files",
        "  are misspelled (`isothem10.json`) — the exporter walks `Library/<doi>/*.json`",
        "  and skips the three registry directories, never filters by filename.",
        "- Uptake units are heterogeneous (mmol/g, cm³(STP)/g, molecules/unitcell,",
        "  g/g, mg/g, wt% …); `molecules/unitcell` needs framework cell composition.",
        "- Material names are non-standard (\"Cu-BTC\", \"MOF-199\", \"MIL-53(Al)…\");",
        "  see the `labels` card for the matching strategy.",
        "",
        "## Feeds",
        "",
        "- `fits` card (D–A parameters) ← pure-water rows in the 280–380 K window.",
        "- `labels` card matching queue (`unmatched_names.csv`) ← adsorbent index.",
        "",
        "## Validation",
        "",
        _validation_block(),
        "",
    ])


def card_fits() -> str:
    qc = CACHE / "fits" / "da_params_qc.md"
    qc_note = (f"`data_cache/fits/da_params_qc.md` ({qc.stat().st_size} bytes)"
               if qc.exists() else "_QC report not built yet._")
    try:
        import pandas as pd  # noqa: PLC0415

        p = CACHE / "fits" / "da_params.csv"
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            by_flag = df["fit_flag"].value_counts().to_dict() if "fit_flag" in df.columns else {}
            n_qst = int(df[df["q_st_j_kg"].notna()]["material_id"].nunique()) \
                if "q_st_j_kg" in df.columns else 0
            counts = {"isotherm_rows": len(df),
                      "adsorbents": int(df["material_id"].nunique()) if "material_id" in df.columns else "—",
                      "adsorbents_with_Q_st": n_qst,
                      **{f"flag:{k}": int(v) for k, v in by_flag.items()}}
        else:
            counts = {}
    except Exception:
        counts = {}
    return "\n".join([
        "# Fits — Dubinin–Astakhov parameters per isotherm",
        "",
        "**Role:** parameter estimation (not model training): one independent,",
        "deterministic fit per isotherm. Output is the label set Stage-1/2",
        "surrogates train against AND the parameter source `harness.physics` consumes.",
        "",
        "## Source",
        "",
        "- Library: `adsorbent-ml/data/fit_da.py` — adsorbate- and unit-agnostic",
        "  (`fit_isotherm_da`, `isosteric_heat`, `fit_all`); the CLI is thin.",
        "- Physics: `q = q_sat · exp(−(A/E)^n)`, `A = R·T·ln(Psat(T)/P)`, `R = 8.314 J/(mol·K)`",
        "  — identical constants to `harness.physics.thermo` by contract.",
        "- `Q_st` via Clausius–Clapeyron on fitted curves at common uptake levels;",
        "  `None` when < 2 usable temperatures ≥ 5 K apart (never imputed).",
        "",
        "## Build",
        "",
        "```bash",
        "python adsorbent-ml/data/fit_da.py   # → data_cache/fits/da_params.csv + da_params_qc.md",
        "```",
        "",
        "## Counts (live probe of `da_params.csv`)",
        "",
        _kv_table(counts) if counts else "_not built yet._",
        "",
        f"QC report: {qc_note}",
        "",
        "## Schema (DESIGN §8.1 subset)",
        "",
        "| column | unit | notes |",
        "|---|---|---|",
        "| `material_id, name, source, isotherm_id, doi` | — | provenance |",
        "| `q_sat_kg_kg` | kg/kg | mass-normalized only where convertible |",
        "| `q_st_j_kg` | J/kg | multi-T subset only |",
        "| `e_char_j_mol, n_da` | J/mol, – | D–A parameters (unit-independent) |",
        "| `t_range_c` | °C | fit validity window; env warns outside it |",
        "| `fit_rmse, n_points, fit_flag, fit_detail, psat_pa` | — | diagnostics are first-class |",
        "",
        "`fit_flag ∈ {ok, poor_fit, S_shaped, insufficient_points, out_of_window}`;",
        "`q_sat_physical` flags mass-normalized q_sat outside 1e-4–2.5 kg/kg",
        "(mislabeled source units — the fit itself stays valid).",
        "",
        "## Gotchas",
        "",
        "- Points at/above Psat are excluded (condensed phase, outside the model).",
        "- S-shaped (Type-V, MIL-101 class) isotherms fail D–A by design — flagged, kept.",
        "- Single-T adsorbents carry no `Q_st`; downstream N1 conditioning skips them.",
        "",
        "## Feeds",
        "",
        "- `labels` card ← usable (`ok` + physical + mass-normalized) fits, median-aggregated.",
        "- `harness.materials.load_materials_csv` ← same CSV (§8.1 schema).",
        "",
        "## Validation",
        "",
        _validation_block(),
        "",
    ])


def card_labels() -> str:
    m = _manifest("n2") or {}
    counts = {k: m.get(k) for k in
              ("n_materials", "n_isodb", "n_anchors", "n_with_Q_st") if k in m}
    if "families" in m and isinstance(m["families"], dict):
        for fam, n in sorted(m["families"].items()):
            counts[f"family:{fam}"] = n
    if not counts:
        try:
            import pandas as pd  # noqa: PLC0415

            p = CACHE / "n2" / "labels.csv"
            if p.exists():
                df = pd.read_csv(p)
                counts = {"n_materials": len(df),
                          "n_with_Q_st": int(df["q_st_j_kg"].notna().sum()) if "q_st_j_kg" in df.columns else 0}
        except Exception:
            pass
    match_note = ""
    try:
        import pandas as pd  # noqa: PLC0415

        mp = CACHE / "n2" / "matched.parquet"
        if mp.exists():
            md = pd.read_parquet(mp)
            by_kind = md["match_kind"].value_counts().to_dict() if "match_kind" in md.columns else {}
            n_match = int((md["match_kind"] != "none").sum()) if "match_kind" in md.columns else 0
            match_note = (f"Matching: **{n_match}/{len(md)}** "
                          f"({n_match / len(md):.0%}); by kind `{by_kind}`.")
        else:
            match_note = "Matching: `matched.parquet` not built yet."
    except Exception as exc:  # noqa: BLE001
        match_note = f"Matching probe failed: {exc}"
    return "\n".join([
        "# Labels — N2 per-material training table",
        "",
        "**Role:** one row per material: the supervised table the tabular baseline",
        "and (gated) the Stage-2 GNN train against.",
        "",
        "## Source",
        "",
        "- Builder: `adsorbent-ml/data/labels.py` — usable ISODB fits",
        "  (`fit_flag == ok`, physical `q_sat`, mass-normalized) median-aggregated",
        "  across isotherms; spread kept as `q_sat_logstd` (honest disagreement proxy).",
        "- 13 commercial anchors (`adsorbent-ml/data/anchors.csv`) appended as gold",
        "  rows (`anchor=True`): silica gel RD, 13X/NaA, AlPO-18/SAPO-34, MIL-101(Cr),",
        "  UiO-66, Mg-MOF-74, CAU-10-H, aluminum fumarate, silicalite, carbon, LiCl/silica.",
        "- Targets are log-space (`log_q_sat`, `log_Q_st`, `log_E`; `n_da` linear);",
        "  `log_Q_st` on the multi-T subset only — single-T rows carry no isosteric",
        "  heat and are never imputed.",
        "- Families: v1 keyword rules (zeolite/silica/carbon/MOF/COF/composite/other),",
        "  coarse but leak-safe for GroupKFold; node/topology families arrive Stage 2.",
        "",
        "## Build",
        "",
        "```bash",
        "python adsorbent-ml/training/train_baseline.py --out data_cache/n2   # writes labels.csv + manifest",
        "python adsorbent-ml/data/match_structures.py                          # writes matched.parquet + unmatched_names.csv",
        "```",
        "",
        "## Counts (from `data_cache/n2/manifest.json` + live probe)",
        "",
        _kv_table(counts) if counts else "_not built yet._",
        "",
        match_note,
        "",
        "Name→structure matching (cheapest first): IZA trade-name rules → verified",
        "CoRE substrings (≥1 filename hit required; <4-char substrings rejected —",
        "the UKEBOD lesson) → anchor join → Ongari CSD tables → QMOF join →",
        "DOI bridge. Remainder → `unmatched_names.csv` curation queue.",
        "",
        "## Schema (`labels.csv`)",
        "",
        "| column | notes |",
        "|---|---|---|",
        "| `material_id, name, source, anchor, family` | identity + split key |",
        "| `q_sat_kg_kg, q_sat_logstd, n_isotherms` | median + spread + support |",
        "| `q_st_j_kg` | NaN when single-T (never imputed) |",
        "| `e_char_j_mol, n_da` | D–A medians |",
        "| `log_q_sat, log_Q_st, log_E` | training targets |",
        "",
        "## Feeds",
        "",
        "- `features/descriptors.py` → 30-col matrix (pores + family + composition).",
        "- `models/tabular_baseline.py` → HistGradientBoosting per target, GroupKFold OOF.",
        "- `eval/rank_eval.py` → OOF predictions through the cycle oracle (top-k hit rate).",
        "",
        "## Validation",
        "",
        _validation_block(),
        "",
    ])


def card_structures() -> str:
    rows: dict = {}
    for sub, label in (("core_mof", "CoRE MOF 2019-ASR"), ("qmof", "QMOF"),
                       ("iza", "IZA-SC"), ("mp", "Materials Project")):
        m = _manifest(sub)
        if not m:
            continue
        for k in ("n_structures", "n_frameworks", "property_columns",
                  "dataset", "figshare_article"):
            if k in m:
                rows[f"{sub}:{k}"] = m[k]
    # Live CIF counts (cheap glob).
    for name, d in (("core_mof", CACHE / "core_mof" / "structures"),
                    ("qmof", CACHE / "qmof" / "relaxed_structures"),
                    ("iza", CACHE / "iza" / "frameworks"),
                    ("mp", CACHE / "mp" / "structures")):
        if d.exists():
            rows[f"{name}:cif_on_disk"] = sum(1 for _ in d.glob("*.cif"))
    # OPTIMADE pulls (one manifest per db).
    optim: dict = {}
    opt_root = CACHE / "optimade"
    if opt_root.exists():
        for sub in sorted(opt_root.glob("*")):
            man = sub / "manifest.json"
            if man.exists():
                m = _read_json(man) or {}
                optim[sub.name] = m.get("n_entries", "?")
    if optim:
        for db, n in optim.items():
            rows[f"optimade/{db}:entries"] = n
    return "\n".join([
        "# Structures — frameworks behind the labels",
        "",
        "**Role:** crystal structures + computed properties for featurization",
        "(Stage-1 pores/composition) and Stage-2 crystal-graph inputs; also the",
        "match targets for ISODB names and the L2 (GCMC) label-generator enabler.",
        "",
        "## Sources / licenses / access",
        "",
        "| source | what | access |",
        "|---|---|---|",
        "| CoRE MOF 2019-ASR | 12k solvent-removed experimental MOFs + 28-col porosity/DDEC table | pip `CoRE-MOF` bulk tar.xz (MIT code / CC BY 4.0 data) |",
        "| QMOF | 20k DFT MOFs × 94 cols (gaps, energies, DDEC6/CM5 charges) + optimized CIFs | Figshare doi 10.6084/m9.figshare.13147324 |",
        "| IZA-SC | ~250 approved zeolite frameworks, free CIFs | `iza_export.py` live scrape (EFN/EWE publish none) |",
        "| OPTIMADE | one client, ~40 structure DBs (MC3D/MC2D/pyrene-MOFs/CURATED-COFs, …) | `optimade_export.py` filtered queries |",
        "| Materials Project | chemsys candidates + CIFs (vendored query layer) | `mp_export.py` (needs `MP_API_KEY`) |",
        "",
        "## Build",
        "",
        "```bash",
        "python adsorbent-ml/data/core_mof_export.py",
        "python adsorbent-ml/data/qmof_export.py",
        "python adsorbent-ml/data/iza_export.py",
        "python adsorbent-ml/data/optimade_export.py --list-providers",
        "python adsorbent-ml/data/mp_export.py --apps datacenter --max-generated-chemsys 3 --limit-per-system 5",
        "```",
        "",
        "## Counts (manifests + CIF globs)",
        "",
        _kv_table(rows) if rows else "_not built yet._",
        "",
        "## Gotchas",
        "",
        "- CoRE \"~14k\" counts include non-public CSD entries; the public ASR pull is 12,020.",
        "- MOF cells are big: OPTIMADE queries need `nsites<2000`, not the `<50` default.",
        "- OPTIMADE index DBs answer `/info` with only `[\"info\",\"links\"]` endpoints —",
        "  filter by `available_endpoints` containing `structures`.",
        "- QMOF↔CoRE join key is formula/composition (imperfect overlap, documented).",
        "",
        "## Feeds",
        "",
        "- Pore descriptors (LCD/PLD/AV) + family/topology splits → `labels` card.",
        "- QMOF partial charges → future L2 (RASPA GCMC) on active-learning shortlists only.",
        "",
        "## Validation",
        "",
        _validation_block(),
        "",
    ])


def card_stability() -> str:
    m = _manifest("stability") or {}
    counts = {k: m.get(k) for k in
              ("rows", "n_with_solvent_label", "n_with_decomp_c") if k in m}
    split_note = ""
    try:
        from stability import feasibility_filter, load_stability_csv  # noqa: PLC0415

        p = CACHE / "stability" / "mofsimplify_ssd_tsd.csv"
        if p.exists():
            recs = load_stability_csv(p)
            ok, bad = feasibility_filter(recs)
            counts["n_pass_default_gate_150C"] = len(ok)
            counts["n_fail_default_gate_150C"] = len(bad)
            split_note = (f"Default gate (`min_decomp_c=150 °C`, unknown water fails): "
                          f"**{len(ok)} pass / {len(bad)} fail** of {len(recs)}.")
        else:
            split_note = "`mofsimplify_ssd_tsd.csv` not built yet."
    except Exception as exc:  # noqa: BLE001
        split_note = f"Live probe failed: {exc}"
    return "\n".join([
        "# Stability — feasibility gate before ranking",
        "",
        "**Role:** N2 must not propose adsorbents that collapse under water cycling",
        "or decompose below regeneration temperature. This gate runs BEFORE",
        "system ranking — never as a training target.",
        "",
        "## Source / license / access",
        "",
        "- MOFSimplify (MIT Kulik lab, Nandy et al. Sci. Data 2022), Zenodo record 5737968",
        "  (`SciData.zip`): `full_SSD_data.csv` (text-mined solvent-removal stability) +",
        "  `full_TSD_data.csv` (TGA decomposition onsets). Manual Zenodo download,",
        "  then normalized by the exporter; raw zip cached under",
        "  `data_cache/stability/_raw/`.",
        "",
        "## Build",
        "",
        "```bash",
        "python adsorbent-ml/data/stability_export.py          # download + build (cached under _raw/)",
        "python adsorbent-ml/data/stability_export.py --local # rebuild from an existing download",
        "```",
        "",
        "## Counts (manifest + live filter probe)",
        "",
        _kv_table(counts) if counts else "_not built yet._",
        "",
        split_note,
        "",
        "## Schema (`mofsimplify_ssd_tsd.csv`)",
        "",
        "| column | notes |",
        "|---|---|---|",
        "| `mof_name` | CSD refcode (one row per refcode, merged SSD+TSD) |",
        "| `water_stable` | intentionally EMPTY — the dataset measures desolvation, not hydrolysis |",
        "| `solvent_removal_stable` | `true/false` text-mined desolvation label |",
        "| `thermal_decomp_c` | TGA onset [°C], empty ⇒ None |",
        "| `source, confidence` | DOI + `text-mined` / `tga-extracted` |",
        "",
        "Loader (`stability.py`): alias-tolerant headers; `water_stable` accepts",
        "`true/false/1/0/yes/no/stable/unstable`; rows without a name are skipped (counted).",
        "",
        "## Gate semantics (`feasibility_filter`)",
        "",
        "- Unknown water stability FAILS by default (`require_water_stable=True`) —",
        "  unknown ≠ stable. Exception: solvent-removal-stable counts as conservative",
        "  desolvation evidence; an explicit `water_stable=false` never passes.",
        "- Missing TGA never fails alone (absence of data is not decomposition).",
        "- Thermal gate: `thermal_decomp_c >= min_decomp_c` (default 150 °C).",
        "",
        "## Feeds",
        "",
        "- `datasets.stability_shortlist()` → `(pass, fail)` DataFrames for N2 screening.",
        "- GUI data explorer → Stability tab (gate + TGA histogram).",
        "",
        "## Validation",
        "",
        _validation_block(),
        "",
    ])


def card_refrigerants() -> str:
    try:
        from refrigerants import REFRIGERANTS  # noqa: PLC0415

        rows = ["| fluid | Antoine window [°C] | Tb [°C] | ASHRAE 34 | GWP100 |",
                "|---|---|---|---|---|"]
        for name in sorted(REFRIGERANTS):
            s = REFRIGERANTS[name]
            rows.append(f"| `{name}` | {s['t_min_c']:g}–{s['t_max_c']:g} | "
                        f"{s['tb_c']:g} | {s['ashrae34']} | {s['gwp100']} |")
        fluid_table = "\n".join(rows)
        probe = []
        try:
            from refrigerants import psat_pa  # noqa: PLC0415

            meoh = float(psat_pa("methanol", 35.0 + 273.15)) / 1e3
            nh3 = float(psat_pa("ammonia", 35.0 + 273.15)) / 1e3
            probe.append(f"Spot check (live): methanol Psat(35 °C) = {meoh:.1f} kPa (~28 kPa); "
                         f"ammonia = {nh3:.0f} kPa (~1350 kPa).")
        except Exception as exc:  # noqa: BLE001
            probe.append(f"Spot check failed: {exc}")
        probe_text = "\n".join(probe)
    except Exception as exc:  # noqa: BLE001
        fluid_table = f"_refrigerants library unavailable: {exc}_"
        probe_text = ""
    return "\n".join([
        "# Refrigerants — working fluids beyond water",
        "",
        "**Role:** trusted thermodynamic reference OUTSIDE the differentiable path",
        "(same bounded role as CoolProp): working-pair expansion + validator.",
        "Never imported by `harness.physics` or the PINN.",
        "",
        "## Source",
        "",
        "- Library: `adsorbent-ml/data/refrigerants.py` (stdlib + numpy).",
        "- Antoine Psat (`log10(P/mmHg) = A − B/(T/°C + C)`, NIST WebBook-consistent",
        "  constants) + Watson latent heat anchored to the tabulated `h_fg` (exact",
        "  at the anchor by construction).",
        "",
        "## Fluids",
        "",
        fluid_table,
        "",
        probe_text,
        "",
        "Water branch agrees with `harness.physics.thermo` to < 3 % over 5–95 °C",
        "(both approximate IAPWS-IF97); boiling points at 1 atm within 2 %.",
        "",
        "## Fail-loudly contract",
        "",
        "- Out-of-window calls RAISE (no silent extrapolation into the desorption band):",
        "  water 1–100 °C, methanol −20–80 °C, ethanol 0–80 °C, ammonia −60–60 °C.",
        "- Unknown fluids raise `KeyError`. Extending a range via CoolProp/REFPROP",
        "  is an explicit follow-up (see ACQUISITION §7).",
        "",
        "## Feeds",
        "",
        "- Validator upgrades (dense-grid reference calls).",
        "- `rank.py` working-pair shortlists; N1 conditioning rows for non-water",
        "  phases (P ranges differ 10–100× across fluids).",
        "- `datasets.refrigerants_table()` exposes this card's numbers as a DataFrame.",
        "",
        "## Validation",
        "",
        _validation_block(),
        "",
    ])


def card_pinn_corpus() -> str:
    return "\n".join([
        "# PINN corpus — synthetic fields + real-condition rows",
        "",
        "**Role:** training inputs for the bed PINN (N1): infinite cheap synthetic",
        "phase fields for the curriculum base + scarce real-condition rows for",
        "physics-residual collocation and hardening.",
        "",
        "## Source",
        "",
        "- Builder: `adsorbent-ml/data/corpus.py` on the trusted solver",
        "  (`harness.physics.bed1d.step_bed` — exact-exponential LDF + RK4 conduction).",
        "- Synthetic: single-phase `T(x,t)` / `q(x,t)` rollouts with FIXED fluid T",
        "  and vapour P (no valve logic — the PINN scope boundary); `dt=None`",
        "  auto-derives from the conduction CFL bound (the `Bed1D` env convention).",
        "- Real: ISODB-derived D–A fits + 13 anchors as *conditioning rows* (no `T(x,t)`",
        "  fields — never reported as field accuracy). Single-T rows without `Q_st`",
        "  are SKIPPED for N1 (the energy equation needs it; imputing would launder",
        "  a guess into a physics label). They remain valid N2 targets.",
        "- Transport where fits carry none: class defaults (`rho_s` 800 kg/m³,",
        "  `c_s` 1000 J/kg/K, `k_eff` 0.2 W/m/K, `k_ldf` 2.5e-4 1/s) — flagged via",
        "  `transport_provenance`, never silent (DESIGN §8.1 honesty rule).",
        "",
        "## Build (in-memory; no cache artifact)",
        "",
        "```python",
        "from corpus import preset_configs, build_corpus, load_real_conditions",
        "data, manifest = build_corpus(preset_configs(), n_data_per_config=800)",
        "conds, report = load_real_conditions()",
        "```",
        "",
        "Preset sweep: 2 materials × 2 thicknesses (mm-scale so CFL dt stays",
        "tractable); kinetics τ ~ 100 s, horizons ~40–100 s — the coupled transient",
        "the PINN must learn.",
        "",
        "## Splits",
        "",
        "- `family_split()`: whole families to train/val — never random",
        "  (random splits leak near-duplicate frameworks).",
        "",
        "## Feeds",
        "",
        "- `models/bed_pinn.py` coordinate PINN (`x_hat, t_hat, cond[15] → T, q`).",
        "- `training/train_pinn.py` curriculum (data warm-up → physics ramp →",
        "  real-condition hardening); `eval/pinn_metrics.py` acceptance gates.",
        "",
        "## Validation",
        "",
        _validation_block(),
        "",
    ])


CARDS: dict[str, tuple[str, object]] = {
    "isodb": ("ISODB — experimental water isotherms", card_isodb),
    "fits": ("Fits — D–A parameters per isotherm", card_fits),
    "labels": ("Labels — N2 per-material training table", card_labels),
    "structures": ("Structures — frameworks behind the labels", card_structures),
    "stability": ("Stability — feasibility gate before ranking", card_stability),
    "refrigerants": ("Refrigerants — working fluids beyond water", card_refrigerants),
    "pinn_corpus": ("PINN corpus — synthetic fields + real rows", card_pinn_corpus),
}


def card_readme() -> str:
    lines = ["# Dataset cards", "",
             "Generated from live `data_cache/` manifests — do not hand-edit numbers here.",
             "Regenerate after any rebuild:",
             "",
             "```bash",
             "python adsorbent-ml/data/cards.py",
             "```",
             "",
             "## Lineage",
             "",
             "```",
             "ISODB raw mirror → normalized isotherms → D–A fits → N2 labels ─┬─→ features → baseline → rank_eval",
             "structures (CoRE/QMOF/IZA/OPTIMADE/MP) ──→ matching ─────────────┘        (top-k hit rate)",
             "stability (MOFSimplify) ──→ feasibility gate ──→ screening shortlists",
             "refrigerants (Antoine/Watson) ──→ validator + working-pair expansion",
             "harness Bed1D solver ──→ PINN corpus (synthetic fields + real conditions)",
             "```",
             "",
             "## Cards",
             ""]
    for slug, (title, _) in CARDS.items():
        lines.append(f"- [`{slug}.md`]({slug}.md) — {title}")
    lines += ["", "## Validation snapshot (generated at card-build time)", "",
              _validation_block(), ""]
    return "\n".join(lines)


def write_dataset_cards(out_dir: str | Path = DEFAULT_OUT) -> dict[str, Path]:
    """Write all cards + README index. Returns ``{slug: path}``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for slug, (_, fn) in CARDS.items():
        p = out / f"{slug}.md"
        p.write_text(fn(), encoding="utf-8")
        written[slug] = p
    readme = out / "README.md"
    readme.write_text(card_readme(), encoding="utf-8")
    written["README"] = readme
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate dataset cards from live manifests.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Output directory for markdown cards.")
    args = ap.parse_args()
    written = write_dataset_cards(args.out)
    for slug, path in written.items():
        print(f"Wrote {path} ({slug})")


if __name__ == "__main__":
    main()
