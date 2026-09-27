"""Validation gates for the data-curation pipeline.

Each ``validate_*`` takes an in-memory table (or live library, for
refrigerants) and returns a :class:`ValidationReport` — ``ok`` plus a list
of human-readable ``issues`` plus machine-readable ``stats``. Builders call
these before writing releases; ``validate_all()`` runs every gate that has
its inputs on disk and never raises for missing caches (missing input is a
failed gate with a build-command hint, not an exception).

Separation rule: stdlib + pandas/numpy only. No harness envs/backends, no
RL, no training imports — the data layer must stay importable without the
``rl`` extra (locked by ``test_data_layer_independence``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "data_cache"

# Physical bands shared with fit_da.py (single source of truth lives there;
# duplicated as literals here so this module never imports the fitter CLI).
Q_SAT_BAND = (1e-4, 2.5)      # kg/kg water (fit_da q_sat_bounds_kg_kg)
E_BOUNDS = (200.0, 60000.0)   # J/mol
N_BOUNDS = (1.0, 6.0)
DECOMP_RANGE_C = (-50.0, 1000.0)
ALLOWED_FAMILIES = {"zeolite", "silica", "carbon", "mof", "cof",
                    "composite", "other"}


@dataclass
class ValidationReport:
    name: str
    ok: bool
    issues: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok,
                "issues": list(self.issues), "stats": dict(self.stats)}


def _finite(s) -> list[float]:
    import math

    vals = []
    for v in list(s):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            vals.append(f)
    return vals


def validate_fits(df) -> ValidationReport:
    """Gate the per-isotherm D-A fit table (``data_cache/fits/da_params.csv``)."""
    issues: list[str] = []
    stats: dict = {"n_rows": int(len(df))}
    required = ["material_id", "name", "q_sat_kg_kg", "e_char_j_mol",
                "n_da", "fit_flag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return ValidationReport("fits", False,
                                [f"missing column(s) {missing}"], stats)
    if len(df) == 0:
        return ValidationReport("fits", False, ["empty fit table"], stats)

    by_flag = df["fit_flag"].value_counts().to_dict()
    stats["by_flag"] = {str(k): int(v) for k, v in by_flag.items()}
    usable = df[df["fit_flag"] == "ok"]
    stats["n_usable"] = int(len(usable))
    stats["usable_frac"] = float(len(usable) / len(df))
    stats["n_adsorbents"] = int(df["material_id"].nunique())
    if "q_st_j_kg" in df.columns:
        stats["n_with_qst"] = int(df[df["q_st_j_kg"].notna()]["material_id"].nunique())
    if "fit_rmse" in df.columns and len(usable):
        vals = _finite(usable["fit_rmse"].dropna())
        if vals:
            vals_sorted = sorted(vals)
            stats["median_nrmse_usable"] = float(vals_sorted[len(vals_sorted) // 2])

    if len(usable) == 0:
        issues.append("no usable (fit_flag == ok) rows")
    if stats["usable_frac"] < 0.10:
        issues.append(f"usable fraction {stats['usable_frac']:.2f} < 0.10 — check fitter or window")

    if len(usable):
        qs = _finite(usable["q_sat_kg_kg"].dropna())
        out = [v for v in qs if not (Q_SAT_BAND[0] <= v <= Q_SAT_BAND[1])]
        stats["n_qsat_outside_band"] = len(out)
        if qs and len(out) > 0.5 * len(qs):
            issues.append(f"{len(out)}/{len(qs)} usable q_sat outside {Q_SAT_BAND} kg/kg")
        es = _finite(usable["e_char_j_mol"].dropna())
        bad_e = [v for v in es if not (E_BOUNDS[0] <= v <= E_BOUNDS[1])]
        if bad_e:
            issues.append(f"{len(bad_e)} usable E outside {E_BOUNDS} J/mol")
        ns = _finite(usable["n_da"].dropna())
        bad_n = [v for v in ns if not (N_BOUNDS[0] <= v <= N_BOUNDS[1])]
        if bad_n:
            issues.append(f"{len(bad_n)} usable n outside {N_BOUNDS}")
        if usable["name"].astype(str).str.strip().eq("").any():
            issues.append("usable rows with empty material name")
    return ValidationReport("fits", not issues, issues, stats)


def validate_labels(df) -> ValidationReport:
    """Gate the per-material N2 label table (``data_cache/n2/labels.csv``)."""
    issues: list[str] = []
    stats: dict = {"n_rows": int(len(df))}
    required = ["material_id", "name", "family", "q_sat_kg_kg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return ValidationReport("labels", False,
                                [f"missing column(s) {missing}"], stats)
    if len(df) == 0:
        return ValidationReport("labels", False, ["empty label table"], stats)
    if int(df["material_id"].nunique()) != len(df):
        issues.append("duplicate material_id rows (one row per material required)")
    if df["name"].astype(str).str.strip().eq("").any():
        issues.append("rows with empty name")
    unknown_fam = sorted(set(df["family"].astype(str)) - ALLOWED_FAMILIES)
    if unknown_fam:
        issues.append(f"unknown families {unknown_fam}")
    stats["families"] = df["family"].value_counts().to_dict()
    qs = _finite(df["q_sat_kg_kg"].dropna())
    bad_q = [v for v in qs if v <= 0]
    if bad_q:
        issues.append(f"{len(bad_q)} non-positive q_sat rows")
    if "anchor" in df.columns:
        stats["n_anchors"] = int(df["anchor"].fillna(False).astype(bool).sum())
    if "q_st_j_kg" in df.columns:
        stats["n_with_Q_st"] = int(df["q_st_j_kg"].notna().sum())
        qsts = _finite(df["q_st_j_kg"].dropna())
        bad_qst = [v for v in qsts if v <= 0]
        if bad_qst:
            issues.append(f"{len(bad_qst)} non-positive Q_st rows (never impute single-T)")
    else:
        stats["n_with_Q_st"] = 0
    for col in ("log_q_sat", "log_E", "log_Q_st"):
        if col in df.columns:
            vals = df[col].dropna()
            import math

            nonfinite = sum(1 for v in vals if not math.isfinite(float(v)))
            if nonfinite:
                issues.append(f"{nonfinite} non-finite {col} values")
    return ValidationReport("labels", not issues, issues, stats)


def validate_matched(matched) -> ValidationReport:
    """Gate the name→structure match table (``data_cache/n2/matched.parquet``)."""
    issues: list[str] = []
    stats: dict = {"n_rows": int(len(matched))}
    if len(matched) == 0:
        return ValidationReport("matched", False, ["empty match table"], stats)
    if "match_kind" not in matched.columns:
        return ValidationReport("matched", False,
                                ["missing match_kind column"], stats)
    by_kind = matched["match_kind"].value_counts().to_dict()
    stats["by_kind"] = {str(k): int(v) for k, v in by_kind.items()}
    n_none = int((matched["match_kind"] == "none").sum())
    stats["n_unmatched"] = n_none
    stats["match_rate"] = float(1.0 - n_none / len(matched)) if len(matched) else 0.0
    if "ambiguous" in matched.columns:
        stats["n_ambiguous"] = int(matched["ambiguous"].fillna(False).astype(bool).sum())
    if "refcode" in matched.columns:
        stats["n_with_refcode"] = int((matched["refcode"].astype(str) != "").sum())
    if stats["match_rate"] < 0.10:
        issues.append(f"match rate {stats['match_rate']:.2f} < 0.10 — matching is the bottleneck")
    return ValidationReport("matched", not issues, issues, stats)


def validate_stability(records_or_df) -> ValidationReport:
    """Gate the MOFSimplify stability table (pass/fail feasibility counts)."""
    import math

    try:
        import pandas as pd  # noqa: PLC0415

        is_df = isinstance(records_or_df, pd.DataFrame)
    except ImportError:
        is_df = False
    if is_df:
        df = records_or_df
        names = df["name"].astype(str) if "name" in df.columns else df.iloc[:, 0].astype(str)
        decs = df["thermal_decomp_c"] if "thermal_decomp_c" in df.columns else None
        n = len(df)
        n_nameless = int(names.str.strip().eq("").sum())
        bad_decomp = 0
        if decs is not None:
            for v in decs.dropna():
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    bad_decomp += 1
                    continue
                if not (DECOMP_RANGE_C[0] <= f <= DECOMP_RANGE_C[1] and math.isfinite(f)):
                    bad_decomp += 1
    else:
        recs = list(records_or_df)
        n = len(recs)
        n_nameless = sum(1 for r in recs if not str(getattr(r, "name", "")).strip())
        bad_decomp = sum(
            1 for r in recs
            if getattr(r, "thermal_decomp_c", None) is not None
            and not (DECOMP_RANGE_C[0] <= float(getattr(r, "thermal_decomp_c")) <= DECOMP_RANGE_C[1]))
    issues: list[str] = []
    stats: dict = {"n_rows": n, "n_nameless": n_nameless,
                   "n_decomp_out_of_range": bad_decomp}
    if n == 0:
        return ValidationReport("stability", False, ["empty stability table"], stats)
    if n_nameless:
        issues.append(f"{n_nameless} nameless rows")
    if bad_decomp:
        issues.append(f"{bad_decomp} decomposition onsets outside {DECOMP_RANGE_C} °C")
    # Feasibility split at the default gate (informational, never a failure).
    try:
        from stability import feasibility_filter  # noqa: PLC0415

        if not is_df:
            ok, bad = feasibility_filter(list(records_or_df))
            stats["n_pass_default_gate"] = len(ok)
            stats["n_fail_default_gate"] = len(bad)
    except Exception:
        pass
    return ValidationReport("stability", not issues, issues, stats)


def validate_refrigerants() -> ValidationReport:
    """Gate the working-fluid library (``data/refrigerants.py``).

    Always available (no cache needed): pins boiling points, the
    ACQUISITION §7 spot pressures (methanol ≈ 28 kPa, ammonia ≈ 1350 kPa at
    35 °C), water-branch agreement with ``harness.physics.thermo``, and the
    fail-loudly contract outside Antoine windows.
    """
    issues: list[str] = []
    stats: dict = {}
    try:
        from refrigerants import REFRIGERANTS, h_fg_j_kg, psat_pa  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ValidationReport("refrigerants", False,
                                [f"cannot import refrigerants library: {exc}"], stats)
    stats["fluids"] = sorted(REFRIGERANTS)
    if set(REFRIGERANTS) != {"water", "methanol", "ethanol", "ammonia"}:
        issues.append(f"unexpected fluid set {sorted(REFRIGERANTS)}")
    for fluid, tb_c in (("water", 100.0), ("methanol", 64.7),
                        ("ethanol", 78.4), ("ammonia", -33.3)):
        try:
            p = float(psat_pa(fluid, tb_c + 273.15))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"{fluid} Psat raised {exc}")
            continue
        if abs(p - 760 * 133.322) / (760 * 133.322) > 0.02:
            issues.append(f"{fluid} boiling point off: Psat(Tb)={p:.0f} Pa")
    try:
        if abs(float(psat_pa("methanol", 35.0 + 273.15)) - 28e3) / 28e3 > 0.05:
            issues.append("methanol Psat(35 °C) ≠ ~28 kPa")
        if abs(float(psat_pa("ammonia", 35.0 + 273.15)) - 1350e3) / 1350e3 > 0.05:
            issues.append("ammonia Psat(35 °C) ≠ ~1350 kPa")
    except Exception as exc:  # noqa: BLE001
        issues.append(f"spot-pressure check raised {exc}")
    try:
        if abs(float(h_fg_j_kg("water", 373.15)) - 2_256_400.0) / 2_256_400.0 > 1e-9:
            issues.append("water Watson anchor moved")
    except Exception as exc:  # noqa: BLE001
        issues.append(f"latent-heat check raised {exc}")
    # Out-of-window must fail loudly (never silently extrapolate).
    for fluid, t_c in (("water", 150.0), ("ammonia", 80.0)):
        try:
            psat_pa(fluid, t_c + 273.15)
            issues.append(f"{fluid} Psat({t_c} °C) did not raise outside window")
        except (ValueError, KeyError):
            pass
    # Water branch vs harness thermo (informational when harness missing).
    try:
        from harness.physics.thermo import water_sat_pressure_pa  # noqa: PLC0415

        import jax.numpy as jnp  # noqa: PLC0415

        worst = 0.0
        for t_c in (5.0, 20.0, 35.0, 60.0, 80.0, 95.0):
            ours = float(psat_pa("water", t_c + 273.15))
            theirs = float(water_sat_pressure_pa(jnp.asarray(t_c + 273.15)))
            worst = max(worst, abs(ours - theirs) / theirs)
        stats["water_vs_harness_max_rel"] = worst
        if worst > 0.03:
            issues.append(f"water branch vs harness {worst:.3f} > 0.03")
    except Exception as exc:  # noqa: BLE001
        stats["water_vs_harness"] = f"skipped ({exc})"
    stats["windows"] = {f: (s["t_min_c"], s["t_max_c"])
                        for f, s in REFRIGERANTS.items()}
    return ValidationReport("refrigerants", not issues, issues, stats)


def validate_all(*, cache_dir: str | Path = CACHE) -> dict[str, ValidationReport]:
    """Run every gate whose inputs exist. Missing inputs are failed gates
    with a build-command hint — never an exception."""
    import pandas as pd  # noqa: PLC0415

    cache = Path(cache_dir)
    out: dict[str, ValidationReport] = {}

    fits_csv = cache / "fits" / "da_params.csv"
    if fits_csv.exists():
        try:
            out["fits"] = validate_fits(pd.read_csv(fits_csv, low_memory=False))
        except Exception as exc:  # noqa: BLE001
            out["fits"] = ValidationReport("fits", False, [f"read failed: {exc}"],
                                           {"path": str(fits_csv)})
    else:
        out["fits"] = ValidationReport("fits", False,
                                       ["da_params.csv missing — build it first: "
                                        "python adsorbent-ml/data/fit_da.py"],
                                       {"path": str(fits_csv)})

    labels_csv = cache / "n2" / "labels.csv"
    if labels_csv.exists():
        try:
            out["labels"] = validate_labels(pd.read_csv(labels_csv))
        except Exception as exc:  # noqa: BLE001
            out["labels"] = ValidationReport("labels", False, [f"read failed: {exc}"],
                                             {"path": str(labels_csv)})
    else:
        out["labels"] = ValidationReport("labels", False,
                                         ["labels.csv missing — build it first: "
                                          "python adsorbent-ml/training/train_baseline.py "
                                          "--out data_cache/n2"],
                                         {"path": str(labels_csv)})

    matched_pq = cache / "n2" / "matched.parquet"
    if matched_pq.exists():
        try:
            out["matched"] = validate_matched(pd.read_parquet(matched_pq))
        except Exception as exc:  # noqa: BLE001
            out["matched"] = ValidationReport("matched", False,
                                              [f"read failed: {exc}"],
                                              {"path": str(matched_pq)})
    else:
        out["matched"] = ValidationReport("matched", False,
                                          ["matched.parquet missing — build it first: "
                                           "python adsorbent-ml/data/match_structures.py"],
                                          {"path": str(matched_pq)})

    stab_csv = cache / "stability" / "mofsimplify_ssd_tsd.csv"
    if stab_csv.exists():
        try:
            from stability import load_stability_csv  # noqa: PLC0415

            out["stability"] = validate_stability(load_stability_csv(stab_csv))
            out["stability"].stats["path"] = str(stab_csv)
        except Exception as exc:  # noqa: BLE001
            out["stability"] = ValidationReport("stability", False,
                                                [f"read failed: {exc}"],
                                                {"path": str(stab_csv)})
    else:
        out["stability"] = ValidationReport("stability", False,
                                            ["mofsimplify_ssd_tsd.csv missing — build it "
                                             "first: python adsorbent-ml/data/"
                                             "stability_export.py --local"],
                                            {"path": str(stab_csv)})

    out["refrigerants"] = validate_refrigerants()
    return out


__all__ = ["ALLOWED_FAMILIES", "DECOMP_RANGE_C", "E_BOUNDS", "N_BOUNDS",
           "Q_SAT_BAND", "ValidationReport", "validate_all", "validate_fits",
           "validate_labels", "validate_matched", "validate_refrigerants",
           "validate_stability"]
