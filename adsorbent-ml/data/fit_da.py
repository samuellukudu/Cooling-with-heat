#!/usr/bin/env python
"""Dubinin–Astakhov fitting: reusable library + thin CLI
(harness/DESIGN.md §12 H1.0 reusability contract).

This is parameter estimation (curve fitting), not model training: each
isotherm is fit independently and deterministically; the output table is
the label set that Stage-1/2 surrogates train against and the parameter
source the harness physics consumes.

Library — adsorbate-agnostic and unit-agnostic, importable without the CLI:

    fit_isotherm_da(p_pa, q, T_K, psat_fn) -> DAFit
        Robust nonlinear least squares for (q_sat, E, n) of
            q = q_sat · exp(−(A/E)^n),   A = R·T·ln(P_sat(T)/P)
        on one single-temperature isotherm. E and n are independent of
        uptake units; q_sat carries the input's uptake units. Points at or
        above P_sat are excluded (condensed phase — outside the model).
    isosteric_heat(pairs, psat_fn) -> float | None
        Absolute isosteric heat [J/mol] via Clausius–Clapeyron from ≥ 2
        usable fits at distinct temperatures, evaluated on the *fitted*
        curves at common uptake levels (robust against point noise).
        Divide by the adsorbate molar mass for J/kg. None when coverage
        is insufficient.
    fit_all(rows, psat_fn, ...) -> pandas.DataFrame
        Batch over isotherm records (dicts with p/q arrays + metadata),
        one output row per isotherm in the harness §8.1 schema plus
        provenance columns. Uptake units are preserved (``q_sat_native``);
        ``q_sat_kg_kg`` is filled only where the unit is mass-normalizable
        for the adsorbate's molar mass.

CLI — deliberately thin (parse → call library → write):

    .venv/bin/python adsorbent-ml/data/fit_da.py             # defaults
    Materials/.venv/bin/python adsorbent-ml/data/fit_da.py   # also works

    data_cache/fits/da_params.csv       §8.1 schema + provenance columns
    data_cache/fits/da_params_qc.md     QC report (flags, RMSE, coverage)

R = 8.314 J/(mol·K) throughout — deliberately identical to
``harness.physics.thermo`` and ``Materials/cooling_physics.py``: parameters
fitted here are consumed by the cycle models there, so both sides must
evaluate the same Polanyi potential.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = REPO_ROOT / "data_cache" / "isodb" / "water_isotherms.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "data_cache" / "fits"

GAS_CONSTANT = 8.314  # J/(mol·K) — matches harness.physics.thermo (see docstring)

MIN_POINTS = 5  # usable points required for a fit
NRMSE_OK = 0.05  # fit nRMSE at or below this fraction of the uptake span ⇒ ok
HILL_WIN_RATIO = 0.70  # Hill alternative must beat D–A by this factor ...
HILL_WIN_NRMSE = 0.10  # ... and be decent itself, to diagnose S_shaped
E_BOUNDS_J_MOL = (200.0, 60000.0)
N_BOUNDS = (1.0, 6.0)

WATER_MOLAR_MASS_KG_MOL = 0.01801528
STP_CM3_PER_MOL = 22414.0  # 273.15 K, 101.325 kPa convention

# kg adsorbate / kg adsorbent per 1 native uptake unit — for WATER. Units
# absent from this table are not mass-normalizable without per-material
# constants (unit-cell mass, density, surface area) and keep native-unit
# fits only (E and n remain valid: uptake scaling does not touch them).
WATER_UPTAKE_TO_KG_KG = {
    "g/g": 1.0,
    "massfraction": 1.0,
    "wt%": 0.01,
    "mg/g": 1e-3,
    "mmol/g": WATER_MOLAR_MASS_KG_MOL * 1e-3 / 1e-3,
    "mol/g": WATER_MOLAR_MASS_KG_MOL / 1e-3,
    "cm3(STP)/g": WATER_MOLAR_MASS_KG_MOL / STP_CM3_PER_MOL * 1e3,
    "ml(STP)/g": WATER_MOLAR_MASS_KG_MOL / STP_CM3_PER_MOL * 1e3,
}
ASSUMED_LIQUID_DENSITY = {"ml/g": 1.0}  # treated as g/g, flagged in QC

# native pressure unit → Pa. The library contract is p_pa (pascals in);
# the batch layer converts using the record's ``pressure_units``.
PRESSURE_TO_PA = {
    "pa": 1.0,
    "kpa": 1e3,
    "mpa": 1e6,
    "bar": 1e5,
    "mbar": 100.0,
    "atm": 101325.0,
    "mmhg": 133.322,
    "torr": 133.322,
    "psi": 6894.76,
}


@dataclass
class DAFit:
    """Result of fitting one isotherm; diagnostics are first-class data."""

    q_sat: float | None  # native uptake units (None when no fit was possible)
    e_char_j_mol: float | None
    n_het: float | None
    rmse: float  # native uptake units (nan when no fit)
    nrmse: float | None  # rmse / uptake span
    n_points: int  # points used in the fit
    n_dropped: int  # excluded: non-finite, q < 0, or P ≥ P_sat
    flag: str  # ok | poor_fit | S_shaped | insufficient_points
    converged: bool
    psat_pa: float | None
    detail: str = ""  # human-readable reason for a non-ok flag

    @property
    def usable(self) -> bool:
        return self.flag == "ok"


# --------------------------------------------------------------------------
# library core
# --------------------------------------------------------------------------


def fit_isotherm_da(p_pa, q, T_K: float, psat_fn) -> DAFit:
    """Fit D–A to one single-temperature isotherm (see module docstring)."""
    p_raw = np.asarray(p_pa, dtype=float).ravel()
    q_raw = np.asarray(q, dtype=float).ravel()
    T_K = float(T_K)
    # ISODB rows can be ragged (pressure/uptake appended per point with
    # independent try/except in the exporter): align by truncating to the
    # shorter array; unpaired points count as dropped.
    n_present = max(len(p_raw), len(q_raw))
    n_total = min(len(p_raw), len(q_raw))
    p_raw, q_raw = p_raw[:n_total], q_raw[:n_total]

    finite = np.isfinite(p_raw) & np.isfinite(q_raw) & (p_raw > 0.0) & (q_raw >= 0.0)
    psat = float(psat_fn(T_K))
    usable = finite & (p_raw < psat)  # at/above saturation: outside the model
    p, q = p_raw[usable], q_raw[usable]
    n_dropped = n_present - len(p)

    if len(p) < MIN_POINTS or not np.any(q > 0.0):
        return DAFit(
            None, None, None, float("nan"), None, len(p), n_dropped,
            "insufficient_points", False, psat,
            f"{len(p)} usable points (< {MIN_POINTS}) or no positive uptake",
        )

    A = GAS_CONSTANT * T_K * np.log(psat / p)  # Polanyi potential, > 0
    span = float(np.max(q) - np.min(q))
    if span <= 0.0:
        span = max(float(np.max(q)), 1e-12)

    def residuals(x):
        q_sat, E, n = x
        return q_sat * np.exp(-((A / E) ** n)) - q

    q_sat0 = max(float(np.quantile(q, 0.95)), float(np.max(q)) * 0.999, 1e-9)
    n0, E0 = _seed_from_linear(A, q, q_sat0)
    lower = [1e-9, E_BOUNDS_J_MOL[0], N_BOUNDS[0]]
    upper = [q_sat0 * 10.0 + 1e-6, E_BOUNDS_J_MOL[1], N_BOUNDS[1]]

    # Multi-start: the D–A landscape in (E, n) is multi-modal for noisy or
    # partial data, and single-start trf routinely stalls in a bad basin
    # with a parameter pinned at a bound. Cheap insurance: the linear seed
    # plus a small spread of (n, E) starts; keep the lowest robust cost.
    seeds = [(n0, E0), (1.5, 3000.0), (2.0, 8000.0), (1.2, 15000.0), (3.0, 4000.0)]
    best = None
    for n_seed, e_seed in seeds:
        x0 = [
            min(max(q_sat0, lower[0]), upper[0]),
            min(max(e_seed, lower[1]), upper[1]),
            min(max(n_seed, lower[2]), upper[2]),
        ]
        try:
            result = least_squares(
                residuals, x0, bounds=(lower, upper), method="trf",
                loss="soft_l1", f_scale=max(1e-12, 0.02 * span),
            )
        except Exception:
            continue
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        return DAFit(
            None, None, None, float("nan"), None, len(p), n_dropped,
            "poor_fit", False, psat, "all least-squares starts failed",
        )

    q_sat_f, E_f, n_f = (float(v) for v in best.x)
    rmse = float(np.sqrt(np.mean(best.fun ** 2)))
    nrmse = rmse / span
    converged = bool(best.success)

    flag, detail = "ok", ""
    if not converged:
        flag, detail = "poor_fit", "least-squares did not converge"
    elif nrmse > NRMSE_OK:
        hill_nrmse = _hill_nrmse(A, q, span)
        if hill_nrmse is not None and hill_nrmse < HILL_WIN_RATIO * nrmse and hill_nrmse <= HILL_WIN_NRMSE:
            flag = "S_shaped"
            detail = (
                f"sigmoidal (Hill) alternative fits much better "
                f"(nRMSE {hill_nrmse:.3f} vs D–A {nrmse:.3f}) — Type-V shape"
            )
        else:
            at_bound = (
                abs(n_f - N_BOUNDS[0]) < 1e-6 or abs(n_f - N_BOUNDS[1]) < 1e-6
                or abs(E_f - E_BOUNDS_J_MOL[0]) < 1e-6
                or abs(E_f - E_BOUNDS_J_MOL[1]) < 1e-6
            )
            if at_bound:
                flag, detail = "poor_fit", "E or n pinned at a parameter bound"
            else:
                flag, detail = "poor_fit", f"nRMSE {nrmse:.3f} > {NRMSE_OK}"

    return DAFit(
        q_sat_f, E_f, n_f, rmse, nrmse, len(p), n_dropped,
        flag, converged, psat, detail,
    )


def _seed_from_linear(A: np.ndarray, q: np.ndarray, q_sat0: float) -> tuple[float, float]:
    """Initial (n, E) from a linear regression in Polanyi coordinates:
    ln(−ln(q/q_sat)) = n·lnA − n·lnE."""
    defaults = (1.8, 5000.0)
    band = (q > 0.02 * q_sat0) & (q < 0.98 * q_sat0) & (A > 0)
    if int(band.sum()) < 3:
        return defaults
    x = np.log(A[band])
    y = np.log(-np.log(q[band] / q_sat0))
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x) < 1e-9:
        return defaults
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # polyfit conditioning noise
            slope, intercept = np.polyfit(x, y, 1)
    except Exception:
        return defaults
    if not np.isfinite(slope) or slope <= 0.1 or not np.isfinite(intercept):
        return defaults
    n0 = float(np.clip(slope, 1.2, 5.0))
    log_e = float(np.clip(-intercept / slope, math.log(500.0), math.log(30000.0)))
    return n0, math.exp(log_e)


def _hill_nrmse(A: np.ndarray, q: np.ndarray, span: float) -> float | None:
    """nRMSE of the Hill (sigmoidal) alternative q = qh·A^m/(K^m + A^m) —
    the specific diagnosis for Type-V (S-shaped) data that D–A cannot bend
    to. None when the alternative cannot be fit."""
    mask = (A > 0) & (q > 0)
    if int(mask.sum()) < MIN_POINTS:
        return None
    A, q = A[mask], q[mask]
    order = np.argsort(A)
    A, q = A[order], q[order]
    qh0 = float(np.max(q))
    K0 = float(np.interp(0.5 * qh0, q, A))
    if not np.isfinite(K0) or K0 <= 0.0:
        K0 = float(np.median(A))

    def residuals(x):
        qh, K, m = x
        return qh * A ** m / (K ** m + A ** m) - q

    lower = [1e-9, 1e-9, 0.5]
    upper = [qh0 * 10.0 + 1e-6, max(float(np.max(A)) * 10.0, 1e-3), 8.0]
    x0 = [qh0, K0, 2.0]
    try:
        result = least_squares(
            residuals, np.clip(x0, lower, upper), bounds=(lower, upper), method="trf",
        )
    except Exception:
        return None
    if not result.success:
        return None
    return float(np.sqrt(np.mean(result.fun ** 2))) / span


def isosteric_heat(pairs, psat_fn, n_levels: int = 20, q_lo: float = 0.25, q_hi: float = 0.75) -> float | None:
    """Absolute isosteric heat [J/mol] from usable D–A fits (Clausius–
    Clapeyron). ``pairs`` is a sequence of (T_K, DAFit); only usable fits
    enter, and the lowest/highest temperatures form the evaluation pair.
    Each fitted curve is inverted at common uptake levels,
        A(q) = E·(−ln(q/q_sat))^{1/n},  lnP(T) = lnP_sat(T) − A/(R·T),
    and  Q_st = R·ΔlnP / Δ(1/T)  is averaged over the levels. Physically
    Q_st ≈ (molar latent heat) + (net binding energy) — always above the
    latent heat. None when fewer than two usable temperatures ≥ 5 K apart
    or an empty common uptake window."""
    usable = sorted(
        ((float(T), f) for T, f in pairs if f.usable), key=lambda item: item[0],
    )
    if len(usable) < 2:
        return None
    T_low, f_low = usable[0]
    T_high, f_high = usable[-1]
    if T_high - T_low < 5.0 or f_low.q_sat is None or f_high.q_sat is None:
        return None

    q_sat_min = min(f_low.q_sat, f_high.q_sat)
    levels = np.linspace(q_lo * q_sat_min, q_hi * q_sat_min, n_levels)
    if not (levels[-1] > levels[0] > 0.0):
        return None

    def ln_p(f: DAFit, T: float, q_level: float) -> float:
        A = f.e_char_j_mol * (-math.log(q_level / f.q_sat)) ** (1.0 / f.n_het)
        return math.log(float(psat_fn(T))) - A / (GAS_CONSTANT * T)

    raw = [
        GAS_CONSTANT * (ln_p(f_high, T_high, lvl) - ln_p(f_low, T_low, lvl))
        / (1.0 / T_low - 1.0 / T_high)
        for lvl in levels
    ]
    # Physical band for the isosteric heat of a vapor-phase adsorbate:
    # positive, and below a generous ceiling (200 kJ/mol ≈ 8 MJ/kg for
    # water). Level values outside it mean the two-temperature fits are
    # inconsistent (e.g. crossing fitted curves) — drop the levels; if too
    # few survive, refuse to report rather than emit a nonsense Q_st.
    values = [v for v in raw if np.isfinite(v) and 0.0 < v < 200e3]
    if len(values) < 3:
        return None
    return float(np.mean(values))


# --------------------------------------------------------------------------
# batch layer (pandas enters only here)
# --------------------------------------------------------------------------


def fit_all(rows, psat_fn, t_min: float = 280.0, t_max: float = 380.0,
            adsorbate_molar_mass_kg_mol: float | None = None,
            uptake_conversions: dict | None = None,
            pressure_conversions: dict | None = None,
            q_sat_bounds_kg_kg: tuple | None = None):
    """Fit many isotherm records → one DataFrame row per isotherm.

    ``rows``: dicts with keys ``pressure`` (array), ``uptake`` (array),
    ``temperature_K``, plus optional metadata (``adsorbent_hashkey``,
    ``adsorbent_name``, ``filename``, ``doi``, ``uptake_units``,
    ``pressure_units``, ``n_points``). Missing keys degrade gracefully.
    Pressures are converted to Pa via ``pressure_conversions`` (keyed by
    the record's ``pressure_units``; records with unknown units are not
    fitted).

    Records outside [t_min, t_max] K are annotated ``out_of_window`` and not
    fitted. ``q_sat_kg_kg`` is filled only for units in
    ``uptake_conversions`` (default: water table); when
    ``q_sat_bounds_kg_kg`` is given, ``q_sat_physical`` flags mass-normalized
    q_sat outside the band (mislabeled units in source records — the fit
    itself stays valid). ``q_st_j_kg`` is filled per adsorbent where ≥ 2
    usable temperatures exist and the adsorbate molar mass is given. The
    ``fit`` column carries the DAFit objects — drop it before serializing.
    """
    import pandas as pd

    conversions = dict(WATER_UPTAKE_TO_KG_KG) if uptake_conversions is None else uptake_conversions
    p_conv = dict(PRESSURE_TO_PA) if pressure_conversions is None else pressure_conversions

    records, fits = [], []
    for row in rows:
        T_K = row.get("temperature_K")
        in_window = T_K is not None and t_min <= float(T_K) <= t_max
        fit = None
        if in_window:
            p_factor = p_conv.get(str(row.get("pressure_units") or "").strip().lower())
            if p_factor is None:
                fit = DAFit(
                    None, None, None, float("nan"), None, 0, 0,
                    "poor_fit", False, None,
                    f"unknown pressure unit {row.get('pressure_units')!r}",
                )
            else:
                p_pa = [float(v) * p_factor for v in row["pressure"]]
                fit = fit_isotherm_da(p_pa, row["uptake"], float(T_K), psat_fn)
        fits.append(fit)
        records.append(_record(row, T_K, in_window, fit, conversions, q_sat_bounds_kg_kg))
    df = pd.DataFrame(records)
    df["fit"] = fits  # DAFit objects — drop before serializing
    df["fit_usable"] = [f is not None and f.usable for f in fits]

    qst_by_adsorbent: dict[str, float] = {}
    if adsorbate_molar_mass_kg_mol and len(df):
        for hk, group in df[df["fit_usable"]].groupby("material_id"):
            pairs = list(zip(group["temperature_K"].astype(float), group["fit"]))
            qst_mol = isosteric_heat(pairs, psat_fn)
            if qst_mol is not None:
                qst_by_adsorbent[hk] = qst_mol / adsorbate_molar_mass_kg_mol
    df["q_st_j_kg"] = df["material_id"].map(qst_by_adsorbent) if len(df) else None
    return df


_COLUMNS = (
    "material_id", "name", "source", "isotherm_id", "doi",
    "temperature_K", "temperature_c", "in_temp_window", "fitted",
    "q_sat_kg_kg", "mass_normalized", "q_sat_physical", "q_sat_native", "uptake_units",
    "q_st_j_kg", "e_char_j_mol", "n_da", "t_range_c",
    "fit_rmse", "rmse_native", "n_points", "n_points_input", "n_dropped",
    "fit_flag", "fit_detail", "psat_pa",
)


def _record(row: dict, T_K, in_window: bool, fit: DAFit | None, conversions: dict, q_sat_bounds=None) -> dict:
    units = str(row.get("uptake_units") or "").strip()
    if units in conversions:
        factor, basis = conversions[units], "conversion_table"
    elif units in ASSUMED_LIQUID_DENSITY:
        factor, basis = ASSUMED_LIQUID_DENSITY[units], "assumed_liquid_density"
    else:
        factor, basis = None, "not_mass_normalizable"

    T_c = float(T_K) - 273.15 if T_K is not None else None
    rec = {c: None for c in _COLUMNS}
    rec.update({
        "material_id": row.get("adsorbent_hashkey"),
        "name": row.get("adsorbent_name"),
        "source": "isodb",
        "isotherm_id": row.get("filename"),
        "doi": row.get("doi"),
        "temperature_K": float(T_K) if T_K is not None else None,
        "temperature_c": T_c,
        "in_temp_window": bool(in_window),
        "fitted": fit is not None,
        "uptake_units": units,
        "n_points_input": row.get("n_points"),
    })
    if fit is not None:
        q_sat_kg = fit.q_sat * factor if (factor is not None and fit.q_sat is not None) else None
        physical = None
        if q_sat_kg is not None and q_sat_bounds is not None:
            physical = bool(q_sat_bounds[0] <= q_sat_kg <= q_sat_bounds[1])
        rec.update({
            "q_sat_kg_kg": q_sat_kg,
            "mass_normalized": factor is not None,
            "q_sat_physical": physical,
            "q_sat_native": fit.q_sat,
            "e_char_j_mol": fit.e_char_j_mol,
            "n_da": fit.n_het,
            "t_range_c": f"{T_c:.0f}-{T_c:.0f}",
            "fit_rmse": fit.nrmse,
            "rmse_native": fit.rmse,
            "n_points": fit.n_points,
            "n_dropped": fit.n_dropped,
            "fit_flag": fit.flag,
            "fit_detail": fit.detail,
            "psat_pa": fit.psat_pa,
        })
    else:
        rec["fit_flag"] = "out_of_window"
        rec["n_points"] = None
    return rec


# --------------------------------------------------------------------------
# water-specific constants + thin CLI
# --------------------------------------------------------------------------


def water_psat_pa(t_k: float) -> float:
    """Saturation pressure of water [Pa] — same two-branch correlation as
    harness.physics.thermo / Materials/cooling_physics.py (Magnus below
    100 °C, NIST Antoine above), scalar form for the fitting loop."""
    t_c = t_k - 273.15
    if t_c < 100.0:
        return 611.2 * math.exp(17.502 * t_c / (240.97 + t_c))
    return (10.0 ** (8.10765 - 1750.286 / (235.0 + t_c))) * 133.322


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Dubinin–Astakhov parameters to ISODB water isotherms (§8.1 schema).")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE, help="Input isotherm table (.parquet or .csv) from nist_isodb.py.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for da_params.csv and the QC report.")
    parser.add_argument("--t-min", type=float, default=280.0, help="Fit window lower bound [K].")
    parser.add_argument("--t-max", type=float, default=380.0, help="Fit window upper bound [K].")
    parser.add_argument("--all-temps", action="store_true", help="Fit every temperature, ignoring the window.")
    return parser.parse_args()


def _qc_report(df, t_min: float, t_max: float) -> str:
    """Markdown QC report; every headline number the H1.0 gate needs."""
    lines = ["# D–A fit QC report (ISODB pure-water isotherms)", ""]
    fitted = df[df["fitted"]]
    usable = df[df["fit_flag"] == "ok"]
    lines += [
        f"- input isotherms: **{len(df)}**; fit window [{t_min:.0f}, {t_max:.0f}] K: "
        f"**{int(df['in_temp_window'].sum())}**; fitted: **{len(fitted)}**",
        f"- flags: " + ", ".join(
            f"`{flag}` **{count}**" for flag, count in df["fit_flag"].value_counts().items()
        ),
    ]

    lines += ["", "## Unit coverage (uptake units)", "",
              "| uptake_units | rows | mass-normalizable |", "|---|---:|---:|"]
    for units, group in df.groupby("uptake_units"):
        lines.append(f"| `{units or '(empty)'}` | {len(group)} | {int(group['mass_normalized'].fillna(False).sum())} |")

    ok_nrmse = usable["fit_rmse"].dropna()
    if len(ok_nrmse):
        lines += [
            "",
            "## Fit quality (usable fits)",
            "",
            f"- nRMSE (relative to uptake span): median **{ok_nrmse.median():.4f}**, "
            f"p25 **{ok_nrmse.quantile(0.25):.4f}**, p75 **{ok_nrmse.quantile(0.75):.4f}**",
            f"- E_char [J/mol]: median **{usable['e_char_j_mol'].median():.0f}**, "
            f"range [{usable['e_char_j_mol'].min():.0f}, {usable['e_char_j_mol'].max():.0f}]",
            f"- n (heterogeneity): median **{usable['n_da'].median():.2f}**, "
            f"range [{usable['n_da'].min():.2f}, {usable['n_da'].max():.2f}]",
        ]
    kg = usable[usable["q_sat_physical"] == True]["q_sat_kg_kg"].dropna()  # noqa: E712
    n_all_kg = usable[usable["mass_normalized"]]["q_sat_kg_kg"].dropna()
    if len(n_all_kg):
        excluded = int((usable["q_sat_physical"] == False).sum())  # noqa: E712
        lines.append(
            f"- q_sat [kg/kg] (physical band only): median **{kg.median():.3f}**, "
            f"range [{kg.min():.3f}, {kg.max():.3f}], n = {len(kg)}"
            + (f" — **{excluded}** rows excluded outside the physical band (mislabeled units)" if excluded else "")
        )

    with_qst = df[df["q_st_j_kg"].notna()]
    n_adsorbents = df["material_id"].nunique()
    lines += [
        "",
        "## Isosteric heat coverage",
        "",
        f"- adsorbents with q_st (≥ 2 usable temperatures ≥ 5 K apart): "
        f"**{with_qst['material_id'].nunique()} / {n_adsorbents}**",
    ]
    if len(with_qst):
        qst = with_qst.groupby("material_id")["q_st_j_kg"].first()
        lines.append(
            f"- q_st [J/kg]: median **{qst.median():.3e}**, "
            f"range [{qst.min():.3e}, {qst.max():.3e}] (sanity: must exceed water latent heat ≈ 2.4e6)"
        )

    non_ok = df[(df["fitted"]) & (df["fit_flag"] != "ok")]
    if len(non_ok):
        lines += ["", "## Worst / flagged fits", "", "| name | isotherm_id | flag | nRMSE | detail |", "|---|---|---|---:|---|"]
        for _, r in non_ok.sort_values("fit_rmse", ascending=False).head(15).iterrows():
            lines.append(f"| {str(r['name'])[:40]} | {r['isotherm_id']} | `{r['fit_flag']}` | "
                         f"{r['fit_rmse'] if r['fit_rmse'] == r['fit_rmse'] else float('nan'):.3f} | {str(r['fit_detail'])[:60]} |")

    mil = df[df["name"].astype(str).str.contains("MIL-101", case=False, na=False)]
    lines += ["", "## MIL-101 spot check (known S-shaped / Type-V material)", ""]
    if len(mil):
        for _, r in mil.iterrows():
            lines.append(f"- {r['name']} @ {r['temperature_K']:.0f} K: `{r['fit_flag']}` — {r['fit_detail'] or 'n/a'}")
    else:
        lines.append("- no MIL-101 isotherms in the water subset")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    import pandas as pd

    if str(args.table).endswith(".parquet"):
        df_in = pd.read_parquet(args.table)
    else:
        df_in = pd.read_csv(args.table)
    t_min, t_max = (0.0, 1e9) if args.all_temps else (args.t_min, args.t_max)

    result = fit_all(
        df_in.to_dict("records"), water_psat_pa, t_min, t_max,
        adsorbate_molar_mass_kg_mol=WATER_MOLAR_MASS_KG_MOL,
        q_sat_bounds_kg_kg=(1e-4, 2.5),  # water: max real capacity ≈ 1.4 kg/kg
    )
    out = result.drop(columns=["fit", "fit_usable"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "da_params.csv"
    out.to_csv(csv_path, index=False)
    qc_path = args.out_dir / "da_params_qc.md"
    qc_path.write_text(_qc_report(result, t_min, t_max), encoding="utf-8")

    usable = result[result["fit_flag"] == "ok"]
    print(f"input isotherms      : {len(result)}")
    print(f"fitted / usable      : {int(result['fitted'].sum())} / {len(usable)}")
    print(f"flag distribution    : {dict(result['fit_flag'].value_counts())}")
    if len(usable):
        print(f"median nRMSE (usable): {usable['fit_rmse'].median():.4f}")
    print(f"adsorbents with q_st : {result[result['q_st_j_kg'].notna()]['material_id'].nunique()} "
          f"/ {result['material_id'].nunique()}")
    print(f"wrote {csv_path}")
    print(f"wrote {qc_path}")


if __name__ == "__main__":
    main()
