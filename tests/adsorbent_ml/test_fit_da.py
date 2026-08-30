"""Unit tests for the D–A fitting library (H1.0 reusability contract,
harness/DESIGN.md §12): synthetic noisy D–A curves recover their true
parameters, Type-V curves are diagnosed as S_shaped (not silently
mis-fit), Clausius–Clapeyron Q_st matches its analytic value, and
degenerate inputs fail loudly via flags."""

import math

import numpy as np
import pandas as pd
import pytest

from fit_da import (
    GAS_CONSTANT,
    WATER_MOLAR_MASS_KG_MOL,
    DAFit,
    fit_all,
    fit_isotherm_da,
    isosteric_heat,
)

PSAT_298 = 3169.0  # Pa, water at 25 °C
NRMSE_OK_THRESHOLD = 0.05


def psat_298(_t_k):
    return PSAT_298


def magnus(t_k):
    """Water Psat below 100 °C (same correlation as harness.physics.thermo)."""
    t_c = t_k - 273.15
    return 611.2 * math.exp(17.502 * t_c / (240.97 + t_c))


def da_curve(q_sat, e_char, n_het, t_k, psat_fn, p):
    a = GAS_CONSTANT * t_k * np.log(psat_fn(t_k) / np.asarray(p, dtype=float))
    return q_sat * np.exp(-((a / e_char) ** n_het))


def test_synthetic_da_recovery():
    rng = np.random.default_rng(0)
    t = 298.0
    p = np.geomspace(0.05 * PSAT_298, 0.99 * PSAT_298, 30)
    q = da_curve(0.35, 4500.0, 1.8, t, psat_298, p) * (1.0 + rng.normal(0.0, 0.005, p.size))

    fit = fit_isotherm_da(p, q, t, psat_298)

    assert isinstance(fit, DAFit)
    assert fit.flag == "ok" and fit.converged
    assert fit.n_dropped == 0 and fit.n_points == 30
    assert fit.q_sat == pytest.approx(0.35, rel=0.03)
    assert fit.e_char_j_mol == pytest.approx(4500.0, rel=0.05)
    assert fit.n_het == pytest.approx(1.8, rel=0.08)
    assert fit.nrmse < NRMSE_OK_THRESHOLD


def test_s_shaped_isotherm_is_diagnosed_not_misfit():
    """Type-V (sigmoidal in Polanyi coordinates) data: D–A must not be
    reported as `ok`; the Hill alternative wins ⇒ flag S_shaped."""
    t = 298.0
    p = np.geomspace(1e-3 * PSAT_298, 0.99 * PSAT_298, 40)
    a = GAS_CONSTANT * t * np.log(PSAT_298 / p)
    q = 0.6 * a**4 / (2500.0**4 + a**4)  # step at A ≈ 2500 J/mol

    fit = fit_isotherm_da(p, q, t, psat_298)

    assert fit.flag == "S_shaped"
    assert "Hill" in fit.detail


def test_too_few_points_is_flagged():
    fit = fit_isotherm_da([100.0, 200.0, 300.0], [0.01, 0.05, 0.1], 298.0, psat_298)
    assert fit.flag == "insufficient_points"
    assert fit.q_sat is None and not fit.converged


def test_points_at_or_above_psat_are_dropped():
    t = 298.0
    p = np.array([50.0, 200.0, 500.0, 1000.0, 2000.0, PSAT_298, 1.2 * PSAT_298])
    q = da_curve(0.35, 4500.0, 1.8, t, psat_298, p)

    fit = fit_isotherm_da(p, q, t, psat_298)

    assert fit.n_dropped == 2
    assert fit.n_points == 5
    assert fit.psat_pa == pytest.approx(PSAT_298)


def test_degenerate_zero_uptake_is_flagged():
    p = np.geomspace(0.05 * PSAT_298, 0.9 * PSAT_298, 20)
    fit = fit_isotherm_da(p, np.zeros(20), 298.0, psat_298)
    assert fit.flag == "insufficient_points"


def test_nan_and_negative_inputs_dropped():
    p = np.array([100.0, 200.0, np.nan, 500.0, 1000.0, 2000.0, 3000.0])
    q = np.array([0.02, 0.05, 0.1, -1.0, 0.2, 0.25, 0.28])
    fit = fit_isotherm_da(p, q, 298.0, psat_298)
    assert fit.n_dropped == 2  # NaN pressure and negative uptake
    assert fit.n_points == 5
    assert fit.flag in ("ok", "poor_fit")  # a fit is attempted on 5 points


def test_ragged_export_rows_are_aligned():
    """ISODB export rows can have pressure/uptake of different lengths (the
    exporter appends per point with independent try/except). The fitter
    must truncate to the overlap, not crash."""
    p = list(np.geomspace(0.05 * PSAT_298, 0.99 * PSAT_298, 30))
    q = list(da_curve(0.35, 4500.0, 1.8, 298.0, psat_298, np.array(p)))[:22]  # 8 unpaired
    fit = fit_isotherm_da(p, q, 298.0, psat_298)
    assert fit.n_points == 22
    assert fit.n_dropped == 8
    assert fit.flag == "ok"
    assert fit.q_sat == pytest.approx(0.35, rel=0.05)

    empty = fit_isotherm_da([], [], 298.0, psat_298)
    assert empty.flag == "insufficient_points" and empty.n_dropped == 0


def test_isosteric_heat_matches_analytic_value():
    """Two isotherms from the *same* D–A parameters: Q_st must equal
    R·ΔlnP_sat/Δ(1/T) + A(q) — latent heat plus binding energy."""
    q_sat, e_char, n_het = 0.35, 5000.0, 1.8
    t_low, t_high = 298.0, 348.0
    pairs = []
    for t in (t_low, t_high):
        p = np.geomspace(1e-4 * magnus(t), 0.99 * magnus(t), 30)
        pairs.append((t, fit_isotherm_da(p, da_curve(q_sat, e_char, n_het, t, magnus, p), t, magnus)))

    qst = isosteric_heat(pairs, magnus)

    assert qst is not None
    # expected value: average over the same uptake levels the function uses
    levels = np.linspace(0.25 * q_sat, 0.75 * q_sat, 20)
    dw = 1.0 / t_low - 1.0 / t_high
    psat_term = GAS_CONSTANT * (math.log(magnus(t_high)) - math.log(magnus(t_low))) / dw
    binding = [e_char * (-math.log(lvl / q_sat)) ** (1.0 / n_het) for lvl in levels]
    expected = psat_term + float(np.mean(binding))
    assert qst == pytest.approx(expected, rel=0.02)
    assert qst > psat_term  # binding always adds heat beyond the latent heat


def test_isosteric_heat_jkg_is_physical_for_water():
    """For water, Q_st ≈ h_fg·M + A ⇒ ~2–4 MJ/kg for plausible params."""
    q_sat, e_char, n_het = 0.35, 5000.0, 1.8
    pairs = []
    for t in (298.0, 348.0):
        p = np.geomspace(1e-4 * magnus(t), 0.99 * magnus(t), 30)
        pairs.append((t, fit_isotherm_da(p, da_curve(q_sat, e_char, n_het, t, magnus, p), t, magnus)))
    qst_kg = isosteric_heat(pairs, magnus) / WATER_MOLAR_MASS_KG_MOL
    assert 2.0e6 < qst_kg < 4.0e6


def test_isosteric_heat_needs_two_usable_temperatures():
    p = np.geomspace(0.05 * PSAT_298, 0.9 * PSAT_298, 15)
    fit = fit_isotherm_da(p, da_curve(0.35, 4500.0, 1.8, 298.0, psat_298, p), 298.0, psat_298)
    poor = DAFit(None, None, None, 0.0, None, 5, 0, "poor_fit", False, PSAT_298)

    assert isosteric_heat([(298.0, fit)], psat_298) is None  # single temperature
    assert isosteric_heat([(298.0, fit), (299.0, fit)], psat_298) is None  # < 5 K apart
    assert isosteric_heat([(298.0, fit), (348.0, poor)], psat_298) is None  # second fit unusable


def test_isosteric_heat_refuses_inconsistent_fits():
    """A high-temperature fit with an absurdly large E implies decreasing
    pressure with rising temperature at fixed uptake ⇒ negative Q_st levels
    ⇒ the guard must refuse (return None), not emit nonsense."""
    t_low, t_high = 298.0, 348.0
    p_low = np.geomspace(0.05 * magnus(t_low), 0.9 * magnus(t_low), 20)
    f_good = fit_isotherm_da(p_low, da_curve(0.35, 5000.0, 1.8, t_low, magnus, p_low), t_low, magnus)
    f_bad = DAFit(0.35, 20000.0, 1.8, 0.01, 0.01, 20, 0, "ok", True, magnus(t_high))

    assert isosteric_heat([(t_low, f_good), (t_high, f_bad)], magnus) is None


def test_fit_all_batch_and_schema():
    """Batch layer: one row per isotherm, §8.1 columns present, unit
    conversion applied, q_st filled per adsorbent only with multi-T
    coverage."""

    def make_row(hk, name, fname, t_k, units, factor):
        p = np.geomspace(0.05 * magnus(t_k), 0.99 * magnus(t_k), 20)
        q_native = da_curve(0.35, 4500.0, 1.8, t_k, magnus, p)
        return {
            "adsorbent_hashkey": hk, "adsorbent_name": name, "filename": fname,
            "doi": "10.0000/test", "temperature_K": t_k, "uptake_units": units,
            "pressure_units": "pa", "n_points": len(p), "pressure": list(p),
            "uptake": list(q_native / factor),
        }

    rows = [
        make_row("hkA", "Adsorbent A", "isoA1", 298.0, "mmol/g", 0.01801528),
        make_row("hkA", "Adsorbent A", "isoA2", 348.0, "mmol/g", 0.01801528),
        make_row("hkB", "Adsorbent B", "isoB1", 298.0, "g/g", 1.0),
        make_row("hkC", "Adsorbent C", "isoC1", 200.0, "mmol/g", 0.01801528),  # out of window
    ]
    rows.append(make_row("hkD", "Adsorbent D", "isoD1", 298.0, "g/g", 1.0e-3))  # mislabeled: q_sat ≈ 350
    df = fit_all(
        rows, magnus, t_min=280.0, t_max=380.0,
        adsorbate_molar_mass_kg_mol=WATER_MOLAR_MASS_KG_MOL,
        q_sat_bounds_kg_kg=(1e-4, 2.5),
    )

    assert len(df) == 5
    for col in ("material_id", "name", "source", "isotherm_id", "q_sat_kg_kg",
                "q_st_j_kg", "e_char_j_mol", "n_da", "fit_flag", "t_range_c"):
        assert col in df.columns

    a_rows = df[df["material_id"] == "hkA"].sort_values("temperature_K")
    assert (a_rows["fit_flag"] == "ok").all()
    assert a_rows["q_sat_kg_kg"].iloc[0] == pytest.approx(0.35, rel=0.03)  # mmol/g → kg/kg
    assert a_rows["q_st_j_kg"].notna().all()  # two temperatures ⇒ q_st filled

    b_row = df[df["material_id"] == "hkB"].iloc[0]
    assert b_row["fit_flag"] == "ok"
    assert b_row["mass_normalized"]
    assert pd.isna(b_row["q_st_j_kg"])  # single temperature ⇒ no q_st

    c_row = df[df["material_id"] == "hkC"].iloc[0]
    assert not c_row["fitted"] and c_row["fit_flag"] == "out_of_window"

    d_row = df[df["material_id"] == "hkD"].iloc[0]
    assert d_row["fit_flag"] == "ok"
    assert d_row["q_sat_kg_kg"] > 2.5 and not d_row["q_sat_physical"]
    assert df[df["material_id"] == "hkA"].iloc[0]["q_sat_physical"]


def test_library_importable_without_cli(capsys):
    """Reusability gate: importing the module and fitting must not require
    or trigger the CLI path (no argparse run, no file I/O)."""
    import fit_da

    p = np.geomspace(0.05 * PSAT_298, 0.99 * PSAT_298, 20)
    fit = fit_da.fit_isotherm_da(p, da_curve(0.35, 4500.0, 1.8, 298.0, psat_298, p), 298.0, psat_298)
    assert fit.flag == "ok"
    assert capsys.readouterr().out == ""
