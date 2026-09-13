"""Tests for composition parsing + feature matrix builder."""

import numpy as np
import pandas as pd
import pytest

from composition import (COMPOSITION_COLUMNS, featurize_composition,
                         formula_for, looks_like_formula, parse_formula)


def test_parser_spots():
    assert parse_formula("H2O") == {"H": 2.0, "O": 1.0}
    assert parse_formula("C115.5H202N14O43Zn4")["C"] == pytest.approx(115.5)
    assert parse_formula("Na12Al12Si12O48")["Si"] == pytest.approx(12.0)
    with pytest.raises(ValueError, match="parens"):
        parse_formula("{[Zn4O(bfbpdc)3]}n")
    with pytest.raises(ValueError, match="not in property table"):
        parse_formula("M2C8H2O6")
    assert looks_like_formula("C24H39N4O60Ni2PW12")
    assert not looks_like_formula("CuBTC")


def test_featurize_finite_and_sensible():
    f = featurize_composition(parse_formula("H2O"))
    assert set(f) == set(COMPOSITION_COLUMNS)
    assert all(np.isfinite(v) for v in f.values())
    assert f["n_elements"] == 2.0 and f["metal_frac"] == pytest.approx(0.0)
    assert f["mean_en"] == pytest.approx((2 * 2.20 + 3.44) / 3, rel=1e-6)
    g = featurize_composition(parse_formula("NaCl"))
    assert g["metal_frac"] == pytest.approx(0.5)


def test_formula_for_sources():
    assert formula_for("C24H39N4O60Ni2PW12", "other") == ("C24H39N4O60Ni2PW12", "parsed")
    assert formula_for("Zeolite 13X", "zeolite")[1] == "class"
    assert formula_for("Activated carbon: BPL", "carbon") == ("C", "class")
    assert formula_for("DMOF-OH", "mof") == (None, "none")


def test_build_matrix_shapes_and_missingness(tmp_path):
    from descriptors import build_feature_matrix  # noqa: PLC0415

    core = pd.DataFrame({
        "filename": ["FOO_ZIF8_BAR_clean"],
        "LCD": [10.0], "PLD": [5.0], "ASA_m2_g": [1500.0], "AV_cm3_g": [0.6],
        "Has_OMS": [True]})
    cp = tmp_path / "core.parquet"
    core.to_parquet(cp, index=False)
    labels = pd.DataFrame([
        {"name": "ZIF-8", "family": "mof"},
        {"name": "Zeolite 13X", "family": "zeolite"},
        {"name": "Mystery stuff", "family": "other"}])
    matched = pd.DataFrame([
        {"name": "ZIF-8", "match_kind": "core", "core_exemplar": "FOO_ZIF8_BAR_clean",
         "ambiguous": False},
        {"name": "Zeolite 13X", "match_kind": "iza", "core_exemplar": "",
         "ambiguous": False},
        {"name": "Mystery stuff", "match_kind": "none", "core_exemplar": "",
         "ambiguous": False}])
    X, names, cov = build_feature_matrix(labels, matched, core_props=cp)
    assert list(X.columns[1:]) == names and len(X) == 3
    assert X.loc[0, "pore_LCD"] == pytest.approx(10.0)
    assert X.loc[0, "pore_has_data"] == 1 and X.loc[1, "pore_has_data"] == 0
    assert np.isnan(X.loc[1, "pore_LCD"])  # IZA: no pore data in v1, honestly NaN
    assert X.loc[0, "fam_mof"] == 1 and X.loc[1, "fam_zeolite"] == 1
    assert X.loc[0, "has_formula"] == 1 and X.loc[2, "has_formula"] == 0
    assert cov["n_with_pores"] == 1 and cov["n_with_formula"] == 2
