"""Tests for levers 1+2: Ongari CSD evidence, DOI bridge, QMOF join,
fuzzy rung, pore-consistency gate, QMOF-aware descriptors."""

import numpy as np
import pandas as pd

from match_structures import (match_materials, pore_consistency,
                              refcode_of_filename)
from ongari import load_ongari_map
from qmof import load_qmof_table, refcode_of


def test_refcode_extraction():
    assert refcode_of_filename("ACECIV_ion_b") == "ACECIV"
    assert refcode_of_filename("RUBTAK03_auto") == "RUBTAK03"
    assert refcode_of_filename("acs.cgd.6b01265_1499490_clean") is None
    assert refcode_of("ABACUF01_FSR") == "ABACUF01"
    assert refcode_of("not-a-refcode") is None


def test_ongari_vendored_map():
    om = load_ongari_map()
    assert len(om) > 1000
    assert set(om.columns) == {"adsorbent_name", "refcode", "source"}
    cubtc = set(om[om["adsorbent_name"] == "CuBTC"]["refcode"])
    assert "FIQCEN" in cubtc  # the paper's desolvated-CuBTC reference


def test_qmof_table_shape():
    qt = load_qmof_table()
    assert len(qt) > 10000 and qt["refcode"].is_unique
    assert qt["formula_dict"].notna().mean() > 0.5
    assert (qt["topology"] != "").mean() > 0.2  # MOFid topology is sparse upstream
    assert qt["n_variants"].min() >= 1


def _toy_core(tmp_path):
    core = pd.DataFrame({
        "filename": ["FIQCEN_clean", "RUBTAK03_auto", "OTHER_clean"],
        "LCD": [10.0, 8.0, 6.0], "PLD": [6.0, 5.0, 4.0],
        "ASA_m2_g": [1500.0, 1200.0, 800.0], "AV_cm3_g": [0.8, 0.5, 0.4],
        "Has_OMS": [True, False, False], "DOI_public": [None, None, None]})
    p = tmp_path / "core.parquet"
    core.to_parquet(p, index=False)
    return p


def test_csd_doi_fuzzy_precedence(tmp_path):
    cp = _toy_core(tmp_path)
    om = pd.DataFrame([
        {"adsorbent_name": "CuBTC", "refcode": "FIQCEN", "source": "step-09.yml"},
        {"adsorbent_name": "UiO-66-(COOH)2", "refcode": "RUBTAK03",
         "source": "step-09.yml"},
    ])
    qt = pd.DataFrame([{"refcode": "FIQCEN", "formula_dict": {"Cu": 1.0},
                        "formula_str": "Cu", "topology": "tbo",
                        "dois": ["10.0000/x"], "pld": 6.0, "lcd": 10.0,
                        "density": 1.0, "n_variants": 1}])
    names = pd.DataFrame({"name": ["CuBTC", "UiO-66-COOH", "Zeolite 13X"],
                          "material_id": ["m1", "m2", "m3"]})
    dmap = {"m3": {"10.0000/x"}}  # shares a DOI with the FIQCEN QMOF row
    matched, rep = match_materials(names, core_props=cp, ongari_map=om,
                                   qmof_table=qt, material_dois=dmap)
    by = matched.set_index("name")
    assert by.loc["CuBTC", "match_kind"] == "csd"  # exact beats everything
    assert by.loc["CuBTC", "core_exemplar"] == "FIQCEN_clean"
    assert by.loc["CuBTC", "qmof_topology"] == "tbo"
    assert by.loc["UiO-66-COOH", "match_kind"] == "csd-fuzzy"  # close, not exact
    assert by.loc["UiO-66-COOH", "ambiguous"] == True
    assert by.loc["UiO-66-COOH", "fuzzy_score"] >= 0.85
    assert by.loc["Zeolite 13X", "match_kind"] == "doi"  # DOI bridge fallback
    assert by.loc["Zeolite 13X", "refcode"] == "FIQCEN"


def test_pore_consistency_gate(tmp_path):
    cp = _toy_core(tmp_path)
    matched = pd.DataFrame([
        {"name": "A", "core_exemplar": "FIQCEN_clean"},   # AV 0.8, q .35 → ok
        {"name": "B", "core_exemplar": "OTHER_clean"},    # AV 0.4, q .9 → bad
        {"name": "C", "core_exemplar": ""}])             # no evidence
    labels = pd.DataFrame({"name": ["A", "B", "C"],
                           "q_sat_kg_kg": [0.35, 0.90, 0.20]})
    out = pore_consistency(matched, labels, core_props=cp)
    assert out.loc[0, "pore_consistent"] is True
    assert out.loc[1, "pore_consistent"] is False
    assert out.loc[2, "pore_consistent"] is None


def test_descriptors_qmof_precedence_and_fallback(tmp_path):
    from descriptors import build_feature_matrix  # noqa: PLC0415

    cp = _toy_core(tmp_path)
    labels = pd.DataFrame([
        {"name": "M1", "family": "mof"},   # qmof formula + qmof pores
        {"name": "M2", "family": "mof"}])  # class formula only
    matched = pd.DataFrame([
        {"name": "M1", "match_kind": "csd", "core_exemplar": "",
         "ambiguous": False, "qmof_formula": "ZnC8H10N4",
         "qmof_topology": "sod", "qmof_density": 1.2,
         "qmof_pld": 3.0, "qmof_lcd": 9.0},
        {"name": "M2", "match_kind": "none", "core_exemplar": "",
         "ambiguous": False, "qmof_formula": "", "qmof_topology": "",
         "qmof_density": float("nan"), "qmof_pld": float("nan"),
         "qmof_lcd": float("nan")}])
    X, names, cov = build_feature_matrix(labels, matched, core_props=cp)
    assert X.loc[0, "pore_has_data"] == 1 and X.loc[0, "pore_qmof"] == 1
    assert X.loc[0, "pore_PLD"] == 3.0 and X.loc[0, "has_formula"] == 1
    assert X.loc[0, "topo_sod"] == 1 and X.loc[0, "has_topology"] == 1
    assert cov["n_with_qmof_formula"] == 1
    # Inference stability: unseen topology → same columns, topo_other bucket
    labels2 = pd.DataFrame([{"name": "M3", "family": "mof"}])
    matched2 = pd.DataFrame([{"name": "M3", "match_kind": "none",
                              "core_exemplar": "", "ambiguous": False,
                              "qmof_formula": "", "qmof_topology": "zzz",
                              "qmof_density": float("nan"),
                              "qmof_pld": float("nan"),
                              "qmof_lcd": float("nan")}])
    X2, names2, _ = build_feature_matrix(
        labels2, matched2, core_props=cp,
        topologies=[t[5:] for t in names if t.startswith("topo_") and t != "topo_other"])
    assert list(X2.columns) == list(X.columns)
    assert X2.loc[0, "topo_other"] == 1
