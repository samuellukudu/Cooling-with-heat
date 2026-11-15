"""OPTIMADE formula-rung tests — the matching round (ROADMAP item 3).

Synthetic candidate index (no corpora, no HTTP): the rung must match
formula-named adsorbents by composition cosine with exact metal sets, flag
ties ambiguous (cosine is scale-invariant — C9...Cd and C18...Cd2 are the
same composition to this metric), never outrank stronger rungs, and count
itself in the report.
"""

import numpy as np
import pandas as pd
import pytest

from match_structures import (match_materials, match_optimade_formula,
                              match_qmoftab_formula)


def _idx_cd_only():
    return [{"db": "pyrene-mofs", "id": "abc-1",
             "formula": "C9H11ClN2O4Cd",
             "counts": {"C": 9.0, "H": 11.0, "Cl": 1.0, "N": 2.0,
                        "O": 4.0, "Cd": 1.0},
             "cif": "/caches/pyrene-mofs/cifs/abc-1.cif"}]


def _idx_cd_scaled_tie():
    # 2× the counts: composition cosine = 1.0 → a tie with abc-1
    return _idx_cd_only() + [
        {"db": "curated-cofs", "id": "abc-2",
         "formula": "C18H22Cl2N4O8Cd2",
         "counts": {"C": 18.0, "H": 22.0, "Cl": 2.0, "N": 4.0,
                    "O": 8.0, "Cd": 2.0},
         "cif": "/caches/curated-cofs/cifs/abc-2.cif"}]


def _names(*rows):
    return pd.DataFrame(rows, columns=["name", "material_id", "family"])


def test_formula_named_adsorbent_matches_uniquely():
    hit = match_optimade_formula("C9H11ClN2O4Cd", "mof", _idx_cd_only())
    assert hit is not None
    assert hit["optimade_id"] == "abc-1"
    assert hit["optimade_score"] == pytest.approx(1.0, abs=1e-6)
    assert hit["ambiguous"] is False  # unique, near-exact composition


def test_scaled_composition_tie_is_ambiguous():
    hit = match_optimade_formula("C9H11ClN2O4Cd", "mof", _idx_cd_scaled_tie())
    assert hit is not None
    assert hit["ambiguous"] is True  # cosine cannot distinguish scale


def test_wrong_metal_set_scores_zero():
    zinc_index = [{"db": "x", "id": "zn-1", "formula": "C9H11N2O4Zn",
                   "counts": {"C": 9.0, "H": 11.0, "N": 2.0, "O": 4.0,
                              "Zn": 1.0}, "cif": "/x/zn-1.cif"}]
    assert match_optimade_formula("C9H11ClN2O4Cd", "mof", zinc_index) is None


def test_low_similarity_carbon_stays_none():
    # an O-containing COF candidate vs a bare-carbon target: cosine < 0.90
    idx = [{"db": "curated-cofs", "id": "cof-7", "formula": "C40H20O8",
            "counts": {"C": 40.0, "H": 20.0, "O": 8.0}, "cif": "/x/cof-7.cif"}]
    assert match_optimade_formula("Sorbonorit 4", "carbon", idx) is None


def test_core_rung_beats_optimade(tmp_path):
    core = pd.DataFrame({
        "filename": ["MIL100_clean"], "LCD": [10.0], "PLD": [6.0],
        "ASA_m2_g": [1500.0], "AV_cm3_g": [0.8], "Has_OMS": [False],
        "DOI_public": [None]})
    cp = tmp_path / "core.parquet"
    core.to_parquet(cp, index=False)
    names = _names({"name": "MIL-100", "material_id": "m1", "family": "mof"},
                   {"name": "C9H11ClN2O4Cd", "material_id": "m2",
                    "family": "mof"})
    matched, rep = match_materials(names, core_props=cp,
                                   optimade_index=_idx_cd_only())
    by = matched.set_index("name")
    assert by.loc["MIL-100", "match_kind"] == "core"  # stronger rung wins
    assert by.loc["C9H11ClN2O4Cd", "match_kind"] == "optimade-formula"
    assert by.loc["C9H11ClN2O4Cd", "optimade_db"] == "pyrene-mofs"
    assert by.loc["C9H11ClN2O4Cd", "optimade_cif"].endswith("abc-1.cif")
    assert rep["by_kind"].get("optimade-formula") == 1
    assert rep["n_with_optimade"] == 1


def test_no_index_keeps_v1_behavior(tmp_path):
    core = pd.DataFrame({
        "filename": ["MIL100_clean"], "LCD": [10.0], "PLD": [6.0],
        "ASA_m2_g": [1500.0], "AV_cm3_g": [0.8], "Has_OMS": [False],
        "DOI_public": [None]})
    cp = tmp_path / "core.parquet"
    core.to_parquet(cp, index=False)
    names = _names({"name": "C9H11ClN2O4Cd", "material_id": "m2",
                    "family": "mof"})
    matched, _ = match_materials(names, core_props=cp, optimade_index=None)
    assert matched.set_index("name").loc["C9H11ClN2O4Cd", "match_kind"] == "none"


# --- round 2: QMOF-table rung + class-table normalization -------------------

def _idx_qmof():
    return [{"db": "qmoftab", "id": "qmof-a9be673", "ref": "DOYBEA_FSR",
             "formula": "Al4C16H12O20",
             "counts": {"Al": 4.0, "C": 16.0, "H": 12.0, "O": 20.0},
             "cif": None}]


def test_qmoftab_rung_matches_literature_composition():
    # "Aluminum fumarate" -> AlC4H3O5 (verified Senkovska/CAU-10 family)
    hit = match_qmoftab_formula("Aluminum fumarate", "mof", _idx_qmof())
    assert hit is not None
    assert hit["qmoftab_id"] == "qmof-a9be673"
    assert hit["qmoftab_score"] >= 0.90
    assert hit["ambiguous"] is False  # 1.0 cosine, single candidate


def test_qmoftab_rung_runs_after_optimade(tmp_path):
    core = pd.DataFrame({
        "filename": ["MIL100_clean"], "LCD": [10.0], "PLD": [6.0],
        "ASA_m2_g": [1500.0], "AV_cm3_g": [0.8], "Has_OMS": [False],
        "DOI_public": [None]})
    cp = tmp_path / "core.parquet"
    core.to_parquet(cp, index=False)
    names = _names({"name": "Aluminum fumarate", "material_id": "m1",
                    "family": "mof"})
    matched, rep = match_materials(names, core_props=cp,
                                   optimade_index=_idx_cd_only(),
                                   qmof_index=_idx_qmof())
    by = matched.set_index("name")
    # OPTIMADE candidate (Cd-MOF, metals {Cd} != {Al}) cannot match; QMOF does
    assert by.loc["Aluminum fumarate", "match_kind"] == "qmoftab-formula"
    assert by.loc["Aluminum fumarate", "qmoftab_ref"] == "DOYBEA_FSR"
    assert rep["n_with_qmoftab"] == 1


def test_class_table_paren_normalization():
    from composition import formula_for

    # round-1 miss: "Cr-MIL(101)" never contained the "mil-101" key raw
    f, src = formula_for("Cr-MIL(101)", "mof")
    assert src == "class" and f == "Cr3C24H16O15F"


def test_class_table_new_keys_resolve():
    from composition import formula_for

    cases = {"DUT-4": "AlC12H7O5", "Zr-MOF-802": "Zr6C30H16N12O28",
             "Zr-MOF-805": "Zr6C72H40O40", "Zr-MOF-806": "Zr6C84H52O40",
             "DMOF-Cl2": "Zn2C22H20N2O8", "CAU-1-NH2": "Al4C28H32N3O18"}
    for nm, want in cases.items():
        f, src = formula_for(nm, "mof")
        assert (f, src) == (want, "class"), nm


def test_class_table_cau10_precedence_over_cau1():
    from composition import formula_for

    f, _ = formula_for("CAU-10-H", "mof")
    assert f == "AlC8H5O5"  # the longer pre-existing key must win


def test_composition_rungs_compete_on_score(tmp_path):
    core = pd.DataFrame({
        "filename": ["MIL100_clean"], "LCD": [10.0], "PLD": [6.0],
        "ASA_m2_g": [1500.0], "AV_cm3_g": [0.8], "Has_OMS": [False],
        "DOI_public": [None]})
    cp = tmp_path / "core.parquet"
    core.to_parquet(cp, index=False)
    # weak OPTIMADE candidate (0.9849-like Al hit) vs exact QMOF candidate:
    # the higher composition score must win regardless of rung order
    optimade = [{"db": "pyrene-mofs", "id": "al-weak", "ref": "",
                 "formula": "C176H96Al8O40",
                 "counts": {"Al": 8.0, "C": 176.0, "H": 96.0, "O": 40.0},
                 "cif": "/x/al-weak.cif"}]
    qmof = [{"db": "qmoftab", "id": "qmof-exact", "ref": "DUT4_like",
             "formula": "Al4C48H28O20",
             "counts": {"Al": 4.0, "C": 48.0, "H": 28.0, "O": 20.0},
             "cif": None}]
    names = _names({"name": "DUT-4", "material_id": "m1", "family": "mof"})
    matched, _ = match_materials(names, core_props=cp,
                                 optimade_index=optimade, qmof_index=qmof)
    row = matched.set_index("name").loc["DUT-4"]
    assert row["match_kind"] == "qmoftab-formula"
    assert row["qmoftab_id"] == "qmof-exact"
    assert row["qmoftab_score"] == pytest.approx(1.0)
    # the losing rung leaves its provenance columns at defaults
    assert row["optimade_id"] == ""
    assert np.isnan(row["optimade_score"])
