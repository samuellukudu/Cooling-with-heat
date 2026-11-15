"""OPTIMADE formula-rung tests — the matching round (ROADMAP item 3).

Synthetic candidate index (no corpora, no HTTP): the rung must match
formula-named adsorbents by composition cosine with exact metal sets, flag
ties ambiguous (cosine is scale-invariant — C9...Cd and C18...Cd2 are the
same composition to this metric), never outrank stronger rungs, and count
itself in the report.
"""

import pandas as pd
import pytest

from match_structures import match_materials, match_optimade_formula


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
