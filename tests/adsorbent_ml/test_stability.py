"""Tests for the stability-label ingest (ACQUISITION §6)."""

import pytest

from stability import StabilityRecord, feasibility_filter, load_stability_csv


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_load_aliases_and_types(tmp_path):
    p = _write(tmp_path / "s.csv",
               "refcode,hydrolytically_stable,decomposition_T_C,doi\n"
               "UiO-66,yes,450,10.0000/x\n"
               "MOF-5,no,,10.0000/y\n"
               "MOF-177,,,10.0000/z\n")
    recs = {r.name: r for r in load_stability_csv(p)}
    assert recs["UiO-66"].water_stable is True
    assert recs["UiO-66"].thermal_decomp_c == pytest.approx(450.0)
    assert recs["MOF-5"].water_stable is False
    assert recs["MOF-5"].thermal_decomp_c is None
    assert recs["MOF-177"].water_stable is None


def test_bad_flag_and_missing_name_column(tmp_path):
    p = _write(tmp_path / "s.csv", "mof_name,water_stable\nX,maybe\n")
    with pytest.raises(ValueError, match="cannot parse water-stability"):
        load_stability_csv(p)
    q = _write(tmp_path / "q.csv", "foo,bar\n1,2\n")
    with pytest.raises(ValueError, match="no name column"):
        load_stability_csv(q)


def test_feasibility_unknown_is_not_stable():
    recs = [StabilityRecord("A", True, 300.0),
            StabilityRecord("B", False, 500.0),
            StabilityRecord("C", None, 500.0),
            StabilityRecord("D", True, 100.0),
            StabilityRecord("E", True, None)]
    ok, bad = feasibility_filter(recs, min_decomp_c=150.0)
    assert [r.name for r in ok] == ["A", "E"]  # C unknown-water fails; D too fragile
    assert [r.name for r in bad] == ["B", "C", "D"]
    ok2, _ = feasibility_filter(recs, require_water_stable=False)
    assert "C" in [r.name for r in ok2]  # lenient mode admits unknowns, still not falses


def test_solvent_removal_label_loads_and_gates(tmp_path):
    p = _write(tmp_path / "s.csv",
               "mof_name,water_stable,solvent_removal_stable,thermal_decomp_c,source,confidence\n"
               "ABAVIJ,,true,413,10.0000/a,text-mined\n"
               "MOF-5,,true,350,10.0000/b,text-mined\n"
               "CRM1,,false,300,10.0000/c,text-mined\n"
               "UNKN,,,,,\n")
    recs = {r.name: r for r in load_stability_csv(p)}
    assert recs["ABAVIJ"].water_stable is None
    assert recs["ABAVIJ"].solvent_removal_stable is True
    assert recs["CRM1"].solvent_removal_stable is False

    ok, bad = feasibility_filter(list(recs.values()), min_decomp_c=150.0)
    # unknown water + solvent-stable passes (conservative desolvation evidence);
    # explicit water_stable=false never passes; unknown-everything fails.
    assert [r.name for r in ok] == ["ABAVIJ", "MOF-5"]
    assert [r.name for r in bad] == ["CRM1", "UNKN"]
