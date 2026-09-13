"""Tests for name→structure matching (rules + self-verifying CoRE hits)."""

import pandas as pd

from match_structures import (match_core, match_iza, match_materials,
                              normalize)


def test_normalize_and_iza_rules():
    assert normalize("MIL-100(Cr)-EG") == "mil100creg"
    assert match_iza("Zeolite 13X") == ("FAU", "high")
    assert match_iza("Zeolite 4A") == ("LTA", "high")
    assert match_iza("Na-ZSM-5") == ("MFI", "high")
    assert match_iza("Silicalite MFI/Na") == ("MFI", "high")
    assert match_iza("Chabazite") == ("CHA", "medium")
    assert match_iza("CuBTC") is None
    assert match_iza("Silica Gel") is None


def test_core_substring_verified_and_ukebod_lesson():
    files = pd.Series(["FOO_ZIF8_BAR_clean", "UKEBOD_X_clean", "OTHER_clean"])
    assert match_core("ZIF-8", files)[0] == "ZIF8"
    # 'BOD' is not a registered synonym (< 4 chars rejected anyway) and no
    # CuBTC synonym hits: UKEBOD must NOT match CuBTC.
    assert match_core("CuBTC", files) is None
    assert match_core("Totally unknown stuff", files) is None


def test_match_materials_real_table_reports_coverage():
    from labels import build_labels  # noqa: PLC0415

    labels, _ = build_labels()
    matched, report = match_materials(
        labels[["name"]], anchor_names=labels[labels["anchor"]]["name"].tolist())
    assert len(matched) == len(labels)
    assert set(matched["match_kind"]) <= {"iza", "core", "anchor", "none"}
    assert report["n_names"] == len(labels)
    assert sum(report["by_kind"].values()) == len(labels)
    # Zeolite trade names resolve; amorphous silica never claims a structure
    by_name = matched.set_index("name")
    assert by_name.loc["Zeolite 13X", "iza_code"] == "FAU"
    assert by_name.loc["Silica Gel", "match_kind"] == "none"
    # Coverage is reported, whatever it is — the moat metric, not a gate
    print("\nmatch coverage:", report["by_kind"],
          "| anchors hit:", report["n_anchor_hits"])
