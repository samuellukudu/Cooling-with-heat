"""Tests for N2 label assembly (real fits CSV)."""

import numpy as np

from labels import build_labels, family_of


def test_family_rules_spot():
    assert family_of("Zeolite 13X") == "zeolite"
    assert family_of("Na-ZSM-5") == "zeolite"
    assert family_of("Grade 03 Silica Gel") == "silica"
    assert family_of("Activated carbon: BPL") == "carbon"
    assert family_of("UiO-66(Zr)-(COOH)2") == "mof"
    assert family_of("COF-1") == "cof"
    assert family_of("LiCl/silica SWS-1L") == "composite"
    assert family_of("??? unknown 123") == "other"


def test_build_labels_counts_and_targets():
    labels, manifest = build_labels()
    assert manifest["n_isodb"] == 143
    assert manifest["n_anchors"] == 13
    assert manifest["n_with_Q_st"] >= 30  # 21 multi-T + 13 anchors
    assert set(manifest["families"]) <= {"zeolite", "silica", "carbon", "mof",
                                         "cof", "composite", "other"}
    assert (labels["q_sat_kg_kg"] > 0).all()
    assert np.isfinite(labels["log_q_sat"]).all()
    assert np.isfinite(labels["log_E"]).all()
    assert labels["log_Q_st"].notna().sum() == manifest["n_with_Q_st"]
    # Anchors are gold rows: full targets, zero spread
    anchors = labels[labels["anchor"]]
    assert anchors[["log_q_sat", "log_Q_st", "log_E", "n_da"]].notna().all().all()
    assert (anchors["q_sat_logstd"] == 0.0).all()
