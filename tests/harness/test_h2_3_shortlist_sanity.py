"""H2.3 gate (DESIGN §12): shortlist sanity for the T2 ranker.

The brute-force rankings that the adsorbent-ml Stage-2 surrogate's top-k
hit rate will be scored against must reproduce the known screening
result on the anchors: **zeolite 13X ranks poorly on the 60 °C
regeneration datacenter profile**. The mechanism is worth restating
because it is the physics the surrogate has to learn: 13X binds water
strongly (E_char = 14 kJ/mol, Q_st = 3.5 MJ/kg), so against a 35 °C
condenser a 60 °C regeneration barely desorbs it (delta_q ≈ 0.006 vs
0.038 for silica gel RD) while its high Q_st also inflates the heat
input. Warm the regeneration to 80 °C (the `human` profile) and 13X
recovers a third of silica's COP — the ranking is regeneration-
temperature-driven, which is exactly the system-level signal the
equilibrium sweep should surface.
"""

import numpy as np
import pytest

from harness.materials import load_anchors
from harness.rank import (
    load_sweep_materials,
    profile_objective,
    shortlist,
    sweep_materials,
)

PROFILES = ("cpu", "human", "vehicle", "datacenter")
X13 = "Zeolite 13X (NaX)"  # the anchor row (an ISODB fit of 13X also exists)
SILICA_RD = "Silica gel RD"


@pytest.fixture(scope="module")
def ranked():
    materials = load_sweep_materials() + load_anchors()
    return sweep_materials(materials, profiles=PROFILES)


def test_sweep_table_is_sane(ranked):
    assert len(ranked) == len(set(ranked["material"])) * len(PROFILES)
    assert ranked["COP"].notna().all() and np.isfinite(ranked["COP"]).all()
    for profile, group in ranked.groupby("profile"):
        n = len(group)
        by_rank = group.sort_values("rank")
        assert by_rank["rank"].tolist() == list(range(1, n + 1))
        # rank must be monotone in the profile-weighted score
        assert (by_rank["score"].iloc[:-1].values >= by_rank["score"].iloc[1:].values).all()


def test_13x_ranks_poorly_on_datacenter(ranked):
    dc = ranked[ranked["profile"] == "datacenter"]
    n = len(dc)
    r13x = int(dc[dc["material"] == X13]["rank"].iloc[0])
    r_sil = int(dc[dc["material"] == SILICA_RD]["rank"].iloc[0])
    print(f"\nH2.3 datacenter: 13X rank {r13x}/{n}, {SILICA_RD} rank {r_sil}/{n}")
    assert r13x > r_sil
    assert r13x > 0.7 * n, "13X must land in the bottom 30 % on the datacenter profile"


def test_13x_gap_closes_with_hotter_regeneration(ranked):
    """The 13X/silica COP ratio must improve from the 60 °C-regeneration
    datacenter profile to the 80 °C-regeneration human profile — the
    ranking is regeneration-temperature-driven."""
    dc = ranked[ranked["profile"] == "datacenter"]
    hu = ranked[ranked["profile"] == "human"]
    cop = lambda df, name: float(df[df["material"] == name]["COP"].iloc[0])  # noqa: E731
    ratio_dc = cop(dc, X13) / cop(dc, SILICA_RD)
    ratio_hu = cop(hu, X13) / cop(hu, SILICA_RD)
    print(f"H2.3 13X/silica COP ratio: datacenter {ratio_dc:.3f} → human {ratio_hu:.3f}")
    assert ratio_hu > 1.4 * ratio_dc


def test_no_zeolite_in_datacenter_top10(ranked):
    top = shortlist(ranked, "datacenter", k=10)
    assert not top["material"].str.contains("Zeolite", case=False).any()


def test_sweep_objective_matches_profile_weights(ranked):
    """The score column is the profile-weighted normalized objective —
    spot-check one row against the shared objective math."""
    from harness.envs.base import objective_value

    prof_row = ranked[(ranked["profile"] == "cpu") & (ranked["material"] == SILICA_RD)].iloc[0]
    obj = profile_objective("cpu")
    metrics = {"COP": prof_row["COP"], "SCP_W_kg": prof_row["SCP_W_kg"]}
    assert float(objective_value(obj, metrics)) == pytest.approx(prof_row["score"], rel=1e-9)
