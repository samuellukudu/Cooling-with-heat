"""Recover-and-run PDE application pages (viz_pde).

The frozen diffheat zoo is executed from git history (never re-added to
the tree) and embedded in standalone offline pages. Tests pin: the frozen
pin/extraction, solver sanity per scenario, the JS live-model mirrors
(TE/absorption) against the physics they mirror, and the page contract
(ids resolve, offline, provenance embedded).
"""

import json
import re

import pytest

from harness import viz_pde
from harness.viz_pde import FROZEN_SHA, PDE_PAGES


def test_frozen_pin_and_extraction():
    pkg = viz_pde.frozen_pkg()
    marker = pkg.parent / "MARKER"
    assert FROZEN_SHA == "42b97c755c37eb3fa301c47a91ef5f769e4e2f8c"
    assert marker.read_text() == FROZEN_SHA + "\nno-top-init\n"
    mods = viz_pde.fro()
    assert set(mods) == {"dh", "fhp", "dhp", "dhs", "dhe"}
    assert hasattr(mods["dh"], "solve_heat_2d")


def test_page_registry_matches_builders():
    assert len(PDE_PAGES) == 13
    assert sorted(s for s, _, _ in PDE_PAGES) == sorted(viz_pde._BUILDERS)


def test_heat1d_eigen_decay():
    d = viz_pde.run_heat1d(quick=True)
    assert d["lambda1"] == pytest.approx(d["lambda1_exact"], rel=0.02)
    assert d["lambda1_exact"] == pytest.approx(9.8696044, rel=1e-6)
    assert len(d["frames"]) == len(d["t"]) == 21
    # Steady line at cell centers: 1-dx/2 … dx/2, monotone decreasing
    # (embedded rounded to 4 decimals).
    assert d["steady"][0] == pytest.approx(1.0 - 0.5 / d["N"], abs=2e-4)
    assert d["steady"][-1] == pytest.approx(0.5 / d["N"], abs=2e-4)
    assert all(b < a for a, b in zip(d["steady"], d["steady"][1:]))


def test_heat2d_approaches_linear_steady():
    d = viz_pde.run_heat2d(quick=True)
    assert d["final_mean"] == pytest.approx(d["steady_mean"], rel=0.1)
    import numpy as np  # noqa: PLC0415

    fr = np.asarray(d["frames"])
    assert fr.max() <= 100.5 and fr.min() >= -0.5


def test_boussinesq_quick_sanity():
    d = viz_pde.run_boussinesq(quick=True)
    assert len(d["runs"]) == 2
    for r in d["runs"]:
        assert r["Nu"] >= 1.0
        assert len(r["glyphs"]) > 0
        assert r["regime"] and r["Nu_corr"] >= 1.0


def test_cloak_objectives_sane():
    d = viz_pde.run_cloak(quick=True)
    assert len(d["runs"]) == 2
    for r in d["runs"]:
        assert r["grad"] >= 0 and r["mismatch"] >= 0
        assert r["klo"] <= r["khi"]


def test_telegrapher_wave_beats_diffusion():
    d = viz_pde.run_telegrapher(quick=True)
    by_label = {r["label"].split("(")[1].split(",")[0].strip(): r for r in d["runs"]}
    assert by_label["wave"]["far_mid"] < by_label["damped"]["far_mid"]
    assert by_label["wave"]["v"] == pytest.approx(1.414, rel=0.01)


def test_te_lumped_spot_and_js_mirror():
    m = viz_pde.fmod("thermoelectric")
    r = m.simulate_te(Tc_C=18.0, Th_C=35.0, current_A=3.0, pair="Bi2Te3")
    assert float(r["Qc_W"]) == pytest.approx(0.154880, rel=1e-6)
    assert float(r["COP"]) == pytest.approx(0.533333, rel=1e-6)
    d = viz_pde.run_te(quick=True)
    assert d["parity"]["leg_COP"] == pytest.approx(d["parity"]["lumped_COP"], rel=0.05)
    html = viz_pde.page_te(d)
    assert "S * Tc * I - 0.5 * I * I * R" in html  # JS closed form present


def test_absorption_anchor_and_js_mirror():
    from harness.physics import absorption as ab  # noqa: PLC0415

    a = ab.simulate_absorption(t_evap_c=7.0, t_cond_c=30.0, t_gen_c=85.0,
                               pair="LiBr-H2O")
    assert float(a["COP"]) == pytest.approx(0.70, rel=0.15)  # V-ABS band
    d = viz_pde.run_absorption(quick=True)
    assert d["anchor"]["COP"] == pytest.approx(0.692, rel=1e-3)
    html = viz_pde.page_absorption(d)
    assert "COP_ideal" in html and "no circulation" in html


def test_metamaterial_search_table():
    d = viz_pde.run_metamaterial(quick=True)
    assert len(d["table"]) == 12
    objs = [r["obj"] for r in d["table"]]
    assert objs == sorted(objs)
    assert d["best"] == d["table"][0]


def test_pde_pages_quick_contract(tmp_path):
    written = viz_pde.write_pde_pages(tmp_path, quick=True)
    assert set(written) == {s for s, _, _ in PDE_PAGES} | {"index"}
    for slug, _, _ in PDE_PAGES:
        html = written[slug].read_text(encoding="utf-8")
        assert "<svg" in html
        # Provenance: frozen pin embedded, or live-tree note for absorption.
        assert FROZEN_SHA[:7] in html or "live tree core" in html
        assert not re.search(r'src="http|href="http|@import|url\(http', html)
        ids = set(re.findall(r'id="([^"]+)"', html))
        scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
        assert len(scripts) == 1
        for ref in set(re.findall(r'getElementById\("([^"]+)"\)', scripts[0])):
            if re.match(r"^(run|meters)-\d+$", ref):
                k = int(ref.split("-")[1])
                base = len(re.findall(rf'id="{ref.split("-")[0]}-\d+"', html))
                assert k < base, (slug, ref)
            else:
                assert ref in ids, (slug, ref)
    idx = written["index"].read_text(encoding="utf-8")
    for slug, _, _ in PDE_PAGES:
        assert f"{slug}.html" in idx
