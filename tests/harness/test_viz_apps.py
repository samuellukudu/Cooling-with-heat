"""Physics-based per-application pages (viz_apps).

Pins the pure-python thermo mirror to harness.physics.thermo, the cycle
mirror to Cycle0D.evaluate, and the page contract (one offline HTML per
profile: inline SVG, embedded physics JSON, no external references).
"""

import json
import re
from pathlib import Path

import pytest

import harness
from harness import viz_apps
from harness.viz_apps import cycle_metrics, da_uptake, h_fg_j_kg, psat_pa

FITS = Path(__file__).resolve().parents[2] / "data_cache" / "fits" / "da_params.csv"
needs_fits = pytest.mark.skipif(not FITS.exists(), reason="needs data_cache/fits/da_params.csv")


def test_thermo_mirror_matches_harness():
    from harness.physics.thermo import (  # noqa: PLC0415
        da_uptake as harness_da,
        water_h_fg_j_kg as harness_hfg,
        water_sat_pressure_pa as harness_psat,
    )

    import jax.numpy as jnp  # noqa: PLC0415

    for t_c in (7.0, 10.0, 16.0, 18.0, 35.0, 45.0, 60.0, 75.0, 80.0, 120.0):
        t_k = t_c + 273.15
        assert psat_pa(t_k) == pytest.approx(float(harness_psat(jnp.asarray(t_k))), rel=1e-9)
        assert h_fg_j_kg(t_k) == pytest.approx(float(harness_hfg(jnp.asarray(t_k))), rel=1e-9)
    for t_c, p_kpa in ((35.0, 5.6), (80.0, 47.3), (120.0, 198.5)):
        t_k = t_c + 273.15
        assert da_uptake(t_k, p_kpa * 1000.0, 0.35, 4500.0, 1.8) == pytest.approx(
            float(harness_da(jnp.asarray(t_k), p_kpa * 1000.0, 0.35, 4500.0, 1.8)), rel=1e-9)


def test_cycle_mirror_matches_oracle():
    for profile in ("cpu", "datacenter"):
        env = harness.make("Cycle0D-v0", material="anchor:Silica gel RD", profile=profile)
        ref = env.evaluate()
        m = env.material
        got = cycle_metrics(float(m.q_sat_kg_kg), float(m.q_st_j_kg),
                            env.profile.t_evap_c, env.profile.t_cond_c,
                            env.profile.t_des_c, env.profile.cycle_time_s,
                            float(m.e_char_j_mol), float(m.n_da))
        for key in ("COP", "SCP_W_kg", "delta_q", "q_ads", "q_des"):
            assert got[key] == pytest.approx(ref[key], rel=1e-9), (profile, key)


def test_cycle_spot_values_guard_js_drift():
    """Hardcoded anchors: if these move, the JS twin drifted too."""
    m = cycle_metrics(0.35, 2.5e6, 16.0, 35.0, 60.0, 300.0, 4500.0, 1.8)
    assert m["COP"] == pytest.approx(0.48476142122233334, rel=1e-9)
    assert m["SCP_W_kg"] == pytest.approx(314.8959830817856, rel=1e-9)
    assert m["delta_q"] == pytest.approx(0.03842269235148649, rel=1e-9)
    assert "17.502" in viz_apps.JS_PHYSICS and "240.97" in viz_apps.JS_PHYSICS
    assert "0.321" in viz_apps.JS_PHYSICS and "8.10765" in viz_apps.JS_PHYSICS


def test_bed_field_sample_is_solver_output():
    # n_cycles=2: metrics come from the last N-1 cycles, so 1 cycle yields zeros.
    field = viz_apps.bed_field_sample("cpu", n_cells=8, n_cycles=2, n_snap=7)
    assert len(field["t_s"]) == 7 and len(field["T_C"]) == 7
    assert all(len(row) == 8 for row in field["T_C"] + field["q"])
    t = field["t_s"]
    assert all(b > a for a, b in zip(t, t[1:]))
    x = field["x_mm"]
    assert x[0] > 0 and x[-1] < 2.0  # mm, cell centers inside the 2 mm bed
    assert field["COP"] > 0 and field["SCP_W_kg"] > 0


def test_bed_sim_channels_aligned():
    """Cross-section channels share the snapshot grid and see both phases."""
    from harness import viz_design  # noqa: PLC0415

    sim = viz_design.bed_sim_sample("cpu", n_cells=8, n_cycles=2, n_snap=13)
    n = len(sim["t_s"])
    for key in ("T_C", "q", "phase", "dq_cool", "dq_in", "t_fluid_c",
                "vapor_kpa", "Qcool_cum", "Qin_cum"):
        assert len(sim[key]) == n, key
    assert set(sim["phase"]) == {0, 1}
    assert sim["Qcool_cum"][-1] >= sim["Qcool_cum"][0]  # cumulative books grow
    assert sim["dqmax"] > 0 and sim["qmax"] > 0


def test_machine_sim_sample_is_system_output():
    """TwoBed replay: counter-phase beds, recovery power, conserved levels."""
    from harness import viz_design  # noqa: PLC0415

    sim = viz_design.machine_sim_sample("cpu", "cpu", n_cells=8, n_cycles=2,
                                        n_snap=13)
    n = len(sim["t_s"])
    assert n == 13
    for key in ("A_T_C", "B_T_C", "A_q", "B_q", "sys_phase", "A_phase",
                "B_phase", "A_dq", "B_dq", "q_rec", "A_fluid_c", "B_fluid_c",
                "evap_lvl", "cond_lvl", "Qcool_cum"):
        assert len(sim[key]) == n, key
    assert all(len(row) == 8 for row in sim["A_T_C"] + sim["B_T_C"])
    assert set(sim["sys_phase"]) == {0, 1}  # both system phases visited
    for i in range(n):  # counter-phase: beds always in opposite phases
        assert sim["A_phase"][i] != sim["B_phase"][i]
    assert sim["recmax"] > 0  # recovery window carries real exchange power
    assert len(sim["segs"]) >= 3  # phase-track segments for the timeline
    assert all(0.3 <= v <= 1.0 for v in sim["evap_lvl"] + sim["cond_lvl"])
    ev, cd = sim["evap_lvl"], sim["cond_lvl"]  # integrated duty: evap drains, cond fills
    assert all(b <= a for a, b in zip(ev, ev[1:]))
    assert all(b >= a for a, b in zip(cd, cd[1:]))
    assert {"COP", "SCP_W_kg", "gain", "Q_rec_kJ_m2"} <= set(sim)


@needs_fits
def test_sweep_top_shape():
    rows = viz_apps.sweep_top("datacenter", k=5)
    assert len(rows) == 5
    ranks = [r["rank"] for r in rows]
    assert ranks == sorted(ranks) == [1, 2, 3, 4, 5]
    for r in rows:
        assert {"material", "COP", "SCP_W_kg", "score", "q_sat_kg_kg",
                "Q_st_MJ_kg", "e_char_j_mol", "n_da", "t_window_c",
                "out_of_window"} <= set(r)
        assert r["COP"] >= 0 and r["SCP_W_kg"] >= 0


@needs_fits
def test_page_generation_contract(tmp_path):
    written = viz_apps.write_app_pages(tmp_path, profiles=["cpu", "datacenter_dynamic"],
                                       k=3, with_fields=False)
    assert set(written) == {"cpu", "datacenter_dynamic", "index"}
    for name in ("cpu", "datacenter_dynamic"):
        html = written[name].read_text(encoding="utf-8")
        assert "<svg" in html and "</svg>" in html
        assert "mat-select" in html and "hx-slider" in html and "rank-table" in html
        assert "Silica gel RD" in html
        # Offline single file: no external fetches, fonts, or CDNs.
        assert not re.search(r'src="http|href="http|@import|url\(http', html)
        assert "sch-cop" in html and "iso-a" in html  # JS-updated ids present
    dyn = written["datacenter_dynamic"].read_text(encoding="utf-8")
    assert "45–70" in dyn  # coolant-loop schedule section
    idx = written["index"].read_text(encoding="utf-8")
    assert "cpu.html" in idx and "datacenter_dynamic.html" in idx


@needs_fits
def test_page_simulation_sections_contract(tmp_path):
    """Full build embeds both simulation views with resolvable element ids."""
    written = viz_apps.write_app_pages(tmp_path, profiles=["cpu"], k=2,
                                       with_fields=True)
    html = written["cpu"].read_text(encoding="utf-8")
    for token in ("bx-cell-0", "bx-qd-0-0", "bx-vd-0", "bx-slider", "bx-play",
                  "mV-AE", "mF-AE", "m-slider", "m-play", "m-track", "m-rec",
                  "m-evap-lvl", "drawBedX", "drawMachine", "inferno",
                  "const BX =", "const MS ="):
        assert token in html, token
    assert not re.search(r'src="http|href="http|@import|url\(http', html)
    ids = set(re.findall(r'id="([^"]+)"', html))
    js = re.findall(r"<script>(.*?)</script>", html, re.S)[0]
    for ref in set(re.findall(r'getElementById\("([^"]+)"\)', js)):
        assert ref in ids, f"JS references missing element {ref!r}"
    for k in range(8):  # dynamically addressed cell/dot ids exist too
        assert f"bx-cell-{k}" in ids and f"mA-cell-{k}" in ids
    assert "heat recovery" in html and "valve" in html.lower()


@needs_fits
def test_page_embeds_ranking_and_profile_json(tmp_path):
    written = viz_apps.write_app_pages(tmp_path, profiles=["vehicle"], k=4,
                                       with_fields=False)
    html = written["vehicle"].read_text(encoding="utf-8")
    m = re.search(r"const MATERIALS = (\{.*?\});\n", html, re.S)
    assert m, "embedded MATERIALS JSON missing"
    mats = json.loads(m.group(1))
    assert len(mats) >= 4  # anchor default + sweep rows
    prof = json.loads(re.search(r"const PROFILE = (\{.*?\});\n", html, re.S).group(1))
    assert prof["t_des_c"] == pytest.approx(120.0) and prof["name"] == "vehicle"
