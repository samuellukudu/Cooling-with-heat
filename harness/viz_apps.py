"""Physics-based interactive pages for each application profile.

One self-contained ``<profile>.html`` per application (plus ``index.html``)
under ``docs/apps/`` — inline SVG + vanilla JS, no external dependencies, so
the pages work offline straight from the file system:

```bash
python -m harness.viz_apps --out docs/apps   # all five profiles
python -m harness.viz_apps --out /tmp/apps --profiles datacenter --no-fields  # fast preview
```

Every number on the page comes from the harness physics, not mock data:

- equilibrium cycle (COP/SCP/Δq) from ``harness.physics.cycle0d`` via
  ``Cycle0D.evaluate`` — and the *live* recomputation in the page mirrors
  the same correlations (Magnus/Antoine Psat, Watson h_fg, D–A uptake) in
  JavaScript, pinned by ``tests/harness/test_viz_apps.py``;
- the bed-dynamics scrubber replays an actual ``Bed1D`` solver rollout
  (``collect_fields=True``) downsampled for embedding;
- the ranking table is the T2 reference sweep (``rank.sweep_materials``)
  for that profile;
- ``datacenter_dynamic`` additionally plots its 45–70 °C coolant-loop
  source schedule.

Matplotlib is never involved: SVG is emitted as text, interactivity is
plain DOM scripting.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Pure-python thermo mirror (curve sampling + page data only).
#
# Constants mirror harness.physics.thermo exactly (Magnus below 100 °C,
# NIST-Antoine above, Watson exponent 0.321, R = 8.314); parity is pinned in
# tests/harness/test_viz_apps.py. The live page recomputation uses the JS
# twin of these same expressions (JS_PHYSICS below).
# --------------------------------------------------------------------------

GAS_CONSTANT = 8.314
CP_ADSORBENT = 1000.0
CP_LIQUID = 4184.0
HX_DEFAULT = 1.35

APP_PROFILES = ("cpu", "human", "vehicle", "datacenter", "datacenter_dynamic")
STEADY_SWEEP = {"cpu": "cpu", "human": "human", "vehicle": "vehicle",
                "datacenter": "datacenter", "datacenter_dynamic": "datacenter"}


def psat_pa(t_k: float) -> float:
    """Water saturation pressure [Pa] (thermo.py two-branch mirror)."""
    t_c = t_k - 273.15
    if t_c < 100.0:
        return 611.2 * math.exp(17.502 * t_c / (240.97 + t_c))
    return (10.0 ** (8.10765 - 1750.286 / (235.0 + t_c))) * 133.322


def h_fg_j_kg(t_k: float) -> float:
    """Latent heat [J/kg] (thermo.py Watson mirror)."""
    t_crit, t_ref, h_ref = 647.1, 373.15, 2_256_400.0
    t = min(t_k, t_crit - 1.0)
    return h_ref * ((t_crit - t) / (t_crit - t_ref)) ** 0.321


def da_uptake(t_k: float, p_pa: float, q_sat: float,
              e_char: float, n: float) -> float:
    """D–A equilibrium uptake [kg/kg] (thermo.py mirror)."""
    ratio = max(psat_pa(t_k) / p_pa, 1.0)
    potential = GAS_CONSTANT * t_k * math.log(ratio)
    return q_sat * math.exp(-((potential / e_char) ** n))


def cycle_metrics(q_sat: float, q_st: float, t_evap_c: float, t_cond_c: float,
                  t_des_c: float, cycle_time_s: float, e_char: float,
                  n: float, hx: float = HX_DEFAULT) -> dict[str, float]:
    """Equilibrium cycle (cycle0d.simulate_cycle mirror, water refrigerant)."""
    t_evap, t_cond, t_des, t_ads = (t + 273.15 for t in
                                    (t_evap_c, t_cond_c, t_des_c, t_cond_c))
    p_evap, p_cond, h_fg = psat_pa(t_evap), psat_pa(t_cond), h_fg_j_kg(t_evap)
    q_ads = da_uptake(t_ads, p_evap, q_sat, e_char, n)
    q_des = da_uptake(t_des, p_cond, q_sat, e_char, n)
    delta_q = max(0.0, q_ads - q_des)
    q_cool = delta_q * h_fg
    q_in = hx * ((CP_ADSORBENT + q_ads * CP_LIQUID) * (t_des - t_ads)
                 + delta_q * q_st)
    return {
        "COP": q_cool / q_in if q_in > 0 else 0.0,
        "SCP_W_kg": q_cool / cycle_time_s if cycle_time_s > 0 else 0.0,
        "delta_q": delta_q, "q_ads": q_ads, "q_des": q_des,
        "P_evap_kPa": p_evap / 1000.0, "P_cond_kPa": p_cond / 1000.0,
        "h_fg_MJ_kg": h_fg / 1e6,
    }


# --------------------------------------------------------------------------
# Data assembly (harness imports stay inside functions so `import harness`
# keeps its startup cost and viz_apps stays import-light like viz.py).
# --------------------------------------------------------------------------

def profile_info(name: str) -> dict:
    """Setpoints/weights/description for one application page."""
    from .profiles import get_profile  # noqa: PLC0415

    p = get_profile(name)
    return {
        "name": p.name, "description": p.description,
        "t_evap_c": p.t_evap_c, "t_cond_c": p.t_cond_c, "t_des_c": p.t_des_c,
        "cycle_time_s": p.cycle_time_s,
        "cop_weight": p.cop_weight, "scp_weight": p.scp_weight,
        "t_fluid_min_c": p.t_fluid_min_c, "t_fluid_max_c": p.t_fluid_max_c,
        "notes": p.notes,
        "dynamic": p.source_schedule is not None,
    }


def sweep_top(profile: str, k: int = 10) -> list[dict]:
    """Top-k T2 reference rows for one profile (steady setpoints)."""
    from . import rank  # noqa: PLC0415

    steady = STEADY_SWEEP[profile]
    mats = rank.load_sweep_materials()
    df = rank.sweep_materials_batched(mats, profiles=[steady])
    top = df.nsmallest(k, "rank")
    rows = []
    for r in top.to_dict("records"):
        rows.append({
            "material": str(r["material"]), "source": str(r["source"]),
            "rank": int(r["rank"]), "score": float(r["score"]),
            "COP": float(r["COP"]), "SCP_W_kg": float(r["SCP_W_kg"]),
            "delta_q": float(r["delta_q"]),
            "q_sat_kg_kg": float(r["q_sat_kg_kg"]),
            "Q_st_MJ_kg": float(r["Q_st_MJ_kg"]),
            "e_char_j_mol": float(r["e_char_j_mol"]),
            "n_da": float(r["n_da"]),
            "t_window_c": str(r["t_window_c"]),
            "out_of_window": bool(r["out_of_window"]),
        })
    return rows


def anchor_params() -> dict:
    """Default cycle material (anchor silica gel RD) parameters."""
    from .materials import get_material  # noqa: PLC0415

    m = get_material("anchor:Silica gel RD")
    return {"material": m.name, "source": m.source,
            "q_sat_kg_kg": float(m.q_sat_kg_kg),
            "Q_st_MJ_kg": float(m.q_st_j_kg) / 1e6,
            "e_char_j_mol": float(m.e_char_j_mol), "n_da": float(m.n_da)}


def bed_field_sample(profile: str, *, n_cells: int = 8, n_cycles: int = 2,
                     n_snap: int = 61) -> dict:
    """Downsampled Bed1D solver rollout for the dynamics scrubber.

    Real ``simulate_bed`` fields (``collect_fields=True``) thinned to
    ``n_snap`` time slices, plus per-snapshot phase/flow/book channels for
    the cross-section view (see ``harness.viz_design``); rounded for
    embedding size.
    """
    from . import viz_design  # noqa: PLC0415

    return viz_design.bed_sim_sample(profile, n_cells=n_cells,
                                     n_cycles=n_cycles, n_snap=n_snap)


def schedule_sample(profile: str, n: int = 120) -> dict | None:
    """Source-schedule samples for dynamic profiles (None when steady)."""
    from .profiles import get_profile  # noqa: PLC0415

    sched = get_profile(profile).source_schedule
    if sched is None:
        return None
    import numpy as np  # noqa: PLC0415

    t = np.linspace(0.0, 14400.0, n)
    return {"t_s": [round(float(v), 1) for v in t],
            "t_c": [round(float(sched(float(v))), 2) for v in t]}


# --------------------------------------------------------------------------
# SVG builders (text emission; dark theme matches harness/viz.py).
# --------------------------------------------------------------------------

_BG, _FG, _AX, _AMB, _GRN, _CYN, _PNL, _SPN = (
    "#0f1115", "#e2e8f0", "#94a3b8", "#f59e0b", "#22c55e", "#06b6d4",
    "#161a21", "#1e222a")


def _svg_open(w: int, h: int) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'style="background:{_BG};border:1px solid {_SPN};border-radius:8px">')


def cycle_schematic_svg() -> str:
    """Four-node cycle diagram; dynamic labels carry ids for JS updates."""
    s = [_svg_open(720, 250)]
    s.append(f'<text x="16" y="24" fill="{_FG}" font-size="13" font-weight="bold">'
             'Adsorption cycle at profile setpoints</text>')
    boxes = [("Evaporator", 30, 60, "T_evap", "P_evap"),
             ("Bed · adsorb", 205, 60, "T_ads", "P_evap"),
             ("Bed · desorb", 380, 60, "T_des", "P_cond"),
             ("Condenser", 555, 60, "T_cond", "P_cond")]
    for title, x, y, t_id, p_id in boxes:
        s.append(f'<rect x="{x}" y="{y}" width="135" height="120" rx="8" '
                 f'fill="{_PNL}" stroke="{_SPN}"/>')
        s.append(f'<text x="{x + 10}" y="{y + 22}" fill="{_FG}" font-size="11">'
                 f'{title}</text>')
        s.append(f'<text id="sch-{t_id}" x="{x + 10}" y="{y + 48}" fill="{_AMB}" '
                 f'font-size="12">–</text>')
        s.append(f'<text id="sch-{p_id}" x="{x + 10}" y="{y + 68}" fill="{_CYN}" '
                 f'font-size="11">–</text>')
        if "Bed" in title:
            q_id = "q_ads" if "adsorb" in title else "q_des"
            s.append(f'<text id="sch-{q_id}" x="{x + 10}" y="{y + 88}" '
                     f'fill="{_GRN}" font-size="11">–</text>')
    for x1, x2, label in ((165, 205, "vapour"), (340, 380, "heat in"),
                          (515, 555, "vapour"), (90, 90, "")):
        if label:
            s.append(f'<line x1="{x1}" y1="120" x2="{x2}" y2="120" '
                     f'stroke="{_AX}" stroke-width="1.5" marker-end="url(#ah)"/>')
            s.append(f'<text x="{(x1 + x2) / 2}" y="110" fill="{_AX}" '
                     f'font-size="9" text-anchor="middle">{label}</text>')
    s.append('<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" '
             'refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" '
             f'fill="{_AX}"/></marker></defs>')
    s.append(f'<text x="16" y="212" fill="{_AX}" font-size="11">Working capacity '
             '<tspan id="sch-dq" fill="#e2e8f0">–</tspan> · Cooling '
             '<tspan id="sch-qcool" fill="#e2e8f0">–</tspan> · Heat in '
             '<tspan id="sch-qin" fill="#e2e8f0">–</tspan></text>')
    s.append(f'<text x="16" y="234" fill="{_AX}" font-size="11">COP '
             '<tspan id="sch-cop" fill="#f59e0b" font-weight="bold">–</tspan>'
             ' · SCP <tspan id="sch-scp" fill="#f59e0b" font-weight="bold">–'
             '</tspan> · score <tspan id="sch-score">–</tspan> '
             '<tspan id="sch-oow"></tspan></text>')
    s.append('</svg>')
    return "\n".join(s)


def chart_svg(chart_id: str, title: str, xlabel: str, ylabel: str,
              w: int = 720, h: int = 300) -> str:
    """Empty log/linear chart skeleton; JS draws paths into it."""
    s = [_svg_open(w, h)]
    s.append(f'<text x="16" y="24" fill="{_FG}" font-size="13" '
             f'font-weight="bold">{title}</text>')
    s.append(f'<g id="{chart_id}-grid"></g>')
    s.append(f'<path id="{chart_id}-a" fill="none" stroke="{_AMB}" '
             f'stroke-width="2"/>')
    s.append(f'<path id="{chart_id}-b" fill="none" stroke="{_CYN}" '
             f'stroke-width="2"/>')
    s.append(f'<g id="{chart_id}-pts"></g>')
    s.append(f'<text x="16" y="{h - 10}" fill="{_AX}" font-size="10">{xlabel}'
             f'</text><text x="16" y="42" fill="{_AX}" font-size="10">{ylabel}'
             f'</text>')
    s.append(f'<g id="{chart_id}-legend"></g>')
    s.append('</svg>')
    return "\n".join(s)


def heatmap_svg(field: dict, w: int = 720, h: int = 190) -> str:
    """Space–time T heatmap rects from the embedded rollout + scrub line."""
    T = field["T_C"]
    lo = min(min(r) for r in T)
    hi = max(max(r) for r in T)
    span = max(hi - lo, 1e-9)
    n_t, n_x = len(T), len(T[0])
    ox, oy, pw, ph = 56, 14, w - 72, h - 52
    s = [_svg_open(w, h)]
    s.append(f'<text x="16" y="24" fill="{_FG}" font-size="13" '
             f'font-weight="bold">Bed temperature history T(x,t) [°C]</text>')
    for i, row in enumerate(T):
        for j, v in enumerate(row):
            f = (v - lo) / span
            r_, g_, b_ = int(20 + 220 * f), int(20 + 120 * (1 - f)), int(40 + 60 * (1 - f))
            x = ox + pw * j / n_x
            y = oy + ph * (1 - (i + 1) / n_t)
            s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{pw / n_x + 0.5:.1f}" '
                     f'height="{ph / n_t + 0.5:.1f}" fill="rgb({r_},{g_},{b_})"/>')
    s.append(f'<line id="hm-scrub" x1="{ox}" y1="{oy}" x2="{ox + pw}" y2="{oy}" '
             f'stroke="#fff" stroke-width="1.5"/>')
    s.append(f'<text x="{ox}" y="{h - 8}" fill="{_AX}" font-size="10">x: wall → '
             f'vapor ({field["x_mm"][0]:.2f}–{field["x_mm"][-1]:.2f} mm)</text>')
    s.append(f'<text x="{ox + pw}" y="{h - 8}" fill="{_AX}" font-size="10" '
             f'text-anchor="end">{lo:.1f}–{hi:.1f} °C</text>')
    s.append('</svg>')
    return "\n".join(s)


def profile_plot_svg() -> str:
    """T(x)/q(x) snapshot skeleton; JS redraws paths on scrub."""
    s = [_svg_open(720, 260)]
    s.append(f'<text x="16" y="24" fill="{_FG}" font-size="13" '
             f'font-weight="bold">Bed snapshot <tspan id="snap-t">–</tspan></text>')
    s.append('<path id="prof-T" fill="none" stroke="#f59e0b" stroke-width="2"/>')
    s.append('<path id="prof-q" fill="none" stroke="#22c55e" '
             'stroke-width="1.6" stroke-dasharray="5 3"/>')
    s.append(f'<g id="prof-grid"></g><text x="16" y="250" fill="{_AX}" '
             f'font-size="10">x [mm, wall → vapor] · left axis T [°C] (amber), '
             f'right axis q [kg/kg] (green)</text>')
    s.append('</svg>')
    return "\n".join(s)


# --------------------------------------------------------------------------
# JavaScript twin of the physics (live recomputation, no network).
# --------------------------------------------------------------------------

JS_PHYSICS = """
const R = 8.314, CP_S = 1000.0, CP_L = 4184.0;
function psatPa(tK) {
  const tC = tK - 273.15;
  if (tC < 100.0) return 611.2 * Math.exp(17.502 * tC / (240.97 + tC));
  return Math.pow(10, 8.10765 - 1750.286 / (235.0 + tC)) * 133.322;
}
function hFg(tK) {
  const Tc = 647.1, Tr = 373.15, Hr = 2256400.0;
  const t = Math.min(tK, Tc - 1.0);
  return Hr * Math.pow((Tc - t) / (Tc - Tr), 0.321);
}
function daUptake(tK, pPa, qSat, E, n) {
  const ratio = Math.max(psatPa(tK) / pPa, 1.0);
  const A = R * tK * Math.log(ratio);
  return qSat * Math.exp(-Math.pow(A / E, n));
}
function cycle(m, P, hx, cyc) {
  const tE = P.t_evap_c + 273.15, tC = P.t_cond_c + 273.15, tD = P.t_des_c + 273.15;
  const pE = psatPa(tE), pCd = psatPa(tC), hfg = hFg(tE);
  const qA = daUptake(tC, pE, m.q_sat_kg_kg, m.e_char_j_mol, m.n_da);
  const qD = daUptake(tD, pCd, m.q_sat_kg_kg, m.e_char_j_mol, m.n_da);
  const dq = Math.max(0, qA - qD), qc = dq * hfg;
  const qin = hx * ((CP_S + qA * CP_L) * (tD - tC) + dq * m.Q_st_MJ_kg * 1e6);
  return { COP: qin > 0 ? qc / qin : 0, SCP: cyc > 0 ? qc / cyc : 0,
    dq, qA, qD, pE, pCd, hfg, qc, qin };
}
function scoreOf(c, P) {
  const w = P.cop_weight + P.scp_weight;
  const nC = Math.min(Math.max((c.COP - 0.05) / (0.85 - 0.05), 0), 1);
  const nS = Math.min(Math.max((c.SCP - 20) / (1600 - 20), 0), 1);
  return (P.cop_weight / w) * nC + (P.scp_weight / w) * nS;
}
"""

JS_PAGE = """
const $ = (id) => document.getElementById(id);
const fmt = (v, d=3) => Number(v).toFixed(d);
function currentMaterial() { return MATERIALS[$("mat-select").value]; }
function outOfWindow(m) {
  const lo = parseFloat(m.t_window_c.split("-")[0]), hi = parseFloat(m.t_window_c.split("-")[1]);
  return (PROFILE.t_des_c > hi || PROFILE.t_cond_c < lo);
}
function updateCycle() {
  const m = currentMaterial();
  const hx = parseFloat($("hx-slider").value), cyc = parseFloat($("cyc-slider").value);
  $("hx-val").textContent = fmt(hx, 2); $("cyc-val").textContent = fmt(cyc, 0) + " s";
  const c = cycle(m, PROFILE, hx, cyc), s = scoreOf(c, PROFILE);
  $("sch-T_evap").textContent = fmt(PROFILE.t_evap_c, 1) + " °C";
  $("sch-P_evap").textContent = fmt(c.pE / 1000, 2) + " kPa · h_fg " + fmt(c.hfg / 1e6, 2) + " MJ/kg";
  $("sch-T_ads").textContent = fmt(PROFILE.t_cond_c, 1) + " °C (bed)";
  $("sch-T_des").textContent = fmt(PROFILE.t_des_c, 1) + " °C (bed)";
  $("sch-P_cond").textContent = fmt(c.pCd / 1000, 2) + " kPa";
  $("sch-q_ads").textContent = "q* = " + fmt(c.qA, 4) + " kg/kg";
  $("sch-q_des").textContent = "q* = " + fmt(c.qD, 4) + " kg/kg";
  $("sch-dq").textContent = "Δq = " + fmt(c.dq, 4) + " kg/kg";
  $("sch-qcool").textContent = fmt(c.qc / 1000, 1) + " kJ/kg";
  $("sch-qin").textContent = fmt(c.qin / 1000, 1) + " kJ/kg";
  $("sch-cop").textContent = fmt(c.COP, 3); $("sch-scp").textContent = fmt(c.SCP, 0) + " W/kg";
  $("sch-score").textContent = fmt(s, 3);
  const oow = outOfWindow(m);
  $("sch-oow").textContent = oow ? "· OUT-OF-WINDOW (extrapolated)" : "";
  $("sch-oow").setAttribute("fill", oow ? "#f87171" : "#22c55e");
  drawIsotherm(m, c);
  document.querySelectorAll("#rank-table tr[data-mat]").forEach((tr) => {
    tr.style.background = tr.dataset.mat === $("mat-select").value ? "#223042" : "";
  });
}
function xLog(p, lo, hi, ox, pw) { return ox + pw * (Math.log(p) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)); }
function yLin(v, lo, hi, oy, ph) { return oy + ph * (1 - (v - lo) / (hi - lo)); }
function drawIsotherm(m, c) {
  const W = 720, ox = 56, oy = 40, pw = W - 76, ph = 200;
  const lo = Math.min(c.pE, c.pCd) * 0.1, hi = Math.max(c.pE, c.pCd) * 1.3;
  let qmax = 0; const ptsA = [], ptsD = [];
  for (let i = 0; i <= 60; i++) {
    const p = lo * Math.pow(hi / lo, i / 60);
    const a = daUptake(PROFILE.t_cond_c + 273.15, p, m.q_sat_kg_kg, m.e_char_j_mol, m.n_da);
    const d = daUptake(PROFILE.t_des_c + 273.15, p, m.q_sat_kg_kg, m.e_char_j_mol, m.n_da);
    qmax = Math.max(qmax, a, d); ptsA.push([p, a]); ptsD.push([p, d]);
  }
  qmax *= 1.1;
  const path = (pts) => pts.map(([p, q], i) =>
    (i ? "L" : "M") + xLog(p, lo, hi, ox, pw).toFixed(1) + "," + yLin(q, 0, qmax, oy, ph).toFixed(1)).join(" ");
  $("iso-a").setAttribute("d", path(ptsA)); $("iso-b").setAttribute("d", path(ptsD));
  const g = $("iso-pts"); g.innerHTML = "";
  [[c.pE, c.qA, "#f59e0b"], [c.pCd, c.qD, "#06b6d4"]].forEach(([p, q]) => {
    const el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    el.setAttribute("cx", xLog(p, lo, hi, ox, pw)); el.setAttribute("cy", yLin(q, 0, qmax, oy, ph));
    el.setAttribute("r", 5); el.setAttribute("fill", "#fff");
    g.appendChild(el);
  });
  const lg = $("iso-legend"); lg.innerHTML = "";
  [["T_ads = " + PROFILE.t_cond_c + " °C", "#f59e0b"], ["T_des = " + PROFILE.t_des_c + " °C", "#06b6d4"]].forEach(([t, col], i) => {
    const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
    el.setAttribute("x", 560); el.setAttribute("y", 56 + 18 * i);
    el.setAttribute("fill", col); el.setAttribute("font-size", 11); el.textContent = t;
    lg.appendChild(el);
  });
}
function init() {
  $("mat-select").addEventListener("change", updateCycle);
  $("hx-slider").addEventListener("input", updateCycle);
  $("cyc-slider").addEventListener("input", updateCycle);
  document.querySelectorAll("#rank-table tr[data-mat]").forEach((tr) => {
    tr.addEventListener("click", () => { $("mat-select").value = tr.dataset.mat; updateCycle(); });
    tr.style.cursor = "pointer";
  });
  updateCycle();
}
document.addEventListener("DOMContentLoaded", init);
"""


def _page_html(info: dict, rows: list[dict], anchor: dict,
               field: dict | None, ms: dict | None, sched: dict | None,
               provenance: dict) -> str:
    from . import viz_design  # noqa: PLC0415 (lazy: page-gen path only)
    mats: dict[str, dict] = {anchor["material"]: {**anchor, "rank": 0,
                                                  "score": float("nan"),
                                                  "COP": float("nan"),
                                                  "SCP_W_kg": float("nan"),
                                                  "delta_q": float("nan"),
                                                  "t_window_c": "20-100",
                                                  "out_of_window": False}}
    for r in rows:
        mats[r["material"]] = r
    mat_opts = "\n".join(
        f'<option value="{m}"{" selected" if m == anchor["material"] else ""}>{m}</option>'
        for m in mats)
    rank_rows = "\n".join(
        f'<tr data-mat="{r["material"]}"><td>{r["rank"]}</td><td>{r["material"]}</td>'
        f'<td>{r["COP"]:.3f}</td><td>{r["SCP_W_kg"]:.0f}</td>'
        f'<td>{r["score"]:.3f}</td><td>{"⚠️" if r["out_of_window"] else ""}</td></tr>'
        for r in rows)
    sched_sec = ""
    if sched is not None:
        pts = " ".join(f"{56 + 644 * t / 14400:.0f},{200 - 140 * (v - 45) / 25:.0f}"
                       for t, v in zip(sched["t_s"], sched["t_c"]))
        sched_sec = f"""
<h2>Source schedule — 45–70 °C coolant loop (4 h period)</h2>
{_svg_open(720, 240)}
<text x="16" y="24" fill="{_FG}" font-size="13" font-weight="bold">Time-varying drive (H2.2 schedule work)</text>
<polyline points="{pts}" fill="none" stroke="{_CYN}" stroke-width="2"/>
<text x="16" y="230" fill="{_AX}" font-size="10">t [0–4 h] · T_source [45–70 °C] — schedules are data; pass your own callable to the env.</text>
</svg>"""
    bed_sec: str
    if field is not None:
        tmax = field["t_s"][-1]
        bed_sec = f"""
<h2>Simulation — bed cross-section (Bed1D solver replay, silica gel RD)</h2>
<p class="dim">One slice through the coated bed: heat-transfer fluid | metal wall |
adsorbent cells colored by the solved temperature | vapour pipe. Cyan dots are
adsorbed water per cell; pipe dots are vapour flowing to/from the bed, sized by
the solved duty rate. Episode COP {field["COP"]:.3f} · SCP {field["SCP_W_kg"]:.0f} W/kg ·
transport defaults flagged (DESIGN §8.1). Drag the slider or press play.</p>
<div class="controls"><button id="bx-play">▶ play</button>
<input id="bx-slider" type="range" min="0" max="{len(field["t_s"]) - 1}" value="0" step="1">
<span class="dim">0 – {tmax:.0f} s</span></div>
{viz_design.bed_xsection_svg("bx", field["n_cells"])}"""
    else:
        bed_sec = "<h2>Simulation — bed cross-section</h2><p class=\"dim\">Solver replay skipped (--no-fields).</p>"
    machine_sec: str
    if ms is not None:
        machine_sec = f"""
<h2>Simulation — two-bed machine (TwoBed system replay, {ms["t_rec_s"]:.0f} s recovery window)</h2>
<p class="dim">Counter-phase pair on silica gel RD: twin cross-sections (cell color =
bed temperature, green bar = mean uptake) · vapor valves open/close with the solved
phases · evaporator/condenser levels integrate the solved vapor duty · the amber link
carries the solved heat-recovery power. System COP {ms["COP"]:.3f} · SCP {ms["SCP_W_kg"]:.0f} W/kg.</p>
<div class="controls"><button id="m-play">▶ play</button>
<input id="m-slider" type="range" min="0" max="{len(ms["t_s"]) - 1}" value="0" step="1">
<span class="dim">0 – {ms["t_s"][-1]:.0f} s · track: cyan = A adsorbs, amber = B adsorbs</span></div>
{viz_design.machine_svg(ms["n_cells"])}"""
    else:
        machine_sec = "<h2>Simulation — two-bed machine</h2><p class=\"dim\">System replay skipped (--no-fields).</p>"
    js_data = ("const PROFILE = " + json.dumps(info) + ";\n"
               + "const MATERIALS = " + json.dumps(mats) + ";\n"
               + ("const BX = " + json.dumps(field) + ";\n" if field else "")
               + ("const MS = " + json.dumps(ms) + ";\n" if ms else ""))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{info["name"]} — {info["description"]} (adsorption cooling)</title>
<style>
body {{ background:{_BG}; color:{_FG}; font-family:system-ui,sans-serif; max-width:760px; margin:0 auto; padding:16px; }}
h1 {{ font-size:22px; }} h2 {{ font-size:16px; margin-top:28px; }}
{viz_design.CSS_ANIM}
.chips span {{ display:inline-block; background:{_PNL}; border:1px solid {_SPN}; border-radius:6px; padding:3px 9px; margin:2px; font-size:12px; }}
.controls {{ display:flex; gap:10px; align-items:center; margin:10px 0; flex-wrap:wrap; font-size:13px; }}
select, input[type=range] {{ accent-color:{_AMB}; }} select {{ background:{_PNL}; color:{_FG}; border:1px solid {_SPN}; border-radius:6px; padding:4px; }}
button {{ background:{_PNL}; color:{_FG}; border:1px solid {_SPN}; border-radius:6px; padding:4px 10px; cursor:pointer; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }} td, th {{ border:1px solid {_SPN}; padding:4px 6px; text-align:left; }}
th {{ color:{_AX}; }} .dim {{ color:{_AX}; font-size:12px; }} a {{ color:{_CYN}; }}
</style>
</head>
<body>
<p class="dim"><a href="index.html">← all applications</a> · generated {provenance["date"]} · {provenance["note"]}</p>
<h1>{info["name"]} — {info["description"]}</h1>
<p class="dim">{info["notes"]}</p>
<div class="chips">
<span>T_evap {info["t_evap_c"]:.0f} °C</span><span>T_cond {info["t_cond_c"]:.0f} °C</span>
<span>T_des {info["t_des_c"]:.0f} °C</span><span>half-cycle {info["cycle_time_s"]:.0f} s</span>
<span>COP wt {info["cop_weight"]:.2f}</span><span>SCP wt {info["scp_weight"]:.2f}</span>
<span>fluid {info["t_fluid_min_c"]:.0f}–{info["t_fluid_max_c"]:.0f} °C</span>
</div>
{bed_sec}
{machine_sec}
<h2>Equilibrium cycle — live physics</h2>
<div class="controls"><label>material <select id="mat-select">{mat_opts}</select></label>
<label>hx <input id="hx-slider" type="range" min="1" max="2" step="0.01" value="1.35"><span id="hx-val">1.35</span></label>
<label>cycle <input id="cyc-slider" type="range" min="30" max="3600" step="5" value="{info["cycle_time_s"]:.0f}"><span id="cyc-val"></span></label></div>
{cycle_schematic_svg()}
<h2>Isotherms — D–A operating window (live)</h2>
<p class="dim">Uptake vs vapour pressure at bed adsorption / desorption temperatures. Dots mark the cycle operating points.</p>
{chart_svg("iso", "q*(T, P) — Dubinin–Astakhov", "P [Pa, log]", "q* [kg/kg]")}
{sched_sec}
<h2>Reference ranking — top {len(rows)} (equilibrium, hx = 1.35)</h2>
<p class="dim">Click a row to load it into the cycle panel. ⚠️ = out-of-window (extrapolated). Ranks never compare across profiles.</p>
<table id="rank-table"><tr><th>rank</th><th>material</th><th>COP</th><th>SCP</th><th>score</th><th>flag</th></tr>
{rank_rows}</table>
<p class="dim">Model proposes; simulator disposes — ML predictions are judged by this physics, never by held-out ML metrics alone.</p>
<script>
{JS_PHYSICS}
{js_data}
{viz_design.JS_DESIGN}
{JS_PAGE}
</script>
</body>
</html>"""


def _pde_links(out_dir: Path) -> str:
    """Index entries for generated PDE pages (globbed, so either generator
    can rebuild the index without ordering constraints)."""
    import re  # noqa: PLC0415

    items = []
    for path in sorted(out_dir.glob("pde-*.html")) + sorted(out_dir.glob("absorption.html")):
        m = re.search(r"<title>(.*?)</title>", path.read_text(encoding="utf-8"))
        title = m.group(1) if m else path.stem
        items.append(f'<li><a href="{path.name}"><b>{path.stem}</b> — {title}</a></li>')
    if not items:
        return ""
    return ("<h2>Frozen-PDE applications (recover-and-run)</h2>"
            '<p class="dim">Field replays from the frozen diffheat zoo '
            "(removed from the tree at f36e929, run from git history) plus the live absorption core.</p>"
            f"<ul>{''.join(items)}</ul>")


def build_index(out_dir: str | Path, profiles: list[str] | None = None) -> Path:
    """(Re)build ``index.html``: application profiles + any PDE pages present."""
    from .profiles import get_profile  # noqa: PLC0415

    out = Path(out_dir)
    profiles = list(profiles or APP_PROFILES)
    items = []
    for name in profiles:
        if not (out / f"{name}.html").exists():
            continue
        p = get_profile(name)
        items.append(f'<li><a href="{name}.html"><b>{name}</b> — {p.description}</a><br>'
                     f'<span class="dim">T_evap {p.t_evap_c:.0f} °C · T_cond {p.t_cond_c:.0f} °C · '
                     f'T_des {p.t_des_c:.0f} °C · half-cycle {p.cycle_time_s:.0f} s</span></li>')
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cooling applications — interactive physics pages</title>
<style>body {{ background:{_BG}; color:{_FG}; font-family:system-ui,sans-serif; max-width:760px; margin:0 auto; padding:16px; }}
h2 {{ font-size:16px; margin-top:28px; }}
.dim {{ color:{_AX}; font-size:12px; }} a {{ color:{_CYN}; }} li {{ margin:10px 0; }}</style></head>
<body><h1>Adsorption-cooling applications</h1>
<p class="dim">One physics-based interactive page per application profile: animated
bed cross-section + two-bed machine replays (solver output), live equilibrium
cycle, D–A window, T2 reference ranking. Offline single files — no external dependencies.</p>
<ul>{"".join(items)}</ul>{_pde_links(out)}</body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")
    return out / "index.html"


def _index_html(profiles: list[str]) -> str:
    from .profiles import get_profile  # noqa: PLC0415

    items = []
    for name in profiles:
        p = get_profile(name)
        items.append(f'<li><a href="{name}.html"><b>{name}</b> — {p.description}</a><br>'
                     f'<span class="dim">T_evap {p.t_evap_c:.0f} °C · T_cond {p.t_cond_c:.0f} °C · '
                     f'T_des {p.t_des_c:.0f} °C · half-cycle {p.cycle_time_s:.0f} s</span></li>')
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Adsorption-cooling applications — interactive physics pages</title>
<style>body {{ background:{_BG}; color:{_FG}; font-family:system-ui,sans-serif; max-width:760px; margin:0 auto; padding:16px; }}
.dim {{ color:{_AX}; font-size:12px; }} a {{ color:{_CYN}; }} li {{ margin:10px 0; }}</style></head>
<body><h1>Adsorption-cooling applications</h1>
<p class="dim\">One physics-based interactive page per application profile: animated
bed cross-section + two-bed machine replays (solver output), live equilibrium
cycle, D–A window, T2 reference ranking. Offline single files — no external dependencies.</p>
<ul>{"".join(items)}</ul></body></html>"""


def write_app_pages(out_dir: str | Path = "docs/apps",
                    profiles: list[str] | None = None,
                    k: int = 10, with_fields: bool = True) -> dict[str, Path]:
    """Generate one HTML page per profile + index. Returns ``{name: path}``."""
    from . import viz_design  # noqa: PLC0415
    from .profiles import get_profile  # noqa: PLC0415

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    profiles = list(profiles or APP_PROFILES)
    written: dict[str, Path] = {}
    anchor = anchor_params()
    for name in profiles:
        get_profile(name)  # fail loudly on unknown keys
        info = profile_info(name)
        rows = sweep_top(STEADY_SWEEP[name], k=k)
        field = bed_field_sample(name) if with_fields else None
        ms = (viz_design.machine_sim_sample(name, STEADY_SWEEP[name])
              if with_fields else None)
        sched = schedule_sample(name)
        prov = {"date": datetime.date.today().isoformat(),
                "note": f"T2 sweep {len(rows)} rows + Bed1D/TwoBed replays"
                        + (" + coolant schedule" if sched else "")}
        path = out / f"{name}.html"
        path.write_text(_page_html(info, rows, anchor, field, ms, sched, prov),
                        encoding="utf-8")
        written[name] = path
    (out / "index.html").write_text(_index_html(profiles), encoding="utf-8")
    build_index(out, profiles)
    written["index"] = out / "index.html"
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate per-application interactive physics pages.")
    ap.add_argument("--out", type=Path, default=Path("docs/apps"))
    ap.add_argument("--profiles", nargs="*", default=list(APP_PROFILES))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-fields", action="store_true",
                    help="skip the solver replays (fast preview)")
    args = ap.parse_args()
    t0 = time.time()
    written = write_app_pages(args.out, profiles=args.profiles, k=args.k,
                              with_fields=not args.no_fields)
    for name, path in written.items():
        print(f"Wrote {path} ({name})")
    print(f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


__all__ = ["APP_PROFILES", "anchor_params", "bed_field_sample", "cycle_metrics",
           "cycle_schematic_svg", "chart_svg", "da_uptake", "h_fg_j_kg",
           "heatmap_svg", "profile_info", "profile_plot_svg", "psat_pa",
           "schedule_sample", "sweep_top", "write_app_pages"]
