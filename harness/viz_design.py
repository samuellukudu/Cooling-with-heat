"""Geometry-based simulation views for the per-application pages.

``viz_apps.py`` owns page assembly and equilibrium content; this module owns
the two animated *simulation* sections:

- **bed cross-section** — fluid channel + metal wall + the 8 adsorbent
  cells + vapor pipe, driven per-snapshot by a ``Bed1D`` solver replay
  (cell temperatures, uptakes, phase, vapor flow, fluid temperature,
  cumulative books);
- **two-bed machine** — twin cross-sections + 4 vapor valves + evaporator /
  condenser vessels + heat-recovery link, driven by a ``TwoBed`` system
  replay (per-bed phases and flows, recovery power, integrated duty).

Sampling helpers return JSON-serializable dicts; SVG builders emit static
skeletons with stable element ids; ``JS_DESIGN`` animates them from the
embedded snapshots. No matplotlib, no network, no new dependencies.
Wall (x=0) is left and vapor-side right, matching ``harness/viz.py``.
"""

from __future__ import annotations

# Dark theme matches harness/viz.py.
_BG, _FG, _AX, _AMB, _GRN, _CYN, _PNL, _SPN, _RED, _WALL = (
    "#0f1115", "#e2e8f0", "#94a3b8", "#f59e0b", "#22c55e", "#06b6d4",
    "#161a21", "#1e222a", "#f87171", "#6b7280")

CSS_ANIM = """
.flow { stroke-dasharray: 7 6; animation: dashfwd 0.9s linear infinite; }
.flow-rev { stroke-dasharray: 7 6; animation: dashrev 0.9s linear infinite; }
@keyframes dashfwd { to { stroke-dashoffset: -26; } }
@keyframes dashrev { to { stroke-dashoffset: 26; } }
"""


def _svg_open(w: int, h: int) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'style="background:{_BG};border:1px solid {_SPN};border-radius:8px">')


# --------------------------------------------------------------------------
# Sampling (harness imports stay inside functions; import-light like viz.py).
# --------------------------------------------------------------------------

def _align(n_series: int, n_snap: int) -> list[int]:
    """Fractional index map: snapshot i -> nearest series row (both uniform)."""
    if n_snap <= 1:
        return [0]
    return [round(i * (n_series - 1) / (n_snap - 1)) for i in range(n_snap)]


def bed_sim_sample(profile: str, *, n_cells: int = 8, n_cycles: int = 2,
                   n_snap: int = 61) -> dict:
    """Bed1D replay + per-snapshot channels for the cross-section view."""
    import numpy as np  # noqa: PLC0415

    from . import make  # noqa: PLC0415

    bed = make("Bed1D-v0", material="anchor:Silica gel RD",
               profile=profile, n_cells=n_cells, n_cycles=n_cycles)
    tr = bed.rollout(collect_fields=True)
    f, s = tr.fields, tr.series
    t = np.asarray(f["t_s"], dtype=float)
    idx = np.unique(np.linspace(0, len(t) - 1, min(n_snap, len(t))).astype(int))
    n_ser = len(np.asarray(s["phase"]))
    amap = _align(n_ser, len(idx))
    phase = np.asarray(s["phase"], dtype=float)
    dq_c = np.asarray(s["dq_cool_j_kg"], dtype=float)
    dq_i = np.asarray(s["dq_in_j_kg"], dtype=float)
    t_fl = np.asarray(s["t_fluid_k"], dtype=float)
    vap = np.asarray(s["vapor_p_Pa"], dtype=float)
    qcool_cum = np.cumsum(dq_c)
    qin_cum = np.cumsum(dq_i)
    T = np.asarray(f["T_K"], dtype=float)[idx] - 273.15
    Q = np.asarray(f["q_kg_kg"], dtype=float)[idx]
    return {
        "t_s": [round(float(t[i]), 3) for i in idx],
        "x_mm": [round(float(v) * 1000.0, 4)
                 for v in np.asarray(f["x_m"], dtype=float)],
        "T_C": [[round(float(v), 2) for v in row] for row in T],
        "q": [[round(float(v), 5) for v in row] for row in Q],
        "phase": [int(phase[j] > 0.5) for j in amap],
        "dq_cool": [round(float(dq_c[j]), 2) for j in amap],
        "dq_in": [round(float(dq_i[j]), 2) for j in amap],
        "t_fluid_c": [round(float(t_fl[j]) - 273.15, 2) for j in amap],
        "vapor_kpa": [round(float(vap[j]) / 1000.0, 3) for j in amap],
        "Qcool_cum": [round(float(qcool_cum[j]), 1) for j in amap],
        "Qin_cum": [round(float(qin_cum[j]), 1) for j in amap],
        "qmax": round(float(Q.max()), 5),
        "dqmax": round(max(float(np.abs(dq_c).max()), 1e-9), 3),
        "COP": round(float(tr.summary["COP"]), 4),
        "SCP_W_kg": round(float(tr.summary["SCP_W_kg"]), 2),
        "n_cells": n_cells, "n_cycles": n_cycles,
    }


def machine_sim_sample(profile: str, steady: str, *, n_cells: int = 8,
                       n_cycles: int = 2, n_snap: int = 61,
                       t_rec_s: float = 20.0, ua_w_m2_k: float = 50.0) -> dict:
    """TwoBed replay + per-snapshot channels for the machine view.

    Small heat-recovery window (``t_rec_s``/``ua_w_m2_k``); the measured
    ``recovery_gain`` is reported whatever its sign — on transient-bound
    profile/material pairs it is negative (DESIGN §4.3), which the page
    states rather than hides.
    """
    import numpy as np  # noqa: PLC0415

    from . import make  # noqa: PLC0415
    from .physics.thermo import water_h_fg_j_kg  # noqa: PLC0415

    import jax.numpy as jnp  # noqa: PLC0415

    m = make("TwoBed-v0", material="anchor:Silica gel RD",
             profile=steady, n_cells=n_cells, n_cycles=n_cycles)
    tr = m.rollout(design={"recovery_ua_w_m2_k": ua_w_m2_k},
                   controls={"t_rec_s": t_rec_s}, collect_fields=True)
    f, s = tr.fields, tr.series
    t_all = np.asarray(f["t_s"], dtype=float)
    idx = np.unique(np.linspace(0, len(t_all) - 1,
                                min(n_snap, len(t_all))).astype(int))
    n_ser = len(np.asarray(s["t_abs_s"]))
    amap = _align(n_ser, len(idx))

    def col(key: str) -> np.ndarray:
        return np.asarray(s[key], dtype=float)

    t_ser, sys_ph = col("t_abs_s"), col("sys_phase")
    A_T = np.asarray(f["A_T_K"], dtype=float)[idx] - 273.15
    B_T = np.asarray(f["B_T_K"], dtype=float)[idx] - 273.15
    A_q = np.asarray(f["A_q_kg_kg"], dtype=float)[idx]
    B_q = np.asarray(f["B_q_kg_kg"], dtype=float)[idx]
    A_ph = col("A_phase")
    A_dq = col("A_dq_cool_j_kg")
    B_dq = col("B_dq_cool_j_kg")
    q_rec = col("q_rec_w_m2")
    h_fg = float(water_h_fg_j_kg(jnp.asarray(m.profile.t_evap_c + 273.15)))
    m_s = 600.0 * float(m.design_space.defaults["A_L_m"])  # kg sorbent / m² wall
    vap_out = np.cumsum(np.maximum(A_dq, 0.0) + np.maximum(B_dq, 0.0)) / h_fg * m_s
    span = max(float(vap_out.max()), 1e-9)
    qmax = max(float(A_q.max()), float(B_q.max()))
    dqmax = max(float(np.abs(A_dq).max()), float(np.abs(B_dq).max()), 1e-9)
    recmax = max(float(np.abs(q_rec).max()), 1e-9)
    # Phase-track segments from the full series (flip instants as fractions).
    flips = [0] + [i + 1 for i in range(len(sys_ph) - 1)
                   if sys_ph[i + 1] != sys_ph[i]] + [len(sys_ph)]
    total = max(float(t_ser[-1]), 1e-9)
    segs = [[round(float(t_ser[flips[k]]) / total, 4),
             round(float(t_ser[flips[k + 1] - 1]) / total, 4),
             int(sys_ph[flips[k]])] for k in range(len(flips) - 1)]
    q_rec_tot = float(np.trapezoid(np.abs(q_rec), t_ser)) / 1000.0  # kJ/m²
    return {
        "t_s": [round(float(t_all[i]), 2) for i in idx],
        "x_mm": [round(float(v) * 1000.0, 4)
                 for v in np.asarray(f["A_x_m"], dtype=float)],
        "A_T_C": [[round(float(v), 1) for v in row] for row in A_T],
        "B_T_C": [[round(float(v), 1) for v in row] for row in B_T],
        "A_q": [[round(float(v), 4) for v in row] for row in A_q],
        "B_q": [[round(float(v), 4) for v in row] for row in B_q],
        "sys_phase": [int(sys_ph[j] > 0.5) for j in amap],
        "A_phase": [int(A_ph[j] > 0.5) for j in amap],
        "B_phase": [int(col("B_phase")[j] > 0.5) for j in amap],
        "A_dq": [round(float(A_dq[j]), 2) for j in amap],
        "B_dq": [round(float(B_dq[j]), 2) for j in amap],
        "q_rec": [round(float(q_rec[j]), 1) for j in amap],
        "A_fluid_c": [round(float(col("A_t_fluid_k")[j]) - 273.15, 1) for j in amap],
        "B_fluid_c": [round(float(col("B_t_fluid_k")[j]) - 273.15, 1) for j in amap],
        "A_vapor_kpa": [round(float(col("A_vapor_p_Pa")[j]) / 1000.0, 2) for j in amap],
        "B_vapor_kpa": [round(float(col("B_vapor_p_Pa")[j]) / 1000.0, 2) for j in amap],
        "evap_lvl": [round(1.0 - 0.4 * float(v) / span, 4) for v in vap_out[amap]],
        "cond_lvl": [round(0.3 + 0.4 * float(v) / span, 4) for v in vap_out[amap]],
        "Qcool_cum": [round(float(col("A_Q_cool_cum_J_kg")[j]
                                  + col("B_Q_cool_cum_J_kg")[j]), 1) for j in amap],
        "segs": segs,
        "qmax": round(qmax, 4), "dqmax": round(dqmax, 3),
        "recmax": round(recmax, 1),
        "COP": round(float(tr.summary["COP"]), 4),
        "SCP_W_kg": round(float(tr.summary["SCP_W_kg"]), 1),
        "gain": round(float(tr.summary.get("recovery_gain", 0.0)), 4),
        "Q_rec_kJ_m2": round(q_rec_tot, 2),
        "t_rec_s": t_rec_s, "ua": ua_w_m2_k,
        "n_cells": n_cells, "n_cycles": n_cycles,
    }


# --------------------------------------------------------------------------
# SVG skeletons (static geometry; JS fills fields per snapshot).
# --------------------------------------------------------------------------

def bed_xsection_svg(pfx: str = "bx", n_cells: int = 8) -> str:
    """Bed slice: fluid channel | wall | adsorbent cells | vapor pipe."""
    s = [_svg_open(720, 320)]
    s.append(f'<text x="16" y="24" fill="{_FG}" font-size="13" font-weight="bold">'
             'Bed cross-section — live solver replay '
             f'<tspan id="{pfx}-badge" font-size="11">–</tspan></text>')
    # Fluid channel + wall.
    s.append(f'<rect id="{pfx}-fluid" x="20" y="60" width="44" height="180" rx="4" '
             f'fill="#334155" stroke="{_SPN}"/>')
    for k in range(3):
        y = 100 + k * 50
        s.append(f'<polyline class="flow" points="26,{y} 38,{y - 10} 26,{y - 20}" '
                 f'fill="none" stroke="#e2e8f0" stroke-width="2" opacity="0.7"/>')
    s.append(f'<text id="{pfx}-flab" x="42" y="255" fill="{_AX}" font-size="10" '
             f'text-anchor="middle">–</text>')
    s.append(f'<rect x="64" y="60" width="14" height="180" fill="{_WALL}" '
             f'stroke="{_SPN}"/><text x="71" y="255" fill="{_AX}" font-size="10" '
             f'text-anchor="middle">wall</text>')
    # Adsorbent cells + uptake dots.
    cw = 400.0 / n_cells
    for i in range(n_cells):
        x = 78 + i * cw
        s.append(f'<rect id="{pfx}-cell-{i}" x="{x:.1f}" y="60" width="{cw - 2:.1f}" '
                 f'height="180" fill="#141428" stroke="{_SPN}"/>')
        s.append(f'<g id="{pfx}-dots-{i}">')
        for d in range(6):
            dx = x + 12 + (d % 2) * (cw / 2 - 6)
            dy = 95 + (d // 2) * 55
            s.append(f'<circle id="{pfx}-qd-{i}-{d}" cx="{dx:.1f}" cy="{dy}" r="6" '
                     f'fill="#67e8f9" opacity="0.9" visibility="hidden"/>')
        s.append('</g>')
    s.append(f'<text x="278" y="255" fill="{_AX}" font-size="10" text-anchor="middle">'
             f'adsorbent (T-colored cells · cyan dots = adsorbed water)</text>')
    # Vapor pipe + drifting dots + connection label.
    s.append(f'<rect x="478" y="130" width="182" height="40" rx="20" fill="{_PNL}" '
             f'stroke="{_SPN}"/>')
    s.append(f'<line id="{pfx}-flow" x1="486" y1="150" x2="652" y2="150" '
             f'stroke="{_CYN}" stroke-width="2.5" class="flow"/>')
    s.append(f'<g id="{pfx}-vapor">')
    for d in range(12):
        s.append(f'<circle id="{pfx}-vd-{d}" cx="500" cy="150" r="3.5" '
                 f'fill="#a5f3fc" visibility="hidden"/>')
    s.append('</g>')
    s.append(f'<text id="{pfx}-conn" x="569" y="120" fill="{_CYN}" font-size="11" '
             f'text-anchor="middle">–</text>')
    # Cumulative books.
    s.append(f'<text x="16" y="292" fill="{_AX}" font-size="11">cooling '
             f'<tspan id="{pfx}-qcool-lab">–</tspan></text>')
    s.append(f'<rect x="150" y="282" width="200" height="10" fill="{_PNL}" '
             f'stroke="{_SPN}"/><rect id="{pfx}-qcool" x="150" y="282" width="0" '
             f'height="10" fill="{_CYN}"/>')
    s.append(f'<text x="366" y="292" fill="{_AX}" font-size="11">heat in '
             f'<tspan id="{pfx}-qin-lab">–</tspan></text>')
    s.append(f'<rect x="470" y="282" width="200" height="10" fill="{_PNL}" '
             f'stroke="{_SPN}"/><rect id="{pfx}-qin" x="470" y="282" width="0" '
             f'height="10" fill="{_AMB}"/>')
    s.append(f'<text id="{pfx}-t" x="704" y="292" fill="{_FG}" font-size="11" '
             f'text-anchor="end">–</text>')
    s.append('</svg>')
    return "\n".join(s)


def machine_svg(n_cells: int = 8) -> str:
    """Counter-phase machine: twin beds, valves, vessels, recovery link."""
    s = [_svg_open(720, 480)]
    s.append(f'<text x="16" y="24" fill="{_FG}" font-size="13" font-weight="bold">'
             'Two-bed machine — live system replay '
             '<tspan id="m-t" font-size="11">–</tspan></text>')
    beds = (("A", 48, "mA"), ("B", 432, "mB"))
    for tag, x0, p in beds:
        s.append(f'<text x="{x0}" y="48" fill="{_FG}" font-size="12">Bed {tag} '
                 f'<tspan id="{p}-badge">–</tspan></text>')
        s.append(f'<rect id="{p}-fluid" x="{x0 - 32}" y="56" width="26" height="114" '
                 f'rx="4" fill="#334155" stroke="{_SPN}"/>')
        s.append(f'<text id="{p}-flab" x="{x0 - 19}" y="182" fill="{_AX}" '
                 f'font-size="9" text-anchor="middle">–</text>')
        s.append(f'<rect x="{x0 - 6}" y="56" width="8" height="114" fill="{_WALL}" '
                 f'stroke="{_SPN}"/>')
        cw = 240.0 / n_cells
        for i in range(n_cells):
            x = x0 + 2 + i * cw
            s.append(f'<rect id="{p}-cell-{i}" x="{x:.1f}" y="56" width="{cw - 2:.1f}" '
                     f'height="114" fill="#141428" stroke="{_SPN}"/>')
        s.append(f'<rect x="{x0}" y="176" width="244" height="8" fill="{_PNL}" '
                 f'stroke="{_SPN}"/><rect id="{p}-qbar" x="{x0}" y="176" width="0" '
                 f'height="8" fill="{_GRN}"/>')
        s.append(f'<text id="{p}-vlab" x="{x0 + 122}" y="198" fill="{_AX}" '
                 f'font-size="9" text-anchor="middle">–</text>')
    # Heat-recovery link between the vapor columns.
    s.append(f'<line id="m-rec" x1="300" y1="210" x2="420" y2="210" stroke="{_AMB}" '
             f'stroke-width="4" opacity="0.15"/>')
    s.append(f'<text id="m-rec-lab" x="360" y="202" fill="{_AMB}" font-size="10" '
             f'text-anchor="middle">heat recovery —</text>')
    # Phase track (static segments + playhead).
    s.append(f'<g id="m-track"></g>')
    s.append(f'<line id="m-head" x1="40" y1="216" x2="40" y2="240" stroke="#fff" '
             f'stroke-width="1.5"/>')
    # Vapor manifold + valves + vessels.
    for bx, side in ((320, "A"), (400, "B")):
        s.append(f'<line x1="{bx}" y1="170" x2="{bx}" y2="252" stroke="{_SPN}" '
                 f'stroke-width="6"/>')
        s.append(f'<line x1="{bx}" y1="262" x2="140" y2="300" stroke="{_SPN}" '
                 f'stroke-width="5"/>')
        s.append(f'<line x1="{bx}" y1="262" x2="580" y2="300" stroke="{_SPN}" '
                 f'stroke-width="5"/>')
        for vx, key in ((232, f"mV-{side}E"), (488, f"mV-{side}C")):
            vy = 281
            s.append(f'<g id="{key}">'
                     f'<polygon points="{vx - 11},{vy - 8} {vx - 11},{vy + 8} {vx},{vy}" '
                     f'fill="{_PNL}" stroke="{_AX}" stroke-width="1.5"/>'
                     f'<polygon points="{vx + 11},{vy - 8} {vx + 11},{vy + 8} {vx},{vy}" '
                     f'fill="{_PNL}" stroke="{_AX}" stroke-width="1.5"/>'
                     f'<line id="{key}-h" x1="{vx}" y1="{vy}" x2="{vx}" y2="{vy - 14}" '
                     f'stroke="{_GRN}" stroke-width="2.5"/></g>')
            s.append(f'<line id="mF-{side}{key[4]}" x1="{vx - 30}" y1="{vy}" '
                     f'x2="{vx + 30}" y2="{vy}" stroke="{_CYN}" stroke-width="2" '
                     f'class="flow" visibility="hidden"/>')
    for vx, name, lvl, lab in ((140, "evaporator", "m-evap-lvl", "m-evap-lab"),
                               (580, "condenser", "m-cond-lvl", "m-cond-lab")):
        s.append(f'<rect x="{vx - 80}" y="300" width="160" height="110" rx="8" '
                 f'fill="{_PNL}" stroke="{_SPN}"/>')
        s.append(f'<text x="{vx}" y="322" fill="{_FG}" font-size="11" '
                 f'text-anchor="middle">{name} <tspan id="{lab}">–</tspan></text>')
        s.append(f'<rect id="{lvl}" x="{vx - 70}" y="340" width="140" height="60" '
                 f'fill="#155e75"/>')
    s.append(f'<text x="16" y="440" fill="{_AX}" font-size="11">system COP '
             f'<tspan id="m-cop">–</tspan> · SCP <tspan id="m-scp">–</tspan> · '
             f'recovery gain <tspan id="m-gain">–</tspan> '
             f'<tspan id="m-gain-note"></tspan></text>')
    s.append(f'<text x="16" y="462" fill="{_AX}" font-size="11">cooling delivered '
             f'<tspan id="m-qcool-lab">–</tspan></text>')
    s.append(f'<rect x="170" y="452" width="300" height="10" fill="{_PNL}" '
             f'stroke="{_SPN}"/><rect id="m-qcool" x="170" y="452" width="0" '
             f'height="10" fill="{_CYN}"/>')
    s.append('</svg>')
    return "\n".join(s)


# --------------------------------------------------------------------------
# Animation (colormaps + per-frame renderers; data consts BX / MS embedded).
# --------------------------------------------------------------------------

JS_DESIGN = """
function inferno(f) {
  f = Math.min(Math.max(f, 0), 1);
  const stops = [[13,13,31],[59,15,112],[140,29,64],[199,92,20],[249,200,70]];
  const x = f * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2), t = x - i;
  const c = stops[i].map((v, k) => Math.round(v + t * (stops[i + 1][k] - v)));
  return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
}
function fluidCol(tC, lo, hi) {
  const f = Math.min(Math.max((tC - lo) / Math.max(hi - lo, 1e-9), 0), 1);
  const cold = [56,189,248], mid = [100,116,139], hot = [249,115,22];
  const a = f < 0.5 ? [cold, mid, f * 2] : [mid, hot, (f - 0.5) * 2];
  const c = a[0].map((v, k) => Math.round(v + a[2] * (a[1][k] - v)));
  return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
}
function valve(key, open) {
  const g = document.getElementById(key);
  const h = document.getElementById(key + "-h");
  const y = parseFloat(g.querySelector("line").getAttribute("y1"));
  h.setAttribute("y2", open ? y - 14 : y);
  h.setAttribute("stroke", open ? "#22c55e" : "#f87171");
  h.setAttribute("transform", open ? "" : "rotate(80 " + g.querySelector("line").getAttribute("x1") + " " + y + ")");
}
function drawBedX(i) {
  const B = BX, T = B.T_C[i], Q = B.q[i], ads = B.phase[i] === 0;
  const tLo = Math.min(...B.T_C.flat()), tHi = Math.max(...B.T_C.flat());
  for (let k = 0; k < T.length; k++) {
    document.getElementById("bx-cell-" + k).setAttribute("fill", inferno((T[k] - tLo) / Math.max(tHi - tLo, 1e-9)));
    const n = Math.round(6 * Q[k] / B.qmax);
    for (let d = 0; d < 6; d++)
      document.getElementById("bx-qd-" + k + "-" + d).setAttribute("visibility", d < n ? "visible" : "hidden");
  }
  document.getElementById("bx-fluid").setAttribute("fill", fluidCol(B.t_fluid_c[i], PROFILE.t_fluid_min_c, PROFILE.t_fluid_max_c));
  document.getElementById("bx-flab").textContent = B.t_fluid_c[i].toFixed(0) + " °C HTF";
  const badge = document.getElementById("bx-badge");
  badge.textContent = ads ? "● ADSORBING" : "● DESORBING";
  badge.setAttribute("fill", ads ? "#06b6d4" : "#f59e0b");
  const conn = document.getElementById("bx-conn");
  conn.textContent = ads ? "← evaporator · " + B.vapor_kpa[i].toFixed(2) + " kPa" : "→ condenser · " + B.vapor_kpa[i].toFixed(2) + " kPa";
  const fl = document.getElementById("bx-flow");
  fl.setAttribute("class", ads ? "flow-rev" : "flow");
  const n = Math.round(12 * Math.abs(B.dq_cool[i]) / B.dqmax);
  for (let d = 0; d < 12; d++) {
    const el = document.getElementById("bx-vd-" + d);
    if (d < n) {
      const x = ads ? 648 - d * 13 : 492 + d * 13;
      el.setAttribute("cx", x); el.setAttribute("cy", 150 + ((d * 37) % 5 - 2) * 6);
      el.setAttribute("visibility", "visible");
    } else el.setAttribute("visibility", "hidden");
  }
  const qMax = Math.max(...B.Qcool_cum), iMax = Math.max(...B.Qin_cum);
  document.getElementById("bx-qcool").setAttribute("width", 200 * B.Qcool_cum[i] / Math.max(qMax, 1e-9));
  document.getElementById("bx-qin").setAttribute("width", 200 * B.Qin_cum[i] / Math.max(iMax, 1e-9));
  document.getElementById("bx-qcool-lab").textContent = (B.Qcool_cum[i] / 1000).toFixed(1) + " kJ/kg";
  document.getElementById("bx-qin-lab").textContent = (B.Qin_cum[i] / 1000).toFixed(1) + " kJ/kg";
  document.getElementById("bx-t").textContent = "t = " + B.t_s[i].toFixed(1) + " s";
}
function bedCells(p, T, Q, M) {
  const tLo = Math.min(...M.A_T_C.flat(), ...M.B_T_C.flat());
  const tHi = Math.max(...M.A_T_C.flat(), ...M.B_T_C.flat());
  for (let k = 0; k < T.length; k++)
    document.getElementById(p + "-cell-" + k).setAttribute("fill", inferno((T[k] - tLo) / Math.max(tHi - tLo, 1e-9)));
  const qm = Q.reduce((a, b) => a + b, 0) / Q.length;
  document.getElementById(p + "-qbar").setAttribute("width", 244 * qm / M.qmax);
}
function drawMachine(i) {
  const M = MS, adsA = M.A_phase[i] === 0, adsB = M.B_phase[i] === 0;
  bedCells("mA", M.A_T_C[i], M.A_q[i], M);
  bedCells("mB", M.B_T_C[i], M.B_q[i], M);
  const bA = document.getElementById("mA-badge"), bB = document.getElementById("mB-badge");
  bA.textContent = adsA ? "● ADSORB" : "● DESORB"; bA.setAttribute("fill", adsA ? "#06b6d4" : "#f59e0b");
  bB.textContent = adsB ? "● ADSORB" : "● DESORB"; bB.setAttribute("fill", adsB ? "#06b6d4" : "#f59e0b");
  document.getElementById("mA-fluid").setAttribute("fill", fluidCol(M.A_fluid_c[i], PROFILE.t_fluid_min_c, PROFILE.t_fluid_max_c));
  document.getElementById("mB-fluid").setAttribute("fill", fluidCol(M.B_fluid_c[i], PROFILE.t_fluid_min_c, PROFILE.t_fluid_max_c));
  document.getElementById("mA-flab").textContent = M.A_fluid_c[i].toFixed(0) + " °C";
  document.getElementById("mB-flab").textContent = M.B_fluid_c[i].toFixed(0) + " °C";
  document.getElementById("mA-vlab").textContent = "P " + M.A_vapor_kpa[i].toFixed(2) + " kPa · q̄ " + (M.A_q[i].reduce((a,b)=>a+b,0)/M.A_q[i].length).toFixed(3);
  document.getElementById("mB-vlab").textContent = "P " + M.B_vapor_kpa[i].toFixed(2) + " kPa · q̄ " + (M.B_q[i].reduce((a,b)=>a+b,0)/M.B_q[i].length).toFixed(3);
  valve("mV-AE", adsA); valve("mV-AC", !adsA); valve("mV-BE", adsB); valve("mV-BC", !adsB);
  const show = (id, on, rev) => {
    const el = document.getElementById(id);
    el.setAttribute("visibility", on ? "visible" : "hidden");
    if (on) el.setAttribute("class", rev ? "flow-rev" : "flow");
  };
  const fA = Math.abs(M.A_dq[i]) / M.dqmax, fB = Math.abs(M.B_dq[i]) / M.dqmax;
  show("mF-AE", adsA && fA > 0.02, true); show("mF-AC", !adsA && fA > 0.02, false);
  show("mF-BE", adsB && fB > 0.02, true); show("mF-BC", !adsB && fB > 0.02, false);
  const rec = document.getElementById("m-rec");
  rec.setAttribute("opacity", 0.15 + 0.85 * Math.abs(M.q_rec[i]) / M.recmax);
  rec.setAttribute("stroke-width", 2 + 5 * Math.abs(M.q_rec[i]) / M.recmax);
  document.getElementById("m-rec-lab").textContent = "heat recovery " + Math.abs(M.q_rec[i]).toFixed(0) + " W/m²";
  const setLvl = (id, f) => {
    const el = document.getElementById(id), H = 60, y0 = 340;
    el.setAttribute("y", y0 + H * (1 - f)); el.setAttribute("height", H * f);
  };
  setLvl("m-evap-lvl", M.evap_lvl[i]); setLvl("m-cond-lvl", M.cond_lvl[i]);
  document.getElementById("m-evap-lab").textContent = PROFILE.t_evap_c.toFixed(0) + " °C · " + Math.round(100 * M.evap_lvl[i]) + "%";
  document.getElementById("m-cond-lab").textContent = PROFILE.t_cond_c.toFixed(0) + " °C · " + Math.round(100 * M.cond_lvl[i]) + "%";
  document.getElementById("m-cop").textContent = M.COP.toFixed(3);
  document.getElementById("m-scp").textContent = M.SCP_W_kg.toFixed(0) + " W/kg";
  const g = document.getElementById("m-gain");
  g.textContent = (M.gain >= 0 ? "+" : "") + M.gain.toFixed(4);
  g.setAttribute("fill", M.gain >= 0 ? "#22c55e" : "#f87171");
  document.getElementById("m-gain-note").textContent = M.gain >= 0 ? "(recovery pays here)" : "(transient-bound: recovery hurts here — DESIGN §4.3)";
  const qcMax = Math.max(...M.Qcool_cum);
  document.getElementById("m-qcool").setAttribute("width", 300 * M.Qcool_cum[i] / Math.max(qcMax, 1e-9));
  document.getElementById("m-qcool-lab").textContent = (M.Qcool_cum[i] / 1000).toFixed(1) + " kJ/kg";
  const head = document.getElementById("m-head");
  const x = 40 + 640 * (M.t_s[i] / M.t_s[M.t_s.length - 1]);
  head.setAttribute("x1", x); head.setAttribute("x2", x);
  document.getElementById("m-t").textContent = "t = " + M.t_s[i].toFixed(1) + " s · σ=" + M.sys_phase[i] + (M.sys_phase[i] === 0 ? " (A adsorbs)" : " (B adsorbs)");
}
function initDesign() {
  const g = document.getElementById("m-track");
  if (g && typeof MS !== "undefined") {
    MS.segs.forEach(([f0, f1, s]) => {
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", 40 + 640 * f0); r.setAttribute("y", 224);
      r.setAttribute("width", Math.max(640 * (f1 - f0), 2)); r.setAttribute("height", 10);
      r.setAttribute("fill", s === 0 ? "#06b6d4" : "#f59e0b"); r.setAttribute("opacity", "0.55");
      g.appendChild(r);
    });
  }
  const bs = document.getElementById("bx-slider"), ms = document.getElementById("m-slider");
  if (bs) bs.addEventListener("input", (e) => drawBedX(parseInt(e.target.value, 10)));
  if (ms) ms.addEventListener("input", (e) => drawMachine(parseInt(e.target.value, 10)));
  const wire = (btn, slider, fn) => {
    let playing = false, timer = null;
    document.getElementById(btn).addEventListener("click", () => {
      playing = !playing; document.getElementById(btn).textContent = playing ? "⏸ pause" : "▶ play";
      if (playing) timer = setInterval(() => {
        const el = document.getElementById(slider), n = el.max * 1;
        el.value = (parseInt(el.value, 10) + 1) % (n + 1); fn(parseInt(el.value, 10));
      }, 140);
      else clearInterval(timer);
    });
  };
  if (document.getElementById("bx-play")) wire("bx-play", "bx-slider", drawBedX);
  if (document.getElementById("m-play")) wire("m-play", "m-slider", drawMachine);
  if (typeof BX !== "undefined") drawBedX(0);
  if (typeof MS !== "undefined") drawMachine(0);
}
"""

__all__ = ["CSS_ANIM", "JS_DESIGN", "bed_sim_sample", "bed_xsection_svg",
           "machine_sim_sample", "machine_svg"]
