"""Literature calibration of the 1-D bed (validation V4, DESIGN §12 H1.4).

Fits the bed-level effective uptake kinetics ``k_LDF`` (grain + vapour-
channel resistance lumped — Open Question 1 evidence) per material, the
wall film coefficient ``h_wall`` per bed type, and a per-rig metal
inventory factor ``hx`` to published experimental silica-gel/water
chiller data. The isotherm stays at the anchor values (``q_sat``,
``Q_st``, ``E``, ``n`` from ``adsorbent-ml/data/anchors.csv``).

Two rigs, two silica gels: Uyun+2009 ran Fuji Davison RD (~2 mm grains),
Sztekler+2021 ran KD gel (0.7-0.8 mm). RD parameters are fitted on the
Uyun standard point (70 C); KD parameters on the Sztekler optimum
(500 s); ``h`` is shared (same bed type: packed fin-tube, water-side).
Each rig's sweep is then a prediction of the other rig's fit.

Geometry is the model design point (L = 2 mm, N = 16, k_eff = 0.3 W/mK,
rho_s = 600 kg/m³): the calibration deliberately does NOT retune geometry,
so ``k_LDF``/``h`` are effective values that map a legacy packed fin-tube
bed onto the fast coated-lamella slab (the mapping caveat is documented
in ``benchmarks.md``).

Rig→model mapping (full derivation in ``benchmarks.md``):

- phase durations are the rig's *pure* adsorption/desorption intervals;
  isolated pre-heat/pre-cool intervals are excluded (the v1 valve has no
  isolated position);
- fluid temperatures are coolant / hot-water INLETs — the fitted ``h``
  absorbs the finite-flow coolant rise;
- evaporating / condensing temperatures are read from the rig's measured
  refrigerant-side traces where published, otherwise estimated from the
  secondary-loop temperatures minus a documented approach;
- legacy rigs report SCP per kg of *total* installed adsorbent while the
  model reports per kg of the *actively adsorbing* bed. For symmetric
  beds each kg adsorbs once per cycle ``t_ads + t_des``, so
  ``SCP_rig = SCP_model · t_ads/(t_ads + t_des)``. COP has no such
  convention gap.

Examples
--------
>>> from harness.calibration import (calibrate, UYUN2009_CASES,
...                                  SZTEKLER2021_CASES, FITTED_BY_RIG,
...                                  model_case_rows)
>>> result = calibrate(UYUN2009_CASES + SZTEKLER2021_CASES)  # refit
>>> rows = model_case_rows(
...     UYUN2009_CASES, {},
...     k_by_rig={r: v["k_ldf_s_1"] for r, v in FITTED_BY_RIG.items()},
...     h_by_rig={r: v["h_wall_w_m2_k"] for r, v in FITTED_BY_RIG.items()},
...     hx_by_rig={r: v["hx_mass_factor"] for r, v in FITTED_BY_RIG.items()})
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .materials import get_material
from .physics import bed1d
from .physics.thermo import CP_ADSORBENT, CP_LIQUID, da_uptake, water_h_fg_j_kg, water_sat_pressure_pa

BENCHMARK_SOURCES = {
    "uyun2009": (
        "Uyun, A.; Miyazaki, T.; Ueda, Y.; Akisawa, A. (2009). Experimental "
        "investigation of a three-bed adsorption refrigeration chiller "
        "employing an advanced mass recovery cycle. Energies 2(3), 531-544. "
        "doi:10.3390/en20300531 (open access). Single-stage cycle data: "
        "Figure 9 (COP/SCP vs heat-source temperature at 540 s phases), "
        "Table 2 (operating conditions, standard point at 70 C), Figure 4 "
        "(T_evap ~ 7-8 C, T_cond ~ 30 C traces); 16 kg silica gel per bed; "
        "COP measurement accuracy +-5.5 %."
    ),
    "sztekler2021": (
        "Sztekler, K.; Bartela, L.; Skroz, E.; Kowalczyk, T.; Maj, P.; "
        "Mika, L.; Stolecka, K. (2021). Optimisation of operation of "
        "adsorption chiller with desalination function. Energies 14(9), "
        "2668. doi:10.3390/en14092668 (open access). Cycle-time sweep "
        "(adsorption = desorption phase, 100-900 s at 80 C heat source): "
        "Figure 24; silica gel KD 700-800 um, 12 kg adsorbent, heating "
        "water 80 +- 3 C, chilled water ~16-18 C, cooling water 20-22 C."
    ),
}


@dataclass(frozen=True)
class BenchmarkCase:
    """One published operating point, mapped onto Bed1D boundary conditions.

    ``COP``/``SCP_W_kg`` are the *published* values (rig convention: SCP
    per kg total adsorbent); :attr:`duty` converts model SCP into the rig
    convention. ``rig`` selects the per-rig ``hx`` accounting factor (the
    rig's metal + water inventory — a rig property, not a material one).
    ``fit=False`` marks validation-only points (never in the calibration
    loss).
    """

    name: str
    source: str
    rig: str
    t_evap_c: float
    t_cond_c: float
    t_f_ads_c: float
    t_f_des_c: float
    t_ads_s: float
    t_des_s: float
    COP: float
    SCP_W_kg: float
    fit: bool = True
    t_cycle_rig_s: "float | None" = None  # full rig cycle incl. isolated
    # switching intervals (defaults to the model cycle t_ads + t_des)

    @property
    def duty(self) -> float:
        """Model SCP → rig SCP factor: the rig reports cooling averaged
        over its FULL cycle (including isolated switching), the model
        over its own back-to-back cycle."""
        t_cycle = self.t_cycle_rig_s if self.t_cycle_rig_s is not None \
            else self.t_ads_s + self.t_des_s
        return self.t_ads_s / t_cycle


_ANCHOR = get_material("anchor:Silica gel RD")

# Anchor isotherm (fixed in V4): q_sat / Q_st / E / n from the anchor table.
RD_ISOTHERM = {
    "q_sat_kg_kg": float(_ANCHOR.q_sat_kg_kg),
    "q_st_j_kg": float(_ANCHOR.q_st_j_kg),
    "e_char_j_mol": float(_ANCHOR.e_char_j_mol),
    "n_da": float(_ANCHOR.n_da),
}

# Fixed bed constants of the calibration runs (the model design point —
# NOT refit; see module docstring).
CALIB_STATIC = {
    "k_eff_w_m_k": 0.3,        # W/(m·K), packed silica gel bed
    "rho_s_kg_m3": 600.0,      # kg/m³ packed-bed adsorbent density
    "c_s_j_kg_k": CP_ADSORBENT,
    "c_pl_j_kg_k": CP_LIQUID,
}
# dt = 100 ms at N = 8: below the RK4 explicit-conduction limit at the wet
# state (dx²·rho·cap_wet/(2·k_eff)·1.39 ≈ 0.16 s) and k_LDF·dt ≤ 0.05 at
# the k_LDF bound — both split-scheme bounds comfortable. The calibrated
# dynamics evolve on tau ≥ 100 s, so this resolves them at ~1e-4 cost.
CALIB_NUMERICS = {"dt_s": 0.1, "n_cycles": 4, "n_cells": 8}

# Fitted-parameter bounds. ``k_ldf_s_1@<rig>`` is the BED-LEVEL effective
# uptake rate of that rig's gel (grain + vapour-channel resistance lumped
# — Open Question 1 evidence); ``hx_mass_factor@<rig>`` is that rig's
# metal + water inventory. ``h_wall_w_m2_k`` is shared (same bed type).
CALIB_BOUNDS = {
    "k_ldf_s_1@uyun2009": (1.0e-4, 0.05),
    "k_ldf_s_1@sztekler2021": (1.0e-4, 0.05),
    "h_wall_w_m2_k@uyun2009": (50.0, 500.0),
    "h_wall_w_m2_k@sztekler2021": (50.0, 500.0),
    "hx_mass_factor@uyun2009": (1.0, 12.0),
    "hx_mass_factor@sztekler2021": (1.0, 12.0),
}

# Default starting points for the fits. h is per rig: the two beds have
# different fin/tube constructions and the fits want 238 vs 101 W/m²K —
# both plausible for packed fin-tube beds (benchmarks.md).
CALIB_DEFAULTS = {
    "k_ldf_s_1@uyun2009": 3.0e-4,
    "k_ldf_s_1@sztekler2021": 1.0e-3,
    "h_wall_w_m2_k@uyun2009": 200.0,
    "h_wall_w_m2_k@sztekler2021": 200.0,
    "hx_mass_factor@uyun2009": 8.0,
    "hx_mass_factor@sztekler2021": 2.5,
}
RIGS = ("uyun2009", "sztekler2021")
RIG_STEMS = ("k_ldf_s_1@", "h_wall_w_m2_k@", "hx_mass_factor@")

# --- published operating points -------------------------------------------
# Uyun+2009 single-stage cycle: COP/SCP read from Figure 9 (single-stage
# curves) at the four heat-source temperatures; 540 s pure phases (the
# 60 s pre-heat/pre-cool intervals are excluded, see module docstring).
# The 70 C standard condition (Table 2) is the calibration point; the
# sweep points are predictions.
UYUN2009_CASES = tuple(
    BenchmarkCase(
        name=f"uyun2009_singlestage_{t_hs:g}C",
        source="uyun2009",
        rig="uyun2009",
        t_evap_c=7.5,
        t_cond_c=30.0,
        t_f_ads_c=14.0,
        t_f_des_c=float(t_hs),
        t_ads_s=540.0,
        t_des_s=540.0,
        COP=cop,
        SCP_W_kg=scp,
        fit=(t_hs == 70.0),
        t_cycle_rig_s=1200.0,  # Table 1b: +60 s pre-cool + 60 s pre-heat
    )
    for t_hs, cop, scp in (
        (65.0, 0.128, 25.5),
        (70.0, 0.169, 37.0),
        (75.0, 0.209, 52.0),
        (80.0, 0.237, 66.0),
    )
)

# Sztekler+2021 cycle-time sweep: equal adsorption / desorption phases
# 100-900 s at 80 C heat source; COP/SCP read from Figure 24. Boundary
# temperatures estimated from the secondary loops (chilled water ~16-18 C
# -> T_evap ~ 14 C; cooling water 20-22 C -> T_cond ~ 25 C). The 500 s
# point (their optimum) is the calibration point; the rest are trend
# predictions. SCP published per kg adsorbent with an ambiguous
# per-bed/total divisor — magnitudes validated loosely, trend strictly.
SZTEKLER2021_CASES = tuple(
    BenchmarkCase(
        name=f"sztekler2021_{t:g}s",
        source="sztekler2021",
        rig="sztekler2021",
        # refrigerant-side estimates from the secondary loops: chilled
        # water 16-18 C in with a ~4 K evaporator approach -> T_evap ~
        # 12.5 C; cooling water 20-22 C with a ~6 K condenser approach
        # -> T_cond ~ 27.5 C (benchmarks.md documents the sensitivity)
        t_evap_c=12.5,
        t_cond_c=27.5,
        t_f_ads_c=20.0,
        t_f_des_c=80.0,
        t_ads_s=float(t),
        t_des_s=float(t),
        COP=cop,
        SCP_W_kg=scp,
        fit=(t == 500.0),
    )
    for t, cop, scp in (
        (100.0, 0.355, 115.0),
        (200.0, 0.415, 128.0),
        (300.0, 0.485, 140.0),
        (500.0, 0.550, 153.5),
        (600.0, 0.585, 143.5),
        (900.0, 0.640, 130.5),
    )
)

BENCHMARK_CASES = UYUN2009_CASES + SZTEKLER2021_CASES

# Committed fit result (2026-08, harness/benchmarks.md): per-rig (k_LDF,
# h, hx) fitted to each rig's standard point. Regression-pinned by
# tests/harness/test_calibration.py.
FITTED_BY_RIG: dict[str, dict[str, float]] = {
    # bed-level effective uptake kinetics [1/s]; wall film [W/(m²·K)];
    # rig metal inventory [-]
    "uyun2009": {"k_ldf_s_1": 2.9297e-4, "h_wall_w_m2_k": 202.85,
                 "hx_mass_factor": 2.4968},
    "sztekler2021": {"k_ldf_s_1": 1.4317e-3, "h_wall_w_m2_k": 100.95,
                     "hx_mass_factor": 1.0},
}


def _case_arrays(cases):
    """Stack the per-case boundary conditions into vmappable arrays."""
    fields = ("t_evap_c", "t_cond_c", "t_f_ads_c", "t_f_des_c", "t_ads_s", "t_des_s")
    return tuple(jnp.asarray([getattr(c, f) for c in cases]) for f in fields)


def _episode_summary(t_evap_c, t_cond_c, t_f_ads_c, t_f_des_c, t_ads_s, t_des_s,
                     k_ldf, h_wall, hx, design, *, dt_s, n_cycles, n_cells,
                     n_steps):
    """One vmapped episode: boundary conditions and the per-rig transport
    scalars are tracers, ``design`` may be tracers (fit path) or floats.
    Returns the BED metric dict."""
    p_evap = water_sat_pressure_pa(t_evap_c + 273.15)
    p_cond = water_sat_pressure_pa(t_cond_c + 273.15)
    h_fg = water_h_fg_j_kg(t_evap_c + 273.15)
    t_init = jnp.full((n_cells,), t_cond_c + 273.15)
    q_init = da_uptake(t_init, p_evap, design["q_sat_kg_kg"],
                       design["e_char_j_mol"], design["n_da"])
    phys = {
        "dx": design["L_m"] / n_cells,
        "L_m": design["L_m"],
        "q_sat_kg_kg": design["q_sat_kg_kg"],
        "q_st_j_kg": design["q_st_j_kg"],
        "e_char_j_mol": design["e_char_j_mol"],
        "n_da": design["n_da"],
        "k_ldf_s_1": k_ldf,
        "rho_s_kg_m3": design["rho_s_kg_m3"],
        "c_s_j_kg_k": design["c_s_j_kg_k"],
        "c_pl_j_kg_k": design["c_pl_j_kg_k"],
        "k_eff_w_m_k": design["k_eff_w_m_k"],
        "h_wall_w_m2_k": h_wall,
        "hx_mass_factor": hx,
        "t_f_ads_c": t_f_ads_c,
        "p_evap_pa": p_evap,
        "p_cond_pa": p_cond,
        "h_fg_evap_j_kg": h_fg,
    }
    carry = bed1d.initial_carry(t_init, q_init, t_phase_end_s=t_ads_s, n_cycles=n_cycles)
    carry_f, _ = bed1d.advance_carry(
        carry, (t_ads_s, t_des_s, t_f_des_c),
        n_steps=n_steps, dt_s=dt_s, phys=phys,
    )
    return bed1d.summary_from_carry(
        carry_f, p_evap_pa=p_evap, p_cond_pa=p_cond, h_fg_evap_j_kg=h_fg
    )


KNOWN_DESIGN_KEYS = frozenset(CALIB_STATIC) | frozenset(RD_ISOTHERM) | {"L_m"}


def _full_design(design):
    """Merge user design over the fixed calibration constants."""
    d = dict(CALIB_STATIC)
    d.update(RD_ISOTHERM)
    d.setdefault("L_m", 0.002)
    unknown = set(design or {}) - KNOWN_DESIGN_KEYS
    if unknown:
        raise ValueError(f"unknown design keys for calibration: {sorted(unknown)}")
    d.update(design or {})
    return d


def _rig_param_arrays(cases, by_rig_maps):
    """Per-case arrays for each ``RIG_STEMS`` parameter from per-rig
    mappings (floats or tracers). Missing rigs fall back to
    :data:`FITTED_BY_RIG`."""
    arrays = []
    for stem in RIG_STEMS:
        key = stem[:-1]
        mapping = {r: FITTED_BY_RIG[r][key] for r in RIGS}
        mapping.update(by_rig_maps.get(key, {}) or {})
        arrays.append(jnp.asarray([mapping[c.rig] for c in cases]))
    return arrays


def simulate_cases(cases, design, *, k_by_rig=None, h_by_rig=None,
                   hx_by_rig=None,
                   dt_s=CALIB_NUMERICS["dt_s"],
                   n_cycles=CALIB_NUMERICS["n_cycles"],
                   n_cells=CALIB_NUMERICS["n_cells"]):
    """Episode metrics for a group of cases in one vmapped rollout.

    All cases share one scan length (set by the longest cycle); shorter-
    phase cases simply complete more cycles, and the metrics always come
    from the last ``n_cycles - 1`` completed cycles (steady periodicity).
    The ``*_by_rig`` maps take ``case.rig`` → that rig's effective uptake
    rate / wall film / metal-inventory factor (module docstring); values
    may be tracers and default to the committed fit.

    Returns ``(COP, SCP_model, delta_q)`` — JAX arrays over cases.
    """
    cases = list(cases)
    if not cases:
        raise ValueError("need at least one case")
    full_design = _full_design(design)
    n_steps = int(round(n_cycles * max(c.t_ads_s + c.t_des_s for c in cases) / float(dt_s))) + 8
    args = list(_case_arrays(cases))
    args.extend(_rig_param_arrays(cases, {"k_ldf_s_1": k_by_rig,
                                          "h_wall_w_m2_k": h_by_rig,
                                          "hx_mass_factor": hx_by_rig}))

    def run(t_evap_c, t_cond_c, t_f_ads_c, t_f_des_c, t_ads_s, t_des_s,
            k_ldf, h_wall, hx):
        return _episode_summary(t_evap_c, t_cond_c, t_f_ads_c, t_f_des_c,
                                t_ads_s, t_des_s, k_ldf, h_wall, hx,
                                full_design,
                                dt_s=dt_s, n_cycles=n_cycles, n_cells=n_cells,
                                n_steps=n_steps)

    out = jax.vmap(run)(*args)
    return out["COP"], out["SCP_W_kg"], out["delta_q"]


def model_case_rows(cases, design, k_by_rig=None, h_by_rig=None,
                    hx_by_rig=None, **numerics) -> list[dict[str, float]]:
    """Error table rows (numpy) for ``cases`` at ``design`` (rig SCP
    convention applied to the model value)."""
    rows = []
    cop_a, scp_a, dq_a = simulate_cases(cases, design, k_by_rig=k_by_rig,
                                        h_by_rig=h_by_rig, hx_by_rig=hx_by_rig,
                                        **numerics)
    cop_a, scp_a, dq_a = (np.asarray(x) for x in (cop_a, scp_a, dq_a))
    for i, c in enumerate(cases):
        scp_rig = float(scp_a[i]) * c.duty
        rows.append({
            "case": c.name,
            "COP_exp": c.COP,
            "COP_model": float(cop_a[i]),
            "COP_rel_err": float(cop_a[i]) / c.COP - 1.0,
            "SCP_exp_W_kg": c.SCP_W_kg,
            "SCP_model_rig_conv_W_kg": scp_rig,
            "SCP_rel_err": scp_rig / c.SCP_W_kg - 1.0,
            "delta_q": float(dq_a[i]),
            "fit": c.fit,
        })
    return rows


@dataclass
class CalibrationResult:
    """Fit outcome: parameters, loss, and the per-case error table."""

    fit_keys: tuple[str, ...]
    best_params: dict[str, float]
    best_loss: float
    n_evals: int
    fixed: dict[str, float]
    by_rig: dict[str, dict[str, float]]
    rows: list[dict[str, float]] = field(default_factory=list)
    history: list[float] = field(default_factory=list)

    @property
    def design(self) -> dict[str, float]:
        return {**self.fixed, **self.best_params}


def _unit_to_design(fit_keys, lo, hi, x01):
    """Unit-box → real parameters (log10 space). Tracer-safe: returns jax
    scalars when ``x01`` is traced (the fit path), floats otherwise."""
    out = {}
    for i, k in enumerate(fit_keys):
        val = jnp.power(10.0, lo[i] + x01[i] * (hi[i] - lo[i]))
        out[k] = val if isinstance(x01, jax.Array) else float(val)
    return out


def calibrate(cases, design=None, *, fit_keys=None,
              bounds=None, dt_s=CALIB_NUMERICS["dt_s"],
              n_cycles=CALIB_NUMERICS["n_cycles"],
              n_cells=CALIB_NUMERICS["n_cells"],
              n_starts=2, n_steps=60, step_size=0.08, seed=0) -> CalibrationResult:
    """Fit ``fit_keys`` to the ``fit=True`` cases by Adam on the relative
    square-error loss, in unit-box log10 coordinates.

    Keys take the form ``k_ldf_s_1@<rig>`` / ``h_wall_w_m2_k@<rig>`` /
    ``hx_mass_factor@<rig>`` (per rig, must cover every rig among the fit
    cases). Validation cases are excluded from the loss but reported in
    the result's error table. Deterministic under ``seed``. Default: fit
    each fit-case rig's full transport triple.
    """
    if fit_keys is None:
        rigs = sorted({c.rig for c in cases if c.fit})
        fit_keys = tuple(k for r in rigs for k in
                         (f"k_ldf_s_1@{r}", f"h_wall_w_m2_k@{r}",
                          f"hx_mass_factor@{r}"))
    try:
        import optax
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the calibrate() fit needs optax (pip install optax)") from exc

    cases = list(cases)
    fit_cases = [c for c in cases if c.fit]
    if not fit_cases:
        raise ValueError("no fit=True cases")
    fit_rigs = sorted({c.rig for c in fit_cases})
    for r in fit_rigs:
        for stem in RIG_STEMS:
            if fit_keys.count(stem + r) > 1:
                raise ValueError(f"fit key {stem + r!r} declared more than once")
    # stems not fitted stay at their documented defaults (e.g. fit only
    # (k_LDF, hx) with h pinned)
    rig_fixed = {stem + r: float(CALIB_DEFAULTS[stem + r])
                 for r in fit_rigs for stem in RIG_STEMS
                 if (stem + r) not in fit_keys}
    bounds = dict(CALIB_BOUNDS)
    for k in fit_keys:
        if k not in bounds:
            raise ValueError(f"no bounds declared for fit key {k!r}")
    lo = np.array([np.log10(bounds[k][0]) for k in fit_keys])
    hi = np.array([np.log10(bounds[k][1]) for k in fit_keys])
    span = hi - lo
    defaults_x = np.clip((np.log10([CALIB_DEFAULTS[k] for k in fit_keys]) - lo) / span,
                         0.0, 1.0)
    cop_exp = jnp.asarray([c.COP for c in fit_cases])
    scp_exp = jnp.asarray([c.SCP_W_kg for c in fit_cases])
    duty = jnp.asarray([c.duty for c in fit_cases])
    rig_index = jnp.asarray([fit_rigs.index(c.rig) for c in fit_cases])
    fixed = _full_design(design)
    for k in fit_keys:
        fixed.pop(k, None)

    def loss_x(x01):
        fit_vals = _unit_to_design(fit_keys, lo, hi, x01)
        per_rig = [jnp.stack([fit_vals[f"{stem}{r}"]
                              if f"{stem}{r}" in fit_vals
                              else rig_fixed[f"{stem}{r}"]
                              for r in fit_rigs])
                   for stem in RIG_STEMS]
        d = {**fixed, **{k: v for k, v in fit_vals.items() if "@" not in k}}
        cop, scp, _ = _sim_rig_params(
            fit_cases, d, *(p[rig_index] for p in per_rig),
            dt_s, n_cycles, n_cells)
        r_cop = (cop - cop_exp) / cop_exp
        r_scp = (scp * duty - scp_exp) / scp_exp
        return jnp.mean(r_cop ** 2) + jnp.mean(r_scp ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(loss_x))
    rng = np.random.default_rng(seed)
    best_loss, best_x, n_evals, history = np.inf, None, 0, []
    for start in range(max(1, n_starts)):
        x = jnp.asarray(defaults_x if start == 0 else rng.uniform(0.0, 1.0, size=len(fit_keys)))
        optimizer = optax.adam(step_size)
        state = optimizer.init(x)
        for _ in range(int(n_steps)):
            value, grad = value_and_grad(x)
            n_evals += 1
            value_f = float(value)
            history.append(value_f)
            if value_f < best_loss:
                best_loss, best_x = value_f, np.array(x)
            updates, state = optimizer.update(grad, state, x)
            # optax updates are descent steps (to be ADDED): minimise the loss.
            x = jnp.clip(x + updates, 0.0, 1.0)
    best_params = _unit_to_design(fit_keys, lo, hi, best_x)
    by_rig = {r: {} for r in fit_rigs}
    for stem in RIG_STEMS:
        for r in fit_rigs:
            key = f"{stem}{r}"
            by_rig[r][stem[:-1]] = best_params.pop(key) if key in best_params \
                else rig_fixed[key]
    rows = model_case_rows(cases, {**fixed, **best_params},
                           k_by_rig={r: v["k_ldf_s_1"] for r, v in by_rig.items()},
                           h_by_rig={r: v["h_wall_w_m2_k"] for r, v in by_rig.items()},
                           hx_by_rig={r: v["hx_mass_factor"] for r, v in by_rig.items()},
                           dt_s=dt_s, n_cycles=n_cycles, n_cells=n_cells)
    return CalibrationResult(
        fit_keys=tuple(fit_keys), best_params=best_params, best_loss=best_loss,
        n_evals=n_evals, fixed=fixed, by_rig=by_rig,
        rows=rows, history=history,
    )


def _sim_rig_params(cases, design, k_cases, h_cases, hx_cases,
                     dt_s, n_cycles, n_cells):
    """``simulate_cases`` with explicit per-case k/h/hx arrays (fit path)."""
    full_design = _full_design(design)
    n_steps = int(round(n_cycles * max(c.t_ads_s + c.t_des_s for c in cases) / float(dt_s))) + 8
    args = list(_case_arrays(cases)) + [k_cases, h_cases, hx_cases]

    def run(t_evap_c, t_cond_c, t_f_ads_c, t_f_des_c, t_ads_s, t_des_s,
            k_ldf, h_wall, hx):
        return _episode_summary(t_evap_c, t_cond_c, t_f_ads_c, t_f_des_c,
                                t_ads_s, t_des_s, k_ldf, h_wall, hx,
                                full_design,
                                dt_s=dt_s, n_cycles=n_cycles, n_cells=n_cells,
                                n_steps=n_steps)

    out = jax.vmap(run)(*args)
    return out["COP"], out["SCP_W_kg"], out["delta_q"]


__all__ = [
    "BENCHMARK_CASES",
    "BENCHMARK_SOURCES",
    "BenchmarkCase",
    "CALIB_BOUNDS",
    "CALIB_DEFAULTS",
    "CALIB_NUMERICS",
    "CALIB_STATIC",
    "CalibrationResult",
    "FITTED_BY_RIG",
    "RD_ISOTHERM",
    "SZTEKLER2021_CASES",
    "UYUN2009_CASES",
    "calibrate",
    "model_case_rows",
    "simulate_cases",
]
