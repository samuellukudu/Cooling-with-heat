"""Two-bed adsorption system — a counter-phase Bed1D pair with optional
heat recovery (``../DESIGN.md`` §4.3, the Env-2 physics).

Two :mod:`.bed1d` episode carries are stepped in lockstep by one
``jax.lax.scan``: at every instant exactly one bed adsorbs (connected to
the evaporator) while the twin desorbs (connected to the condenser and
the heat source). The **system phase** σ ∈ {0, 1} records which bed
adsorbs; both beds always flip together, so the pair stays in exact
counter-phase. The beds may be built from different materials and
geometries — the basis for the composite-bed experiments.

**Timing.** The system phase σ = 0 ("bed A adsorbs") lasts ``D0`` seconds
and σ = 1 lasts ``D1`` — the asymmetric two-duration parameterization of
a counter-phase machine. Durations are *role-keyed* per bed: bed A's
adsorption phases coincide with σ = 0, bed B's with σ = 1, so bed B is
stepped with the swapped duration pair and every per-bed book (phase
clocks, adsorption-time histories, phase-fraction channels) stays
role-consistent. A phase ends at its declared duration — read at phase
start, so per-step duration series must be constant within a phase —
or earlier when a **request bit** demands the other phase. Request bits
are the valve-level hook that schedule policies (H2.2) use to gate
desorption on a time-varying heat source; they respect a minimum dwell
so a policy that chatters at its own threshold cannot vibrate the
valves every substep.

**Heat recovery.** During the first ``t_rec`` seconds after each flip
the two beds are disconnected from their fluids (the recovery loop owns
the wall circuits — the wall film is scaled to zero, so the booked wall
flux and hence the source draw are exactly zero) and exchange heat
through a lumped conductance ``UA_rec`` [W/(m²·K)] between their mean
temperatures:

    Q̇_rec = UA_rec · (T̄_hot − T̄_cold)   [W/m² of coupled wall]

applied as an exactly antisymmetric per-cell extra source (bed A loses
what bed B gains, frozen over the step like every other split-scheme
term). The films must be off for this to work: they would otherwise
re-pin each bed to its own fluid within seconds and reverse the
exchange. The exchange conserves ``U_A + U_B`` to machine precision,
and it is *free* in the accounting: ``Q_in`` books only each bed's own
wall flux and metal term — recovery heat is a transfer between beds,
not source input, which is exactly why it raises COP. The vapor valves
keep their phase connections during the recovery window (v1
simplification: no vapor-pressure dynamics — the lumped vapor
inventory of Open Question 2 would be its extension).

**Initialization.** Both beds start at a steady-state-like flip instant:
bed A at its desorption-end state (hot, equilibrated on the condenser
isotherm) entering adsorption; bed B at its adsorption-end state (cold,
equilibrated on the evaporator isotherm) entering desorption. With this
both beds contribute from the first substep — there is no zero-duty
warm-up stretch, which is what the V6 duty-continuity criterion
requires. The first per-bed cycle is still transient (the true periodic
state is reached after one cycle) and stays excluded from metrics via
the usual last-``n_cycles−1``-cycles convention.

**Metrics.** Per-bed books are recombined in absolute energy (each bed
is 1 m² of wall): ``COP = ΣQ_cool/ΣQ_in`` and
``SCP = ΣQ_cool/(Σ t_ads·m_s)`` — the single-bed convention
generalized; identical beds reproduce the ``Bed1D`` value exactly. For
identical symmetric beds the system is the single-bed cycle
phase-shifted by half a period, so in the equilibrium limit the system
COP matches the ``Cycle0D`` oracle (the V3 correspondence, re-checked
at system level by V6).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import bed1d
from .thermo import da_uptake, water_h_fg_j_kg, water_sat_pressure_pa

# Per-step series: system block + one full bed1d block per bed + per-bed
# cumulative books. Index helpers below are the single source of truth.
SYSTEM_SERIES_CHANNELS = (
    "t_abs_s",
    "sys_phase",
    "req",
    "q_rec_w_m2",
    *(f"A_{c}" for c in bed1d.SERIES_CHANNELS),
    *(f"B_{c}" for c in bed1d.SERIES_CHANNELS),
    "A_Q_cool_cum_J_kg",
    "A_Q_in_cum_J_kg",
    "B_Q_cool_cum_J_kg",
    "B_Q_in_cum_J_kg",
)
_N_BED_CHANNELS = len(bed1d.SERIES_CHANNELS)
A_BLOCK = slice(4, 4 + _N_BED_CHANNELS)
B_BLOCK = slice(4 + _N_BED_CHANNELS, 4 + 2 * _N_BED_CHANNELS)
A_QCOOL_CUM_IDX = 4 + 2 * _N_BED_CHANNELS
A_QIN_CUM_IDX = A_QCOOL_CUM_IDX + 1
B_QCOOL_CUM_IDX = A_QCOOL_CUM_IDX + 2
B_QIN_CUM_IDX = A_QCOOL_CUM_IDX + 3

# Per-bed physics/configuration keys (one dict per bed; the refrigerant
# circuit — evaporator/condenser setpoints — is system-level).
BED_KEYS = (
    "q_sat_kg_kg",
    "q_st_j_kg",
    "e_char_j_mol",
    "n_da",
    "k_ldf_s_1",
    "rho_s_kg_m3",
    "c_s_j_kg_k",
    "c_pl_j_kg_k",
    "k_eff_w_m_k",
    "h_wall_w_m2_k",
    "L_m",
    "n_cells",
    "hx_mass_factor",
)


def bed_phys(
    *,
    q_sat_kg_kg,
    q_st_j_kg,
    e_char_j_mol,
    n_da,
    k_ldf_s_1,
    rho_s_kg_m3,
    c_s_j_kg_k,
    c_pl_j_kg_k,
    k_eff_w_m_k,
    h_wall_w_m2_k,
    L_m,
    n_cells,
    hx_mass_factor,
    t_evap_c,
    t_cond_c,
    t_f_ads_c,
    soft_switch=False,
):
    """Assemble one bed's configuration dict for :func:`advance_two_carry`.

    The refrigerant-circuit pressures and latent heat are computed from
    the shared evaporator/condenser setpoints; ``t_f_ads_c`` is the
    adsorption-phase fluid temperature (a construction constant — the
    desorption fluid temperature arrives per step as a control).
    """
    n_cells = int(n_cells)
    return {
        "q_sat_kg_kg": q_sat_kg_kg,
        "q_st_j_kg": q_st_j_kg,
        "e_char_j_mol": e_char_j_mol,
        "n_da": n_da,
        "k_ldf_s_1": k_ldf_s_1,
        "rho_s_kg_m3": rho_s_kg_m3,
        "c_s_j_kg_k": c_s_j_kg_k,
        "c_pl_j_kg_k": c_pl_j_kg_k,
        "k_eff_w_m_k": k_eff_w_m_k,
        "h_wall_w_m2_k": h_wall_w_m2_k,
        "L_m": L_m,
        "n_cells": n_cells,
        "dx": L_m / n_cells,
        "hx_mass_factor": hx_mass_factor,
        "t_evap_c": t_evap_c,
        "t_cond_c": t_cond_c,
        "t_f_ads_c": t_f_ads_c,
        "p_evap_pa": water_sat_pressure_pa(t_evap_c + 273.15),
        "p_cond_pa": water_sat_pressure_pa(t_cond_c + 273.15),
        "h_fg_evap_j_kg": water_h_fg_j_kg(t_evap_c + 273.15),
        "soft_switch": soft_switch,
    }


def _set_carry(carry, idx, value):
    return carry[:idx] + (value,) + carry[idx + 1 :]


def initial_two_carry(phys_a, phys_b, *, t_phase0_s, t_des_end_k, n_cycles):
    """Carry pair at a steady-state-like flip instant (module docstring).

    Bed A enters adsorption hot and empty (desorption-end state at the
    initial desorption-fluid level ``t_des_end_k`` [K]), bed B enters
    desorption cold and loaded (adsorption-end state), so both beds
    contribute duty from the first substep. ``t_phase0_s`` seeds both
    beds' internal phase clocks with the σ = 0 duration.
    """
    p_evap = phys_a["p_evap_pa"]
    p_cond = phys_a["p_cond_pa"]
    t_a = jnp.full((phys_a["n_cells"],), t_des_end_k)
    q_a = da_uptake(t_a, p_cond, phys_a["q_sat_kg_kg"],
                    phys_a["e_char_j_mol"], phys_a["n_da"])
    t_b = jnp.full((phys_b["n_cells"],), phys_b["t_f_ads_c"] + 273.15)
    q_b = da_uptake(t_b, p_evap, phys_b["q_sat_kg_kg"],
                    phys_b["e_char_j_mol"], phys_b["n_da"])
    carry_a = bed1d.initial_carry(t_a, q_a, t_phase_end_s=t_phase0_s,
                                  n_cycles=n_cycles)
    carry_b = bed1d.initial_carry(t_b, q_b, t_phase_end_s=t_phase0_s,
                                  n_cycles=n_cycles)
    carry_b = _set_carry(carry_b, bed1d._CARRY_PHASE,
                         jnp.asarray(bed1d.DES_PHASE))
    return carry_a, carry_b, jnp.asarray(0.0)


def advance_two_carry(carry, xs, *, n_steps, dt_s, phys_a, phys_b,
                      recovery_ua_w_m2_k=0.0, use_req=False):
    """Advance a two-bed carry pair by ``n_steps`` lockstep physics steps.

    ``xs = (t_d0_s, t_d1_s, t_f_des_c, t_rec_s, t_dwell_min_s, req)`` —
    durations of the σ = 0 / σ = 1 system phases [s], desorption fluid
    temperature [°C], recovery window [s], request minimum dwell [s],
    and the request bit (desired system phase). Each entry is a scalar
    broadcast over the steps or a ``(n_steps,)`` array; ``req`` is
    ignored unless ``use_req``. ``phys_a``/``phys_b`` are the per-bed
    configuration dicts (:func:`bed_phys`); ``recovery_ua_w_m2_k`` is
    the lumped recovery conductance (0 disables the coupling).

    Returns ``((carry_a, carry_b, q_rec_acc), ys)`` with ``ys`` shaped
    ``(n_steps, len(SYSTEM_SERIES_CHANNELS))``.
    """

    def _col(v):
        arr = jnp.asarray(v)
        return jnp.full((n_steps,), arr) if arr.ndim == 0 else arr

    d0_x, d1_x, tfd_x, trec_x, tdw_x, req_x = (_col(v) for v in xs)
    ua = float(recovery_ua_w_m2_k)
    n_a = phys_a["n_cells"]
    n_b = phys_b["n_cells"]

    def body(carry, x):
        carry_a, carry_b, q_rec_acc = carry
        d0, d1, t_f_des, t_rec, t_dwell, req = x
        t_abs = carry_a[bed1d._CARRY_T_ABS]
        t_phase_start = carry_a[bed1d._CARRY_T_PHASE_START]
        sigma = carry_a[bed1d._CARRY_PHASE]

        # System-level flip decision; both beds always flip together.
        # The internal bed clocks are aligned by construction (both
        # were seeded with the phase duration at the last flip), so
        # either one firing means the system phase is over; request
        # bits demand the other phase after the minimum dwell.
        t_next = t_abs + dt_s
        flip_timeout = ((t_next >= carry_a[bed1d._CARRY_T_END])
                        | (t_next >= carry_b[bed1d._CARRY_T_END]))
        if use_req:
            flip_req = ((req > 0.5) != (sigma > 0.5)) & (
                (t_abs - t_phase_start) >= t_dwell)
        else:
            flip_req = jnp.asarray(False)
        flip_now = flip_timeout | flip_req
        carry_a = _set_carry(carry_a, bed1d._CARRY_T_END,
                             jnp.where(flip_now, t_abs,
                                       carry_a[bed1d._CARRY_T_END]))
        carry_b = _set_carry(carry_b, bed1d._CARRY_T_END,
                             jnp.where(flip_now, t_abs,
                                       carry_b[bed1d._CARRY_T_END]))

        # Heat recovery: during the first t_rec of the phase the beds are
        # disconnected from their fluids (the recovery loop owns the wall
        # circuits — wall film scaled to zero, so the booked wall flux and
        # hence the source draw are exactly zero) and exchange heat through
        # a lumped UA between their mean temperatures, applied as exactly
        # antisymmetric per-cell sources (bed A loses what bed B gains).
        # The exchange direction follows the instantaneous mean-temperature
        # difference; with the films off it decays monotonically instead of
        # being swamped by the films within seconds.
        if ua > 0.0:
            rec_on = (t_rec > 0.0) & ((t_abs - t_phase_start) < t_rec)
            q_rec = jnp.where(
                rec_on,
                ua * (jnp.mean(carry_a[bed1d._CARRY_T])
                      - jnp.mean(carry_b[bed1d._CARRY_T])),
                0.0,
            )
            film = jnp.where(rec_on, 0.0, 1.0)
        else:
            q_rec = 0.0
            film = None
        extra_a = -q_rec / n_a
        extra_b = q_rec / n_b

        # Role-keyed controls: bed B's adsorption phases coincide with
        # the σ = 1 system phase, so its duration pair is swapped.
        carry_a, ys_a = bed1d.step_episode(
            carry_a, (d0, d1, t_f_des), dt_s=dt_s, phys=phys_a,
            extra_source_w_m2=extra_a, wall_film_scale=film)
        carry_b, ys_b = bed1d.step_episode(
            carry_b, (d1, d0, t_f_des), dt_s=dt_s, phys=phys_b,
            extra_source_w_m2=extra_b, wall_film_scale=film)
        # Accumulate the ABSOLUTE transfer: the hot bed is whichever just
        # finished desorption, so the exchange direction alternates with
        # the system phase and a signed integral would cancel to zero over
        # a cycle while ~UA·ΔT·t_rec of heat moves each window.
        q_rec_acc = q_rec_acc + jnp.abs(q_rec) * dt_s

        row = jnp.concatenate((
            jnp.stack((t_next, carry_a[bed1d._CARRY_PHASE], req, q_rec)),
            ys_a,
            ys_b,
            jnp.stack((carry_a[bed1d._CARRY_QCOOL_CUM],
                       carry_a[bed1d._CARRY_QIN_CUM],
                       carry_b[bed1d._CARRY_QCOOL_CUM],
                       carry_b[bed1d._CARRY_QIN_CUM])),
        ))
        return (carry_a, carry_b, q_rec_acc), row

    return jax.lax.scan(
        body,
        carry,
        (d0_x, d1_x, tfd_x, trec_x, tdw_x, req_x),
        length=n_steps,
    )


def summary_from_two_carry(carry, *, m_s_a, m_s_b, p_evap_pa, p_cond_pa,
                           h_fg_evap_j_kg):
    """Two-bed episode metrics from the final carry (DESIGN §5.3 schema).

    Per-bed books are recombined in absolute energy (each bed is 1 m² of
    wall). ``Q_rec_J_m2`` is the signed cumulative recovery transfer
    (positive = bed A → bed B). ``unmet_load_frac`` is 0 in v1: there is
    no chilled-water load model yet (stochastic/load experiments, H2.4+).
    """
    carry_a, carry_b, q_rec_acc = carry
    ta = bed1d.cycle_totals_from_carry(carry_a)
    tb = bed1d.cycle_totals_from_carry(carry_b)
    m_tot = m_s_a + m_s_b

    q_cool = ta["Q_cool_J_kg"] * m_s_a + tb["Q_cool_J_kg"] * m_s_b
    q_in = ta["Q_in_J_kg"] * m_s_a + tb["Q_in_J_kg"] * m_s_b
    t_ads_w = ta["t_ads_s"] * m_s_a + tb["t_ads_s"] * m_s_b

    safe_qin = jnp.where(q_in > 0.0, q_in, 1.0)
    cop = jnp.where(q_in > 0.0, q_cool / safe_qin, 0.0)
    safe_t = jnp.where(t_ads_w > 0.0, t_ads_w, 1.0)
    scp = jnp.where(t_ads_w > 0.0, q_cool / safe_t, 0.0)
    n_last = carry_a[bed1d._CARRY_HIST_QCOOL].shape[0] - 1
    safe_n = max(n_last, 1)
    return {
        "COP": cop,
        "SCP_W_kg": scp,
        "delta_q": (ta["delta_q"] * m_s_a + tb["delta_q"] * m_s_b)
        / (m_tot * safe_n),
        "q_ads": (ta["q_ads"] * m_s_a + tb["q_ads"] * m_s_b) / (m_tot * safe_n),
        "q_des": (ta["q_des"] * m_s_a + tb["q_des"] * m_s_b) / (m_tot * safe_n),
        "P_evap_kPa": p_evap_pa / 1000.0,
        "P_cond_kPa": p_cond_pa / 1000.0,
        "h_fg_MJ_kg": h_fg_evap_j_kg / 1e6,
        "Q_cool_J_kg": q_cool / m_tot,
        "Q_in_J_kg": q_in / m_tot,
        "Q_rec_J_m2": q_rec_acc,
        "unmet_load_frac": 0.0,
    }


def simulate_two_bed(
    *,
    bed_a,
    bed_b,
    t_evap_c,
    t_cond_c,
    t_f_ads_c,
    t_f_des_c=None,
    t_ads_s=None,
    t_des_s=None,
    t_rec_s=0.0,
    t_dwell_min_s=0.0,
    recovery_ua_w_m2_k=0.0,
    dt_s,
    n_cycles=4,
    n_steps=None,
    collect_trace=False,
    soft_switch=False,
    series=None,
    req=None,
):
    """Roll out a whole two-bed episode with :func:`jax.lax.scan`.

    ``bed_a``/``bed_b`` are per-bed physics dicts (:func:`bed_phys`
    keyword set minus the shared refrigerant setpoints, which are passed
    at the top level). Time controls are scalars — ``t_ads_s``/``t_des_s``
    are the σ = 0 / σ = 1 phase durations, ``t_f_des_c`` the desorption
    fluid temperature — or per-step arrays via ``series =
    {"t_ads_s": …, "t_des_s": …, "t_f_des_c": …, "t_rec_s": …,
    "t_dwell_min_s": …}`` (any subset) and ``req`` (the request-bit
    array; passing it enables request flips). A scalar-only call infers
    the horizon as ``n_cycles·(t_ads_s + t_des_s)/dt_s`` and requires
    static durations (the scan length must be static); a series-driven
    call takes the horizon from the series length.

    Returns ``{"summary": …, "series": …, "m_s_a_kg_m2": …,
    "m_s_b_kg_m2": …}`` with ``summary`` mapping the metric keys to JAX
    scalars (gradients flow) and ``series`` a dict of decimated per-step
    channels or ``None``.
    """
    for name, bed in (("bed_a", bed_a), ("bed_b", bed_b)):
        missing = [k for k in BED_KEYS if k not in bed]
        if missing:
            raise KeyError(f"{name} is missing per-bed keys {missing}")
    t_f_des_c = 0.0 if t_f_des_c is None else t_f_des_c

    phys_a = bed_phys(**{k: bed_a[k] for k in BED_KEYS},
                      t_evap_c=t_evap_c, t_cond_c=t_cond_c,
                      t_f_ads_c=t_f_ads_c, soft_switch=soft_switch)
    phys_b = bed_phys(**{k: bed_b[k] for k in BED_KEYS},
                      t_evap_c=t_evap_c, t_cond_c=t_cond_c,
                      t_f_ads_c=t_f_ads_c, soft_switch=soft_switch)

    series = {k: jnp.asarray(v) for k, v in (series or {}).items()}
    if req is not None:
        req = jnp.asarray(req)
        n_steps = int(req.shape[0])
        if series and any(int(a.shape[0]) != n_steps for a in series.values()):
            raise ValueError("series and req must share the horizon length")
    elif series:
        n_steps = int(next(iter(series.values())).shape[0])
    elif n_steps is None:
        try:
            t_total = float(t_ads_s) + float(t_des_s)
        except TypeError as exc:  # tracer durations need a static horizon
            raise TypeError(
                "simulate_two_bed needs static t_ads_s/t_des_s (or an "
                "explicit n_steps / series) because the scan length must "
                "be static"
            ) from exc
        if min(float(t_ads_s), float(t_des_s)) < 2.0 * float(dt_s):
            raise ValueError(
                "phase durations must exceed ~2 dt_s for the flip logic "
                "(got t_ads_s=%g, t_des_s=%g, dt_s=%g)"
                % (float(t_ads_s), float(t_des_s), float(dt_s))
            )
        n_steps = int(round(n_cycles * t_total / float(dt_s))) + 8

    t_f_des0 = series["t_f_des_c"][0] if "t_f_des_c" in series else t_f_des_c
    t_phase0 = series["t_ads_s"][0] if "t_ads_s" in series else t_ads_s
    carry = initial_two_carry(phys_a, phys_b, t_phase0_s=t_phase0,
                              t_des_end_k=t_f_des0 + 273.15,
                              n_cycles=n_cycles)

    def _s(key, default):
        return series[key] if key in series else default

    xs = (
        _s("t_ads_s", t_ads_s),
        _s("t_des_s", t_des_s),
        _s("t_f_des_c", t_f_des_c),
        _s("t_rec_s", t_rec_s),
        _s("t_dwell_min_s", t_dwell_min_s),
        0.0 if req is None else req,
    )
    (carry_a, carry_b, q_rec_acc), ys = advance_two_carry(
        carry, xs, n_steps=n_steps, dt_s=dt_s, phys_a=phys_a, phys_b=phys_b,
        recovery_ua_w_m2_k=recovery_ua_w_m2_k, use_req=req is not None,
    )
    summary = summary_from_two_carry(
        (carry_a, carry_b, q_rec_acc),
        m_s_a=phys_a["rho_s_kg_m3"] * phys_a["L_m"],
        m_s_b=phys_b["rho_s_kg_m3"] * phys_b["L_m"],
        p_evap_pa=phys_a["p_evap_pa"],
        p_cond_pa=phys_a["p_cond_pa"],
        h_fg_evap_j_kg=phys_a["h_fg_evap_j_kg"],
    )

    out_series = None
    if collect_trace:
        stride = max(1, n_steps // 2048)
        thinned = ys[::stride]
        out_series = {
            name: thinned[:, i] for i, name in enumerate(SYSTEM_SERIES_CHANNELS)
        }
    return {
        "summary": summary,
        "series": out_series,
        "m_s_a_kg_m2": phys_a["rho_s_kg_m3"] * phys_a["L_m"],
        "m_s_b_kg_m2": phys_b["rho_s_kg_m3"] * phys_b["L_m"],
    }


__all__ = [
    "A_BLOCK",
    "A_QCOOL_CUM_IDX",
    "A_QIN_CUM_IDX",
    "B_BLOCK",
    "B_QCOOL_CUM_IDX",
    "B_QIN_CUM_IDX",
    "BED_KEYS",
    "SYSTEM_SERIES_CHANNELS",
    "advance_two_carry",
    "bed_phys",
    "initial_two_carry",
    "simulate_two_bed",
    "summary_from_two_carry",
]
