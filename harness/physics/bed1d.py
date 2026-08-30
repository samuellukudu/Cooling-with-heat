"""Dynamic 1-D adsorber bed — method of lines on a wall-normal slab
(``../DESIGN.md`` §4.2, the Env-1 physics).

Geometry (v1, coated fin / wall-normal slab): ``x ∈ [0, L]`` with the
heat-transfer fluid at ``x = 0`` (convective wall) and an adiabatic far
face at ``x = L`` (vapour-side symmetry). Cell-centred finite volumes,
explicit RK4 in time inside ``jax.lax.scan`` — a whole episode is one
jittable, ``vmap``-able rollout. Everything is differentiable: the D–A
isotherm and latent-heat correlations come from :mod:`.thermo`, and the
scan carries temperature and uptake fields through every valve flip.

Governing equations (per unit wall area; ``m_s = ρ_s·L`` kg adsorbent/m²):

    (c_s + c_pl·q)·ρ_s·∂T/∂t = ∂/∂x(k_eff·∂T/∂x) + ρ_s·q̇·Q_st
    q̇(x,t) = k_LDF·( q*(T(x,t), P(t)) − q(x,t) )
    wall:  −k_eff·∂T/∂x|₀ = h·(T_f(t) − T(0,t));   far:  ∂T/∂x|_L = 0
    valve: ads → P = P_sat(T_evap);   des → P = P_sat(T_cond)

Source sign: adsorption (q̇ > 0) releases ``Q_st`` and *heats* the bed.
With the adsorbed-phase enthalpy convention ``h_a = h_g − Q_st`` this PDE
is the exact bed energy balance — the vapour-enthalpy and storage
cross-terms cancel identically — which is what makes the desorption-phase
wall-flux integral reduce to the ``Cycle0D`` heat-input accounting in the
equilibrium limit (validation V3, ``../DESIGN.md`` §9).

Stiffness. The LDF uptake is integrated with an exact exponential substep
per step (``q*`` frozen at the start-of-step state):

    q_{n+1} = q* + (q_n − q*)·exp(−k_LDF·Δt)

which removes the ``k·Δt < 2`` explicit-Euler bound of the uptake ODE
itself (any ``k·Δt`` is stable in ``q``). The remaining limit is the
two-way adsorption-heat ↔ temperature coupling of the split scheme:
linearising the one-step map gives roughly
``|Q_st/cap · dq*/dT|·(1 − e^{−k_LDF·Δt}) ≲ 2``, i.e. ``k_LDF·Δt ≲ 0.3``
for typical parameters. The equilibrium limit that V3 exercises is
therefore τ_kin = 1/k_LDF ≪ cycle time — reached with ``k_LDF = 5 1/s``
(τ_kin = 0.2 s against 600 s phases) — and k-convergence is asserted in
the V3 test rather than taking k literally to infinity.

Energy accounting (per kg adsorbent, mirroring the ``Cycle0D`` oracle):

- cooling: ``Q_cool`` accumulates ``h_fg(T_evap)·d(mean q)`` during the
  adsorption phase only (signed — the valve-flip burst that first pushes
  vapour *to* the evaporator nets out against the later uptake, exactly
  as the oracle's instantaneous-isosteric approximation assumes);
- heat input: ``Q_in`` accumulates ``hx_mass_factor ×`` the wall-flux
  integral during the desorption phase only. The per-step identity
  ``Σ cap·ΔT·Δx = Φ_wall + ρ_s·Q_st·Σ Δq·Δx`` holds to machine precision
  (the RK4 wall-flux quadrature is the same one the T-update uses), which
  the V2 tests pin down.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .thermo import da_uptake, water_h_fg_j_kg, water_sat_pressure_pa

ADS_PHASE = 0.0
DES_PHASE = 1.0

# Per-step series channels recorded by :func:`simulate_bed` (DESIGN §5.2).
SERIES_CHANNELS = (
    "t_wall_k",
    "t_bed_mean_k",
    "t_bed_max_k",
    "q_mean_kg_kg",
    "q_star_mean_kg_kg",
    "p_over_p_evap",
    "t_fluid_k",
    "phase",
    "phase_fraction",
    "dq_cool_j_kg",
    "dq_in_j_kg",
)


def volumetric_capacity(rho_s_kg_m3, c_s_j_kg_k, c_pl_j_kg_k, q_kg_kg):
    """Bed volumetric heat capacity [J/(m³·K)] at uptake ``q``."""
    return rho_s_kg_m3 * (c_s_j_kg_k + c_pl_j_kg_k * q_kg_kg)


def max_timestep(L_m, n_cells, k_eff_w_m_k, rho_cp_eff_j_m3_k):
    """Explicit conduction limit ``Δt ≤ Δx²/(2α)`` (diffheat rule of thumb)."""
    dx = L_m / n_cells
    alpha = k_eff_w_m_k / rho_cp_eff_j_m3_k
    return dx * dx / (2.0 * alpha)


def check_timestep(L_m, n_cells, k_eff_w_m_k, rho_cp_eff_j_m3_k, dt_s) -> bool:
    """True when ``dt_s`` satisfies the explicit conduction CFL bound.

    Mirrors ``diffheat.check_cfl`` so backend sweeps fail loudly the same
    way the frozen 1-D solver does (DESIGN §4.2).
    """
    return bool(dt_s <= max_timestep(L_m, n_cells, k_eff_w_m_k, rho_cp_eff_j_m3_k))


def _conduction_div(T, dx, k_eff_w_m_k, h_wall_w_m2_k, t_f_k):
    """Net energy flux per cell [W/m² of wall]: conduction + wall BC.

    Cell-centred finite volumes. The convective wall BC is enforced with a
    ghost cell (cell-centred Robin), which carries the half-cell film
    resistance — using the raw cell average in ``h·(T_f − T_0)`` instead
    would over-drive the wall flux by ~``h·Δx/(2k)``. The far face is
    adiabatic (mirror ghost). The operator telescopes exactly:
    ``sum(div) = wall flux``, with the wall flux taken as the face flux
    ``k·(T_g − T_0)/Δx = h·(T_f − T_surface)``.
    """
    a = k_eff_w_m_k / dx
    denom = a + 0.5 * h_wall_w_m2_k
    # Tg = T0 + h·(T_f − T0)/(k/Δx + h/2); the where guards the
    # k_eff = h = 0 (fully decoupled) corner, not a physical branch.
    T_ghost = jnp.where(
        denom > 0.0,
        T[0] + h_wall_w_m2_k * (t_f_k - T[0]) / denom,
        T[0],
    )
    Tp = jnp.concatenate([jnp.reshape(T_ghost, (1,)), T, T[-1:]])
    div = k_eff_w_m_k * (Tp[:-2] - 2.0 * T + Tp[2:]) / dx
    wall = k_eff_w_m_k * (T_ghost - T[0]) / dx
    return div, wall


BED_RHS_PARAMS = (
    "dx",
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
)


def bed_rhs(T, q, t_f_k, p_pa, *, dx, q_sat_kg_kg, q_st_j_kg, e_char_j_mol,
            n_da, k_ldf_s_1, rho_s_kg_m3, c_s_j_kg_k, c_pl_j_kg_k,
            k_eff_w_m_k, h_wall_w_m2_k):
    """Semi-discrete right-hand side at state ``(T, q)`` under lumped
    pressure ``p_pa`` and fluid temperature ``t_f_k``.

    Returns ``(dT/dt [K/s], q̇ [kg/kg/s], wall flux [W/m² into the bed])``.
    """
    div, wall = _conduction_div(T, dx, k_eff_w_m_k, h_wall_w_m2_k, t_f_k)
    cap = rho_s_kg_m3 * (c_s_j_kg_k + c_pl_j_kg_k * q)
    q_star = da_uptake(T, p_pa, q_sat_kg_kg, e_char_j_mol, n_da)
    q_dot = k_ldf_s_1 * (q_star - q)
    d_t = (div + rho_s_kg_m3 * q_dot * q_st_j_kg * dx) / (cap * dx)
    return d_t, q_dot, wall


def step_bed(T, q, t_f_k, p_pa, dt_s, *, dx, q_sat_kg_kg, q_st_j_kg,
             e_char_j_mol, n_da, k_ldf_s_1, rho_s_kg_m3, c_s_j_kg_k,
             c_pl_j_kg_k, k_eff_w_m_k, h_wall_w_m2_k):
    """One full time step: exact-exponential LDF substep + RK4 on T.

    The uptake update freezes ``q*`` at the start-of-step state (see
    module docstring); the temperature update is classic RK4 on the
    conduction operator with the step-averaged adsorption source and a
    capacity frozen at the midpoint uptake — both constants across the
    RK4 stages, which keeps the per-step energy identity exact.

    Returns ``(T_new, q_new, info)`` where ``info`` carries the RK4
    wall-flux quadrature ``Φ`` [J/m²], the adsorption heat released over
    the step [J/m²], and the start-of-step equilibrium uptake field.
    """
    q_star = da_uptake(T, p_pa, q_sat_kg_kg, e_char_j_mol, n_da)
    decay = jnp.exp(-k_ldf_s_1 * dt_s)
    q_new = q_star + (q - q_star) * decay
    q_dot_avg = (q_new - q) / dt_s

    cap = rho_s_kg_m3 * (c_s_j_kg_k + c_pl_j_kg_k * 0.5 * (q + q_new))
    src = rho_s_kg_m3 * q_dot_avg * q_st_j_kg * dx  # W/m² per cell

    def stage(t_s):
        # The wall flux comes from the same stencil evaluation as the
        # stage RHS, so the quadrature below is consistent with the
        # T-update and the per-step energy identity holds exactly.
        div, wall = _conduction_div(t_s, dx, k_eff_w_m_k, h_wall_w_m2_k, t_f_k)
        return (div + src) / (cap * dx), wall

    a1, w1 = stage(T)
    t2 = T + 0.5 * dt_s * a1
    a2, w2 = stage(t2)
    t3 = T + 0.5 * dt_s * a2
    a3, w3 = stage(t3)
    t4 = T + dt_s * a3
    a4, w4 = stage(t4)
    T_new = T + (dt_s / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)

    phi = (dt_s / 6.0) * (w1 + 2.0 * w2 + 2.0 * w3 + w4)
    ads_heat = rho_s_kg_m3 * q_st_j_kg * dx * jnp.sum(q_new - q)
    info = {
        "wall_flux_integral": phi,
        "adsorption_heat": ads_heat,
        "q_star": q_star,
    }
    return T_new, q_new, info


# Index map of the episode scan carry (kept as a plain tuple for jit; the
# names here are the single source of truth for the layout).
_CARRY_T = 0
_CARRY_Q = 1
_CARRY_PHASE = 2
_CARRY_T_ABS = 3
_CARRY_T_END = 4
_CARRY_QCOOL_ACC = 5
_CARRY_QIN_ACC = 6
_CARRY_CYCLES = 7
_CARRY_HIST_QCOOL = 8
_CARRY_HIST_QIN = 9
_CARRY_HIST_DQ = 10
_CARRY_HIST_QADS = 11
_CARRY_HIST_QDES = 12
_CARRY_HIST_TADS = 13
_CARRY_QM_START = 14
_CARRY_QCOOL_CUM = 15
_CARRY_QIN_CUM = 16


def _push(hist, value):
    """Shift a fixed-length history left and append ``value``."""
    return jnp.concatenate([hist[1:], jnp.reshape(value, (1,))])


def initial_carry(T_init_k, q_init_kg_kg, *, t_phase_end_s, n_cycles):
    """Episode carry: fields + phase clock + per-cycle accounting histories."""
    n_cells = T_init_k.shape[0]
    zeros = jnp.zeros(n_cycles)
    return (
        T_init_k,
        q_init_kg_kg,
        jnp.asarray(ADS_PHASE),
        jnp.asarray(0.0),
        jnp.asarray(t_phase_end_s),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        jnp.mean(q_init_kg_kg),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )


def advance_carry(carry, controls, *, n_steps, dt_s, phys):
    """Advance an episode carry by ``n_steps`` physics steps under constant
    per-call controls ``controls = (t_ads_s, t_des_s, t_f_des_c)``.

    ``phys`` is a dict of static physics/configuration values (see
    :func:`simulate_bed` for the keys). Phase flips happen inside the scan
    when the absolute clock reaches ``t_phase_end``; the entered phase's
    duration is taken from the controls at flip time, so a caller may
    update the directive between calls (the env's action semantics).
    Returns ``(new_carry, ys)`` with ``ys`` shaped ``(n_steps,
    len(SERIES_CHANNELS))``.
    """
    dx = phys["dx"]
    m_s = phys["rho_s_kg_m3"] * phys["L_m"]
    p_evap = phys["p_evap_pa"]
    p_cond = phys["p_cond_pa"]
    h_fg = phys["h_fg_evap_j_kg"]
    t_f_ads = phys["t_f_ads_c"] + 273.15
    hx = phys["hx_mass_factor"]
    params = {k: phys[k] for k in BED_RHS_PARAMS}

    t_ads_x, t_des_x, t_f_des_x = (
        jnp.full((n_steps,), c) for c in controls
    )

    def body(carry, x):
        (T, q, phase, t_abs, t_end, qca, qia, ncyc, hqc, hqi, hdq, hqa, hqd,
         hta, qms, qcc, qic) = carry
        t_ads, t_des, t_f_des = x
        t_f_des = t_f_des + 273.15  # controls arrive in °C (§5.2 action)
        in_ads = phase < 0.5
        t_f = jnp.where(in_ads, t_f_ads, t_f_des)
        p_pa = jnp.where(in_ads, p_evap, p_cond)

        qm_pre = jnp.mean(q)
        T_new, q_new, info = step_bed(T, q, t_f, p_pa, dt_s, **params)
        qm_new = jnp.mean(q_new)

        # Energy accounting, J per kg adsorbent (DESIGN §4.2).
        # Cooling: h_fg·Δq̄ is already specific (kg/kg uptake swing); the
        # wall-flux integral is per m² of wall, hence its /m_s.
        dq_cool = h_fg * (qm_new - qm_pre) * in_ads
        dq_in = hx * info["wall_flux_integral"] * (1.0 - in_ads) / m_s
        qca = qca + dq_cool
        qia = qia + dq_in
        qcc = qcc + dq_cool
        qic = qic + dq_in

        t_new_abs = t_abs + dt_s
        flip = t_new_abs >= t_end
        entering = 1.0 - phase
        dur_enter = jnp.where(entering < 0.5, t_ads, t_des)
        t_end_new = jnp.where(flip, t_end + dur_enter, t_end)
        phase_new = jnp.where(flip, entering, phase)

        flip_a2d = flip * in_ads  # adsorption phase completed
        push = flip * (1.0 - in_ads)  # desorption phase completed = cycle done
        qm_now = jnp.mean(q_new)
        hqa = jnp.where(flip_a2d, _push(hqa, qm_now), hqa)
        hdq = jnp.where(flip_a2d, _push(hdq, qm_now - qms), hdq)
        hqd = jnp.where(push, _push(hqd, qm_now), hqd)
        hqc = jnp.where(push, _push(hqc, qca), hqc)
        hqi = jnp.where(push, _push(hqi, qia), hqi)
        hta = jnp.where(push, _push(hta, t_ads), hta)
        qca = jnp.where(push, 0.0, qca)
        qia = jnp.where(push, 0.0, qia)
        qms = jnp.where(push, qm_now, qms)
        ncyc = ncyc + push

        # Observation-oriented series (post-flip state).
        p_new = jnp.where(phase_new < 0.5, p_evap, p_cond)
        q_star_mean = jnp.mean(da_uptake(T_new, p_new, phys["q_sat_kg_kg"],
                                         phys["e_char_j_mol"], phys["n_da"]))
        dur_new = jnp.where(phase_new < 0.5, t_ads, t_des)
        elapsed = t_new_abs - (t_end_new - dur_new)
        ys = jnp.stack(
            (
                T_new[0],
                jnp.mean(T_new),
                jnp.max(T_new),
                qm_new,
                q_star_mean,
                p_new / p_evap,
                jnp.where(phase_new < 0.5, t_f_ads, t_f_des),
                phase_new,
                elapsed / dur_new,
                dq_cool,
                dq_in,
            )
        )
        return (
            T_new, q_new, phase_new, t_new_abs, t_end_new, qca, qia, ncyc,
            hqc, hqi, hdq, hqa, hqd, hta, qms, qcc, qic,
        ), ys

    return jax.lax.scan(body, carry, (t_ads_x, t_des_x, t_f_des_x),
                        length=n_steps)


def summary_from_carry(carry, *, p_evap_pa, p_cond_pa, h_fg_evap_j_kg):
    """Episode metrics from the final carry (DESIGN §5.2 schema).

    Metrics cover the last ``n_cycles − 1`` completed cycles (the first
    cycle is warm-up: the episode starts on the adsorption isotherm, so
    its swing is zero by construction).
    """
    hqc = carry[_CARRY_HIST_QCOOL]
    hqi = carry[_CARRY_HIST_QIN]
    hdq = carry[_CARRY_HIST_DQ]
    hqa = carry[_CARRY_HIST_QADS]
    hqd = carry[_CARRY_HIST_QDES]
    hta = carry[_CARRY_HIST_TADS]

    q_cool = jnp.sum(hqc[1:])
    q_in = jnp.sum(hqi[1:])
    t_ads = jnp.sum(hta[1:])
    safe_qin = jnp.where(q_in > 0.0, q_in, 1.0)
    cop = jnp.where(q_in > 0.0, q_cool / safe_qin, 0.0)
    safe_t = jnp.where(t_ads > 0.0, t_ads, 1.0)
    scp = jnp.where(t_ads > 0.0, q_cool / safe_t, 0.0)
    n_last = hqc.shape[0] - 1
    safe_n = max(n_last, 1)
    return {
        "COP": cop,
        "SCP_W_kg": scp,
        "delta_q": jnp.sum(hdq[1:]) / safe_n,
        "q_ads": jnp.sum(hqa[1:]) / safe_n,
        "q_des": jnp.sum(hqd[1:]) / safe_n,
        "P_evap_kPa": p_evap_pa / 1000.0,
        "P_cond_kPa": p_cond_pa / 1000.0,
        "h_fg_MJ_kg": h_fg_evap_j_kg / 1e6,
        "Q_cool_J_kg": q_cool,
        "Q_in_J_kg": q_in,
    }


def simulate_bed(
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
    t_f_des_c,
    t_ads_s,
    t_des_s,
    dt_s,
    n_cycles=4,
    T_init_k=None,
    q_init_kg_kg=None,
    n_steps=None,
    collect_trace=False,
):
    """Roll out whole adsorption episodes with :func:`jax.lax.scan`.

    The episode runs ``n_cycles`` adsorption/desorption pairs starting on
    the adsorption phase at ``(T_init, q*(T_init, P_evap))`` — by default
    the bed is pre-equilibrated with the adsorption-side fluid at
    ``t_cond_c``. Returns ``{"summary": …, "series": …}``; ``summary`` maps
    the episode metric keys to JAX scalars (gradients flow), ``series`` is
    a dict of per-step channels (decimated to ≤ 2048 samples) or ``None``.
    """
    n_cells = int(n_cells)
    n_cycles = int(n_cycles)
    dx = L_m / n_cells
    m_s = rho_s_kg_m3 * L_m
    p_evap = water_sat_pressure_pa(t_evap_c + 273.15)
    p_cond = water_sat_pressure_pa(t_cond_c + 273.15)
    h_fg = water_h_fg_j_kg(t_evap_c + 273.15)

    if n_steps is None:
        try:
            t_total = float(t_ads_s) + float(t_des_s)
        except TypeError as exc:  # tracer durations need a static horizon
            raise TypeError(
                "simulate_bed needs static t_ads_s/t_des_s (or an explicit "
                "n_steps) because the scan length must be static"
            ) from exc
        if min(float(t_ads_s), float(t_des_s)) < 2.0 * float(dt_s):
            raise ValueError(
                "phase durations must exceed ~2 dt_s for the valve-flip "
                "logic (got t_ads_s=%g, t_des_s=%g, dt_s=%g)"
                % (float(t_ads_s), float(t_des_s), float(dt_s))
            )
        n_steps = int(round(n_cycles * t_total / float(dt_s))) + 8

    if T_init_k is None:
        T_init_k = jnp.full((n_cells,), t_cond_c + 273.15)
    if q_init_kg_kg is None:
        q_init_kg_kg = da_uptake(
            T_init_k, p_evap, q_sat_kg_kg, e_char_j_mol, n_da
        )

    phys = {
        "dx": dx,
        "L_m": L_m,
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
        "hx_mass_factor": hx_mass_factor,
        "t_f_ads_c": t_f_ads_c,
        "p_evap_pa": p_evap,
        "p_cond_pa": p_cond,
        "h_fg_evap_j_kg": h_fg,
    }

    carry = initial_carry(
        T_init_k, q_init_kg_kg, t_phase_end_s=t_ads_s, n_cycles=n_cycles
    )
    controls = (t_ads_s, t_des_s, t_f_des_c)
    carry_f, ys = advance_carry(
        carry, controls, n_steps=n_steps, dt_s=dt_s, phys=phys
    )
    summary = summary_from_carry(
        carry_f, p_evap_pa=p_evap, p_cond_pa=p_cond, h_fg_evap_j_kg=h_fg
    )

    series = None
    if collect_trace:
        stride = max(1, n_steps // 2048)
        thinned = ys[::stride]
        series = {
            name: thinned[:, i] for i, name in enumerate(SERIES_CHANNELS)
        }
    return {"summary": summary, "series": series, "m_s_kg_m2": m_s}


__all__ = [
    "ADS_PHASE",
    "BED_RHS_PARAMS",
    "DES_PHASE",
    "SERIES_CHANNELS",
    "advance_carry",
    "bed_rhs",
    "check_timestep",
    "initial_carry",
    "max_timestep",
    "simulate_bed",
    "step_bed",
    "summary_from_carry",
    "volumetric_capacity",
]
