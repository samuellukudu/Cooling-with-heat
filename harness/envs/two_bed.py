"""TwoBed-v0 — the counter-phase two-bed system as a Problem + Gym env
(DESIGN §4.3/§5.3).

One class, two faces, exactly like :mod:`.bed1d`:

- harness Problem (``spec``/``design_space``/``evaluate``/``metrics_jax``/
  ``rollout``): the whole episode is one ``jax.lax.scan`` rollout of
  :func:`harness.physics.system.simulate_two_bed`, so the grad backend
  differentiates episode metrics through every valve flip and through
  the heat-recovery coupling;
- dynamic control face (``reset``/``step``, NumPy boundary): a
  Gymnasium-compatible environment. Wrap with :class:`TwoBedGymEnv` for
  ``check_env``.

Action semantics (§5.3, v0 continuous): the action is
``(t_switch [s], T_f,des [°C], t_rec [s])``. ``t_switch`` sets the
duration of both system phases that *start* afterwards (the running
phase keeps the end time it was started with); ``T_f,des`` and
``t_rec`` apply from the next physics substep. ``t_rec = 0`` disables
heat recovery for phases started afterwards; the recovery conductance
itself (``recovery_ua_w_m2_k``) is a design parameter. The §5.3
alternatives — per-step valve bits, source-allocation fraction, and a
chilled-water load model — are deferred (the valve-bit hook exists at
the physics level for schedule policies, H2.2; the load model arrives
with the stochastic-profile experiments).

Reward (§5.3, v1): ``r_t = ΔQ_cool − λ·ΔQ_in`` per step, J per kg total
adsorbent (both beds' books summed in absolute energy). The
``unmet_setpoint`` term is 0 — there is no load model in v1 — and
``unmet_load_frac`` is reported as 0.0 for schema stability.

Metrics (§5.3): the per-bed keys plus ``Q_rec_J_m2`` (total heat moved
between beds), ``unmet_load_frac`` (0.0 in v1) and ``recovery_gain`` =
COP with recovery − COP of the identical machine with
``recovery_ua_w_m2_k = 0``. The gain needs a counterfactual rollout, so
it is computed in ``evaluate``/``rollout`` (and in ``metrics_jax`` when
the problem is built with ``counterfactual=True``); build with
``counterfactual=False`` for cheap optimization loops that don't need
it.

Known trade-off (V6/H2.1 finding): heat recovery raises COP only when
the swings complete with slack (phases long against the swing
timescales) — the exchanged heat then directly offsets source draw at
unchanged swing. In transient-bound regimes (short phases, slow
kinetics) the film-off window delays the swings, and the throughput
loss outweighs the heat saving: ``recovery_gain`` goes negative. Report
it alongside ``t_rec``/``recovery_ua_w_m2_k``; schedule optimization
(H2.2) picks the operating regime.

Numerics: ``dt_phys`` is fixed at construction from the worst case
within the declared per-bed design bounds (same rule as Bed1D).
"""

from __future__ import annotations

import functools
from typing import Any, Mapping

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from ..materials import MaterialParams, get_material
from ..physics import system
from ..physics.thermo import (
    CP_ADSORBENT,
    CP_LIQUID,
    da_uptake,
    water_h_fg_j_kg,
    water_sat_pressure_pa,
)
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import ActionSpec, DesignSpace, EpisodeTrace, ProblemSpec, Sensor
from .bed1d import (
    BED_DESIGN_BOUNDS,
    BED_N_CELLS,
    BED_RHO_S_KG_M3,
    DESIGN_KEYS as BED_DESIGN_KEYS,
    OBS_SENSORS as BED_OBS_SENSORS,
    T_SWITCH_BOUNDS,
)

TWO_BED_SCHEMA_VERSION = 1

# Episode metric schema: the per-bed keys plus the system keys (§5.3).
TWO_BED_METRIC_KEYS = (
    "COP",
    "SCP_W_kg",
    "delta_q",
    "q_ads",
    "q_des",
    "P_evap_kPa",
    "P_cond_kPa",
    "h_fg_MJ_kg",
    "Q_cool_J_kg",
    "Q_in_J_kg",
    "Q_rec_J_m2",
    "recovery_gain",
    "unmet_load_frac",
)
_NO_COUNTERFACTUAL_KEYS = tuple(k for k in TWO_BED_METRIC_KEYS if k != "recovery_gain")

DESIGN_KEYS = (
    *(f"A_{k}" for k in BED_DESIGN_KEYS),
    *(f"B_{k}" for k in BED_DESIGN_KEYS),
    "recovery_ua_w_m2_k",
)

DESIGN_BOUNDS = {
    **{f"A_{k}": v for k, v in BED_DESIGN_BOUNDS.items()},
    **{f"B_{k}": v for k, v in BED_DESIGN_BOUNDS.items()},
    "recovery_ua_w_m2_k": (0.0, 500.0),
}

T_REC_BOUNDS = (0.0, 600.0)

# §5.3 observation: per-bed §5.2 sensor vectors + system channels.
OBS_SENSORS = (
    *(Sensor(f"A_{s.name}", s.unit, s.lo, s.hi) for s in BED_OBS_SENSORS),
    *(Sensor(f"B_{s.name}", s.unit, s.lo, s.hi) for s in BED_OBS_SENSORS),
    Sensor("T_source", "K", 250.0, 450.0),
    Sensor("t_abs_s", "s", 0.0, 1.0e7),
)
_OBS_LO = np.array([s.lo for s in OBS_SENSORS], dtype=np.float32)
_OBS_HI = np.array([s.hi for s in OBS_SENSORS], dtype=np.float32)

_N_BED_OBS = len(BED_OBS_SENSORS)
_A_DQ_COOL = 4 + 9  # system series: A block offset + dq_cool_j_kg
_A_DQ_IN = _A_DQ_COOL + 1
_B_DQ_COOL = 15 + 9
_B_DQ_IN = _B_DQ_COOL + 1


def _bed_defaults(material: MaterialParams, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}q_sat_kg_kg": float(material.q_sat_kg_kg),
        f"{prefix}Q_st_j_kg": float(material.q_st_j_kg),
        f"{prefix}e_char_j_mol": float(material.e_char_j_mol),
        f"{prefix}n_da": float(material.n_da),
        f"{prefix}k_ldf_s_1": float(material.k_ldf_s_1) if material.k_ldf_s_1 is not None else 0.05,
        f"{prefix}L_m": 0.002,
        f"{prefix}k_eff_w_m_k": float(material.k_eff_w_m_k) if material.k_eff_w_m_k is not None else 0.3,
        f"{prefix}h_wall_w_m2_k": 500.0,
        f"{prefix}hx_mass_factor": 1.35,
    }


class TwoBed:
    """Counter-phase two-bed adsorption system over materials × profile."""

    def __init__(
        self,
        material: "str | MaterialParams" = "anchor:Silica gel RD",
        material_b: "str | MaterialParams | None" = None,
        profile: "str | ApplicationProfile" = "cpu",
        design: Mapping[str, float] | None = None,
        *,
        dt_ctrl_s: float = 5.0,
        n_cycles: int = 4,
        lam: float = 0.0,
        n_cells: int = BED_N_CELLS,
        dt_phys_s: float | None = None,
        counterfactual: bool = True,
    ):
        self.material = get_material(material)
        self.material_b = get_material(material_b) if material_b is not None else self.material
        self.profile = get_profile(profile)
        self.dt_ctrl_s = float(dt_ctrl_s)
        self.n_cycles = int(n_cycles)
        self.lam = float(lam)
        self.n_cells = int(n_cells)
        self.counterfactual = bool(counterfactual)

        metric_keys = TWO_BED_METRIC_KEYS if self.counterfactual else _NO_COUNTERFACTUAL_KEYS
        self.spec = ProblemSpec(
            name="TwoBed-v0",
            kind="dynamic",
            obs_spec=OBS_SENSORS,
            action_spec=ActionSpec(
                kind="continuous",
                names=("t_switch_s", "t_f_des_c", "t_rec_s"),
                lo=(T_SWITCH_BOUNDS[0], self.profile.t_fluid_min_c, T_REC_BOUNDS[0]),
                hi=(T_SWITCH_BOUNDS[1], self.profile.t_fluid_max_c, T_REC_BOUNDS[1]),
            ),
            metric_keys=metric_keys,
            schema_version=TWO_BED_SCHEMA_VERSION,
        )

        defaults = self._design_defaults()
        if design is not None:
            defaults = DesignSpace(keys=DESIGN_KEYS, defaults=defaults).merge(design)
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS, defaults=defaults, bounds=DESIGN_BOUNDS
        )

        if dt_phys_s is None:
            lo_L, _ = BED_DESIGN_BOUNDS["L_m"]
            _, hi_k = BED_DESIGN_BOUNDS["k_eff_w_m_k"]
            dx = lo_L / self.n_cells
            dt_phys_s = dx * dx * BED_RHO_S_KG_M3 * CP_ADSORBENT / (2.0 * hi_k)
        self.dt_phys_s = float(dt_phys_s)
        self._n_sub = max(1, int(round(self.dt_ctrl_s / self.dt_phys_s)))
        self._carry: tuple | None = None
        self._ctrl = self.default_controls()
        self._m_s_a = BED_RHO_S_KG_M3 * self.design_space.defaults["A_L_m"]
        self._m_s_b = BED_RHO_S_KG_M3 * self.design_space.defaults["B_L_m"]
        self._advance_jit = jax.jit(
            functools.partial(
                system.advance_two_carry,
                n_steps=self._n_sub,
                dt_s=self.dt_phys_s,
                phys_a=self._bed_phys("A_"),
                phys_b=self._bed_phys("B_"),
                recovery_ua_w_m2_k=self.design_space.defaults["recovery_ua_w_m2_k"],
                use_req=False,
            )
        )

    # -- design/controls ---------------------------------------------------

    def _design_defaults(self) -> dict[str, float]:
        return {
            **_bed_defaults(self.material, "A_"),
            **_bed_defaults(self.material_b, "B_"),
            "recovery_ua_w_m2_k": 0.0,
        }

    def default_controls(self) -> dict[str, float]:
        """§5.3 controls at profile defaults: shared phase duration, the
        profile's regeneration temperature, recovery off."""
        return {
            "t_ads_s": self.profile.cycle_time_s,
            "t_des_s": self.profile.cycle_time_s,
            "t_f_des_c": self.profile.t_des_c,
            "t_rec_s": 0.0,
        }

    def _bed_phys(self, prefix: str) -> dict[str, Any]:
        d = self.design_space.defaults
        return system.bed_phys(
            q_sat_kg_kg=d[f"{prefix}q_sat_kg_kg"],
            q_st_j_kg=d[f"{prefix}Q_st_j_kg"],
            e_char_j_mol=d[f"{prefix}e_char_j_mol"],
            n_da=d[f"{prefix}n_da"],
            k_ldf_s_1=d[f"{prefix}k_ldf_s_1"],
            rho_s_kg_m3=BED_RHO_S_KG_M3,
            c_s_j_kg_k=CP_ADSORBENT,
            c_pl_j_kg_k=CP_LIQUID,
            k_eff_w_m_k=d[f"{prefix}k_eff_w_m_k"],
            h_wall_w_m2_k=d[f"{prefix}h_wall_w_m2_k"],
            L_m=d[f"{prefix}L_m"],
            n_cells=self.n_cells,
            hx_mass_factor=d[f"{prefix}hx_mass_factor"],
            t_evap_c=self.profile.t_evap_c,
            t_cond_c=self.profile.t_cond_c,
            t_f_ads_c=self.profile.t_cond_c,
        )

    def _bed_dicts(self, merged: Mapping[str, Any]) -> tuple[dict, dict]:
        """Per-bed keyword dicts for :func:`system.simulate_two_bed` (values
        may be tracers on the grad path)."""
        def one(prefix: str) -> dict:
            return {
                "q_sat_kg_kg": merged[f"{prefix}q_sat_kg_kg"],
                "q_st_j_kg": merged[f"{prefix}Q_st_j_kg"],
                "e_char_j_mol": merged[f"{prefix}e_char_j_mol"],
                "n_da": merged[f"{prefix}n_da"],
                "k_ldf_s_1": merged[f"{prefix}k_ldf_s_1"],
                "rho_s_kg_m3": BED_RHO_S_KG_M3,
                "c_s_j_kg_k": CP_ADSORBENT,
                "c_pl_j_kg_k": CP_LIQUID,
                "k_eff_w_m_k": merged[f"{prefix}k_eff_w_m_k"],
                "h_wall_w_m2_k": merged[f"{prefix}h_wall_w_m2_k"],
                "L_m": merged[f"{prefix}L_m"],
                "n_cells": self.n_cells,
                "hx_mass_factor": merged[f"{prefix}hx_mass_factor"],
            }
        return one("A_"), one("B_")

    # -- harness Problem face ----------------------------------------------

    def _simulate(self, merged, ctrl, *, recovery_ua_w_m2_k, collect_trace=False):
        bed_a, bed_b = self._bed_dicts(merged)
        return system.simulate_two_bed(
            bed_a=bed_a,
            bed_b=bed_b,
            t_evap_c=self.profile.t_evap_c,
            t_cond_c=self.profile.t_cond_c,
            t_f_ads_c=self.profile.t_cond_c,
            t_f_des_c=ctrl["t_f_des_c"],
            t_ads_s=ctrl["t_ads_s"],
            t_des_s=ctrl["t_des_s"],
            t_rec_s=ctrl["t_rec_s"],
            recovery_ua_w_m2_k=recovery_ua_w_m2_k,
            dt_s=self.dt_phys_s,
            n_cycles=self.n_cycles,
            collect_trace=collect_trace,
        )

    def metrics_jax(self, design: Mapping[str, Any] | None = None,
                    controls: Mapping[str, float] | None = None) -> dict[str, Any]:
        merged = self.design_space.merge(design)
        ctrl = {**self.default_controls(), **(controls or {})}
        out = self._simulate(merged, ctrl,
                             recovery_ua_w_m2_k=merged["recovery_ua_w_m2_k"])
        summary = dict(out["summary"])
        if self.counterfactual:
            norec = self._simulate(merged, ctrl, recovery_ua_w_m2_k=0.0)
            summary["recovery_gain"] = summary["COP"] - norec["summary"]["COP"]
        return summary

    def evaluate(self, design: Mapping[str, float] | None = None,
                 controls: Mapping[str, float] | None = None) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics_jax(design, controls).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del n_steps  # episode length is n_cycles system cycles (§5.3)
        merged = self.design_space.merge(design)
        ctrl = {**self.default_controls(), **(controls or {})}
        out = self._simulate(merged, ctrl,
                             recovery_ua_w_m2_k=merged["recovery_ua_w_m2_k"],
                             collect_trace=True)
        summary = {k: float(v) for k, v in out["summary"].items()}
        if self.counterfactual:
            norec = self._simulate(merged, ctrl, recovery_ua_w_m2_k=0.0)
            summary["recovery_gain"] = summary["COP"] - float(norec["summary"]["COP"])
        series = {k: np.asarray(v) for k, v in (out["series"] or {}).items()}
        return EpisodeTrace(series=series, summary=summary)

    # -- dynamic face (numpy boundary) --------------------------------------

    def reset(self, *, seed=None, options=None):
        """Start a fresh episode at a steady-state-like flip instant
        (``physics.system`` module docstring): bed A hot and empty entering
        adsorption, bed B cold and loaded entering desorption."""
        del seed, options  # deterministic physics — nothing to sample in v0
        ctrl = self.default_controls()
        self._ctrl = ctrl
        phys_a, phys_b = self._bed_phys("A_"), self._bed_phys("B_")
        self._carry = jax.device_get(
            system.initial_two_carry(
                phys_a, phys_b,
                t_phase0_s=ctrl["t_ads_s"],
                t_des_end_k=ctrl["t_f_des_c"] + 273.15,
                n_cycles=self.n_cycles,
            )
        )
        return self._observation(), {"phase": "A_ads", "t_abs": 0.0}

    def step(self, action):
        if self._carry is None:
            raise RuntimeError("call reset() before step()")
        action = np.asarray(action)
        if action.shape != (3,):
            raise ValueError(f"continuous action must have shape (3,), got {action.shape}")
        t_switch = float(np.clip(action[0], *T_SWITCH_BOUNDS))
        t_f_des = float(np.clip(action[1], self.profile.t_fluid_min_c,
                                self.profile.t_fluid_max_c))
        t_rec = float(np.clip(action[2], *T_REC_BOUNDS))
        # t_switch sets the duration of phases starting afterwards; the
        # fluid temperature and recovery window apply from the next
        # substep (module docstring).
        self._ctrl = {**self._ctrl, "t_ads_s": t_switch, "t_des_s": t_switch,
                      "t_f_des_c": t_f_des, "t_rec_s": t_rec}

        ctrl = self._ctrl
        carry, ys = self._advance_jit(
            self._carry,
            (jnp.asarray(ctrl["t_ads_s"]), jnp.asarray(ctrl["t_des_s"]),
             jnp.asarray(ctrl["t_f_des_c"]), jnp.asarray(ctrl["t_rec_s"]),
             jnp.asarray(0.0), jnp.asarray(0.0)),
        )
        self._carry = jax.device_get(carry)
        ys = jax.device_get(ys)

        dqc = float(ys[:, _A_DQ_COOL].sum()) * self._m_s_a \
            + float(ys[:, _B_DQ_COOL].sum()) * self._m_s_b
        dqi = float(ys[:, _A_DQ_IN].sum()) * self._m_s_a \
            + float(ys[:, _B_DQ_IN].sum()) * self._m_s_b
        m_tot = self._m_s_a + self._m_s_b
        reward = dqc / m_tot - self.lam * dqi / m_tot  # §5.3 v1 reward

        cycles_done = float(self._carry[0][7])  # bed A's completed cycles
        terminated = bool(cycles_done >= self.n_cycles)
        info = {
            "t_abs_s": float(self._carry[0][3]),
            "phase": "A_ads" if self._carry[0][2] < 0.5 else "A_des",
            "cycles_done": int(cycles_done),
            "dq_cool_j_kg": dqc / m_tot,
            "dq_in_j_kg": dqi / m_tot,
        }
        if terminated:
            info["metrics"] = self._summary_numpy()
        return self._observation(), float(reward), terminated, False, info

    def _summary_numpy(self) -> dict[str, float]:
        """Episode metrics from the numpy carry pair (same schema as
        ``physics.system.summary_from_two_carry``, no jax dispatch)."""
        ca, cb, q_rec = self._carry
        n = ca[8].shape[0] - 1
        safe_n = max(n, 1)
        m_a, m_b = self._m_s_a, self._m_s_b
        m_tot = m_a + m_b
        qc = float(ca[8][1:].sum()) * m_a + float(cb[8][1:].sum()) * m_b
        qi = float(ca[9][1:].sum()) * m_a + float(cb[9][1:].sum()) * m_b
        t_ads_w = float(ca[13][1:].sum()) * m_a + float(cb[13][1:].sum()) * m_b
        prof = self.profile
        p_evap = float(water_sat_pressure_pa(prof.t_evap_c + 273.15))
        p_cond = float(water_sat_pressure_pa(prof.t_cond_c + 273.15))
        return {
            "COP": qc / qi if qi > 0 else 0.0,
            "SCP_W_kg": qc / t_ads_w if t_ads_w > 0 else 0.0,
            "delta_q": (float(ca[10][1:].sum()) * m_a + float(cb[10][1:].sum()) * m_b)
            / (m_tot * safe_n),
            "q_ads": (float(ca[11][1:].sum()) * m_a + float(cb[11][1:].sum()) * m_b)
            / (m_tot * safe_n),
            "q_des": (float(ca[12][1:].sum()) * m_a + float(cb[12][1:].sum()) * m_b)
            / (m_tot * safe_n),
            "P_evap_kPa": p_evap / 1e3,
            "P_cond_kPa": p_cond / 1e3,
            "h_fg_MJ_kg": float(water_h_fg_j_kg(prof.t_evap_c + 273.15)) / 1e6,
            "Q_cool_J_kg": qc / m_tot,
            "Q_in_J_kg": qi / m_tot,
            "Q_rec_J_m2": float(q_rec),
            "recovery_gain": 0.0,  # counterfactual needs a second rollout; see metrics_jax
            "unmet_load_frac": 0.0,
        }

    def _bed_obs(self, carry, prefix: str, swapped: bool) -> list[float]:
        """§5.2 sensor vector for one bed from its numpy carry."""
        ctrl = self._ctrl
        d = self.design_space.defaults
        T, q = carry[0], carry[1]
        phase = float(carry[2])
        in_ads = phase < 0.5
        p_evap = float(water_sat_pressure_pa(self.profile.t_evap_c + 273.15))
        p_cond = float(water_sat_pressure_pa(self.profile.t_cond_c + 273.15))
        p = p_evap if in_ads else p_cond
        q_sat = d[f"{prefix}q_sat_kg_kg"]
        e_char = d[f"{prefix}e_char_j_mol"]
        n_da = d[f"{prefix}n_da"]
        q_star_mean = float(jnp.mean(da_uptake(T, p, q_sat, e_char, n_da)))
        # Role-keyed phase durations: bed B's adsorption phases are the
        # σ = 1 system phases (module docstring).
        if swapped:
            dur = ctrl["t_des_s"] if in_ads else ctrl["t_ads_s"]
        else:
            dur = ctrl["t_ads_s"] if in_ads else ctrl["t_des_s"]
        frac = (float(carry[3]) - float(carry[18])) / dur if dur > 0 else 0.0
        t_f = (ctrl["t_f_des_c"] if not in_ads else self.profile.t_cond_c) + 273.15
        return [
            float(T[0]), float(T.mean()), float(T.max()),
            float(q.mean()), q_star_mean, p / p_evap, frac, t_f,
            float(carry[15]), float(carry[16]),
        ]

    def _observation(self) -> np.ndarray:
        """Sensors are recomputed from the carries (reset and step alike)."""
        ca, cb, _ = self._carry
        row = [
            *self._bed_obs(ca, "A_", swapped=False),
            *self._bed_obs(cb, "B_", swapped=True),
            self._ctrl["t_f_des_c"] + 273.15,
            float(ca[3]),
        ]
        return np.array(row, dtype=np.float32)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(_OBS_LO, _OBS_HI, dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=np.array([T_SWITCH_BOUNDS[0], self.profile.t_fluid_min_c,
                          T_REC_BOUNDS[0]], dtype=np.float32),
            high=np.array([T_SWITCH_BOUNDS[1], self.profile.t_fluid_max_c,
                           T_REC_BOUNDS[1]], dtype=np.float32),
            dtype=np.float32,
        )


def build_two_bed(material: "str | MaterialParams" = "anchor:Silica gel RD",
                  material_b: "str | MaterialParams | None" = None,
                  profile: "str | ApplicationProfile" = "cpu",
                  **kwargs) -> TwoBed:
    return TwoBed(material, material_b, profile, **kwargs)


REGISTRIES["envs"].register("TwoBed-v0", build_two_bed)


class TwoBedGymEnv(gym.Env):
    """Gymnasium wrapper around :class:`TwoBed` (numpy boundary, §5.3)."""

    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        self._env = TwoBed(**kwargs)

    @property
    def problem(self) -> TwoBed:
        return self._env

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        return self._env.step(action)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._env.action_space


def _register_gymnasium() -> None:
    try:
        gym.register(
            "TwoBed-v0",
            entry_point="harness.envs.two_bed:TwoBedGymEnv",
            kwargs={"material": "anchor:Silica gel RD", "material_b": None,
                    "profile": "cpu"},
        )
    except gym.error.Error:
        pass  # already registered (e.g. re-import)


_register_gymnasium()
