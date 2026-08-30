"""Bed1D-v0 — the dynamic 1-D adsorber bed as a Problem + Gym env
(DESIGN §4.2/§5.2).

One class, two faces:

- harness Problem (``spec``/``design_space``/``evaluate``/``metrics_jax``/
  ``rollout``): the whole episode is one ``jax.lax.scan`` rollout
  (:func:`harness.physics.bed1d.simulate_bed`), so the grad backend can
  differentiate episode metrics through every valve flip;
- dynamic control face (``reset``/``step``, NumPy boundary): a Gymnasium
  -compatible environment for schedule/control experiments (and the rl
  backend from H2 on). Wrap with :class:`Bed1DGymEnv` for
  ``gymnasium.utils.env_checker.check_env``.

Action semantics (§5.2, continuous mode): the action is
``(T_f,des [°C], t_switch [s])``. The fluid temperature applies from the
next physics substep (the loop's lag is ignored in v1); ``t_switch`` sets
the duration of every phase that *starts* afterwards — the running phase
keeps the end time it was started with. Discrete mode (``action_mode=
"discrete"``) exposes the §5.2 alternative ``{connect_ads, connect_des}``:
the requested valve connection is taken at the next substep, durations
unchanged.

Reward (§5.2): ``r_t = ΔQ_cool − λ·ΔQ_in`` per step, J per kg adsorbent.
``λ = 0`` maximises SCP; ``λ = 1/COP_target`` prices heat. The
setpoint-violation term ``μ·violation`` activates with TwoBed (H2) — there
is no load setpoint in a single-bed episode, so it is zero here.

Numerics: ``dt_phys`` is fixed at construction from the *worst case within
the declared design bounds* (thinnest bed, highest conductivity, dry
capacity) so every design a backend may propose satisfies the conduction
CFL and the ``k_LDF·Δt ≲ 0.25`` split-scheme bound without re-jitting; see
``harness.physics.bed1d`` for the stability analysis.
"""

from __future__ import annotations

import functools
from typing import Any, Mapping

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from ..materials import MaterialParams, get_material
from ..physics import bed1d
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

BED1D_SCHEMA_VERSION = 1

# Episode metric schema (superset of the Cycle0D oracle keys, same order
# of meaning — see physics.bed1d.summary_from_carry). Q_cool/Q_in are
# totals over the last n_cycles−1 (warm-up) cycles, per kg adsorbent.
BED_METRIC_KEYS = (
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
)

DESIGN_KEYS = (
    "q_sat_kg_kg",
    "Q_st_j_kg",
    "e_char_j_mol",
    "n_da",
    "k_ldf_s_1",
    "L_m",
    "k_eff_w_m_k",
    "h_wall_w_m2_k",
    "hx_mass_factor",
)

BED_DESIGN_BOUNDS = {
    "q_sat_kg_kg": (0.08, 0.90),  # mirrored from the Cycle0D screen envelope
    "Q_st_j_kg": (2.30e6, 4.10e6),
    "e_char_j_mol": (2000.0, 20000.0),
    "n_da": (1.0, 4.0),
    "k_ldf_s_1": (0.01, 20.0),  # keeps k·dt ≤ 0.04 at the worst-case dt
    "L_m": (0.001, 0.004),
    "k_eff_w_m_k": (0.15, 0.6),
    "h_wall_w_m2_k": (50.0, 3000.0),
    "hx_mass_factor": (1.0, 2.0),
}

# Fixed v1 bed solids/geometry (not design keys): packed silica-gel-like
# bed; rho_s is the dry adsorbent mass per bed volume.
BED_RHO_S_KG_M3 = 600.0
BED_N_CELLS = 16

# §5.2 action bounds.
T_SWITCH_BOUNDS = (60.0, 1800.0)

# §5.2 observation: sensor-like channels a controller could measure.
OBS_SENSORS = (
    Sensor("T_wall", "K", 250.0, 500.0),
    Sensor("T_bed_mean", "K", 250.0, 500.0),
    Sensor("T_bed_max", "K", 250.0, 650.0),  # headroom for valve-burst spikes
    Sensor("q_mean", "kg/kg", 0.0, 2.0),
    Sensor("q_star_mean", "kg/kg", 0.0, 2.0),
    Sensor("P_over_P_evap", "-", 0.0, 60.0),
    Sensor("time_in_phase_over_t_switch", "-", 0.0, 2.0),
    Sensor("T_fluid", "K", 250.0, 450.0),
    Sensor("Q_cool_cum", "J/kg", 0.0, 1.0e8),
    Sensor("Q_in_cum", "J/kg", 0.0, 1.0e8),
)

_OBS_LO = np.array([s.lo for s in OBS_SENSORS], dtype=np.float32)
_OBS_HI = np.array([s.hi for s in OBS_SENSORS], dtype=np.float32)


def _bed_phys(material: MaterialParams, profile: ApplicationProfile,
              merged: Mapping[str, float], t_f_ads_c: float, t_f_des_c: float,
              t_ads_s: float, t_des_s: float, n_cells: int) -> dict[str, Any]:
    """Assemble the ``simulate_bed``/``advance_carry`` keyword set.

    Values may be JAX tracers (grad path) or floats; the bed-level
    fallbacks exist because anchor materials carry no transport data.
    """
    return dict(
        q_sat_kg_kg=merged["q_sat_kg_kg"],
        q_st_j_kg=merged["Q_st_j_kg"],
        e_char_j_mol=merged["e_char_j_mol"],
        n_da=merged["n_da"],
        k_ldf_s_1=merged["k_ldf_s_1"],
        rho_s_kg_m3=BED_RHO_S_KG_M3,
        c_s_j_kg_k=CP_ADSORBENT,
        c_pl_j_kg_k=CP_LIQUID,
        k_eff_w_m_k=merged["k_eff_w_m_k"],
        h_wall_w_m2_k=merged["h_wall_w_m2_k"],
        L_m=merged["L_m"],
        n_cells=n_cells,
        hx_mass_factor=merged["hx_mass_factor"],
        t_evap_c=profile.t_evap_c,
        t_cond_c=profile.t_cond_c,
        t_f_ads_c=t_f_ads_c,
        t_f_des_c=t_f_des_c,
        t_ads_s=t_ads_s,
        t_des_s=t_des_s,
    )


class Bed1D:
    """Dynamic single-bed adsorption episode over a material × profile pair."""

    def __init__(
        self,
        material: "str | MaterialParams" = "anchor:Silica gel RD",
        profile: "str | ApplicationProfile" = "cpu",
        design: Mapping[str, float] | None = None,
        *,
        action_mode: str = "continuous",
        dt_ctrl_s: float = 5.0,
        n_cycles: int = 4,
        lam: float = 0.0,
        n_cells: int = BED_N_CELLS,
        dt_phys_s: float | None = None,
    ):
        if action_mode not in ("continuous", "discrete"):
            raise ValueError(f"unknown action_mode {action_mode!r}")
        self.material = get_material(material)
        self.profile = get_profile(profile)
        self.action_mode = action_mode
        self.dt_ctrl_s = float(dt_ctrl_s)
        self.n_cycles = int(n_cycles)
        self.lam = float(lam)
        self.n_cells = int(n_cells)

        self.spec = ProblemSpec(
            name="Bed1D-v0",
            kind="dynamic",
            obs_spec=OBS_SENSORS,
            action_spec=(
                ActionSpec(
                    kind="continuous",
                    names=("t_f_des_c", "t_switch_s"),
                    lo=(self.profile.t_fluid_min_c, T_SWITCH_BOUNDS[0]),
                    hi=(self.profile.t_fluid_max_c, T_SWITCH_BOUNDS[1]),
                )
                if action_mode == "continuous"
                else ActionSpec(kind="discrete", choices=("connect_ads", "connect_des"))
            ),
            metric_keys=BED_METRIC_KEYS,
            schema_version=BED1D_SCHEMA_VERSION,
        )
        defaults = self._design_defaults()
        if design is not None:
            # Constructor design overrides become the problem's defaults
            # (the base backends vary); unknown keys raise here.
            defaults = DesignSpace(keys=DESIGN_KEYS, defaults=defaults).merge(design)
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS,
            defaults=defaults,
            bounds=BED_DESIGN_BOUNDS,
        )

        # Static numerics: worst case within the declared bounds so any
        # design a backend proposes stays inside the conduction CFL and
        # the k_LDF·dt split-scheme bound (module docstring).
        if dt_phys_s is None:
            lo_L, _ = BED_DESIGN_BOUNDS["L_m"]
            _, hi_k = BED_DESIGN_BOUNDS["k_eff_w_m_k"]
            dx = lo_L / self.n_cells
            dt_phys_s = dx * dx * BED_RHO_S_KG_M3 * CP_ADSORBENT / (2.0 * hi_k)
        self.dt_phys_s = float(dt_phys_s)
        self._n_sub = max(1, int(round(self.dt_ctrl_s / self.dt_phys_s)))
        self._carry: tuple | None = None
        self._ctrl = self.default_controls()
        self._advance_jit = jax.jit(
            functools.partial(
                bed1d.advance_carry,
                n_steps=self._n_sub,
                dt_s=self.dt_phys_s,
                phys=self._static_phys(),
            )
        )

    # -- design/controls --------------------------------------------------

    def _design_defaults(self) -> dict[str, float]:
        m = self.material
        return {
            "q_sat_kg_kg": float(m.q_sat_kg_kg),
            "Q_st_j_kg": float(m.q_st_j_kg),
            "e_char_j_mol": float(m.e_char_j_mol),
            "n_da": float(m.n_da),
            "k_ldf_s_1": float(m.k_ldf_s_1) if m.k_ldf_s_1 is not None else 0.05,
            "L_m": 0.002,
            "k_eff_w_m_k": float(m.k_eff_w_m_k) if m.k_eff_w_m_k is not None else 0.3,
            "h_wall_w_m2_k": 500.0,
            "hx_mass_factor": 1.35,
        }

    def default_controls(self) -> dict[str, float]:
        """§5.2 controls at profile defaults: fluid setpoints and the
        profile's half-cycle time for both phases."""
        return {
            "t_f_ads_c": self.profile.t_cond_c,
            "t_f_des_c": self.profile.t_des_c,
            "t_ads_s": self.profile.cycle_time_s,
            "t_des_s": self.profile.cycle_time_s,
        }

    def _static_phys(self) -> dict[str, Any]:
        """Phys dict with the static (non-design) entries for the jitted
        step advance; design entries are filled per call."""
        prof = self.profile
        p_evap = float(water_sat_pressure_pa(prof.t_evap_c + 273.15))
        p_cond = float(water_sat_pressure_pa(prof.t_cond_c + 273.15))
        phys = _bed_phys(
            self.material, prof, self.design_space.defaults,
            self._ctrl["t_f_ads_c"], self._ctrl["t_f_des_c"],
            self._ctrl["t_ads_s"], self._ctrl["t_des_s"],
            self.n_cells,
        )
        phys.update(
            dx=self.design_space.defaults["L_m"] / self.n_cells,
            p_evap_pa=p_evap,
            p_cond_pa=p_cond,
            h_fg_evap_j_kg=float(water_h_fg_j_kg(prof.t_evap_c + 273.15)),
        )
        return phys

    # -- harness Problem face ---------------------------------------------

    def metrics_jax(self, design: Mapping[str, Any] | None = None,
                    controls: Mapping[str, float] | None = None) -> dict[str, Any]:
        merged = self.design_space.merge(design)
        ctrl = {**self.default_controls(), **(controls or {})}
        out = bed1d.simulate_bed(
            **_bed_phys(self.material, self.profile, merged,
                        ctrl["t_f_ads_c"], ctrl["t_f_des_c"],
                        ctrl["t_ads_s"], ctrl["t_des_s"],
                        self.n_cells),
            dt_s=self.dt_phys_s,
            n_cycles=self.n_cycles,
        )
        return out["summary"]

    def evaluate(self, design: Mapping[str, float] | None = None,
                 controls: Mapping[str, float] | None = None) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics_jax(design, controls).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del n_steps  # episode length is n_cycles of the profile (§5.2)
        merged = self.design_space.merge(design)
        ctrl = {**self.default_controls(), **(controls or {})}
        out = bed1d.simulate_bed(
            **_bed_phys(self.material, self.profile, merged,
                        ctrl["t_f_ads_c"], ctrl["t_f_des_c"],
                        ctrl["t_ads_s"], ctrl["t_des_s"],
                        self.n_cells),
            dt_s=self.dt_phys_s,
            n_cycles=self.n_cycles,
            collect_trace=True,
        )
        summary = {k: float(v) for k, v in out["summary"].items()}
        series = {k: np.asarray(v) for k, v in (out["series"] or {}).items()}
        return EpisodeTrace(series=series, summary=summary)

    # -- dynamic face (numpy boundary) -------------------------------------

    def reset(self, *, seed=None, options=None):
        """Start a fresh episode: bed pre-equilibrated on the adsorption
        isotherm at ``(t_cond, P_evap)``; phase = adsorption."""
        del seed, options  # deterministic physics — nothing to sample in v0
        ctrl = self.default_controls()
        self._ctrl = ctrl
        p_evap = float(water_sat_pressure_pa(self.profile.t_evap_c + 273.15))
        T_init = jnp.full((self.n_cells,), self.profile.t_cond_c + 273.15)
        q_init = da_uptake(T_init, p_evap, self.design_space.defaults["q_sat_kg_kg"],
                           self.design_space.defaults["e_char_j_mol"],
                           self.design_space.defaults["n_da"])
        self._carry = jax.device_get(
            bed1d.initial_carry(T_init, q_init,
                                t_phase_end_s=ctrl["t_ads_s"],
                                n_cycles=self.n_cycles)
        )
        return self._observation(None), {"phase": "ads", "t_abs": 0.0}

    def step(self, action):
        if self._carry is None:
            raise RuntimeError("call reset() before step()")
        action = np.asarray(action)
        if self.action_mode == "continuous":
            if action.shape != (2,):
                raise ValueError(f"continuous action must have shape (2,), got {action.shape}")
            t_f_des = float(np.clip(action[0], self.profile.t_fluid_min_c,
                                    self.profile.t_fluid_max_c))
            t_switch = float(np.clip(action[1], *T_SWITCH_BOUNDS))
            # T_f,des takes effect from the next substep; t_switch sets the
            # duration of phases starting afterwards (module docstring).
            self._ctrl = {**self._ctrl, "t_f_des_c": t_f_des,
                          "t_ads_s": t_switch, "t_des_s": t_switch}
        else:
            want_des = int(action) == 1
            if want_des != bool(self._carry[bed1d._CARRY_PHASE] > 0.5):
                c = list(self._carry)
                c[bed1d._CARRY_T_END] = c[bed1d._CARRY_T_ABS]  # flip next substep
                self._carry = tuple(c)

        ctrl = self._ctrl
        carry, ys = self._advance_jit(
            self._carry,
            (jnp.asarray(ctrl["t_ads_s"]), jnp.asarray(ctrl["t_des_s"]),
             jnp.asarray(ctrl["t_f_des_c"])),
        )
        self._carry = jax.device_get(carry)
        ys = jax.device_get(ys)

        dq_cool = float(ys[:, 9].sum())  # whole control step (§5.2 reward)
        dq_in = float(ys[:, 10].sum())
        reward = dq_cool - self.lam * dq_in  # μ·violation ≡ 0 in Bed1D (§5.2)

        cycles_done = float(self._carry[bed1d._CARRY_CYCLES])
        terminated = bool(cycles_done >= self.n_cycles)
        info = {
            "t_abs_s": float(self._carry[bed1d._CARRY_T_ABS]),
            "phase": "des" if self._carry[bed1d._CARRY_PHASE] > 0.5 else "ads",
            "cycles_done": int(cycles_done),
            "dq_cool_j_kg": dq_cool,
            "dq_in_j_kg": dq_in,
        }
        if terminated:
            info["metrics"] = self._summary_numpy()
        return self._observation(ys[-1]), float(reward), terminated, False, info

    def _summary_numpy(self) -> dict[str, float]:
        """Episode metrics from the numpy carry (same schema as
        ``physics.bed1d.summary_from_carry``, evaluated without jax dispatch)."""
        c = self._carry
        n = c[8].shape[0] - 1
        q_cool = float(c[8][1:].sum())
        q_in = float(c[9][1:].sum())
        t_ads = float(c[13][1:].sum())
        prof = self.profile
        return {
            "COP": q_cool / q_in if q_in > 0 else 0.0,
            "SCP_W_kg": q_cool / t_ads if t_ads > 0 else 0.0,
            "delta_q": float(c[10][1:].sum()) / max(n, 1),
            "q_ads": float(c[11][1:].sum()) / max(n, 1),
            "q_des": float(c[12][1:].sum()) / max(n, 1),
            "P_evap_kPa": float(water_sat_pressure_pa(prof.t_evap_c + 273.15)) / 1e3,
            "P_cond_kPa": float(water_sat_pressure_pa(prof.t_cond_c + 273.15)) / 1e3,
            "h_fg_MJ_kg": float(water_h_fg_j_kg(prof.t_evap_c + 273.15)) / 1e6,
            "Q_cool_J_kg": q_cool,
            "Q_in_J_kg": q_in,
        }

    def _observation(self, ys_last) -> np.ndarray:
        c = self._carry
        T, q = c[bed1d._CARRY_T], c[bed1d._CARRY_Q]
        phase = float(c[bed1d._CARRY_PHASE])
        defaults = self.design_space.defaults
        if ys_last is None:  # reset: evaluate the sensors directly
            p_evap = float(water_sat_pressure_pa(self.profile.t_evap_c + 273.15))
            p_cond = float(water_sat_pressure_pa(self.profile.t_cond_c + 273.15))
            p = p_evap if phase < 0.5 else p_cond
            q_star_mean = float(
                jnp.mean(da_uptake(T, p, defaults["q_sat_kg_kg"],
                                   defaults["e_char_j_mol"], defaults["n_da"]))
            )
            t_wall, t_mean, t_max = float(T[0]), float(T.mean()), float(T.max())
            q_mean = float(q.mean())
            p_ratio = p / p_evap
            t_f = (self._ctrl["t_f_ads_c"] if phase < 0.5
                   else self._ctrl["t_f_des_c"]) + 273.15
            frac = 0.0
        else:
            (t_wall, t_mean, t_max, q_mean, q_star_mean, p_ratio, t_f_k,
             _phase, frac, _dq_c, _dq_i) = ys_last
            p_ratio = float(p_ratio)
            t_f = float(t_f_k)
        return np.array(
            [
                t_wall,
                t_mean,
                t_max,
                q_mean,
                q_star_mean,
                p_ratio,
                frac,
                t_f,
                float(c[bed1d._CARRY_QCOOL_CUM]),
                float(c[bed1d._CARRY_QIN_CUM]),
            ],
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(_OBS_LO, _OBS_HI, dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces.Space:
        if self.action_mode == "continuous":
            return gym.spaces.Box(
                low=np.array([self.profile.t_fluid_min_c, T_SWITCH_BOUNDS[0]], dtype=np.float32),
                high=np.array([self.profile.t_fluid_max_c, T_SWITCH_BOUNDS[1]], dtype=np.float32),
                dtype=np.float32,
            )
        return gym.spaces.Discrete(2)


def build_bed1d(material: "str | MaterialParams" = "anchor:Silica gel RD",
                profile: "str | ApplicationProfile" = "cpu",
                **kwargs) -> Bed1D:
    return Bed1D(material, profile, **kwargs)


REGISTRIES["envs"].register("Bed1D-v0", build_bed1d)


class Bed1DGymEnv(gym.Env):
    """Gymnasium wrapper around :class:`Bed1D` (numpy boundary, §5.2).

    Separate from the Problem class so the harness ``spec`` (a
    ProblemSpec) never collides with ``gym.Env.spec``.
    """

    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        self._env = Bed1D(**kwargs)

    @property
    def problem(self) -> Bed1D:
        return self._env

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)  # seeds gym's _np_random
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        return self._env.step(action)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._env.action_space


def _register_gymnasium() -> None:
    try:
        gym.register(
            "Bed1D-v0",
            entry_point="harness.envs.bed1d:Bed1DGymEnv",
            kwargs={"material": "anchor:Silica gel RD", "profile": "cpu"},
        )
    except gym.error.Error:
        pass  # already registered (e.g. re-import)


_register_gymnasium()
