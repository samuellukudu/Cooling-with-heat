"""AbsorptionCycle-v0 — single-effect absorption chiller as a static Problem (T-A3).

Env = (lumped 3-T cycle physics) × (working pair) × (profile scenario).
Discovery axes: **design** (``t_gen_c`` generator temperature) and
**materials** (working-pair shortlist per profile — LiBr vs NH3 ranking
flips with regeneration temperature, the T-A3 analog of the H2.3 13X
finding). The algorithm axis (source-following schedules) is deferred.

Discipline: ``t_evap_c``/``t_cond_c``/``cycle_time_s`` are scenario
(profile) parameters, not design keys. ``material`` accepts a working-pair
key (``"LiBr-H2O"`` …); harness anchor refs (``"anchor:…"``) and
``MaterialParams`` fall back to the default pair so GUI graphs wired with
adsorption materials still run — the fallback is explicit, not silent
(see ``pair_name`` attribute / ``pair_provenance``).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..physics import absorption as ab
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec

DESIGN_KEYS = ("t_gen_c",)
ABSORPTION_METRIC_KEYS = ab.ABSORPTION_METRIC_KEYS
ABSORPTION_SCHEMA_VERSION = 1


def _resolve_pair_name(material: Any) -> tuple[str, str]:
    """Map the ``material`` slot to a working-pair key + provenance flag."""
    if material is None:
        return ab.DEFAULT_PAIR, "default"
    if isinstance(material, str):
        try:
            p = ab.resolve_pair(material.strip())
            return p["name"], "pair"
        except KeyError:
            # Harness anchor ref (e.g. "anchor:Silica gel RD") — documented
            # fallback so GUI graphs wired with adsorption materials still run.
            return ab.DEFAULT_PAIR, "anchor-fallback"
    # MaterialParams instance or anything else → fallback (adsorption rows
    # carry D-A isotherms, not absorption pairs).
    return ab.DEFAULT_PAIR, "anchor-fallback"


class AbsorptionCycle:
    """Single-effect absorption point cycle over pair × profile."""

    def __init__(
        self,
        material: Any = ab.DEFAULT_PAIR,
        profile: "str | ApplicationProfile" = "human",
        *,
        cycle_time_s: float | None = None,
        ua_loss_W_K_per_kg: float = 0.0,
        t_amb_c: float = ab.T_AMB_C,
    ):
        self.pair_name, self.pair_provenance = _resolve_pair_name(material)
        self.material = material
        self.profile = get_profile(profile)
        self.cycle_time_s = float(cycle_time_s if cycle_time_s is not None else self.profile.cycle_time_s)
        self.ua_loss = float(ua_loss_W_K_per_kg)
        self.t_amb_c = float(t_amb_c)
        self.spec = ProblemSpec(name="AbsorptionCycle-v0", kind="static",
                                metric_keys=ABSORPTION_METRIC_KEYS,
                                schema_version=ABSORPTION_SCHEMA_VERSION,
                                spatial_dim=0, time_resolved=False, grid_type="none")
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS,
            defaults={"t_gen_c": float(self.profile.t_des_c)},
            bounds={"t_gen_c": (60.0, 95.0)},
        )
        from ..physics.adapters import ModelSpec  # noqa: WPS433
        self.model_spec = ModelSpec(
            name="absorption",
            state_vars=("delta_x", "COP"),
            units={"t_gen_c": "degC", "COP": "-", "SCP_W_kg": "W/kg"},
            dt_min_s=1e-3,
            dt_max_s=1e3,
            description="lumped single-effect absorption cycle (T-A3 trial)",
        )

    def metrics_jax(self, design: Mapping[str, Any] | None = None):
        merged = self.design_space.merge(design)
        return ab.simulate_absorption(
            t_evap_c=self.profile.t_evap_c,
            t_cond_c=self.profile.t_cond_c,
            t_gen_c=merged["t_gen_c"],
            pair=self.pair_name,
            cycle_time_s=self.cycle_time_s,
            ua_loss_W_K_per_kg=self.ua_loss,
            t_amb_c=self.t_amb_c,
        )

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del controls, n_steps
        return EpisodeTrace(summary=self.evaluate(design))


def build_absorption(material: Any = ab.DEFAULT_PAIR,
                     profile: "str | ApplicationProfile" = "human",
                     **kwargs) -> AbsorptionCycle:
    allowed = {"cycle_time_s", "ua_loss_W_K_per_kg", "t_amb_c"}
    ctor = {k: v for k, v in kwargs.items() if k in allowed}
    return AbsorptionCycle(material=material, profile=profile, **ctor)


REGISTRIES["envs"].register("AbsorptionCycle-v0", build_absorption)

try:
    from ..physics.adapters import Fluxes as _Fluxes  # noqa: WPS433
    from ..physics.adapters import ModelSpec as _MS  # noqa: WPS433

    class _AbsorptionAdapter:
        def __init__(self, pair: str = ab.DEFAULT_PAIR):
            self.pair = pair
            self.spec = _MS(name="absorption", state_vars=("delta_x", "COP"),
                            units={"COP": "-"}, dt_min_s=1e-3, dt_max_s=1e3,
                            description="absorption adapter (T-A3)")

        def step(self, state, control, dt_s):
            out = ab.simulate_absorption(
                t_gen_c=float(control.get("t_gen_c", 85.0)), pair=self.pair)
            return {k: float(v) for k, v in out.items()}, _Fluxes()

        def metrics(self, trace):
            return dict(trace)

    try:
        REGISTRIES["models"].register("absorption", _AbsorptionAdapter)
    except ValueError:
        pass
except Exception:
    pass

__all__ = ["AbsorptionCycle", "build_absorption", "DESIGN_KEYS", "ABSORPTION_METRIC_KEYS"]
