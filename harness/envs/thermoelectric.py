"""Thermoelectric-v0 — Peltier pair as a static harness Problem (T-A2).

Env = (lumped pair physics) × (TE pair row) × (profile scenario with
``Th = t_cond_c``, ``Tc = t_evap_c`` as nominal module boundary
temperatures — hot-side rejection to the ambient loop, cold side to the
load). Discovery axes: **design** (drive current — interior
optimum at ``I_opt = S·Tc/R``) and **materials** (pair shortlist per
scenario, including validity-driven selection via ``out_of_window``).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..physics import thermoelectric as te
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec

DESIGN_KEYS = ("current_A",)
TE_ENV_METRIC_KEYS = te.TE_METRIC_KEYS
TE_SCHEMA_VERSION = 1


def _resolve_pair_name(material: Any) -> tuple[str, str]:
    if isinstance(material, str):
        try:
            p = te.resolve_te_pair(material.strip())
            return p["name"], "pair"
        except KeyError:
            return te.DEFAULT_TE_PAIR, "anchor-fallback"
    if material is None:
        return te.DEFAULT_TE_PAIR, "default"
    return te.DEFAULT_TE_PAIR, "anchor-fallback"


class Thermoelectric:
    """Lumped Peltier pair over pair × profile."""

    def __init__(
        self,
        material: Any = te.DEFAULT_TE_PAIR,
        profile: "str | ApplicationProfile" = "cpu",
    ):
        self.pair_name, self.pair_provenance = _resolve_pair_name(material)
        self.material = material
        self.profile = get_profile(profile)
        self.Th_C = float(self.profile.t_cond_c)
        self.Tc_C = float(self.profile.t_evap_c)
        self.spec = ProblemSpec(name="Thermoelectric-v0", kind="static",
                                metric_keys=TE_ENV_METRIC_KEYS,
                                schema_version=TE_SCHEMA_VERSION,
                                spatial_dim=0, time_resolved=False, grid_type="none")
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS,
            defaults={"current_A": 2.0},
            bounds={"current_A": (0.5, 8.0)},
        )
        from ..physics.adapters import ModelSpec  # noqa: WPS433
        self.model_spec = ModelSpec(
            name="thermoelectric",
            state_vars=("Qc_W", "COP"),
            units={"current_A": "A", "Qc_W": "W", "COP": "-"},
            dt_min_s=1e-3,
            dt_max_s=1e3,
            description="lumped Peltier pair (T-A2 trial)",
        )

    def metrics_jax(self, design: Mapping[str, Any] | None = None):
        merged = self.design_space.merge(design)
        return te.simulate_te(Tc_C=self.Tc_C, Th_C=self.Th_C,
                              current_A=merged["current_A"], pair=self.pair_name)

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del controls, n_steps
        return EpisodeTrace(summary=self.evaluate(design))


def build_thermoelectric(material: Any = te.DEFAULT_TE_PAIR,
                         profile: "str | ApplicationProfile" = "cpu",
                         **kwargs) -> Thermoelectric:
    del kwargs
    return Thermoelectric(material=material, profile=profile)


REGISTRIES["envs"].register("Thermoelectric-v0", build_thermoelectric)

try:
    from ..physics.adapters import Fluxes as _Fluxes  # noqa: WPS433
    from ..physics.adapters import ModelSpec as _MS  # noqa: WPS433

    class _TEAdapter:
        def __init__(self, pair: str = te.DEFAULT_TE_PAIR):
            self.pair = pair
            self.spec = _MS(name="thermoelectric", state_vars=("Qc_W", "COP"),
                            units={"Qc_W": "W"}, dt_min_s=1e-3, dt_max_s=1e3,
                            description="thermoelectric adapter (T-A2)")

        def step(self, state, control, dt_s):
            out = te.simulate_te(current_A=float(control.get("current_A", 2.0)), pair=self.pair)
            return {k: float(v) for k, v in out.items()}, _Fluxes()

        def metrics(self, trace):
            return dict(trace)

    try:
        REGISTRIES["models"].register("thermoelectric", _TEAdapter)
    except ValueError:
        pass
except Exception:
    pass

__all__ = ["Thermoelectric", "build_thermoelectric", "DESIGN_KEYS", "TE_ENV_METRIC_KEYS"]
