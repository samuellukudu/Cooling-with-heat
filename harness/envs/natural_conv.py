"""NaturalConv-v0 — buoyancy-driven enclosure as a static Problem (T-A1).

Env = (Nusselt-correlation physics) × (fluid row) × (profile scenario).
Discovery axes: **design** (hot-bottom/cold-top ΔT — interior optimum
under a heater-cost penalty) and **materials** (fluid shortlist per
scenario: water ≫ oil ≫ air in specific flux). The algorithm axis
(heating-pattern control under time-varying Ra) and resolved CFD both
stay pull-driven H3 heads.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..physics import natural_conv as nc
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec

DESIGN_KEYS = ("delta_T_K",)
NATCONV_ENV_METRIC_KEYS = nc.NATCONV_METRIC_KEYS
NATCONV_SCHEMA_VERSION = 1


def _resolve_fluid_name(material: Any) -> tuple[str, str]:
    if isinstance(material, str):
        try:
            f = nc.resolve_fluid(material.strip())
            return f["name"], "fluid"
        except KeyError:
            return nc.DEFAULT_FLUID, "anchor-fallback"
    if material is None:
        return nc.DEFAULT_FLUID, "default"
    return nc.DEFAULT_FLUID, "anchor-fallback"


class NaturalConv:
    """Correlated RB enclosure over fluid × scenario."""

    def __init__(
        self,
        material: Any = nc.DEFAULT_FLUID,
        profile: "str | ApplicationProfile" = "human",
        *,
        L_m: float = 0.1,
    ):
        self.fluid_name, self.fluid_provenance = _resolve_fluid_name(material)
        self.material = material
        self.profile = get_profile(profile)
        self.L_m = float(L_m)
        self.spec = ProblemSpec(name="NaturalConv-v0", kind="static",
                                metric_keys=NATCONV_ENV_METRIC_KEYS,
                                schema_version=NATCONV_SCHEMA_VERSION,
                                spatial_dim=0, time_resolved=False, grid_type="none")
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS,
            defaults={"delta_T_K": 10.0},
            bounds={"delta_T_K": (1.0, 60.0)},
        )
        from ..physics.adapters import ModelSpec  # noqa: WPS433
        self.model_spec = ModelSpec(
            name="natural_conv",
            state_vars=("Nu", "Ra"),
            units={"delta_T_K": "K", "heat_flux_W_m2": "W/m2"},
            dt_min_s=1e-3,
            dt_max_s=1e3,
            description="correlated RB enclosure (T-A1 trial)",
        )

    def regime(self, design: Mapping[str, float] | None = None) -> str:
        """Human-readable regime (NOT a metric — keep out of Objective)."""
        m = self.evaluate(design)
        return nc.regime_of(m["Ra"])

    def metrics_jax(self, design: Mapping[str, Any] | None = None):
        merged = self.design_space.merge(design)
        return nc.simulate_natconv(delta_T_K=merged["delta_T_K"],
                                   L_m=self.L_m, fluid=self.fluid_name)

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        del controls, n_steps
        return EpisodeTrace(summary=self.evaluate(design))


def build_natural_conv(material: Any = nc.DEFAULT_FLUID,
                       profile: "str | ApplicationProfile" = "human",
                       **kwargs) -> NaturalConv:
    allowed = {"L_m"}
    ctor = {k: v for k, v in kwargs.items() if k in allowed}
    return NaturalConv(material=material, profile=profile, **ctor)


REGISTRIES["envs"].register("NaturalConv-v0", build_natural_conv)

try:
    from ..physics.adapters import Fluxes as _Fluxes  # noqa: WPS433
    from ..physics.adapters import ModelSpec as _MS  # noqa: WPS433

    class _NatConvAdapter:
        def __init__(self, fluid: str = nc.DEFAULT_FLUID):
            self.fluid = fluid
            self.spec = _MS(name="natural_conv", state_vars=("Nu", "Ra"),
                            units={"Nu": "-"}, dt_min_s=1e-3, dt_max_s=1e3,
                            description="natural-convection adapter (T-A1)")

        def step(self, state, control, dt_s):
            out = nc.simulate_natconv(delta_T_K=float(control.get("delta_T_K", 10.0)),
                                      fluid=self.fluid)
            return {k: float(v) for k, v in out.items()}, _Fluxes()

        def metrics(self, trace):
            return dict(trace)

    try:
        REGISTRIES["models"].register("natural_conv", _NatConvAdapter)
    except ValueError:
        pass
except Exception:
    pass

__all__ = ["NaturalConv", "build_natural_conv", "DESIGN_KEYS", "NATCONV_ENV_METRIC_KEYS"]
