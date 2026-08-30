"""Cycle0D-v0 — the equilibrium oracle as a static Problem (DESIGN §4.1/§5.1).

A faithful wrapping of ``harness.physics.cycle0d.simulate_cycle`` (itself the
exact JAX mirror of ``Materials/cooling_physics.simulate_adsorption_cycle``).
Not a dynamic env: an equilibrium model has no meaningful ``step`` — faking
dynamics where there are none is exactly what the naming discipline forbids.
The rl backend wraps static problems as single-step envs if ever needed.

Design-parameter bounds for ``q_sat``/``Q_st`` are the canonical screening
envelope mirrored from ``cooling_physics.pareto_target_window`` defaults, so
harness optima are directly comparable with the legacy screen (validation
V7).
"""

from __future__ import annotations

from typing import Any, Mapping

from ..materials import MaterialParams, get_material
from ..physics import METRIC_KEYS, simulate_cycle
from ..profiles import ApplicationProfile, get_profile
from ..registry import REGISTRIES
from .base import DesignSpace, EpisodeTrace, ProblemSpec

DESIGN_KEYS = (
    "q_sat_kg_kg",
    "Q_st_j_kg",
    "e_char_j_mol",
    "n_da",
    "cycle_time_s",
    "hx_mass_factor",
)

# Canonical screening envelope — mirrored from
# Materials/cooling_physics.py::pareto_target_window defaults.
Q_SAT_BOUNDS = (0.08, 0.90)
Q_ST_BOUNDS = (2.30e6, 4.10e6)

# Canonical min-max ranges used by the legacy screen's weighted score
# (cooling_physics.pareto_target_window) — reused verbatim so V7 compares
# the same quantity on both sides.
CANONICAL_NORMALIZATION = {
    "COP": (0.05, 0.85),
    "SCP_W_kg": (20.0, 1600.0),
}


class Cycle0D:
    """Equilibrium single-bed cycle over a material × profile pair."""

    def __init__(self, material: "str | MaterialParams", profile: "str | ApplicationProfile", hx_mass_factor: float = 1.35):
        self.material = get_material(material)
        self.profile = get_profile(profile)
        self.spec = ProblemSpec(name="Cycle0D-v0", kind="static", metric_keys=METRIC_KEYS)
        self.design_space = DesignSpace(
            keys=DESIGN_KEYS,
            defaults={
                "q_sat_kg_kg": self.material.q_sat_kg_kg,
                "Q_st_j_kg": self.material.q_st_j_kg,
                "e_char_j_mol": self.material.e_char_j_mol,
                "n_da": self.material.n_da,
                "cycle_time_s": self.profile.cycle_time_s,
                "hx_mass_factor": float(hx_mass_factor),
            },
            bounds={
                "q_sat_kg_kg": Q_SAT_BOUNDS,
                "Q_st_j_kg": Q_ST_BOUNDS,
                "e_char_j_mol": (2000.0, 20000.0),
                "n_da": (1.0, 4.0),
                "cycle_time_s": (30.0, 3600.0),
                "hx_mass_factor": (1.0, 2.0),
            },
        )

    def metrics_jax(self, design: Mapping[str, Any] | None = None) -> dict[str, Any]:
        merged = self.design_space.merge(design)
        return simulate_cycle(
            q_sat=merged["q_sat_kg_kg"],
            q_st=merged["Q_st_j_kg"],
            t_evap_c=self.profile.t_evap_c,
            t_cond_c=self.profile.t_cond_c,
            t_des_c=self.profile.t_des_c,
            cycle_time_sec=merged["cycle_time_s"],
            e_char_j_mol=merged["e_char_j_mol"],
            n_heterogeneity=merged["n_da"],
            hx_mass_factor=merged["hx_mass_factor"],
        )

    def evaluate(self, design: Mapping[str, float] | None = None) -> dict[str, float]:
        return {k: float(v) for k, v in self.metrics_jax(design).items()}

    def rollout(self, design=None, controls=None, *, n_steps: int = 1) -> EpisodeTrace:
        # Static problem: one evaluation, no time series (DESIGN §4.1).
        return EpisodeTrace(summary=self.evaluate(design))


def build_cycle0d(
    material: "str | MaterialParams" = "anchor:Silica gel RD",
    profile: "str | ApplicationProfile" = "datacenter",
    hx_mass_factor: float = 1.35,
) -> Cycle0D:
    return Cycle0D(material, profile, hx_mass_factor)


REGISTRIES["envs"].register("Cycle0D-v0", build_cycle0d)
