"""Application profiles — the "whatever application is of interest" axis.

Built-ins mirror ``Materials/heat_cooling_screen.py`` with the **same four
keys** (``cpu``, ``human``, ``vehicle``, ``datacenter``) and identical
steady-setpoint and weight values (DESIGN §7.3/§8.2), so the same key means
the same physical scenario across subprojects. The screening-only weights
(stability/conductivity/density/pareto) are carried verbatim for
cross-tool comparability; harness physics does not consume them.

Dynamic envs additionally use ``t_fluid_min_c`` / ``t_fluid_max_c`` as the
heat-transfer-fluid actuator band; ``source_schedule`` / ``load_schedule``
arrive with the dynamic envs (H2+) and are ``None`` for steady use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .registry import REGISTRIES


@dataclass(frozen=True)
class ApplicationProfile:
    name: str  # registry key, e.g. "datacenter"
    description: str  # display name, mirrored from heat_cooling_screen
    t_evap_c: float
    t_cond_c: float
    t_des_c: float
    cycle_time_s: float
    cop_weight: float
    scp_weight: float
    # Screening weights carried verbatim from heat_cooling_screen.py —
    # metadata for cross-tool comparability, not consumed by harness physics.
    stability_weight: float = 0.0
    conductivity_weight: float = 0.0
    density_weight: float = 0.0
    pareto_weight: float = 0.0
    # Actuator band for the heat-transfer fluid (dynamic envs, H1+).
    t_fluid_min_c: float = 45.0
    t_fluid_max_c: float = 95.0
    # H2+: optional time schedules, t [s] -> source temperature [°C] / load [kW].
    source_schedule: Callable | None = None
    load_schedule: Callable | None = None
    notes: str = ""


def get_profile(key_or_profile: "str | ApplicationProfile") -> ApplicationProfile:
    """Resolve a registry key or pass an instance through."""
    if isinstance(key_or_profile, ApplicationProfile):
        return key_or_profile
    factory = REGISTRIES["profiles"].resolve(str(key_or_profile))
    value = factory()
    if not isinstance(value, ApplicationProfile):
        raise TypeError(f"harness.profiles factory for {key_or_profile!r} returned {type(value).__name__}")
    return value


# Steady setpoints, weights and notes are copied verbatim from
# Materials/heat_cooling_screen.py::APPLICATIONS. Fluid bands: derived from
# each profile's regeneration temperature and stated heat-source range in
# the notes (datacenter: 45–70 °C loops; solar/HVAC: 75–95 °C collectors;
# vehicle: exhaust-grade heat up to ~130 °C; cpu: above the 75 °C
# regeneration setpoint).
BUILTIN_PROFILES = {
    "cpu": ApplicationProfile(
        name="cpu",
        description="CPU / electronics cold plate assist",
        t_evap_c=18.0,
        t_cond_c=35.0,
        t_des_c=75.0,
        cycle_time_s=120.0,
        cop_weight=0.20,
        scp_weight=0.45,
        stability_weight=0.15,
        conductivity_weight=0.15,
        density_weight=0.05,
        pareto_weight=0.15,
        t_fluid_min_c=60.0,
        t_fluid_max_c=85.0,
        notes="Needs very high specific cooling power and excellent bed heat transfer; practical use is more likely at rack/cold-plate scale than inside a chip package.",
    ),
    "human": ApplicationProfile(
        name="human",
        description="Human thermal comfort / HVAC",
        t_evap_c=10.0,
        t_cond_c=35.0,
        t_des_c=80.0,
        cycle_time_s=600.0,
        cop_weight=0.40,
        scp_weight=0.20,
        stability_weight=0.20,
        conductivity_weight=0.05,
        density_weight=0.15,
        pareto_weight=0.15,
        t_fluid_min_c=75.0,
        t_fluid_max_c=95.0,
        notes="Best fit for solar thermal or waste-heat regenerated adsorption chillers using water refrigerant.",
    ),
    "vehicle": ApplicationProfile(
        name="vehicle",
        description="Vehicle waste-heat cooling",
        t_evap_c=7.0,
        t_cond_c=45.0,
        t_des_c=120.0,
        cycle_time_s=180.0,
        cop_weight=0.20,
        scp_weight=0.40,
        stability_weight=0.20,
        conductivity_weight=0.15,
        density_weight=0.05,
        pareto_weight=0.15,
        t_fluid_min_c=90.0,
        t_fluid_max_c=130.0,
        notes="Can exploit hotter exhaust/coolant heat, but compactness, vibration tolerance, and fast cycling dominate.",
    ),
    "datacenter": ApplicationProfile(
        name="datacenter",
        description="Data-center waste-heat cooling",
        t_evap_c=16.0,
        t_cond_c=35.0,
        t_des_c=60.0,
        cycle_time_s=300.0,
        cop_weight=0.35,
        scp_weight=0.30,
        stability_weight=0.20,
        conductivity_weight=0.10,
        density_weight=0.05,
        pareto_weight=0.20,
        t_fluid_min_c=45.0,
        t_fluid_max_c=70.0,
        notes="Hard low-grade heat case: candidates must regenerate from warm liquid loops around 45-70 C.",
    ),
}


def _register_builtins() -> None:
    for profile in BUILTIN_PROFILES.values():
        REGISTRIES["profiles"].register(profile.name, lambda p=profile: p, overwrite=True)


_register_builtins()
