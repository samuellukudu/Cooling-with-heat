"""Pure-JAX physics of heat-driven cooling (``../DESIGN.md`` §4).

This subpackage imports only ``jax``/``numpy`` and itself — no env, backend,
data, or optimizer imports (enforced by import-linter, root pyproject).
"""

from . import bed1d, cycle0d, system, thermo
from .bed1d import (
    SERIES_CHANNELS,
    advance_carry,
    bed_rhs,
    check_timestep,
    cycle_totals_from_carry,
    initial_carry,
    max_timestep,
    simulate_bed,
    step_bed,
    step_episode,
    summary_from_carry,
    volumetric_capacity,
)
from .cycle0d import METRIC_KEYS, simulate_cycle
from .system import (
    SYSTEM_SERIES_CHANNELS,
    advance_two_carry,
    bed_phys,
    initial_two_carry,
    simulate_two_bed,
    summary_from_two_carry,
)
from .thermo import (
    CP_ADSORBENT,
    CP_LIQUID,
    GAS_CONSTANT,
    da_uptake,
    water_h_fg_j_kg,
    water_sat_pressure_pa,
)

__all__ = [
    "CP_ADSORBENT",
    "CP_LIQUID",
    "GAS_CONSTANT",
    "METRIC_KEYS",
    "SERIES_CHANNELS",
    "SYSTEM_SERIES_CHANNELS",
    "advance_carry",
    "advance_two_carry",
    "bed1d",
    "bed_phys",
    "bed_rhs",
    "check_timestep",
    "cycle0d",
    "cycle_totals_from_carry",
    "da_uptake",
    "initial_carry",
    "initial_two_carry",
    "max_timestep",
    "simulate_bed",
    "simulate_cycle",
    "simulate_two_bed",
    "step_bed",
    "step_episode",
    "summary_from_carry",
    "summary_from_two_carry",
    "system",
    "thermo",
    "volumetric_capacity",
    "water_h_fg_j_kg",
    "water_sat_pressure_pa",
]
