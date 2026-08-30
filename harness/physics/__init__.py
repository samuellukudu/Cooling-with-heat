"""Pure-JAX physics of heat-driven cooling (``../DESIGN.md`` §4).

This subpackage imports only ``jax``/``numpy`` and itself — no env, backend,
data, or optimizer imports (enforced by import-linter, root pyproject).
"""

from . import cycle0d, thermo
from .cycle0d import METRIC_KEYS, simulate_cycle
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
    "cycle0d",
    "da_uptake",
    "simulate_cycle",
    "thermo",
    "water_h_fg_j_kg",
    "water_sat_pressure_pa",
]
