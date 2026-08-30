"""Environment layer: problems, specs, registries wiring (``DESIGN.md`` §5)."""

from . import base, bed1d, cycle0d
from .base import (
    ActionSpec,
    DesignSpace,
    EpisodeTrace,
    Objective,
    Problem,
    ProblemSpec,
    Sensor,
    objective_value,
    validate_problem,
)
from .bed1d import BED1D_SCHEMA_VERSION, BED_METRIC_KEYS, Bed1D, Bed1DGymEnv

__all__ = [
    "ActionSpec",
    "BED1D_SCHEMA_VERSION",
    "BED_METRIC_KEYS",
    "Bed1D",
    "Bed1DGymEnv",
    "DesignSpace",
    "EpisodeTrace",
    "Objective",
    "Problem",
    "ProblemSpec",
    "Sensor",
    "base",
    "bed1d",
    "cycle0d",
    "objective_value",
    "validate_problem",
]
