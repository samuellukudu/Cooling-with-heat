"""Environment layer: problems, specs, registries wiring (``DESIGN.md`` §5)."""

from . import base, bed1d, cycle0d, two_bed
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
from .two_bed import (
    TWO_BED_SCHEMA_VERSION,
    TWO_BED_METRIC_KEYS,
    TwoBed,
    TwoBedGymEnv,
    TwoBedSchedule,
)

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
    "TWO_BED_SCHEMA_VERSION",
    "TWO_BED_METRIC_KEYS",
    "TwoBed",
    "TwoBedGymEnv",
    "TwoBedSchedule",
    "base",
    "bed1d",
    "cycle0d",
    "objective_value",
    "two_bed",
    "validate_problem",
]
