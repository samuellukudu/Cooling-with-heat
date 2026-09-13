"""Environment layer: problems, specs, registries wiring (``DESIGN.md`` §5)."""

from . import absorption, adapter, base, bed1d, boussinesq, cloak2d, cycle0d, forced_conv, heat1d, heat2d, heat3d, natural_conv, te_1d, telegrapher, thermoelectric, two_bed
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
from .absorption import ABSORPTION_METRIC_KEYS, AbsorptionCycle
from .bed1d import BED1D_SCHEMA_VERSION, BED_METRIC_KEYS, Bed1D, Bed1DGymEnv
from .boussinesq import BOUSSINESQ_ENV_METRIC_KEYS, Boussinesq
from .cloak2d import CLOAK_METRIC_KEYS, Cloak2D
from .forced_conv import FORCED_CONV_METRIC_KEYS, ForcedConv
from .heat1d import HEAT1D_METRIC_KEYS, Heat1D
from .heat2d import HEAT2D_METRIC_KEYS, Heat2D
from .heat3d import HEAT3D_METRIC_KEYS, Heat3D
from .natural_conv import NATCONV_ENV_METRIC_KEYS, NaturalConv
from .te_1d import TE1D_ENV_METRIC_KEYS, Thermoelectric1D
from .telegrapher import TELEGRAPHER_ENV_METRIC_KEYS, TelegrapherCancel
from .thermoelectric import TE_ENV_METRIC_KEYS, Thermoelectric
from .two_bed import (
    TWO_BED_SCHEMA_VERSION,
    TWO_BED_METRIC_KEYS,
    TwoBed,
    TwoBedGymEnv,
    TwoBedSchedule,
)

__all__ = [
    "ABSORPTION_METRIC_KEYS",
    "AbsorptionCycle",
    "ActionSpec",
    "BED1D_SCHEMA_VERSION",
    "BED_METRIC_KEYS",
    "BOUSSINESQ_ENV_METRIC_KEYS",
    "Bed1D",
    "Bed1DGymEnv",
    "Boussinesq",
    "CLOAK_METRIC_KEYS",
    "Cloak2D",
    "FORCED_CONV_METRIC_KEYS",
    "ForcedConv",
    "HEAT1D_METRIC_KEYS",
    "HEAT2D_METRIC_KEYS",
    "HEAT3D_METRIC_KEYS",
    "Heat1D",
    "Heat2D",
    "Heat3D",
    "NATCONV_ENV_METRIC_KEYS",
    "NaturalConv",
    "TELEGRAPHER_ENV_METRIC_KEYS",
    "TE1D_ENV_METRIC_KEYS",
    "TE_ENV_METRIC_KEYS",
    "TelegrapherCancel",
    "Thermoelectric",
    "Thermoelectric1D",
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
    "absorption",
    "bed1d",
    "boussinesq",
    "cloak2d",
    "cycle0d",
    "forced_conv",
    "heat1d",
    "heat2d",
    "heat3d",
    "natural_conv",
    "objective_value",
    "te_1d",
    "telegrapher",
    "thermoelectric",
    "two_bed",
    "validate_problem",
]
