"""A differentiable optimization harness for heat-driven cooling, with
gym-compatible environments.

RL is one of three optimization backends — not the identity of the project.
That framing keeps us honest about which problems need which solver, and
makes the project legible to people who (rightly) raise an eyebrow at
"we used RL" on a 3-parameter physics function. See DESIGN.md.

Typical use::

    import harness

    env = harness.make("Cycle0D-v0",
                       material="anchor:Silica gel RD",
                       profile="datacenter")
    result = harness.optimize(env, harness.Objective.single("COP"),
                              backend="grad")
    print(result.best_metrics["COP"])
"""

from __future__ import annotations

import jax

# Physics parity < 1e-12 against the canonical NumPy implementation (V1)
# requires float64; enable it before any jnp array is created. This matches
# the root tests/conftest.py convention and the frozen diffheat lineage.
jax.config.update("jax_enable_x64", True)

from . import backends, envs, materials, physics, profiles, registry, report  # noqa: E402
from .backends import OptimizeResult, optimize  # noqa: E402
from .envs.base import (  # noqa: E402
    ActionSpec,
    DesignSpace,
    EpisodeTrace,
    Objective,
    Problem,
    ProblemSpec,
    Sensor,
    objective_value,
)
from .materials import MaterialParams, get_material, load_materials_csv  # noqa: E402
from .profiles import ApplicationProfile, get_profile  # noqa: E402
from .registry import (  # noqa: E402
    REGISTRIES,
    make,
    register,
    register_backend,
    register_env,
    register_material,
    register_model,
    register_profile,
)

__version__ = "0.1.0"

__all__ = [
    "ActionSpec",
    "ApplicationProfile",
    "EpisodeTrace",
    "MaterialParams",
    "Objective",
    "OptimizeResult",
    "Problem",
    "ProblemSpec",
    "REGISTRIES",
    "DesignSpace",
    "Sensor",
    "__version__",
    "backends",
    "envs",
    "get_material",
    "get_profile",
    "load_materials_csv",
    "make",
    "materials",
    "objective_value",
    "optimize",
    "physics",
    "profiles",
    "register",
    "register_backend",
    "register_env",
    "register_material",
    "register_model",
    "register_profile",
    "registry",
    "report",
]
