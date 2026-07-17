# diffheat/physics/__init__.py
"""Physical problem definitions."""
from ..operators.laplacian import make_laplacian
from .bessel import (
    bessel_j_zero,
    eigenfunction_disc,
    eigenfunction_norm,
    eigenvalue_disc,
)
from .heat1d import HeatEquation1D, apply_boundary_conditions
from .heat2d import HeatEquation2D
from .heat3d import HeatEquation3D
from .wave import WaveEquation1D, WaveEquation2D, WaveEquation3D
from .telegrapher import TelegrapherEquation1D, TelegrapherEquation2D, TelegrapherEquation3D

__all__ = [
    "HeatEquation1D",
    "HeatEquation2D",
    "HeatEquation3D",
    "WaveEquation1D",
    "WaveEquation2D",
    "WaveEquation3D",
    "TelegrapherEquation1D",
    "TelegrapherEquation2D",
    "TelegrapherEquation3D",
    "apply_boundary_conditions",
    "make_laplacian",
    "bessel_j_zero",
    "eigenvalue_disc",
    "eigenfunction_disc",
    "eigenfunction_norm",
]


