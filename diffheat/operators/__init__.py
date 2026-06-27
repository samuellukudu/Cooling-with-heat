# diffheat/operators/__init__.py
"""Discrete differential operators for finite difference PDEs."""
from .divergence import divergence_2d
from .gradient import gradient_1d, gradient_2d, gradient_x, gradient_y
from .laplacian import laplacian_1d, laplacian_2d, make_laplacian

__all__ = [
    "make_laplacian",
    "laplacian_1d",
    "laplacian_2d",
    "gradient_1d",
    "gradient_x",
    "gradient_y",
    "gradient_2d",
    "divergence_2d",
]
