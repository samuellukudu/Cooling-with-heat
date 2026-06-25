# diffheat/operators/__init__.py
"""Discrete differential operators for finite difference PDEs."""
from .laplacian import laplacian_2d, make_laplacian

__all__ = ["make_laplacian", "laplacian_2d"]
