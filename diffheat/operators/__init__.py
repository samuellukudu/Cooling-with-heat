# diffheat/operators/__init__.py
"""Discrete differential operators for finite difference PDEs."""
from .laplacian import make_laplacian

__all__ = ["make_laplacian"]
