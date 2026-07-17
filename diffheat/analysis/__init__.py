# diffheat/analysis/__init__.py
"""Analysis tools for PDE solutions — nodal lines, validators, etc."""
from .nodal import compute_nodal_lines_2d, compute_nodal_lines_disc
from .validator import check_maximum_principle_2d, check_mean_value_2d

__all__ = [
    "compute_nodal_lines_2d",
    "compute_nodal_lines_disc",
    "check_mean_value_2d",
    "check_maximum_principle_2d",
]
