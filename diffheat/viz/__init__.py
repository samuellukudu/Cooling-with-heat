# diffheat/viz/__init__.py
"""PyQt6-based visualization for heat equation trajectories.

This is the ONLY package in diffheat that imports PyQt or matplotlib.
The core library remains headless and works without these dependencies.
"""
from .heatmap1d import HeatmapWidget
from .heatmap2d import Heatmap2DWidget
from .window import ViewerWindow, ViewerWindow2D, run_viewer, run_viewer_2d

__all__ = [
    "HeatmapWidget",
    "Heatmap2DWidget",
    "ViewerWindow",
    "ViewerWindow2D",
    "run_viewer",
    "run_viewer_2d",
]
