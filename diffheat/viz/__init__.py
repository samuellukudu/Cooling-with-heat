# diffheat/viz/__init__.py
"""PyQt6-based visualization for heat equation trajectories.

This is the ONLY package in diffheat that imports PyQt or matplotlib.
The core library remains headless and works without these dependencies.
"""
from .heatmap1d import HeatmapWidget
from .window import ViewerWindow, run_viewer

__all__ = ["HeatmapWidget", "ViewerWindow", "run_viewer"]
