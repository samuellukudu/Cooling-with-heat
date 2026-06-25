# diffheat/viz/canvas.py
"""Matplotlib canvas embedded in a Qt widget."""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib figure embedded in a Qt widget."""

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
