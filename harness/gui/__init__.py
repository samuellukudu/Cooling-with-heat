"""harness.gui — PyQt6 apps for the cooling-with-heat project.

Two separate applications share :mod:`harness.gui.kit`:

- ``harness.gui.app`` — the RL & simulation launcher over the harness
  environments (``python -m harness.gui`` or the ``harness-gui`` script).
- ``adsorbent-ml/gui/data_app.py`` — the dataset explorer (lives next to
  the adsorbent-ml pipeline; imports this kit).
"""

__version__ = "0.2.0"
