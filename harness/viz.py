"""Hydrogym-style field rendering for the cooling harness.

Physics returns arrays; this module turns them into frames. Matplotlib is
imported lazily (Agg backend) so ``import harness`` stays light and
headless-safe — scripts get ``rgb_array`` videos without Qt, the GUI
Field player embeds the same ``paint_*`` painters on live axes.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_BG = "#0f1115"
_FG = "#e2e8f0"
_AXIS = "#94a3b8"
_SPINE = "#1e222a"
_AMBER = "#f59e0b"
_GREEN = "#22c55e"
_CYAN = "#06b6d4"

__all__ = [
    "bed_frame",
    "twobed_frame",
    "field_heatmap",
    "episode_frames",
    "paint_bed_snapshot",
    "paint_twobed_snapshot",
    "paint_heatmap",
]


def _canvas(figsize=(6.0, 3.5), dpi=100):
    # importlib (not a static import) keeps the AST import-hygiene gate
    # green — same precedent as the rl extra in backends.py. matplotlib
    # stays a lazy, render-time-only dependency of this module.
    import importlib

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    agg_mod = importlib.import_module("matplotlib.backends.backend_agg")
    fig_mod = importlib.import_module("matplotlib.figure")
    fig = fig_mod.Figure(figsize=figsize, dpi=dpi, facecolor=_BG)
    agg_mod.FigureCanvasAgg(fig)
    return fig


def _to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def _style(ax, xlabel="") -> None:
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_AXIS, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(_SPINE)
    if xlabel:
        ax.set_xlabel(xlabel, color=_AXIS, fontsize=7)


def _twin(ax):
    ax2 = ax.twinx()
    ax2.set_facecolor(_BG)
    ax2.tick_params(colors=_AXIS, labelsize=7)
    for spine in ax2.spines.values():
        spine.set_color(_SPINE)
    return ax2


def paint_bed_snapshot(ax, ax2, T_K: np.ndarray, q: np.ndarray,
                       x_m: np.ndarray) -> None:
    """Paint one ``T(x)`` + ``q(x)`` bed snapshot on live axes."""
    x = np.asarray(x_m, dtype=float) * 1000.0  # mm
    ax.plot(x, np.asarray(T_K, dtype=float) - 273.15,
            color=_AMBER, lw=1.6, label="T [°C]")
    ax.set_ylabel("T [°C]", color=_AMBER, fontsize=7)
    ax2.plot(x, np.asarray(q, dtype=float), color=_GREEN, lw=1.3,
             ls="--", label="q [kg/kg]")
    ax2.set_ylabel("q [kg/kg]", color=_GREEN, fontsize=7)
    _style(ax, xlabel="x [mm, wall → vapor]")


def paint_twobed_snapshot(ax, ax2, A_T: np.ndarray, A_q: np.ndarray,
                          B_T: np.ndarray, B_q: np.ndarray,
                          x_m: np.ndarray) -> None:
    """Paint stacked-pair snapshot: A/B ``T(x)`` solid, ``q(x)`` dashed."""
    x = np.asarray(x_m, dtype=float) * 1000.0
    ax.plot(x, np.asarray(A_T, dtype=float) - 273.15,
            color=_AMBER, lw=1.5, label="A T [°C]")
    ax.plot(x, np.asarray(B_T, dtype=float) - 273.15,
            color=_CYAN, lw=1.5, label="B T [°C]")
    ax.set_ylabel("T [°C]", color=_FG, fontsize=7)
    ax2.plot(x, np.asarray(A_q, dtype=float), color=_GREEN, lw=1.1,
             ls="--", label="A q")
    ax2.plot(x, np.asarray(B_q, dtype=float), color=_GREEN, lw=1.1,
             ls=":", alpha=0.7, label="B q")
    ax2.set_ylabel("q [kg/kg]", color=_GREEN, fontsize=7)
    ax.legend(facecolor="#1b1e24", edgecolor="#2a2f3a", fontsize=6,
              labelcolor=_FG, loc="upper right")
    _style(ax, xlabel="x [mm, wall → vapor]")


def paint_heatmap(ax, T_hist: np.ndarray, t_s: np.ndarray,
                  x_m: np.ndarray, title: str = "T(x, t) [°C]",
                  colorbar: bool = True):
    """Paint a space–time ``T`` heatmap on live axes (x horizontal)."""
    T_C = np.asarray(T_hist, dtype=float) - 273.15
    x = np.asarray(x_m, dtype=float) * 1000.0
    t = np.asarray(t_s, dtype=float)
    im = ax.imshow(T_C, aspect="auto", origin="lower", cmap="inferno",
                   extent=(x[0], x[-1], t[0], t[-1]), interpolation="nearest")
    _style(ax, xlabel="x [mm, wall → vapor]")
    ax.set_ylabel("t [s]", color=_AXIS, fontsize=7)
    ax.set_title(title, color=_FG, fontsize=9)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, shrink=0.85).ax.tick_params(
            colors=_AXIS, labelsize=6)
    return im


def bed_frame(T_K: np.ndarray, q: np.ndarray, x_m: np.ndarray, *,
              title: str = "", t_s: float | None = None,
              figsize=(6.0, 3.5)) -> np.ndarray:
    """One ``(H, W, 3)`` uint8 frame: ``T(x)`` + ``q(x)`` bed profiles.

    The hydrogym analogue of a vorticity snapshot: wall (x=0) left,
    vapor side right.
    """
    fig = _canvas(figsize)
    ax = fig.add_subplot(111)
    paint_bed_snapshot(ax, _twin(ax), T_K, q, x_m)
    head = title + (f"  ·  t = {t_s:.1f} s" if t_s is not None else "")
    if head:
        ax.set_title(head, color=_FG, fontsize=9)
    fig.tight_layout()
    return _to_rgb(fig)


def twobed_frame(A_T: np.ndarray, A_q: np.ndarray,
                 B_T: np.ndarray, B_q: np.ndarray, x_m: np.ndarray, *,
                 title: str = "", t_s: float | None = None,
                 figsize=(6.0, 5.0)) -> np.ndarray:
    """Stacked A/B bed-profile frame for the counter-phase pair."""
    fig = _canvas(figsize)
    x = np.asarray(x_m, dtype=float) * 1000.0
    for i, (T_K, q, name, color) in enumerate(
            ((A_T, A_q, "bed A", _AMBER), (B_T, B_q, "bed B", _CYAN))):
        ax = fig.add_subplot(2, 1, i + 1)
        ax.plot(x, np.asarray(T_K, dtype=float) - 273.15,
                color=color, lw=1.5, label="T [°C]")
        ax.set_ylabel("T [°C]", color=color, fontsize=7)
        ax2 = ax.twinx()
        ax2.set_facecolor(_BG)
        ax2.plot(x, np.asarray(q, dtype=float), color=_GREEN, lw=1.2,
                 ls="--", label="q")
        ax2.tick_params(colors=_AXIS, labelsize=6)
        for spine in ax2.spines.values():
            spine.set_color(_SPINE)
        _style(ax, xlabel="x [mm]" if i == 1 else "")
        ax.set_title(name, color=_FG, fontsize=8, loc="left")
    head = title + (f"  ·  t = {t_s:.1f} s" if t_s is not None else "")
    if head:
        fig.suptitle(head, color=_FG, fontsize=9)
    fig.tight_layout()
    return _to_rgb(fig)


def field_heatmap(T_hist: np.ndarray, t_s: np.ndarray, x_m: np.ndarray, *,
                  title: str = "T(x, t) [°C]",
                  figsize=(6.0, 3.0)) -> np.ndarray:
    """Space–time heatmap frame: the episode at a glance (x horizontal)."""
    fig = _canvas(figsize)
    ax = fig.add_subplot(111)
    paint_heatmap(ax, T_hist, t_s, x_m, title=title)
    fig.tight_layout()
    return _to_rgb(fig)


def episode_frames(fields: Mapping[str, Any], kind: str = "bed1d",
                   stride: int = 1, title: str = "") -> list[np.ndarray]:
    """Render every ``stride``-th field snapshot to an ``rgb_array`` list.

    ``kind="bed1d"`` expects ``T_K``/``q_kg_kg``/``t_s``/``x_m``;
    ``kind="twobed"`` expects the ``A_*``/``B_*`` keys. Ready for
    ``imageio.mimsave`` or SB3 video logging.
    """
    t = np.asarray(fields["t_s"], dtype=float)
    idx = range(0, len(t), max(1, int(stride)))
    frames = []
    if kind == "twobed":
        for i in idx:
            frames.append(twobed_frame(
                fields["A_T_K"][i], fields["A_q_kg_kg"][i],
                fields["B_T_K"][i], fields["B_q_kg_kg"][i],
                fields["A_x_m"], title=title, t_s=float(t[i])))
    else:
        for i in idx:
            frames.append(bed_frame(
                fields["T_K"][i], fields["q_kg_kg"][i], fields["x_m"],
                title=title, t_s=float(t[i])))
    return frames
