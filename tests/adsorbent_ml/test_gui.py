"""Offscreen tests for the adsorbent data explorer (adsorbent-ml/gui)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def app_module(qapp):
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "data_app_gui", repo / "adsorbent-ml" / "gui" / "data_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def window(app_module):
    return app_module.MainWindow()


class _StubWorker:
    """Mimics harness.gui.kit.Worker's surface for synchronous runs."""

    class _Signal:
        @staticmethod
        def emit(message: str) -> None:
            pass

    log = _Signal()
    cancel_requested = False


def test_all_pages_construct(window):
    assert [window.nav.item(i).text() for i in range(window.nav.count())] == [
        "Overview", "Materials", "Isotherms & fits", "Stability",
        "Structures", "Rankings"]


def test_overview_lists_built_datasets(window):
    assert window.overview.table.table.rowCount() >= 5


def test_materials_table_and_filter(window):
    assert window.materials.Y is not None and len(window.materials.Y) >= 150
    window.materials.family.setCurrentText("zeolite")
    rows = window.materials.table.table.rowCount()
    assert 0 < rows < 50
    window.materials.family.setCurrentText("(all families)")
    assert window.materials.table.table.rowCount() == len(window.materials.Y)


def test_isotherms_plot_real_fit(window):
    assert window.isotherms.listw.count() > 20
    first = window.isotherms.listw.item(0).text()
    window.isotherms._select(first)
    assert len(window.isotherms.fig.axes[0].lines) > 0


def test_stability_and_structures_populate(window):
    assert window.stability.pass_table.table.rowCount() > 1000
    assert window.structures.inv_table.table.rowCount() >= 4


def test_rankings_run_synchronous(app_module, window):
    """Anchor ranking through the Cycle0D oracle, fed into the view."""
    df = app_module._run_ranking(_StubWorker(), "datacenter", full=False)
    assert len(df) >= 10
    assert {"material", "score"} <= set(df.columns)
    window.rankings.on_finished(df)
    assert window.rankings.sweep.table.table.rowCount() >= 10
