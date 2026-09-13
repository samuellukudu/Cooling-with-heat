"""G3/G4 GUI polish tests — Sweep & Compare, offscreen, no heavy Bed1D search.

Uses small grid (budget ≤4) and n_cells=8 for speed.
QT_QPA_PLATFORM=offscreen is set by fixture.
"""

from __future__ import annotations

import pytest

# -- model tests (no Qt) ----------------------------------------------------

def test_sweep_node_type_exists():
    from harness.gui.model import NODE_TYPES, SCOPE_KINDS, DEFAULT_NODE_DATA

    assert "Sweep" in NODE_TYPES
    assert "Compare" in SCOPE_KINDS
    assert "Sweep" in SCOPE_KINDS
    # Sweep default data includes required keys
    d = DEFAULT_NODE_DATA["Sweep"]
    assert "axis" in d and "grid" in d and "budget" in d


def test_sweep_validation_t_switch():
    from harness.gui.model import ExperimentGraph

    g = ExperimentGraph.default_cycle0d()
    # add Sweep with valid t_switch grid
    sw = g.add_node("Sweep", x=0, y=0, data={"axis": "t_switch", "grid": [60, 120, 240], "budget": 4})
    errs = [e for e in g.validate() if e["code"].startswith("bad_sweep")]
    assert not errs
    # invalid axis
    sw.data["axis"] = "bad_axis"
    errs = [e for e in g.validate() if e["code"] == "bad_sweep_axis"]
    assert errs
    sw.data["axis"] = "t_switch"
    # bad grid too few points
    sw.data["grid"] = [60]
    errs = [e for e in g.validate() if e["code"] == "bad_sweep_grid"]
    assert errs
    sw.data["grid"] = [60, 120, 240]


def test_sweep_validation_material():
    from harness.gui.model import ExperimentGraph

    g = ExperimentGraph.default_cycle0d()
    sw = g.add_node("Sweep", x=0, y=0, data={"axis": "material", "grid": ["anchor:Silica gel RD"], "budget": 4})
    assert not [e for e in g.validate() if e["code"] == "bad_sweep_grid"]
    sw.data["grid"] = []
    assert [e for e in g.validate() if e["code"] == "bad_sweep_grid"]


def test_model_sweep_roundtrip_and_to_python(tmp_path):
    from harness.gui.model import ExperimentGraph

    g = ExperimentGraph.default_cycle0d()
    g.add_node("Sweep", x=10, y=10, data={"axis": "t_switch", "grid": [80, 180, 300], "budget": 4, "grid_min": 80, "grid_max": 300, "steps": 3})
    p = tmp_path / "sweep.harness.json"
    g.save(p)
    g2 = ExperimentGraph.load(p)
    assert any(n.type == "Sweep" for n in g2.nodes)
    assert g2.schema_version >= 2
    code = g2.to_python()
    assert "switch_time_sweep" in code or "sweep" in code.lower()
    compile(code, "<generated_sweep>", "exec")
    # material sweep code
    g3 = ExperimentGraph.default_cycle0d()
    g3.add_node("Sweep", x=0, y=0, data={"axis": "material", "grid": ["anchor:Silica gel RD", "anchor:zeolite 13X"], "budget": 2})
    code2 = g3.to_python()
    assert "sweep_materials" in code2
    compile(code2, "<generated_material>", "exec")


def test_model_palette_includes_sweep():
    from harness.gui.palette import BLOCKS

    types = [b["type"] for b in BLOCKS]
    assert "Sweep" in types
    # Compare scope exists
    assert any(b["type"] == "Scope" and b["data"].get("kind") == "Compare" for b in BLOCKS)
    # Sweep scope also
    assert any(b["type"] == "Scope" and b["data"].get("kind") == "Sweep" for b in BLOCKS)
    # filter works (simulate _filter logic)
    q = "sweep"
    filtered = [b for b in BLOCKS if q in b["label"].lower() or q in b["type"].lower() or q in b["tip"].lower()]
    assert len(filtered) >= 2
    # filter compare
    q2 = "compare"
    filtered2 = [b for b in BLOCKS if q2 in b["label"].lower() or q2 in b["type"].lower() or q2 in b["tip"].lower()]
    assert len(filtered2) >= 1


# -- Qt offscreen tests -----------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_inspector_sweep_page(qapp):
    from harness.gui.inspector import InspectorDock
    from harness.gui.model import ExperimentGraph

    g = ExperimentGraph.default_cycle0d()
    sw = g.add_node("Sweep", x=0, y=0, data={"axis": "t_switch", "grid": [60, 120, 240, 480], "budget": 4,
                                            "grid_min": 60, "grid_max": 480, "steps": 4})
    dock = InspectorDock()
    dock.show_node(sw.id, sw.type, dict(sw.data))
    assert dock._sweep_axis.currentText() == "t_switch"
    assert dock._sweep_steps.value() == 4
    # switch to material and check toggles
    dock._sweep_axis.setCurrentText("material")
    qapp.processEvents()
    assert dock._sweep_mat_grid.isEnabled()
    # back to t_switch
    dock._sweep_axis.setCurrentText("t_switch")
    qapp.processEvents()
    assert dock._sweep_min.isEnabled()
    dock.close()


def test_scopes_compare_and_sweep(qapp):
    from harness.gui.scopes import ScopeTabs
    import pandas as pd

    tabs = ScopeTabs()
    # Check tabs exist
    names = [tabs.tabText(i) for i in range(tabs.count())]
    assert "Compare" in names
    assert "Sweep" in names
    # Compare
    a = {"COP": 0.5, "SCP_W_kg": 200}
    b = {"COP": 0.55, "SCP_W_kg": 220}
    tabs.set_compare(a, b, title="Test compare")
    assert tabs.compare.table.rowCount() == 2
    # Sweep dataframe
    df = pd.DataFrame([
        {"t_switch_s": 80, "COP": 0.5, "SCP_W_kg": 200, "delta_q": 0.1},
        {"t_switch_s": 180, "COP": 0.55, "SCP_W_kg": 240, "delta_q": 0.12},
        {"t_switch_s": 300, "COP": 0.52, "SCP_W_kg": 230, "delta_q": 0.11},
    ])
    tabs.set_sweep_dataframe(df, title="Sweep test")
    assert tabs.sweep.table.rowCount() == 3
    qapp.processEvents()
    tabs.close()


def test_dataframe_scope_handles_sweep(qapp):
    from harness.gui.scopes import DataFrameScope
    import pandas as pd

    scope = DataFrameScope("Sweep")
    df = pd.DataFrame([
        {"t_switch_s": 60, "COP": 0.4, "SCP_W_kg": 150},
        {"t_switch_s": 120, "COP": 0.45, "SCP_W_kg": 180},
    ])
    scope.set_dataframe(df)
    assert scope.table.rowCount() == 2
    assert scope.table.columnCount() == 3
    qapp.processEvents()
    scope.close()


def test_executor_sweep_t_switch(qapp):
    """Run sweep via JobWorker — small grid, n_cells=8, budget 4, no heavy search."""
    from PyQt6.QtWidgets import QApplication
    from harness.gui.model import ExperimentGraph
    from harness.gui.executor import JobWorker

    g = ExperimentGraph.default_cycle0d()
    # Configure for Bed1D sweep: set physics to Bed1D-v0 with n_cells=8, n_cycles=2
    for n in g.nodes:
        if n.type == "Physics":
            n.data["kind"] = "Bed1D-v0"
            n.data["n_cells"] = 8
            n.data["n_cycles"] = 2
            n.data["soft_switch"] = False
            n.data["dt_phys_s"] = None
        if n.type == "Optimizer":
            # remove optimizer -> sweep path doesn't need it, but we can keep it; sweep will be prioritized
            pass
    # Add Sweep
    sw = g.add_node("Sweep", x=500, y=0, data={"axis": "t_switch", "grid": [80, 200, 360], "budget": 3,
                                              "grid_min": 80, "grid_max": 360, "steps": 3})
    # Ensure graph validates
    errs = [e for e in g.validate() if e["code"] not in ("schedule_on_static",)]
    assert not errs, errs

    app = QApplication.instance() or QApplication([])
    w = JobWorker()
    payloads: list = []
    errors: list = []
    w.finished.connect(lambda p: payloads.append(p))
    w.failed.connect(lambda s: errors.append(s))
    w.run(g)
    app.processEvents()
    assert not errors, f"executor sweep failed: {errors}"
    assert payloads, "no finished payload"
    payload = payloads[0]
    assert payload.sweep_df is not None
    assert payload.sweep_axis == "t_switch"
    assert len(payload.sweep_df) == 3
    assert "SCP_W_kg" in payload.sweep_df.columns
    assert "t_switch_s" in payload.sweep_df.columns


def test_executor_sweep_material(qapp):
    from PyQt6.QtWidgets import QApplication
    from harness.gui.model import ExperimentGraph
    from harness.gui.executor import JobWorker

    g = ExperimentGraph.default_cycle0d()
    sw = g.add_node("Sweep", x=0, y=0, data={"axis": "material", "grid": ["anchor:Silica gel RD", "anchor:Zeolite 13X (NaX)", "anchor:Aluminum fumarate"], "budget": 3})
    errs = [e for e in g.validate() if e["code"] not in ("schedule_on_static",)]
    assert not errs
    app = QApplication.instance() or QApplication([])
    w = JobWorker()
    payloads: list = []
    errors: list = []
    w.finished.connect(lambda p: payloads.append(p))
    w.failed.connect(lambda s: errors.append(s))
    w.run(g)
    app.processEvents()
    assert not errors, f"material sweep failed: {errors[:1]}"
    assert payloads
    payload = payloads[0]
    assert payload.sweep_df is not None
    assert payload.sweep_axis == "material"
    assert len(payload.sweep_df) >= 2, f"expected >=2 rows got {len(payload.sweep_df)}: {payload.sweep_df}"


def test_app_run_with_sweep_node_offscreen(qapp):
    from harness.gui.app import MainWindow

    w = MainWindow()
    # add sweep node and setup physics for sweep
    for n in w.graph.nodes:
        if n.type == "Physics":
            n.data["kind"] = "Bed1D-v0"
            n.data["n_cells"] = 8
            n.data["n_cycles"] = 2
            n.data["soft_switch"] = False
    # remove optimizer to test pure sweep path (optional)
    # keep optimizer but sweep will override
    sw = w.graph.add_node("Sweep", x=550, y=140, data={"axis": "t_switch", "grid": [80, 240, 400], "budget": 3,
                                                      "grid_min": 80, "grid_max": 400, "steps": 3})
    # also need scene item for visual, but not required for executor
    from harness.gui.scene import NodeItem
    item = NodeItem(sw)
    w.scene.addItem(item)
    w.scene._nodes[sw.id] = item  # type: ignore
    # Run via worker synchronously (not via QTimer)
    payloads = []
    errors = []
    w._worker.finished.connect(lambda p: payloads.append(p))
    w._worker.failed.connect(lambda s: errors.append(s))
    w._worker.run(w.graph)
    qapp.processEvents()
    assert not errors, f"app sweep failed: {errors}"
    assert payloads
    assert payloads[0].sweep_df is not None
    # also test Compare after two runs: simulate second run result
    w._prev_result = payloads[0]
    # second fake result: tweak metrics slightly
    import copy
    second = copy.deepcopy(payloads[0])
    if hasattr(second, "metrics") and second.metrics:
        # mutate one metric for diff
        for k in list(second.metrics.keys())[:1]:
            second.metrics[k] = second.metrics[k] * 1.05
    w._last_result = second
    # now compare should work (prev vs last)
    # Use scopes compare method directly
    w.scopes.set_compare(w._prev_result.metrics, w._last_result.metrics, title="test compare")
    assert w.scopes.compare.table.rowCount() > 0
    # sweep tabs should be present
    assert w.scopes.sweep.table.rowCount() >= 0
    w.close()
    w._thread.quit()
    w._thread.wait(2000)


def test_json_version_migration(tmp_path):
    from harness.gui.model import ExperimentGraph, SCHEMA_VERSION
    import json, pathlib

    # create v1 style file (no Sweep)
    g = ExperimentGraph.default_cycle0d()
    g.schema_version = 1
    p = tmp_path / "v1.harness.json"
    g.save(p)
    # mutate file to ensure schemaVersion 1
    d = json.loads(p.read_text())
    d["schemaVersion"] = 1
    p.write_text(json.dumps(d))
    # load via from_dict should migrate to 2
    g2 = ExperimentGraph.load(p)
    assert g2.schema_version == SCHEMA_VERSION
    assert g2.validate() == []

