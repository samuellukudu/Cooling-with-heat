"""Executor tests — headless, no canvas required."""

import pytest

from harness.gui.executor import JobWorker
from harness.gui.model import ExperimentGraph


def _run_graph(graph: ExperimentGraph):
    """Helper that runs synchronously (no QThread) and returns payload or raises."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    w = JobWorker()
    payloads: list = []
    errors: list = []
    w.finished.connect(lambda p: payloads.append(p))
    w.failed.connect(lambda s: errors.append(s))
    w.run(graph)
    app.processEvents()
    if errors:
        raise AssertionError(f"executor failed: {errors[0][:2000]}")
    assert payloads, "executor did not emit finished"
    return payloads[0]


def test_executor_cycle0d_search():
    g = ExperimentGraph.default_cycle0d()
    for n in g.nodes:
        if n.type == "Optimizer":
            n.data["budget"] = 12
            n.data["method"] = "cmaes"
    payload = _run_graph(g)
    assert payload.result is not None
    assert "COP" in payload.metrics
    assert payload.result.n_evals > 0
    assert "harness.make" in payload.python_code


def test_executor_cycle0d_evaluate_no_optimizer():
    g = ExperimentGraph.default_cycle0d()
    # remove optimizer -> evaluate path
    g.nodes = [n for n in g.nodes if n.type != "Optimizer"]
    g.edges = [e for e in g.edges if "optimizer" not in e.from_node]
    # also need to ensure no dangling edge ids reference removed node
    # (remove edges to nowhere)
    node_ids = {n.id for n in g.nodes}
    g.edges = [e for e in g.edges if e.from_node in node_ids and e.to_node in node_ids]
    payload = _run_graph(g)
    assert payload.result is None
    assert "COP" in payload.metrics


def test_executor_bed1d_evaluate():
    g = ExperimentGraph.default_bed1d()
    # keep only evaluate path (remove optimizer)
    g.nodes = [n for n in g.nodes if n.type != "Optimizer"]
    node_ids = {n.id for n in g.nodes}
    g.edges = [e for e in g.edges if e.from_node in node_ids and e.to_node in node_ids]
    for n in g.nodes:
        if n.type == "Physics":
            n.data["n_cells"] = 8
            n.data["n_cycles"] = 2
    payload = _run_graph(g)
    assert "COP" in payload.metrics
    assert "SCP_W_kg" in payload.metrics
