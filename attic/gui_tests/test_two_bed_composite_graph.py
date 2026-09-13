"""TwoBed composite wiring on the ExperimentGraph (parked with the GUI).

Moved from tests/harness/test_h3_composite.py when the GUI was parked in
attic/ — these exercise harness.gui.model/executor, not harness itself.
"""

from harness.gui.model import ExperimentGraph


def test_gui_executor_material_b_forwarding():
    from harness.gui.executor import JobWorker

    g = ExperimentGraph(name="TwoBed composite demo")
    m_a = g.add_node("Material", x=0, y=0, data={"ref": "anchor:Silica gel RD"})
    m_b = g.add_node("Material", x=0, y=80, data={"ref": "anchor:Zeolite 13X (NaX)"})
    p = g.add_node("Profile", x=0, y=160, data={"ref": "datacenter"})
    phys = g.add_node("Physics", x=200, y=80, data={"kind": "TwoBed-v0", "n_cells": 4, "n_cycles": 1})
    # Wire material -> material, second material -> material_b
    g.add_edge(m_a.id, "material", phys.id, "material")
    g.add_edge(m_b.id, "material_b", phys.id, "material_b")
    g.add_edge(p.id, "profile", phys.id, "profile")

    worker = JobWorker()
    # Use the graph helpers
    assert g.material_ref() == "anchor:Silica gel RD"
    assert g.material_b_ref() == "anchor:Zeolite 13X (NaX)"
    # Build problem via executor's _make_problem with material_b forwarding
    prob = worker._make_problem("TwoBed-v0", g.material_ref(), g.profile_ref(), g.physics_kwargs(), material_b=g.material_b_ref())
    assert prob.material.name == "Silica gel RD"
    assert prob.material_b.name == "Zeolite 13X (NaX)"


def test_gui_material_b_via_data_field():
    g = ExperimentGraph(name="TwoBed data-field composite")
    g.add_node("Material", x=0, y=0, data={"ref": "anchor:Silica gel RD"})
    g.add_node("Profile", x=0, y=80, data={"ref": "datacenter"})
    phys = g.add_node("Physics", x=200, y=40, data={"kind": "TwoBed-v0", "n_cells": 4, "n_cycles": 1, "material_b": "anchor:Zeolite 13X (NaX)"})
    # Even without a second Material node, material_b data field should be read
    assert g.material_b_ref() == "anchor:Zeolite 13X (NaX)"
    # Fallback when not set
    g2 = ExperimentGraph(name="No composite")
    g2.add_node("Material", x=0, y=0, data={"ref": "anchor:Silica gel RD"})
    g2.add_node("Profile", x=0, y=80, data={"ref": "datacenter"})
    g2.add_node("Physics", x=200, y=40, data={"kind": "Bed1D-v0", "n_cells": 4})
    assert g2.material_b_ref() is None


def test_to_python_includes_material_b():
    g = ExperimentGraph(name="TwoBed to_python")
    m_a = g.add_node("Material", x=0, y=0, data={"ref": "anchor:Silica gel RD"})
    m_b = g.add_node("Material", x=0, y=80, data={"ref": "anchor:Zeolite 13X (NaX)"})
    p = g.add_node("Profile", x=0, y=160, data={"ref": "datacenter"})
    phys = g.add_node("Physics", x=200, y=80, data={"kind": "TwoBed-v0", "n_cells": 4})
    g.add_edge(m_a.id, "material", phys.id, "material")
    g.add_edge(m_b.id, "material_b", phys.id, "material_b")
    g.add_edge(p.id, "profile", phys.id, "profile")
    src = g.to_python()
    assert "material_b=" in src
    assert "Zeolite 13X" in src
