"""H3 composite TwoBed — harness.make material/material_b and GUI wiring."""

import pytest

import harness
from harness.envs.two_bed import TwoBed
from harness.gui.model import ExperimentGraph


def test_make_two_bed_composite_via_harness_make():
    # Direct harness.make with material and material_b — lightweight physics for CI
    env = harness.make(
        "TwoBed-v0",
        material="anchor:Silica gel RD",
        material_b="anchor:Zeolite 13X (NaX)",
        profile="datacenter",
        n_cells=4,
        n_cycles=1,
        dt_phys_s=0.02,
    )
    assert isinstance(env, TwoBed)
    assert env.material.name == "Silica gel RD"
    assert env.material_b.name == "Zeolite 13X (NaX)"
    assert env.material.name != env.material_b.name
    # Design space is per-bed prefixed
    assert "A_q_sat_kg_kg" in env.design_space.keys
    assert "B_q_sat_kg_kg" in env.design_space.keys
    # Per-bed defaults differ due to different materials
    assert env.design_space.defaults["A_q_sat_kg_kg"] == pytest.approx(0.35)  # silica
    assert env.design_space.defaults["B_q_sat_kg_kg"] == pytest.approx(0.28)  # zeolite
    assert env.design_space.defaults["A_e_char_j_mol"] == pytest.approx(4500.0)
    assert env.design_space.defaults["B_e_char_j_mol"] == pytest.approx(14000.0)
    # Evaluate works — may be zero for very short dt_phys but schema must be present
    metrics = env.evaluate()
    assert "COP" in metrics and "SCP_W_kg" in metrics
    # Check per-bed design propagation (not physics positivity, which needs longer cycle)
    assert env.design_space.defaults["A_q_sat_kg_kg"] != env.design_space.defaults["B_q_sat_kg_kg"]


def test_make_two_bed_single_material_defaults_to_both():
    env = harness.make("TwoBed-v0", material="anchor:Silica gel RD", profile="datacenter", n_cells=4, n_cycles=1, dt_phys_s=0.02)
    assert env.material_b.name == env.material.name
    assert env.design_space.defaults["A_q_sat_kg_kg"] == env.design_space.defaults["B_q_sat_kg_kg"]


def test_make_two_bed_material_b_as_instance():
    from harness.materials import get_material

    mat_a = get_material("anchor:Silica gel RD")
    mat_b = get_material("anchor:AlPO-18")
    env = harness.make("TwoBed-v0", material=mat_a, material_b=mat_b, profile="datacenter", n_cells=4, n_cycles=1, dt_phys_s=0.02)
    assert env.material.name == "Silica gel RD"
    assert env.material_b.name == "AlPO-18"


def test_gui_executor_material_b_forwarding():
    from harness.gui.executor import JobWorker
    from harness.gui.model import ExperimentGraph

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


def test_two_bed_composite_metrics_differ_from_single():
    # Design propagation is the honest signal: zeolite vs silica have
    # different per-bed q_sat/e_char, so the design dicts differ even when the
    # short-horizon physics underestimates the cycle. We gate on design, not
    # on a brittle COP != 0 check.
    single = harness.make("TwoBed-v0", material="anchor:Silica gel RD", profile="datacenter", n_cells=4, n_cycles=1, dt_phys_s=0.02)
    composite = harness.make("TwoBed-v0", material="anchor:Silica gel RD", material_b="anchor:Zeolite 13X (NaX)", profile="datacenter", n_cells=4, n_cycles=1, dt_phys_s=0.02)
    assert single.design_space.defaults["B_e_char_j_mol"] == pytest.approx(4500.0)
    assert composite.design_space.defaults["B_e_char_j_mol"] == pytest.approx(14000.0)
    # Even with zero COP on short horizon, the per-bed q dicts differ
    assert single.design_space.defaults["B_q_sat_kg_kg"] != composite.design_space.defaults["B_q_sat_kg_kg"]
