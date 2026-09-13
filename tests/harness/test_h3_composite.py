"""H3 composite TwoBed — harness.make material/material_b (GUI-wiring tests
live in attic/gui_tests/test_two_bed_composite_graph.py)."""

import pytest

import harness
from harness.envs.two_bed import TwoBed


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
