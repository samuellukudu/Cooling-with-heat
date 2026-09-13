"""Command palette parser tests."""

from harness.gui.command_palette import apply_patch, parse_command
from harness.gui.model import ExperimentGraph


def test_parse_material():
    assert parse_command("material silica gel rd")["material"] == "anchor:Silica gel RD"
    assert parse_command("material 13X")["material"] == "anchor:zeolite 13X"


def test_parse_profile():
    assert parse_command("profile datacenter")["profile"] == "datacenter"
    assert parse_command("optimize for SCP under human")["profile"] == "human"


def test_parse_physics():
    assert parse_command("physics Bed1D")["physics"] == "Bed1D-v0"
    assert parse_command("two-bed with heat recovery")["physics"] == "TwoBed-v0"
    assert parse_command("cycle oracle")["physics"] == "Cycle0D-v0"


def test_parse_objective():
    assert parse_command("optimize for COP")["objective"] == "COP"
    assert parse_command("optimize for SCP")["objective"] == "SCP_W_kg"
    assert parse_command("optimize for COP and SCP")["objective"] == "COP+SCP"


def test_parse_optimizer():
    assert parse_command("search cmaes")["optimizer"] == "search:cmaes"
    assert parse_command("grad adam")["optimizer"] == "grad"


def test_apply_patch_changes_graph():
    g = ExperimentGraph.default_cycle0d()
    patch = parse_command("optimize Bed1D for SCP under datacenter")
    apply_patch(g, patch)
    assert g.physics_node().data["kind"] == "Bed1D-v0"
    assert g.profile_ref() == "datacenter"
    obj = g.objective_node()
    assert obj.data["weights"].get("SCP_W_kg") == 1.0
