"""Model tests — toolkit-agnostic, no Qt required."""

from pathlib import Path

from harness.gui.model import ExperimentGraph


def test_default_cycle0d_valid():
    g = ExperimentGraph.default_cycle0d()
    assert not g.validate()  # Cycle0D demo should be valid


def test_default_bed1d_valid():
    g = ExperimentGraph.default_bed1d()
    assert not g.validate()


def test_roundtrip_json(tmp_path: Path):
    g = ExperimentGraph.default_cycle0d()
    g.name = "roundtrip"
    p = tmp_path / "graph.harness.json"
    g.save(p)
    g2 = ExperimentGraph.load(p)
    assert g2.name == "roundtrip"
    assert len(g2.nodes) == len(g.nodes)
    assert len(g2.edges) == len(g.edges)
    assert g2.validate() == []


def test_to_python_cycle0d():
    g = ExperimentGraph.default_cycle0d()
    code = g.to_python()
    assert "harness.make" in code
    assert "Cycle0D-v0" in code
    assert "harness.optimize" in code
    # should be valid python
    compile(code, "<generated>", "exec")


def test_to_python_bed1d():
    g = ExperimentGraph.default_bed1d()
    code = g.to_python()
    assert "Bed1D-v0" in code
    compile(code, "<generated>", "exec")


def test_apply_patch_material():
    from harness.gui.command_palette import apply_patch, parse_command

    g = ExperimentGraph.default_cycle0d()
    patch = parse_command("material 13X")
    assert "material" in patch
    apply_patch(g, patch)
    # material ref should have changed to 13X anchor
    refs = [n.data.get("ref") for n in g.nodes if n.type == "Material"]
    assert any("13X" in r for r in refs)


def test_validation_missing_physics():
    g = ExperimentGraph(name="empty")
    errs = g.validate()
    assert any(e["code"] == "no_physics" for e in errs)


def test_validation_rl_on_static():
    g = ExperimentGraph.default_cycle0d()
    for n in g.nodes:
        if n.type == "Optimizer":
            n.data["kind"] = "rl"
    errs = g.validate()
    assert any(e["code"] == "rl_on_static" for e in errs)


def test_add_remove_node():
    g = ExperimentGraph.default_cycle0d()
    n0 = len(g.nodes)
    e0 = len(g.edges)
    m = g.add_node("Material", x=0, y=0, data={"ref": "anchor:zeolite 13X"})
    assert len(g.nodes) == n0 + 1
    g.remove_node(m.id)
    assert len(g.nodes) == n0
    # edges incident to removed node also gone
    assert len(g.edges) <= e0


def test_physics_kwargs_filtering():
    g = ExperimentGraph.default_cycle0d()
    # Cycle0D should only expose hx_mass_factor, not n_cells
    kw = g.physics_kwargs()
    assert "hx_mass_factor" in kw
    assert "n_cells" not in kw

    g2 = ExperimentGraph.default_bed1d()
    kw2 = g2.physics_kwargs()
    # Bed1D exposes n_cells etc filtered via PHYSICS_KWARGS, but hx goes to design via executor split
    assert "n_cells" in kw2
