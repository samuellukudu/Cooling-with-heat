"""Registries, built-ins, and make() round-trips (DESIGN §7.1/§7.3)."""

import pytest

import harness
from harness import registry


def test_builtin_profiles_registered():
    assert set(harness.REGISTRIES["profiles"].names()) >= {"cpu", "human", "vehicle", "datacenter"}
    profile = harness.get_profile("datacenter")
    # Values mirrored verbatim from Materials/heat_cooling_screen.py.
    assert (profile.t_evap_c, profile.t_cond_c, profile.t_des_c, profile.cycle_time_s) == (16.0, 35.0, 60.0, 300.0)
    assert (profile.cop_weight, profile.scp_weight) == (0.35, 0.30)
    assert profile.description == "Data-center waste-heat cooling"


def test_builtin_anchor_materials_registered():
    names = harness.REGISTRIES["materials"].names()
    assert len(names) >= 13
    silica = harness.get_material("anchor:Silica gel RD")
    assert silica.q_sat_kg_kg == 0.35
    assert silica.q_st_j_kg == pytest.approx(2.5e6)
    assert silica.e_char_j_mol == 4500.0
    assert silica.n_da == 1.8
    assert silica.source == "anchor"
    assert silica.t_range_c == (20.0, 100.0)
    assert silica.transport_provenance == "none"


def test_unknown_name_error_lists_available():
    with pytest.raises(KeyError) as excinfo:
        harness.REGISTRIES["envs"].resolve("Nope-v0")
    message = str(excinfo.value)
    assert "Nope-v0" in message
    assert "Cycle0D-v0" in message  # the error teaches what exists


def test_register_make_round_trip():
    registry.register_env("TestDummy-v0", lambda **kw: {"received": kw}, overwrite=True)
    try:
        assert harness.make("TestDummy-v0", material="x") == {"received": {"material": "x"}}
        assert "TestDummy-v0" in harness.REGISTRIES["envs"]
    finally:
        del registry.REGISTRIES["envs"]._factories["TestDummy-v0"]
        registry.REGISTRIES["envs"]._entry_points_scanned = False


def test_collision_raises_without_overwrite():
    registry.register_profile("TestDup", lambda: None)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_profile("TestDup", lambda: None)
    registry.register_profile("TestDup", lambda: None, overwrite=True)  # allowed
    del registry.REGISTRIES["profiles"]._factories["TestDup"]
    registry.REGISTRIES["profiles"]._entry_points_scanned = False


def test_get_material_passthrough_instance():
    silica = harness.get_material("anchor:Silica gel RD")
    assert harness.get_material(silica) is silica


def test_get_profile_passthrough_instance():
    profile = harness.get_profile("cpu")
    assert harness.get_profile(profile) is profile
