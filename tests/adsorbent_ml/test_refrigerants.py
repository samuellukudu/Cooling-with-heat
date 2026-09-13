"""Tests for the working-fluid library (ACQUISITION §7).

Pins physical spot values (boiling points, the ACQUISITION §7 kPa anchors),
water-branch agreement with the harness correlations, and the fail-loudly
contract outside Antoine windows.
"""

import math

import numpy as np
import pytest

from refrigerants import h_fg_j_kg, list_refrigerants, psat_pa


def test_known_fluids_listed():
    assert set(list_refrigerants()) == {"water", "methanol", "ethanol", "ammonia"}


def test_boiling_points_at_1_atm():
    """Each fluid boils at its tabulated Tb: Psat(Tb) = 760 mmHg."""
    for fluid, tb_c in (("water", 100.0), ("methanol", 64.7),
                        ("ethanol", 78.4), ("ammonia", -33.3)):
        assert psat_pa(fluid, tb_c + 273.15) == pytest.approx(760 * 133.322, rel=0.02)


def test_acquisition_spot_pressures():
    """ACQUISITION.md §7 anchors: methanol ≈ 28 kPa, ammonia ≈ 1350 kPa at 35 °C."""
    assert psat_pa("methanol", 35.0 + 273.15) == pytest.approx(28e3, rel=0.05)
    assert psat_pa("ammonia", 35.0 + 273.15) == pytest.approx(1350e3, rel=0.05)


def test_water_branch_matches_harness_thermo():
    """Water Antoine vs harness Magnus/Antoine: both approximate IF97, so they
    must agree to a few percent over the shared 5–95 °C window."""
    from harness.physics.thermo import water_sat_pressure_pa

    import jax.numpy as jnp

    for t_c in (5.0, 20.0, 35.0, 60.0, 80.0, 95.0):
        ours = psat_pa("water", t_c + 273.15)
        theirs = float(water_sat_pressure_pa(jnp.asarray(t_c + 273.15)))
        assert ours == pytest.approx(theirs, rel=0.03)


def test_psat_monotone_and_array_path():
    grid = np.linspace(10.0, 90.0, 9) + 273.15
    p = psat_pa("water", grid)
    assert isinstance(p, np.ndarray) and p.shape == grid.shape
    assert bool(np.all(np.diff(p) > 0.0))


def test_out_of_window_fails_loudly():
    with pytest.raises(ValueError, match="outside Antoine window"):
        psat_pa("water", 150.0 + 273.15)
    with pytest.raises(ValueError, match="outside Antoine window"):
        psat_pa("ammonia", 80.0 + 273.15)
    with pytest.raises(KeyError):
        psat_pa("r134a", 300.0)


def test_latent_heat_anchors_and_order():
    """Watson is exact at its anchor; magnitudes follow water > ammonia >
    methanol > ethanol at 25 °C (tabulated order)."""
    assert h_fg_j_kg("water", 373.15) == pytest.approx(2_256_400.0, rel=1e-12)
    assert h_fg_j_kg("ammonia", 239.85) == pytest.approx(1_369_000.0, rel=1e-12)
    t = 25.0 + 273.15
    assert h_fg_j_kg("water", t) == pytest.approx(2.44e6, rel=0.03)
    assert 1.0e6 < h_fg_j_kg("ammonia", t) < 1.3e6
    assert 1.0e6 < h_fg_j_kg("methanol", t) < 1.3e6
    assert 0.85e6 < h_fg_j_kg("ethanol", t) < 1.0e6
    assert h_fg_j_kg("water", t) > h_fg_j_kg("methanol", t) > 500e3
    with pytest.raises(ValueError):
        h_fg_j_kg("water", 700.0)
