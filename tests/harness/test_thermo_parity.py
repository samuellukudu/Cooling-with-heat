"""V1 (DESIGN §9): harness.physics.thermo is an exact mirror of
Materials/cooling_physics.py — max relative error < 1e-12 on a dense grid —
plus an optional CoolProp reference check (the ``cool`` extra)."""

import numpy as np
import pytest
import jax.numpy as jnp

import cooling_physics
from harness.physics import thermo

# -40 .. 150 °C dense; includes the 100 °C branch point exactly.
T_GRID_K = np.linspace(233.15, 423.15, 1201)
BRANCH_T_K = np.array([372.149, 373.15, 373.151])  # just below / at / above


def _canonical_sat(t_k):
    return np.array([cooling_physics.water_sat_pressure_pa(float(t)) for t in t_k])


def _canonical_hfg(t_k):
    return np.array([cooling_physics.water_h_fg_j_kg(float(t)) for t in t_k])


def test_sat_pressure_parity_dense_grid():
    ref = _canonical_sat(T_GRID_K)
    got = np.asarray(thermo.water_sat_pressure_pa(jnp.asarray(T_GRID_K)))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)


def test_sat_pressure_branch_point_exact():
    """Magnus and Antoine differ by ~1.8 % at 100 °C, so this pins the branch."""
    ref = _canonical_sat(BRANCH_T_K)
    got = np.asarray(thermo.water_sat_pressure_pa(jnp.asarray(BRANCH_T_K)))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)


def test_h_fg_parity_dense_grid():
    ref = _canonical_hfg(T_GRID_K)
    got = np.asarray(thermo.water_h_fg_j_kg(jnp.asarray(T_GRID_K)))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0)


def test_constants_match_canonical():
    assert thermo.GAS_CONSTANT == 8.314
    assert thermo.CP_ADSORBENT == 1000.0
    assert thermo.CP_LIQUID == 4184.0


def test_coolprop_reference():
    """Optional reference check (CoolProp = IAPWS-IF97); skipped unless the
    ``cool`` extra is installed. Tolerances = the correlation error budgets
    documented in thermo.py (<0.3 % Psat, <0.5 % h_fg)."""
    CoolProp = pytest.importorskip("CoolProp")
    for t_k in (273.15 + 0.0, 273.15 + 16.0, 273.15 + 35.0, 273.15 + 60.0, 273.15 + 100.0, 273.15 + 150.0):
        p_ref = CoolProp.CoolProp.PropsSI("P", "T", t_k, "Q", 1, "Water")
        p_got = float(thermo.water_sat_pressure_pa(jnp.float64(t_k)))
        assert p_got == pytest.approx(p_ref, rel=0.005), f"Psat mismatch at {t_k} K"
        h_ref = CoolProp.CoolProp.PropsSI("H", "T", t_k, "Q", 1, "Water") - CoolProp.CoolProp.PropsSI(
            "H", "T", t_k, "Q", 0, "Water"
        )
        h_got = float(thermo.water_h_fg_j_kg(jnp.float64(t_k)))
        assert h_got == pytest.approx(h_ref, rel=0.005), f"h_fg mismatch at {t_k} K"
