"""sweep_materials_batched ≡ sweep_materials (the vmapped fast path).

The batched kernel must reproduce the per-material reference loop exactly —
same rows, same scores, same ranks — so the GUI/benchmark fast path and the
H2.3 reference numbers stay the *same* ranking, not a similar one.
"""

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")  # noqa: F841

from harness.materials import MaterialParams, load_anchors  # noqa: E402
from harness.rank import sweep_materials, sweep_materials_batched  # noqa: E402

NUMERIC_COLS = ["COP", "SCP_W_kg", "delta_q", "q_ads", "q_des", "score"]


def _sweep_set() -> list[MaterialParams]:
    mats = list(load_anchors())
    # extremes to exercise the degenerate/edge branches under vmap:
    mats.append(MaterialParams(
        name="edge-high", source="synthetic", q_sat_kg_kg=0.90,
        q_st_j_kg=4.1e6, e_char_j_mol=20000.0, n_da=4.0,
        t_range_c=(20.0, 100.0)))
    mats.append(MaterialParams(
        name="edge-low", source="synthetic", q_sat_kg_kg=0.08,
        q_st_j_kg=2.3e6, e_char_j_mol=2000.0, n_da=1.0,
        t_range_c=(20.0, 100.0)))
    return mats


@pytest.mark.parametrize("profiles", [("datacenter",), ("cpu", "vehicle")])
def test_batched_matches_loop(profiles):
    mats = _sweep_set()
    loop = sweep_materials(mats, profiles=profiles)
    batched = sweep_materials_batched(mats, profiles=profiles)
    assert len(loop) == len(batched) > 0
    pd.testing.assert_frame_equal(
        loop[["profile", "material"]], batched[["profile", "material"]])
    np.testing.assert_allclose(
        loop[NUMERIC_COLS].to_numpy(dtype=float),
        batched[NUMERIC_COLS].to_numpy(dtype=float),
        rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(loop["rank"].to_numpy(), batched["rank"].to_numpy())
    np.testing.assert_array_equal(
        loop["out_of_window"].to_numpy(), batched["out_of_window"].to_numpy())


def test_batched_empty_and_single():
    assert sweep_materials_batched([]).empty
    one = sweep_materials_batched([_sweep_set()[0]], profiles=("cpu",))
    assert len(one) == 1
