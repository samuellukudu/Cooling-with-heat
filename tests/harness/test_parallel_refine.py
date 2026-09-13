"""CPU process-pool parallelism: parallel_map basics + refine parity.

The parallel Bed1D refinement must reproduce the serial loop exactly —
same rows, same values, same order — so ``workers=N`` changes wall-clock,
never numbers. Spawn-pool tests stay tiny (2 workers, 3 rows, a coarse
bed) because every child pays interpreter + jax import cost.
"""

import pandas as pd
import pytest

from harness import parallel
from harness.materials import load_anchors
from harness.rank import refine_with_bed1d, sweep_materials


def test_parallel_map_ordered_and_picked_up():
    items = [-3, 2, -1, 0]
    assert parallel.parallel_map(abs, items, workers=2) == [3, 2, 1, 0]


def test_parallel_map_serial_fallback_for_tiny_work():
    assert parallel.parallel_map(abs, [-1], workers=8) == [1]
    assert parallel.parallel_map(abs, [], workers=4) == []


def test_resolve_workers_clamps():
    assert parallel.resolve_workers(None, 0) == 1
    assert parallel.resolve_workers(64, 3) == 3
    assert parallel.resolve_workers(0, 5) == 1
    assert parallel.resolve_workers(None, 10_000) == parallel.default_workers()


def test_partition_shapes():
    assert parallel.partition([1, 2, 3, 4, 5], 2) == [[1, 2, 3], [4, 5]]
    assert parallel.partition([1, 2], 8) == [[1], [2]]
    assert parallel.partition([], 3) == []


@pytest.mark.parametrize("workers", [1, 2])
def test_refine_workers_match(workers):
    ranked = sweep_materials(load_anchors(), profiles=("datacenter",))
    ref = refine_with_bed1d(ranked, "datacenter", k=3, n_cells=8,
                            dt_phys_s=0.05, workers=1)
    alt = refine_with_bed1d(ranked, "datacenter", k=3, n_cells=8,
                            dt_phys_s=0.05, workers=workers)
    assert len(ref) == len(alt) == 3
    pd.testing.assert_frame_equal(ref, alt, rtol=1e-12, atol=0,
                                  check_exact=False)


def test_refine_empty_profile():
    ranked = sweep_materials(load_anchors(), profiles=("cpu",))
    empty = refine_with_bed1d(ranked, "vehicle", k=5, workers=2)
    assert isinstance(empty, pd.DataFrame) and empty.empty
