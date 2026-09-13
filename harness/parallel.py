"""CPU process-pool parallelism for per-item-expensive harness work.

Two CPU regimes live in this repo, and only one of them needs this module:

- *Inside one big jitted call* — PINN training, the vmapped ranking kernel —
  XLA's Eigen threadpool already uses every core. One process is enough; a
  pool would only oversubscribe the machine.
- *Across many independent, per-item-expensive calls* — ``Bed1D`` episode
  evaluations (~1 s per material at default resolution), seed sweeps,
  cross-validation folds — the glue between jitted calls is Python and the
  extra cores sit idle. :func:`parallel_map` distributes those items over a
  pool of fresh interpreter processes.

Measure before reaching for either lever: the Cycle0D sweep that once
motivated "just parallelize it" costs ~60 ms for the whole fitted table.

Design notes:

- Spawn context, never fork: the caller may already hold an initialized JAX
  backend, and forking after XLA starts its thread pools can deadlock.
  Spawned workers are fresh interpreters that import what the task needs.
- Workers are pinned to CPU (``JAX_PLATFORMS=cpu`` — this module *is* the
  CPU path; a parent's GPU opt-in must not leak into children).
- ``fn`` is pickled by reference, so it must be importable at module level:
  lambdas, closures, and ``__main__``-defined functions cannot cross the
  pool. Arguments and return values must pickle too (plain data, dataclasses).
- The usual spawn contract applies to callers: keep script entry points
  under ``if __name__ == "__main__":`` so children re-importing ``__main__``
  don't re-run them.
- Results return in input order; worker exceptions re-raise in the parent.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def default_workers() -> int:
    """Half the cores: each spawned worker runs its own XLA/Eigen thread
    pool, so one worker per core oversubscribes the machine."""
    return max(1, (os.cpu_count() or 2) // 2)


def resolve_workers(workers: int | None, n_items: int) -> int:
    """Clamp a workers request to something the item count can actually use."""
    n = default_workers() if workers is None else max(1, int(workers))
    return max(1, min(n, int(n_items)))


def partition(items: Sequence[T], n_chunks: int) -> list[list[T]]:
    """Contiguous chunks — at most ``n_chunks`` of them, never empty."""
    if not items:
        return []
    n = max(1, min(int(n_chunks), len(items)))
    size, rem = divmod(len(items), n)
    chunks: list[list[T]] = []
    start = 0
    for i in range(n):
        end = start + size + (1 if i < rem else 0)
        chunks.append(list(items[start:end]))
        start = end
    return chunks


def _init_worker() -> None:
    # Runs before any task unpickles (and thus before anything imports jax
    # in the child): force the CPU backend regardless of what the parent
    # process had selected.
    os.environ["JAX_PLATFORMS"] = "cpu"


def parallel_map(fn: Callable[[T], R], items, *,
                 workers: int | None = None) -> list[R]:
    """Map ``fn`` over ``items``, in order, across CPU-pinned processes.

    Falls back to a plain loop when the item count (or the worker clamp)
    leaves nothing to distribute. ``workers=None`` means
    :func:`default_workers`.
    """
    items = list(items)
    n = resolve_workers(workers, len(items))
    if n <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    # Inherited by spawned children even before the initializer runs.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx,
                             initializer=_init_worker) as pool:
        return list(pool.map(fn, items))


__all__ = ["default_workers", "partition", "parallel_map", "resolve_workers"]
