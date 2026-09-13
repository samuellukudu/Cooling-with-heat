"""GPU memory hygiene for the harness (small-VRAM / shared-GPU machines).

The project's GPU is a 4 GB laptop RTX 3050 shared with the display, and the
physics runs float64 (V1 parity requirement). Three levers, in one place:

- :func:`configure` — called by the GUI entry points (and safe to call from
  scripts) *before the first jax device use*: disables XLA preallocation so
  JAX takes VRAM as needed instead of grabbing ~75% of the card, and selects
  the platform allocator so freed device memory is actually returned to the
  driver (the caching default holds it until process exit). Trade-off: each
  allocation is slower; for harness-sized problems this is irrelevant.
- :func:`is_oom` — recognizes CUDA/XLA out-of-memory errors from anywhere in
  an exception chain (XlaRuntimeError, RESOURCE_EXHAUSTED, ...).
- :func:`clear_caches` — drops jitted executables after a job finishes so
  VRAM is reclaimed while a long-lived process (the GUIs) stays open.
  Scripts reclaim everything at process exit regardless.

CPU fallback note: the JAX backend is fixed at first device use inside a
process. OOM handling therefore means *fail loudly with the remedy* (set
``JAX_PLATFORMS=cpu`` for the process, or free VRAM) — there is no in-process
GPU→CPU switch.
"""

from __future__ import annotations

import os
import sys
from typing import Any

OOM_MARKERS = (
    "out of memory",
    "oom",
    "resource_exhausted",
    "memory limit",
    "cuda_error_out_of_memory",
    "failed to allocate",
)


def configure(*, preallocate: bool = False, platform_allocator: bool = True) -> None:
    """Set XLA env vars before jax initializes its backend (idempotent).

    Only sets variables the user has not set — explicit environment always
    wins. Import this module and call :func:`configure` before importing jax
    anywhere in the process (GUI entry points do; jax is imported lazily by
    the rest of the package).
    """
    if not preallocate:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if platform_allocator:
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


def is_oom(exc: BaseException | None) -> bool:
    """True if the exception (or any nested cause/context) is a VRAM OOM."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(marker in text for marker in OOM_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def clear_caches() -> None:
    """Free jitted executables + their device buffers (best-effort)."""
    try:
        import jax  # noqa: PLC0415

        jax.clear_caches()
    except Exception:
        pass


def devices() -> str:
    """One-line device summary for logs/status bars."""
    try:
        import jax  # noqa: PLC0415

        return ", ".join(str(d) for d in jax.devices())
    except Exception as exc:  # noqa: BLE001
        return f"(unavailable: {exc})"


def oom_message(exc: BaseException) -> str:
    """Actionable guidance wrapped around an OOM failure."""
    return (
        f"GPU out of memory: {type(exc).__name__}: {exc}\n"
        "remedies, in order:\n"
        "  1. free VRAM (close the other process using the GPU)\n"
        "  2. re-run with JAX_PLATFORMS=cpu (backend is fixed per process —\n"
        "     the fallback must be a fresh process)\n"
        "  3. smaller problem (n_cells / budget / batch)"
    )


def guard(callable_, *, log: "Any | None" = None, clear_on_error: bool = True):
    """Run ``callable_()``; on OOM clear caches and raise the actionable
    message. Non-OOM errors pass through untouched."""
    try:
        return callable_()
    except Exception as exc:  # noqa: BLE001
        if is_oom(exc):
            if clear_on_error:
                clear_caches()
            if log is not None:
                log(oom_message(exc))
            raise RuntimeError(oom_message(exc)) from exc
        raise


__all__ = ["clear_caches", "configure", "devices", "is_oom", "oom_message", "guard"]


if __name__ == "__main__":  # quick manual check: python -m harness.gpu
    configure()
    import jax  # noqa: E402  (deliberately after configure)

    print(f"jax {jax.__version__} · devices: {devices()}")
    x = __import__("jax.numpy", fromlist=["jnp"]).linspace(0.0, 1.0, 1_000_000)
    print(f"x64 float64 array on {x.devices()}: {float(x.sum()):.6f}")
    sys.exit(0)
