"""Name registries and the ``make()`` entry point (``DESIGN.md`` §7.1).

Five registries, all one shape: ``models``, ``envs``, ``backends``,
``materials``, ``profiles``. Built-ins register at import time; third-party
packages register at import time or advertise entry points in the
``harness.<kind>`` groups (loaded lazily on first miss). Unknown names fail
with the list of what is available.

The registries live in this neutral module — not in ``envs`` — so that
``materials`` and ``profiles`` can register without importing the env layer
(keeping the import direction rules in the root pyproject satisfiable).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Callable

_ENTRY_POINT_PREFIX = "harness"

_KINDS = ("models", "envs", "backends", "materials", "profiles")


@dataclass
class Registry:
    """One name → zero-argument-factory mapping for a plugin kind."""

    kind: str
    _factories: dict[str, Callable[[], Any]] = field(default_factory=dict)
    _entry_points_scanned: bool = False

    def register(self, name: str, factory: Callable[[], Any], *, overwrite: bool = False) -> Callable[[], Any]:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{self.kind}: registry names must be non-empty strings, got {name!r}")
        if name in self._factories and not overwrite:
            raise ValueError(
                f"{name!r} is already registered in harness.{self.kind}; "
                "pass overwrite=True to replace it"
            )
        self._factories[name] = factory
        return factory

    def names(self) -> tuple[str, ...]:
        self._scan_entry_points()
        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        return name in self._factories or name in self.names()

    def resolve(self, name: str) -> Callable[[], Any]:
        if name in self._factories:
            return self._factories[name]
        self._scan_entry_points()
        if name in self._factories:
            return self._factories[name]
        raise KeyError(
            f"unknown harness.{self.kind} name {name!r}; "
            f"available: {', '.join(self.names()) or '(none)'}"
        )

    def _scan_entry_points(self) -> None:
        if self._entry_points_scanned:
            return
        self._entry_points_scanned = True
        try:
            eps = entry_points(group=f"{_ENTRY_POINT_PREFIX}.{self.kind}")
        except Exception:  # metadata unavailable in exotic environments
            return
        for ep in eps:
            if ep.name in self._factories:
                continue
            try:
                self._factories[ep.name] = ep.load()
            except Exception as exc:  # a broken plugin must not break make()
                warnings.warn(
                    f"harness.{self.kind} entry point {ep.name!r} failed to "
                    f"load ({exc!r}); skipping",
                    stacklevel=2,
                )


REGISTRIES: dict[str, Registry] = {kind: Registry(kind) for kind in _KINDS}


def register(kind: str, name: str, factory: Callable[[], Any], *, overwrite: bool = False):
    return REGISTRIES[kind].register(name, factory, overwrite=overwrite)


def register_model(name: str, factory: Callable[[], Any], *, overwrite: bool = False):
    return register("models", name, factory, overwrite=overwrite)


def register_env(name: str, factory: Callable[..., Any], *, overwrite: bool = False):
    return register("envs", name, factory, overwrite=overwrite)


def register_backend(name: str, factory: Callable[[], Any], *, overwrite: bool = False):
    return register("backends", name, factory, overwrite=overwrite)


def register_material(name: str, factory: Callable[[], Any], *, overwrite: bool = False):
    return register("materials", name, factory, overwrite=overwrite)


def register_profile(name: str, factory: Callable[[], Any], *, overwrite: bool = False):
    return register("profiles", name, factory, overwrite=overwrite)


def make(name: str, **config) -> Any:
    """Build a registered environment (``DESIGN.md`` §3).

    ``material=`` / ``profile=`` values may be registry keys (e.g.
    ``"anchor:Silica gel RD"``, ``"datacenter"``) or already-built instances;
    resolution is the env factory's job.
    """
    factory = REGISTRIES["envs"].resolve(name)
    return factory(**config)
