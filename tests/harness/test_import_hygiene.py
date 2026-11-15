"""Import hygiene: the harness stays independent of the legacy/frozen
packages (import-linter covers internal layering; this covers modules that
are not importable packages, e.g. Materials/cooling_physics.py)."""

import ast
from pathlib import Path

import harness

FORBIDDEN_ROOTS = (
    "diffheat",
    "cooling_physics",
    "heat_cooling_screen",
    "matplotlib",
    "torch",
    "stable_baselines3",
)

# Per-file allowances: forbidden roots a named module may import. viz_pde.py
# runs the frozen diffheat PDE zoo **recovered from git history** into the
# gitignored data_cache/_frozen/ extract and imports the original solvers from
# there (recover-and-run, harness/viz_pde.py docstring) — never a tree
# dependency. Only "diffheat" is allowed for it; matplotlib/torch stay banned,
# and every other module keeps the full ban list.
ALLOWED_ROOTS: dict[str, set[str]] = {"viz_pde.py": {"diffheat"}}


def _import_roots(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                yield node.module


def test_no_legacy_or_heavy_imports_in_harness_source():
    package_dir = Path(harness.__file__).resolve().parent
    offenders = []
    for source_file in sorted(package_dir.rglob("*.py")):
        # GUI is UI-only and legitimately imports matplotlib (harness[gui] extra)
        if "gui" in source_file.parts:
            continue
        allowed = ALLOWED_ROOTS.get(source_file.name, set())
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for module in _import_roots(tree):
            if module.split(".")[0] in FORBIDDEN_ROOTS and module.split(".")[0] not in allowed:
                offenders.append(f"{source_file.name}: import of {module!r}")
    assert not offenders, f"forbidden imports found: {offenders}"


def test_public_api_importable_from_top_level():
    for name in harness.__all__:
        assert hasattr(harness, name), f"harness.__all__ member {name!r} missing"
