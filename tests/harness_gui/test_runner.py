"""Qt-free runner tests: catalog, evaluate, optimize, sweeps, script."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from harness.gui import runner


def test_env_catalog_is_registry_driven():
    names = runner.env_names()
    assert "Cycle0D-v0" in names and "Bed1D-v0" in names
    params = runner.env_factory_params("Bed1D-v0")
    assert "material" in params and "profile" in params
    assert "n_cells" in params and params["n_cells"]["kind"] == "int"
    backends = runner.backend_params()
    assert {"grad", "search", "rl"} <= set(backends)
    assert "budget" in backends["search"]


def test_evaluate_cycle0d():
    spec = runner.RunSpec(env_name="Cycle0D-v0", collect_trace=False)
    payload = runner.execute(spec, log=lambda _s: None)
    assert "COP" in payload.metrics
    assert payload.sweep_df is None
    compile(payload.python_code, "script", "exec")


def test_optimize_search_small_budget():
    spec = runner.RunSpec(env_name="Cycle0D-v0", mode="optimize",
                          backend="search", seed=0, collect_trace=False,
                          backend_kwargs={"budget": 8, "method": "cmaes"})
    payload = runner.execute(spec, log=lambda _s: None)
    assert payload.result is not None
    assert payload.result.n_evals >= 8 - 1  # budget respected approximately
    assert payload.trace is None  # collect_trace off


def test_t_switch_sweep_cancel_checks():
    spec = runner.RunSpec(env_name="Bed1D-v0", sweep_axis="t_switch",
                          sweep_min=120.0, sweep_max=360.0, sweep_steps=3,
                          env_kwargs={"n_cells": 4, "n_cycles": 1},
                          collect_trace=False)
    payload = runner.execute(spec, log=lambda _s: None)
    assert len(payload.sweep_df) == 3
    assert {"t_switch_s", "COP", "SCP_W_kg"} <= set(payload.sweep_df.columns)
    # cancelled-before-start raises loudly
    spec_cancel = runner.RunSpec(env_name="Bed1D-v0", sweep_axis="t_switch",
                                 sweep_steps=2, collect_trace=False)
    with pytest.raises(RuntimeError, match="cancelled"):
        runner.execute(spec_cancel, log=lambda _s: None,
                       is_cancelled=lambda: True)


def test_material_sweep_two_anchors():
    spec = runner.RunSpec(env_name="Cycle0D-v0", sweep_axis="material",
                          material_ref="anchor:Silica gel RD, anchor:Zeolite 13X (NaX)",
                          profile_ref="datacenter", collect_trace=False)
    payload = runner.execute(spec, log=lambda _s: None)
    assert len(payload.sweep_df) == 2
    assert "material" in payload.sweep_df.columns


def test_design_overrides_log_and_apply():
    logs: list[str] = []
    spec = runner.RunSpec(env_name="Cycle0D-v0",
                          design_overrides={"q_sat_kg_kg": 0.4},
                          collect_trace=False)
    payload = runner.execute(spec, log=logs.append)
    assert any("design defaults overridden" in line for line in logs)
    assert payload.metrics["COP"] == payload.metrics["COP"]
    with pytest.raises(KeyError):
        runner.execute(runner.RunSpec(env_name="Cycle0D-v0",
                                      design_overrides={"nope": 1.0}),
                       log=lambda _s: None)
