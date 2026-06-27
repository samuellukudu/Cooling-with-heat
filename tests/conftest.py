# tests/conftest.py
"""Shared test configuration."""
import pytest
import jax

# Enable float64 for JAX to prevent numerical precision issues during CPU tests
jax.config.update("jax_enable_x64", True)

_ROS_PLUGINS = (
    "ament_copyright",
    "ament_flake8",
    "ament_lint",
    "ament_pep257",
    "ament_xmllint",
    "launch_ros",
    "launch_testing",
)


def pytest_configure(config: pytest.Config) -> None:
    for name in _ROS_PLUGINS:
        if config.pluginmanager.has_plugin(name):
            config.pluginmanager.set_blocked(name)
