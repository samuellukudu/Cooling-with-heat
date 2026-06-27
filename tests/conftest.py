# tests/conftest.py
"""Shared test configuration.

ROS 2 packages installed system-wide leak pytest plugins into virtual
environments via PYTHONPATH.  We disable them here so that ``pytest`` works
regardless of whether the ROS setup script has been sourced.
"""

import pytest

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
