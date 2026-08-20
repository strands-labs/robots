"""Behavior tests for the ``strands_robots.tools`` lazy-import contract.

The tools package defers importing each tool module until first attribute
access so that ``import strands_robots.tools`` never pulls in numpy, pyserial,
psutil, and the other heavy per-tool dependencies. These tests pin that public
contract:

- Every name advertised in ``__all__`` is lazily importable as a real attribute.
- The first access materializes the tool via ``__getattr__`` and caches it into
  the package namespace, so a repeat access returns the identical object.
- An unknown attribute raises ``AttributeError`` naming the package and the
  missing attribute, matching the standard module-level ``__getattr__`` protocol.
"""

from __future__ import annotations

import importlib

import pytest

import strands_robots.tools as tools_pkg


def test_all_lists_every_lazy_import_name() -> None:
    """``__all__`` advertises exactly the lazily-importable tool names."""
    assert set(tools_pkg.__all__) == set(tools_pkg._LAZY_IMPORTS)
    # The advertised fleet of tools, so a silent drop is caught here.
    assert set(tools_pkg.__all__) == {
        "create_judge_agent",
        "download_assets",
        "gr00t_inference",
        "harness_memory",
        "lerobot_calibrate",
        "lerobot_camera",
        "lerobot_teleoperate",
        "lerobot_train",
        "load_episode",
        "pose_tool",
        "read_predicate_verdict",
        "robot_mesh",
        "run_policy",
        "sample_frames",
        "serial_tool",
        "train_policy",
        "use_lerobot",
        "use_ros",
        "use_rosbridge",
        "use_rtps",
        "write_label",
    }


@pytest.mark.parametrize("name", sorted(tools_pkg._LAZY_IMPORTS))
def test_each_tool_is_lazily_importable(name: str) -> None:
    """Each advertised name resolves to the matching object via ``__getattr__``.

    The package caches resolved tools into its namespace, and a prior test (or
    import) may already have triggered that cache, so first drop any cached
    binding to force a fresh ``__getattr__`` resolution. Then assert the access
    materializes the documented target and re-caches it for subsequent reads.
    """
    # Force the next access to go through __getattr__ rather than a cached glob.
    vars(tools_pkg).pop(name, None)
    assert name not in vars(tools_pkg)

    value = getattr(tools_pkg, name)
    assert value is not None

    # The lazy target maps to the documented (relative module, attribute) pair.
    rel_module, attr_name = tools_pkg._LAZY_IMPORTS[name]
    submodule = importlib.import_module(rel_module, tools_pkg.__name__)
    assert value is getattr(submodule, attr_name)

    # First access caches into the package namespace; second access is identical.
    assert name in vars(tools_pkg)
    assert getattr(tools_pkg, name) is value


def test_unknown_attribute_raises_attribute_error() -> None:
    """An unknown attribute raises ``AttributeError`` naming package + attr."""
    with pytest.raises(AttributeError) as excinfo:
        tools_pkg.definitely_not_a_tool  # noqa: B018 - trigger __getattr__

    message = str(excinfo.value)
    assert "strands_robots.tools" in message
    assert "definitely_not_a_tool" in message
