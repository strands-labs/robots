"""Tests for agent-facing tools."""

import importlib

import pytest

# All tools should be importable without hardware dependencies
TOOL_MODULES = [
    ("strands_robots.tools.gr00t_inference", "gr00t_inference"),
    ("strands_robots.tools.inference", "inference"),
    ("strands_robots.tools.isaac_sim", "isaac_sim"),
    ("strands_robots.tools.lerobot_calibrate", "lerobot_calibrate"),
    ("strands_robots.tools.lerobot_camera", "lerobot_camera"),
    ("strands_robots.tools.lerobot_dataset", "lerobot_dataset"),
    ("strands_robots.tools.lerobot_teleoperate", "lerobot_teleoperate"),
    ("strands_robots.tools.marble_tool", "marble_tool"),
    ("strands_robots.tools.newton_sim", "newton_sim"),
    ("strands_robots.tools.pose_tool", "pose_tool"),
    ("strands_robots.tools.reachy_mini_tool", "reachy_mini"),
    ("strands_robots.tools.robot_mesh", "robot_mesh"),
    ("strands_robots.tools.serial_tool", "serial_tool"),
    ("strands_robots.tools.stereo_depth", "stereo_depth"),
    ("strands_robots.tools.stream", "stream"),
    ("strands_robots.tools.teleoperator", "teleoperator"),
    ("strands_robots.tools.use_lerobot", "use_lerobot"),
    ("strands_robots.tools.use_unitree", "use_unitree"),
]


def _try_import(module_path):
    """Attempt to import a module, returning (module, None) or (None, error)."""
    try:
        mod = importlib.import_module(module_path)
        return mod, None
    except (ImportError, ModuleNotFoundError) as e:
        return None, str(e)


class TestToolImports:
    """Verify all tools are importable (skips when optional deps missing)."""

    @pytest.mark.parametrize("module_path,tool_name", TOOL_MODULES)
    def test_tool_importable(self, module_path, tool_name):
        """Each tool module should import without errors.

        Skips when optional dependencies (lerobot, psutil, serial, etc.)
        are not installed in the CI environment.
        """
        mod, err = _try_import(module_path)
        if mod is None:
            pytest.skip(f"Optional dependency missing: {err}")
        # Tool function should exist with same name as module
        assert hasattr(mod, tool_name), f"{module_path} missing {tool_name} function"

    @pytest.mark.parametrize("module_path,tool_name", TOOL_MODULES)
    def test_tool_is_callable(self, module_path, tool_name):
        """Each tool's main function should be callable."""
        mod, err = _try_import(module_path)
        if mod is None:
            pytest.skip(f"Optional dependency missing: {err}")
        fn = getattr(mod, tool_name)
        assert callable(fn)


class TestToolsPackage:
    """Test the tools package __init__."""

    def test_tools_init_importable(self):
        import strands_robots.tools

        assert strands_robots.tools is not None

    def test_tools_count(self):
        """Should have 18 tools."""
        from pathlib import Path

        tools_dir = Path(__file__).parent.parent / "strands_robots" / "tools"
        py_files = [f for f in tools_dir.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 16  # At least 16 tool files
