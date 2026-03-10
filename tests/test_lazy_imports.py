"""Tests for lazy import patterns across all strands_robots subpackages.

Covers:
1. strands_robots.tools.__init__.__getattr__  — lazy tool loading
2. strands_robots.newton.__init__.__getattr__ — lazy Newton backend loading
3. strands_robots.isaac.__init__.__getattr__  — lazy Isaac Sim loading
4. strands_robots.__init__ export verification

Note: OpenCV 4.12 has cv2.dnn.DictValue / cv2.mat_wrapper bugs that trigger
on reimport. We pre-mock cv2 at module scope before any strands_robots imports.
"""

# ─────────────────────────────────────────────────────────────────
# Pre-mock cv2 at module scope BEFORE any strands_robots imports
# This prevents OpenCV 4.12 circular import crash (cv2.mat_wrapper)
# ─────────────────────────────────────────────────────────────────
import importlib.machinery as _im_fix
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

_mock_cv2 = MagicMock()
_mock_cv2.__spec__ = _im_fix.ModuleSpec("cv2", None)
_mock_cv2.__version__ = "4.12.0"
_mock_cv2.dnn = MagicMock()
_mock_cv2.dnn.DictValue = MagicMock()
_mock_cv2.typing = MagicMock()
_mock_cv2.mat_wrapper = MagicMock()

# Install cv2 mock before any strands_robots module touches it
for _key in list(sys.modules.keys()):
    if _key.startswith("cv2"):
        sys.modules.pop(_key, None)
sys.modules["cv2"] = _mock_cv2
sys.modules["cv2.dnn"] = _mock_cv2.dnn
sys.modules["cv2.typing"] = _mock_cv2.typing
sys.modules["cv2.mat_wrapper"] = _mock_cv2.mat_wrapper

# Now it's safe to import strands_robots packages
import strands_robots  # noqa: E402
import strands_robots.isaac  # noqa: E402
import strands_robots.newton  # noqa: E402
import strands_robots.tools  # noqa: E402

# Grab __getattr__ references ONCE
_tools_getattr = strands_robots.tools.__getattr__
_newton_getattr = strands_robots.newton.__getattr__
_isaac_getattr = strands_robots.isaac.__getattr__


# ─────────────────────────────────────────────────────────────────
# 1. strands_robots.tools.__init__.__getattr__
# ─────────────────────────────────────────────────────────────────


class TestToolsLazyImports:
    """Test the tools package __getattr__ lazy loading."""

    def test_getattr_known_tool_returns_attr(self):
        """__getattr__ should import and return the tool for known names."""
        mock_mod = types.ModuleType("strands_robots.tools.newton_sim")
        mock_tool = MagicMock(name="newton_sim_tool")
        mock_mod.newton_sim = mock_tool

        with patch.dict("sys.modules", {"strands_robots.tools.newton_sim": mock_mod}):
            result = _tools_getattr("newton_sim")
            assert result is mock_tool

    def test_getattr_unknown_raises_attribute_error(self):
        """__getattr__ should raise AttributeError for unknown names."""
        with pytest.raises(AttributeError, match="has no attribute"):
            _tools_getattr("nonexistent_tool_xyz")

    def test_getattr_caches_in_globals(self):
        """After first access, the tool should be cached in module globals."""
        mock_mod = types.ModuleType("strands_robots.tools.inference")
        mock_tool = MagicMock(name="inference_tool")
        mock_mod.inference = mock_tool

        pkg = strands_robots.tools
        pkg.__dict__.pop("inference", None)

        with patch.dict("sys.modules", {"strands_robots.tools.inference": mock_mod}):
            result = _tools_getattr("inference")
            assert result is mock_tool
            assert pkg.__dict__.get("inference") is mock_tool

    def test_all_lazy_imports_listed(self):
        """Every tool in _LAZY_IMPORTS should be in __all__."""
        from strands_robots.tools import _LAZY_IMPORTS, __all__

        for name in _LAZY_IMPORTS:
            assert name in __all__, f"{name} in _LAZY_IMPORTS but not in __all__"

    def test_lazy_imports_dict_not_empty(self):
        """_LAZY_IMPORTS should contain the expected tools."""
        from strands_robots.tools import _LAZY_IMPORTS

        expected = {
            "gr00t_inference",
            "inference",
            "lerobot_camera",
            "pose_tool",
            "serial_tool",
            "teleoperator",
            "lerobot_dataset",
            "robot_mesh",
            "reachy_mini",
            "newton_sim",
            "isaac_sim",
            "marble_tool",
            "lerobot_teleoperate",
            "lerobot_calibrate",
        }
        assert expected.issubset(set(_LAZY_IMPORTS.keys()))


# ─────────────────────────────────────────────────────────────────
# 2. strands_robots.newton.__init__.__getattr__
# ─────────────────────────────────────────────────────────────────


class TestNewtonLazyImports:
    """Test the newton package __getattr__ lazy loading."""

    def test_newton_backend_lazy_import(self):
        mock_backend = MagicMock(name="NewtonBackend")
        mock_mod = types.ModuleType("strands_robots.newton.newton_backend")
        mock_mod.NewtonBackend = mock_backend

        with patch.dict("sys.modules", {"strands_robots.newton.newton_backend": mock_mod}):
            result = _newton_getattr("NewtonBackend")
            assert result is mock_backend

    def test_newton_config_lazy_import(self):
        mock_config = MagicMock(name="NewtonConfig")
        mock_mod = types.ModuleType("strands_robots.newton.newton_backend")
        mock_mod.NewtonConfig = mock_config

        with patch.dict("sys.modules", {"strands_robots.newton.newton_backend": mock_mod}):
            result = _newton_getattr("NewtonConfig")
            assert result is mock_config

    def test_newton_gym_env_lazy_import(self):
        mock_env = MagicMock(name="NewtonGymEnv")
        mock_mod = types.ModuleType("strands_robots.newton.newton_gym_env")
        mock_mod.NewtonGymEnv = mock_env

        with patch.dict("sys.modules", {"strands_robots.newton.newton_gym_env": mock_mod}):
            result = _newton_getattr("NewtonGymEnv")
            assert result is mock_env

    def test_solver_map_lazy_import(self):
        mock_map = {"mujoco": MagicMock()}
        mock_mod = types.ModuleType("strands_robots.newton.newton_backend")
        mock_mod.SOLVER_MAP = mock_map

        with patch.dict("sys.modules", {"strands_robots.newton.newton_backend": mock_mod}):
            result = _newton_getattr("SOLVER_MAP")
            assert result is mock_map

    def test_render_backends_lazy_import(self):
        mock_backends = ["opengl", "vulkan"]
        mock_mod = types.ModuleType("strands_robots.newton.newton_backend")
        mock_mod.RENDER_BACKENDS = mock_backends

        with patch.dict("sys.modules", {"strands_robots.newton.newton_backend": mock_mod}):
            result = _newton_getattr("RENDER_BACKENDS")
            assert result is mock_backends

    def test_broad_phase_options_lazy_import(self):
        mock_opts = ["sap", "bvh"]
        mock_mod = types.ModuleType("strands_robots.newton.newton_backend")
        mock_mod.BROAD_PHASE_OPTIONS = mock_opts

        with patch.dict("sys.modules", {"strands_robots.newton.newton_backend": mock_mod}):
            result = _newton_getattr("BROAD_PHASE_OPTIONS")
            assert result is mock_opts

    def test_newton_unknown_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _newton_getattr("nonexistent_solver_xyz")

    def test_newton_all_exports(self):
        expected = {
            "NewtonBackend",
            "NewtonConfig",
            "NewtonGymEnv",
            "SOLVER_MAP",
            "RENDER_BACKENDS",
            "BROAD_PHASE_OPTIONS",
        }
        assert expected == set(strands_robots.newton.__all__)


# ─────────────────────────────────────────────────────────────────
# 3. strands_robots.isaac.__init__.__getattr__
# ─────────────────────────────────────────────────────────────────


class TestIsaacLazyImports:
    """Test the isaac package __getattr__ lazy loading."""

    def test_isaac_sim_backend_lazy_import(self):
        mock_backend = MagicMock(name="IsaacSimBackend")
        mock_mod = types.ModuleType("strands_robots.isaac.isaac_sim_backend")
        mock_mod.IsaacSimBackend = mock_backend

        with patch.dict("sys.modules", {"strands_robots.isaac.isaac_sim_backend": mock_mod}):
            result = _isaac_getattr("IsaacSimBackend")
            assert result is mock_backend

    def test_isaac_gym_env_lazy_import(self):
        mock_env = MagicMock(name="IsaacGymEnv")
        mock_mod = types.ModuleType("strands_robots.isaac.isaac_gym_env")
        mock_mod.IsaacGymEnv = mock_env

        with patch.dict("sys.modules", {"strands_robots.isaac.isaac_gym_env": mock_mod}):
            result = _isaac_getattr("IsaacGymEnv")
            assert result is mock_env

    def test_isaac_lab_env_lazy_import(self):
        mock_env = MagicMock(name="IsaacLabEnv")
        mock_mod = types.ModuleType("strands_robots.isaac.isaac_lab_env")
        mock_mod.IsaacLabEnv = mock_env

        with patch.dict("sys.modules", {"strands_robots.isaac.isaac_lab_env": mock_mod}):
            result = _isaac_getattr("IsaacLabEnv")
            assert result is mock_env

    def test_isaac_lab_trainer_lazy_import(self):
        mock_trainer = MagicMock(name="IsaacLabTrainer")
        mock_config = MagicMock(name="IsaacLabTrainerConfig")
        mock_mod = types.ModuleType("strands_robots.isaac.isaac_lab_trainer")
        mock_mod.IsaacLabTrainer = mock_trainer
        mock_mod.IsaacLabTrainerConfig = mock_config

        with patch.dict("sys.modules", {"strands_robots.isaac.isaac_lab_trainer": mock_mod}):
            result = _isaac_getattr("IsaacLabTrainer")
            assert result is mock_trainer

    def test_isaac_lab_trainer_config_lazy_import(self):
        mock_trainer = MagicMock(name="IsaacLabTrainer")
        mock_config = MagicMock(name="IsaacLabTrainerConfig")
        mock_mod = types.ModuleType("strands_robots.isaac.isaac_lab_trainer")
        mock_mod.IsaacLabTrainer = mock_trainer
        mock_mod.IsaacLabTrainerConfig = mock_config

        with patch.dict("sys.modules", {"strands_robots.isaac.isaac_lab_trainer": mock_mod}):
            result = _isaac_getattr("IsaacLabTrainerConfig")
            assert result is mock_config

    def test_asset_converter_lazy_import(self):
        mock_converter = MagicMock(name="AssetConverter")
        mock_mod = types.ModuleType("strands_robots.isaac.asset_converter")
        mock_mod.AssetConverter = mock_converter

        with patch.dict("sys.modules", {"strands_robots.isaac.asset_converter": mock_mod}):
            result = _isaac_getattr("AssetConverter")
            assert result is mock_converter

    def test_create_isaac_env_lazy_import(self):
        mock_fn = MagicMock(name="create_isaac_env")
        mock_mod = types.ModuleType("strands_robots.isaac.isaac_lab_env")
        mock_mod.create_isaac_env = mock_fn

        with patch.dict("sys.modules", {"strands_robots.isaac.isaac_lab_env": mock_mod}):
            result = _isaac_getattr("create_isaac_env")
            assert result is mock_fn

    def test_list_isaac_tasks_lazy_import(self):
        mock_fn = MagicMock(name="list_isaac_tasks")
        mock_mod = types.ModuleType("strands_robots.isaac.isaac_lab_env")
        mock_mod.list_isaac_tasks = mock_fn

        with patch.dict("sys.modules", {"strands_robots.isaac.isaac_lab_env": mock_mod}):
            result = _isaac_getattr("list_isaac_tasks")
            assert result is mock_fn

    def test_convert_mjcf_to_usd_lazy_import(self):
        mock_fn = MagicMock(name="convert_mjcf_to_usd")
        mock_mod = types.ModuleType("strands_robots.isaac.asset_converter")
        mock_mod.convert_mjcf_to_usd = mock_fn
        mock_mod.convert_usd_to_mjcf = MagicMock()
        mock_mod.convert_all_robots_to_usd = MagicMock()

        with patch.dict("sys.modules", {"strands_robots.isaac.asset_converter": mock_mod}):
            result = _isaac_getattr("convert_mjcf_to_usd")
            assert result is mock_fn

    def test_convert_usd_to_mjcf_lazy_import(self):
        mock_fn = MagicMock(name="convert_usd_to_mjcf")
        mock_mod = types.ModuleType("strands_robots.isaac.asset_converter")
        mock_mod.convert_mjcf_to_usd = MagicMock()
        mock_mod.convert_usd_to_mjcf = mock_fn
        mock_mod.convert_all_robots_to_usd = MagicMock()

        with patch.dict("sys.modules", {"strands_robots.isaac.asset_converter": mock_mod}):
            result = _isaac_getattr("convert_usd_to_mjcf")
            assert result is mock_fn

    def test_convert_all_robots_to_usd_lazy_import(self):
        mock_fn = MagicMock(name="convert_all_robots_to_usd")
        mock_mod = types.ModuleType("strands_robots.isaac.asset_converter")
        mock_mod.convert_mjcf_to_usd = MagicMock()
        mock_mod.convert_usd_to_mjcf = MagicMock()
        mock_mod.convert_all_robots_to_usd = mock_fn

        with patch.dict("sys.modules", {"strands_robots.isaac.asset_converter": mock_mod}):
            result = _isaac_getattr("convert_all_robots_to_usd")
            assert result is mock_fn

    def test_isaac_unknown_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            _isaac_getattr("nonexistent_isaac_xyz")

    def test_get_isaac_sim_path_no_install(self):
        from strands_robots.isaac import get_isaac_sim_path

        result = get_isaac_sim_path()
        assert result is None or isinstance(result, str)

    def test_is_isaac_sim_available(self):
        from strands_robots.isaac import is_isaac_sim_available

        result = is_isaac_sim_available()
        assert isinstance(result, bool)

    def test_isaac_all_exports(self):
        expected = {
            "IsaacSimBackend",
            "IsaacSimBridgeClient",
            "IsaacSimBridgeServer",
            "IsaacGymEnv",
            "IsaacLabEnv",
            "IsaacLabTrainer",
            "IsaacLabTrainerConfig",
            "AssetConverter",
            "create_isaac_env",
            "list_isaac_tasks",
            "convert_mjcf_to_usd",
            "convert_usd_to_mjcf",
            "convert_all_robots_to_usd",
            "get_isaac_sim_path",
            "is_isaac_sim_available",
        }
        assert expected == set(strands_robots.isaac.__all__)


# ─────────────────────────────────────────────────────────────────
# 4. strands_robots.__init__ export verification
# ─────────────────────────────────────────────────────────────────


class TestRootInitStructure:
    """Verify strands_robots.__init__ exports are correct."""

    def test_core_exports_always_present(self):
        core = {
            "Robot",
            "Policy",
            "MockPolicy",
            "create_policy",
            "register_policy",
            "list_providers",
            "list_robots",
            "resolve_policy",
        }
        for name in core:
            assert name in strands_robots.__all__, f"{name} missing from __all__"
            assert hasattr(strands_robots, name), f"{name} not accessible"

    def test_simulation_export_when_available(self):
        if hasattr(strands_robots, "Simulation"):
            assert "Simulation" in strands_robots.__all__

    def test_hardware_robot_export(self):
        if hasattr(strands_robots, "HardwareRobot"):
            assert "HardwareRobot" in strands_robots.__all__

    def test_training_exports(self):
        if hasattr(strands_robots, "Trainer"):
            assert "Trainer" in strands_robots.__all__
            assert "TrainConfig" in strands_robots.__all__
            assert "create_trainer" in strands_robots.__all__

    def test_video_exports(self):
        if hasattr(strands_robots, "VideoEncoder"):
            assert "VideoEncoder" in strands_robots.__all__
            assert "encode_frames" in strands_robots.__all__

    def test_processor_exports(self):
        if hasattr(strands_robots, "ProcessorBridge"):
            assert "ProcessorBridge" in strands_robots.__all__
            assert "ProcessedPolicy" in strands_robots.__all__

    def test_kinematics_exports(self):
        if hasattr(strands_robots, "Kinematics"):
            assert "Kinematics" in strands_robots.__all__
            assert "create_kinematics" in strands_robots.__all__

    def test_envs_export(self):
        if hasattr(strands_robots, "StrandsSimEnv"):
            assert "StrandsSimEnv" in strands_robots.__all__

    def test_record_exports(self):
        if hasattr(strands_robots, "RecordSession"):
            assert "RecordSession" in strands_robots.__all__
            assert "RecordMode" in strands_robots.__all__
            assert "EpisodeStats" in strands_robots.__all__

    def test_visualizer_exports(self):
        if hasattr(strands_robots, "RecordingVisualizer"):
            assert "RecordingVisualizer" in strands_robots.__all__

    def test_motion_library_exports(self):
        if hasattr(strands_robots, "MotionLibrary"):
            assert "MotionLibrary" in strands_robots.__all__
            assert "Motion" in strands_robots.__all__

    def test_rl_trainer_exports(self):
        if hasattr(strands_robots, "SB3Trainer"):
            assert "SB3Trainer" in strands_robots.__all__
            assert "RLConfig" in strands_robots.__all__
            assert "create_rl_trainer" in strands_robots.__all__

    def test_robometer_exports(self):
        if hasattr(strands_robots, "StrandsRobometerRewardWrapper"):
            assert "StrandsRobometerRewardWrapper" in strands_robots.__all__
            assert "robometer_reward_fn" in strands_robots.__all__

    def test_leisaac_exports(self):
        if hasattr(strands_robots, "LeIsaacEnv"):
            assert "LeIsaacEnv" in strands_robots.__all__
            assert "create_leisaac_env" in strands_robots.__all__

    def test_marble_exports(self):
        if hasattr(strands_robots, "MarblePipeline"):
            assert "MarblePipeline" in strands_robots.__all__
            assert "MarbleConfig" in strands_robots.__all__

    def test_dataset_recorder_export(self):
        if hasattr(strands_robots, "DatasetRecorder"):
            assert "DatasetRecorder" in strands_robots.__all__

    def test_dreamgen_exports(self):
        if hasattr(strands_robots, "DreamGenPipeline"):
            assert "DreamGenPipeline" in strands_robots.__all__
            assert "DreamGenConfig" in strands_robots.__all__

    def test_zenoh_mesh_exports(self):
        if "Mesh" in strands_robots.__all__:
            assert hasattr(strands_robots, "Mesh")
            assert "get_peers" in strands_robots.__all__

    def test_init_source_has_try_except_blocks(self):
        import inspect

        source = inspect.getsource(strands_robots)
        assert source.count("except ImportError") >= 15
        assert source.count("try:") >= 15

    def test_no_duplicate_all_entries(self):
        seen = set()
        for name in strands_robots.__all__:
            assert name not in seen, f"Duplicate in __all__: {name}"
            seen.add(name)

    def test_all_entries_are_accessible(self):
        """Every item in __all__ should be accessible as an attribute."""
        for name in strands_robots.__all__:
            assert hasattr(strands_robots, name), f"{name} in __all__ but not accessible via hasattr"
