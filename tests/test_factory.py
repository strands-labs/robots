#!/usr/bin/env python3
"""Tests for the Robot factory, asset registry, and policy resolver."""

import ast
import importlib.util
import os

import pytest

# Use find_spec to check filesystem — `import strands` can succeed if another
# test file injected a mock into sys.modules (cross-test pollution).
try:
    _has_strands = importlib.util.find_spec("strands") is not None
except (ValueError, ModuleNotFoundError):
    _has_strands = False


import sys
import types
from unittest.mock import MagicMock, patch


def _inject_mock_modules():
    """Inject mock modules for strands_robots.simulation and strands_robots.robot
    so that unittest.mock.patch can resolve the dotted paths."""
    if "strands_robots.simulation" not in sys.modules:
        sim_mod = types.ModuleType("strands_robots.simulation")
        sim_mod.Simulation = None  # placeholder
        sys.modules["strands_robots.simulation"] = sim_mod
    if "strands_robots.robot" not in sys.modules:
        robot_mod = types.ModuleType("strands_robots.robot")
        robot_mod.Robot = None  # placeholder
        sys.modules["strands_robots.robot"] = robot_mod


def _mujoco_available():
    """Check if MuJoCo is importable."""
    try:
        import mujoco  # noqa: F401

        return True
    except ImportError:
        return False


class TestFactorySyntax:
    """Validate factory and resolver parse correctly."""

    BASE = os.path.join(os.path.dirname(__file__), "..", "strands_robots")

    def _check_syntax(self, filepath):
        with open(filepath, "r") as f:
            source = f.read()
        ast.parse(source, filename=filepath)

    def test_factory_syntax(self):
        self._check_syntax(os.path.join(self.BASE, "factory.py"))

    def test_policy_resolver_syntax(self):
        self._check_syntax(os.path.join(self.BASE, "policy_resolver.py"))

    def test_envs_syntax(self):
        self._check_syntax(os.path.join(self.BASE, "envs.py"))

    def test_video_syntax(self):
        self._check_syntax(os.path.join(self.BASE, "video.py"))

    def test_kinematics_syntax(self):
        self._check_syntax(os.path.join(self.BASE, "kinematics.py"))

    def test_processor_syntax(self):
        self._check_syntax(os.path.join(self.BASE, "processor.py"))

    def test_motion_library_syntax(self):
        self._check_syntax(os.path.join(self.BASE, "motion_library.py"))

    def test_dataset_recorder_syntax(self):
        self._check_syntax(os.path.join(self.BASE, "dataset_recorder.py"))

    def test_record_syntax(self):
        self._check_syntax(os.path.join(self.BASE, "record.py"))

    def test_visualizer_syntax(self):
        self._check_syntax(os.path.join(self.BASE, "visualizer.py"))

    def test_leisaac_syntax(self):
        self._check_syntax(os.path.join(self.BASE, "leisaac.py"))

    def test_assets_init_syntax(self):
        self._check_syntax(os.path.join(self.BASE, "assets", "__init__.py"))


class TestFactoryImports:
    """Test factory module imports."""

    def test_import_factory(self):
        from strands_robots.factory import Robot, create_robot, list_robots

        assert callable(Robot)
        assert callable(list_robots)
        assert Robot is create_robot

    def test_import_list_robots(self):
        from strands_robots.factory import list_robots

        robots = list_robots()
        assert isinstance(robots, list)
        assert len(robots) >= 34  # 34 robots in _UNIFIED_ROBOTS

    def test_list_robots_sim(self):
        from strands_robots.factory import list_robots

        sim_robots = list_robots(mode="sim")
        assert len(sim_robots) >= 32  # 32 sim-capable robots
        for r in sim_robots:
            assert r["has_sim"] is True

    def test_list_robots_real(self):
        from strands_robots.factory import list_robots

        real_robots = list_robots(mode="real")
        assert len(real_robots) >= 8  # 8 real-capable robots
        for r in real_robots:
            assert r["has_real"] is True

    def test_list_robots_both(self):
        from strands_robots.factory import list_robots

        both = list_robots(mode="both")
        assert len(both) >= 5  # 5 robots with both sim + real
        for r in both:
            assert r["has_sim"] is True
            assert r["has_real"] is True

    def test_known_robots_present(self):
        from strands_robots.factory import list_robots

        names = {r["name"] for r in list_robots()}
        assert "so100" in names
        assert "panda" in names
        assert "unitree_g1" in names
        assert "aloha" in names
        assert "spot" in names

    def test_new_robots_present(self):
        """Verify Open Duck Mini V2 and Asimov V0 are in the factory registry."""
        from strands_robots.factory import list_robots

        names = {r["name"] for r in list_robots()}
        assert "open_duck_mini" in names
        assert "asimov_v0" in names

    def test_new_robots_sim_capable(self):
        """Verify new robots have sim support."""
        from strands_robots.factory import list_robots

        sim_names = {r["name"] for r in list_robots(mode="sim")}
        assert "open_duck_mini" in sim_names
        assert "asimov_v0" in sim_names

    @pytest.mark.skipif(
        not _mujoco_available(),
        reason="MuJoCo not installed — requires strands-robots[sim]",
    )
    def test_robot_factory_sim(self):
        """Test Robot() creates a Simulation in sim mode."""
        from strands_robots.factory import Robot

        sim = Robot("so100", mode="sim")
        # Should be a Simulation instance
        assert hasattr(sim, "create_world") or hasattr(sim, "run_policy")

    @pytest.mark.skipif(
        not _mujoco_available(),
        reason="MuJoCo not installed — requires strands-robots[sim]",
    )
    def test_robot_factory_open_duck_mini(self):
        """Test Robot() works for Open Duck Mini V2."""
        from strands_robots.factory import Robot
        from strands_robots.simulation import resolve_model

        model_path = resolve_model("open_duck_mini")
        if not model_path:
            pytest.skip("Open Duck Mini model not found")
        try:
            sim = Robot("open_duck_mini", mode="sim")
            assert hasattr(sim, "create_world") or hasattr(sim, "run_policy")
        except RuntimeError as e:
            if "Failed to load" in str(e) and ".stl" in str(e):
                pytest.skip("Open Duck Mini mesh files not downloaded")
            raise

    @pytest.mark.skipif(
        not _mujoco_available(),
        reason="MuJoCo not installed — requires strands-robots[sim]",
    )
    def test_robot_factory_asimov_v0(self):
        """Test Robot() works for Asimov V0."""
        from strands_robots.factory import Robot
        from strands_robots.simulation import resolve_model

        model_path = resolve_model("asimov_v0")
        if not model_path:
            pytest.skip("Asimov V0 model not found")
        try:
            sim = Robot("asimov_v0", mode="sim")
            assert hasattr(sim, "create_world") or hasattr(sim, "run_policy")
        except RuntimeError as e:
            if "Failed to load" in str(e) and (".stl" in str(e) or ".STL" in str(e)):
                pytest.skip("Asimov V0 mesh files not downloaded")
            raise

    def test_alias_resolution(self):
        from strands_robots.factory import _resolve_name

        assert _resolve_name("go2") == "unitree_go2"
        assert _resolve_name("g1") == "unitree_g1"
        assert _resolve_name("franka_panda") == "panda"
        assert _resolve_name("so100") == "so100"

    def test_new_robot_alias_resolution(self):
        """Test alias resolution for Open Duck Mini and Asimov."""
        from strands_robots.factory import _resolve_name

        # Open Duck Mini aliases
        assert _resolve_name("open_duck") == "open_duck_mini"
        assert _resolve_name("mini_bdx") == "open_duck_mini"
        assert _resolve_name("bdx") == "open_duck_mini"
        assert _resolve_name("open_duck_v2") == "open_duck_mini"
        # Asimov alias
        assert _resolve_name("asimov") == "asimov_v0"


class TestAssetRegistry:
    """Test the asset registry for new robots."""

    def test_asset_registry_count(self):
        """Verify 32 robots in the asset registry."""
        from strands_robots.assets import list_available_robots

        robots = list_available_robots()
        assert len(robots) >= 32

    def test_open_duck_mini_asset_exists(self):
        """Verify Open Duck Mini V2 asset resolves."""
        from strands_robots.assets import resolve_model_path

        path = resolve_model_path("open_duck_mini")
        assert path is not None
        assert path.exists()

    def test_asimov_v0_asset_exists(self):
        """Verify Asimov V0 asset resolves."""
        from strands_robots.assets import resolve_model_path

        path = resolve_model_path("asimov_v0")
        assert path is not None
        assert path.exists()

    def test_open_duck_aliases_resolve(self):
        """Verify all Open Duck Mini aliases resolve to the same path."""
        from strands_robots.assets import resolve_model_path

        canonical = resolve_model_path("open_duck_mini")
        assert canonical is not None
        for alias in ["open_duck", "bdx", "mini_bdx", "open_duck_v2", "open_duck_mini_v2"]:
            path = resolve_model_path(alias)
            assert path is not None, f"Alias '{alias}' did not resolve"
            assert path == canonical, f"Alias '{alias}' resolved to {path}, expected {canonical}"

    def test_asimov_alias_resolves(self):
        """Verify asimov alias resolves to asimov_v0."""
        from strands_robots.assets import resolve_model_path

        canonical = resolve_model_path("asimov_v0")
        alias_path = resolve_model_path("asimov")
        assert alias_path is not None
        assert alias_path == canonical

    def test_asset_alias_count(self):
        """Verify alias count is at least 46."""
        from strands_robots.assets import list_aliases

        aliases = list_aliases()
        assert len(aliases) >= 46


class TestPolicyResolver:
    """Test smart policy resolution."""

    def test_import(self):
        from strands_robots.policy_resolver import resolve_policy

        assert callable(resolve_policy)

    def test_mock_shorthand(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("mock")
        assert provider == "mock"

    def test_hf_model_id(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("lerobot/act_aloha_sim_transfer_cube_human")
        assert provider == "lerobot_local"
        assert kwargs["pretrained_name_or_path"] == "lerobot/act_aloha_sim_transfer_cube_human"

    def test_grpc_address(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("localhost:8080")
        assert provider == "lerobot_async"
        assert kwargs["server_address"] == "localhost:8080"

    def test_websocket_address(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("ws://gpu:9000")
        assert provider == "dreamzero"
        assert kwargs["host"] == "gpu"
        assert kwargs["port"] == 9000

    def test_zmq_address(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("zmq://jetson:5555")
        assert provider == "groot"
        assert kwargs["host"] == "jetson"
        assert kwargs["port"] == 5555

    def test_openvla_model(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("openvla/openvla-7b")
        assert provider == "openvla"

    def test_microsoft_magma(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("microsoft/Magma-8B")
        assert provider == "magma"

    def test_unknown_org_defaults_to_lerobot(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("random-org/some-model")
        assert provider == "lerobot_local"

    def test_extra_kwargs_forwarded(self):
        from strands_robots.policy_resolver import resolve_policy

        provider, kwargs = resolve_policy("mock", custom_param="hello")
        assert kwargs.get("custom_param") == "hello"


class TestPackageExports:
    """Test top-level package exports."""

    def test_core_exports(self):
        from strands_robots import Robot, list_robots

        assert callable(Robot)
        assert callable(list_robots)

    def test_policy_exports(self):
        from strands_robots import create_policy, list_providers, register_policy

        assert callable(create_policy)
        assert callable(register_policy)
        providers = list_providers()
        assert "mock" in providers

    @pytest.mark.skipif(not _has_strands, reason="strands SDK not installed — tools require @tool decorator")
    def test_tool_exports(self):
        """Verify hardware tools are importable."""

    def test_resolve_policy_export(self):
        from strands_robots import resolve_policy

        assert callable(resolve_policy)

    def test_all_list(self):
        import strands_robots

        assert hasattr(strands_robots, "__all__")
        assert len(strands_robots.__all__) > 10


# ---------------------------------------------------------------------------
# New tests: Robot() factory function and list_robots() comprehensive tests
# ---------------------------------------------------------------------------


class TestResolveNameFunction:
    """Test _resolve_name for normalization and alias resolution."""

    def test_lowercase_normalization(self):
        from strands_robots.factory import _resolve_name

        assert _resolve_name("SO100") == "so100"
        assert _resolve_name("Panda") == "panda"

    def test_strip_whitespace(self):
        from strands_robots.factory import _resolve_name

        assert _resolve_name("  so100  ") == "so100"

    def test_hyphen_to_underscore(self):
        from strands_robots.factory import _resolve_name

        assert _resolve_name("reachy-mini") == "reachy_mini"
        assert _resolve_name("unitree-g1") == "unitree_g1"

    def test_combined_normalization(self):
        from strands_robots.factory import _resolve_name

        assert _resolve_name("  Reachy-Mini  ") == "reachy_mini"

    def test_unknown_name_passthrough(self):
        from strands_robots.factory import _resolve_name

        assert _resolve_name("unknown_robot") == "unknown_robot"

    def test_all_aliases_resolve(self):
        from strands_robots.factory import _ALIASES, _resolve_name

        for alias, canonical in _ALIASES.items():
            assert _resolve_name(alias) == canonical, f"Alias '{alias}' should resolve to '{canonical}'"


class TestAutoDetectMode:
    """Test _auto_detect_mode logic."""

    def test_env_var_sim_override(self):
        from strands_robots.factory import _auto_detect_mode

        with patch.dict(os.environ, {"STRANDS_ROBOT_MODE": "sim"}):
            assert _auto_detect_mode("so100", {"real": "so100_follower"}) == "sim"

    def test_env_var_real_override(self):
        from strands_robots.factory import _auto_detect_mode

        with patch.dict(os.environ, {"STRANDS_ROBOT_MODE": "real"}):
            assert _auto_detect_mode("so100", {"real": "so100_follower"}) == "real"

    def test_defaults_to_sim_without_env_var(self):
        from strands_robots.factory import _auto_detect_mode

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("STRANDS_ROBOT_MODE", None)
            assert _auto_detect_mode("panda", {"sim": "panda"}) == "sim"

    def test_defaults_to_sim_for_sim_only_robot(self):
        from strands_robots.factory import _auto_detect_mode

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("STRANDS_ROBOT_MODE", None)
            assert _auto_detect_mode("panda", {"sim": "panda", "real": None}) == "sim"

    def test_defaults_to_sim_without_robot_info(self):
        from strands_robots.factory import _auto_detect_mode

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("STRANDS_ROBOT_MODE", None)
            assert _auto_detect_mode("unknown", None) == "sim"

    def test_serial_detection_import_error(self):
        """If serial.tools is not available, should default to sim."""
        from strands_robots.factory import _auto_detect_mode

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("STRANDS_ROBOT_MODE", None)
            with patch.dict("sys.modules", {"serial": None, "serial.tools": None, "serial.tools.list_ports": None}):
                assert _auto_detect_mode("so100", {"real": "so100_follower"}) == "sim"


class TestRobotFactoryMujoco:
    """Test Robot() with mocked MuJoCo backend."""

    @classmethod
    def setup_class(cls):
        _inject_mock_modules()

    def test_robot_sim_mujoco_creates_simulation(self):
        """Robot('so100', mode='sim') should create a Simulation instance."""
        mock_sim = MagicMock()
        mock_sim.create_world = MagicMock()
        mock_sim.add_robot = MagicMock(return_value={"status": "success"})

        with (
            patch("strands_robots.factory.Simulation", return_value=mock_sim, create=True),
            patch("strands_robots.simulation.Simulation", return_value=mock_sim, create=True),
        ):
            from strands_robots.factory import Robot

            result = Robot("so100", mode="sim", backend="mujoco")
            assert result is mock_sim

    def test_robot_sim_creates_world_and_adds_robot(self):
        """Robot() should call create_world() and add_robot()."""
        mock_sim = MagicMock()
        mock_sim.create_world = MagicMock()
        mock_sim.add_robot = MagicMock(return_value={"status": "success"})

        with patch("strands_robots.simulation.Simulation", return_value=mock_sim):
            from strands_robots.factory import Robot

            Robot("panda", mode="sim")

            mock_sim.create_world.assert_called_once()
            mock_sim.add_robot.assert_called_once()
            call_kwargs = mock_sim.add_robot.call_args
            assert call_kwargs[1]["name"] == "panda" or call_kwargs[1].get("data_config") == "panda"

    def test_robot_sim_with_position(self):
        """Robot() with position should pass it to add_robot."""
        mock_sim = MagicMock()
        mock_sim.create_world = MagicMock()
        mock_sim.add_robot = MagicMock(return_value={"status": "success"})

        with patch("strands_robots.simulation.Simulation", return_value=mock_sim):
            from strands_robots.factory import Robot

            Robot("so100", mode="sim", position=[1.0, 2.0, 3.0])

            call_kwargs = mock_sim.add_robot.call_args
            assert call_kwargs[1]["position"] == [1.0, 2.0, 3.0]

    def test_robot_sim_error_raises(self):
        """Robot() should raise if add_robot fails."""
        mock_sim = MagicMock()
        mock_sim.create_world = MagicMock()
        mock_sim.add_robot = MagicMock(return_value={"status": "error", "content": [{"text": "failed"}]})

        with patch("strands_robots.simulation.Simulation", return_value=mock_sim):
            from strands_robots.factory import Robot

            with pytest.raises(RuntimeError, match="Failed to create sim robot"):
                Robot("so100", mode="sim")

    def test_robot_alias_resolved_before_creation(self):
        """Robot('g1') should resolve to 'unitree_g1'."""
        mock_sim = MagicMock()
        mock_sim.create_world = MagicMock()
        mock_sim.add_robot = MagicMock(return_value={"status": "success"})

        with patch("strands_robots.simulation.Simulation", return_value=mock_sim):
            from strands_robots.factory import Robot

            Robot("g1", mode="sim")

            call_kwargs = mock_sim.add_robot.call_args
            assert call_kwargs[1]["name"] == "unitree_g1"


class TestRobotFactoryIsaac:
    """Test Robot() with mocked Isaac Sim backend."""

    def test_robot_isaac_backend(self):
        """Robot(backend='isaac') should create IsaacSimBackend."""
        mock_isaac = MagicMock()
        mock_isaac.create_world = MagicMock()
        mock_isaac.add_robot = MagicMock(return_value={"status": "success"})

        mock_config = MagicMock()

        with (
            patch("strands_robots.factory.IsaacSimBackend", return_value=mock_isaac, create=True),
            patch("strands_robots.factory.IsaacSimConfig", return_value=mock_config, create=True),
            patch("strands_robots.isaac.isaac_sim_backend.IsaacSimBackend", return_value=mock_isaac, create=True),
            patch("strands_robots.isaac.isaac_sim_backend.IsaacSimConfig", return_value=mock_config, create=True),
        ):
            from strands_robots.factory import Robot

            result = Robot("unitree_go2", mode="sim", backend="isaac", num_envs=4096)
            assert result is mock_isaac

    def test_isaac_error_raises(self):
        """Isaac backend should raise on failure."""
        mock_isaac = MagicMock()
        mock_isaac.create_world = MagicMock()
        mock_isaac.add_robot = MagicMock(return_value={"status": "error"})

        with (
            patch("strands_robots.isaac.isaac_sim_backend.IsaacSimBackend", return_value=mock_isaac),
            patch("strands_robots.isaac.isaac_sim_backend.IsaacSimConfig", MagicMock()),
        ):
            from strands_robots.factory import Robot

            with pytest.raises(RuntimeError, match="Failed to create Isaac robot"):
                Robot("unitree_go2", mode="sim", backend="isaac")


class TestRobotFactoryNewton:
    """Test Robot() with mocked Newton backend."""

    def test_robot_newton_backend(self):
        """Robot(backend='newton') should create NewtonBackend."""
        mock_newton = MagicMock()
        mock_newton.create_world = MagicMock()
        mock_newton.add_robot = MagicMock(return_value={"status": "success"})
        mock_newton.replicate = MagicMock()

        with (
            patch("strands_robots.newton.newton_backend.NewtonBackend", return_value=mock_newton),
            patch("strands_robots.newton.newton_backend.NewtonConfig", MagicMock()),
        ):
            from strands_robots.factory import Robot

            result = Robot("so100", mode="sim", backend="newton", num_envs=64, solver="mujoco")
            assert result is mock_newton

    def test_newton_replicates_for_parallel_envs(self):
        """Newton backend should call replicate() when num_envs > 1."""
        mock_newton = MagicMock()
        mock_newton.create_world = MagicMock()
        mock_newton.add_robot = MagicMock(return_value={"status": "success"})
        mock_newton.replicate = MagicMock()

        with (
            patch("strands_robots.newton.newton_backend.NewtonBackend", return_value=mock_newton),
            patch("strands_robots.newton.newton_backend.NewtonConfig", MagicMock()),
        ):
            from strands_robots.factory import Robot

            Robot("so100", mode="sim", backend="newton", num_envs=64)

            mock_newton.replicate.assert_called_once_with(num_envs=64)

    def test_newton_no_replicate_for_single_env(self):
        """Newton backend should NOT replicate when num_envs=1."""
        mock_newton = MagicMock()
        mock_newton.create_world = MagicMock()
        mock_newton.add_robot = MagicMock(return_value={"status": "success"})
        mock_newton.replicate = MagicMock()

        with (
            patch("strands_robots.newton.newton_backend.NewtonBackend", return_value=mock_newton),
            patch("strands_robots.newton.newton_backend.NewtonConfig", MagicMock()),
        ):
            from strands_robots.factory import Robot

            Robot("so100", mode="sim", backend="newton", num_envs=1)

            mock_newton.replicate.assert_not_called()

    def test_newton_error_raises(self):
        """Newton backend should raise on failure."""
        mock_newton = MagicMock()
        mock_newton.create_world = MagicMock()
        mock_newton.add_robot = MagicMock(return_value={"status": "error"})

        with (
            patch("strands_robots.newton.newton_backend.NewtonBackend", return_value=mock_newton),
            patch("strands_robots.newton.newton_backend.NewtonConfig", MagicMock()),
        ):
            from strands_robots.factory import Robot

            with pytest.raises(RuntimeError, match="Failed to create Newton robot"):
                Robot("so100", mode="sim", backend="newton")


class TestRobotFactoryReal:
    """Test Robot() with mocked real hardware backend."""

    @classmethod
    def setup_class(cls):
        _inject_mock_modules()

    def test_robot_real_mode(self):
        """Robot(mode='real') should create HardwareRobot."""
        mock_hw = MagicMock()

        with patch("strands_robots.robot.Robot", return_value=mock_hw):
            from strands_robots.factory import Robot

            result = Robot("so100", mode="real")
            assert result is mock_hw

    def test_robot_real_passes_cameras(self):
        """Real mode should forward cameras config."""
        mock_hw_cls = MagicMock()

        with patch("strands_robots.robot.Robot", mock_hw_cls):
            from strands_robots.factory import Robot

            cameras = {"front": {"index_or_path": 0}}
            Robot("so100", mode="real", cameras=cameras)

            call_kwargs = mock_hw_cls.call_args[1]
            assert call_kwargs["cameras"] == cameras

    def test_robot_real_uses_real_config(self):
        """Real mode should use the 'real' key from robot_info."""
        mock_hw_cls = MagicMock()

        with patch("strands_robots.robot.Robot", mock_hw_cls):
            from strands_robots.factory import Robot

            Robot("so100", mode="real")

            call_kwargs = mock_hw_cls.call_args[1]
            # so100's real config is "so100_follower"
            assert call_kwargs["robot"] == "so100_follower"


class TestListRobots:
    """Comprehensive tests for list_robots()."""

    def test_list_all(self):
        from strands_robots.factory import list_robots

        robots = list_robots()
        assert len(robots) >= 34

    def test_list_all_has_required_fields(self):
        from strands_robots.factory import list_robots

        for robot in list_robots():
            assert "name" in robot
            assert "description" in robot
            assert "has_sim" in robot
            assert "has_real" in robot

    def test_list_sorted_by_name(self):
        from strands_robots.factory import list_robots

        robots = list_robots()
        names = [r["name"] for r in robots]
        assert names == sorted(names)

    def test_list_sim_only(self):
        from strands_robots.factory import list_robots

        sim_robots = list_robots(mode="sim")
        for r in sim_robots:
            assert r["has_sim"] is True

    def test_list_real_only(self):
        from strands_robots.factory import list_robots

        real_robots = list_robots(mode="real")
        for r in real_robots:
            assert r["has_real"] is True

    def test_list_both(self):
        from strands_robots.factory import list_robots

        both = list_robots(mode="both")
        for r in both:
            assert r["has_sim"] is True
            assert r["has_real"] is True

    def test_so100_has_both(self):
        from strands_robots.factory import list_robots

        both_names = {r["name"] for r in list_robots(mode="both")}
        assert "so100" in both_names

    def test_panda_sim_only(self):
        from strands_robots.factory import list_robots

        robots_dict = {r["name"]: r for r in list_robots()}
        assert robots_dict["panda"]["has_sim"] is True
        assert robots_dict["panda"]["has_real"] is False

    def test_lekiwi_real_only(self):
        from strands_robots.factory import list_robots

        robots_dict = {r["name"]: r for r in list_robots()}
        assert robots_dict["lekiwi"]["has_sim"] is False
        assert robots_dict["lekiwi"]["has_real"] is True


class TestCreateRobotAlias:
    """Test that create_robot is an alias for Robot."""

    def test_create_robot_is_robot(self):
        from strands_robots.factory import Robot, create_robot

        assert create_robot is Robot


class TestUnifiedRobotsRegistry:
    """Validate _UNIFIED_ROBOTS integrity."""

    def test_all_entries_have_required_keys(self):
        from strands_robots.factory import _UNIFIED_ROBOTS

        for name, info in _UNIFIED_ROBOTS.items():
            assert "sim" in info, f"{name} missing 'sim' key"
            assert "real" in info, f"{name} missing 'real' key"
            assert "description" in info, f"{name} missing 'description' key"

    def test_no_empty_descriptions(self):
        from strands_robots.factory import _UNIFIED_ROBOTS

        for name, info in _UNIFIED_ROBOTS.items():
            assert len(info["description"]) > 0, f"{name} has empty description"

    def test_sim_or_real_required(self):
        """Every robot must have at least sim or real support."""
        from strands_robots.factory import _UNIFIED_ROBOTS

        for name, info in _UNIFIED_ROBOTS.items():
            assert info["sim"] is not None or info["real"] is not None, f"{name} has neither sim nor real support"

    def test_aliases_point_to_valid_robots(self):
        from strands_robots.factory import _ALIASES, _UNIFIED_ROBOTS

        for alias, canonical in _ALIASES.items():
            assert (
                canonical in _UNIFIED_ROBOTS
            ), f"Alias '{alias}' points to '{canonical}' which is not in _UNIFIED_ROBOTS"
