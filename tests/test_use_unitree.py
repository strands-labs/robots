"""Tests for use_unitree tool — universal Unitree SDK interface.

Tests cover:
- Discovery actions (list_robots, list_services, list_methods, diagnose)
- Action parsing (dot-separated strings)
- G1 arm gesture resolution
- Velocity clamping safety guards
- Dangerous action rejection
- Mock mode execution
- Client caching and connection management
- Error handling (missing SDK, invalid actions, unknown methods)
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────────
# Pre-mock strands SDK (may not be installed in CI)
# ─────────────────────────────────────────────────────────────────────
_mock_strands = types.ModuleType("strands")
_mock_strands.tool = lambda f: f
sys.modules.setdefault("strands", _mock_strands)

import strands_robots.tools.use_unitree as _use_unitree_mod  # noqa: E402
from strands_robots.tools.use_unitree import (  # noqa: E402
    _DANGEROUS_ACTIONS,
    _G1_ARM_GESTURES,
    _ROBOT_INFO,
    _SERVICE_REGISTRY,
    _VELOCITY_LIMITS,
    _clamp_velocity,
    _ConnectionManager,
    _handle_discovery,
    _is_mock_mode,
    _MockClient,
    _parse_action,
    _resolve_gesture,
    use_unitree,
)

# ─────────────────────────────────────────────────────────────────────
# Helpers: fake SDK client classes for introspection tests
# ─────────────────────────────────────────────────────────────────────


class _FakeSportClient:
    """Mimics SportClient method signatures for introspection tests."""

    def Init(self):  # noqa: N802
        pass

    def Move(self, vx: float, vy: float, vyaw: float):  # noqa: N802
        pass

    def StandUp(self):  # noqa: N802
        pass

    def StandDown(self):  # noqa: N802
        pass

    def BackFlip(self):  # noqa: N802
        pass

    def Sit(self):  # noqa: N802
        pass

    def RecoveryStand(self):  # noqa: N802
        pass


class _FakeArmClient:
    """Mimics G1ArmActionClient method signatures for introspection tests."""

    def Init(self):  # noqa: N802
        pass

    def ExecuteAction(self, action_id: int):  # noqa: N802
        pass

    def GetActionList(self):  # noqa: N802
        pass


# ─────────────────────────────────────────────────────────────────────
# Action parsing
# ─────────────────────────────────────────────────────────────────────


class TestParseAction:
    def test_three_parts(self):
        assert _parse_action("go2.sport.Move") == ("go2", "sport", "Move")

    def test_three_parts_g1(self):
        assert _parse_action("g1.loco.StandUp") == ("g1", "loco", "StandUp")

    def test_two_parts(self):
        assert _parse_action("go2.sport") == ("go2", "sport", "")

    def test_one_part_discovery(self):
        assert _parse_action("list_robots") == ("__discovery__", "", "list_robots")

    def test_one_part_diagnose(self):
        assert _parse_action("diagnose") == ("__discovery__", "", "diagnose")

    def test_invalid_too_many_parts(self):
        with pytest.raises(ValueError, match="Invalid action format"):
            _parse_action("a.b.c.d")


# ─────────────────────────────────────────────────────────────────────
# Gesture resolution
# ─────────────────────────────────────────────────────────────────────


class TestGestureResolution:
    def test_hug(self):
        assert _resolve_gesture("hug") == 19

    def test_high_five(self):
        assert _resolve_gesture("high_five") == 18

    def test_space_normalized(self):
        assert _resolve_gesture("high five") == 18

    def test_hyphen_normalized(self):
        assert _resolve_gesture("high-five") == 18

    def test_case_insensitive(self):
        assert _resolve_gesture("HUG") == 19

    def test_shake_hand(self):
        assert _resolve_gesture("shake_hand") == 27

    def test_unknown_returns_none(self):
        assert _resolve_gesture("moonwalk") is None

    def test_all_gestures_have_ids(self):
        for name, action_id in _G1_ARM_GESTURES.items():
            assert isinstance(action_id, int)
            assert _resolve_gesture(name) == action_id


# ─────────────────────────────────────────────────────────────────────
# Velocity clamping
# ─────────────────────────────────────────────────────────────────────


class TestVelocityClamping:
    def test_within_limits_unchanged(self):
        params = {"vx": 0.5, "vy": 0.2, "vyaw": 0.3}
        result = _clamp_velocity("go2", params)
        assert result == params

    def test_exceeds_positive_limit(self):
        params = {"vx": 5.0, "vy": 0, "vyaw": 0}
        result = _clamp_velocity("go2", params)
        assert result["vx"] == 1.5  # Go2 vx limit

    def test_exceeds_negative_limit(self):
        params = {"vx": -5.0, "vy": 0, "vyaw": 0}
        result = _clamp_velocity("go2", params)
        assert result["vx"] == -1.5

    def test_g1_limits(self):
        params = {"vx": 2.0, "vy": 1.0, "vyaw": 2.0}
        result = _clamp_velocity("g1", params)
        assert result["vx"] == 0.6
        assert result["vy"] == 0.3
        assert result["vyaw"] == 0.5

    def test_unknown_robot_no_clamping(self):
        params = {"vx": 100.0}
        result = _clamp_velocity("unknown_robot", params)
        assert result == params

    def test_no_velocity_params_unchanged(self):
        params = {"level": 1}
        result = _clamp_velocity("go2", params)
        assert result == params

    def test_original_not_mutated(self):
        params = {"vx": 5.0}
        _clamp_velocity("go2", params)
        assert params["vx"] == 5.0  # Original unchanged


# ─────────────────────────────────────────────────────────────────────
# Discovery actions
# ─────────────────────────────────────────────────────────────────────


class TestDiscovery:
    def test_list_robots(self):
        result = _handle_discovery("list_robots", {})
        assert result["status"] == "success"
        robots = result["content"][0]["json"]["robots"]
        assert len(robots) == len(_ROBOT_INFO)
        names = [r["name"] for r in robots]
        assert "go2" in names
        assert "g1" in names
        assert "h1" in names
        assert "b2" in names

    def test_list_services_go2(self):
        result = _handle_discovery("list_services", {"robot": "go2"})
        assert result["status"] == "success"
        services = result["content"][0]["json"]["services"]
        svc_names = [s["name"] for s in services]
        assert "sport" in svc_names
        assert "video" in svc_names

    def test_list_services_missing_robot(self):
        result = _handle_discovery("list_services", {})
        assert result["status"] == "error"

    def test_list_services_unknown_robot(self):
        result = _handle_discovery("list_services", {"robot": "nonexistent"})
        assert result["status"] == "error"

    def test_list_methods_go2_sport(self):
        """Test list_methods with mocked SDK client for introspection."""
        mock_mod = types.ModuleType("unitree_sdk2py.go2.sport.sport_client")
        mock_mod.SportClient = _FakeSportClient

        with patch("importlib.import_module", return_value=mock_mod):
            result = _handle_discovery("list_methods", {"robot": "go2", "service": "sport"})

        assert result["status"] == "success"
        methods = result["content"][0]["json"]["methods"]
        method_names = [m["name"] for m in methods]
        assert "Move" in method_names
        assert "StandUp" in method_names
        assert "BackFlip" in method_names
        # BackFlip should be flagged as dangerous
        backflip = next(m for m in methods if m["name"] == "BackFlip")
        assert backflip["dangerous"] is True

    def test_list_methods_g1_arm_includes_gestures(self):
        """Test list_methods for G1 arm includes gesture shortcuts."""
        mock_mod = types.ModuleType("unitree_sdk2py.g1.arm.g1_arm_action_client")
        mock_mod.G1ArmActionClient = _FakeArmClient

        with patch("importlib.import_module", return_value=mock_mod):
            result = _handle_discovery("list_methods", {"robot": "g1", "service": "arm"})

        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Gesture shortcuts" in text

    def test_list_methods_without_sdk_returns_error(self):
        """When SDK is not installed, list_methods returns error gracefully."""
        result = _handle_discovery("list_methods", {"robot": "go2", "service": "sport"})
        # Without unitree_sdk2py installed, _get_client_methods returns []
        assert result["status"] == "error"
        assert "No methods found" in result["content"][0]["text"]

    def test_list_methods_missing_params(self):
        result = _handle_discovery("list_methods", {})
        assert result["status"] == "error"

    def test_diagnose(self):
        result = _handle_discovery("diagnose", {})
        assert result["status"] == "success"
        diag = result["content"][0]["json"]
        assert "sdk_installed" in diag
        assert "cyclonedds_installed" in diag
        assert "mock_mode" in diag
        assert "registered_services" in diag
        assert diag["registered_services"] == len(_SERVICE_REGISTRY)

    def test_unknown_discovery_action(self):
        result = _handle_discovery("foobar", {})
        assert result["status"] == "error"


# ─────────────────────────────────────────────────────────────────────
# Mock mode
# ─────────────────────────────────────────────────────────────────────


class TestMockMode:
    def test_is_mock_mode_false_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _is_mock_mode() is False

    def test_is_mock_mode_true(self):
        with patch.dict(os.environ, {"UNITREE_MOCK": "true"}):
            assert _is_mock_mode() is True

    def test_is_mock_mode_yes(self):
        with patch.dict(os.environ, {"UNITREE_MOCK": "yes"}):
            assert _is_mock_mode() is True

    def test_is_mock_mode_1(self):
        with patch.dict(os.environ, {"UNITREE_MOCK": "1"}):
            assert _is_mock_mode() is True

    def test_mock_client_call(self):
        client = _MockClient("go2", "sport")
        result = client.Move(vx=0.3, vy=0, vyaw=0)
        assert result == 0
        assert len(client._calls) == 1
        assert client._calls[0]["method"] == "Move"
        assert client._calls[0]["parameters"] == {"vx": 0.3, "vy": 0, "vyaw": 0}

    def test_mock_client_no_args(self):
        client = _MockClient("go2", "sport")
        result = client.StandUp()
        assert result == 0
        assert client._calls[0]["method"] == "StandUp"

    def test_mock_client_private_attr_raises(self):
        client = _MockClient("go2", "sport")
        with pytest.raises(AttributeError):
            client._private_method()

    def test_mock_client_multiple_calls(self):
        client = _MockClient("go2", "sport")
        client.StandUp()
        client.Move(vx=0.5, vy=0, vyaw=0)
        client.Sit()
        assert len(client._calls) == 3
        assert client._calls[0]["method"] == "StandUp"
        assert client._calls[1]["method"] == "Move"
        assert client._calls[2]["method"] == "Sit"


# ─────────────────────────────────────────────────────────────────────
# Tool integration (mock mode)
# ─────────────────────────────────────────────────────────────────────


class TestUseUnitreeMockMode:
    """Test the main use_unitree function with UNITREE_MOCK=true."""

    @pytest.fixture(autouse=True)
    def _enable_mock(self):
        with patch.dict(os.environ, {"UNITREE_MOCK": "true"}):
            yield

    def test_move_go2(self):
        result = use_unitree(action="go2.sport.Move", parameters={"vx": 0.3, "vy": 0, "vyaw": 0})
        assert result["status"] == "success"
        assert "go2.sport.Move" in result["content"][0]["text"]

    def test_standup_go2(self):
        result = use_unitree(action="go2.sport.StandUp")
        assert result["status"] == "success"

    def test_g1_loco_move(self):
        result = use_unitree(action="g1.loco.Move", parameters={"vx": 0.2, "vy": 0, "vyaw": 0})
        assert result["status"] == "success"

    def test_g1_arm_gesture_hug(self):
        result = use_unitree(action="g1.arm.hug")
        assert result["status"] == "success"
        assert "g1.arm.ExecuteAction" in result["content"][0]["text"]

    def test_g1_arm_gesture_high_five(self):
        result = use_unitree(action="g1.arm.high_five")
        assert result["status"] == "success"

    def test_g1_arm_gesture_shake_hand(self):
        result = use_unitree(action="g1.arm.shake_hand")
        assert result["status"] == "success"
        assert "ExecuteAction" in result["content"][0]["text"]

    def test_g1_arm_gesture_preserves_extra_params(self):
        """Gesture resolution should merge with user params."""
        result = use_unitree(action="g1.arm.hug", parameters={"extra_key": "value"})
        assert result["status"] == "success"

    def test_dangerous_action_rejected_without_confirm(self):
        result = use_unitree(action="go2.sport.BackFlip")
        assert result["status"] == "error"
        assert "dangerous" in result["content"][0]["text"].lower()

    def test_dangerous_action_allowed_with_confirm(self):
        result = use_unitree(action="go2.sport.BackFlip", confirm=True)
        assert result["status"] == "success"

    def test_all_dangerous_actions_rejected(self):
        """Every action in _DANGEROUS_ACTIONS should be rejected without confirm."""
        for action_name in _DANGEROUS_ACTIONS:
            result = use_unitree(action=f"go2.sport.{action_name}")
            assert result["status"] == "error", f"{action_name} was not rejected"

    def test_velocity_clamping_applied(self):
        result = use_unitree(action="go2.sport.Move", parameters={"vx": 10.0, "vy": 0, "vyaw": 0})
        assert result["status"] == "success"

    def test_set_velocity_clamped(self):
        result = use_unitree(action="g1.loco.SetVelocity", parameters={"vx": 5.0, "vy": 5.0, "vyaw": 5.0})
        assert result["status"] == "success"

    def test_discovery_list_robots(self):
        result = use_unitree(action="list_robots")
        assert result["status"] == "success"
        assert "go2" in result["content"][0]["text"]

    def test_discovery_list_services(self):
        result = use_unitree(action="list_services", parameters={"robot": "g1"})
        assert result["status"] == "success"
        assert "loco" in result["content"][0]["text"]

    def test_discovery_diagnose(self):
        result = use_unitree(action="diagnose")
        assert result["status"] == "success"

    def test_invalid_action_format(self):
        result = use_unitree(action="a.b.c.d")
        assert result["status"] == "error"

    def test_custom_interface_and_domain(self):
        result = use_unitree(
            action="go2.sport.StandUp",
            interface="eth0",
            domain_id=1,
        )
        assert result["status"] == "success"

    def test_env_interface(self):
        with patch.dict(os.environ, {"UNITREE_NETWORK_INTERFACE": "wlan0", "UNITREE_MOCK": "true"}):
            result = use_unitree(action="go2.sport.StandUp")
            assert result["status"] == "success"

    def test_env_domain_id(self):
        with patch.dict(os.environ, {"UNITREE_DOMAIN_ID": "5", "UNITREE_MOCK": "true"}):
            result = use_unitree(action="go2.sport.StandUp")
            assert result["status"] == "success"

    def test_none_parameters_default(self):
        result = use_unitree(action="go2.sport.StandUp", parameters=None)
        assert result["status"] == "success"

    def test_b2_sport(self):
        result = use_unitree(action="b2.sport.StandUp")
        assert result["status"] == "success"

    def test_h1_loco(self):
        result = use_unitree(action="h1.loco.Move", parameters={"vx": 0.1, "vy": 0, "vyaw": 0})
        assert result["status"] == "success"

    def test_common_motion_switcher(self):
        result = use_unitree(action="common.motion_switcher.SwitchMode")
        assert result["status"] == "success"


# ─────────────────────────────────────────────────────────────────────
# Connection manager
# ─────────────────────────────────────────────────────────────────────


class TestConnectionManager:
    def test_init(self):
        mgr = _ConnectionManager()
        assert mgr._channel_initialized is False
        assert len(mgr._clients) == 0

    def test_close_all(self):
        mgr = _ConnectionManager()
        mgr._clients["test"] = MagicMock()
        mgr.close_all()
        assert len(mgr._clients) == 0

    def test_get_client_unknown_service(self):
        mgr = _ConnectionManager()
        mgr._channel_initialized = True  # Skip DDS init
        with pytest.raises(ValueError, match="Unknown service"):
            mgr.get_client("nonexistent", "nonexistent")

    def test_get_client_caches(self):
        mgr = _ConnectionManager()
        mgr._channel_initialized = True

        mock_client = MagicMock()
        mock_client.Init = MagicMock()
        mock_module = MagicMock()
        mock_module.SportClient.return_value = mock_client

        with patch("importlib.import_module", return_value=mock_module):
            client1 = mgr.get_client("go2", "sport", "eth0", 0)
            client2 = mgr.get_client("go2", "sport", "eth0", 0)
            assert client1 is client2  # Cached

    def test_different_interfaces_different_clients(self):
        mgr = _ConnectionManager()
        mgr._channel_initialized = True

        mock_module = MagicMock()

        with patch("importlib.import_module", return_value=mock_module):
            client1 = mgr.get_client("go2", "sport", "eth0", 0)
            client2 = mgr.get_client("go2", "sport", "wlan0", 0)
            assert client1 is not None
            assert client2 is not None

    def test_different_domains_different_clients(self):
        mgr = _ConnectionManager()
        mgr._channel_initialized = True

        mock_module = MagicMock()

        with patch("importlib.import_module", return_value=mock_module):
            client1 = mgr.get_client("go2", "sport", "eth0", 0)
            client2 = mgr.get_client("go2", "sport", "eth0", 1)
            assert client1 is not None
            assert client2 is not None


# ─────────────────────────────────────────────────────────────────────
# Registry completeness
# ─────────────────────────────────────────────────────────────────────


class TestRegistryCompleteness:
    def test_all_robot_services_registered(self):
        """Every service listed in _ROBOT_INFO should have a _SERVICE_REGISTRY entry."""
        for robot, info in _ROBOT_INFO.items():
            for svc in info["services"]:
                assert (robot, svc) in _SERVICE_REGISTRY, f"Missing registry entry for ({robot}, {svc})"

    def test_all_registry_entries_have_robot_info(self):
        """Every _SERVICE_REGISTRY entry should map to a known robot."""
        for robot, service in _SERVICE_REGISTRY:
            assert robot in _ROBOT_INFO, f"Registry entry ({robot}, {service}) has no _ROBOT_INFO"

    def test_velocity_limits_for_all_quadrupeds_and_humanoids(self):
        """All non-common robots should have velocity limits."""
        for robot, info in _ROBOT_INFO.items():
            if robot == "common":
                continue
            assert robot in _VELOCITY_LIMITS, f"Missing velocity limits for {robot}"

    def test_gesture_ids_unique(self):
        """All G1 arm gesture IDs should be unique."""
        ids = list(_G1_ARM_GESTURES.values())
        assert len(ids) == len(set(ids)), "Duplicate gesture IDs found"

    def test_dangerous_actions_are_strings(self):
        """All dangerous actions should be string method names."""
        for action in _DANGEROUS_ACTIONS:
            assert isinstance(action, str)

    def test_service_count(self):
        """Verify total service count matches documentation."""
        assert len(_SERVICE_REGISTRY) >= 15  # 5 go2 + 5 b2 + 3 g1 + 1 h1 + 1 common

    def test_robot_info_has_required_fields(self):
        """Every _ROBOT_INFO entry should have all required fields."""
        required = {"type", "description", "services", "joints"}
        for robot, info in _ROBOT_INFO.items():
            for field in required:
                assert field in info, f"{robot} missing field: {field}"

    def test_gesture_count(self):
        """Verify we have all 16 documented G1 arm gestures."""
        assert len(_G1_ARM_GESTURES) == 16


# ─────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_import_error_message(self):
        """When SDK not installed, should give helpful install instructions."""
        with patch.object(_use_unitree_mod, "_is_mock_mode", return_value=False):
            with patch.object(
                _use_unitree_mod._conn,
                "get_client",
                side_effect=ImportError("No module named 'unitree_sdk2py'"),
            ):
                result = use_unitree(action="go2.sport.StandUp")
                assert result["status"] == "error"
                assert "unitree_sdk2py" in result["content"][0]["text"]
                assert "pip install" in result["content"][0]["text"]

    def test_runtime_error(self):
        """Runtime errors should be caught and reported."""
        with patch.object(_use_unitree_mod, "_is_mock_mode", return_value=False):
            with patch.object(
                _use_unitree_mod._conn,
                "get_client",
                side_effect=RuntimeError("DDS connection failed"),
            ):
                result = use_unitree(action="go2.sport.StandUp")
                assert result["status"] == "error"
                assert "DDS connection failed" in result["content"][0]["text"]

    def test_unknown_method_on_mock(self):
        """Mock client accepts any method (dynamic dispatch)."""
        with patch.dict(os.environ, {"UNITREE_MOCK": "true"}):
            result = use_unitree(action="go2.sport.NonexistentMethod")
            assert result["status"] == "success"

    def test_unknown_service(self):
        """Unknown robot.service should error with helpful message."""
        with patch.object(_use_unitree_mod, "_is_mock_mode", return_value=False):
            with patch.object(
                _use_unitree_mod._conn,
                "get_client",
                side_effect=ValueError("Unknown service: fake.fake"),
            ):
                result = use_unitree(action="fake.fake.Method")
                assert result["status"] == "error"


# ─────────────────────────────────────────────────────────────────────
# Integration: full workflow simulation
# ─────────────────────────────────────────────────────────────────────


class TestFullWorkflow:
    """Simulate a realistic agent workflow in mock mode."""

    @pytest.fixture(autouse=True)
    def _enable_mock(self):
        with patch.dict(os.environ, {"UNITREE_MOCK": "true"}):
            yield

    def test_discover_then_control_go2(self):
        # 1. Discover robots
        r = use_unitree(action="list_robots")
        assert r["status"] == "success"

        # 2. List Go2 services
        r = use_unitree(action="list_services", parameters={"robot": "go2"})
        assert r["status"] == "success"

        # 3. Stand up
        r = use_unitree(action="go2.sport.StandUp")
        assert r["status"] == "success"

        # 4. Move forward
        r = use_unitree(action="go2.sport.Move", parameters={"vx": 0.3, "vy": 0, "vyaw": 0})
        assert r["status"] == "success"

        # 5. Stop
        r = use_unitree(action="go2.sport.StopMove")
        assert r["status"] == "success"

        # 6. Sit
        r = use_unitree(action="go2.sport.Sit")
        assert r["status"] == "success"

    def test_g1_humanoid_workflow(self):
        # Start locomotion
        r = use_unitree(action="g1.loco.Start")
        assert r["status"] == "success"

        # Walk
        r = use_unitree(action="g1.loco.Move", parameters={"vx": 0.2, "vy": 0, "vyaw": 0})
        assert r["status"] == "success"

        # Wave
        r = use_unitree(action="g1.arm.high_wave")
        assert r["status"] == "success"

        # Hug
        r = use_unitree(action="g1.arm.hug")
        assert r["status"] == "success"

        # Stop
        r = use_unitree(action="g1.loco.StopMove")
        assert r["status"] == "success"

    def test_b2_industrial_workflow(self):
        r = use_unitree(action="b2.sport.StandUp")
        assert r["status"] == "success"

        r = use_unitree(action="b2.sport.Move", parameters={"vx": 0.3, "vy": 0, "vyaw": 0})
        assert r["status"] == "success"

    def test_diagnose_workflow(self):
        r = use_unitree(action="diagnose")
        assert r["status"] == "success"
        diag = r["content"][0]["json"]
        assert diag["mock_mode"] is True


# ─────────────────────────────────────────────────────────────────────
# Return format handling
# ─────────────────────────────────────────────────────────────────────


class TestReturnFormats:
    def test_tuple_return(self):
        """Test handling of (code, data) tuple returns."""
        mock_client = MagicMock()
        mock_client.AutoRecoveryGet.return_value = (0, True)

        with patch.object(_use_unitree_mod, "_is_mock_mode", return_value=False):
            with patch.object(_use_unitree_mod._conn, "get_client", return_value=mock_client):
                result = use_unitree(action="go2.sport.AutoRecoveryGet")
                assert result["status"] == "success"
                assert result["content"][0]["json"]["return_code"] == 0
                assert result["content"][0]["json"]["data"] is True

    def test_int_return(self):
        """Test handling of simple int return codes."""
        mock_client = MagicMock()
        mock_client.StandUp.return_value = 0

        with patch.object(_use_unitree_mod, "_is_mock_mode", return_value=False):
            with patch.object(_use_unitree_mod._conn, "get_client", return_value=mock_client):
                result = use_unitree(action="go2.sport.StandUp")
                assert result["status"] == "success"
                assert result["content"][0]["json"]["return_code"] == 0

    def test_nonzero_return_code(self):
        """Non-zero return codes should indicate error."""
        mock_client = MagicMock()
        mock_client.Move.return_value = -1

        with patch.object(_use_unitree_mod, "_is_mock_mode", return_value=False):
            with patch.object(_use_unitree_mod._conn, "get_client", return_value=mock_client):
                result = use_unitree(action="go2.sport.Move", parameters={"vx": 0, "vy": 0, "vyaw": 0})
                assert result["status"] == "error"

    def test_none_return_treated_as_success(self):
        """None return (non-int, non-tuple) should be treated as success."""
        mock_client = MagicMock()
        mock_client.SomeMethod.return_value = None

        with patch.object(_use_unitree_mod, "_is_mock_mode", return_value=False):
            with patch.object(_use_unitree_mod._conn, "get_client", return_value=mock_client):
                result = use_unitree(action="go2.sport.SomeMethod")
                assert result["status"] == "success"

    def test_method_not_found_on_real_client(self):
        """When method not found on real client, should return error."""
        mock_client = MagicMock(spec=[])  # Empty spec = no attributes
        # getattr with default returns None for unlisted attrs with spec
        mock_client.configure_mock(**{})

        with patch.object(_use_unitree_mod, "_is_mock_mode", return_value=False):
            with patch.object(_use_unitree_mod._conn, "get_client") as get_mock:
                # Create a client where getattr(client, "Foo", None) returns None
                real_mock = MagicMock()
                real_mock.Foo = None  # Explicitly set to None
                get_mock.return_value = real_mock

                result = use_unitree(action="go2.sport.Foo")
                assert result["status"] == "error"
                assert "not found" in result["content"][0]["text"]


# ─────────────────────────────────────────────────────────────────────
# Deadlock regression + gesture override
# ─────────────────────────────────────────────────────────────────────


class TestDeadlockRegression:
    """Verify _ensure_channel is called outside get_client's lock to avoid deadlock."""

    def _mock_unitree_sdk(self):
        """Create mock unitree_sdk2py modules for sys.modules injection."""
        mock_channel_init = MagicMock()
        mock_channel_mod = types.ModuleType("unitree_sdk2py.core.channel")
        mock_channel_mod.ChannelFactoryInitialize = mock_channel_init

        mock_sport_client = MagicMock()
        mock_sport_client.Init = MagicMock()
        mock_sport_mod = types.ModuleType("unitree_sdk2py.go2.sport.sport_client")
        mock_sport_mod.SportClient = MagicMock(return_value=mock_sport_client)

        modules = {
            "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
            "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
            "unitree_sdk2py.core.channel": mock_channel_mod,
            "unitree_sdk2py.go2": types.ModuleType("unitree_sdk2py.go2"),
            "unitree_sdk2py.go2.sport": types.ModuleType("unitree_sdk2py.go2.sport"),
            "unitree_sdk2py.go2.sport.sport_client": mock_sport_mod,
        }
        return modules, mock_channel_init, mock_sport_client

    def test_get_client_does_not_deadlock(self):
        """Calling get_client with uninitialized channel should not deadlock.

        Previously, get_client held self._lock and called _ensure_channel
        which also tried to acquire self._lock (non-reentrant). This would
        deadlock. The fix moves _ensure_channel before the lock.
        """
        mgr = _ConnectionManager()
        assert mgr._channel_initialized is False

        modules, mock_channel_init, mock_sport_client = self._mock_unitree_sdk()

        with patch.dict(sys.modules, modules):
            client = mgr.get_client("go2", "sport", "eth0", 0)

        assert mgr._channel_initialized is True
        mock_channel_init.assert_called_once_with(0, "eth0")
        assert client is not None

    def test_ensure_channel_idempotent(self):
        """Multiple _ensure_channel calls should only initialize once."""
        mgr = _ConnectionManager()

        modules, mock_channel_init, _ = self._mock_unitree_sdk()

        with patch.dict(sys.modules, modules):
            mgr._ensure_channel("eth0", 0)
            mgr._ensure_channel("eth0", 0)
            mgr._ensure_channel("wlan0", 1)

        # Only one call despite three invocations
        mock_channel_init.assert_called_once()

    def test_lock_not_held_during_ensure_channel(self):
        """Verify the lock is free when _ensure_channel runs."""
        mgr = _ConnectionManager()
        lock_was_free = []

        original_ensure = mgr._ensure_channel

        def spy_ensure(interface, domain_id):
            # If the lock were held, acquire(blocking=False) would return False
            acquired = mgr._lock.acquire(blocking=False)
            lock_was_free.append(acquired)
            if acquired:
                mgr._lock.release()
            original_ensure(interface, domain_id)

        modules, _, _ = self._mock_unitree_sdk()

        with patch.dict(sys.modules, modules):
            mgr._ensure_channel = spy_ensure
            mgr.get_client("go2", "sport", "eth0", 0)

        # _ensure_channel should have been called with lock free
        assert len(lock_was_free) == 1
        assert lock_was_free[0] is True, "Lock was held during _ensure_channel — deadlock risk!"


class TestGestureOverride:
    """Verify gesture ID takes precedence over user-supplied action_id."""

    @pytest.fixture(autouse=True)
    def _enable_mock(self):
        with patch.dict(os.environ, {"UNITREE_MOCK": "true"}):
            yield

    def test_gesture_id_wins_over_user_action_id(self):
        """Gesture name should determine action_id, not user params."""
        result = use_unitree(
            action="g1.arm.hug",
            parameters={"action_id": 999},
        )
        assert result["status"] == "success"
        # hug = 19, user tried 999, gesture should win
        assert "ExecuteAction" in result["content"][0]["text"]

    def test_gesture_preserves_non_conflicting_params(self):
        """Extra params should survive gesture resolution."""
        result = use_unitree(
            action="g1.arm.clap",
            parameters={"speed": 0.5},
        )
        assert result["status"] == "success"
