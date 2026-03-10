"""Tests for strands_robots/policies/lerobot_async/__init__.py — gRPC async inference.

All gRPC, LeRobot, and torch dependencies are mocked. CPU-only.
Covers:
- _validate_policy_type()
- LerobotAsyncPolicy initialization and properties
- set_robot_state_keys()
- _ensure_connected()
- get_actions() (async)
- _build_raw_observation()
- _convert_actions()
- _generate_zero_actions()
- disconnect()

Fix (2026-03-03): Import module ONCE at module scope with cv2 mocked to prevent
OpenCV 4.12 DictValue / circular import crash triggered by strands_robots.__init__
eagerly importing lerobot_camera.py.
"""

import asyncio
import os
import pickle
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Mock strands if not installed ───
try:
    import strands

    HAS_STRANDS = hasattr(strands, "Agent")
except ImportError:
    import types as _types

    _mock_strands = _types.ModuleType("strands")
    _mock_strands.tool = lambda f: f
    sys.modules["strands"] = _mock_strands
    HAS_STRANDS = False


# ─── Pre-mock cv2 to prevent OpenCV 4.12 crash during strands_robots.__init__ ───
# strands_robots/__init__.py eagerly imports lerobot_camera.py which does `import cv2`.
# On OpenCV 4.12, this crashes with "cv2.dnn has no attribute DictValue" or circular
# import errors when cv2 is already partially loaded by another test.
_cv2_needs_mock = "cv2" not in sys.modules
if _cv2_needs_mock:
    import importlib.machinery as _im_fix

    _mock_cv2 = MagicMock()
    _mock_cv2.__spec__ = _im_fix.ModuleSpec("cv2", None)
    _mock_cv2.dnn = MagicMock()
    _mock_cv2.dnn.DictValue = MagicMock()
    sys.modules["cv2"] = _mock_cv2

# ─── Import the module ONCE at module scope, then reuse across all tests ───
# This avoids re-triggering the cv2 import chain on every test function.
# Remove any stale cached modules first.
for _key in list(sys.modules.keys()):
    if "strands_robots.policies.lerobot_async" in _key:
        del sys.modules[_key]

try:  # noqa: E402
    import strands_robots.policies.lerobot_async as _lerobot_async_mod

    LerobotAsyncPolicy = _lerobot_async_mod.LerobotAsyncPolicy
    _validate_policy_type = _lerobot_async_mod._validate_policy_type
except (ImportError, AttributeError):
    __import__("pytest").skip("Requires PR #8 (policy-abstraction)", allow_module_level=True)

# Restore cv2 if we mocked it
if _cv2_needs_mock and "cv2" in sys.modules and sys.modules["cv2"] is _mock_cv2:
    del sys.modules["cv2"]


# ─── Module-level picklable classes (needed for pickle.dumps in tests) ───


class _FakeRemotePolicyConfig:
    """Picklable stand-in for lerobot RemotePolicyConfig."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeTimedAction:
    """Picklable stand-in for lerobot TimedAction."""

    def __init__(self, values):
        self._values = values

    def get_action(self):
        return self._values


class _FakeTimedObservation:
    """Picklable stand-in for lerobot TimedObservation."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _build_lerobot_async_mocks():
    """Build a complete mock lerobot package hierarchy for _ensure_connected / get_actions.

    Uses real ModuleType for parent packages so that 'from X import Y' works
    correctly with sys.modules patching.
    """
    mocks = {}

    # Top-level lerobot package
    mock_lerobot = ModuleType("lerobot")
    mocks["lerobot"] = mock_lerobot

    # grpc
    mock_grpc = MagicMock()
    mock_channel = MagicMock()
    mock_grpc.insecure_channel.return_value = mock_channel
    mocks["grpc"] = mock_grpc

    # transport — parent must be ModuleType with attributes for 'from X import Y'
    mock_services_pb2 = MagicMock()
    mock_services_pb2_grpc = MagicMock()
    mock_stub = MagicMock()
    mock_services_pb2_grpc.AsyncInferenceStub.return_value = mock_stub

    mock_transport = ModuleType("lerobot.transport")
    mock_transport.services_pb2 = mock_services_pb2
    mock_transport.services_pb2_grpc = mock_services_pb2_grpc
    mocks["lerobot.transport"] = mock_transport
    mocks["lerobot.transport.services_pb2"] = mock_services_pb2
    mocks["lerobot.transport.services_pb2_grpc"] = mock_services_pb2_grpc

    # transport utils
    mock_transport_utils = MagicMock()
    mock_transport_utils.send_bytes_in_chunks.return_value = iter([b"chunk"])
    mocks["lerobot.transport.utils"] = mock_transport_utils

    # async_inference — parent must be ModuleType with helpers attribute
    mock_helpers = MagicMock()
    mock_helpers.RemotePolicyConfig = _FakeRemotePolicyConfig
    mock_helpers.TimedObservation = _FakeTimedObservation

    mock_async_inference = ModuleType("lerobot.async_inference")
    mock_async_inference.helpers = mock_helpers
    mocks["lerobot.async_inference"] = mock_async_inference
    mocks["lerobot.async_inference.helpers"] = mock_helpers

    return mocks, mock_grpc, mock_stub, mock_services_pb2


# ═════════════════════════════════════════════════════════════════════════════
# _validate_policy_type Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestValidatePolicyType:
    """Test policy type validation against LeRobot registry."""

    def test_valid_policy_type(self):
        mock_factory = MagicMock()
        mock_factory.get_policy_class = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.policies": MagicMock(),
                "lerobot.policies.factory": mock_factory,
            },
        ):
            _validate_policy_type("pi0")
            mock_factory.get_policy_class.assert_called_once_with("pi0")

    def test_lerobot_not_installed(self):
        # _validate_policy_type silently succeeds when lerobot is missing
        with patch.dict("sys.modules", {"lerobot.policies.factory": None}):
            _validate_policy_type("any_type")

    def test_lerobot_runtime_error(self):
        mock_factory = MagicMock()
        mock_factory.get_policy_class = MagicMock(side_effect=RuntimeError("torch not found"))

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.policies": MagicMock(),
                "lerobot.policies.factory": mock_factory,
            },
        ):
            _validate_policy_type("pi0")

    def test_invalid_policy_type_with_known_choices(self):
        mock_factory = MagicMock()
        mock_factory.get_policy_class = MagicMock(side_effect=ValueError("not found"))

        mock_config = MagicMock()
        mock_config.PreTrainedConfig.get_known_choices.return_value = [
            "act",
            "pi0",
            "diffusion",
        ]

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.policies": MagicMock(),
                "lerobot.policies.factory": mock_factory,
                "lerobot.configs": MagicMock(),
                "lerobot.configs.policies": mock_config,
            },
        ):
            with pytest.raises(ValueError, match="Unsupported policy type"):
                _validate_policy_type("nonexistent_policy")

    def test_invalid_policy_type_choices_unavailable(self):
        mock_factory = MagicMock()
        mock_factory.get_policy_class = MagicMock(side_effect=ValueError("not found"))

        mock_config = MagicMock()
        mock_config.PreTrainedConfig.get_known_choices.side_effect = Exception("no choices")

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.policies": MagicMock(),
                "lerobot.policies.factory": mock_factory,
                "lerobot.configs": MagicMock(),
                "lerobot.configs.policies": mock_config,
            },
        ):
            with pytest.raises(ValueError, match="Unsupported policy type"):
                _validate_policy_type("bad_type")


# ═════════════════════════════════════════════════════════════════════════════
# LerobotAsyncPolicy Initialization Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestLerobotAsyncPolicyInit:
    """Test LerobotAsyncPolicy initialization and properties."""

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_basic_init(self, mock_validate):
        policy = LerobotAsyncPolicy(
            server_address="localhost:8080",
            policy_type="pi0",
            pretrained_name_or_path="lerobot/pi0-so100-wipe",
            actions_per_chunk=10,
        )

        assert policy.server_address == "localhost:8080"
        assert policy.policy_type == "pi0"
        assert policy.pretrained_name_or_path == "lerobot/pi0-so100-wipe"
        assert policy.actions_per_chunk == 10
        assert policy.device == "cuda"
        assert policy.fps == 30
        assert policy.task == ""
        assert policy.lerobot_features == {}
        assert policy.rename_map == {}
        assert policy.robot_state_keys == []
        assert policy._connected is False
        assert policy._timestep == 0
        mock_validate.assert_called_once_with("pi0")

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_custom_init(self, mock_validate):
        features = {"observation.state": {"dim": 6}}
        rename = {"joint_pos": "observation.state"}

        policy = LerobotAsyncPolicy(
            server_address="gpu-server:9090",
            policy_type="act",
            pretrained_name_or_path="lerobot/act_aloha",
            actions_per_chunk=20,
            device="cpu",
            fps=60,
            task="pick and place",
            lerobot_features=features,
            rename_map=rename,
        )

        assert policy.device == "cpu"
        assert policy.fps == 60
        assert policy.task == "pick and place"
        assert policy.lerobot_features == features
        assert policy.rename_map == rename

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_provider_name(self, mock_validate):
        policy = LerobotAsyncPolicy()
        assert policy.provider_name == "lerobot_async"

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_set_robot_state_keys(self, mock_validate):
        policy = LerobotAsyncPolicy()
        keys = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ]
        policy.set_robot_state_keys(keys)
        assert policy.robot_state_keys == keys


# ═════════════════════════════════════════════════════════════════════════════
# _ensure_connected Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestEnsureConnected:
    """Test gRPC connection establishment."""

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_already_connected(self, mock_validate):
        policy = LerobotAsyncPolicy()
        policy._connected = True

        policy._ensure_connected()
        assert policy._connected is True

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_connect_success(self, mock_validate):
        mocks, mock_grpc, mock_stub, mock_services_pb2 = _build_lerobot_async_mocks()

        with patch.dict("sys.modules", mocks):
            policy = LerobotAsyncPolicy(server_address="localhost:8080", policy_type="pi0")
            policy._ensure_connected()

            assert policy._connected is True
            mock_grpc.insecure_channel.assert_called_once_with("localhost:8080")
            mock_stub.Ready.assert_called_once()
            mock_stub.SendPolicyInstructions.assert_called_once()

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_connect_grpc_not_installed(self, mock_validate):
        policy = LerobotAsyncPolicy()

        with patch.dict("sys.modules", {"grpc": None}):
            with pytest.raises(ImportError, match="dependencies not available"):
                policy._ensure_connected()

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_connect_server_unreachable(self, mock_validate):
        mocks, mock_grpc, mock_stub, mock_services_pb2 = _build_lerobot_async_mocks()
        mock_stub.Ready.side_effect = ConnectionError("Server unreachable")

        with patch.dict("sys.modules", mocks):
            policy = LerobotAsyncPolicy(server_address="bad-host:9999")
            with pytest.raises(ConnectionError):
                policy._ensure_connected()


# ═════════════════════════════════════════════════════════════════════════════
# _build_raw_observation Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestBuildRawObservation:
    """Test observation format conversion."""

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_builds_with_state_and_images(self, mock_validate):
        policy = LerobotAsyncPolicy()
        policy.robot_state_keys = ["joint_0", "joint_1"]

        observation = {
            "joint_0": np.array([0.5]),
            "joint_1": np.array([1.0]),
            "camera_front": np.zeros((480, 640, 3), dtype=np.uint8),
        }

        raw = policy._build_raw_observation(observation, "pick up the cube")

        assert "joint_0" in raw
        assert "joint_1" in raw
        assert "camera_front" in raw
        assert "task" in raw
        assert raw["task"] == "pick up the cube"

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_builds_without_instruction(self, mock_validate):
        policy = LerobotAsyncPolicy()
        policy.robot_state_keys = ["joint_0"]

        raw = policy._build_raw_observation({"joint_0": np.array([0.5])}, "")
        assert "task" not in raw

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_skips_non_image_non_state(self, mock_validate):
        policy = LerobotAsyncPolicy()
        policy.robot_state_keys = ["joint_0"]

        observation = {
            "joint_0": np.array([0.5]),
            "scalar_value": 42.0,
            "1d_array": np.array([1, 2, 3]),
        }

        raw = policy._build_raw_observation(observation, "")
        assert "joint_0" in raw
        assert "scalar_value" not in raw
        assert "1d_array" not in raw


# ═════════════════════════════════════════════════════════════════════════════
# _convert_actions Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestConvertActions:
    """Test TimedAction to action dict conversion."""

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_convert_with_numpy(self, mock_validate):
        policy = LerobotAsyncPolicy()
        policy.robot_state_keys = ["j0", "j1", "j2"]

        mock_action1 = MagicMock()
        mock_tensor1 = MagicMock()
        mock_tensor1.numpy.return_value = np.array([0.1, 0.2, 0.3])
        mock_action1.get_action.return_value = mock_tensor1

        mock_action2 = MagicMock()
        mock_tensor2 = MagicMock()
        mock_tensor2.numpy.return_value = np.array([0.4, 0.5, 0.6])
        mock_action2.get_action.return_value = mock_tensor2

        actions = policy._convert_actions([mock_action1, mock_action2])

        assert len(actions) == 2
        assert actions[0] == {
            "j0": pytest.approx(0.1),
            "j1": pytest.approx(0.2),
            "j2": pytest.approx(0.3),
        }
        assert actions[1] == {
            "j0": pytest.approx(0.4),
            "j1": pytest.approx(0.5),
            "j2": pytest.approx(0.6),
        }

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_convert_with_plain_list(self, mock_validate):
        policy = LerobotAsyncPolicy()
        policy.robot_state_keys = ["j0", "j1"]

        mock_action = MagicMock()
        mock_action.get_action.return_value = [1.5, 2.5]

        actions = policy._convert_actions([mock_action])

        assert len(actions) == 1
        assert actions[0]["j0"] == pytest.approx(1.5)
        assert actions[0]["j1"] == pytest.approx(2.5)

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_convert_pads_missing_keys(self, mock_validate):
        policy = LerobotAsyncPolicy()
        policy.robot_state_keys = ["j0", "j1", "j2", "j3"]

        mock_action = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.numpy.return_value = np.array([0.1, 0.2])
        mock_action.get_action.return_value = mock_tensor

        actions = policy._convert_actions([mock_action])

        assert actions[0]["j0"] == pytest.approx(0.1)
        assert actions[0]["j1"] == pytest.approx(0.2)
        assert actions[0]["j2"] == 0.0
        assert actions[0]["j3"] == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# _generate_zero_actions Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestGenerateZeroActions:
    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_generates_correct_count(self, mock_validate):
        policy = LerobotAsyncPolicy(actions_per_chunk=5)
        policy.robot_state_keys = ["j0", "j1"]

        zeros = policy._generate_zero_actions()
        assert len(zeros) == 5
        for action in zeros:
            assert action == {"j0": 0.0, "j1": 0.0}

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_empty_state_keys(self, mock_validate):
        policy = LerobotAsyncPolicy(actions_per_chunk=3)
        policy.robot_state_keys = []

        zeros = policy._generate_zero_actions()
        assert len(zeros) == 3
        assert all(z == {} for z in zeros)


# ═════════════════════════════════════════════════════════════════════════════
# get_actions Tests (async)
# ═════════════════════════════════════════════════════════════════════════════


class TestGetActions:
    """Test the async get_actions method."""

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_get_actions_success(self, mock_validate):
        policy = LerobotAsyncPolicy(server_address="localhost:8080")
        policy.robot_state_keys = ["j0", "j1"]
        policy._connected = True

        mocks, mock_grpc, mock_stub, mock_services_pb2 = _build_lerobot_async_mocks()

        # Build a picklable action response
        fake_actions = [_FakeTimedAction(np.array([0.5, 0.7]))]
        response_data = pickle.dumps(fake_actions)

        mock_response = MagicMock()
        mock_response.data = response_data
        mock_stub.GetActions.return_value = mock_response

        policy._stub = mock_stub

        with patch.dict("sys.modules", mocks):
            result = asyncio.run(
                policy.get_actions(
                    {"j0": np.array([0.1]), "j1": np.array([0.2])},
                    "pick up cube",
                )
            )

        assert len(result) == 1
        assert result[0]["j0"] == pytest.approx(0.5)
        assert result[0]["j1"] == pytest.approx(0.7)
        assert policy._timestep == 1

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_get_actions_empty_response(self, mock_validate):
        policy = LerobotAsyncPolicy(actions_per_chunk=3)
        policy.robot_state_keys = ["j0"]
        policy._connected = True

        mocks, mock_grpc, mock_stub, mock_services_pb2 = _build_lerobot_async_mocks()

        mock_response = MagicMock()
        mock_response.data = b""
        mock_stub.GetActions.return_value = mock_response

        policy._stub = mock_stub

        with patch.dict("sys.modules", mocks):
            result = asyncio.run(policy.get_actions({"j0": np.array([0.0])}, ""))

        assert len(result) == 3
        assert all(a["j0"] == 0.0 for a in result)

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_get_actions_exception_returns_zeros(self, mock_validate):
        policy = LerobotAsyncPolicy(actions_per_chunk=2)
        policy.robot_state_keys = ["j0"]
        policy._connected = True

        mocks, mock_grpc, mock_stub, mock_services_pb2 = _build_lerobot_async_mocks()

        mock_stub.GetActions.side_effect = RuntimeError("inference failed")
        policy._stub = mock_stub

        with patch.dict("sys.modules", mocks):
            result = asyncio.run(policy.get_actions({"j0": np.array([0.0])}, ""))

        assert len(result) == 2
        assert all(a["j0"] == 0.0 for a in result)


# ═════════════════════════════════════════════════════════════════════════════
# disconnect Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestDisconnect:
    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_disconnect_closes_channel(self, mock_validate):
        policy = LerobotAsyncPolicy()
        mock_channel = MagicMock()
        policy._channel = mock_channel
        policy._connected = True

        policy.disconnect()

        mock_channel.close.assert_called_once()
        assert policy._connected is False

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_disconnect_no_channel(self, mock_validate):
        policy = LerobotAsyncPolicy()
        policy._channel = None
        policy._connected = False

        policy.disconnect()
        assert policy._connected is False

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_del_calls_disconnect(self, mock_validate):
        policy = LerobotAsyncPolicy()
        mock_channel = MagicMock()
        policy._channel = mock_channel
        policy._connected = True

        policy.__del__()
        mock_channel.close.assert_called_once()

    @patch("strands_robots.policies.lerobot_async._validate_policy_type")
    def test_del_handles_exception(self, mock_validate):
        policy = LerobotAsyncPolicy()
        mock_channel = MagicMock()
        mock_channel.close.side_effect = RuntimeError("already closed")
        policy._channel = mock_channel
        policy._connected = True

        policy.__del__()


# ═════════════════════════════════════════════════════════════════════════════
# Module-level Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestModuleExports:
    def test_all_exports(self):
        assert "LerobotAsyncPolicy" in _lerobot_async_mod.__all__
