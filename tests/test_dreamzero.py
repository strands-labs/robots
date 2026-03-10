#!/usr/bin/env python3
"""Comprehensive mock-based tests for DreamzeroPolicy WebSocket client.

Tests every method, branch, and error path in policies/dreamzero/__init__.py
WITHOUT requiring a running DreamZero server, GPU, websockets, or msgpack.

Mocking strategy:
  - sys.modules patches for websockets.sync.client and msgpack_numpy
  - MagicMock chain: ws_root.sync.client IS ws_sync_client (Python import resolution)
  - PIL.Image mocked inline for resize tests only
"""

import asyncio
import sys
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Async helper — avoids deprecated asyncio.get_event_loop()
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run a coroutine synchronously, creating and closing its own event loop.

    This avoids the Python 3.10+ DeprecationWarning from get_event_loop()
    and is thread-safe (each call owns its own loop).
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Module-level mocking helpers
# ---------------------------------------------------------------------------


def _make_ws_modules():
    """Create a properly chained websockets mock for sys.modules.

    Python's `import websockets.sync.client` resolves through:
        sys.modules["websockets"].sync.client
    So we must wire the chain: ws_root.sync.client IS ws_sync_client.
    """
    ws_sync_client = MagicMock()
    ws_sync = MagicMock()
    ws_sync.client = ws_sync_client
    ws_root = MagicMock()
    ws_root.sync = ws_sync
    ws_root.sync.client = ws_sync_client
    return {
        "websockets": ws_root,
        "websockets.sync": ws_sync,
        "websockets.sync.client": ws_sync_client,
    }


def _make_msgpack_modules(*, with_openpi: bool = True):
    """Create msgpack_numpy mock modules.

    Args:
        with_openpi: If True, include openpi_client.msgpack_numpy in sys.modules.
    """
    mp = MagicMock()
    mp.unpackb = MagicMock()
    mp.Packer = MagicMock

    mods = {"msgpack_numpy": mp}
    if with_openpi:
        openpi = MagicMock()
        openpi.msgpack_numpy = mp
        mods["openpi_client"] = openpi
        mods["openpi_client.msgpack_numpy"] = mp
    return mods, mp


def _all_modules(*, with_openpi: bool = True):
    """Merge ws + msgpack modules for a single patch.dict."""
    ws = _make_ws_modules()
    mp_mods, mp_mock = _make_msgpack_modules(with_openpi=with_openpi)
    ws.update(mp_mods)
    return ws, mp_mock


def _server_config(
    image_resolution=(180, 320),
    n_external_cameras=2,
    needs_wrist_camera=True,
):
    """Default server config dict returned on connect."""
    return {
        "image_resolution": image_resolution,
        "n_external_cameras": n_external_cameras,
        "needs_wrist_camera": needs_wrist_camera,
    }


def _import_policy(modules_dict):
    """Import DreamzeroPolicy under mocked modules, returning the class."""
    sys.modules.pop("strands_robots.policies.dreamzero", None)
    with patch.dict(sys.modules, modules_dict):
        from strands_robots.policies.dreamzero import DreamzeroPolicy

        return DreamzeroPolicy


def _connected_policy_from_mods(mods, mp, **kwargs):
    """Create a fully connected DreamzeroPolicy with WebSocket mock.

    Returns (policy, mock_ws) so callers can configure additional recv().
    """
    from strands_robots.policies.dreamzero import DreamzeroPolicy

    mock_ws = MagicMock()
    mock_ws.recv.return_value = b"config"
    mods["websockets.sync.client"].connect.return_value = mock_ws
    mp.unpackb = MagicMock(return_value=_server_config())

    p = DreamzeroPolicy(**kwargs)
    p._ensure_connected()
    return p, mock_ws


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_module_cache():
    """Remove cached dreamzero module between tests."""
    yield
    sys.modules.pop("strands_robots.policies.dreamzero", None)


@pytest.fixture
def modules_and_mp():
    """Provide all mocked modules + the msgpack mock."""
    mods, mp = _all_modules()
    return mods, mp


@pytest.fixture
def policy_cls(modules_and_mp):
    """Import and return DreamzeroPolicy class under mocks."""
    mods, _ = modules_and_mp
    return _import_policy(mods)


# ---------------------------------------------------------------------------
# TestInit — Constructor & properties
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self, policy_cls):
        p = policy_cls()
        assert p._host == "localhost"
        assert p._port == 8000
        assert p._instruction == ""
        assert p._image_resize is None
        assert p._action_horizon == 24
        assert p._ws is None
        assert p._connected is False
        assert p._step == 0
        assert len(p._session_id) == 36  # UUID format

    def test_custom_params(self, policy_cls):
        p = policy_cls(
            host="gpu-server",
            port=9000,
            instruction="pick cube",
            session_id="my-session",
            image_resize=(240, 320),
            action_horizon=16,
        )
        assert p._host == "gpu-server"
        assert p._port == 9000
        assert p._instruction == "pick cube"
        assert p._session_id == "my-session"
        assert p._image_resize == (240, 320)
        assert p._action_horizon == 16

    def test_provider_name(self, policy_cls):
        p = policy_cls()
        assert p.provider_name == "dreamzero"

    def test_set_robot_state_keys(self, policy_cls):
        p = policy_cls()
        keys = ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"]
        p.set_robot_state_keys(keys)
        assert p._robot_state_keys == keys

    def test_is_policy_subclass(self, modules_and_mp):
        """DreamzeroPolicy must inherit from the Policy ABC."""
        mods, _ = modules_and_mp
        with patch.dict(sys.modules, mods):
            sys.modules.pop("strands_robots.policies.dreamzero", None)
            from strands_robots.policies import Policy
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            assert issubclass(DreamzeroPolicy, Policy)

    def test_unique_session_ids(self, policy_cls):
        """Two independently constructed policies get different session IDs."""
        p1 = policy_cls()
        p2 = policy_cls()
        assert p1._session_id != p2._session_id

    def test_accepts_extra_kwargs(self, policy_cls):
        """Constructor should accept arbitrary **kwargs without error (API stability)."""
        p = policy_cls(host="h", port=1234, some_future_kwarg="value")
        assert p._host == "h"


# ---------------------------------------------------------------------------
# TestEnsureConnected — WebSocket connection + config handshake
# ---------------------------------------------------------------------------


class TestEnsureConnected:
    def test_connects_via_ws(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            mock_ws = MagicMock()
            mock_ws.recv.return_value = b"config"
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(return_value=_server_config())

            p = DreamzeroPolicy(host="server1", port=7777)
            p._ensure_connected()

            assert p._connected is True
            assert p._ws is mock_ws
            assert p._server_config == _server_config()
            mods["websockets.sync.client"].connect.assert_called_once_with(
                "ws://server1:7777",
                compression=None,
                max_size=None,
                ping_interval=60,
                ping_timeout=600,
            )

    def test_noop_if_already_connected(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            p = DreamzeroPolicy()
            p._connected = True
            p._ensure_connected()
            mods["websockets.sync.client"].connect.assert_not_called()

    def test_idempotent_connection(self, modules_and_mp):
        """Calling _ensure_connected() twice does NOT create a second WebSocket."""
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            mock_ws = MagicMock()
            mock_ws.recv.return_value = b"config"
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(return_value=_server_config())

            p = DreamzeroPolicy()
            p._ensure_connected()
            p._ensure_connected()

            # connect called only once
            mods["websockets.sync.client"].connect.assert_called_once()

    def test_connection_refused_raises(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            mods["websockets.sync.client"].connect.side_effect = ConnectionRefusedError()

            p = DreamzeroPolicy()
            with pytest.raises(ConnectionError, match="DreamZero server not running"):
                p._ensure_connected()

    def test_ws_fails_then_wss_succeeds(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            mock_ws = MagicMock()
            mock_ws.recv.return_value = b"config"
            mp.unpackb = MagicMock(return_value=_server_config())

            mods["websockets.sync.client"].connect.side_effect = [
                OSError("ws failed"),
                mock_ws,
            ]

            p = DreamzeroPolicy()
            p._ensure_connected()

            assert p._connected is True
            assert mods["websockets.sync.client"].connect.call_count == 2
            calls = mods["websockets.sync.client"].connect.call_args_list
            assert "ws://" in calls[0][0][0]
            assert "wss://" in calls[1][0][0]

    def test_ws_and_wss_both_fail(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            mods["websockets.sync.client"].connect.side_effect = [
                OSError("ws failed"),
                OSError("wss failed"),
            ]

            p = DreamzeroPolicy()
            with pytest.raises(ConnectionError, match="Cannot connect"):
                p._ensure_connected()

    def test_websockets_import_error(self):
        """When websockets is not installed, get a clear ImportError."""
        mp_mods, mp = _make_msgpack_modules()
        blocked = {
            "websockets": None,
            "websockets.sync": None,
            "websockets.sync.client": None,
        }
        blocked.update(mp_mods)

        sys.modules.pop("strands_robots.policies.dreamzero", None)
        with patch.dict(sys.modules, blocked):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            p = DreamzeroPolicy()
            with pytest.raises(ImportError, match="websockets"):
                p._ensure_connected()

    def test_msgpack_import_fallback_bare_module(self):
        """When openpi_client is not available, fall back to bare msgpack_numpy."""
        ws_mods = _make_ws_modules()
        mp_mods, mp = _make_msgpack_modules(with_openpi=False)
        all_mods = {}
        all_mods.update(ws_mods)
        all_mods.update(mp_mods)
        all_mods["openpi_client"] = None

        mock_ws = MagicMock()
        mock_ws.recv.return_value = b"config"
        mp.unpackb = MagicMock(return_value=_server_config())

        sys.modules.pop("strands_robots.policies.dreamzero", None)
        with patch.dict(sys.modules, all_mods):
            ws_mods["websockets.sync.client"].connect.return_value = mock_ws

            from strands_robots.policies.dreamzero import DreamzeroPolicy

            p = DreamzeroPolicy()
            p._ensure_connected()
            assert p._connected is True


# ---------------------------------------------------------------------------
# TestBuildObservation — Camera mapping, image resize, joint extraction
# ---------------------------------------------------------------------------


class TestBuildObservation:
    def _connected_policy(self, policy_cls, server_config=None):
        """Create a policy that appears already connected with a given config."""
        p = policy_cls()
        p._connected = True
        p._server_config = server_config or _server_config()
        return p

    def test_basic_camera_mapping(self, policy_cls):
        p = self._connected_policy(policy_cls)
        obs = {
            "camera_left": np.zeros((180, 320, 3), dtype=np.uint8),
            "camera_right": np.zeros((180, 320, 3), dtype=np.uint8),
        }
        result = p._build_observation(obs, "pick cube")

        assert "observation/exterior_image_0_left" in result
        assert "observation/exterior_image_1_left" in result
        assert "observation/wrist_image_left" in result
        assert result["prompt"] == "pick cube"
        assert result["endpoint"] == "infer"
        assert result["session_id"] == p._session_id

    def test_wrist_camera_detected(self, policy_cls):
        p = self._connected_policy(policy_cls)
        obs = {
            "wrist_camera": np.ones((180, 320, 3), dtype=np.uint8) * 128,
            "camera_ext": np.zeros((180, 320, 3), dtype=np.uint8),
        }
        result = p._build_observation(obs, "task")

        wrist = result["observation/wrist_image_left"]
        assert wrist.mean() == 128
        ext0 = result["observation/exterior_image_0_left"]
        assert ext0.mean() == 0

    def test_excess_cameras_ignored(self, policy_cls):
        """More cameras than n_external_cameras → extras are dropped."""
        cfg = _server_config(n_external_cameras=1)
        p = self._connected_policy(policy_cls, cfg)
        obs = {
            "camera_a": np.zeros((180, 320, 3), dtype=np.uint8),
            "camera_b": np.ones((180, 320, 3), dtype=np.uint8) * 200,
            "camera_c": np.ones((180, 320, 3), dtype=np.uint8) * 100,
        }
        result = p._build_observation(obs, "task")

        assert "observation/exterior_image_0_left" in result
        assert "observation/exterior_image_1_left" not in result

    def test_missing_cameras_filled_with_zeros(self, policy_cls):
        """When no camera images provided, all slots filled with zeros."""
        p = self._connected_policy(policy_cls)
        result = p._build_observation({}, "task")

        for i in range(2):
            img = result[f"observation/exterior_image_{i}_left"]
            assert img.shape == (180, 320, 3)
            assert img.sum() == 0
        assert result["observation/wrist_image_left"].sum() == 0

    def test_wrist_disabled_by_server_config(self, policy_cls):
        cfg = _server_config(needs_wrist_camera=False)
        p = self._connected_policy(policy_cls, cfg)
        result = p._build_observation({}, "task")
        assert "observation/wrist_image_left" not in result

    def test_image_resize_with_pil(self, policy_cls):
        """When image dimensions don't match server config, PIL resize is attempted."""
        p = self._connected_policy(policy_cls)
        wrong_size = np.ones((100, 200, 3), dtype=np.uint8) * 42
        correct_size = np.ones((180, 320, 3), dtype=np.uint8) * 42

        mock_img_instance = MagicMock()
        mock_img_instance.resize.return_value = mock_img_instance
        mock_image_mod = MagicMock()
        mock_image_mod.fromarray.return_value = mock_img_instance

        with patch.dict(sys.modules, {"PIL": MagicMock(Image=mock_image_mod), "PIL.Image": mock_image_mod}):
            with patch("numpy.array", side_effect=lambda x: correct_size if x is mock_img_instance else np.asarray(x)):
                obs = {"camera_ext": wrong_size}
                p._build_observation(obs, "task")
                mock_image_mod.fromarray.assert_called_once()
                mock_img_instance.resize.assert_called_once_with((320, 180))

    def test_image_resize_pil_unavailable_uses_original(self, policy_cls):
        """When PIL fails to import, image is used as-is (wrong size)."""
        p = self._connected_policy(policy_cls)
        wrong_size = np.ones((100, 200, 3), dtype=np.uint8) * 42

        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            obs = {"camera_ext": wrong_size}
            result = p._build_observation(obs, "task")
            ext = result["observation/exterior_image_0_left"]
            assert ext.shape == (100, 200, 3)

    def test_non_ndarray_camera_skipped(self, policy_cls):
        """Non-ndarray camera values are skipped."""
        p = self._connected_policy(policy_cls)
        obs = {
            "camera_string": "not_an_image",
            "image_none": None,
        }
        result = p._build_observation(obs, "task")
        assert result["observation/exterior_image_0_left"].sum() == 0
        assert result["observation/exterior_image_1_left"].sum() == 0

    def test_joint_position_from_observation(self, policy_cls):
        p = self._connected_policy(policy_cls)
        obs = {
            "joint_position": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        }
        result = p._build_observation(obs, "task")
        jp = result["observation/joint_position"]
        np.testing.assert_allclose(jp, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    def test_default_instruction_used_when_empty(self, policy_cls):
        """When instruction is empty, falls back to self._instruction."""
        p = self._connected_policy(policy_cls)
        p._instruction = "default pick"
        result = p._build_observation({}, "")
        assert result["prompt"] == "default pick"

    def test_no_server_config_defaults(self, policy_cls):
        """When server_config is None, use hardcoded defaults."""
        p = policy_cls()
        p._connected = True
        p._server_config = None
        result = p._build_observation({}, "task")
        assert result["observation/exterior_image_0_left"].shape == (180, 320, 3)

    def test_custom_image_resize_overrides_server(self, policy_cls):
        """Policy-level image_resize overrides server config."""
        p = policy_cls(image_resize=(240, 320))
        p._connected = True
        p._server_config = _server_config(image_resolution=(180, 320))
        result = p._build_observation({}, "task")
        assert result["observation/exterior_image_0_left"].shape == (240, 320, 3)

    def test_prompt_and_session_metadata(self, policy_cls):
        """Observation includes prompt, endpoint, and session_id metadata."""
        p = self._connected_policy(policy_cls)
        result = p._build_observation({}, "grasp the bottle")

        assert result["prompt"] == "grasp the bottle"
        assert result["endpoint"] == "infer"
        assert result["session_id"] == p._session_id


# ---------------------------------------------------------------------------
# TestExtractJointState — Key matching, type coercion, robot_state_keys
# ---------------------------------------------------------------------------


class TestExtractJointState:
    def _policy(self, policy_cls, robot_state_keys=None):
        p = policy_cls()
        if robot_state_keys:
            p.set_robot_state_keys(robot_state_keys)
        return p

    def test_ndarray_match(self, policy_cls):
        p = self._policy(policy_cls)
        obs = {"joint_position": np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)}
        result = p._extract_joint_state(obs, "joint_position", 7)
        assert result.dtype == np.float32
        assert len(result) == 7
        np.testing.assert_allclose(result, [1, 2, 3, 4, 5, 6, 7])

    def test_list_match(self, policy_cls):
        p = self._policy(policy_cls)
        obs = {"my_joint_position_data": [0.1, 0.2, 0.3]}
        result = p._extract_joint_state(obs, "joint_position", 3)
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3], atol=1e-6)

    def test_tuple_match(self, policy_cls):
        p = self._policy(policy_cls)
        obs = {"joint_position": (1.0, 2.0)}
        result = p._extract_joint_state(obs, "joint_position", 2)
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_scalar_match(self, policy_cls):
        p = self._policy(policy_cls)
        obs = {"gripper_position": 0.75}
        result = p._extract_joint_state(obs, "gripper_position", 1)
        assert result.shape == (1,)
        np.testing.assert_allclose(result, [0.75])

    def test_int_scalar(self, policy_cls):
        p = self._policy(policy_cls)
        obs = {"gripper_position": 1}
        result = p._extract_joint_state(obs, "gripper_position", 1)
        np.testing.assert_allclose(result, [1.0])

    def test_no_match_returns_zeros(self, policy_cls):
        p = self._policy(policy_cls)
        obs = {"unrelated_key": 42}
        result = p._extract_joint_state(obs, "joint_position", 7)
        assert result.sum() == 0
        assert result.shape == (7,)

    def test_robot_state_keys_fallback(self, policy_cls):
        p = self._policy(policy_cls, ["j0", "j1", "j2", "j3", "j4", "j5", "j6"])
        obs = {"j0": 0.1, "j1": 0.2, "j2": 0.3, "j3": 0.4, "j4": 0.5, "j5": 0.6, "j6": 0.7}
        result = p._extract_joint_state(obs, "joint_position", 7)
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], atol=1e-6)

    def test_robot_state_keys_padded(self, policy_cls):
        """When robot state keys give fewer values than expected, pad with zeros."""
        p = self._policy(policy_cls, ["j0", "j1"])
        obs = {"j0": 1.0, "j1": 2.0}
        result = p._extract_joint_state(obs, "joint_position", 7)
        np.testing.assert_allclose(result, [1.0, 2.0, 0, 0, 0, 0, 0])

    def test_robot_state_keys_not_used_for_non_joint(self, policy_cls):
        """Robot state keys only activate for 'joint' hints."""
        p = self._policy(policy_cls, ["j0", "j1"])
        obs = {"j0": 1.0, "j1": 2.0}
        result = p._extract_joint_state(obs, "cartesian_position", 6)
        assert result.sum() == 0

    def test_robot_state_keys_fewer_than_action_dim(self, policy_cls):
        """When fewer robot_state_keys than expected_dim, pad remaining with zeros."""
        p = self._policy(policy_cls, ["j0", "j1", "j2"])
        obs = {"j0": 0.5, "j1": 0.6, "j2": 0.7}
        result = p._extract_joint_state(obs, "joint_position", 7)
        np.testing.assert_allclose(result, [0.5, 0.6, 0.7, 0, 0, 0, 0])

    def test_unrecognized_value_type_falls_through(self, policy_cls):
        """Dict or other unrecognized types in observation fall through."""
        p = self._policy(policy_cls)
        obs = {"joint_position": {"nested": "dict"}}
        result = p._extract_joint_state(obs, "joint_position", 7)
        assert result.sum() == 0

    def test_unrecognized_type_then_valid_match(self, policy_cls):
        """First match has dict type (falls through), second has valid ndarray."""
        p = self._policy(policy_cls)
        obs = {
            "a_joint_position_bad": {"nested": True},
            "b_joint_position_good": np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32),
        }
        result = p._extract_joint_state(obs, "joint_position", 7)
        np.testing.assert_allclose(result, [1, 2, 3, 4, 5, 6, 7])


# ---------------------------------------------------------------------------
# TestGetActions — Full send/recv cycle with action decoding
# ---------------------------------------------------------------------------


class TestGetActions:
    """Test get_actions() with full WebSocket mock.

    Key: patch.dict(sys.modules) must be active during the await,
    because get_actions() does `from openpi_client import msgpack_numpy`
    at call time.
    """

    def test_get_actions_generic_keys(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            actions_array = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] * 3)
            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"actions"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(side_effect=[_server_config(), actions_array])

            p = DreamzeroPolicy()
            p._ensure_connected()

            obs = {"camera_left": np.zeros((180, 320, 3), dtype=np.uint8)}
            actions = _run_async(p.get_actions(obs, "pick cube"))

            assert len(actions) == 3
            assert "joint_0" in actions[0]
            assert "gripper" in actions[0]
            assert actions[0]["gripper"] == pytest.approx(0.8)
            assert p._step == 1

    def test_get_actions_with_robot_state_keys(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            actions_array = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] * 2)
            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"actions"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(side_effect=[_server_config(), actions_array])

            keys = ["j0", "j1", "j2", "j3", "j4", "j5", "j6"]
            p = DreamzeroPolicy()
            p.set_robot_state_keys(keys)
            p._ensure_connected()

            actions = _run_async(p.get_actions({}, "pick"))

            assert "j0" in actions[0]
            assert "j6" in actions[0]
            assert "gripper" in actions[0]
            assert actions[0]["j0"] == pytest.approx(0.1)
            assert actions[0]["j6"] == pytest.approx(0.7)

    def test_get_actions_ensures_connected(self, modules_and_mp):
        """get_actions() should trigger lazy connection via _ensure_connected."""
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            actions_array = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"actions"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(side_effect=[_server_config(), actions_array])

            p = DreamzeroPolicy()
            assert p._connected is False

            # get_actions should connect automatically
            actions = _run_async(p.get_actions({}, "pick"))
            assert p._connected is True
            assert len(actions) == 1

    def test_get_actions_values_are_floats(self, modules_and_mp):
        """Returned action values must be Python floats (not numpy types)."""
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            actions_array = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"actions"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(side_effect=[_server_config(), actions_array])

            p = DreamzeroPolicy()
            p._ensure_connected()

            actions = _run_async(p.get_actions({}, "pick"))
            for key, value in actions[0].items():
                assert isinstance(value, float), f"{key} is {type(value)}, expected float"

    def test_get_actions_sends_packed_observation(self, modules_and_mp):
        """get_actions() should pack the observation and send it over the WebSocket."""
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            actions_array = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"actions"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(side_effect=[_server_config(), actions_array])

            p = DreamzeroPolicy()
            p._ensure_connected()

            obs = {"camera_left": np.zeros((180, 320, 3), dtype=np.uint8)}
            _run_async(p.get_actions(obs, "pick"))

            # WebSocket.send should have been called (once for actions, after config recv)
            assert mock_ws.send.call_count >= 1

    def test_get_actions_server_error_string(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", "Server OOM error"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(return_value=_server_config())

            p = DreamzeroPolicy()
            p._ensure_connected()

            with pytest.raises(RuntimeError, match="DreamZero server error"):
                _run_async(p.get_actions({}, "pick"))

    def test_get_actions_non_ndarray_response(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"bad"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(side_effect=[_server_config(), {"not": "an array"}])

            p = DreamzeroPolicy()
            p._ensure_connected()

            with pytest.raises(ValueError, match="Expected numpy array"):
                _run_async(p.get_actions({}, "pick"))

    def test_get_actions_robot_state_keys_exceed_action_dims(self, modules_and_mp):
        """When robot_state_keys has more entries than action vector dimensions."""
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            small_actions = np.array([[0.1, 0.2, 0.3, 0.9]])
            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"actions"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(side_effect=[_server_config(), small_actions])

            p = DreamzeroPolicy()
            p.set_robot_state_keys([f"j{i}" for i in range(10)])
            p._ensure_connected()

            actions = _run_async(p.get_actions({}, "pick"))
            # Only j0, j1, j2 should be in output (3 = 4-1 gripper)
            assert "j0" in actions[0]
            assert "j1" in actions[0]
            assert "j2" in actions[0]
            assert "j3" not in actions[0]
            assert "gripper" in actions[0]
            assert actions[0]["gripper"] == pytest.approx(0.9)

    def test_step_counter_increments(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            actions_array = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"a1", b"a2"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(side_effect=[_server_config(), actions_array, actions_array])

            p = DreamzeroPolicy()
            p._ensure_connected()

            _run_async(p.get_actions({}, "step1"))
            assert p._step == 1
            _run_async(p.get_actions({}, "step2"))
            assert p._step == 2

    def test_single_dim_action_vector(self, modules_and_mp):
        """Action vector with just 1 element."""
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            single_action = np.array([[0.5]])
            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"data"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(side_effect=[_server_config(), single_action])

            p = DreamzeroPolicy()
            p._ensure_connected()

            actions = _run_async(p.get_actions({}, "task"))
            assert len(actions) == 1
            assert "gripper" in actions[0]
            assert actions[0]["gripper"] == pytest.approx(0.5)

    def test_large_action_horizon(self, modules_and_mp):
        """100 action steps in one response."""
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            large_actions = np.random.rand(100, 8).astype(np.float32)
            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"data"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(side_effect=[_server_config(), large_actions])

            p = DreamzeroPolicy()
            p._ensure_connected()

            actions = _run_async(p.get_actions({}, "task"))
            assert len(actions) == 100
            for a in actions:
                assert len(a) == 8  # 7 joints + gripper


# ---------------------------------------------------------------------------
# TestReset — Session reset protocol
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_sends_message(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"reset successful"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(return_value=_server_config())

            p = DreamzeroPolicy()
            p._ensure_connected()

            old_session = p._session_id
            p._step = 10
            p.reset()

            assert p._step == 0
            assert p._session_id != old_session
            assert len(p._session_id) == 36

    def test_reset_noop_when_not_connected(self, policy_cls):
        p = policy_cls()
        p.reset()
        assert p._step == 0

    def test_reset_noop_when_ws_is_none(self, policy_cls):
        p = policy_cls()
        p._connected = True
        p._ws = None
        p.reset()

    def test_multiple_reset_cycles(self, modules_and_mp):
        """Multiple reset cycles produce unique session IDs each time."""
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"reset1", b"reset2", b"reset3"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(return_value=_server_config())

            p = DreamzeroPolicy()
            p._ensure_connected()

            session_ids = set()
            session_ids.add(p._session_id)
            for _ in range(3):
                p._step = 5
                p.reset()
                session_ids.add(p._session_id)
                assert p._step == 0

            # All 4 session IDs (initial + 3 resets) should be unique
            assert len(session_ids) == 4

    def test_reset_msgpack_fallback_to_bare_module(self):
        """When openpi_client is unavailable during reset, fall back to bare msgpack_numpy."""
        ws_mods = _make_ws_modules()
        mp_mods, mp = _make_msgpack_modules(with_openpi=False)
        all_mods = {}
        all_mods.update(ws_mods)
        all_mods.update(mp_mods)
        all_mods["openpi_client"] = None

        mock_ws = MagicMock()
        mock_ws.recv.side_effect = [b"config", b"reset ok"]
        ws_mods["websockets.sync.client"].connect.return_value = mock_ws
        mp.unpackb = MagicMock(return_value=_server_config())

        sys.modules.pop("strands_robots.policies.dreamzero", None)
        with patch.dict(sys.modules, all_mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            p = DreamzeroPolicy()
            p._ensure_connected()
            p._step = 5
            p.reset()
            assert p._step == 0


# ---------------------------------------------------------------------------
# TestClose — Connection cleanup
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_cleans_up(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            mock_ws = MagicMock()
            mock_ws.recv.return_value = b"config"
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(return_value=_server_config())

            p = DreamzeroPolicy()
            p._ensure_connected()
            assert p._connected is True

            p.close()
            assert p._ws is None
            assert p._connected is False
            mock_ws.close.assert_called_once()

    def test_close_noop_when_no_ws(self, policy_cls):
        p = policy_cls()
        p.close()
        assert p._connected is False

    def test_close_handles_exception(self, modules_and_mp):
        """close() should not raise even if ws.close() fails."""
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            mock_ws = MagicMock()
            mock_ws.close.side_effect = RuntimeError("close failed")
            mock_ws.recv.return_value = b"config"
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(return_value=_server_config())

            p = DreamzeroPolicy()
            p._ensure_connected()
            p.close()
            assert p._ws is None

    def test_del_calls_close(self, modules_and_mp):
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            mock_ws = MagicMock()
            mock_ws.recv.return_value = b"config"
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(return_value=_server_config())

            p = DreamzeroPolicy()
            p._ensure_connected()
            p.__del__()
            assert p._ws is None
            assert p._connected is False


# ---------------------------------------------------------------------------
# TestAliases — DreamZeroPolicy alias & __all__
# ---------------------------------------------------------------------------


class TestAliases:
    def test_dreamzeropolicy_alias(self, modules_and_mp):
        mods, _ = modules_and_mp
        with patch.dict(sys.modules, mods):
            sys.modules.pop("strands_robots.policies.dreamzero", None)
            from strands_robots.policies.dreamzero import DreamZeroPolicy, DreamzeroPolicy

            assert DreamZeroPolicy is DreamzeroPolicy

    def test_all_exports(self, modules_and_mp):
        mods, _ = modules_and_mp
        with patch.dict(sys.modules, mods):
            sys.modules.pop("strands_robots.policies.dreamzero", None)
            from strands_robots.policies import dreamzero

            assert "DreamzeroPolicy" in dreamzero.__all__
            assert "DreamZeroPolicy" in dreamzero.__all__


# ---------------------------------------------------------------------------
# TestRegistration — PolicyRegistry integration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_create_policy_dreamzero(self, modules_and_mp):
        """create_policy('dreamzero') should return a DreamzeroPolicy instance."""
        mods, _ = modules_and_mp
        with patch.dict(sys.modules, mods):
            sys.modules.pop("strands_robots.policies.dreamzero", None)
            from strands_robots.policies import create_policy

            p = create_policy("dreamzero", host="myhost", port=9999)
            assert p.provider_name == "dreamzero"
            assert p._host == "myhost"
            assert p._port == 9999

    def test_create_policy_alias_dream_zero(self, modules_and_mp):
        """create_policy('dream_zero') should resolve via alias."""
        mods, _ = modules_and_mp
        with patch.dict(sys.modules, mods):
            sys.modules.pop("strands_robots.policies.dreamzero", None)
            from strands_robots.policies import create_policy

            p = create_policy("dream_zero")
            assert p.provider_name == "dreamzero"

    def test_create_policy_alias_world_action_model(self, modules_and_mp):
        """create_policy('world_action_model') should resolve via alias."""
        mods, _ = modules_and_mp
        with patch.dict(sys.modules, mods):
            sys.modules.pop("strands_robots.policies.dreamzero", None)
            from strands_robots.policies import create_policy

            p = create_policy("world_action_model")
            assert p.provider_name == "dreamzero"

    def test_provider_name_matches_registration(self, modules_and_mp):
        """provider_name from create_policy must match the canonical name."""
        mods, _ = modules_and_mp
        with patch.dict(sys.modules, mods):
            sys.modules.pop("strands_robots.policies.dreamzero", None)
            from strands_robots.policies import create_policy

            for alias in ("dreamzero", "dream_zero", "world_action_model"):
                p = create_policy(alias)
                assert p.provider_name == "dreamzero", f"Alias '{alias}' → provider_name={p.provider_name}"


# ---------------------------------------------------------------------------
# TestLifecycle — Full connect→infer→reset→close flow
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_full_lifecycle(self, modules_and_mp):
        """Test the complete lifecycle: connect → get_actions → reset → close."""
        mods, mp = modules_and_mp
        with patch.dict(sys.modules, mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            actions_array = np.array(
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    [0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81],
                ]
            )

            mock_ws = MagicMock()
            mock_ws.recv.side_effect = [b"config", b"actions", b"reset ok"]
            mods["websockets.sync.client"].connect.return_value = mock_ws
            mp.unpackb = MagicMock(side_effect=[_server_config(), actions_array])

            p = DreamzeroPolicy(
                host="gpu-server",
                port=8000,
                instruction="pick up the red cube",
            )
            p.set_robot_state_keys(["j0", "j1", "j2", "j3", "j4", "j5", "gripper"])

            # 1. Connect + get_actions
            obs = {
                "camera_left": np.zeros((180, 320, 3), dtype=np.uint8),
                "camera_right": np.zeros((180, 320, 3), dtype=np.uint8),
            }
            actions = _run_async(p.get_actions(obs, "pick up the red cube"))
            assert p._connected is True
            assert len(actions) == 2
            assert actions[0]["j0"] == pytest.approx(0.1)
            assert actions[0]["gripper"] == pytest.approx(0.8)
            assert p._step == 1

            # 2. Reset
            old_session = p._session_id
            p.reset()
            assert p._step == 0
            assert p._session_id != old_session

            # 3. Close
            p.close()
            assert p._ws is None
            assert p._connected is False


# ---------------------------------------------------------------------------
# TestEdgeCases — Boundary conditions
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_observation_dict(self, policy_cls):
        """Completely empty observation should produce valid (zero) outputs."""
        p = policy_cls()
        p._connected = True
        p._server_config = _server_config()
        result = p._build_observation({}, "task")

        assert result["observation/joint_position"].sum() == 0
        assert result["observation/cartesian_position"].sum() == 0
        assert result["observation/gripper_position"].sum() == 0
        assert result["prompt"] == "task"

    def test_session_id_is_uuid(self, policy_cls):
        """Auto-generated session_id should be a valid UUID."""
        p = policy_cls()
        parsed = uuid.UUID(p._session_id)
        assert str(parsed) == p._session_id

    def test_get_actions_msgpack_fallback_no_openpi(self):
        """get_actions() falls back to bare msgpack_numpy when openpi_client missing."""
        ws_mods = _make_ws_modules()
        mp_mods, mp = _make_msgpack_modules(with_openpi=False)
        all_mods = {}
        all_mods.update(ws_mods)
        all_mods.update(mp_mods)
        all_mods["openpi_client"] = None

        mock_ws = MagicMock()
        actions_array = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        mock_ws.recv.side_effect = [b"config", b"actions"]
        ws_mods["websockets.sync.client"].connect.return_value = mock_ws
        mp.unpackb = MagicMock(side_effect=[_server_config(), actions_array])

        sys.modules.pop("strands_robots.policies.dreamzero", None)
        with patch.dict(sys.modules, all_mods):
            from strands_robots.policies.dreamzero import DreamzeroPolicy

            p = DreamzeroPolicy()
            p._ensure_connected()

            actions = _run_async(p.get_actions({}, "pick"))
            assert len(actions) == 1
            assert actions[0]["gripper"] == pytest.approx(0.8)
