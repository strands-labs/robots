"""Tests for strands_robots/tools/reachy_mini_tool.py — Reachy Mini robot control tool.

Tests cover:
- Connection pool (Zenoh session management)
- Transport helpers (REST API, Zenoh pub/sub)
- Pose math (RPY to matrix)
- All 30+ actions (status, movement, sensors, camera, audio, motors, moves, expressions)
- Multi-robot support (different hosts/prefixes)
- Error handling (network failures, missing deps)
"""

import importlib
import socket
import sys
from unittest.mock import MagicMock, patch

import pytest

# Skip if tools PR (#11) not merged yet
try:
    import strands_robots.tools.reachy_mini_tool  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("Requires PR #11 (agent-tools)", allow_module_level=True)

# --- Guard: strands must be mockable ---
_mock_strands = MagicMock()
_mock_tool_decorator = lambda f: f  # noqa: E731
_mock_strands.tool = _mock_tool_decorator


@pytest.fixture(autouse=True)
def _mock_strands_import(monkeypatch):
    """Mock strands import so the module can load without strands-agents installed."""
    monkeypatch.setitem(sys.modules, "strands", _mock_strands)
    # Force reload to pick up mock
    mod_name = "strands_robots.tools.reachy_mini_tool"
    if mod_name in sys.modules:
        del sys.modules[mod_name]


@pytest.fixture
def rmt(_mock_strands_import):
    """Import the module fresh with mocked strands.

    Explicitly depends on _mock_strands_import to guarantee sys.modules["strands"]
    is patched before importlib.import_module runs. Without this dependency, pytest's
    DAG-based fixture resolution may evaluate rmt before the autouse fixture when rmt
    is pulled in as a transitive dependency (e.g. via _clear_sessions).
    """
    return importlib.import_module("strands_robots.tools.reachy_mini_tool")


@pytest.fixture(autouse=True)
def _clear_sessions(rmt):
    """Clear session pool before each test."""
    with rmt._SESSIONS_LOCK:
        rmt._SESSIONS.clear()
    yield
    with rmt._SESSIONS_LOCK:
        rmt._SESSIONS.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pose Math
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPoseMath:
    """Test RPY to 4x4 pose matrix conversion and identity."""

    def test_identity_pose(self, rmt):
        m = rmt._identity_pose()
        assert m == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def test_rpy_zero(self, rmt):
        m = rmt._rpy_to_pose(0, 0, 0, 0, 0, 0)
        # Should be close to identity
        for i in range(4):
            for j in range(4):
                expected = 1.0 if i == j else 0.0
                assert abs(m[i][j] - expected) < 1e-10

    def test_rpy_pitch_90(self, rmt):
        m = rmt._rpy_to_pose(90, 0, 0)
        # pitch=90 means sp=1, cp≈0
        assert abs(m[2][0] - (-1.0)) < 1e-10  # -sin(pitch)

    def test_rpy_yaw_90(self, rmt):
        m = rmt._rpy_to_pose(0, 0, 90)
        # yaw=90 means cy≈0, sy=1
        assert abs(m[0][0]) < 1e-10  # cy*cp ≈ 0
        assert abs(m[1][0] - 1.0) < 1e-10  # sy*cp ≈ 1

    def test_rpy_roll_90(self, rmt):
        m = rmt._rpy_to_pose(0, 90, 0)
        assert abs(m[2][1] - 1.0) < 1e-10  # cp*sr ≈ 1

    def test_rpy_with_translation(self, rmt):
        m = rmt._rpy_to_pose(0, 0, 0, 100, 200, 300)
        # Translation in meters (mm / 1000)
        assert abs(m[0][3] - 0.1) < 1e-10
        assert abs(m[1][3] - 0.2) < 1e-10
        assert abs(m[2][3] - 0.3) < 1e-10

    def test_rpy_combined(self, rmt):
        m = rmt._rpy_to_pose(30, 45, 60, 50, 75, 100)
        # Should be a valid 4x4 matrix
        assert len(m) == 4
        assert all(len(row) == 4 for row in m)
        assert m[3] == [0, 0, 0, 1]

    def test_rpy_negative_angles(self, rmt):
        m = rmt._rpy_to_pose(-45, -30, -60)
        assert len(m) == 4
        assert m[3] == [0, 0, 0, 1]

    def test_rpy_360_wrap(self, rmt):
        m0 = rmt._rpy_to_pose(0, 0, 0)
        m360 = rmt._rpy_to_pose(360, 360, 360)
        for i in range(3):
            for j in range(4):
                assert abs(m0[i][j] - m360[i][j]) < 1e-8


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Host Resolution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestResolveHost:
    def test_ip_passthrough(self, rmt):
        with patch("socket.gethostbyname", return_value="192.168.1.10"):
            assert rmt._resolve_host("192.168.1.10") == "192.168.1.10"

    def test_hostname_resolution(self, rmt):
        with patch("socket.gethostbyname", return_value="10.0.0.5"):
            assert rmt._resolve_host("reachy-mini.local") == "10.0.0.5"

    def test_unresolvable_fallback(self, rmt):
        with patch("socket.gethostbyname", side_effect=socket.gaierror):
            assert rmt._resolve_host("bad-host") == "bad-host"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Zenoh Session Pool
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestZenohSessionPool:
    def test_session_created_on_first_call(self, rmt):
        mock_zenoh = MagicMock()
        mock_session = MagicMock()
        mock_zenoh.open.return_value = mock_session
        mock_zenoh.Config.from_json5.return_value = MagicMock()

        with patch("socket.gethostbyname", return_value="10.0.0.1"):
            with patch.dict(sys.modules, {"zenoh": mock_zenoh}):
                session = rmt._get_zenoh_session("reachy.local", 7447)
                assert session is mock_session
                assert "10.0.0.1:7447" in rmt._SESSIONS

    def test_session_reused_on_second_call(self, rmt):
        mock_session = MagicMock()
        with patch("socket.gethostbyname", return_value="10.0.0.1"):
            with rmt._SESSIONS_LOCK:
                rmt._SESSIONS["10.0.0.1:7447"] = mock_session
            result = rmt._get_zenoh_session("reachy.local", 7447)
            assert result is mock_session

    def test_session_none_when_zenoh_missing(self, rmt):
        with patch("socket.gethostbyname", return_value="10.0.0.1"):
            with patch("importlib.import_module", side_effect=ImportError("no zenoh")):
                result = rmt._get_zenoh_session("reachy.local", 7447)
                assert result is None

    def test_session_none_on_connect_error(self, rmt):
        mock_zenoh = MagicMock()
        mock_zenoh.open.side_effect = RuntimeError("Connection refused")
        mock_zenoh.Config.from_json5.return_value = MagicMock()

        with patch("socket.gethostbyname", return_value="10.0.0.1"):
            with patch.dict(sys.modules, {"zenoh": mock_zenoh}):
                result = rmt._get_zenoh_session("reachy.local", 7447)
                assert result is None

    def test_session_fallback_config(self, rmt):
        """Test fallback when Config.from_json5 is not available."""
        mock_zenoh = MagicMock()
        mock_zenoh.Config.from_json5.side_effect = AttributeError("no from_json5")
        mock_config = MagicMock()
        mock_zenoh.Config.return_value = mock_config
        mock_session = MagicMock()
        mock_zenoh.open.return_value = mock_session

        with patch("socket.gethostbyname", return_value="10.0.0.2"):
            with patch.dict(sys.modules, {"zenoh": mock_zenoh}):
                session = rmt._get_zenoh_session("host2", 7447)
                assert session is mock_session
                mock_config.insert_json5.assert_called_once()

    def test_close_all_sessions(self, rmt):
        mock_s1 = MagicMock()
        mock_s2 = MagicMock()
        with rmt._SESSIONS_LOCK:
            rmt._SESSIONS["10.0.0.1:7447"] = mock_s1
            rmt._SESSIONS["10.0.0.2:7447"] = mock_s2

        rmt._close_all_sessions()

        mock_s1.close.assert_called_once()
        mock_s2.close.assert_called_once()
        assert len(rmt._SESSIONS) == 0

    def test_close_all_sessions_handles_errors(self, rmt):
        mock_s = MagicMock()
        mock_s.close.side_effect = RuntimeError("already closed")
        with rmt._SESSIONS_LOCK:
            rmt._SESSIONS["10.0.0.1:7447"] = mock_s

        # Should not raise
        rmt._close_all_sessions()
        assert len(rmt._SESSIONS) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Transport Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAPI:
    def test_api_get_success(self, rmt):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"state": "running"}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = rmt._api("10.0.0.1", 8000, "/api/daemon/status")
            assert result == {"state": "running"}

    def test_api_post_with_data(self, rmt):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ok": true}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = rmt._api("10.0.0.1", 8000, "/api/daemon/start", "POST", {"wake_up": True})
            assert result == {"ok": True}

    def test_api_http_error(self, rmt):
        import urllib.error

        error = urllib.error.HTTPError("url", 500, "err", {}, MagicMock(read=lambda: b"server error"))
        with patch("urllib.request.urlopen", side_effect=error):
            result = rmt._api("10.0.0.1", 8000, "/api/fail")
            assert "error" in result

    def test_api_connection_error(self, rmt):
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
            result = rmt._api("10.0.0.1", 8000, "/api/fail")
            assert "error" in result


class TestZenohTransport:
    def test_zenoh_put_success(self, rmt):
        mock_session = MagicMock()
        with patch.object(rmt, "_get_zenoh_session", return_value=mock_session):
            result = rmt._zenoh_put("host", "prefix", "topic", {"key": "val"})
            assert result == {"ok": True}
            mock_session.put.assert_called_once()

    def test_zenoh_put_no_session(self, rmt):
        with patch.object(rmt, "_get_zenoh_session", return_value=None):
            result = rmt._zenoh_put("host", "prefix", "topic", {"key": "val"})
            assert result == {"error": "zenoh unavailable"}

    def test_zenoh_put_error(self, rmt):
        mock_session = MagicMock()
        mock_session.put.side_effect = RuntimeError("network")
        with patch.object(rmt, "_get_zenoh_session", return_value=mock_session):
            result = rmt._zenoh_put("host", "prefix", "topic", {})
            assert "error" in result

    def test_zenoh_cmd_delegates(self, rmt):
        with patch.object(rmt, "_zenoh_put", return_value={"ok": True}) as mock_put:
            rmt._zenoh_cmd("host", "pfx", {"torque": True}, 7447)
            mock_put.assert_called_once_with("host", "pfx", "command", {"torque": True}, 7447)

    def test_zenoh_sub_collects_messages(self, rmt):
        mock_session = MagicMock()

        def fake_subscribe(topic_expr, handler):
            # Simulate receiving messages
            sample = MagicMock()
            sample.key_expr = "prefix/joint_positions"
            sample.payload.to_bytes.return_value = b'{"head_joint_positions": [0.1, 0.2, 0.3]}'
            handler(sample)
            return MagicMock()

        mock_session.declare_subscriber = fake_subscribe

        with patch.object(rmt, "_get_zenoh_session", return_value=mock_session):
            with patch("time.sleep"):
                msgs = rmt._zenoh_sub("host", "prefix", "joint_positions", 0.1)
                assert len(msgs) == 1
                assert "head_joint_positions" in msgs[0][1]

    def test_zenoh_sub_no_session(self, rmt):
        with patch.object(rmt, "_get_zenoh_session", return_value=None):
            msgs = rmt._zenoh_sub("host", "prefix", "topic", 0.1)
            assert msgs[0][0] == "error"

    def test_zenoh_sub_error(self, rmt):
        mock_session = MagicMock()
        mock_session.declare_subscriber.side_effect = RuntimeError("fail")
        with patch.object(rmt, "_get_zenoh_session", return_value=mock_session):
            msgs = rmt._zenoh_sub("host", "prefix", "topic", 0.1)
            assert msgs[0][0] == "error"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Actions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestStatusActions:
    def test_status(self, rmt):
        api_response = {
            "state": "running",
            "version": "1.2.3",
            "wlan_ip": "192.168.1.10",
            "backend_status": {
                "motor_control_mode": "stiff",
                "control_loop_stats": {"mean_control_loop_frequency": 100.5},
            },
        }
        with patch.object(rmt, "_api", return_value=api_response):
            result = rmt.reachy_mini(action="status", host="10.0.0.1")
            assert result["status"] == "success"
            assert "Reachy Mini" in result["content"][0]["text"]
            assert "100.5Hz" in result["content"][0]["text"]

    def test_state(self, rmt):
        joint_data = [
            ("pfx/joint_positions", {"head_joint_positions": [0.1, -0.2, 0.3], "antennas_joint_positions": [0.0, 0.0]})
        ]
        pose_data = [("pfx/head_pose", {"head_pose": [[1, 0, 0, 0]]})]
        imu_data = [("pfx/imu_data", {"accelerometer": [0.0, 0.0, 9.8], "temperature": 42})]

        with patch.object(rmt, "_zenoh_sub", side_effect=[joint_data, pose_data, imu_data]):
            result = rmt.reachy_mini(action="state", host="10.0.0.1")
            assert result["status"] == "success"
            text = result["content"][0]["text"]
            assert "Head:" in text
            assert "Antennas:" in text
            assert "Temp:" in text

    def test_state_with_errors(self, rmt):
        error_msg = [("error", {"error": "zenoh unavailable"})]
        with patch.object(rmt, "_zenoh_sub", return_value=error_msg):
            result = rmt.reachy_mini(action="state", host="10.0.0.1")
            assert result["status"] == "success"


class TestDaemonActions:
    def test_daemon_start(self, rmt):
        with patch.object(rmt, "_api", return_value={"ok": True}):
            result = rmt.reachy_mini(action="daemon_start")
            assert result["status"] == "success"
            assert "▶️" in result["content"][0]["text"]

    def test_daemon_stop(self, rmt):
        with patch.object(rmt, "_api", return_value={"ok": True}):
            result = rmt.reachy_mini(action="daemon_stop")
            assert result["status"] == "success"
            assert "⏹️" in result["content"][0]["text"]

    def test_daemon_restart(self, rmt):
        with patch.object(rmt, "_api", return_value={"ok": True}):
            result = rmt.reachy_mini(action="daemon_restart")
            assert result["status"] == "success"
            assert "🔄" in result["content"][0]["text"]


class TestMovementActions:
    def test_look(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            result = rmt.reachy_mini(action="look", pitch=-15, yaw=30)
            assert result["status"] == "success"
            assert "pitch=-15" in result["content"][0]["text"]
            assert "yaw=30" in result["content"][0]["text"]

    def test_goto_pose(self, rmt):
        with patch.object(rmt, "_api", return_value={"ok": True}):
            result = rmt.reachy_mini(action="goto_pose", pitch=10, roll=5, yaw=-20, x=50, y=0, z=100, duration=2.0)
            assert result["status"] == "success"
            assert "Goto" in result["content"][0]["text"]

    def test_antennas(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            result = rmt.reachy_mini(action="antennas", left_antenna=45, right_antenna=-45)
            assert result["status"] == "success"
            assert "L=45" in result["content"][0]["text"]
            assert "R=-45" in result["content"][0]["text"]

    def test_body(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            result = rmt.reachy_mini(action="body", body_yaw=30)
            assert result["status"] == "success"
            assert "Body=30" in result["content"][0]["text"]

    def test_auto_body_yaw_on(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            result = rmt.reachy_mini(action="auto_body_yaw", enabled=True)
            assert result["status"] == "success"
            assert "ON" in result["content"][0]["text"]

    def test_auto_body_yaw_off(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            result = rmt.reachy_mini(action="auto_body_yaw", enabled=False)
            assert result["status"] == "success"
            assert "OFF" in result["content"][0]["text"]


class TestGazeActions:
    def test_look_at_world(self, rmt):
        with patch.object(rmt, "_api", return_value={"ok": True}):
            result = rmt.reachy_mini(action="look_at_world", x=1.0, y=0.5, z=0.3, duration=1.5)
            assert result["status"] == "success"
            assert "World" in result["content"][0]["text"]

    def test_look_at_image(self, rmt):
        with patch.object(rmt, "_api", return_value={"ok": True}):
            result = rmt.reachy_mini(action="look_at_image", x=320, y=240, duration=0.5)
            assert result["status"] == "success"
            assert "Pixel" in result["content"][0]["text"]
            assert "320" in result["content"][0]["text"]


class TestSensorActions:
    def test_joints_success(self, rmt):
        joint_msg = [
            (
                "pfx/joint_positions",
                {
                    "head_joint_positions": [0.1, -0.2, 0.3],
                    "antennas_joint_positions": [0.0, 0.0],
                },
            )
        ]
        with patch.object(rmt, "_zenoh_sub", return_value=joint_msg):
            result = rmt.reachy_mini(action="joints")
            assert result["status"] == "success"
            assert "Head:" in result["content"][0]["text"]
            assert "json" in result["content"][1]

    def test_joints_no_data(self, rmt):
        with patch.object(rmt, "_zenoh_sub", return_value=[("error", {"error": "zenoh unavailable"})]):
            result = rmt.reachy_mini(action="joints")
            assert result["status"] == "error"
            assert "No joint data" in result["content"][0]["text"]

    def test_head_pose_success(self, rmt):
        pose_msg = [("pfx/head_pose", {"head_pose": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})]
        with patch.object(rmt, "_zenoh_sub", return_value=pose_msg):
            result = rmt.reachy_mini(action="head_pose")
            assert result["status"] == "success"

    def test_head_pose_no_data(self, rmt):
        with patch.object(rmt, "_zenoh_sub", return_value=[("error", {"error": "fail"})]):
            result = rmt.reachy_mini(action="head_pose")
            assert result["status"] == "error"

    def test_imu_success(self, rmt):
        imu_msg = [
            (
                "pfx/imu_data",
                {
                    "accelerometer": [0.0, 0.0, 9.81],
                    "gyroscope": [0.0, 0.0, 0.0],
                    "quaternion": [1, 0, 0, 0],
                    "temperature": 38.5,
                },
            )
        ]
        with patch.object(rmt, "_zenoh_sub", return_value=imu_msg):
            result = rmt.reachy_mini(action="imu")
            assert result["status"] == "success"
            assert "38.5" in result["content"][0]["text"]

    def test_imu_no_data(self, rmt):
        with patch.object(rmt, "_zenoh_sub", return_value=[("error", {"error": "fail"})]):
            result = rmt.reachy_mini(action="imu")
            assert result["status"] == "error"


class TestCameraAction:
    def test_camera_capture(self, rmt):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"\xff\xd8\xff\xe0JPEG_DATA"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = rmt.reachy_mini(action="camera", host="10.0.0.1")
            assert result["status"] == "success"
            assert "bytes" in result["content"][0]["text"]
            assert result["content"][1]["image"]["format"] == "jpeg"

    def test_camera_with_save(self, rmt, tmp_path):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"\xff\xd8\xff\xe0"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        save_file = str(tmp_path / "snap.jpg")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = rmt.reachy_mini(action="camera", save_path=save_file)
            assert result["status"] == "success"
            assert save_file in result["content"][0]["text"]

    def test_camera_error(self, rmt):
        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError("refused")):
            result = rmt.reachy_mini(action="camera")
            assert result["status"] == "error"


class TestAudioActions:
    def test_play_sound(self, rmt):
        with patch.object(rmt, "_api", return_value={"ok": True}):
            result = rmt.reachy_mini(action="play_sound", sound_file="beep.wav")
            assert result["status"] == "success"
            assert "🔊" in result["content"][0]["text"]

    def test_play_sound_missing_file(self, rmt):
        result = rmt.reachy_mini(action="play_sound")
        assert result["status"] == "error"
        assert "sound_file required" in result["content"][0]["text"]

    def test_record_audio(self, rmt):
        with patch.object(rmt, "_api", return_value={"ok": True}):
            result = rmt.reachy_mini(action="record_audio", duration=3.0)
            assert result["status"] == "success"
            assert "🎤" in result["content"][0]["text"]


class TestMotorActions:
    def test_enable_motors(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            result = rmt.reachy_mini(action="enable_motors")
            assert result["status"] == "success"
            assert "⚡" in result["content"][0]["text"]

    def test_enable_motors_specific(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            result = rmt.reachy_mini(action="enable_motors", motor_ids="head_pitch,head_yaw")
            assert result["status"] == "success"
            assert "head_pitch,head_yaw" in result["content"][0]["text"]

    def test_disable_motors(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            result = rmt.reachy_mini(action="disable_motors")
            assert result["status"] == "success"
            assert "💤" in result["content"][0]["text"]

    def test_gravity_compensation(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            result = rmt.reachy_mini(action="gravity_compensation")
            assert result["status"] == "success"
            assert "compliant" in result["content"][0]["text"]

    def test_stiff(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            result = rmt.reachy_mini(action="stiff")
            assert result["status"] == "success"
            assert "Stiff" in result["content"][0]["text"]

    def test_stop(self, rmt):
        with patch.object(rmt, "_api", return_value={"ok": True}):
            result = rmt.reachy_mini(action="stop")
            assert result["status"] == "success"
            assert "🛑" in result["content"][0]["text"]


class TestMoveLibraryActions:
    def test_list_moves_emotions(self, rmt):
        with patch.object(rmt, "_api", return_value=["happy", "sad", "wave"]):
            result = rmt.reachy_mini(action="list_moves", library="emotions")
            assert result["status"] == "success"
            assert "3" in result["content"][0]["text"]
            assert "happy" in result["content"][0]["text"]

    def test_list_moves_dance(self, rmt):
        with patch.object(rmt, "_api", return_value=["waltz", "disco"]):
            result = rmt.reachy_mini(action="list_moves", library="dance")
            assert result["status"] == "success"

    def test_list_moves_non_list_response(self, rmt):
        with patch.object(rmt, "_api", return_value={"error": "not found"}):
            result = rmt.reachy_mini(action="list_moves")
            assert result["status"] == "success"

    def test_play_move(self, rmt):
        with patch.object(rmt, "_api", return_value={"ok": True}):
            result = rmt.reachy_mini(action="play_move", move_name="wave", library="emotions")
            assert result["status"] == "success"
            assert "wave" in result["content"][0]["text"]

    def test_play_move_missing_name(self, rmt):
        result = rmt.reachy_mini(action="play_move")
        assert result["status"] == "error"
        assert "move_name required" in result["content"][0]["text"]


class TestRecordingActions:
    def test_start_recording(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            result = rmt.reachy_mini(action="start_recording")
            assert result["status"] == "success"
            assert "Recording started" in result["content"][0]["text"]

    def test_stop_recording(self, rmt):
        msgs = [("pfx/recorded_data", [{"frame": 1}, {"frame": 2}, {"frame": 3}])]
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            with patch.object(rmt, "_zenoh_sub", return_value=msgs):
                result = rmt.reachy_mini(action="stop_recording")
                assert result["status"] == "success"
                assert "3 frames" in result["content"][0]["text"]

    def test_stop_recording_with_save(self, rmt, tmp_path):
        msgs = [("pfx/recorded_data", [{"frame": 1}])]
        save_file = str(tmp_path / "rec.json")
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            with patch.object(rmt, "_zenoh_sub", return_value=msgs):
                result = rmt.reachy_mini(action="stop_recording", save_path=save_file)
                assert result["status"] == "success"
                assert save_file in result["content"][0]["text"]

    def test_stop_recording_no_data(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            with patch.object(rmt, "_zenoh_sub", return_value=[("error", {"error": "fail"})]):
                result = rmt.reachy_mini(action="stop_recording")
                assert result["status"] == "success"
                assert "0 frames" in result["content"][0]["text"]


class TestExpressionActions:
    def test_wake_up(self, rmt):
        with patch.object(rmt, "_api", return_value={"ok": True}):
            result = rmt.reachy_mini(action="wake_up")
            assert result["status"] == "success"
            assert "☀️" in result["content"][0]["text"]

    def test_sleep(self, rmt):
        with patch.object(rmt, "_api", return_value={"ok": True}):
            result = rmt.reachy_mini(action="sleep")
            assert result["status"] == "success"
            assert "😴" in result["content"][0]["text"]

    def test_nod(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            with patch("time.sleep"):
                result = rmt.reachy_mini(action="nod")
                assert result["status"] == "success"
                assert "nods" in result["content"][0]["text"]

    def test_shake(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            with patch("time.sleep"):
                result = rmt.reachy_mini(action="shake")
                assert result["status"] == "success"
                assert "shakes" in result["content"][0]["text"]

    def test_happy(self, rmt):
        with patch.object(rmt, "_zenoh_cmd", return_value={"ok": True}):
            with patch("time.sleep"):
                result = rmt.reachy_mini(action="happy")
                assert result["status"] == "success"
                assert "happy" in result["content"][0]["text"]


class TestUnknownAction:
    def test_unknown_action(self, rmt):
        result = rmt.reachy_mini(action="fly")
        assert result["status"] == "error"
        assert "Unknown: fly" in result["content"][0]["text"]
        # Validate it lists valid actions
        assert "status" in result["content"][0]["text"]
        assert "look" in result["content"][0]["text"]


class TestExceptionHandling:
    def test_general_exception(self, rmt):
        with patch.object(rmt, "_api", side_effect=RuntimeError("catastrophic")):
            result = rmt.reachy_mini(action="status")
            assert result["status"] == "error"
            assert "catastrophic" in result["content"][0]["text"]


class TestMultiRobot:
    """Test that different host/prefix combinations work independently."""

    def test_different_hosts(self, rmt):
        calls = []

        def mock_api(host, port, path, method="GET", data=None):
            calls.append((host, path))
            return {
                "state": "running",
                "version": "1.0",
                "wlan_ip": host,
                "backend_status": {
                    "motor_control_mode": "stiff",
                    "control_loop_stats": {"mean_control_loop_frequency": 100},
                },
            }

        with patch.object(rmt, "_api", side_effect=mock_api):
            r1 = rmt.reachy_mini(action="status", host="robot-1.local")
            r2 = rmt.reachy_mini(action="status", host="robot-2.local")

            assert r1["status"] == "success"
            assert r2["status"] == "success"
            assert calls[0][0] == "robot-1.local"
            assert calls[1][0] == "robot-2.local"

    def test_different_prefixes(self, rmt):
        calls = []

        def mock_cmd(host, prefix, cmd, zenoh_port=7447):
            calls.append(prefix)
            return {"ok": True}

        with patch.object(rmt, "_zenoh_cmd", side_effect=mock_cmd):
            rmt.reachy_mini(action="look", prefix="robot_a", pitch=10)
            rmt.reachy_mini(action="look", prefix="robot_b", pitch=-10)

            assert calls == ["robot_a", "robot_b"]

    def test_different_ports(self, rmt):
        calls = []

        def mock_cmd(host, prefix, cmd, zenoh_port=7447):
            calls.append(zenoh_port)
            return {"ok": True}

        with patch.object(rmt, "_zenoh_cmd", side_effect=mock_cmd):
            rmt.reachy_mini(action="look", zenoh_port=7447, pitch=10)
            rmt.reachy_mini(action="look", zenoh_port=7448, pitch=10)

            assert calls == [7447, 7448]
