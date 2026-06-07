"""Tests for camera schema in mesh presence + per-camera publish loop."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_robot_with_camera():
    """A duck-typed robot with one camera in its inner config."""
    inner = SimpleNamespace(
        is_connected=True,
        name="so101_test",
        config=SimpleNamespace(cameras={"wrist": {"index": 0}}),
        get_observation=MagicMock(return_value={"wrist": _make_frame()}),
    )
    return SimpleNamespace(
        tool_name_str="so101",
        robot=inner,
    )


def _make_frame(h=4, w=4, c=3):
    import numpy as np

    return np.arange(h * w * c, dtype=np.uint8).reshape(h, w, c)


def test_presence_includes_cameras(fake_robot_with_camera):
    """Cameras list is advertised in the presence payload."""
    from strands_robots.mesh import Mesh

    m = Mesh(fake_robot_with_camera, peer_id="test-cam-1", peer_type="robot")
    payload = m._build_presence()

    assert payload["robot_id"] == "test-cam-1"
    assert payload["cameras"] == ["wrist"]


def test_presence_no_cameras_when_inner_has_none():
    """No 'cameras' key when the inner robot has no cameras configured."""
    from strands_robots.mesh import Mesh

    inner = SimpleNamespace(
        is_connected=True,
        name="so101_test",
        config=SimpleNamespace(cameras={}),
    )
    r = SimpleNamespace(tool_name_str="so101", robot=inner)
    m = Mesh(r, peer_id="test-cam-2", peer_type="robot")
    payload = m._build_presence()
    assert "cameras" not in payload


def test_resolve_camera_hz_default_disabled(fake_robot_with_camera, monkeypatch):
    """STRANDS_MESH_CAMERA_HZ unset → loop disabled (returns 0)."""
    from strands_robots.mesh import Mesh

    monkeypatch.delenv("STRANDS_MESH_CAMERA_HZ", raising=False)
    m = Mesh(fake_robot_with_camera, peer_id="test-cam-3")
    assert m._resolve_camera_hz() == 0.0


def test_resolve_camera_hz_from_env(fake_robot_with_camera, monkeypatch):
    """Valid env value enables the loop."""
    from strands_robots.mesh import Mesh

    monkeypatch.setenv("STRANDS_MESH_CAMERA_HZ", "5")
    m = Mesh(fake_robot_with_camera, peer_id="test-cam-4")
    assert m._resolve_camera_hz() == 5.0


def test_resolve_camera_hz_invalid_disables(fake_robot_with_camera, monkeypatch):
    """Garbage env value → loop disabled, no exception."""
    from strands_robots.mesh import Mesh

    monkeypatch.setenv("STRANDS_MESH_CAMERA_HZ", "not-a-number")
    m = Mesh(fake_robot_with_camera, peer_id="test-cam-5")
    assert m._resolve_camera_hz() == 0.0


def test_publish_cameras_once_calls_put(fake_robot_with_camera):
    """One frame is read per camera and forwarded via mesh_session.put.

    Note: outgoing camera frames are wrapped in a signed envelope; we
    unwrap them here so the rest of the assertions stay readable.
    """
    from strands_robots.mesh import Mesh

    m = Mesh(fake_robot_with_camera, peer_id="test-cam-6")
    with patch("strands_robots.mesh.core.put") as mock_put:
        m._publish_cameras_once()

    assert mock_put.called, "put() should have been called for the camera"
    topic, payload = mock_put.call_args[0]
    assert topic == "strands/test-cam-6/camera/wrist"
    assert payload["peer_id"] == "test-cam-6"
    assert payload["cam"] == "wrist"
    assert payload["shape"] == [4, 4, 3]
    assert payload["dtype"] == "uint8"
    assert payload["encoding"] in ("jpeg", "raw")
    assert isinstance(payload["data"], str) and len(payload["data"]) > 0


def test_publish_cameras_once_skips_disconnected():
    """No-op when the inner robot is disconnected."""
    from strands_robots.mesh import Mesh

    inner = SimpleNamespace(
        is_connected=False,
        config=SimpleNamespace(cameras={"wrist": {}}),
        get_observation=MagicMock(),
    )
    r = SimpleNamespace(tool_name_str="so101", robot=inner)
    m = Mesh(r, peer_id="test-cam-7")
    with patch("strands_robots.mesh.core.put") as mock_put:
        m._publish_cameras_once()
    assert not mock_put.called
    assert not inner.get_observation.called


def test_publish_cameras_once_handles_missing_frame(fake_robot_with_camera):
    """A camera that returns None is skipped silently."""
    from strands_robots.mesh import Mesh

    fake_robot_with_camera.robot.get_observation = MagicMock(return_value={"wrist": None})
    m = Mesh(fake_robot_with_camera, peer_id="test-cam-8")
    with patch("strands_robots.mesh.core.put") as mock_put:
        m._publish_cameras_once()
    assert not mock_put.called


def test_publish_cameras_once_kill_switch_blocks_publish(fake_robot_with_camera, monkeypatch):
    """STRANDS_MESH_CAMERA_DISABLED=true short-circuits before any publish.

    Pins the privacy kill-switch gate at the top of _publish_cameras_once:
    when the env var is truthy, no frame is collected and put() is never
    called, even though the inner robot is connected and has a camera.
    """
    from strands_robots.mesh import Mesh

    monkeypatch.setenv("STRANDS_MESH_CAMERA_DISABLED", "true")
    m = Mesh(fake_robot_with_camera, peer_id="test-cam-killswitch")
    with patch("strands_robots.mesh.core.put") as mock_put:
        m._publish_cameras_once()
    assert not mock_put.called
    assert not fake_robot_with_camera.robot.get_observation.called


def test_publish_cameras_once_publishes_when_kill_switch_unset(fake_robot_with_camera, monkeypatch):
    """With the kill-switch unset, the camera frame is published normally.

    Companion to test_publish_cameras_once_kill_switch_blocks_publish: proves
    the gate is the reason for the no-op above, not an unrelated short-circuit.
    """
    from strands_robots.mesh import Mesh

    monkeypatch.delenv("STRANDS_MESH_CAMERA_DISABLED", raising=False)
    m = Mesh(fake_robot_with_camera, peer_id="test-cam-killswitch-off")
    with patch("strands_robots.mesh.core.put") as mock_put:
        m._publish_cameras_once()
    assert mock_put.called


def test_publish_cameras_once_kill_switch_lenient_truthy(fake_robot_with_camera, monkeypatch):
    """Lenient truthy values (on/1/yes) also engage the kill-switch."""
    from strands_robots.mesh import Mesh

    for raw in ("on", "1", "YES", "True"):
        monkeypatch.setenv("STRANDS_MESH_CAMERA_DISABLED", raw)
        m = Mesh(fake_robot_with_camera, peer_id="test-cam-killswitch-lenient")
        with patch("strands_robots.mesh.core.put") as mock_put:
            m._publish_cameras_once()
        assert not mock_put.called, f"value {raw!r} should disable publishing"
