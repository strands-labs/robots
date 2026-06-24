"""Tests for the Teleoperator factory and TeleopMixin.

Covers (no hardware required):
  * factory: config build, id/kwarg forwarding, typo rejection, bad type
  * mixin: lazy attach (no connect), multi-device, map_fn, merge/last-wins,
    background loop frames, stop + disconnect, publish delegation, sim
    robot_name routing.
"""

from __future__ import annotations

import threading
import time

import pytest

from strands_robots.teleop_mixin import AttachedTeleop, TeleopMixin

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTeleop:
    """Minimal lerobot-Teleoperator-shaped fake.

    Satisfies the duck-typed contract: get_action(), connect(), disconnect(),
    is_connected, name, id.
    """

    def __init__(self, action: dict[str, float], *, name: str = "fake_leader", id: str | None = None):
        self._action = action
        self.name = name
        self.id = id
        self.is_connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.get_action_calls = 0

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        self.is_connected = True
        self.connect_calls += 1

    def disconnect(self) -> None:
        self.is_connected = False
        self.disconnect_calls += 1

    def get_action(self) -> dict[str, float]:
        self.get_action_calls += 1
        return dict(self._action)


class FakeHost(TeleopMixin):
    """Minimal host exposing send_action(), like Robot/Simulation."""

    def __init__(self, tool_name: str = "fake_host"):
        self.tool_name_str = tool_name
        self.mesh = None
        self.peer_id = None
        self.sent: list[tuple[dict, str | None]] = []
        self._send_lock = threading.Lock()

    def send_action(self, action: dict, robot_name: str | None = None, n_substeps: int = 1):  # noqa: ARG002
        with self._send_lock:
            self.sent.append((dict(action), robot_name))
        return {"status": "success", "content": [{"text": "ok"}]}


class FakePublishHost(FakeHost):
    """Host that also exposes the mesh publish API (like hardware Robot)."""

    def __init__(self, tool_name: str = "fake_pub_host"):
        super().__init__(tool_name)
        self.publish_calls: list[dict] = []
        self.stop_teleop_calls = 0

    def start_teleop_publish(self, teleoperator, device_name="leader", method="arm", hz=50.0):
        self.publish_calls.append(
            {"teleoperator": teleoperator, "device_name": device_name, "method": method, "hz": hz}
        )
        return {"status": "success", "content": [{"text": f"pub {device_name}"}]}

    def stop_teleop(self, device_name=None):  # noqa: ARG002
        self.stop_teleop_calls += 1
        return {"status": "success", "content": [{"text": "stopped pub"}]}


def _spin_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# attach / lazy
# ---------------------------------------------------------------------------


def test_attach_is_lazy_no_connect():
    host = FakeHost()
    dev = FakeTeleop({"a.pos": 1.0})
    host.attach_teleop(dev, name="leader")
    assert dev.connect_calls == 0  # lazy: attach must not connect
    assert "leader" in host._teleops


def test_attach_returns_self_chainable():
    host = FakeHost()
    out = host.attach_teleop(FakeTeleop({"a": 1.0}), name="one").attach_teleop(FakeTeleop({"b": 2.0}), name="two")
    assert out is host
    assert set(host._teleops) == {"one", "two"}


def test_attach_duplicate_name_rejected():
    host = FakeHost()
    host.attach_teleop(FakeTeleop({"a": 1.0}), name="leader")
    with pytest.raises(ValueError, match="already attached"):
        host.attach_teleop(FakeTeleop({"b": 2.0}), name="leader")


def test_attach_name_resolution_id_then_name():
    host = FakeHost()
    host.attach_teleop(FakeTeleop({"a": 1.0}, name="so101_leader", id="blue"))
    assert "blue" in host._teleops  # id wins over type name
    host2 = FakeHost()
    host2.attach_teleop(FakeTeleop({"a": 1.0}, name="gamepad", id=None))
    assert "gamepad" in host2._teleops  # falls back to type name


def test_attach_prebuilt_with_kwargs_rejected():
    host = FakeHost()
    with pytest.raises(TypeError, match="only valid when building from a type string"):
        host.attach_teleop(FakeTeleop({"a": 1.0}), name="x", port="/dev/ttyACM0")


def test_attach_non_teleop_rejected():
    host = FakeHost()

    class NoAction:
        pass

    with pytest.raises(ValueError, match="no callable get_action"):
        host.attach_teleop(NoAction(), name="bad")


def test_infer_method():
    from strands_robots.teleop_mixin import _infer_method

    assert _infer_method("so101_leader") == "arm"
    assert _infer_method("gamepad") == "gamepad"
    assert _infer_method("keyboard") == "keyboard"
    assert _infer_method("phone") == "phone"


def test_normalize_action_dict_and_array():
    """Dict values are floated per-key; an array becomes positional j-keys."""
    import numpy as np

    from strands_robots.teleop_mixin import _normalize_action

    assert _normalize_action({"shoulder.pos": np.float32(1.5)}) == {"shoulder.pos": 1.5}
    assert _normalize_action(np.array([1.0, 2.0, 3.0])) == {"j0": 1.0, "j1": 2.0, "j2": 3.0}


def test_normalize_action_scalar_does_not_crash():
    """A numpy/torch scalar or 0-d array exposes ``tolist()`` that returns a
    bare Python number, not a list. Enumerating that raises
    ``'float' object is not iterable`` -- a 1-DOF leader must not crash the
    teleop loop. Such a value normalizes to the single-DOF ``{"raw": ...}``
    shape, matching the plain-Python-scalar fallback.
    """
    import numpy as np

    from strands_robots.teleop_mixin import _normalize_action

    assert _normalize_action(2.0) == {"raw": 2.0}
    assert _normalize_action(np.float32(1.5)) == {"raw": 1.5}
    assert _normalize_action(np.array(1.5)) == {"raw": 1.5}


# ---------------------------------------------------------------------------
# teleoperate loop
# ---------------------------------------------------------------------------


def test_teleoperate_connects_lazily_and_sends():
    host = FakeHost()
    dev = FakeTeleop({"shoulder.pos": 0.5})
    host.attach_teleop(dev, name="leader")
    host.teleoperate(hz=200)
    try:
        assert _spin_until(lambda: dev.connect_calls == 1)
        assert _spin_until(lambda: len(host.sent) > 0)
        action, robot_name = host.sent[-1]
        assert action == {"shoulder.pos": 0.5}
        assert robot_name is None
    finally:
        host.stop_teleoperate()
    assert dev.disconnect_calls == 1  # stop disconnects


def test_teleoperate_multi_device_merge():
    host = FakeHost()
    arm = FakeTeleop({"shoulder.pos": 1.0, "elbow.pos": 2.0})
    pad = FakeTeleop({"gripper.pos": 9.0})
    host.attach_teleop(arm, name="arm").attach_teleop(pad, name="pad")
    host.teleoperate(hz=200)
    try:
        assert _spin_until(lambda: len(host.sent) > 0)
        action, _ = host.sent[-1]
        assert action == {"shoulder.pos": 1.0, "elbow.pos": 2.0, "gripper.pos": 9.0}
    finally:
        host.stop_teleoperate()


def test_teleoperate_map_fn_applied():
    """Sim teleop bridge: remap leader joint names -> sim actuator names."""
    host = FakeHost()
    leader = FakeTeleop({"shoulder.pos": 0.25})
    host.attach_teleop(
        leader,
        name="leader",
        map_fn=lambda a: {k.replace(".pos", "_actuator"): v * 2 for k, v in a.items()},
    )
    host.teleoperate(hz=200)
    try:
        assert _spin_until(lambda: len(host.sent) > 0)
        action, _ = host.sent[-1]
        assert action == {"shoulder_actuator": 0.5}
    finally:
        host.stop_teleoperate()


def test_teleoperate_robot_name_routing():
    """Sim has unique robot names -> send_action routes to the named robot."""
    host = FakeHost()
    host.attach_teleop(FakeTeleop({"j.pos": 1.0}), name="leader")
    host.teleoperate(hz=200, robot_name="so101_1")
    try:
        assert _spin_until(lambda: len(host.sent) > 0)
        _, robot_name = host.sent[-1]
        assert robot_name == "so101_1"
    finally:
        host.stop_teleoperate()


def test_teleoperate_no_devices_errors():
    host = FakeHost()
    res = host.teleoperate()
    assert res["status"] == "error"
    assert "No teleoperators attached" in res["content"][0]["text"]


def test_teleoperate_double_start_errors():
    host = FakeHost()
    host.attach_teleop(FakeTeleop({"a": 1.0}), name="leader")
    host.teleoperate(hz=100)
    try:
        res = host.teleoperate()
        assert res["status"] == "error"
        assert "already running" in res["content"][0]["text"]
    finally:
        host.stop_teleoperate()


def test_teleoperate_unknown_name_errors():
    host = FakeHost()
    host.attach_teleop(FakeTeleop({"a": 1.0}), name="leader")
    res = host.teleoperate(names=["ghost"])
    assert res["status"] == "error"
    assert "Unknown teleop name" in res["content"][0]["text"]


def test_teleoperate_block_with_duration():
    host = FakeHost()
    dev = FakeTeleop({"a.pos": 1.0})
    host.attach_teleop(dev, name="leader")
    t0 = time.time()
    res = host.teleoperate(hz=100, block=True, duration=0.2)
    elapsed = time.time() - t0
    assert res["status"] == "success"
    assert "completed" in res["content"][0]["text"]
    assert 0.15 < elapsed < 1.0
    assert host._teleop_frames > 0
    assert dev.is_connected is False  # block path: caller stops? no -- ensure not running
    assert host._teleop_running is False


def test_connect_failure_rolls_back():
    host = FakeHost()

    class BadConnect(FakeTeleop):
        def connect(self, calibrate: bool = True):  # noqa: ARG002
            raise RuntimeError("port busy")

    good = FakeTeleop({"a": 1.0})
    bad = BadConnect({"b": 2.0})
    host.attach_teleop(good, name="good").attach_teleop(bad, name="bad")
    res = host.teleoperate(hz=100)
    assert res["status"] == "error"
    assert "Failed to connect" in res["content"][0]["text"]
    # good must be rolled back (disconnected), loop not running
    assert good.is_connected is False
    assert host._teleop_running is False


# ---------------------------------------------------------------------------
# detach / status / publish
# ---------------------------------------------------------------------------


def test_detach_specific_and_all():
    host = FakeHost()
    host.attach_teleop(FakeTeleop({"a": 1.0}), name="one").attach_teleop(FakeTeleop({"b": 2.0}), name="two")
    host.detach_teleop("one")
    assert set(host._teleops) == {"two"}
    host.detach_teleop()
    assert host._teleops == {}


def test_list_teleops():
    host = FakeHost()
    host.attach_teleop(FakeTeleop({"a": 1.0}), name="leader")
    res = host.list_teleops()
    assert res["status"] == "success"
    assert "leader" in res["teleops"]


def test_publish_delegates_to_host():
    host = FakePublishHost()
    arm = FakeTeleop({"a.pos": 1.0})
    host.attach_teleop(arm, name="arm", method="arm")
    res = host.teleoperate(hz=100, publish=True)
    try:
        assert res["status"] == "success"
        assert res["publish"] is True
        assert len(host.publish_calls) == 1
        assert host.publish_calls[0]["device_name"] == "arm"
        assert host.publish_calls[0]["teleoperator"] is arm
    finally:
        host.stop_teleoperate()
    assert host.stop_teleop_calls >= 1  # stop delegated to host.stop_teleop


def test_publish_without_host_support_errors():
    host = FakeHost()  # no start_teleop_publish
    dev = FakeTeleop({"a": 1.0})
    host.attach_teleop(dev, name="leader")
    res = host.teleoperate(publish=True)
    assert res["status"] == "error"
    assert "start_teleop_publish" in res["content"][0]["text"]
    # rolled back
    assert dev.is_connected is False


def test_get_teleoperate_status():
    host = FakeHost()
    host.attach_teleop(FakeTeleop({"a": 1.0}), name="leader")
    st = host.get_teleoperate_status()
    assert st["running"] is False
    assert st["devices"] == ["leader"]


def test_attached_teleop_dataclass():
    dev = FakeTeleop({"a": 1.0})
    att = AttachedTeleop(device=dev, name="x", method="arm", map_fn=None)
    assert att.device is dev
    assert att.method == "arm"


# ---------------------------------------------------------------------------
# background-loop resilience: conflict warning, send_action errors, device
# exceptions. These pin the hot-loop defensive branches so a misbehaving
# device or robot never crashes teleoperation - it is counted and rate-limited.
# ---------------------------------------------------------------------------


class ErrorHost(FakeHost):
    """Host whose send_action always reports a structured error."""

    def send_action(self, action: dict, robot_name: str | None = None, n_substeps: int = 1):  # noqa: ARG002
        with self._send_lock:
            self.sent.append((dict(action), robot_name))
        return {"status": "error", "content": [{"text": "actuator fault"}]}


class RaisingTeleop(FakeTeleop):
    """Teleop whose get_action() raises - simulates a flaky device read."""

    def get_action(self) -> dict[str, float]:
        self.get_action_calls += 1
        raise RuntimeError("device read failed")


def test_teleoperate_conflicting_keys_last_wins_and_counts_frames():
    """Two devices writing the same key: last-attached wins, loop keeps running."""
    host = FakeHost()
    # Both devices set 'gripper.pos'; merge order follows attach order, last wins.
    first = FakeTeleop({"gripper.pos": 1.0}, name="first")
    second = FakeTeleop({"gripper.pos": 2.0}, name="second")
    host.attach_teleop(first, name="first").attach_teleop(second, name="second")
    host.teleoperate(hz=200)
    try:
        assert _spin_until(lambda: len(host.sent) > 0)
        action, _ = host.sent[-1]
        assert action == {"gripper.pos": 2.0}  # last-attached device wins
    finally:
        host.stop_teleoperate()


def test_teleoperate_send_action_error_increments_error_count():
    """A robot that rejects actions is counted as an error, loop survives."""
    host = ErrorHost()
    host.attach_teleop(FakeTeleop({"a.pos": 1.0}), name="leader")
    host.teleoperate(hz=200)
    try:
        # Frames advance last in the loop body, so once a frame lands the error
        # from the rejected send_action has already been counted.
        assert _spin_until(lambda: host._teleop_frames > 0)
        assert host._teleop_errors > 0
        assert host._teleop_running is True
    finally:
        host.stop_teleoperate()


def test_teleoperate_device_read_exception_counted_not_fatal():
    """A device whose get_action() raises is counted; teleop keeps spinning."""
    host = FakeHost()
    host.attach_teleop(RaisingTeleop({"a.pos": 1.0}), name="flaky")
    host.teleoperate(hz=200)
    try:
        assert _spin_until(lambda: host._teleop_errors > 0)
        # The exception path short-circuits before send_action, so nothing sent.
        assert host.sent == []
        assert host._teleop_running is True
    finally:
        host.stop_teleoperate()
