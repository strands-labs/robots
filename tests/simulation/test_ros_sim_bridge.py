"""Behavior tests for the ROS 2 simulation bridge (``SimEngine(ros2_bridge=...)``).

``rclpy`` is a system-provided, non-PyPI dependency, so these tests inject a
fake ``rclpy`` + ``sensor_msgs.msg`` into ``sys.modules`` to exercise the
publisher wiring with NO ROS 2 installed. They assert that:

* :class:`SimRosBridge` builds the right topics and ``JointState`` / ``Image``
  message fields and routes them to per-robot publishers.
* :meth:`SimEngine._publish_ros_telemetry` reads joint state from
  ``get_observation`` and forwards it, and is a no-op when the bridge is off.
* Enabling the bridge with no ``rclpy`` raises a clear :class:`ImportError`.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import strands_robots.utils as utils_mod
from strands_robots.simulation.base import SimEngine


class _FakePublisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.messages: list[Any] = []

    def publish(self, msg: Any) -> None:
        self.messages.append(msg)


class _FakeClock:
    def now(self) -> Any:
        return self

    def to_msg(self) -> str:
        return "stamp"


class _FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name
        self.publishers: list[_FakePublisher] = []
        self.destroyed = False

    def get_clock(self) -> _FakeClock:
        return _FakeClock()

    def create_publisher(self, _msg_type: Any, topic: str, _depth: int) -> _FakePublisher:
        pub = _FakePublisher(topic)
        self.publishers.append(pub)
        return pub

    def destroy_node(self) -> None:
        self.destroyed = True


class _Header:
    def __init__(self) -> None:
        self.stamp: Any = None
        self.frame_id: str = ""


class _JointState:
    def __init__(self) -> None:
        self.header = _Header()
        self.name: list[str] = []
        self.position: list[float] = []


class _Image:
    def __init__(self) -> None:
        self.header = _Header()
        self.height = 0
        self.width = 0
        self.encoding = ""
        self.is_bigendian = 0
        self.step = 0
        self.data = b""


@pytest.fixture
def fake_ros(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject a fake rclpy + sensor_msgs.msg and clear require_optional's cache."""
    state: dict[str, Any] = {"inited": False, "shutdown": False, "nodes": []}

    rclpy = ModuleType("rclpy")
    rclpy.ok = lambda: state["inited"]  # type: ignore[attr-defined]

    def _init() -> None:
        state["inited"] = True

    def _shutdown() -> None:
        state["shutdown"] = True
        state["inited"] = False

    def _create_node(name: str) -> _FakeNode:
        node = _FakeNode(name)
        state["nodes"].append(node)
        return node

    rclpy.init = _init  # type: ignore[attr-defined]
    rclpy.shutdown = _shutdown  # type: ignore[attr-defined]
    rclpy.create_node = _create_node  # type: ignore[attr-defined]

    sensor_pkg = ModuleType("sensor_msgs")
    sensor_msg = ModuleType("sensor_msgs.msg")
    sensor_msg.JointState = _JointState  # type: ignore[attr-defined]
    sensor_msg.Image = _Image  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_pkg)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msg)
    # require_optional memoizes resolved modules; drop any cached real entries.
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)
    return state


def test_sim_ros_bridge_publishes_joint_states(fake_ros: dict[str, Any]) -> None:
    from strands_robots.simulation.ros_bridge import SimRosBridge

    bridge = SimRosBridge(domain_id=7)
    bridge.publish_joint_states("so101", ["shoulder_pan", "elbow"], [0.1, 0.2])

    node = fake_ros["nodes"][0]
    (pub,) = node.publishers
    assert pub.topic == "/so101/joint_states"
    (msg,) = pub.messages
    assert msg.name == ["shoulder_pan", "elbow"]
    assert msg.position == [0.1, 0.2]
    assert msg.header.frame_id == "so101"


def test_sim_ros_bridge_sets_domain_env(fake_ros: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    from strands_robots.simulation.ros_bridge import SimRosBridge

    SimRosBridge(domain_id=42)
    assert os.environ["ROS_DOMAIN_ID"] == "42"


def test_sim_ros_bridge_publishes_rgb_image(fake_ros: dict[str, Any]) -> None:
    from strands_robots.simulation.ros_bridge import SimRosBridge

    bridge = SimRosBridge()
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    bridge.publish_image("so101", "front", frame)

    node = fake_ros["nodes"][0]
    (pub,) = node.publishers
    assert pub.topic == "/so101/front/image_raw"
    (msg,) = pub.messages
    assert (msg.height, msg.width, msg.encoding, msg.step) == (4, 5, "rgb8", 15)
    assert len(msg.data) == 4 * 5 * 3


def test_sim_ros_bridge_ignores_non_rgb_image(fake_ros: dict[str, Any]) -> None:
    from strands_robots.simulation.ros_bridge import SimRosBridge

    bridge = SimRosBridge()
    bridge.publish_image("so101", "depth", np.zeros((4, 5), dtype=np.uint8))
    assert fake_ros["nodes"][0].publishers == []


def test_sim_ros_bridge_shutdown_destroys_node(fake_ros: dict[str, Any]) -> None:
    from strands_robots.simulation.ros_bridge import SimRosBridge

    bridge = SimRosBridge()
    node = fake_ros["nodes"][0]
    bridge.shutdown()
    assert node.destroyed is True
    assert fake_ros["shutdown"] is True
    bridge.shutdown()  # idempotent


class _FakeEngine(SimEngine):
    """Minimal concrete engine exercising the telemetry helper only."""

    def __init__(self, observation: dict[str, Any], *, ros2_bridge: bool = False, ros2_domain: int = 0) -> None:
        self._obs = observation
        self._init_ros_bridge(ros2_bridge=ros2_bridge, ros2_domain=ros2_domain)

    def list_robots(self) -> list[str]:
        return ["so101"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["shoulder_pan", "elbow"]

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        return self._obs

    # Unused abstract methods for this focused test (never called).
    def create_world(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def destroy(self) -> dict[str, Any]:
        raise NotImplementedError

    def reset(self) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def add_robot(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def remove_robot(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def add_object(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def remove_object(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def render(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def send_action(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def physics_timestep(self) -> float:
        return 0.002


def test_publish_telemetry_forwards_joint_state(fake_ros: dict[str, Any]) -> None:
    obs = {"shoulder_pan": 0.5, "elbow": -0.25, "front": np.zeros((2, 2, 3), dtype=np.uint8)}
    engine = _FakeEngine(obs, ros2_bridge=True, ros2_domain=3)
    engine._publish_ros_telemetry()

    node = fake_ros["nodes"][0]
    topics = {p.topic: p for p in node.publishers}
    assert "/so101/joint_states" in topics
    js = topics["/so101/joint_states"].messages[0]
    assert js.position == [0.5, -0.25]
    assert "/so101/front/image_raw" in topics


def test_publish_telemetry_is_noop_when_disabled(fake_ros: dict[str, Any]) -> None:
    engine = _FakeEngine({"shoulder_pan": 0.0, "elbow": 0.0})  # ros2_bridge defaults False
    engine._publish_ros_telemetry()
    assert engine._ros_bridge is None
    assert fake_ros["nodes"] == []


def test_enabling_bridge_without_rclpy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # No fake_ros fixture here: ensure rclpy is absent and the cache is clear.
    monkeypatch.setitem(sys.modules, "rclpy", None)
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)
    with pytest.raises(ImportError):
        _FakeEngine({}, ros2_bridge=True)


def test_publish_telemetry_safe_without_bridge_init(fake_ros: dict[str, Any]) -> None:
    """Telemetry hooks tolerate an engine that never called ``_init_ros_bridge``.

    The simulation ABC defines no ``__init__``, so a lightweight subclass (or a
    backend constructed via an unusual path) may never initialize the bridge
    attributes. ``step`` calls ``_publish_ros_telemetry`` unconditionally, so it
    - and ``_shutdown_ros_bridge`` - must be safe no-ops in that case rather
    than raising ``AttributeError`` on a missing ``_ros_bridge``. Here we strip
    the attribute off a normally-built engine to model that uninitialized state.
    """
    engine = _FakeEngine({"shoulder_pan": 0.0, "elbow": 0.0})
    del engine._ros_bridge

    engine._publish_ros_telemetry()  # must not raise
    engine._shutdown_ros_bridge()  # must not raise
    assert fake_ros["nodes"] == []


class _TwoRobotEngine(SimEngine):
    """Engine with two robots where one robot's observation always fails.

    Models a transient per-robot render fault (EGL/GL context loss, a camera
    that produced no frame) so we can assert the documented contract: a single
    robot's failure must not interrupt the loop or escape ``step()``.
    """

    def __init__(self, *, ros2_bridge: bool = True, ros2_domain: int = 0) -> None:
        self._init_ros_bridge(ros2_bridge=ros2_bridge, ros2_domain=ros2_domain)

    def list_robots(self) -> list[str]:
        return ["broken", "healthy"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["shoulder_pan", "elbow"]

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        if robot_name == "broken":
            raise RuntimeError("camera render failed (EGL context lost)")
        return {"shoulder_pan": 0.5, "elbow": -0.25}

    # Unused abstract methods for this focused test (never called).
    def create_world(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def destroy(self) -> dict[str, Any]:
        raise NotImplementedError

    def reset(self) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        raise NotImplementedError

    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def add_robot(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def remove_robot(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def add_object(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def remove_object(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def render(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def send_action(self, *a: Any, **k: Any) -> dict[str, Any]:
        raise NotImplementedError

    def physics_timestep(self) -> float:
        return 0.002


def test_publish_telemetry_per_robot_failure_does_not_interrupt_loop(fake_ros: dict[str, Any]) -> None:
    """A failing robot is skipped; healthy robots still publish; step() never crashes.

    Pins the docstring contract on the hot ``ros2_bridge=True`` path: per-robot
    failures (e.g. a camera that did not render) never interrupt the loop. The
    "broken" robot raises in ``get_observation``; the "healthy" robot - listed
    after it - must still publish its joint_states, and no exception escapes.
    """
    engine = _TwoRobotEngine(ros2_bridge=True, ros2_domain=5)

    engine._publish_ros_telemetry()  # must not raise despite the broken robot

    node = fake_ros["nodes"][0]
    topics = {p.topic: p for p in node.publishers}
    # Broken robot never published; healthy one published despite being second.
    assert "/broken/joint_states" not in topics
    assert "/healthy/joint_states" in topics
    assert topics["/healthy/joint_states"].messages[0].position == [0.5, -0.25]


def test_publish_telemetry_skip_images_publishes_joints_only(fake_ros: dict[str, Any]) -> None:
    """``skip_images=True`` publishes joint_states but never publishes camera images.

    Backends call ``_publish_ros_telemetry(skip_images=True)`` on the hot step
    path to skip the render/publish cost when no camera subscriber needs frames.
    Pin the contract: joint_states still go out, but no ``image_raw`` publisher
    is created for camera keys present in the observation.
    """
    obs = {"shoulder_pan": 0.5, "elbow": -0.25, "front": np.zeros((2, 2, 3), dtype=np.uint8)}
    engine = _FakeEngine(obs, ros2_bridge=True, ros2_domain=3)

    engine._publish_ros_telemetry(skip_images=True)

    node = fake_ros["nodes"][0]
    topics = {p.topic: p for p in node.publishers}
    assert "/so101/joint_states" in topics
    assert topics["/so101/joint_states"].messages[0].position == [0.5, -0.25]
    assert "/so101/front/image_raw" not in topics


def test_shutdown_ros_bridge_tears_down_active_bridge_idempotently(fake_ros: dict[str, Any]) -> None:
    """``_shutdown_ros_bridge`` destroys an active bridge and clears the handle.

    Pins the documented "safe to call repeatedly" contract: the first call
    forwards to :meth:`SimRosBridge.shutdown` (destroying the ROS 2 node) and
    resets ``_ros_bridge`` to None, so a second call is a no-op rather than a
    double shutdown on a dead node.
    """
    engine = _FakeEngine({"shoulder_pan": 0.0, "elbow": 0.0}, ros2_bridge=True, ros2_domain=2)
    node = fake_ros["nodes"][0]
    assert engine._ros_bridge is not None

    engine._shutdown_ros_bridge()

    assert node.destroyed is True
    assert engine._ros_bridge is None

    engine._shutdown_ros_bridge()  # idempotent: must not raise, handle stays None
    assert engine._ros_bridge is None
