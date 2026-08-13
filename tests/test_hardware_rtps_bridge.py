"""Behavior tests for the pure-RTPS hardware bridge (``HardwareRtpsBridge``).

``cyclonedds`` is an optional pip dependency, so these tests inject a fake
``cyclonedds`` (domain/pub/sub/topic) into ``sys.modules`` to exercise the
bridge wiring with NO cyclonedds installed. They assert that the RTPS bridge:

* publishes ``/<robot>/joint_states`` (+ per-camera ``image_raw``) with the
  ROS-mangled DDS topic names and the right ``sensor_msgs`` field layout;
* subscribes ``/<robot>/joint_command`` and forwards an inbound JointState into
  ``robot.send_action`` as a flat ``{name: pos}`` dict, ignoring empty/mismatched
  samples;
* is telemetry-only when no robot is bound or ``enable_commands=False``;
* tears down cleanly and is idempotent;
* raises a clear ImportError when cyclonedds is absent.

The IDL bundle's own type layouts are validated separately against a real ROS 2
node (see the PR's live cross-stack verification); here the IDL classes are
trivial fakes so the tests stay ROS-free.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import strands_robots.utils as utils_mod


class _FakeWriter:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.samples: list[Any] = []

    def write(self, msg: Any) -> None:
        self.samples.append(msg)


class _FakeReader:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self._queue: list[Any] = []

    def feed(self, msg: Any) -> None:
        self._queue.append(msg)

    def take(self, N: int = 10) -> list[Any]:
        out, self._queue = self._queue[:N], self._queue[N:]
        return out


class _FakeTopic:
    def __init__(self, participant: Any, name: str, idl_cls: Any) -> None:
        self.name = name
        self.idl_cls = idl_cls


class _FakeParticipant:
    def __init__(self, domain_id: int = 0, qos: Any = None) -> None:
        self.domain_id = domain_id
        self.qos = qos


# Minimal IDL stand-ins (the real layouts are validated against live ROS 2).
class _JointState:
    def __init__(self, header=None, name=None, position=None, velocity=None, effort=None) -> None:
        self.header = header
        self.name = name or []
        self.position = position or []
        self.velocity = velocity or []
        self.effort = effort or []


class _Image:
    def __init__(self, header=None, height=0, width=0, encoding="", is_bigendian=0, step=0, data=b"") -> None:
        self.header = header
        self.height = height
        self.width = width
        self.encoding = encoding
        self.is_bigendian = is_bigendian
        self.step = step
        self.data = data


class _Header:
    def __init__(self, stamp=None, frame_id="") -> None:
        self.stamp = stamp
        self.frame_id = frame_id


class _Time:
    def __init__(self, sec=0, nanosec=0) -> None:
        self.sec = sec
        self.nanosec = nanosec


_IDL = {
    "sensor_msgs/msg/JointState": _JointState,
    "sensor_msgs/msg/Image": _Image,
    "std_msgs/msg/Header": _Header,
    "builtin_interfaces/msg/Time": _Time,
}


@pytest.fixture
def fake_cyclonedds(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject a fake cyclonedds + patch the IDL bundle's get_type/have_cyclonedds."""
    state: dict[str, Any] = {"writers": [], "readers": [], "participants": []}

    # require_optional("cyclonedds") must succeed -> inject a module + clear cache.
    cyclonedds = ModuleType("cyclonedds")
    domain_mod = ModuleType("cyclonedds.domain")
    pub_mod = ModuleType("cyclonedds.pub")
    sub_mod = ModuleType("cyclonedds.sub")
    topic_mod = ModuleType("cyclonedds.topic")

    def _make_participant(domain_id: int = 0, qos: Any = None) -> _FakeParticipant:
        p = _FakeParticipant(domain_id, qos=qos)
        state["participants"].append(p)
        return p

    def _make_writer(_participant: Any, topic: _FakeTopic) -> _FakeWriter:
        w = _FakeWriter(topic.name)
        state["writers"].append(w)
        return w

    def _make_reader(_participant: Any, topic: _FakeTopic) -> _FakeReader:
        r = _FakeReader(topic.name)
        state["readers"].append(r)
        return r

    domain_mod.DomainParticipant = _make_participant  # type: ignore[attr-defined]
    pub_mod.DataWriter = _make_writer  # type: ignore[attr-defined]
    sub_mod.DataReader = _make_reader  # type: ignore[attr-defined]
    topic_mod.Topic = _FakeTopic  # type: ignore[attr-defined]

    # Fake cyclonedds.qos so _build_security_qos can assemble Property policies.
    qos_mod = ModuleType("cyclonedds.qos")

    class _FakeQos:
        def __init__(self, *policies: Any) -> None:
            self.policies = list(policies)

    class _FakePolicy:
        class Property:
            def __init__(self, name: str, value: str) -> None:
                self.name = name
                self.value = value

    qos_mod.Qos = _FakeQos  # type: ignore[attr-defined]
    qos_mod.Policy = _FakePolicy  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "cyclonedds", cyclonedds)
    monkeypatch.setitem(sys.modules, "cyclonedds.domain", domain_mod)
    monkeypatch.setitem(sys.modules, "cyclonedds.pub", pub_mod)
    monkeypatch.setitem(sys.modules, "cyclonedds.sub", sub_mod)
    monkeypatch.setitem(sys.modules, "cyclonedds.topic", topic_mod)
    monkeypatch.setitem(sys.modules, "cyclonedds.qos", qos_mod)
    monkeypatch.setattr(utils_mod, "_lazy_modules", {"cyclonedds": cyclonedds}, raising=False)

    # Patch the IDL bundle resolver so it returns our trivial classes.
    import strands_robots.rtps.idl as idl_mod

    monkeypatch.setattr(idl_mod, "get_type", lambda t: _IDL[t], raising=True)
    monkeypatch.setattr(idl_mod, "have_cyclonedds", lambda: True, raising=True)

    # The command-behavior tests below exercise the inbound surface, which is
    # now gated on DDS Security. They are not security tests, so run them under
    # the explicit insecure opt-out; the dedicated gate tests control this env
    # and a dds_security_config themselves.
    monkeypatch.setenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", "1")
    return state


class _FakeInner:
    name = "test_arm"


class _FakeRobot:
    def __init__(self) -> None:
        self.tool_name_str = "test_arm"
        self.robot = _FakeInner()
        self.sent_actions: list[dict[str, Any]] = []

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.sent_actions.append(action)
        return {"status": "success", "content": [{"text": "ok"}]}


def _bridge(robot=None, **kw):
    from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge

    return HardwareRtpsBridge(robot, **kw)  # type: ignore[arg-type]


def test_publish_joint_states_uses_mangled_topic_and_fields(fake_cyclonedds: dict[str, Any]) -> None:
    b = _bridge(enable_commands=False)
    b.publish_joint_states("test_arm", ["a", "b"], [0.1, 0.2])
    writer = next(w for w in fake_cyclonedds["writers"] if w.topic == "rt/test_arm/joint_states")
    (msg,) = writer.samples
    assert msg.name == ["a", "b"]
    assert msg.position == [0.1, 0.2]
    assert msg.header.frame_id == "test_arm"


def test_publish_image_fields(fake_cyclonedds: dict[str, Any]) -> None:
    b = _bridge(enable_commands=False)
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    b.publish_image("test_arm", "wrist", frame)
    writer = next(w for w in fake_cyclonedds["writers"] if w.topic == "rt/test_arm/wrist/image_raw")
    (msg,) = writer.samples
    assert (msg.height, msg.width, msg.encoding, msg.step) == (4, 6, "rgb8", 6 * 3)
    assert len(msg.data) == 4 * 6 * 3


def test_publish_image_drops_non_rgb_frames(fake_cyclonedds: dict[str, Any]) -> None:
    """publish_image silently drops frames that are not HxWx3 RGB.

    A ``sensor_msgs/Image`` published with a mismatched shape/step would make a
    downstream ROS 2 consumer misread the raw byte buffer, so the bridge rejects
    grayscale (2D) and non-3-channel frames before it ever creates a writer.
    """
    b = _bridge(enable_commands=False)
    topic = "rt/test_arm/wrist/image_raw"

    # 2D grayscale, 4-channel RGBA, and a singleton channel are all malformed
    # for an rgb8 Image -> each must be dropped without publishing.
    b.publish_image("test_arm", "wrist", np.zeros((4, 6), dtype=np.uint8))
    b.publish_image("test_arm", "wrist", np.zeros((4, 6, 4), dtype=np.uint8))
    b.publish_image("test_arm", "wrist", np.zeros((4, 6, 1), dtype=np.uint8))
    assert not any(w.topic == topic for w in fake_cyclonedds["writers"])

    # A well-formed RGB frame afterwards still publishes -> the guard is
    # specific to malformed frames, not a blanket mute of the camera topic.
    b.publish_image("test_arm", "wrist", np.zeros((4, 6, 3), dtype=np.uint8))
    writer = next(w for w in fake_cyclonedds["writers"] if w.topic == topic)
    (msg,) = writer.samples
    assert (msg.height, msg.width, msg.encoding) == (4, 6, "rgb8")


def test_command_subscription_drives_send_action(fake_cyclonedds: dict[str, Any]) -> None:
    robot = _FakeRobot()
    b = _bridge(robot)
    reader = next(r for r in fake_cyclonedds["readers"] if r.topic == "rt/test_arm/joint_command")
    reader.feed(_JointState(name=["a", "b"], position=[0.5, -0.5]))
    b._on_command(reader.take()[0])
    assert robot.sent_actions == [{"a": 0.5, "b": -0.5}]
    b.shutdown()


def test_empty_sample_is_skipped_silently(fake_cyclonedds: dict[str, Any]) -> None:
    robot = _FakeRobot()
    b = _bridge(robot)
    b._on_command(_JointState(name=[], position=[]))
    assert robot.sent_actions == []
    b.shutdown()


def test_length_mismatch_is_ignored(fake_cyclonedds: dict[str, Any]) -> None:
    robot = _FakeRobot()
    b = _bridge(robot)
    b._on_command(_JointState(name=["a", "b"], position=[0.5]))
    assert robot.sent_actions == []
    b.shutdown()


def test_read_only_when_disabled(fake_cyclonedds: dict[str, Any]) -> None:
    robot = _FakeRobot()
    b = _bridge(robot, enable_commands=False)
    assert not [r for r in fake_cyclonedds["readers"] if r.topic.endswith("joint_command")]
    b.shutdown()


def test_pure_publisher_has_no_command_surface(fake_cyclonedds: dict[str, Any]) -> None:
    b = _bridge()  # robot=None
    assert fake_cyclonedds["readers"] == []
    b.shutdown()


def test_shutdown_is_idempotent(fake_cyclonedds: dict[str, Any]) -> None:
    b = _bridge(_FakeRobot())
    b.shutdown()
    assert b._poll_thread is None
    b.shutdown()


def test_missing_cyclonedds_raises_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils_mod, "_lazy_modules", {}, raising=False)
    monkeypatch.setitem(sys.modules, "cyclonedds", None)
    from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge

    with pytest.raises(ImportError):
        HardwareRtpsBridge(_FakeRobot())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DDS Security gate on the inbound command surface
# ---------------------------------------------------------------------------

_VALID_SECURITY = {
    "identity_ca": "file:/etc/dds/identity_ca.pem",
    "certificate": "file:/etc/dds/participant_cert.pem",
    "private_key": "file:/etc/dds/participant_key.pem",
    "governance": "file:/etc/dds/governance.p7s",
    "permissions": "file:/etc/dds/permissions.p7s",
}


def test_command_surface_refuses_without_security_or_optout(
    fake_cyclonedds: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # An enabled inbound command surface drives the physical arm, so without a
    # dds_security_config and without the explicit opt-out the bridge refuses.
    monkeypatch.delenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", raising=False)
    with pytest.raises(ValueError, match="unsecured DDS graph"):
        _bridge(_FakeRobot())


def test_command_surface_allowed_with_insecure_optout(
    fake_cyclonedds: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", "yes")
    b = _bridge(_FakeRobot())
    assert b._enable_commands is True
    b.shutdown()


def test_command_surface_allowed_with_security_config_when_env_unset(
    fake_cyclonedds: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", raising=False)
    b = _bridge(_FakeRobot(), dds_security_config=dict(_VALID_SECURITY))
    assert b._enable_commands is True
    b.shutdown()


def test_telemetry_only_bridge_is_not_gated(fake_cyclonedds: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    # enable_commands=False is publish-only: no inbound surface, so no gate.
    monkeypatch.delenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", raising=False)
    b = _bridge(_FakeRobot(), enable_commands=False)
    assert b._enable_commands is False
    b.shutdown()


def test_security_config_missing_required_key_raises(fake_cyclonedds: dict[str, Any]) -> None:
    incomplete = dict(_VALID_SECURITY)
    del incomplete["private_key"]
    with pytest.raises(ValueError, match="missing required keys"):
        _bridge(_FakeRobot(), dds_security_config=incomplete)


def test_security_config_wires_plugins_and_credentials_into_participant_qos(
    fake_cyclonedds: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE", raising=False)
    b = _bridge(_FakeRobot(), dds_security_config=dict(_VALID_SECURITY))
    participant = fake_cyclonedds["participants"][-1]
    assert participant.qos is not None
    props = {p.name: p.value for p in participant.qos.policies}
    # Builtin DDS-Security plugins are wired automatically.
    assert props["dds.sec.auth.library.path"] == "dds_security_auth"
    assert props["dds.sec.crypto.library.path"] == "dds_security_crypto"
    assert props["dds.sec.access.library.path"] == "dds_security_ac"
    # Operator credentials are mapped to their dds.sec.* property names.
    assert props["dds.sec.auth.identity_ca"] == _VALID_SECURITY["identity_ca"]
    assert props["dds.sec.auth.identity_certificate"] == _VALID_SECURITY["certificate"]
    assert props["dds.sec.auth.private_key"] == _VALID_SECURITY["private_key"]
    assert props["dds.sec.access.governance"] == _VALID_SECURITY["governance"]
    assert props["dds.sec.access.permissions"] == _VALID_SECURITY["permissions"]
    # Optional permissions_ca absent -> property not set.
    assert "dds.sec.access.permissions_ca" not in props
    b.shutdown()


def test_telemetry_only_participant_has_no_security_qos(fake_cyclonedds: dict[str, Any]) -> None:
    b = _bridge(enable_commands=False)  # no robot, no config
    assert fake_cyclonedds["participants"][-1].qos is None
    b.shutdown()


# ---------------------------------------------------------------------------
# Joint position bounds enforcement on the inbound command surface
# ---------------------------------------------------------------------------


def test_joint_limits_apply_in_range_command(fake_cyclonedds: dict[str, Any]) -> None:
    robot = _FakeRobot()
    b = _bridge(robot, joint_limits={"a": (-1.0, 1.0), "b": (-1.0, 1.0)})
    b._on_command(_JointState(name=["a", "b"], position=[0.5, -0.5]))
    assert robot.sent_actions == [{"a": 0.5, "b": -0.5}]
    b.shutdown()


def test_joint_limits_reject_whole_command_when_one_joint_out_of_range(
    fake_cyclonedds: dict[str, Any],
) -> None:
    # One joint out of range -> the ENTIRE command is dropped (no partial apply).
    robot = _FakeRobot()
    b = _bridge(robot, joint_limits={"a": (-1.0, 1.0), "b": (-1.0, 1.0)})
    b._on_command(_JointState(name=["a", "b"], position=[0.5, 9.0]))
    assert robot.sent_actions == []
    b.shutdown()


def test_joint_limits_do_not_constrain_undeclared_joints(fake_cyclonedds: dict[str, Any]) -> None:
    robot = _FakeRobot()
    b = _bridge(robot, joint_limits={"a": (-1.0, 1.0)})
    # "b" has no declared bound -> not constrained.
    b._on_command(_JointState(name=["a", "b"], position=[0.5, 100.0]))
    assert robot.sent_actions == [{"a": 0.5, "b": 100.0}]
    b.shutdown()


def test_invalid_joint_limits_raise_at_construction(fake_cyclonedds: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="min .* > max|must be a"):
        _bridge(_FakeRobot(), joint_limits={"a": (1.0, -1.0)})
    with pytest.raises(ValueError, match="must be a"):
        _bridge(_FakeRobot(), joint_limits={"a": 5.0})  # type: ignore[dict-item]


def test_start_poll_is_idempotent(fake_cyclonedds: dict[str, Any]) -> None:
    """A second start while the poll thread is alive spawns no second thread.

    The cyclonedds mirror of ``test_start_spin_is_idempotent``: two threads on
    one reader would each ``take()`` and split the command stream between them.
    """
    b = _bridge(_FakeRobot())  # enable_commands default -> poll thread started
    first = b._poll_thread
    assert first is not None and first.is_alive()

    b._start_poll()
    assert b._poll_thread is first

    b.shutdown()
    assert b._poll_thread is None
