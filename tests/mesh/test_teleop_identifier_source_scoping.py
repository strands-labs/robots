"""Teleop identifiers are validated wherever a stream is built, not only on the wire.

``InputPublisher`` / ``InputReceiver`` interpolate ``device_name`` and
``source_peer_id`` straight into the Zenoh key expression
``strands/{peer_id}/input/{device_name}``. Zenoh reads ``*`` and ``**`` as
key-expression wildcards, so an unvalidated ``source_peer_id`` converts
"follow this one leader" into "apply joint commands from every peer that
publishes an input frame" - the receiver hands them to
``robot.send_action()`` with the source no longer scoped at all.

The wire ``teleop_receive`` command already rejected those values through
``validate_command``; the constructors and
``HardwareRobot.start_teleop_publish`` / ``start_teleop_receive`` - the API the
wire path itself delegates to, and the one a local caller uses - accepted them.
These tests pin the shared contract: one validator
(:func:`strands_robots.mesh.security.validate_mesh_identifier`) guards every
surface, so the accepted domains cannot diverge.

Validation sits ahead of the teardown of any stream already registered under
that key, so a refused call cannot stop a live one. That guarantee only means
something next to its mirror - an accepted call *does* stop and replace the
stream it supersedes - because "the live stream survived" is equally true of an
implementation that never tears anything down. Both halves are pinned here for
the receive surface; the publish mirror lives with the rate guard that shares
its entry point.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from strands_robots.mesh.input import InputPublisher, InputReceiver
from strands_robots.mesh.security import (
    MAX_PEER_ID_LEN,
    ValidationError,
    validate_command,
    validate_mesh_identifier,
)

#: Values the wire ``teleop_receive`` surface has always rejected. The
#: wildcards are the safety-critical ones: they widen the subscription.
BAD_IDENTIFIERS = [
    "**",  # Zenoh "any number of segments" wildcard
    "*",  # Zenoh single-segment wildcard
    "leader*",  # partial wildcard still matches many keys
    "a/b",  # extra key-expression segment
    "evil; rm -rf /",
    "evil$(whoami)",
    "with space",
    "with\nnewline",
    "with\x00null",
    "",
    "a" * (MAX_PEER_ID_LEN + 1),
]

GOOD_IDENTIFIERS = ["leader", "leader-1", "peer.with.dots", "robot_99", "a"]


class _FakeMesh:
    """Minimal mesh: a peer_id plus a subscribe that records the key expression."""

    def __init__(self, peer_id: str = "follower-1") -> None:
        self.peer_id = peer_id
        self.alive = True
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []

    def subscribe(self, topic: str, callback: Any = None, name: str | None = None) -> str:
        self.subscribed.append(topic)
        return name or topic

    def unsubscribe(self, name: str) -> None:
        self.unsubscribed.append(name)


class TestValidatorContract:
    """The shared validator's accepted domain and message shape."""

    @pytest.mark.parametrize("good", GOOD_IDENTIFIERS)
    def test_clean_identifier_returned_unchanged(self, good: str) -> None:
        assert validate_mesh_identifier(good, "p") == good

    @pytest.mark.parametrize("bad", BAD_IDENTIFIERS)
    def test_bad_identifier_rejected_and_names_the_parameter(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="InputReceiver.source_peer_id"):
            validate_mesh_identifier(bad, "InputReceiver.source_peer_id")

    def test_wildcard_rejection_message_names_wildcards(self) -> None:
        """The message tells the caller why a wildcard is refused."""
        with pytest.raises(ValidationError, match="Zenoh wildcards"):
            validate_mesh_identifier("**", "p")

    def test_non_string_rejected_with_type_name(self) -> None:
        with pytest.raises(ValidationError, match="must be a string"):
            validate_mesh_identifier(7, "p")

    def test_identifier_at_length_limit_accepted(self) -> None:
        value = "a" * MAX_PEER_ID_LEN
        assert validate_mesh_identifier(value, "p") == value


class TestReceiverSourceScoping:
    """A receiver cannot be built with an identifier that widens its subscription."""

    @pytest.mark.parametrize("bad", BAD_IDENTIFIERS)
    def test_bad_source_peer_id_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="InputReceiver.source_peer_id"):
            InputReceiver(_FakeMesh(), object(), source_peer_id=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", BAD_IDENTIFIERS)
    def test_bad_device_name_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="InputReceiver.device_name"):
            InputReceiver(_FakeMesh(), object(), source_peer_id="leader-1", device_name=bad)  # type: ignore[arg-type]

    def test_wildcard_source_never_subscribes(self) -> None:
        """Rejection happens before construction, so no wildcard key is declared."""
        mesh = _FakeMesh()
        with pytest.raises(ValidationError):
            InputReceiver(mesh, object(), source_peer_id="**")  # type: ignore[arg-type]
        assert mesh.subscribed == []

    def test_clean_source_subscribes_to_the_exact_leader_key(self) -> None:
        """The point-to-point key expression is unchanged by the guard."""
        mesh = _FakeMesh()
        recv = InputReceiver(mesh, object(), source_peer_id="leader-1", device_name="leader")  # type: ignore[arg-type]
        assert recv.topic == "strands/leader-1/input/leader"
        recv.start()
        assert mesh.subscribed == ["strands/leader-1/input/leader"]
        recv.stop()


class TestPublisherDeviceName:
    """A publisher cannot advertise actuator data on a wildcard key either."""

    @pytest.mark.parametrize("bad", BAD_IDENTIFIERS)
    def test_bad_device_name_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="InputPublisher.device_name"):
            InputPublisher(_FakeMesh(), object(), device_name=bad)  # type: ignore[arg-type]

    def test_clean_device_name_keeps_canonical_topic(self) -> None:
        pub = InputPublisher(_FakeMesh(peer_id="leader-1"), object(), device_name="gamepad")  # type: ignore[arg-type]
        assert pub.topic == "strands/leader-1/input/gamepad"


class TestWireAndDirectApiAgree:
    """Whatever the wire rejects, the API it delegates to rejects too."""

    @pytest.mark.parametrize("bad", BAD_IDENTIFIERS)
    def test_source_peer_id_domains_match(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            validate_command({"action": "teleop_receive", "source_peer_id": bad})
        with pytest.raises(ValidationError):
            InputReceiver(_FakeMesh(), object(), source_peer_id=bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("good", GOOD_IDENTIFIERS)
    def test_accepted_shapes_match(self, good: str) -> None:
        out = validate_command({"action": "teleop_receive", "source_peer_id": good, "device_name": good})
        assert out["source_peer_id"] == good
        recv = InputReceiver(_FakeMesh(), object(), source_peer_id=good, device_name=good)  # type: ignore[arg-type]
        assert recv.source_peer_id == good
        assert recv.device_name == good


class _LiveStream:
    """Stand-in for a running publisher/receiver that records being stopped."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> dict[str, Any]:
        self.stopped = True
        return {}


@pytest.fixture
def hardware_robot() -> Any:
    """A Robot carrying only the teleop state the guard needs (no hardware init)."""
    from strands_robots.hardware_robot import Robot as HardwareRobot
    from strands_robots.hardware_robot import RobotTaskState

    hw = HardwareRobot.__new__(HardwareRobot)
    hw.tool_name_str = "test_arm"
    hw.mesh = _FakeMesh(peer_id="follower-1")
    hw.peer_id = "follower-1"
    hw.robot = object()
    hw._task_state = RobotTaskState()
    hw._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="teleop_guard")
    hw._shutdown_event = threading.Event()
    hw._task_admission = threading.Lock()
    hw._task_claimed = False
    return hw


class TestHardwareRobotReportsThroughTheToolEnvelope:
    """The agent-facing entry points reject the value without raising."""

    @pytest.mark.parametrize("bad", ["**", "*", "a/b", ""])
    def test_receive_returns_error_envelope(self, hardware_robot: Any, bad: str) -> None:
        hw = hardware_robot
        result = hw.start_teleop_receive(source_peer_id=bad)
        assert result["status"] == "error"
        assert "start_teleop_receive.source_peer_id" in result["content"][0]["text"]
        # No receiver registered, no subscription declared.
        assert getattr(hw, "_input_receivers", {}) == {}
        assert hw.mesh.subscribed == []

    def test_receive_rejects_wildcard_device_name(self, hardware_robot: Any) -> None:
        hw = hardware_robot
        result = hw.start_teleop_receive(source_peer_id="leader-1", device_name="**")
        assert result["status"] == "error"
        assert "start_teleop_receive.device_name" in result["content"][0]["text"]

    def test_publish_returns_error_envelope(self, hardware_robot: Any) -> None:
        hw = hardware_robot
        result = hw.start_teleop_publish(teleoperator=object(), device_name="**")
        assert result["status"] == "error"
        assert "start_teleop_publish.device_name" in result["content"][0]["text"]
        assert getattr(hw, "_input_publishers", {}) == {}

    def test_rejected_receive_leaves_a_live_stream_running(self, hardware_robot: Any) -> None:
        """Validation precedes the stop-existing-stream step, so nothing is lost."""
        hw = hardware_robot
        live = _LiveStream()
        hw._input_receivers = {"leader-1/leader": live}
        result = hw.start_teleop_receive(source_peer_id="**", device_name="leader")
        assert result["status"] == "error"
        assert live.stopped is False
        assert hw._input_receivers == {"leader-1/leader": live}

    def test_rejected_publish_leaves_a_live_stream_running(self, hardware_robot: Any) -> None:
        hw = hardware_robot
        live = _LiveStream()
        hw._input_publishers = {"leader": live}
        result = hw.start_teleop_publish(teleoperator=object(), device_name="a/b")
        assert result["status"] == "error"
        assert live.stopped is False
        assert hw._input_publishers == {"leader": live}


class TestAnAcceptedCallReplacesTheLiveStream:
    """A receive that passes validation supersedes the stream it replaces.

    ``start_teleop_receive`` keys its registry on ``source_peer_id/device_name``
    and stops whatever is registered under that key before installing the new
    receiver. Nothing drove that step, so the refusal guarantee above stood
    alone - and it holds just as well for a body with no teardown, where the
    superseded receiver would keep applying the old leader's frames to the same
    hardware alongside the new one.
    """

    def test_an_accepted_receive_stops_and_replaces_the_live_receiver(self, hardware_robot: Any) -> None:
        """The stream under that key is stopped and the new receiver takes it over."""
        hw = hardware_robot
        live = _LiveStream()
        hw._input_receivers = {"leader-1/leader": live}

        result = hw.start_teleop_receive(source_peer_id="leader-1", device_name="leader")

        assert result["status"] == "success"
        assert live.stopped is True
        replacement = hw._input_receivers["leader-1/leader"]
        assert replacement is not live
        assert isinstance(replacement, InputReceiver)
        assert hw.mesh.subscribed == ["strands/leader-1/input/leader"]
        # One entry per key: the replacement takes the slot, it does not add one.
        assert sorted(hw._input_receivers) == ["leader-1/leader"]

    def test_a_second_leader_under_the_same_device_name_leaves_the_first_running(self, hardware_robot: Any) -> None:
        """The key is the pair, so two leaders can drive one follower.

        A registry keyed on ``device_name`` alone would silently stop the first
        leader's stream here - the follower would end up following whichever
        leader connected last, with nothing reported.
        """
        hw = hardware_robot
        first = _LiveStream()
        hw._input_receivers = {"leader-1/leader": first}

        result = hw.start_teleop_receive(source_peer_id="leader-2", device_name="leader")

        assert result["status"] == "success"
        assert first.stopped is False
        assert sorted(hw._input_receivers) == ["leader-1/leader", "leader-2/leader"]

    def test_the_refusal_assertion_is_backed_by_a_reachable_teardown(self, hardware_robot: Any) -> None:
        """One key, both directions: refused leaves it, accepted replaces it.

        ``test_rejected_receive_leaves_a_live_stream_running`` asserts a live
        receiver is *not* stopped by a refused call. On this surface both
        refusable arguments are part of the registry key, so a refused value
        cannot name a registered entry and that assertion holds for either
        ordering of validation and teardown - and for a body with no teardown
        at all. Driving both directions on one key is what shows the teardown
        is reachable, and so what the refusal assertion rules out. (The
        publish surface differs: ``hz`` is refused while ``device_name`` still
        names the live stream, so ordering is observable there and is pinned
        with the rate guard.)
        """
        hw = hardware_robot
        live = _LiveStream()
        hw._input_receivers = {"leader-1/leader": live}

        refused = hw.start_teleop_receive(source_peer_id="**", device_name="leader")
        assert refused["status"] == "error"
        assert live.stopped is False

        accepted = hw.start_teleop_receive(source_peer_id="leader-1", device_name="leader")
        assert accepted["status"] == "success"
        assert live.stopped is True

    def test_the_replacement_leaves_exactly_one_live_subscription(self, hardware_robot: Any) -> None:
        """Replacing a real receiver drops its subscription rather than leaking it.

        Driven with a real :class:`InputReceiver` rather than a stand-in: a
        superseded receiver that stayed subscribed would keep delivering frames
        into ``robot.send_action`` from a stream the caller believes is gone.
        """
        hw = hardware_robot
        superseded = InputReceiver(mesh=hw.mesh, robot=object(), source_peer_id="leader-1", device_name="leader")
        superseded.start()
        hw._input_receivers = {"leader-1/leader": superseded}
        assert hw.mesh.unsubscribed == []

        result = hw.start_teleop_receive(source_peer_id="leader-1", device_name="leader")

        assert result["status"] == "success"
        assert hw.mesh.unsubscribed == ["input:leader-1/leader"]
        replacement = hw._input_receivers["leader-1/leader"]
        assert replacement is not superseded
        assert replacement.topic == superseded.topic
