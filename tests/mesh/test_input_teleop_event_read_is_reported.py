"""A failed teleop-event read must be reported, not published as ``events: None``.

:class:`~strands_robots.mesh.input.InputPublisher` streams a leader's joint
action and, alongside it, the operator's control signals from the
teleoperator's ``get_teleop_events()`` - ``terminate_episode`` (the operator
asking to end the episode), ``success``, ``rerecord_episode``,
``is_intervention``. The published topic schema declares that field nullable,
and ``null`` is the correct value for a plain leader arm that exposes no event
surface at all.

That made a failed read indistinguishable from "the operator signalled
nothing": a teleoperator whose event surface stops answering (a keyboard
listener thread that died, a gamepad unplugged mid-session) published
``events: None`` on every frame while joint commands kept flowing, and left no
trace anywhere - ``stats`` reported a clean session and no log line was emitted.
An operator pressing quit could not tell their signal was being dropped.

The read stays best-effort: the event surface is secondary to the joint stream,
so a failure must not drop the frame the arm is following. What the failure must
not do is disappear. These tests pin both halves - the frame survives, and the
failure is counted in ``stats["event_read_errors"]`` and logged.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import pytest

from strands_robots.mesh.input import InputPublisher

_ACTION = {"shoulder.pos": 0.25, "elbow.pos": -0.10}
_LOGGER = "strands_robots.mesh.input"


class _StopAfterMesh:
    """Mesh double that ends the publish loop after ``limit`` frames.

    Driving the loop synchronously (rather than through the real background
    thread) keeps the assertions on exact frame counts deterministic.
    """

    peer_id = "leader-01"

    def __init__(self, limit: int = 3) -> None:
        self.limit = limit
        self.published: list[dict[str, Any]] = []
        self.publisher: InputPublisher | None = None

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published.append(payload)
        if len(self.published) >= self.limit and self.publisher is not None:
            self.publisher._running = False


class _Leader:
    """Leader arm whose ``get_teleop_events`` either answers or raises."""

    def __init__(self, *, raises: BaseException | None = None, events: Any = None) -> None:
        self._raises = raises
        self._events = events
        self.event_calls = 0

    def get_action(self) -> dict[str, float]:
        return dict(_ACTION)

    def get_teleop_events(self) -> Any:
        self.event_calls += 1
        if self._raises is not None:
            raise self._raises
        return self._events


class _EventlessLeader:
    """Plain leader arm: no ``get_teleop_events`` surface at all."""

    def get_action(self) -> dict[str, float]:
        return dict(_ACTION)


def _drive(teleoperator: Any, frames: int = 3) -> tuple[_StopAfterMesh, InputPublisher]:
    """Run the real publish loop for exactly ``frames`` frames, synchronously."""
    mesh = _StopAfterMesh(limit=frames)
    # A high rate leaves no sleep budget, so the loop never blocks in tests.
    publisher = InputPublisher(cast(Any, mesh), teleoperator, device_name="leader", hz=10_000.0)
    mesh.publisher = publisher
    publisher._running = True
    publisher._publish_loop()
    assert len(mesh.published) == frames, f"premise: expected {frames} frames, got {len(mesh.published)}"
    return mesh, publisher


def _event_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "teleop-event" in r.getMessage()]


class TestAFailedEventReadIsDistinguishableFromNoEventSurface:
    """The two states that both publish ``events: None`` must not look alike."""

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("keyboard listener thread is dead"),
            OSError("gamepad /dev/input/js0 disconnected"),
        ],
        ids=["listener-died", "device-unplugged"],
    )
    def test_a_raising_event_read_is_counted(self, error: BaseException, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            mesh, publisher = _drive(_Leader(raises=error))

        # ``.get(..., 0)`` so a build that does not report the failure at all
        # fails on the count rather than on a missing key.
        assert publisher.stats.get("event_read_errors", 0) == 3, (
            "every frame published events=None because the read raised "
            f"{error!r}, but stats reported "
            f"event_read_errors={publisher.stats.get('event_read_errors', 0)} - "
            "indistinguishable from a leader with no event surface"
        )
        assert all(frame["events"] is None for frame in mesh.published)

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("keyboard listener thread is dead"), OSError("js0 disconnected")],
        ids=["listener-died", "device-unplugged"],
    )
    def test_a_raising_event_read_is_logged_with_the_device_and_the_cause(
        self, error: BaseException, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            _drive(_Leader(raises=error))

        warnings = _event_warnings(caplog)
        assert warnings, "a failed teleop-event read emitted no warning at all"
        assert "leader" in warnings[0], warnings[0]
        assert str(error) in warnings[0], warnings[0]

    def test_no_event_surface_is_not_reported_as_a_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """The legitimate ``events: None`` must stay silent - no over-reporting."""
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            mesh, publisher = _drive(_EventlessLeader())

        assert publisher.stats.get("event_read_errors", 0) == 0
        assert _event_warnings(caplog) == []
        assert all(frame["events"] is None for frame in mesh.published)

    def test_a_successful_read_is_silent_and_carries_the_events(self, caplog: pytest.LogCaptureFixture) -> None:
        signals = {"terminate_episode": True, "success": False}
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            mesh, publisher = _drive(_Leader(events=dict(signals)))

        assert publisher.stats.get("event_read_errors", 0) == 0
        assert _event_warnings(caplog) == []
        assert all(frame["events"] == signals for frame in mesh.published)


class TestTheJointStreamSurvivesAFailedEventRead:
    """The event surface is secondary: its failure must not stop the arm."""

    def test_every_frame_is_still_published_with_its_action(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            mesh, publisher = _drive(_Leader(raises=RuntimeError("listener dead")), frames=4)

        assert publisher.stats["frames"] == 4
        assert [frame["action"] for frame in mesh.published] == [_ACTION] * 4
        assert [frame["seq"] for frame in mesh.published] == [0, 1, 2, 3]

    def test_the_publish_error_counter_is_not_conflated(self, caplog: pytest.LogCaptureFixture) -> None:
        """``errors`` counts lost frames; a lost event read loses no frame."""
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            _mesh, publisher = _drive(_Leader(raises=RuntimeError("listener dead")))

        assert publisher.stats["errors"] == 0


class TestTheLogIsBoundedButTheCounterIsNot:
    """A persistent fault at control rate must not flood the console."""

    def test_warnings_are_capped_while_the_count_stays_exact(self, caplog: pytest.LogCaptureFixture) -> None:
        frames = 30
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            _mesh, publisher = _drive(_Leader(raises=RuntimeError("listener dead")), frames=frames)

        assert publisher.stats.get("event_read_errors", 0) == frames, (
            "the counter must record every failed read, not just the logged ones"
        )
        assert len(_event_warnings(caplog)) <= 5
