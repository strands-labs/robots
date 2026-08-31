"""A degraded state probe names itself in the published snapshot.

:meth:`strands_robots.mesh.core.Mesh._read_state` probes a robot defensively --
hardware joints, task state, sim world, sim joints -- and omits the section of
any probe that raises. An omitted section is ambiguous by itself: a robot with no
joints and a robot whose joint read just failed produce the same snapshot.

Reporting the failure in the peer's own log closed half of that. It could not
close the other half, because the observer that needs to explain the absence is
on another machine: a fleet view could only distinguish the two cases by reading
that peer's log, so one grew a regex over mesh's own log lines and used it as an
API. These tests pin the contract that removes the need:

* while a probe is failing, the snapshot carries ``degraded[<category>]`` with
  ``reason`` (the exception's type name -- the discriminator
  ``_warn_read_state_once`` documents as selecting the operator's next move),
  ``detail`` (its message, bounded), ``failures`` and ``for_seconds``;
* ``for_seconds`` is the interval that has elapsed since the fault began,
  measured on the monotonic clock -- a duration a renderer could report as a
  constant and satisfy every other assertion here;
* the entry disappears on the tick the probe answers again;
* a healthy peer's snapshot is unchanged -- no key is added when nothing failed;
* the block makes the snapshot non-empty, so a hardware-only peer whose one
  section is the one that raised keeps publishing instead of going silent. That
  silence is the whole reason a log regex was the only way to explain it: there
  was no message to inspect.

The report is deliberately NOT changed with it. ``_read_state_warned`` arms once
per category for the life of the peer, so a probe flapping at ``STATE_HZ`` costs
the same one warning it always did while the wire tracks every transition.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.mesh.core import Mesh

CORE_LOGGER = "strands_robots.mesh.core"

#: Every category the probes report, taken from the source in
#: :mod:`tests.mesh.test_read_state_probe_failures_are_reported` for the same
#: reason: a fifth probe must not be able to appear without being held to this.
EXPECTED_CATEGORIES = {"hw_joints", "task_state", "sim_world", "sim_joints"}


class _Bus:
    """A device whose joint read fails, or answers, on demand."""

    is_connected = True

    def __init__(self) -> None:
        self.config = SimpleNamespace(cameras={})
        # read_joints prefers a bus that can sync_read; None routes it through
        # get_observation, which is the seam these tests drive.
        self.bus = None
        self.exc: BaseException | None = None
        self.reads = 0

    def get_observation(self) -> dict[str, Any]:
        self.reads += 1
        if self.exc is not None:
            raise self.exc
        return {"shoulder_pan.pos": 0.5, "elbow_flex.pos": -0.25}


def _mesh(host: Any, peer_id: str = "arm-a1") -> Mesh:
    """A Mesh with only what ``_read_state`` reads.

    Built through ``__new__`` on purpose: the state machinery has to work on a
    peer whose ``__init__`` never ran, which is how the sim paths reach it.
    """
    m = Mesh.__new__(Mesh)
    m.peer_id = peer_id  # type: ignore[misc]
    m.robot = host  # type: ignore[misc]
    return m


def _hardware_peer() -> tuple[Mesh, _Bus]:
    """A peer whose only section is ``joints`` -- the shape that went silent."""
    bus = _Bus()
    return _mesh(SimpleNamespace(robot=bus)), bus


class TestAFailingProbeNamesItselfOnTheWire:
    """The block is present, keyed by category, while the probe is failing."""

    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionError("Port is in use!"),
            RuntimeError("this arm has no calibration registered"),
            TimeoutError("no response from motors"),
            OSError("[Errno 6] Device not configured"),
        ],
        ids=["port-contention", "uncalibrated", "no-response", "unplugged"],
    )
    def test_the_reason_is_the_exception_type(self, exc: BaseException) -> None:
        """Four faults an operator answers differently, told apart on the wire.

        The type is the discriminator the reporter's docstring already names: a
        contended port is a different job from an uncalibrated arm. Before this,
        all four arrived as the same absent section.
        """
        m, bus = _hardware_peer()
        bus.exc = exc
        out = m._read_state()

        assert out is not None
        record = out["degraded"]["hw_joints"]
        assert record["reason"] == type(exc).__name__
        assert record["detail"] == str(exc)
        assert record["failures"] == 1
        assert record["for_seconds"] >= 0.0

    def test_only_the_probe_that_failed_is_named(self) -> None:
        """A working section must not be reported as degraded."""

        class _RaisingTask:
            @property
            def status(self) -> Any:
                raise RuntimeError("task state unreadable")

        bus = _Bus()
        m = _mesh(SimpleNamespace(robot=bus, _task_state=_RaisingTask()))
        out = m._read_state()

        assert out is not None
        assert out["joints"], "the readable section still publishes"
        assert set(out["degraded"]) == {"task_state"}

    def test_the_failure_count_accumulates_across_ticks(self) -> None:
        """``failures`` distinguishes one unlucky read from a standing fault."""
        m, bus = _hardware_peer()
        bus.exc = ConnectionError("Port is in use!")

        counts = []
        for _ in range(3):
            out = m._read_state()
            assert out is not None
            counts.append(out["degraded"]["hw_joints"]["failures"])

        assert counts == [1, 2, 3], f"failures must count ticks, got {counts}"

    def test_a_changed_failure_mode_replaces_the_reason(self) -> None:
        """A fault that becomes a different fault must not keep the old reason.

        A contended port that is resolved into an arm nobody calibrated is two
        different jobs, and reporting the first one forever would send the
        operator back to a port that is now free.
        """
        m, bus = _hardware_peer()
        bus.exc = ConnectionError("Port is in use!")
        first = m._read_state()
        bus.exc = RuntimeError("this arm has no calibration registered")
        second = m._read_state()

        assert first is not None and second is not None
        assert first["degraded"]["hw_joints"]["reason"] == "ConnectionError"
        assert second["degraded"]["hw_joints"]["reason"] == "RuntimeError"
        assert second["degraded"]["hw_joints"]["detail"] == "this arm has no calibration registered"

    def test_the_detail_is_bounded(self) -> None:
        """The text is a third-party message on a 10 Hz topic, so it is capped."""
        from strands_robots.mesh.core import MAX_DEGRADED_DETAIL_LEN

        m, bus = _hardware_peer()
        bus.exc = RuntimeError("x" * (MAX_DEGRADED_DETAIL_LEN * 4))
        out = m._read_state()

        assert out is not None
        assert len(out["degraded"]["hw_joints"]["detail"]) == MAX_DEGRADED_DETAIL_LEN


class TestTheBlockClearsWhenTheProbeAnswers:
    """A stale diagnosis is worse than none: the entry is removed on recovery."""

    def test_a_recovered_probe_is_no_longer_reported(self) -> None:
        m, bus = _hardware_peer()
        bus.exc = ConnectionError("Port is in use!")
        during = m._read_state()
        bus.exc = None
        after = m._read_state()

        assert during is not None and "hw_joints" in during["degraded"]
        assert after is not None
        assert "degraded" not in after, "a probe that answered must not still be reported degraded"
        assert after["joints"], "and its section is back"

    def test_a_second_fault_after_recovery_starts_a_fresh_count(self) -> None:
        """``failures`` and ``for_seconds`` describe the CURRENT fault."""
        m, bus = _hardware_peer()
        bus.exc = ConnectionError("Port is in use!")
        m._read_state()
        m._read_state()
        bus.exc = None
        m._read_state()
        bus.exc = ConnectionError("Port is in use!")
        out = m._read_state()

        assert out is not None
        assert out["degraded"]["hw_joints"]["failures"] == 1

    def test_a_peer_with_no_hardware_is_not_a_recovery(self) -> None:
        """A probe that did not run must not clear another peer shape's fault.

        The clear happens where the read returned, not where the probe was
        skipped, so a sim peer with no motor bus cannot read as an ``hw_joints``
        recovery -- and an arm that has gone away keeps its last diagnosis.
        """
        m, bus = _hardware_peer()
        bus.exc = ConnectionError("Port is in use!")
        m._read_state()
        bus.is_connected = False  # the arm is gone; the probe is skipped entirely
        out = m._read_state()

        assert out is not None
        assert out["degraded"]["hw_joints"]["reason"] == "ConnectionError", (
            "a skipped probe is not an answer, so the standing fault survives"
        )
        assert bus.reads == 1, "premise: the second tick did not reach the device"


class TestAHealthyPeerIsUnchanged:
    """No key is added when nothing failed."""

    def test_a_readable_arm_publishes_no_degraded_key(self) -> None:
        m, _bus = _hardware_peer()
        out = m._read_state()

        assert out is not None
        assert set(out) == {"peer_id", "t", "joints"}

    def test_a_peer_with_nothing_to_report_still_publishes_nothing(self) -> None:
        """The empty-snapshot guard is about having nothing to say, not silence.

        A host with no hardware, no task and no world has no diagnosis either,
        so it still returns ``None`` and the loop still publishes nothing. What
        changed is that a probe FAILURE is now something to say.
        """
        m = _mesh(SimpleNamespace(robot=None))
        out = m._read_state()
        assert out is None


class TestTheSilencedPeerKeepsPublishing:
    """The headline consequence: a diagnosis is content, so the peer speaks."""

    def test_a_hardware_only_peer_publishes_its_fault(self) -> None:
        """Its one section is the one that raised, and it used to vanish.

        ``_read_state`` returns ``None`` when only ``peer_id`` and ``t``
        survived and the loop publishes nothing for a ``None``, so this peer
        stopped appearing on the state topic while its presence heartbeat kept
        advertising it -- with nothing on the wire to inspect, which is what
        left a log regex as the only way to explain it.
        """
        m, bus = _hardware_peer()
        bus.exc = ConnectionError("Port is in use!")
        out = m._read_state()

        assert out is not None, "a peer with a fault to report must not go silent"
        assert out["degraded"]["hw_joints"]["reason"] == "ConnectionError"

    def test_the_state_loop_publishes_that_snapshot(self) -> None:
        """End to end through the loop's own publish gate, not just the builder."""
        m, bus = _hardware_peer()
        bus.exc = ConnectionError("Port is in use!")
        published: list[tuple[str, Any]] = []
        # Parameter named as Mesh.publish names it: the assignment is checked.
        m.publish = lambda key, payload: published.append((key, payload))  # type: ignore[method-assign]

        state = m._read_state()
        if state:
            m.publish(f"strands/{m.peer_id}/state", state)

        assert len(published) == 1, "the loop's `if state:` gate must let a diagnosis through"
        topic, payload = published[0]
        assert topic == "strands/arm-a1/state"
        assert payload["degraded"]["hw_joints"]["detail"] == "Port is in use!"


class TestTheDurationIsMeasuredNotStamped:
    """``for_seconds`` is a duration, so it is measured on the monotonic clock.

    Two independent properties, and the second one is what stops the first from
    being satisfiable by a literal: the duration must not track the wall clock,
    AND it must actually be derived from a clock rather than reported as a
    constant. A renderer that returns ``0.0`` unconditionally has the right
    clock domain vacuously.
    """

    def test_the_duration_is_the_interval_the_probe_has_been_failing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A constant satisfies every other assertion this file makes about it.

        ``for_seconds >= 0.0`` in
        :meth:`TestAFailingProbeNamesItselfOnTheWire.test_the_reason_is_the_exception_type`,
        ``for_seconds < 60.0`` in the wall-clock test beside this one, and the
        key-set check below are all true of a hardcoded ``0.0``, so the suite
        graded the field's presence and its clock DOMAIN while asserting nothing
        about whether it had been measured. Under that grading a renderer whose
        duration is always zero -- because it subtracts a key no record carries,
        which is the shape a second renderer of this block took in review -- is
        green, and the operator-visible consequence is the one
        :meth:`strands_robots.mesh.core.Mesh._degraded_probes` exists to
        prevent: "this probe failed" and "this probe has been failing since it
        was plugged in" want different responses, and a duration pinned at zero
        reports the second as the first.

        The clock is driven rather than slept on, so the interval is exact and
        the test does not trade runtime for the assertion's strength.
        """
        from strands_robots.mesh import core

        elapsed = [1000.0]
        monkeypatch.setattr(core.time, "monotonic", lambda: elapsed[0])

        m, bus = _hardware_peer()
        bus.exc = ConnectionError("Port is in use!")
        first = m._read_state()
        elapsed[0] += 3.5
        second = m._read_state()

        assert first is not None and second is not None
        assert first["degraded"]["hw_joints"]["for_seconds"] == pytest.approx(0.0), (
            "premise: the tick that records the fault has measured no interval yet"
        )
        assert second["degraded"]["hw_joints"]["for_seconds"] == pytest.approx(3.5), (
            "for_seconds must be the measured interval since the fault began, "
            "not a constant that happens to be non-negative"
        )
        assert second["degraded"]["hw_joints"]["failures"] == 2, (
            "premise: the second tick is the same standing fault, not a new one"
        )

    def test_a_wall_clock_step_does_not_move_the_duration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An NTP correction mid-fault must not invent hours of downtime.

        ``t`` is an absolute stamp another machine correlates and stays on
        ``time.time()``; ``for_seconds`` answers how long the probe has been
        failing and must not move when the date does.
        """
        from strands_robots.mesh import core

        real_time = core.time.time
        m, bus = _hardware_peer()
        bus.exc = ConnectionError("Port is in use!")
        first = m._read_state()

        monkeypatch.setattr(core.time, "time", lambda: real_time() + 86_400.0)
        second = m._read_state()

        assert first is not None and second is not None
        assert second["degraded"]["hw_joints"]["for_seconds"] < 60.0, (
            "for_seconds tracked the wall clock instead of elapsed time"
        )
        assert second["t"] - first["t"] > 3600.0, "premise: the wall clock really moved"

    def test_the_monotonic_stamp_stays_off_the_wire(self) -> None:
        """Seconds of local process uptime mean nothing to another machine."""
        m, bus = _hardware_peer()
        bus.exc = ConnectionError("Port is in use!")
        out = m._read_state()

        assert out is not None
        assert set(out["degraded"]["hw_joints"]) == {"reason", "detail", "failures", "for_seconds"}


class TestTheReportIsUnchanged:
    """This changes the snapshot, not the log."""

    def test_a_flapping_probe_still_warns_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """The wire tracks every transition; the log keeps its one line.

        Clearing the log gate on recovery would re-arm the warning, so a probe
        flapping at ``STATE_HZ`` would trade ten silent ticks for a warning per
        tick -- which is the noise the once-per-category gate exists to prevent.
        """
        m, bus = _hardware_peer()
        with caplog.at_level(logging.DEBUG, logger=CORE_LOGGER):
            for _ in range(4):
                bus.exc = ConnectionError("Port is in use!")
                m._read_state()
                bus.exc = None
                m._read_state()

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, f"a flapping probe must not warn per tick, got {len(warnings)}"


class TestEveryProbeIsHeldToTheContract:
    """Derived from the source, so a fifth probe cannot skip half of it."""

    def test_every_reported_category_also_notes_its_recovery(self) -> None:
        """A category that can be set and never cleared would go stale forever."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(Mesh._read_state)))
        reported: set[str] = set()
        cleared: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            arg = node.args[0]
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                continue
            name = ast.unparse(node.func)
            if name.endswith("_warn_read_state_once"):
                reported.add(arg.value)
            elif name.endswith("_note_read_state_ok"):
                cleared.add(arg.value)

        assert reported == EXPECTED_CATEGORIES, f"reported categories drifted: {reported}"
        assert reported - cleared == set(), (
            f"these probes report a failure and never clear it: {sorted(reported - cleared)}"
        )

    def test_the_block_is_attached_unconditionally(self) -> None:
        """Attaching it inside a probe's ``try`` would lose it to that probe."""
        body = textwrap.dedent(inspect.getsource(Mesh._read_state))
        tree = ast.parse(body)
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)
        guarded = {
            id(inner)
            for node in ast.walk(func)
            if isinstance(node, ast.Try)
            for stmt in node.body
            for inner in ast.walk(stmt)
        }
        attaches = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("_degraded_probes")
        ]
        assert attaches, "premise: _read_state builds the block"
        for call in attaches:
            assert id(call) not in guarded, "the block must not be built inside a probe's try"
