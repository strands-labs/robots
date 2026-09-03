"""``lockout_elapsed_s`` is a duration, so it rides the monotonic clock.

``Mesh`` stamps two timestamps together whenever the e-stop lockout engages:
``_last_estop_ts`` (wall) and ``_last_estop_mono`` (monotonic). The split
exists because the two answer different questions. ``_last_estop_ts`` is an
absolute instant, published as the envelope's ``t`` where the wall clock is the
only domain a remote peer can interpret. ``_last_estop_mono`` is what durations
are measured from, because the wall clock can be adjusted underneath a running
process.

``test_corroboration_clock_domain`` already pins that rule for the 0.2s
corroboration window. The resume path reports a second duration --
``lockout_elapsed_s``, the field the audit trail keeps to answer how long the
fleet was halted, echoed into the resume envelope and the operator log line --
and it was reconstructed from two wall-clock reads. A clock adjustment during
the lockout (chrony stepping an RTC-less robot at its first NTP sync, a VM
resumed from suspend, an operator correcting the clock) therefore moved the
reported duration without any time being held: inflated by a forward step,
negative by a backward one.

The pins below drive the real ``_resume_lockout`` and read the three places the
duration surfaces, plus the two controls that keep the fix honest -- an
undisturbed clock still reports the time actually held, and ``t`` stays on the
wall clock.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import core

_CODE = "operator-secret"
_WALL_AT_ENGAGE = 1_800_000_000.0
_MONO_AT_ENGAGE = 500_000.0


class _Clock:
    """A wall clock that can be stepped beside a monotonic clock that cannot."""

    def __init__(self) -> None:
        self.wall = _WALL_AT_ENGAGE
        self.mono = _MONO_AT_ENGAGE

    def hold(self, seconds: float, *, wall_step: float = 0.0) -> None:
        """Hold the lockout for *seconds*, with the wall clock also stepped."""
        self.mono += seconds
        self.wall += seconds + wall_step

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clk = _Clock()
    monkeypatch.setattr(core.time, "time", clk.time)
    monkeypatch.setattr(core.time, "monotonic", clk.monotonic)
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)
    return clk


def _engaged_operator(monkeypatch: pytest.MonkeyPatch) -> tuple[core.Mesh, dict[str, Any], list[dict]]:
    """An operator peer holding an engaged lockout, with its sinks captured.

    The lockout is stamped exactly as ``emergency_stop`` stamps it -- both
    timestamps, from the clock under test -- so the pins read production state
    rather than a half-wired approximation of it.
    """
    mesh = core.Mesh(robot=object(), peer_id="operator-1")
    mesh._running = True
    audited: list[dict] = []
    mesh.publish_safety_event = lambda **kw: audited.append(kw)  # type: ignore[method-assign]
    monkeypatch.setattr(mesh, "_local_session_zid", lambda: None)
    monkeypatch.setattr(mesh, "_safety_wire_zid", lambda key: None)
    published: dict[str, Any] = {}
    monkeypatch.setattr(core, "put", lambda key, payload: published.update(payload))

    mesh._estop_lockout.set()
    mesh._last_estop_ts = core.time.time()
    mesh._last_estop_mono = core.time.monotonic()
    return mesh, published, audited


def _resume(mesh: core.Mesh) -> None:
    assert mesh._resume_lockout(_CODE) == {"status": "ok"}


class TestTheReportedLockoutIsTheTimeTheFleetWasHeld:
    """The duration cannot be moved by an adjustment of the wall clock."""

    def test_an_undisturbed_clock_reports_the_time_the_fleet_was_held(
        self, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: with no clock adjustment both domains agree, so this
        passes with or without the fix -- it is what gives the two pins below
        their meaning."""
        mesh, published, _ = _engaged_operator(monkeypatch)
        clock.hold(12.5)
        _resume(mesh)
        assert published["lockout_elapsed_s"] == pytest.approx(12.5)

    @pytest.mark.parametrize(
        ("adjustment", "wall_step"),
        [
            ("chrony steps the clock forward one hour", 3600.0),
            ("an operator corrects the clock back two hours", -7200.0),
            ("an RTC-less robot syncs to NTP for the first time", -100_000_000.0),
        ],
    )
    def test_a_wall_clock_adjustment_does_not_move_the_reported_lockout(
        self, clock: _Clock, monkeypatch: pytest.MonkeyPatch, adjustment: str, wall_step: float
    ) -> None:
        """The fleet is held for 12.5s in every row; only the wall clock moves."""
        mesh, published, _ = _engaged_operator(monkeypatch)
        clock.hold(12.5, wall_step=wall_step)
        _resume(mesh)
        assert published["lockout_elapsed_s"] == pytest.approx(12.5), adjustment

    def test_a_backward_correction_cannot_report_a_negative_lockout(
        self, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A duration read off a monotonic clock cannot be negative, so the
        audit trail can never claim the fleet resumed before it stopped."""
        mesh, published, audited = _engaged_operator(monkeypatch)
        clock.hold(12.5, wall_step=-7200.0)
        _resume(mesh)
        assert published["lockout_elapsed_s"] >= 0.0
        resume_ok = [e for e in audited if e["event_type"] == "resume_ok"]
        assert [e["payload"]["lockout_elapsed_s"] for e in resume_ok] == [pytest.approx(12.5)]

    def test_every_sink_reports_the_same_duration(
        self, clock: _Clock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The audit record, the resume envelope and the operator log line are
        three views of one measurement, so a forward step must not leave them
        describing different lockouts."""
        mesh, published, audited = _engaged_operator(monkeypatch)
        clock.hold(30.0, wall_step=3600.0)
        with caplog.at_level(logging.WARNING, logger=core.logger.name):
            _resume(mesh)

        audited_elapsed = [e["payload"]["lockout_elapsed_s"] for e in audited if e["event_type"] == "resume_ok"]
        assert audited_elapsed == [pytest.approx(30.0)]
        assert published["lockout_elapsed_s"] == pytest.approx(30.0)
        assert "resume after 30.0s lockout" in caplog.text


class TestTheEnvelopeInstantStaysOnTheWallClock:
    """The complement of the rule: ``t`` is an instant, not a duration."""

    def test_the_envelope_timestamp_is_a_wall_clock_instant(
        self, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A receiver compares ``t`` against its own wall clock for freshness,
        so moving it into the monotonic domain would make every resume look
        decades stale. This fails if the two timestamps are conflated."""
        mesh, published, _ = _engaged_operator(monkeypatch)
        clock.hold(12.5)
        _resume(mesh)
        assert published["t"] == pytest.approx(clock.wall)

    def test_the_proof_binds_the_duration_the_envelope_carries(
        self, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The override proof is recomputed by every receiver over the body it
        actually sees, so the duration the issuer measured is the duration
        bound into the MAC."""
        mesh, published, _ = _engaged_operator(monkeypatch)
        clock.hold(12.5, wall_step=3600.0)
        _resume(mesh)
        expected = core.hmac.new(
            _CODE.encode(),
            json.dumps(
                {
                    "peer_id": "operator-1",
                    "t": published["t"],
                    "lockout_elapsed_s": published["lockout_elapsed_s"],
                    "proof_nonce": published["proof_nonce"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            "sha256",
        ).hexdigest()
        assert published["override_proof"] == expected


class TestAResumeIsStillAcceptedEndToEnd:
    """A receiver clears its lockout from the envelope the issuer minted."""

    def test_a_receiver_recovers_from_the_minted_envelope(self, clock: _Clock, monkeypatch: pytest.MonkeyPatch) -> None:
        mesh, published, _ = _engaged_operator(monkeypatch)
        clock.hold(12.5, wall_step=-7200.0)
        _resume(mesh)

        receiver = core.Mesh(robot=object(), peer_id="robot-1")
        receiver.publish_safety_event = MagicMock()  # type: ignore[method-assign]
        receiver._estop_lockout.set()
        sample = MagicMock()
        sample.payload.to_bytes.return_value = json.dumps(published).encode()
        receiver._on_safety_resume(sample)

        assert receiver._estop_lockout.is_set() is False, "the fleet would stay e-stopped"
