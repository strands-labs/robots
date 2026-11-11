"""Every ``on_disconnect`` applied leaves one structured ``strands_robots.audit`` record: who it recalled the arm
from, why, what it wrote and whether that worked."""

from __future__ import annotations

import logging
import time

import pytest

from strands_robots import Robot
from strands_robots.mesh import wire
from strands_robots.mesh.watchdog import MeshWatchdog
from strands_robots.robot.safety import Watchdog, apply_on_disconnect


@pytest.fixture
def arm():
    robot = Robot("so101")
    yield robot
    robot.close()


def _records(caplog) -> list[dict]:
    keys = ("tenant", "device_id", "peer", "reason", "applied", "outcome")
    return [
        {k: getattr(r, k) for k in keys}
        for r in caplog.records
        if r.name == "strands_robots.audit" and getattr(r, "rpc", None) == "on_disconnect"
    ]


def _wait_disarmed(dog: MeshWatchdog) -> None:
    end = time.monotonic() + 2.0
    while dog._armed.is_set() and time.monotonic() < end:
        time.sleep(0.005)


def test_a_quiet_mover_leaves_a_record_naming_it(arm, caplog):
    dog = MeshWatchdog(arm, 100, [], "lab")
    try:
        with caplog.at_level(logging.INFO, logger="strands_robots.audit"):
            dog.on_rpc("act", "mover")(True)
            _wait_disarmed(dog)
    finally:
        dog.close()
    [rec] = _records(caplog)
    assert rec["tenant"] == "lab" and rec["device_id"] == arm.id and rec["peer"] == "mover"
    assert "quiet" in rec["reason"] and rec["applied"] == "hold" and rec["outcome"] == "ok"


def test_a_vanished_mover_leaves_a_record_naming_it(arm, caplog):
    dog = MeshWatchdog(arm, 10_000, [], "lab", "left-arm")
    try:
        with caplog.at_level(logging.INFO, logger="strands_robots.audit"):
            dog.on_rpc("act", "mover")(True)
            dog._on_peer(wire.key("peer", "lab", "mover"), False)
    finally:
        dog.close()
    [rec] = _records(caplog)
    assert rec["device_id"] == "left-arm", "the record names the device id the arm is mounted under"
    assert rec["peer"] == "mover" and "vanished" in rec["reason"] and rec["outcome"] == "ok"


def test_an_apply_that_fails_is_recorded_with_its_error(arm, caplog):
    def broken() -> None:
        raise OSError("bus gone")

    arm.body.hold = broken
    dog = MeshWatchdog(arm, 10_000, [], "lab")
    try:
        with caplog.at_level(logging.INFO, logger="strands_robots.audit"):
            dog.on_rpc("act", "mover")(True)
            dog._on_peer(wire.key("peer", "lab", "mover"), False)
    finally:
        dog.close()
    [rec] = _records(caplog)
    assert rec["peer"] == "mover" and rec["outcome"] == "OSError"


def test_a_latched_estop_is_recorded_as_writing_nothing(arm, caplog):
    arm.estop()
    with caplog.at_level(logging.INFO, logger="strands_robots.audit"):
        apply_on_disconnect(arm, reason="test")
    [rec] = _records(caplog)
    assert rec["applied"].startswith("nothing") and rec["outcome"] == "ok"


def test_a_local_watchdog_lapse_is_recorded(arm, caplog):
    now = [0.0]
    dog = Watchdog(arm, 100, clock=lambda: now[0])
    now[0] = 0.5
    with caplog.at_level(logging.INFO, logger="strands_robots.audit"):
        assert dog.poll()
    [rec] = _records(caplog)
    assert rec["reason"] == "watchdog lapse" and rec["peer"] is None and rec["outcome"] == "ok"
