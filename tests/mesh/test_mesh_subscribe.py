"""subscribe(): a device's state and events arrive over the mesh with the in-process payload."""

from __future__ import annotations

import threading
import time

import pytest

from strands_robots import Robot


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.01)
    return pred()


@pytest.fixture
def served(mesh_pair):
    a, b = mesh_pair
    arm = Robot("so101")
    a.mount(arm, device_id="so101-a", state_hz=50)
    assert b.discover("so101-a", min_count=1, timeout=3.0)
    yield arm, a, b
    a.close()  # unmount before the robot goes
    arm.close()


def test_state_payload_is_the_in_process_payload(served):
    arm, _, b = served
    local, remote = [], []
    unsub_local = arm.subscribe("state", local.append, hz=50)
    unsub = b.subscribe("so101-a", "state", remote.append)
    assert _wait(lambda: len(remote) >= 5 and local)
    unsub()
    unsub_local()
    assert set(remote[-1]) == set(local[-1])
    assert remote[-1]["shoulder_pan.pos"] == pytest.approx(arm.observe().joints["shoulder_pan"], abs=1e-6)
    seqs = [p["seq"] for p in remote]
    assert seqs == sorted(seqs)
    n = len(remote)
    time.sleep(0.2)
    assert len(remote) == n, "unsubscribe stops delivery"


def test_events_are_forwarded(served):
    arm, _, b = served
    got = []
    done = threading.Event()
    b.subscribe("so101-a", "estop", lambda p: (got.append(p), done.set()))
    time.sleep(0.2)
    arm.estop()
    assert done.wait(3.0)
    assert got[0]["engaged"] is True and got[0]["device_id"] == arm.id


def test_unknown_event_is_refused(served):
    _, _, b = served
    with pytest.raises(ValueError, match="unknown event"):
        b.subscribe("so101-a", "teleport", print)
