"""A camera is read on its own capture thread: a tick takes the latest frame and never waits on a device.

Checked by structure, not wall-clock: devices whose open waits for the other device's open (only side-by-side opens
get through), and devices that block on an event (only a read that never touches the device returns).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from test_drivers import make_driver

from strands_robots import Robot
from strands_robots.robot.drivers import cameras as mod
from strands_robots.robot.drivers.cameras import Cameras


class _Device:
    """A 640x480 device; ``gate`` set: every read after the first waits on it (a camera between two frames)."""

    barrier: threading.Barrier | None = None
    gate: threading.Event | None = None
    made: list = []

    def __init__(self, index, spec):
        self.index, self.opened_on, self.read_on, self.closed_on, self.reads = (
            index,
            threading.get_ident(),
            set(),
            None,
            0,
        )
        if _Device.barrier is not None:
            _Device.barrier.wait()  # every camera's open must be under way at once
        _Device.made.append(self)

    def read(self):
        self.read_on.add(threading.get_ident())
        if self.reads and _Device.gate is not None:
            _Device.gate.wait(5)
        time.sleep(0.002)
        self.reads += 1
        return np.full((480, 640, 3), self.reads % 256, np.uint8)

    def close(self):
        self.closed_on = threading.get_ident()


@pytest.fixture
def device(monkeypatch):
    _Device.barrier, _Device.gate, _Device.made = None, None, []
    monkeypatch.setattr(mod, "_CVCapture", _Device)
    yield _Device
    if _Device.gate is not None:
        _Device.gate.set()


TWO = {"front": {"index": 0, "width": 640, "height": 480}, "wrist": {"index": 1, "width": 640, "height": 480}}


def test_cameras_open_side_by_side(device):
    device.barrier = threading.Barrier(2, timeout=2)  # serial opens break it: the 2nd never begins
    cams = Cameras(TWO)
    cams.open()
    assert sorted(d.index for d in device.made) == [0, 1]
    assert all(d.opened_on != threading.get_ident() for d in device.made)
    cams.close()


def test_a_read_returns_the_latest_frames_while_every_device_waits_for_its_next(device):
    cams = Cameras(TWO)
    cams.open()
    device.gate = threading.Event()  # from here each device blocks between frames
    done: dict = {}
    reader = threading.Thread(target=lambda: done.update(frames=[cams.read() for _ in range(3)]), daemon=True)
    reader.start()
    reader.join(2.0)  # generous: no device delivers in this time, so only a read of the kept frame finishes
    assert not reader.is_alive(), "Cameras.read waited on a device"
    assert all(set(f) == {"front", "wrist"} for f in done["frames"])
    assert all(threading.get_ident() not in d.read_on for d in device.made)
    assert set(cams.stamps) == {"front", "wrist"} and all(a >= 0 for a in cams.ages().values())
    device.gate.set()
    cams.close()


def test_close_stops_every_capture_thread_and_each_device_closes_on_its_own_thread(device):
    cams = Cameras(TWO)
    cams.read()
    threads = [g.thread for g in cams._caps.values()]
    cams.close()
    assert not any(t.is_alive() for t in threads)
    assert all(d.closed_on is not None and d.closed_on == d.opened_on for d in device.made)
    cams.close()  # twice is fine


def test_a_camera_that_stopped_delivering_is_left_out_and_a_failing_one_is_named(device, monkeypatch):
    cams = Cameras(TWO, max_age_s=0.2)
    cams.open()
    device.gate = threading.Event()
    time.sleep(0.4)
    assert cams.read() == {}  # as a failed read today: no stale frame passed off as now
    device.gate.set()
    cams.close()

    def broken(self):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(_Device, "read", broken)
    with pytest.raises(OSError, match=r"camera 'front': .*Input/output error"):
        Cameras({"front": 0}).open()


def test_a_robot_opens_its_cameras_at_connect(device):
    d, _ = make_driver()
    d.cameras = Cameras(TWO)
    arm = Robot("so101", driver=d, twin=False)
    assert len(device.made) == 2 and all(dev.reads for dev in device.made)  # open, and delivering, before any observe
    assert set(arm.observe().images) == {"front", "wrist"}
    arm.close()
    assert all(dev.closed_on is not None for dev in device.made)


def test_a_device_with_no_frame_is_not_spun_on_and_concurrent_readers_open_each_device_once(device, monkeypatch):
    cams = Cameras(TWO)
    readers = [threading.Thread(target=cams.read) for _ in range(4)]
    for r in readers:
        r.start()
    for r in readers:
        r.join(5)
    assert sorted(d.index for d in device.made) == [0, 1]
    cams.close()

    unplugged = {"n": 0}

    def nothing(self):
        unplugged["n"] += 1
        return None  # what OpenCV answers, at once, for a camera pulled out

    monkeypatch.setattr(_Device, "read", nothing)
    grabber = mod.Grabber("wrist", lambda: _Device(1, None))
    time.sleep(0.2)
    grabber.stop()
    assert unplugged["n"] < 200  # a spin would be ~10^5 calls; the idle wait keeps it to ~40
