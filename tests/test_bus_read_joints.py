"""A broken camera must not be able to hide an arm's joints.

Measured on real hardware: an arm had published ZERO joints for eleven hours while its
mesh presence showed it connected and non-stale. The only trace was one line five
seconds after it started - `state probe 'hw_joints' failed, that section of the snapshot is
omitted (further failures logged at debug): RuntimeError('OpenCVCamera(1) read failed')` -
and then silence by design.

Cause, read in lerobot's source: SOFollower.get_observation() sync-reads the motors FIRST,
then loops the cameras calling read_latest(). A camera that raises throws away the joint
positions already in hand. Everything joints-shaped went through that one call - the fleet
snapshot, the history traces, motion detection and teleop's publisher - so one dead USB
camera silently disarmed the whole arm.
"""

from __future__ import annotations

import threading

import pytest

from strands_robots.bus_access import bus_lock, read_joints


class _Bus:
    def __init__(self, positions: dict[str, float], *, accepts_retry: bool = True) -> None:
        self.positions = positions
        self.accepts_retry = accepts_retry
        self.calls: list[dict] = []
        self.locked_during_read: bool | None = None
        self.owner: object | None = None

    def sync_read(self, register: str, num_retry: int | None = None):
        if num_retry is not None and not self.accepts_retry:
            raise TypeError("sync_read() got an unexpected keyword argument 'num_retry'")
        self.calls.append({"register": register, "num_retry": num_retry})
        if self.owner is not None:
            # The lock must be HELD while the wire is in use, or this reintroduces
            # the multi-reader collision bus_access exists to prevent. Probed from
            # ANOTHER thread, because the lock is an RLock: the reading thread can
            # always re-acquire it, so asking here would prove nothing.
            lock = bus_lock(self.owner)
            result: list[bool] = []

            def probe() -> None:
                got = lock.acquire(blocking=False)
                result.append(got)
                if got:
                    lock.release()

            t = threading.Thread(target=probe)
            t.start()
            t.join(2)
            self.locked_during_read = result == [False]
        return dict(self.positions)


class _Cfg:
    def __init__(self, retries: int | None = 3) -> None:
        if retries is not None:
            self.num_read_retries = retries


class _Arm:
    """An arm whose camera is dead - exactly arm-1's state for eleven hours."""

    def __init__(self, bus: _Bus | None = None, *, retries: int | None = 3) -> None:
        self.bus = bus
        self.config = _Cfg(retries)
        self.is_connected = True
        self.observation_calls = 0

    def get_observation(self):
        self.observation_calls += 1
        raise RuntimeError("OpenCVCamera(1) read failed (status=False).")


POSITIONS = {
    "shoulder_pan": 1.0,
    "shoulder_lift": -46.0,
    "elbow_flex": 92.0,
    "wrist_flex": 3.0,
    "wrist_roll": 170.0,
    "gripper": 2.0,
}


class TestJointsSurviveADeadCamera:
    def test_the_incident_joints_are_returned_while_get_observation_raises(self) -> None:
        arm = _Arm(_Bus(POSITIONS))
        joints = read_joints(arm)
        assert joints == {f"{m}.pos": v for m, v in POSITIONS.items()}
        assert arm.observation_calls == 0, "the camera path must not be touched at all"

    def test_the_old_path_really_did_lose_them(self) -> None:
        # Pins the bug itself: through get_observation there are no joints to have.
        from strands_robots.bus_access import read_observation

        with pytest.raises(RuntimeError, match="OpenCVCamera"):
            read_observation(_Arm(_Bus(POSITIONS)))

    def test_the_shape_matches_get_observations_joint_half(self) -> None:
        # Callers (positions_from_observation, the hw_joints probe) parse ".pos".
        joints = read_joints(_Arm(_Bus(POSITIONS)))
        assert all(k.endswith(".pos") for k in joints)
        assert "wrist_roll.pos" in joints


class TestItStillSharesTheBusLock:
    def test_the_read_happens_under_the_device_lock(self) -> None:
        bus = _Bus(POSITIONS)
        arm = _Arm(bus)
        bus.owner = arm
        read_joints(arm)
        assert bus.locked_during_read is True

    def test_a_thread_holding_the_lock_blocks_the_read(self) -> None:
        arm = _Arm(_Bus(POSITIONS))
        started = threading.Event()
        done = threading.Event()

        def reader() -> None:
            started.set()
            read_joints(arm)
            done.set()

        with bus_lock(arm):
            t = threading.Thread(target=reader, daemon=True)
            t.start()
            started.wait(1)
            assert not done.wait(0.2), "read_joints must wait its turn on the wire"
        assert done.wait(2)


class TestDriverVariations:
    def test_the_configured_retry_count_is_passed_through(self) -> None:
        bus = _Bus(POSITIONS)
        read_joints(_Arm(bus, retries=5))
        assert bus.calls[-1] == {"register": "Present_Position", "num_retry": 5}

    def test_a_bus_without_the_retry_keyword_is_still_read(self) -> None:
        bus = _Bus(POSITIONS, accepts_retry=False)
        joints = read_joints(_Arm(bus))
        assert joints["gripper.pos"] == 2.0
        assert bus.calls[-1]["num_retry"] is None

    def test_a_driver_with_no_num_read_retries_is_read_plainly(self) -> None:
        bus = _Bus(POSITIONS)
        read_joints(_Arm(bus, retries=None))
        assert bus.calls[-1]["num_retry"] is None

    def test_a_driver_with_no_bus_falls_back_to_the_full_observation(self) -> None:
        # A driver whose only reader is get_observation cannot be asked for less;
        # returning nothing would be worse than returning frames we ignore.
        class _NoBus:
            def __init__(self) -> None:
                self.reads = 0

            def get_observation(self):
                self.reads += 1
                return {"shoulder_pan.pos": 1.0, "top": object()}

        dev = _NoBus()
        obs = read_joints(dev)
        assert dev.reads == 1
        assert obs["shoulder_pan.pos"] == 1.0

    def test_a_bus_answering_with_a_non_mapping_takes_the_observation_fallback(self) -> None:
        """A ``bus`` that is not a motor bus is detected, not trusted.

        Handing the non-mapping straight back was worse than useless: this
        function documents a mapping and every joints consumer iterates it, so
        the caller failed on ``.items()`` a frame later with nothing naming the
        cause. Measured on the state probe, which swallowed the resulting
        ``TypeError`` and published no state message at all -- so the fleet saw
        a peer that had gone quiet rather than one with an unreadable bus.
        ``bus`` is an attribute name a wrapper or a proxy can also use, so this
        is reachable without a mock.
        """

        class _Weird(_Bus):
            def sync_read(self, register: str, num_retry: int | None = None):
                return ["not", "a", "mapping"]

        class _WeirdBusArm:
            def __init__(self) -> None:
                self.bus = _Weird({})
                self.config = _Cfg(3)
                self.is_connected = True
                self.observation_calls = 0

            def get_observation(self):
                self.observation_calls += 1
                return {"shoulder_pan.pos": 1.0, "top": object()}

        dev = _WeirdBusArm()
        out = read_joints(dev)

        assert isinstance(out, dict), "a non-mapping bus answer must not reach the caller"
        assert out["shoulder_pan.pos"] == 1.0
        assert dev.observation_calls == 1, "the fallback observation is what produced the reading"

    def test_the_fallback_reports_the_drivers_own_error_rather_than_a_non_mapping(self) -> None:
        """The fallback is a route to the reading, not a way to swallow a failure.

        A device with an unreadable bus AND a raising ``get_observation()`` has
        nothing to give, and the honest answer is the driver's own exception --
        which is what a caller can act on. Returning the non-mapping instead
        reported success and failed later somewhere else.
        """

        class _Weird(_Bus):
            def sync_read(self, register: str, num_retry: int | None = None):
                return ["not", "a", "mapping"]

        arm = _Arm(_Weird({}))  # its get_observation() raises the dead-camera error

        with pytest.raises(RuntimeError, match="OpenCVCamera"):
            read_joints(arm)
