"""The run kernel: one body read per tick (K1), no catch-up burst (K3), one clock per body (K5), one step (K7).

A hardware body here is a ``CountingDriver`` on a ``VirtualClock`` the test owns: every bus read and write is
logged with its time, and nothing waits for real.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from strands_robots import JointState, Robot, Simulation
from strands_robots.description import Description
from strands_robots.robot.clock import MONOTONIC
from strands_robots.robot.run import lockstep
from strands_robots.robot.run.pace import Pacer


class VirtualClock:
    """``sleep`` and ``advance`` move it; nothing waits for real."""

    def __init__(self, t0: float = 1000.0) -> None:
        self.t = t0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)

    advance = sleep


class CountingDriver:
    """A hardware body on no bus: a read returns the last write; every read and write is logged with its time."""

    supports_position_hold, calibration, resolution = True, None, None

    def __init__(self, clock=None) -> None:
        self.clock = clock
        self.names = tuple(Description.load("so101").action_keys)
        self.q = np.zeros(len(self.names))
        self.reads: list[float] = []
        self.writes: list[tuple[float, np.ndarray]] = []
        self.holds = 0

    def _now(self) -> float:
        return self.clock.now() if self.clock is not None else time.monotonic()

    def read(self) -> JointState:
        self.reads.append(self._now())
        z = np.zeros(len(self.names))
        return JointState(self.names, self.q.copy(), z, z.copy())

    def write(self, targets) -> None:
        self.q = np.asarray(targets, dtype=float).copy()
        self.writes.append((self._now(), self.q))

    def hold(self) -> None:
        self.holds += 1

    def pose(self):
        return None

    def images(self):
        return {}

    def extras(self):
        return {}

    def estop(self) -> None: ...

    def calibrate(self, *, interactive: bool = True):
        raise NotImplementedError

    def close(self) -> None: ...


def _hw():
    clock = VirtualClock()
    drv = CountingDriver(clock)
    return Robot("so101", driver=drv, twin=False, clock=clock), drv


def _stalls(drv, at: int, seconds: float, base: float = 0.0):
    """A policy whose call ``at`` takes ``seconds`` of inference on the body's clock (the others ``base``)."""
    calls = [0]

    def pol(obs):
        calls[0] += 1
        drv.clock.advance(seconds if calls[0] == at else base)
        return {"shoulder_pan": 0.0}

    return pol


def test_k1_one_bus_read_per_tick():
    robot, drv = _hw()
    robot.run(lambda obs: {"shoulder_pan": 0.0}, steps=25, hz=200)
    assert len(drv.reads) == 1 + 25 + 1  # bind's requirements check, one per tick, the RunResult's final observation
    assert len(drv.writes) == 25


@pytest.mark.parametrize(("action", "reads"), [(np.zeros(6), 1), ({"shoulder_pan": 0.1}, 2)])
def test_k1_act_reads_once_for_its_answer_and_once_more_to_fill_a_mapping(action, reads):
    robot, drv = _hw()
    robot.act(action)
    assert len(drv.reads) == reads and len(drv.writes) == 1


def test_k1_command_never_reads():
    robot, drv = _hw()
    robot.command(robot.coerce({"shoulder_pan": 0.1}, measured=np.zeros(6)))
    assert drv.reads == [] and len(drv.writes) == 1


def test_k3_no_catch_up_burst_after_a_stall():
    robot, drv = _hw()
    res = robot.run(_stalls(drv, 5, 0.3, base=0.002), steps=20, hz=50)  # call 5 stalls 15 periods
    gaps = np.diff([t for t, _ in drv.writes])
    assert gaps.min() >= (1 - 1e-9) / 50, f"burst: min gap {gaps.min() * 1e3:.1f} ms at 50 Hz"
    assert res.deadline_misses == 1 and res.steps == 20
    assert res.worst_late_s == pytest.approx(0.3 - 1 / 50)  # landed 300 ms after its start, deadline at 20 ms


def test_k3_on_miss_fault_ends_the_run_at_the_late_tick():
    robot, drv = _hw()
    res = robot.run(_stalls(drv, 3, 0.3), steps=20, hz=50, on_miss="fault")
    assert (res.stopped_by, res.steps, res.deadline_misses) == ("deadline", 3, 1)


def test_k3_seconds_are_the_commanded_span_of_the_writes():
    robot, drv = _hw()
    res = robot.run(_stalls(drv, 3, 0.1), steps=10, hz=50)
    w = [t for t, _ in drv.writes]
    assert res.seconds == pytest.approx(w[-1] - w[0] + 1 / 50) == pytest.approx(10 / 50 + 0.1)
    assert res.wall_seconds >= 0


@pytest.mark.parametrize("hz", [10.0, 30.0, 50.0, 100.0, 200.0])
def test_k3_pacer_writes_are_never_closer_than_one_period_for_any_inference_times(hz):
    """100 random sequences per rate: gaps >= dt, a miss is exactly a late write, and the writes never drift."""
    rng, dt = np.random.default_rng(int(hz)), 1 / hz
    for _ in range(100):
        clock, work = VirtualClock(), rng.uniform(0.0, 0.5, rng.integers(1, 60)) * (rng.random() < 0.5 or 0.01)
        p = Pacer(dt, now=clock.now, sleep=clock.sleep)
        writes, misses = [], 0
        for w in work:
            p.start()
            start = clock.now()
            clock.advance(w)
            p.before_write()
            writes.append(clock.now())
            p.after_write()
            misses += writes[-1] > start + dt + 1e-12
        assert (np.diff(writes) >= dt - 1e-9).all() and p.misses == misses
        assert writes[-1] - writes[0] <= sum(max(w, dt) for w in work) + 1e-9


def test_k3_a_write_that_takes_time_or_a_sleep_that_overshoots_is_not_a_miss():
    """A real bus write takes ms and a real sleep overshoots: neither is lateness, before or after a re-phase."""
    clock, dt = VirtualClock(), 1 / 50
    p = Pacer(dt, now=clock.now, sleep=lambda s: clock.advance(s + 0.001))  # every sleep 1 ms late
    for k in range(50):
        p.start()
        clock.advance(0.3 if k == 10 else 0.005)  # inference; tick 10 stalls
        p.before_write()
        clock.advance(0.002)  # the write itself
        p.after_write()
    assert p.misses == 1 and p.worst_late == pytest.approx(0.3 - dt, abs=0.002)


@pytest.mark.parametrize(("eps", "healthy"), [(100e-6, 2000), (1.5e-3, 500)])
def test_k3_lateness_is_measured_from_the_write_that_landed_not_a_grid_the_writes_left_behind(eps, healthy):
    """One tick that nearly fills its period (17.95 ms + 2 ms write at 50 Hz, sleeps overshooting by ``eps``) used
    to tip the pacer into a regime where every later write was held to its floor and the grid stayed one overshoot
    further behind each tick, so a 1 ms overrun hundreds of ticks later was reported as hundreds of ms late."""
    clock, dt = VirtualClock(), 1 / 50
    p = Pacer(dt, now=clock.now, sleep=lambda s: clock.advance(s + eps))
    writes: list[float] = []

    def tick(inference: float) -> None:
        p.start()
        clock.advance(inference)
        p.before_write()
        writes.append(clock.now())
        clock.advance(0.002)  # the write itself
        p.after_write()

    for k in range(healthy + 2):
        tick(0.01795 if k == 0 else 0.005)  # one tick that almost saturates its period, then healthy ticks
    tick(0.019)  # a 1 ms overrun of dt - write
    assert p.misses == 1
    late_ms = f"{p.worst_late * 1e3:.1f} ms reported for a 1 ms overrun"
    assert p.worst_late == pytest.approx(0.001, abs=eps + 1e-9), late_ms
    assert min(np.diff(writes)) >= dt - 1e-9


def test_k3_writes_stay_one_period_apart_across_runs_estops_and_resumes():
    """A random history of runs, stalls, e-stops and resumes on one body: no two writes < one period apart."""
    rng = np.random.default_rng(0)
    robot, drv = _hw()
    for _ in range(40):
        verb = rng.choice(["run", "run", "estop", "resume"])
        if verb == "estop":
            robot.estop()
        elif verb == "resume":
            robot.resume(approved_by="op")
        else:
            res = robot.run(_stalls(drv, int(rng.integers(1, 6)), float(rng.uniform(0, 0.2))), steps=8, hz=50)
            assert res.stopped_by == ("estop" if robot.estopped else "steps")
    gaps = np.diff([t for t, _ in drv.writes])
    assert len(gaps) > 20 and gaps.min() >= 1 / 50 - 1e-9


def test_k3_estop_during_a_pacing_wait_writes_nothing_more():
    drv = CountingDriver()
    robot = Robot("so101", driver=drv, twin=False, clock=MONOTONIC)
    calls = [0]

    def pol(obs):
        calls[0] += 1
        if calls[0] == 2:
            threading.Timer(0.05, robot.estop).start()  # lands mid-wait at 2 Hz (the period is 500 ms)
        return {}

    t0 = time.monotonic()
    res = robot.run(pol, steps=10, hz=2)
    assert res.stopped_by == "estop" and len(drv.writes) <= 2
    assert time.monotonic() - t0 < 1.5  # it did not wait out the remaining periods


def test_k5_observation_t_is_capture_time_on_the_body_clock():
    robot, drv = _hw()
    drv.clock.advance(1.5)
    assert robot.observe().t == drv.clock.now() == drv.reads[-1]


def test_k5_latch_and_run_seconds_on_the_body_clock():
    robot, drv = _hw()
    t = drv.clock.now()
    robot.estop()
    assert robot.estop_latch.engaged_at == t
    robot.resume(approved_by="op")
    res = robot.run(lambda obs: (drv.clock.advance(0.001), {})[1], steps=10, hz=50)
    assert res.deadline_misses == 0 and res.seconds == pytest.approx(10 / 50) and res.hz_achieved == pytest.approx(50)


class BrokenClock(VirtualClock):
    def now(self) -> float:
        raise OSError("ptp: no grandmaster")


def _announced(robot) -> list[tuple[str, dict]]:
    got: list[tuple[str, dict]] = []
    for event in ("estop", "fault"):
        robot.subscribe(event, lambda payload, event=event: got.append((event, payload)))
    return got


def test_k5_a_clock_that_raises_never_stops_the_estop_from_latching_reaching_the_body_or_being_announced():
    drv = CountingDriver()
    stops = []
    drv.estop = lambda: stops.append(1)  # type: ignore[method-assign]
    robot = Robot("so101", driver=drv, twin=False, clock=BrokenClock())
    got = _announced(robot)
    robot.estop(by="op")  # returns: the stop latched and reached the body, only its time stamp is missing
    assert robot.estopped and robot.estop_latch.count == 1 and robot.estop_latch.engaged_at is None and stops == [1]
    assert got[0] == ("estop", {"device_id": "so101", "engaged": True, "by": "op"})
    assert [e for e, _ in got] == ["estop", "fault"] and "OSError('ptp: no grandmaster')" in got[1][1]["reason"]
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    sim.close()
    got = _announced(arm)
    arm.estop()  # a closed world has no time to read; the stop latches and is announced regardless
    assert arm.estopped and arm.estop_latch.engaged_at is None
    assert [e for e, _ in got] == ["estop", "fault"] and "AssertionError" in got[1][1]["reason"]


def test_k5_a_body_stop_that_raises_is_still_raised_when_the_clock_raises_too():
    drv = CountingDriver()

    def bus_down() -> None:
        raise ConnectionError("the bus did not answer")

    drv.estop = bus_down  # type: ignore[method-assign]
    robot = Robot("so101", driver=drv, twin=False, clock=BrokenClock())
    with pytest.raises(ConnectionError, match="did not answer"):  # the body may not have stopped: the caller hears it
        robot.estop()
    assert robot.estopped and robot.estop_latch.engaged_at is None


def test_k5_a_world_robot_reads_world_time_and_refuses_a_clock():
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    arm.act({})
    assert arm.observe().t == arm.clock.now() == sim.time > 0
    with pytest.raises(TypeError, match="clock="):
        Robot("so101", sim=sim, clock=MONOTONIC)
    sim.close()


@pytest.mark.parametrize("clock", [time.monotonic, lambda: 1.0, type(MONOTONIC)], ids=["function", "lambda", "class"])
def test_k5_a_hardware_robot_refuses_a_clock_that_is_not_a_clock_instance_and_says_what_to_pass(clock):
    with pytest.raises(TypeError, match=r"clock= must be a Clock instance.*now\(\).*sleep\(seconds\).*leave clock="):
        Robot("so101", driver=CountingDriver(), twin=False, clock=clock)  # taken before: act() wrote, then failed


def test_a_faulted_hardware_run_still_puts_the_body_in_its_on_disconnect_state():
    robot, drv = _hw()
    k = [0]

    def nan_at_3(obs):
        k[0] += 1
        return {"shoulder_pan": float("nan") if k[0] == 3 else 0.0}

    res = robot.run(nan_at_3, steps=10, hz=50)
    assert res.stopped_by == "fault" and len(drv.writes) == 2 and drv.holds == 1  # so101: every control holds


def _pair():
    sim = Simulation()
    return sim, Robot("so101", sim=sim, id="a"), Robot("so101", sim=sim, id="b", position=(0.5, 0, 0))


def test_k7_lockstep_reads_each_body_once_per_tick():
    sim, a, b = _pair()
    n = {"a": 0, "b": 0}
    orig = sim.backend.joint_state

    def counted(rid):
        n[rid] += 1
        return orig(rid)

    sim.backend.joint_state = counted
    res = sim.run({a: lambda o: {"shoulder_pan": 0.1}, b: lambda o: {}}, steps=7)
    assert res["a"].steps == res["b"].steps == 7
    assert n == {"a": 1 + 7 + 1, "b": 1 + 7 + 1}  # bind's requirements check, one per tick, the final observation
    sim.close()


def test_k7_a_fault_on_one_robot_stops_both_and_the_other_never_acts_that_tick():
    sim, a, b = _pair()
    k = {"a": 0, "b": 0}

    def pa(obs):
        k["a"] += 1
        return {"shoulder_pan": float("nan")} if k["a"] == 3 else {}

    def pb(obs):
        k["b"] += 1
        return {}

    res = sim.run({a: pa, b: pb}, steps=10)
    assert {r.stopped_by for r in res.values()} == {"fault"}
    assert k == {"a": 3, "b": 2} and res["a"].steps == res["b"].steps == 2
    sim.close()


def test_k7_lockstep_honours_a_stop_event():
    sim, a, b = _pair()
    ev = threading.Event()
    k = [0]

    def pa(obs):
        k[0] += 1
        if k[0] == 4:
            ev.set()
        return {}

    res = lockstep(sim, {a: pa, b: lambda o: {}}, steps=50, stop_event=ev)
    assert res["a"].stopped_by == "stopped" and res["a"].steps == 4
    sim.close()
