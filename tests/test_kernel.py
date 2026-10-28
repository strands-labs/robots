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


def test_k5_a_clock_that_raises_never_stops_the_estop_from_latching_and_reaching_the_body():
    drv = CountingDriver()
    stops = []
    drv.estop = lambda: stops.append(1)  # type: ignore[method-assign]
    robot = Robot("so101", driver=drv, twin=False, clock=BrokenClock())
    with pytest.raises(OSError, match="grandmaster"):
        robot.estop()
    assert robot.estopped and robot.estop_latch.count == 1 and robot.estop_latch.engaged_at is None and stops == [1]
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    sim.close()
    with pytest.raises(AssertionError):  # a closed world has no time to read; the stop latches regardless
        arm.estop()
    assert arm.estopped and arm.estop_latch.engaged_at is None


def test_k5_a_world_robot_reads_world_time_and_refuses_a_clock():
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    arm.act({})
    assert arm.observe().t == arm.clock.now() == sim.time > 0
    with pytest.raises(TypeError, match="clock="):
        Robot("so101", sim=sim, clock=MONOTONIC)
    sim.close()


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


def test_k7_a_start_pose_ramps_each_robot_before_the_lockstep_loop():
    """Simulation.run takes a start_pose policy to its pose before step one, as Robot.run does; a robot whose policy
    names no start_pose is left where it is."""
    sim, a, b = _pair()
    res = sim.run({a: _FromReady(), b: lambda o: {}}, steps=1)
    assert res["a"].steps == res["b"].steps == 1
    q = a.observe().joints.positions
    assert q[0] == pytest.approx(0.6, abs=0.1) and q[2] == pytest.approx(-0.4, abs=0.1)
    assert abs(b.observe().joints.positions[0]) < 0.1  # b named no start_pose: not moved
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


def test_k3_a_policy_that_returns_none_holds_without_a_write_or_a_miss():
    """A remote policy waiting for its chunk returns None: the body keeps its last target, the tick counts as a hold,
    the pacer's grid moves on and nothing is booked as a deadline miss."""
    robot, drv = _hw()
    calls = [0]

    def pol(obs):
        calls[0] += 1
        return None if 3 <= calls[0] <= 7 else {"shoulder_pan": 0.1 * calls[0]}  # five ticks of waiting

    res = robot.run(pol, steps=10, hz=50)
    assert (res.steps, res.holds, res.deadline_misses) == (10, 5, 0)
    assert len(drv.writes) == 5 and len(drv.reads) == 1 + 10 + 1  # one read per tick, held or not
    gaps = np.diff([t for t, _ in drv.writes])
    assert gaps[1] == pytest.approx(6 / 50)  # the five held periods sit between write 2 and write 3
    assert res.held_seconds == pytest.approx(5 / 50) and res.seconds == pytest.approx(10 / 50)
    assert res.hz_achieved == pytest.approx(50.0)


def test_k3_a_hold_on_the_pacer_moves_the_grid_one_period_without_a_miss():
    clock, dt = VirtualClock(), 1 / 50
    p = Pacer(dt, now=clock.now, sleep=clock.sleep)
    p.start()
    p.before_write()
    p.after_write()
    for _ in range(3):  # three held ticks: each starts at its grid point and writes nothing
        p.start()
        clock.advance(0.004)
        p.hold()
    p.start()
    assert clock.now() == pytest.approx(1000.0 + 4 * dt)
    p.before_write()
    p.after_write()
    assert p.misses == 0 and p.span() == pytest.approx(5 * dt)


def test_k3_a_slow_hold_does_not_fault_a_run_that_faults_on_a_miss():
    """``on_miss="fault"`` is how a real run refuses to command a body late. A sequential remote policy holds for a
    whole WAN round trip (here: ticks longer than the period), and that wait must not end the run."""
    robot, drv = _hw()
    calls = [0]

    def pol(obs):
        calls[0] += 1
        if 3 <= calls[0] <= 7:
            time.sleep(0.03)  # the chunk is in flight: this tick outlasts its 20 ms period
            return None
        return {"shoulder_pan": 0.1 * calls[0]}

    res = robot.run(pol, steps=10, hz=50, on_miss="fault")
    assert (res.steps, res.holds, res.stopped_by) == (10, 5, "steps")
    assert res.deadline_misses == 0 and res.fault is None


def test_k3_a_hold_longer_than_a_period_is_still_not_a_miss():
    """The chunk a sequential run waits for can take longer than one tick to land (a WAN round trip is 20+ periods).

    The waiting is deliberate, so the write that ends it is not late: a hold re-phases the grid onto now rather
    than leaving the next tick's grid point in the past, which would book the reply as the run's own lateness.
    """
    clock, dt = VirtualClock(), 1 / 30
    p = Pacer(dt, now=clock.now, sleep=clock.sleep)
    p.start()
    p.before_write()
    p.after_write()
    p.start()
    clock.advance(2.5 * dt)  # the reply lands mid-tick, two and a half periods after this hold began
    p.hold()
    p.start()
    p.before_write()
    p.after_write()
    assert p.misses == 0 and p.worst_late == 0.0


def test_k3_a_sim_hold_still_advances_the_world():
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    t0 = sim.time
    res = arm.run(lambda obs: None, steps=10, hz=50)
    assert (res.steps, res.holds) == (10, 10) and sim.time - t0 == pytest.approx(10 / 50, abs=1e-6)


class _FromReady:
    """A policy that names where its episodes begin (radians here: no units declared) and then holds there."""

    start_pose = {"shoulder_pan": 0.6, "elbow_flex": -0.4}
    start_seconds = 1.0

    def act(self, obs, *, instruction=None):
        return obs.joints.positions

    def reset(self):
        pass


def test_a_policy_with_a_start_pose_is_taken_there_before_step_one():
    robot, drv = _hw()
    res = robot.run(_FromReady(), steps=5, hz=50)
    ramp, steps = drv.writes[:-5], drv.writes[-5:]
    assert res.steps == 5 and len(ramp) == 30  # 1 s at reach's 30 Hz, then the five policy steps
    assert ramp[0][1][0] < 0.6 < ramp[-1][1][0] + 1e-9 and ramp[-1][1][0] == pytest.approx(0.6)  # smoothstep ends AT it
    assert ramp[-1][1][2] == pytest.approx(-0.4) and ramp[-1][1][1] == 0.0  # a joint not named holds its position
    assert all(w[0] == pytest.approx(0.6) for _, w in steps)  # step one began from the start pose
    with pytest.raises(KeyError, match="start_pose names \\['elbow'\\]"):
        bad = _FromReady()
        bad.start_pose = {"elbow": 0.1}
        robot.run(bad, steps=1)


def test_a_start_pose_is_in_the_policys_units():
    from strands_robots.policy import Policy as Base
    from strands_robots.units import lerobot_units

    class Norm(Base):
        units = "lerobot"
        start_pose = {"shoulder_pan.pos": 50.0, "gripper.pos": 10.0}  # LeRobot names and units are fine too
        start_seconds = 0.5

        def act(self, obs, *, instruction=None):
            return obs.joints.positions

    robot, drv = _hw()
    robot.run(Norm(), steps=1, hz=50)
    u = lerobot_units(robot.description)
    want = u.to_rad(np.array([50.0, 0, 0, 0, 0, 10.0]))
    assert drv.writes[-1][1][0] == pytest.approx(want[0], abs=1e-6) and drv.writes[-1][1][5] == pytest.approx(want[5])
