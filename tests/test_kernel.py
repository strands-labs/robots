"""The run kernel: one body read per tick (K1), no catch-up burst (K3), one clock per body (K5), one step (K7).

A hardware body here is a ``CountingDriver`` on a ``VirtualClock`` the test owns: every bus read and write is
logged with its time, and nothing waits for real.
"""

from __future__ import annotations

import contextlib
import threading
import time

import numpy as np
import pytest
from test_drivers import FakeSerial, make_driver, writes

from strands_robots import JointState, Robot, Simulation
from strands_robots.description import Description
from strands_robots.robot.clock import MONOTONIC
from strands_robots.robot.drivers import feetech as ft
from strands_robots.robot.run import lockstep
from strands_robots.robot.run.pace import Pacer
from strands_robots.robot.safety import NonFiniteTarget


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


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_k1_command_refuses_a_non_finite_target_before_the_bus_as_act_does(bad):
    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False)
    with pytest.raises(NonFiniteTarget, match="shoulder_pan"):
        real.command(np.array([bad, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert writes(s) == []  # no torque, no goal (the driver used to refuse it after Torque_Enable)


@pytest.mark.parametrize("n", [5, 7])
def test_k1_command_refuses_a_target_that_is_not_one_value_per_action_key_before_the_bus(n):
    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False)
    with pytest.raises(ValueError, match=f"action has {n} values; so101 has 6 action keys"):
        real.command(np.zeros(n))
    assert writes(s) == []  # it used to fail in the driver's zip(), after Torque_Enable


def test_k1_command_clips_every_joint_into_its_limits_and_still_never_reads():
    d, s = make_driver(cal=None)  # uncalibrated: full-turn counts, so only the limits keep 3 rad off the servo
    real = Robot("so101", driver=d, twin=False)
    hi = real.description.limits["shoulder_lift"][1]
    s.frames.clear()
    real.command(np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0]))
    goal = int.from_bytes(s.regs[2][ft.REG_GOAL_POSITION : ft.REG_GOAL_POSITION + 2], "little")
    assert goal == d.rad_to_counts("shoulder_lift", hi)  # 3186; 3.0 rad used to go out as 4004
    assert [f for f in s.frames if f[4] in (ft.READ, ft.SYNC_READ)] == []


def test_k1_command_never_puts_a_nan_or_an_out_of_limit_target_into_a_world():
    sim = Simulation()
    arm = Robot("so101", sim=sim)
    t0 = sim.time
    with pytest.raises(NonFiniteTarget, match="shoulder_pan"):
        arm.command(np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]))
    # while one ctrl is NaN, MuJoCo drives every actuator in the world as if its ctrl were 0
    assert np.isfinite(sim.backend.data.ctrl[:6]).all() and sim.time == t0
    arm.command(np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0]))
    assert sim.backend.data.ctrl[1] == pytest.approx(arm.description.limits["shoulder_lift"][1])
    sim.close()


@pytest.mark.parametrize("n", [5, 7])
def test_k1_coerce_refuses_a_measured_vector_that_is_not_one_value_per_action_key(n):
    robot, drv = _hw()
    with pytest.raises(ValueError, match=f"measured has {n} values; so101 has 6 action keys"):
        robot.coerce({"gripper": 0.2}, measured=np.zeros(n))  # 5 used to raise IndexError, 7 to return 7 targets
    assert drv.reads == [] and drv.writes == []


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


@pytest.mark.parametrize(("eps", "healthy", "misses"), [(100e-6, 2000, 0), (1.5e-3, 500, 1)])
def test_k3_lateness_is_measured_from_the_write_that_landed_not_a_grid_the_writes_left_behind(eps, healthy, misses):
    """One tick that nearly fills its period (17.95 ms + 2 ms write at 50 Hz, sleeps overshooting by ``eps``) used
    to tip the pacer into a regime where every later write was held to its floor and the grid stayed one overshoot
    further behind each tick, so a 1 ms overrun hundreds of ticks later was reported as hundreds of ms late. The
    last tick's 19 ms of inference start within one late wake (``eps``) of its grid point, as on a fresh pacer:
    late only when that wake is over 1 ms (``misses``), and by no more than the excess."""
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
    tick(0.019)  # 19 ms of inference: 1 ms before its deadline, less how late the tick started
    assert p.misses == misses
    late_ms = f"{p.worst_late * 1e3:.2f} ms reported for 19 ms of inference after a {eps * 1e3:g} ms late wake"
    assert p.worst_late <= max(0.0, 0.019 + eps - dt) + 1e-9, late_ms
    assert min(np.diff(writes)) >= dt - 1e-9


class LateWaking(VirtualClock):
    """Every sleep wakes ``late`` seconds late, as a real one does; ``advance`` (inference) stays exact."""

    def __init__(self, late: float) -> None:
        super().__init__()
        self.late = late

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds) + self.late


@pytest.mark.parametrize("late", [0.1e-3, 1.5e-3, 6e-3])
def test_k3_a_paced_run_keeps_its_rate_and_fresh_observations_when_every_sleep_wakes_late(late):
    """50 Hz on the fake Feetech bus, 3-8 ms of inference, every sleep waking ``late``: no miss, so the writes span
    their periods to within one (as ``RunResult`` says), each acts on an observation one inference old (not one
    period), and no two are closer than a period less that lateness."""
    clock, stamps, seen = LateWaking(late), [], []

    class Stamped(FakeSerial):  # each Goal_Position write stamped on the body's clock as it reaches the bus
        def write(self, data):
            if data[4] == ft.SYNC_WRITE and data[5] == ft.REG_GOAL_POSITION:
                stamps.append(clock.now())
            super().write(data)

    d, _ = make_driver(serial=Stamped())
    arm = Robot("so101", driver=d, twin=False, clock=clock)
    work = iter(np.random.default_rng(0).uniform(0.003, 0.008, 300))

    def pol(obs):
        seen.append(obs.t)
        clock.advance(next(work))
        return {"shoulder_pan": 0.1}

    res = arm.run(pol, steps=300, hz=50)
    w = np.array(stamps)
    assert res.deadline_misses == 0 and len(w) == len(seen) == 300
    rate = f"{res.hz_achieved:.2f} Hz achieved, the bus written at {299 / (w[-1] - w[0]):.2f} Hz"
    assert abs(res.seconds - 300 / 50) <= 1 / 50 and abs(w[-1] - w[0] - 299 / 50) <= 1 / 50, rate
    age = w - np.array(seen)
    assert age.max() <= 0.008 + late + 1e-9, f"writes act on observations up to {age.max() * 1e3:.1f} ms old"
    assert np.diff(w).min() >= 1 / 50 - late - 1e-9


@pytest.mark.parametrize("hz", [10.0, 50.0, 200.0])
def test_k3_sleeps_that_wake_late_never_cost_a_run_its_rate_or_a_gap_more_than_one_wake(hz):
    """100 random histories per rate (inference up to 1.2 periods, some stalls, every sleep 0..max_late late): with
    no miss the span is n periods to within one, and no two writes are closer than a period less ``max_late``."""
    rng, dt = np.random.default_rng(int(hz)), 1 / hz
    spans = []
    for _ in range(100):
        clock, max_late = VirtualClock(), rng.uniform(0, dt / 2)
        p = Pacer(dt, now=clock.now, sleep=lambda s, c=clock, m=max_late: c.advance(max(0.0, s) + rng.uniform(0, m)))
        work = rng.uniform(0, rng.uniform(0.2, 1.2) * dt, rng.integers(2, 80))
        if rng.random() < 0.25:
            work[rng.integers(0, len(work))] = rng.uniform(0, 0.5)
        writes = []
        for w in work:
            p.start()
            clock.advance(w)
            p.before_write()
            writes.append(clock.now())
            clock.advance(rng.uniform(0, dt / 10))  # the write itself
            p.after_write()
        assert np.diff(writes).min() >= dt - max_late - 1e-9
        if p.misses == 0:
            spans.append(abs(p.span() - len(work) * dt) / dt)
    assert len(spans) > 30 and max(spans) <= 1 + 1e-9, f"a run with no miss was {max(spans):.2f} periods off its rate"


class _Dataset:
    """What a run's recording calls on LeRobot's dataset, with no LeRobot: frames counted, the directory made."""

    def __init__(self, root) -> None:
        from types import SimpleNamespace

        self.root, self.frames, self.meta = root, 0, SimpleNamespace(total_episodes=0)
        root.mkdir(parents=True)

    def add_frame(self, frame) -> None:
        self.frames += 1

    def save_episode(self, parallel_encoding: bool = True) -> None:
        self.meta.total_episodes += 1

    def finalize(self) -> None: ...

    def clear_episode_buffer(self) -> None: ...


def _no_lerobot(monkeypatch, open_dataset) -> None:
    """A recording whose dataset ``open_dataset`` opens: no LeRobot, PyAV or ``datasets`` imported."""
    from strands_robots.recording import lerobot_v3

    monkeypatch.setattr(lerobot_v3, "require_lerobot", lambda: None)
    monkeypatch.setattr(lerobot_v3, "open_dataset", open_dataset)
    monkeypatch.setattr(lerobot_v3, "quiet", contextlib.nullcontext)


def _slow_dataset(monkeypatch, clock, seconds: float) -> list:
    """Open a recording's dataset in ``seconds`` of the body's clock (LeRobot took 0.3-1 s in a fresh process)."""
    made: list = []

    def open_dataset(ep, first):
        clock.advance(seconds)
        made.append(_Dataset(ep.root))
        return made[-1], True

    _no_lerobot(monkeypatch, open_dataset)
    return made


def test_k3_a_recording_opens_its_dataset_before_tick_one_so_its_setup_is_no_deadline_miss(tmp_path, monkeypatch):
    """The dataset used to open at the first frame, after tick 1's write: its setup made tick 2 late, so
    ``on_miss="fault"`` ended a process's first recording there."""
    robot, drv = _hw()
    made = _slow_dataset(monkeypatch, drv.clock, 0.5)
    res = robot.run(lambda obs: {"shoulder_pan": 0.1}, steps=30, hz=30, record=tmp_path / "ds", on_miss="fault")
    assert (res.stopped_by, res.steps, res.deadline_misses, res.fault) == ("steps", 30, 0, None)
    assert made[0].frames == 30 and res.recorded == str(made[0].root) and len(drv.writes) == 30


def test_k3_a_recording_that_ends_before_its_first_frame_leaves_no_dataset(tmp_path, monkeypatch):
    robot, drv = _hw()
    made = _slow_dataset(monkeypatch, drv.clock, 0.5)
    res = robot.run(lambda obs: {}, steps=30, hz=30, record=tmp_path / "ds", stop_when=lambda obs: True)
    assert res.stopped_by == "stop_when" and res.recorded is None and drv.writes == []
    assert not (tmp_path / "ds").exists() and all(d.frames == 0 for d in made)


class _WithGains:
    """A policy that carries its own PD gains (kp 33), as holosoma's checkpoint does."""

    gains = ([33.0] * 6, [1.0] * 6)

    def act(self, obs, *, instruction=None):
        return {"shoulder_pan": 0.6}

    def reset(self) -> None: ...


def test_k3_a_dataset_the_run_cannot_extend_is_refused_before_a_write_or_the_policys_gains(tmp_path, monkeypatch):
    """Opening the dataset is one of the run's checks: refused, the arm has not moved and keeps the gains it had (a
    policy's go on after the last check). It used to open at tick 1's frame, after both."""

    def refuse(ep, first):
        raise ValueError(f"{ep.target}: the dataset there has other features")

    _no_lerobot(monkeypatch, refuse)
    arm = Robot("so101")

    def body():  # shoulder_pan's position and kp
        m = arm.sim.backend.model
        kp = m.actuator_gainprm[m.actuator(f"{arm.id}/{arm.description.actuators[0]}").id, 0]
        return arm.observe(images=False)["shoulder_pan"], float(kp)

    found = body()
    with pytest.raises(ValueError, match="other features"):
        arm.run(_WithGains(), steps=3, hz=20, record=tmp_path / "ds")
    assert body() == found  # dev: the arm moved, under kp 33
    arm.sim.close()


def test_k3_writes_stay_one_period_apart_across_runs_estops_and_resumes():
    """A random history of runs, stalls, e-stops and resumes on one body whose sleeps are exact: no two writes < one
    period apart (when the wait before a write wakes late, the next one may come that much closer)."""
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


def _pan_target(sim, robot) -> float:
    """The shoulder_pan target the world applies to ``robot`` on its next step."""
    return float(sim.backend.data.ctrl[sim.backend._index[robot.id].actuator_ids[0]])


def _b_decides_first(k):
    def pb(obs):  # tick 1: 0.1, tick 2: 0.2, tick 3: 0.3
        k["b"] += 1
        return {"shoulder_pan": 0.1 * k["b"]}

    return pb


def test_k7_a_robot_listed_before_the_one_that_faults_writes_nothing_that_tick():
    sim, a, b = _pair()
    k = {"a": 0, "b": 0}

    def pa(obs):
        k["a"] += 1
        return {"shoulder_pan": float("nan")} if k["a"] == 3 else {}

    res = sim.run({b: _b_decides_first(k), a: pa}, steps=10)  # b has decided tick 3 when a refuses it
    assert (res["b"].stopped_by, res["b"].steps, k) == ("fault", 2, {"a": 3, "b": 3})
    assert _pan_target(sim, b) == pytest.approx(0.2)  # tick 2's target: no later step of the world applies 0.3
    sim.close()


@pytest.mark.parametrize("resumed", [False, True], ids=["pressed", "pressed-and-resumed"])
def test_k7_an_estop_while_the_robots_decide_leaves_that_tick_unwritten(resumed):
    sim, a, b = _pair()
    k = {"a": 0, "b": 0}

    def pa(obs):
        k["a"] += 1
        if k["a"] == 3:  # the operator stops a while it decides tick 3, after b decided it
            a.estop()
            if resumed:
                a.resume(approved_by="op")
        return {}

    res = sim.run({b: _b_decides_first(k), a: pa}, steps=10)
    assert (res["b"].stopped_by, res["b"].steps, res["b"].fault) == ("estop", 2, "e-stop engaged on a")
    assert _pan_target(sim, b) == pytest.approx(0.2)
    sim.close()


def test_k7_an_estop_on_a_robot_yet_to_decide_ends_the_tick_before_it_decides():
    sim, a, b = _pair()
    k = {"a": 0, "b": 0}

    def pa(obs):  # a decides first
        k["a"] += 1
        if k["a"] == 3:
            b.estop()  # the operator stops b while a decides tick 3
        return {"shoulder_pan": 0.1 * k["a"]}

    def pb(obs):
        k["b"] += 1
        return {}

    res = sim.run({a: pa, b: pb}, steps=10)
    assert (res["a"].stopped_by, res["a"].fault, k) == ("estop", "e-stop engaged on b", {"a": 3, "b": 2})
    assert _pan_target(sim, a) == pytest.approx(0.2)
    sim.close()


@pytest.mark.parametrize("confirmed", [False, True], ids=["agent", "dashboard"])
def test_k7_a_lockstep_tick_the_gate_refuses_for_one_robot_is_written_for_none(confirmed):
    """An agent's run ends as a fault at the refused step (K6), a dashboard's (one operator yes) raises it: either
    way the robot listed before the refused one has not written that tick."""
    from contextlib import nullcontext

    from strands_robots.robot.run.source import from_agent
    from strands_robots.robot.safety import RequestRefused

    sim, a, b = _pair()
    k = {"a": 0, "b": 0}

    def third_refused(robot, request):
        k["a"] += 1
        if k["a"] == 3:
            raise RequestRefused(f"{robot.id}: refused at tick 3")

    a.validators.append(third_refused)  # every gated write of a passes it, after b decided the same tick
    raised = pytest.raises(RequestRefused, match="tick 3") if confirmed else nullcontext()
    with from_agent(confirmed=confirmed), raised:
        res = sim.run({b: _b_decides_first(k), a: lambda o: {}}, steps=10)
    if not confirmed:
        assert (res["a"].stopped_by, res["a"].steps, res["a"].fault) == ("fault", 2, "a: refused at tick 3")
    assert _pan_target(sim, b) == pytest.approx(0.2)
    sim.close()


def test_k7_a_robot_that_holds_in_lockstep_writes_nothing_and_the_world_still_steps_once_a_tick():
    """In lock-step a hold writes nothing and adds no step of its own: the world steps once a tick for every robot."""
    sim, a, b = _pair()
    k = {"b": 0}

    def pb(obs):  # ticks 2 to 4 hold: tick 1's target stays in the world
        k["b"] += 1
        return None if 2 <= k["b"] <= 4 else {"shoulder_pan": 0.1 * k["b"]}

    t0 = sim.time
    res = sim.run({a: lambda o: {"shoulder_pan": 0.3}, b: pb}, steps=4, hz=50)
    assert (res["b"].stopped_by, res["b"].steps, k["b"]) == ("steps", 4, 4)
    assert sim.time - t0 == pytest.approx(4 / 50, abs=1e-6)
    assert (_pan_target(sim, a), _pan_target(sim, b)) == pytest.approx((0.3, 0.1))
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
