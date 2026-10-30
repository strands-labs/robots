"""One safety chain for every write, an e-stop that always wins, a watchdog that applies on_disconnect."""

from __future__ import annotations

import threading
import time
from decimal import Decimal

import numpy as np
import pytest
from test_drivers import CorruptingSerial, make_driver

from strands_robots import Robot
from strands_robots.robot.device import manifest_for
from strands_robots.robot.drivers import feetech as ft
from strands_robots.robot.safety import RequestRefused, Watchdog


def test_out_of_range_write_is_refused_naming_joint_value_and_bound():
    arm = Robot("so101")
    lo, hi = arm.description.limits["elbow_flex"]
    with pytest.raises(
        RequestRefused, match=rf"elbow_flex={hi + 1:.4f} rad is outside its range \[{lo:.4f}, {hi:.4f}\]"
    ):
        arm.confirm_act({"elbow_flex": hi + 1.0})


def _saturated_sim_arm():
    arm = Robot("so101")
    hi = np.array([arm.description.limits[k][1] for k in arm.action_keys])
    for _ in range(50):  # legal: act() clips; the soft limit lets the joints settle a hair past it
        arm.act(hi)
    return arm


def _servo_past_range_min_arm():
    d, s = make_driver()
    s.actual[4] = 1024 - 40  # wrist_flex rests 40 counts below its calibrated sweep
    return Robot("so101", driver=d, twin=False, confirm=lambda a: True)


@pytest.mark.parametrize("make", [_saturated_sim_arm, _servo_past_range_min_arm])
def test_a_joint_measured_past_its_limit_does_not_refuse_a_write_that_never_named_it(make):
    arm = make()
    arm.confirm_act({"gripper": 0.5})  # the joints it did not name are held, clipped into range
    hi = arm.description.limits["gripper"][1]
    with pytest.raises(RequestRefused, match=r"^so101: gripper="):  # a value it did send is still graded
        arm.confirm_act({"gripper": hi + 1.0})
    arm.close()


def test_validator_refusal_reaches_the_caller_verbatim():
    arm = Robot("so101")

    def no_wrist(robot, request):
        if abs(dict(request.items())["wrist_roll"]) > 0.1:
            raise RequestRefused("wrist_roll must stay within 0.1 rad of zero while the cup is held")

    arm.validators.append(no_wrist)
    with pytest.raises(RequestRefused) as e:
        arm.confirm_act({"wrist_roll": 0.5})
    assert str(e.value) == "wrist_roll must stay within 0.1 rad of zero while the cup is held"
    arm.confirm_act({"wrist_roll": 0.05})


def test_estop_latches_refuses_act_and_resume_needs_an_operator():
    arm = Robot("so101")
    arm.estop()
    assert arm.estopped and arm.describe()["estop"] is True
    with pytest.raises(RequestRefused, match="e-stop is engaged"):
        arm.act({"shoulder_pan": 0.2})
    with pytest.raises(RequestRefused, match="approved_by"):
        arm.resume()
    with pytest.raises(RequestRefused, match="approved_by"):
        arm.resume(approved_by="  ")
    assert arm.estopped
    arm.resume(approved_by="cagatay")
    arm.act({"shoulder_pan": 0.2})


def test_run_stops_with_estop_within_one_tick():
    arm = Robot("so101")
    ticks: list[int] = []

    def pol(obs):
        ticks.append(len(ticks))
        if len(ticks) == 5:
            arm.estop()
        return {"shoulder_pan": 0.4}

    res = arm.run(pol, hz=50, steps=100)
    assert res.stopped_by == "estop" and res.steps == 4 and len(ticks) == 5


def test_estop_from_another_thread_stops_a_background_run():
    arm = Robot("so101")
    task = arm.start(lambda o: {"shoulder_pan": 0.5}, hz=100, seconds=30, realtime=True)
    time.sleep(0.15)
    t0 = time.perf_counter()
    threading.Thread(target=arm.estop).start()
    while task.running and time.perf_counter() - t0 < 2.0:
        time.sleep(0.001)
    assert not task.running and task.result.stopped_by == "estop"
    assert time.perf_counter() - t0 < 0.1  # one tick at 100 Hz is 10 ms; generous for CI


def test_hold_and_watchdog_lapse_apply_on_disconnect_in_manual_time():
    arm = Robot("so101")
    arm.act({"shoulder_pan": 0.6})
    with arm.sim.manual_time():
        dog = Watchdog(arm, timeout_ms=100)
        dog.beat()
        arm.sim.step(int(0.05 / arm.sim.physics.timestep))
        assert dog.poll() is False
        arm.sim.step(int(0.06 / arm.sim.physics.timestep))
        q = arm.observe().joints.positions
        assert dog.poll() is True and dog.poll() is False  # applied once per lapse
        np.testing.assert_allclose(arm.sim.backend.data.ctrl[: len(q)], q)


def test_hardware_robot_manifest_and_estop_on_a_faked_bus():
    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False)
    m = manifest_for(real)
    assert m.real and m.world is None and len(m.controls) == 6
    real.estop()  # torque was never on: a limp arm stays limp, nothing written
    assert d.extras()["torque"] is False
    real.resume(approved_by="op")
    real.act({"shoulder_pan": 0.1})
    real.estop()  # torque on: hold where it is
    assert d.extras()["torque"] is True
    real.close()


def _feetech_arm(torque):
    """A so101 on the fake bus: limp (torque never on), or powered by the developer's own write."""
    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False)
    if torque:
        real.act(real.observe().joints.positions)  # torque on, Goal_Position := Present_Position
    return real, d, s


def _lapse(robot):
    """One watchdog lapse on a fake clock: 200 ms of silence against a 100 ms timeout."""
    now = [0.0]
    dog = Watchdog(robot, timeout_ms=100, clock=lambda: now[0])
    now[0] = 0.2
    return dog.poll()


@pytest.mark.parametrize("torque", [False, True], ids=["limp", "torque-on"])
def test_a_hold_while_the_estop_is_latched_is_refused_and_writes_nothing(torque):
    from test_operator_gate import goal_writes

    from strands_robots.robot.safety import EStopped

    real, d, s = _feetech_arm(torque)
    real.estop()
    s.frames.clear()
    with pytest.raises(EStopped, match="e-stop is engaged"):
        real.hold()  # the verb an agent tool and a mesh peer call
    assert goal_writes(s) == [] and d.extras()["torque"] is torque
    real.resume(approved_by="op")
    real.hold()  # released: a hold is accepted again
    real.close()


@pytest.mark.parametrize("torque", [False, True], ids=["limp", "torque-on"])
def test_a_watchdog_lapse_while_the_estop_is_latched_writes_nothing(torque):
    from test_operator_gate import goal_writes

    real, d, s = _feetech_arm(torque)
    real.estop()
    s.frames.clear()
    assert _lapse(real) is True  # the controller goes quiet after an e-stop: the lapse is seen, and adds nothing
    assert goal_writes(s) == [] and d.extras()["torque"] is torque
    real.close()


def _narrow_safe(robot, key, safe):
    from dataclasses import replace

    from strands_robots.robot.device.build import manifest_key

    m = robot.manifest
    controls = tuple(replace(c, safe=safe) if c.name == f"{key}.pos" else c for c in m.controls)
    robot.__dict__["_manifest"] = (manifest_key(robot), replace(m, controls=controls))


def test_outside_safe_inside_range_needs_the_operators_yes():
    d, _ = make_driver()
    asked: list[np.ndarray] = []
    answer = {"yes": False}
    real = Robot("so101", driver=d, twin=False, confirm=lambda a: asked.append(a) or answer["yes"])
    _narrow_safe(real, "shoulder_pan", (-0.5, 0.5))
    with pytest.raises(RequestRefused, match=r"shoulder_pan=0\.8000 rad outside safe \[-0\.5000, 0\.5000\]"):
        real.confirm_act({"shoulder_pan": 0.8})
    assert len(asked) == 1
    answer["yes"] = True
    real.confirm_act({"shoulder_pan": 0.8})
    assert len(asked) == 2  # asked once for this write, not again by the gate
    sim = Robot("so101")
    _narrow_safe(sim, "shoulder_pan", (-0.5, 0.5))
    with pytest.raises(RequestRefused, match="outside safe"):
        sim.confirm_act({"shoulder_pan": 0.8})
    sim.confirm_act({"shoulder_pan": 0.8}, confirmed=True)  # an operator said yes out of band
    real.close()


def test_a_write_already_past_the_latch_check_is_refused_after_estop():
    arm = Robot("so101")
    coerce = arm.coerce

    def estop_mid_command(action, **kw):
        out = coerce(action, **kw)
        arm.estop()  # another thread presses stop between the check and the write
        return out

    arm.coerce = estop_mid_command  # type: ignore[method-assign]
    before = np.array(arm.sim.backend.data.ctrl[:6])
    with pytest.raises(RequestRefused, match="e-stop is engaged"):
        arm.act({"shoulder_pan": 0.5})
    np.testing.assert_allclose(arm.sim.backend.data.ctrl[:6][0], arm.observe().joints.positions[0])
    assert arm.sim.backend.data.ctrl[0] != 0.5 and before.shape == (6,)


def test_the_latch_exists_before_any_thread_touches_it():
    arm = Robot("so101")
    assert "estop_latch" in arm.__dict__ and arm.estop_latch is arm.estop_latch


def test_lockstep_estop_names_the_robot_that_latched():
    from strands_robots import Simulation

    sim = Simulation()
    a, b = Robot("so101", sim=sim, id="a"), Robot("so101", sim=sim, id="b", position=(0.5, 0, 0))
    n = {"k": 0}

    def pol(obs):
        n["k"] += 1
        if n["k"] == 3:
            a.estop()
        return obs.joints.positions

    res = sim.run({a: pol, b: lambda o: o.joints.positions}, steps=20)
    assert {r.stopped_by for r in res.values()} == {"estop"}
    assert res["a"].fault == res["b"].fault == "e-stop engaged on a"
    sim.close()


def test_an_estop_resumed_before_the_next_tick_still_stops_the_run():
    arm = Robot("so101")
    task = arm.start(lambda o: {"shoulder_pan": 0.5}, hz=50, seconds=30, realtime=True)
    time.sleep(0.1)
    arm.estop()
    arm.resume(approved_by="op")  # released at once: the run must still have seen the stop
    task.halt(2.0)
    assert task.result.stopped_by == "estop"


def test_estop_from_another_thread_never_collides_on_the_half_duplex_wire():
    from test_operator_gate import OneWire

    d, s = make_driver(serial=OneWire())
    real = Robot("so101", driver=d, twin=False)
    real.act(real.observe().joints.positions)  # torque on, so estop() is a hold on the bus
    errors: list[BaseException] = []

    def observe():
        try:
            for _ in range(60):
                real.observe(images=False)
        except Exception as exc:  # the test reports what a thread raised
            errors.append(exc)

    def stop():
        for _ in range(20):
            real.estop()

    threads = [threading.Thread(target=observe), threading.Thread(target=stop)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and s.collisions == 0
    real.close()


def test_estop_holds_and_announces_itself_through_one_corrupted_status_packet():
    from test_operator_gate import goal_writes

    d, s = make_driver(serial=CorruptingSerial())
    real = Robot("so101", driver=d, twin=False)
    real.act({"shoulder_pan": 0.5})  # torque on, so estop() is a hold on the bus
    s.actual[1] = 2400  # servo 1 is still on its way to its goal
    heard: list[dict] = []
    real.subscribe("estop", heard.append)
    s.frames.clear()
    s.corrupt = 1
    real.estop()
    goal = int.from_bytes(s.regs[1][ft.REG_GOAL_POSITION : ft.REG_GOAL_POSITION + 2], "little")
    assert real.estopped and len(goal_writes(s)) == 1 and goal == 2400  # held where it was measured
    assert [(h["device_id"], h["engaged"]) for h in heard] == [("so101", True)] and s.corrupt == 0
    real.close()


def test_a_watchdog_refuses_a_real_body_with_no_declared_on_disconnect():
    from test_drivers import FakeLowState, FakeSub

    from strands_robots import Description
    from strands_robots.robot.drivers.unitree import UnitreeDriver

    g1 = Robot(
        "g1", driver=UnitreeDriver(Description.load("g1"), port="en7", subscriber=FakeSub(FakeLowState())), twin=False
    )
    with pytest.raises(RequestRefused, match="no proven safe on_disconnect"):
        Watchdog(g1, timeout_ms=200)
    g1.close()
    d, _ = make_driver()
    so = Robot("so101", driver=d, twin=False)
    Watchdog(so, timeout_ms=200)  # feetech declares hold in the registry
    so.close()


def _held_arm_whose_next_hold_fails(error: Exception | None = None, fails: list[int] | None = None):
    """An SO-101 on the fake bus under torque, whose next ``fails[0]`` holds (default 1) raise ``error``.

    ``error`` defaults to an ``OSError``; setting ``fails[0]`` again makes later holds fail too."""
    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False)
    real.act({"shoulder_pan": 0.3})  # torque on: a hold is one Goal_Position write
    hold, left = d.hold, fails if fails is not None else [1]

    def hold_failing():
        if left[0] > 0:
            left[0] -= 1
            raise error if error is not None else OSError("the USB adapter re-enumerated")
        hold()

    d.hold = hold_failing  # type: ignore[method-assign]
    return real, s


def _holds_after(s, n):
    """Goal writes on ``s`` past the first ``n``, once one lands (5 s at most)."""
    from test_operator_gate import goal_writes

    t0 = time.monotonic()
    while len(goal_writes(s)) == n and time.monotonic() - t0 < 5.0:
        time.sleep(0.005)
    return len(goal_writes(s)) - n


def _watch_until_it_holds(real, s, period_s=0.005):
    """A watchdog thread on ``real`` until a hold lands (5 s at most), then closed: (holds, thread alive, watchdog)."""
    from test_operator_gate import goal_writes

    n = len(goal_writes(s))
    dog = Watchdog(real, timeout_ms=30).start(period_s=period_s)
    held, alive = _holds_after(s, n), dog._thread is not None and dog._thread.is_alive()
    dog.stop()
    real.close()
    return held, alive, dog


def test_a_watchdog_thread_outlives_a_failed_on_disconnect_and_retries_it_next_period():
    real, s = _held_arm_whose_next_hold_fails()
    faults: list[dict] = []
    real.subscribe("fault", faults.append)
    # Subscribed last: Emitter.emit stops at the first subscriber that raises, which would hide the event.
    real.subscribe("fault", lambda f: 1 / 0)  # a subscriber's own bug does not end the watchdog either
    held, alive, dog = _watch_until_it_holds(real, s)
    assert held == 1 and alive and dog.lapsed  # the retry held, and the thread is still watching
    assert dog.errors == 1 and len(faults) == 1 and faults[0]["device_id"] == "so101"
    assert "watchdog" in faults[0]["reason"] and "OSError: the USB adapter re-enumerated" in faults[0]["reason"]


def test_a_run_of_failed_watchdog_retries_is_one_fault_event_and_a_later_run_is_another():
    from test_operator_gate import goal_writes

    fails = [3]
    real, s = _held_arm_whose_next_hold_fails(fails=fails)
    faults: list[dict] = []
    real.subscribe("fault", faults.append)
    n = len(goal_writes(s))
    dog = Watchdog(real, timeout_ms=30).start(period_s=0.005)
    assert _holds_after(s, n) == 1 and dog.errors == 3 and len(faults) == 1  # three failed tries, one event
    fails[0], n = 1, len(goal_writes(s))
    dog.beat()  # the commander comes back once, then goes quiet: a new lapse whose first apply fails
    assert _holds_after(s, n) == 1 and dog.errors == 4 and len(faults) == 2
    dog.stop()
    real.close()


def test_a_poll_after_a_failed_on_disconnect_retries_it_in_the_same_lapse():
    from test_operator_gate import goal_writes

    real, s = _held_arm_whose_next_hold_fails()
    now = {"t": 0.0}
    dog = Watchdog(real, timeout_ms=50, clock=lambda: now["t"])
    now["t"] = 0.1
    with pytest.raises(OSError, match="re-enumerated"):
        dog.poll()
    n = len(goal_writes(s))
    now["t"] = 0.2  # no beat since: the same lapse, still not applied
    assert dog.poll() is True and len(goal_writes(s)) == n + 1
    assert dog.poll() is False  # applied once per lapse
    real.close()


@pytest.mark.parametrize("torque", [False, True], ids=["limp", "torque-on"])
def test_a_latched_lapse_is_left_to_the_estop_so_a_failed_read_is_never_retried_after_resume(torque):
    from test_operator_gate import goal_writes

    real, d, s = _feetech_arm(torque)
    real.estop()
    read, garbled = d.read, [True]

    def read_garbled():
        if garbled[0]:
            raise OSError("servo 6: bad checksum")
        return read()

    d.read = read_garbled  # type: ignore[method-assign]
    now = [0.0]
    dog = Watchdog(real, timeout_ms=100, clock=lambda: now[0])
    s.frames.clear()
    now[0] = 0.2
    assert dog.poll() is True  # seen, and left to the e-stop: nothing read that could fail and be retried
    garbled[0] = False
    real.resume(approved_by="op")
    now[0] = 0.4  # no beat since: that lapse was the e-stop's, so resume() lets nothing land
    assert dog.poll() is False and goal_writes(s) == [] and d.extras()["torque"] is torque
    dog.beat()
    now[0] = 0.7  # the commander came back, then went quiet again: this lapse is the watchdog's
    assert dog.poll() is True and len(goal_writes(s)) == (1 if torque else 2)  # a limp arm's hold enables torque first
    assert d.extras()["torque"] is True
    real.close()


@pytest.mark.parametrize("torque", [False, True], ids=["limp", "torque-on"])
def test_a_lapse_whose_apply_an_estop_interrupted_is_left_to_it_and_never_applied_after_resume(torque):
    from test_operator_gate import goal_writes

    real, d, s = _feetech_arm(torque)
    read, first = d.read, [True]

    def read_as_the_estop_lands():
        if first[0]:
            first[0] = False
            real.estop()  # another thread presses stop while the apply reads, and that read fails
            raise OSError("servo 6: bad checksum")
        return read()

    d.read = read_as_the_estop_lands  # type: ignore[method-assign]
    now = [0.0]
    dog = Watchdog(real, timeout_ms=100, clock=lambda: now[0])
    now[0] = 0.2
    with pytest.raises(OSError, match="bad checksum"):
        dog.poll()
    real.resume(approved_by="op")
    s.frames.clear()
    now[0] = 0.4  # no beat since: the e-stop took that lapse, as it takes one that starts while latched
    assert dog.poll() is False and goal_writes(s) == [] and d.extras()["torque"] is torque
    real.close()


def test_a_lapse_whose_write_an_estop_interrupted_is_never_applied_after_resume():
    from dataclasses import replace

    from test_operator_gate import goal_writes

    from strands_robots.robot.device.build import manifest_key

    real, d, s = _feetech_arm(True)
    m = real.manifest  # every control zero: the apply is a guarded write, which refuses once a stop lands in it
    real.__dict__["_manifest"] = (
        manifest_key(real),
        replace(m, controls=tuple(replace(c, on_disconnect="zero") for c in m.controls)),
    )
    write, first = d.write, [True]

    def write_as_the_estop_lands(targets):
        if first[0]:
            first[0] = False
            real.estop()  # another thread presses stop between the apply's latch check and its write
        write(targets)

    d.write = write_as_the_estop_lands  # type: ignore[method-assign]
    now = [0.0]
    dog = Watchdog(real, timeout_ms=100, clock=lambda: now[0])
    now[0] = 0.2
    with pytest.raises(RequestRefused, match="e-stop is engaged"):
        dog.poll()
    real.resume(approved_by="op")
    s.frames.clear()
    now[0] = 0.4  # no beat since: the stop that won the write keeps the lapse
    assert dog.poll() is False and goal_writes(s) == []
    real.close()


def test_a_beat_while_on_disconnect_is_applied_is_kept_so_the_next_lapse_applies_too():
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    now = {"t": 0.0}
    dog = Watchdog(real, timeout_ms=50, clock=lambda: now["t"])
    hold = d.hold

    def hold_as_the_commander_returns():
        hold()
        dog.beat()  # the commander's next command lands mid-apply (beat() takes no lock)

    d.hold = hold_as_the_commander_returns  # type: ignore[method-assign]
    now["t"] = 0.1
    assert dog.poll() is True
    d.hold = hold  # type: ignore[method-assign]
    now["t"] = 0.3  # the commander went quiet again after that one command: a new lapse
    assert dog.poll() is True
    real.close()


def test_closing_a_robot_ends_its_watchdog_thread_quietly():
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    errors: list[BaseException] = []
    old = threading.excepthook
    threading.excepthook = lambda a: errors.append(a.exc_value)
    try:
        thread = Watchdog(real, timeout_ms=20).start(period_s=0.005)._thread
        real.close()
        assert thread is not None
        thread.join(2.0)
    finally:
        threading.excepthook = old
    assert errors == [] and not thread.is_alive()


@pytest.mark.parametrize("period_s", [0.0, -0.01, float("nan"), float("inf"), 1e10])
def test_a_watchdog_period_outside_zero_to_timeout_max_is_refused(period_s):
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    with pytest.raises(ValueError, match="period_s must be"):
        Watchdog(real, timeout_ms=100).start(period_s=period_s).stop()
    real.close()


@pytest.mark.parametrize("timeout_ms", [float("nan"), float("inf")])
def test_a_watchdog_timeout_that_is_not_a_finite_number_is_refused(timeout_ms):
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    with pytest.raises(ValueError, match="timeout_ms must be a finite number > 0"):
        Watchdog(real, timeout_ms=timeout_ms)
    real.close()


@pytest.mark.parametrize("period_s", [np.float32(0.005), Decimal("0.005")], ids=["np.float32", "Decimal"])
def test_a_watchdog_period_given_as_another_number_type_still_paces_its_retries(period_s):
    real, s = _held_arm_whose_next_hold_fails()
    held, alive, dog = _watch_until_it_holds(real, s, period_s)
    assert held == 1 and alive and dog.errors == 1


def test_an_error_that_cannot_print_itself_does_not_end_the_watchdog_thread_and_is_named_by_its_type():
    class Unprintable(OSError):
        def __str__(self) -> str:
            raise RuntimeError("this error cannot format its own message")

    real, s = _held_arm_whose_next_hold_fails(Unprintable())
    faults: list[dict] = []
    real.subscribe("fault", faults.append)
    held, alive, dog = _watch_until_it_holds(real, s)
    assert held == 1 and alive and dog.errors == 1
    assert [f["reason"] for f in faults] == ["watchdog: poll failed (Unprintable); retrying every 0.005 s"]


def test_closing_a_robot_while_its_watchdog_applies_a_lapse_reports_no_fault():
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    faults: list[dict] = []
    real.subscribe("fault", faults.append)
    holding, closed = threading.Event(), threading.Event()

    def hold_as_the_robot_closes():
        holding.set()
        closed.wait(5.0)
        raise OSError("Attempting to use a port that is not open")  # pyserial, once close() has run

    d.hold = hold_as_the_robot_closes  # type: ignore[method-assign]
    dog = Watchdog(real, timeout_ms=20).start(period_s=0.005)
    assert holding.wait(5.0)
    real.close()
    closed.set()
    assert dog._thread is not None
    dog._thread.join(2.0)
    assert not dog._thread.is_alive() and dog.errors == 0 and faults == []


def test_a_fault_subscriber_can_stop_the_watchdog_whose_thread_it_runs_on():
    real, s = _held_arm_whose_next_hold_fails(fails=[10**6])
    box: dict = {}
    real.subscribe("fault", lambda f: box["dog"].stop())
    later: list[dict] = []
    real.subscribe("fault", later.append)
    box["dog"] = dog = Watchdog(real, timeout_ms=20)
    dog.start(period_s=0.005)
    assert dog._thread is not None
    dog._thread.join(5.0)
    assert not dog._thread.is_alive() and len(later) == 1 and dog.errors == 1
    real.close()


def test_workspace_bounds_validator_refuses_a_target_whose_tool_leaves_the_box():
    from strands_robots.robot.safety import WorkspaceBoundsValidator

    arm = Robot("so101")
    reach = WorkspaceBoundsValidator(
        lambda q: np.array([0.1 + 0.2 * q[0], 0.0, 0.2]), lo=(-0.2, -0.2, 0.0), hi=(0.2, 0.2, 0.4)
    )
    arm.validators.append(reach)
    arm.confirm_act({"shoulder_pan": 0.2})
    with pytest.raises(RequestRefused, match=r"tool x=0\.3000 m is outside the workspace \[-0\.2000, 0\.2000\]"):
        arm.confirm_act({"shoulder_pan": 1.0})


def test_resume_can_pin_the_stop_it_clears_so_a_stale_yes_is_refused():
    # A7: an operator's yes for the stop they read must not clear a later, different stop they never saw.
    arm = Robot("so101")
    arm.estop(by="op")  # stop #1, the one the operator reads
    seen = arm.estop_latch.count
    arm.resume(approved_by="op")  # someone clears #1 early
    arm.estop(by="validator")  # stop #2, a different hazard, never shown to op
    with pytest.raises(RequestRefused, match=r"resume is for stop #1, but the current stop is #2"):
        arm.resume(approved_by="op", stop=seen)
    assert arm.estopped  # #2 is still latched
    arm.resume(approved_by="op", stop=arm.estop_latch.count)  # pinned to #2: clears it
    assert not arm.estopped
    arm.close()


def test_a_stop_landing_between_the_pin_check_and_the_release_stays_latched():
    # A7 under the module's own concurrency: the pin check and the clear are ONE critical section, so a hazard
    # engaged from another thread between them is never cleared by a yes meant for the stop before it.
    arm = Robot("so101")
    latch = arm.estop_latch
    armed, engaged = threading.Event(), threading.Event()
    second = threading.Thread(target=lambda: (arm.estop(by="validator"), engaged.set()))

    class PausingEvent(threading.Event):
        def clear(self) -> None:  # the gap between the pin check and the clear
            if armed.is_set():
                second.start()
                engaged.wait(timeout=0.5)  # before the fix #2 lands here; after it the engage waits on the lock
            super().clear()

    latch._set = PausingEvent()
    arm.estop(by="op")  # stop #1, the one the operator reads
    seen = latch.count
    arm.resume(approved_by="op")  # someone clears #1 early
    armed.set()
    arm.resume(approved_by="op", stop=seen)  # the operator's yes for #1: passes the check, then #2 lands
    second.join(timeout=5)
    assert latch.count == 2 and latch.stopped_by == "validator"
    assert latch.engaged, "a hazard stopped between the check and the clear was silently released"
    with pytest.raises(RequestRefused, match=r"resume is for stop #1, but the current stop is #2"):
        arm.resume(approved_by="op", stop=seen)
    arm.close()


def test_the_latch_names_who_stopped_it_and_carries_a_wall_clock():
    # E1(b): the latch must be placeable on a calendar and name the stopper, so a stop has an audit trail.
    arm = Robot("so101")
    events: list[dict] = []
    arm.subscribe("estop", events.append)
    before = time.time()
    arm.estop(by="alice@corp")
    latch = arm.estop_latch
    assert latch.stopped_by == "alice@corp"
    assert latch.engaged_wall is not None and before <= latch.engaged_wall <= time.time()
    assert latch.engaged_at is not None  # the monotonic clock is kept for elapsed math
    assert events and events[-1]["by"] == "alice@corp"  # the event carries the stopper too
    arm.resume(approved_by="op")
    arm.close()


def test_a_local_press_records_no_operator_and_resume_without_a_pin_is_unchanged():
    # by/stop are additive: an unattributed local press and an unpinned resume behave as before.
    arm = Robot("so101")
    arm.estop()
    assert arm.estop_latch.stopped_by is None and arm.estop_latch.engaged_wall is not None
    arm.resume(approved_by="cagatay")  # no stop= pin: clears whatever is latched, as before
    assert not arm.estopped
    arm.close()


def test_k4_latch_release_clears_derived_state():
    """K4: release left stale reason/engaged_at behind; a resumed latch must read as fully clear."""
    arm = Robot("so101")
    arm.estop_latch.engage("test")
    assert arm.estop_latch.reason == "test" and arm.estop_latch.engaged_at is not None
    arm.estop_latch.release("operator")
    assert arm.estop_latch.reason is None
    assert arm.estop_latch.engaged_at is None
    assert arm.estop_latch.engaged_wall is None
    assert arm.estop_latch.resumed_by == "operator"
    arm.close()
