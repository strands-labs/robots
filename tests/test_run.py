import time

import pytest

from strands_robots import JointState, Policy, Robot, Simulation
from strands_robots.policy import SequencePolicy
from strands_robots.robot import RunResult


def test_robot_is_a_class_and_a_body_in_the_default_world():
    arm = Robot("so101")
    assert type(arm) is Robot and arm.sim is Simulation.default() and not arm.is_real
    d = arm.describe()
    assert d["id"] == "so101" and d["world"] == "mujoco" and d["real"] is False and d["task"] is None
    assert repr(arm) == "Robot('so101', id='so101', mujoco, dof=6)"
    arm.close()
    assert arm.sim is None and "so101" not in Simulation.default().robots


@pytest.mark.parametrize("limit", [{"steps": 6}, {"seconds": 0.2}])
@pytest.mark.parametrize("world", [False, True])
def test_a_run_reports_the_limit_its_caller_gave(limit, world):
    arm = Robot("so101")
    hold = lambda o: o.joints.positions  # noqa: E731
    res = arm.sim.run({arm: hold}, hz=30, **limit)[arm.id] if world else arm.run(hold, hz=30, **limit)
    assert (res.stopped_by, res.steps) == (next(iter(limit)), 6)


def test_run_for_steps_reaches_target_and_measures():
    arm = Robot("so101")
    res = arm.run(lambda o: {"shoulder_pan": 0.4, "gripper": 0.8}, hz=50, steps=60)
    assert isinstance(res, RunResult) and res.steps == 60 and res.stopped_by == "steps"
    assert res.seconds == pytest.approx(60 / 50)  # sim time, not wall time
    assert res.hz_achieved == pytest.approx(50)  # the rate the body saw, on its clock (sim time)
    assert res.steps / res.wall_seconds > 50  # and the host ran it faster than real time
    assert res.final is not None and res.final["shoulder_pan"] == pytest.approx(0.4, abs=0.02)


def test_run_for_seconds_sets_the_sim_step_count():
    arm = Robot("so101")
    t0 = arm.sim.time
    arm.run(lambda o: o.joints.positions, hz=25, seconds=1.0)
    assert arm.sim.time == pytest.approx(t0 + 1.0, abs=1e-9)


def test_run_stop_when_and_instruction():
    arm = Robot("so101")
    seen = []

    def pol(obs, instruction=None):
        seen.append(instruction)
        return {"elbow_flex": 1.0}

    res = arm.run(pol, instruction="bend", hz=50, stop_when=lambda o: o["elbow_flex"] > 0.5, steps=500)
    assert res.stopped_by == "stop_when" and 0 < res.steps < 500 and seen[0] == "bend"


def test_run_needs_a_bound_and_valid_hz():
    arm = Robot("so101")
    with pytest.raises(ValueError, match="seconds=, steps= or stop_when="):
        arm.run(lambda o: o.joints.positions)
    with pytest.raises(ValueError, match="hz"):
        arm.run(lambda o: o.joints.positions, hz=0, steps=1)
    with pytest.raises(ValueError, match="hz"):  # refused before the spec is resolved, so nothing is loaded
        arm.run("zenoh://127.0.0.1:1/nobody", hz=-1, steps=1)


def test_start_stop_background_task():
    arm = Robot("so101")
    task = arm.start(lambda o: {"wrist_roll": 0.5}, hz=200, seconds=30, realtime=True)  # paced: still running at 0.2 s
    assert task.status == "running"
    time.sleep(0.2)
    res = arm.stop()
    assert res is not None and res.stopped_by == "stopped" and 5 < res.steps < 6000
    assert task.status == "done" and not task.running
    arm.start(lambda o: o.joints.positions, seconds=5)
    with pytest.raises(RuntimeError, match="already has a running task"):
        arm.start(lambda o: o.joints.positions, seconds=5)
    arm.stop()


def test_two_threads_starting_one_robot_at_once_start_one_run():
    """``start`` checked for a running task and then stored its own: two threads released together both started a
    run on one bus (the fake Feetech bus), as two tool calls from one model turn do."""
    import threading

    from test_drivers import make_driver

    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    together, got = threading.Barrier(2), []

    def start() -> None:
        together.wait()
        try:
            got.append(type(real.start(lambda o: {}, steps=20, hz=20)).__name__)
        except RuntimeError as e:
            got.append(str(e))

    threads = [threading.Thread(target=start) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    real.stop(timeout=5)
    assert sorted(got) == ["Task", "robot 'so101' already has a running task; call stop() first"], got
    real.close()


def test_background_task_error_surfaces_in_result():
    arm = Robot("so101")

    def bad(obs):
        raise RuntimeError("policy exploded")

    task = arm.start(bad, steps=3)
    task._thread.join(2)
    assert task.status == "error"
    with pytest.raises(RuntimeError, match="policy exploded"):
        _ = task.result
    arm.close()  # teardown does not re-raise the task's error, and the body leaves the world
    assert arm.sim is None and "so101" not in Simulation.default().robots


def test_set_gains_takes_one_kp_and_kd_per_actuator():
    arm = Robot("so101")
    arm.set_gains([50.0] * 6, [1.0] * 6)
    with pytest.raises(ValueError, match="for 6 actuators"):
        arm.set_gains([50.0] * 5, [1.0] * 6)


def test_the_operator_gate_passes_a_simulated_body():
    arm = Robot("so101")
    assert arm.gate({"shoulder_pan": 0.2})[0] == pytest.approx(0.2)
    assert arm.confirm_act({"shoulder_pan": 0.2}).t > 0


def test_realtime_paces_a_sim_run_to_the_wall_clock():
    arm = Robot("so101")
    t0 = time.perf_counter()
    res = arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=6, realtime=True)
    wall = time.perf_counter() - t0
    assert res.steps == 6 and 0.2 <= wall <= 1.0  # 6 steps at 20 Hz ≈ 0.3 s, not the ~5 ms a free-running sim takes
    t0 = time.perf_counter()
    arm.run(SequencePolicy([{"shoulder_pan": 0.1}]), hz=20, steps=6)
    assert time.perf_counter() - t0 < 0.2


def test_act_takes_the_joints_an_observation_measured():
    arm = Robot("so101")
    arm.act({"shoulder_pan": 0.4}, steps=200)
    measured = arm.observe().joints
    assert arm.coerce(measured) == pytest.approx(measured.positions)  # a hold at what was read
    with pytest.raises(KeyError, match="unknown joint"):
        arm.coerce(JointState.zeros(["shoulder_pan", "not_a_joint"]))


class _At30(SequencePolicy):
    hz = 30.0


def test_both_loops_replay_at_the_rate_the_policy_declares_and_refuse_another():
    arm = Robot("so101")
    assert arm.run(_At30([{"shoulder_pan": 0.1}]), seconds=1).hz_target == 30.0
    assert arm.sim.run({arm: _At30([{"shoulder_pan": 0.1}])}, seconds=1)[arm.id].hz_target == 30.0
    with pytest.raises(ValueError, match="runs at 30 Hz; hz=50"):
        arm.run(_At30([{"shoulder_pan": 0.1}]), hz=50, steps=1)
    with pytest.raises(ValueError, match="runs at 30 Hz; hz=50"):
        arm.sim.run({arm: _At30([{"shoulder_pan": 0.1}])}, hz=50, steps=1)
    other = Robot("so101", id="so101_2")

    class At20(SequencePolicy):
        hz = 20.0

    with pytest.raises(ValueError, match="different rates"):
        arm.sim.run({arm: _At30([{"shoulder_pan": 0.1}]), other: At20([{"shoulder_pan": 0.1}])}, steps=1)


class _Holds:
    def act(self, obs, *, instruction=None):
        return obs.joints.positions

    def reset(self):
        pass


@pytest.mark.parametrize(
    "name, value, said",
    [
        ("hz", 0, "hz must be a number > 0, got 0"),
        ("hz", "50", "hz must be a number > 0, got '50'"),
        ("hz", True, "hz must be a number > 0, got True"),
        ("realtime", "no", "realtime must be True or False, got 'no'"),
        ("requires_position_hold", "0", "requires_position_hold must be True or False"),
        ("gains", "strong", r"gains must be a \(kp, kd\) pair, got 'strong'"),
        ("attach", True, "attach must be a method, got True"),
        ("detach", "later", "detach must be a method, got 'later'"),
        ("requirements", object(), "requirements must be a method"),
        ("needs_observation", False, "needs_observation must be a method, got False"),
        ("requirements", lambda: {}, r"requirements\(\) must return Requirements, got \{\}"),
    ],
)
def test_a_policy_hook_of_the_wrong_type_is_refused_by_name_before_step_one(name, value, said):
    arm, pol = Robot("so101"), _Holds()
    setattr(pol, name, value)
    with pytest.raises(TypeError, match=said):
        arm.run(pol, steps=1)
    with pytest.raises(TypeError, match=said):
        arm.sim.run({arm: pol}, steps=1)
    arm.close()


def test_stop_raises_a_run_error_once_and_a_run_that_will_not_end_times_out():
    arm = Robot("so101")

    def bad(obs):
        raise ZeroDivisionError("policy bug")

    arm.start(bad, steps=3)._thread.join(2)
    with pytest.raises(ZeroDivisionError, match="policy bug"):
        arm.stop()
    assert arm.stop() is None  # reported once; the robot is free again
    gate = __import__("threading").Event()
    task = arm.start(lambda o: gate.wait(5) and o.joints.positions, steps=2)
    with pytest.raises(TimeoutError, match="still going after 0.2 s"):
        arm.stop(timeout=0.2)
    assert task.running
    gate.set()
    assert arm.stop().stopped_by == "stopped"
    arm.close()


def test_a_policy_that_emits_nan_ends_the_run_as_a_fault_before_the_write():
    arm = Robot("so101")
    calls = []

    def pol(obs):
        calls.append(1)
        return {"shoulder_pan": float("nan") if len(calls) == 3 else 0.1}

    res = arm.run(pol, hz=50, steps=10)
    assert res.stopped_by == "fault" and res.steps == 2 and "non-finite" in res.fault
    assert all(abs(v) < 10 for v in arm.observe().joints.positions)  # the world never saw the NaN


def _zero_on_disconnect(robot, joint="shoulder_pan.pos"):
    """Give one of a sim body's controls on_disconnect='zero' so a fault's effect is observable."""
    import dataclasses as dc

    from strands_robots.robot.device.build import manifest_key

    m = robot.manifest
    controls = tuple(dc.replace(c, on_disconnect="zero") if c.name == joint else c for c in m.controls)
    robot.__dict__["_manifest"] = (manifest_key(robot), dc.replace(m, controls=controls))


def test_a_faulted_run_puts_the_body_in_its_declared_on_disconnect_state():
    arm = Robot("so101")
    _zero_on_disconnect(arm)
    arm.run(lambda o: {"shoulder_pan": 0.8, "shoulder_lift": 0.3}, hz=50, steps=10)
    assert arm.sim.backend.data.ctrl[0] == pytest.approx(0.8)  # commanded away from the disconnect target
    calls = []

    def pol(o):
        calls.append(1)
        return {"shoulder_pan": float("nan") if len(calls) == 3 else 0.8}

    res = arm.run(pol, hz=50, steps=10)
    assert res.stopped_by == "fault"
    assert arm.sim.backend.data.ctrl[0] == pytest.approx(0.0)  # on_disconnect='zero' drove shoulder_pan to 0
    held = arm.observe().joints.positions
    assert arm.sim.backend.data.ctrl[1] == pytest.approx(held[1])  # a 'hold' control keeps its measured value


def test_lockstep_applies_on_disconnect_only_to_the_robot_that_faulted():
    sim = Simulation()
    a, b = Robot("so101", sim=sim), Robot("so101", sim=sim)
    _zero_on_disconnect(a)
    a.act({"shoulder_pan": 0.8})
    calls = []

    def faulting(o):
        calls.append(1)
        return {"shoulder_pan": float("nan") if len(calls) == 3 else 0.8}

    res = sim.run({a: faulting, b: lambda o: {"shoulder_pan": 0.5}}, steps=10)
    assert res[a.id].stopped_by == "fault" and "non-finite" in res[a.id].fault
    assert res[b.id].fault is None  # b did not fault; the world's run ended with a
    ia = sim.backend._index[a.id].actuator_ids[0]  # a's shoulder_pan actuator
    ib = sim.backend._index[b.id].actuator_ids[0]
    assert sim.backend.data.ctrl[ia] == pytest.approx(0.0)  # a faulted: on_disconnect='zero' applied
    assert sim.backend.data.ctrl[ib] != pytest.approx(0.0)  # b kept running, no disconnect applied


@pytest.mark.parametrize(
    ("spec", "says"),
    [
        ("hold", "use robot.hold() to stand still, robot.act(..., seconds=) to reach a pose"),
        ("nope", "known presets: holosoma, microduck;"),  # hold is not offered as a preset
    ],
)
def test_hold_is_not_a_policy_spec_and_the_refusal_names_the_verbs(spec, says):
    with pytest.raises(ValueError) as err:
        Policy(spec)
    assert says in str(err.value) and "stand still at the pose" not in str(err.value)


def test_act_seconds_reaches_the_pose_in_one_call_and_one_plain_act_is_one_step_along():
    one, ramp = Robot("so101", sim=Simulation()), Robot("so101", sim=Simulation())
    home = {"shoulder_lift": 0.8, "elbow_flex": -0.8, "gripper": 1.0}
    stepped = one.act(home).joints.as_dict()
    t0, reached = ramp.sim.time, ramp.act(home, seconds=1.0).joints.as_dict()
    assert ramp.sim.time - t0 == pytest.approx(1.0, abs=0.04)  # the world advanced the seconds asked for
    for j, q in home.items():
        assert reached[j] == pytest.approx(q, abs=0.05), j
        assert abs(stepped[j] - q) > 0.1, j  # one step of a far target is still on its way
    with pytest.raises(ValueError, match="not both"):
        ramp.act(home, steps=2, seconds=1.0)
    with pytest.raises(ValueError, match="> 0"):
        ramp.act(home, seconds=0)
