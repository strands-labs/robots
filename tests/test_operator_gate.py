"""The operator gate: a real body never moves because of a default (#4749 part 5)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
import test_drivers
from test_drivers import make_driver

from strands_robots import Robot
from strands_robots.robot.drivers import feetech as ft


def goal_writes(serial: object) -> list[bytes]:
    """Frames that would move a servo: a SYNC_WRITE of Goal_Position or of Torque_Enable."""
    return [
        f
        for f in serial.frames  # type: ignore[attr-defined]
        if f[4] == ft.SYNC_WRITE and f[5] in (ft.REG_GOAL_POSITION, ft.REG_TORQUE_ENABLE)
    ]


def test_direct_act_is_the_developers_call_and_needs_no_gate():
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    real.act({"shoulder_pan": 0.2})  # plain Python: the developer is the operator
    assert real.observe()["shoulder_pan"] == pytest.approx(0.2, abs=1e-2)


@pytest.mark.parametrize(
    "confirm, match",
    [(None, "pass confirm="), (lambda action: False, "refused")],
    ids=["no-callable", "operator-says-no"],
)
def test_a_refused_hold_writes_nothing(confirm, match):
    d, s = make_driver()
    real = Robot("so101", driver=d, confirm=confirm)
    before = real.observe().joints.positions
    s.frames.clear()
    with pytest.raises(PermissionError, match=match):
        real.confirm_act(before)
    assert goal_writes(s) == [] and d.extras()["torque"] is False
    np.testing.assert_array_equal(real.observe().joints.positions, before)
    real.close()


def test_an_allowed_hold_writes_and_the_twin_mirrors_it():
    d, s = make_driver()
    asked: list[np.ndarray] = []
    real = Robot("so101", driver=d, confirm=lambda action: asked.append(np.array(action)) or True)
    measured = real.observe().joints
    real.confirm_act(measured.positions)
    assert len(asked) == 1 and len(goal_writes(s)) == 2  # torque on, then one Goal_Position write
    for name in measured.names:  # the hold lands on the bus within one count of where it was read
        goal = int.from_bytes(s.regs[d.ids[measured.names.index(name)]][ft.REG_GOAL_POSITION :][:2], "little")
        assert abs(goal - d.rad_to_counts(name, measured[name])) <= 1
    twin = real.twin
    assert twin is not None and twin.sim is not None
    twin.sim.step(200)
    np.testing.assert_allclose(twin.observe().joints.positions, measured.positions, atol=0.05)
    real.close()


def test_a_servo_resting_past_its_limit_can_still_hold_and_return():
    """A joint that reads a hair past its hard range (no clamp in counts_to_rad) can still be commanded to
    where it is, or back toward the range; only a target driving it further out is refused (Yin #4774)."""
    d, s = make_driver()
    s.actual[d.ids[3]] = 3072 + 40  # wrist_flex encoder past range_max: it reads outside [lo, hi]
    real = Robot("so101", driver=d, confirm=lambda action: True, twin=False)
    hold = real.observe().joints
    assert hold["wrist_flex"] > real.description.limits["wrist_flex"][1]  # genuinely past its hard limit
    s.frames.clear()
    real.confirm_act(hold.positions)  # "hold where you are" is allowed, not refused for where the body is
    assert len(goal_writes(s)) == 2  # torque on, then one Goal_Position write
    toward = {"wrist_flex": real.description.limits["wrist_flex"][1]}  # moving back toward the range: allowed
    real.confirm_act(toward)
    with pytest.raises(PermissionError, match="wrist_flex"):  # driving it further out: refused
        real.confirm_act({"wrist_flex": float(hold["wrist_flex"]) + 0.2})
    real.close()


def test_confirmed_out_of_band_and_sim_bodies_are_ungated():
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    real.confirm_act({"shoulder_pan": 0.3}, confirmed=True)  # an operator said yes elsewhere (a dashboard dialog)
    assert real.observe()["shoulder_pan"] == pytest.approx(0.3, abs=1e-2)
    Robot("so101").confirm_act({"shoulder_pan": 0.3})  # simulated: no gate
    with pytest.raises(TypeError, match="confirm"):
        Robot("so101", confirm=lambda a: True)  # a gate on a simulated body is a mistake worth naming


class OneWire(test_drivers.FakeSerial):
    """A half-duplex wire that notices a second request sent before the first reply was read."""

    def __init__(self):
        super().__init__()
        self.waiting = False
        self.collisions = 0

    def write(self, data):
        if self.waiting:
            self.collisions += 1
        super().write(data)
        self.waiting = data[4] != ft.SYNC_WRITE
        time.sleep(0.0005)  # the reply is on the wire for a moment

    def read(self, n):
        self.waiting = False
        return super().read(n)


def test_two_threads_on_one_hardware_robot_take_turns_on_the_bus():
    d, s = make_driver(serial=OneWire())
    real = Robot("so101", driver=d, confirm=lambda action: True, twin=False)
    hold = real.observe().joints.positions
    errors: list[BaseException] = []

    def hammer(verb):
        try:
            for _ in range(40):
                verb()
        except Exception as exc:  # the test reports what a thread raised
            errors.append(exc)

    threads = [
        threading.Thread(target=hammer, args=(lambda: real.observe(images=False),)),
        threading.Thread(target=hammer, args=(lambda: real.confirm_act({"shoulder_pan": hold[0]}),)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and s.collisions == 0
    real.close()


# --- K6: an agent's run on a real arm asks one approval per run, not confirm= every tick -----------------------


def test_k6_agent_run_asks_approve_once_not_confirm_every_tick():
    """The old path asked confirm() on every tick of an agent run (30 dialogs a second); approve= is asked once."""
    from strands_robots.robot.run import RunPlan, from_agent

    plans: list = []
    per_tick: list = []
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False, approve=lambda p: plans.append(p) or True,
                 confirm=lambda t: per_tick.append(t) or True)  # fmt: skip
    with from_agent():
        res = real.run(lambda obs: {}, steps=4, hz=50, instruction="wave")
    assert res.stopped_by == "steps" and res.steps == 4
    assert len(plans) == 1 and per_tick == []  # one plan, the per-command confirm never asked
    assert plans[0] == RunPlan("so101", plans[0].policy, "wave", 50, None, 4)
    real.close()


def test_k6_without_approve_an_agent_run_moves_nothing():
    from strands_robots.robot.run import from_agent
    from strands_robots.robot.safety import RequestRefused

    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False)
    s.frames.clear()
    with from_agent(), pytest.raises(RequestRefused, match="approve="):
        real.run(lambda obs: {}, steps=3, hz=50)
    assert goal_writes(s) == []  # refused before anything was acquired or written
    real.close()


def test_k6_operator_refuses_the_run():
    from strands_robots.robot.run import from_agent
    from strands_robots.robot.safety import RequestRefused

    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False, approve=lambda p: False)
    s.frames.clear()
    with from_agent(), pytest.raises(RequestRefused, match="refused the run"):
        real.run(lambda obs: {}, steps=3, hz=50)
    assert goal_writes(s) == []
    real.close()


class _LeRobotUnits:
    """A policy in LeRobot range units: holds what it sees, 99999 on shoulder_pan at tick 4."""

    units = "lerobot"

    def __init__(self):
        self.k = 0

    def act(self, obs, instruction=None):
        self.k += 1
        out = np.array(obs.joints.positions, dtype=float)
        if self.k == 4:
            out[0] = 99999.0
        return out

    def reset(self):
        pass


def _rad_policy():
    k = [0]

    def pol(obs):
        k[0] += 1
        return {"shoulder_pan": 99.0} if k[0] == 4 else {}  # beyond the hard range on tick 4

    return pol


@pytest.mark.parametrize("make_policy", [_rad_policy, _LeRobotUnits], ids=["rad", "lerobot_units"])
def test_k6_every_step_still_passes_limits_and_validators(make_policy):
    from strands_robots.robot.run import from_agent

    seen: list[np.ndarray] = []
    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False, approve=lambda p: True)
    real.validators.append(lambda r, req: seen.append(req.targets.copy()))
    pol = make_policy()

    with from_agent():
        res = real.run(pol, steps=10, hz=50)
    assert len(seen) == 3  # the validator saw every written step; tick 4 refused before it
    assert res.stopped_by == "fault" and res.fault is not None and "range" in res.fault
    real.close()


def test_k6_developer_run_is_not_gated():
    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False)  # no approve, no confirm: the developer is the operator
    before = len(goal_writes(s))
    real.run(lambda obs: {}, steps=3, hz=50)
    assert len(goal_writes(s)) > before  # the hold wrote goal frames with no gate
    real.close()


def test_k6_start_carries_the_agent_mark_to_its_thread():
    from strands_robots.robot.run import from_agent
    from strands_robots.robot.safety import RequestRefused

    d, _ = make_driver()
    real = Robot("so101", driver=d, twin=False)
    with from_agent():
        task = real.start(lambda obs: {}, steps=3, hz=50)
    task._thread.join(5)
    with pytest.raises(RequestRefused, match="approve="):
        _ = task.result
    real.close()
