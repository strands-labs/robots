"""A hardware Robot's twin shares Simulation.default() with simulated bodies: it never pushes them, one bad
sample never resets their world, and a world that closes under another thread refuses cleanly."""

import threading
import time

import numpy as np
import pytest
from test_drivers import FakeLowState, FakeSub, make_driver

from strands_robots import Description, Robot, Simulation
from strands_robots.robot.drivers.unitree import UnitreeDriver
from strands_robots.types import JointState


def _real_g1(msg):
    return Robot("g1", driver=UnitreeDriver(Description.load("g1"), port="fake", subscriber=FakeSub(msg)))


def _touching(world, rid):
    """The robots (or "world", the floor) that ``rid``'s bodies touch right now."""
    out = set()
    for c in world.contacts():
        a, b = c.body_a.split("/")[0], c.body_b.split("/")[0]
        if rid in (a, b):
            out.add(b if a == rid else a)
    return out


class _MemoryDriver:
    """A third-party Driver held in memory: it reads ``q``, which the test sets, and its other verbs do nothing."""

    supports_position_hold, calibration, resolution = True, None, None

    def __init__(self, name):
        self.names = Description.load(name).action_keys
        self.q = np.zeros(len(self.names))
        for verb in ("write", "hold", "estop", "calibrate", "close"):
            setattr(self, verb, lambda *a, **k: None)
        self.pose, self.images, self.extras = (lambda: None), dict, dict

    def read(self):
        return JointState(self.names, self.q.copy(), np.zeros_like(self.q), np.zeros_like(self.q))


def _on_the_floor(robot):
    """The bodies of ``robot`` that touch the floor over 1 s of its world's steps, with nobody observing it."""
    bodies = set()
    for _ in range(50):
        robot.sim.control_step()
        for c in robot.sim.contacts():
            if "world" in (c.body_a, c.body_b):
                bodies |= {b.split("/", 1)[1] for b in (c.body_a, c.body_b) if b.startswith(f"{robot.id}/")}
    return bodies


@pytest.mark.parametrize("arm_first", [True, False], ids=["arm_first", "real_first"])
def test_a_twin_never_pushes_a_simulated_arm_off_its_targets(arm_first):
    # both stand on the so101 spawn point and every observe() places the twin there; #4749's headline is arm first
    arm = Robot("so101") if arm_first else None
    real = Robot("so101", driver=make_driver()[0], id="arm_real")
    arm = arm or Robot("so101")
    worst = 0.0
    for k in range(100):  # the sim arm sweeps for 50 ticks, then holds; the real arm idles, observed every tick
        s = min(1.0, k / 50)
        target = {"shoulder_pan": 1.2 * s, "shoulder_lift": -1.0 * s, "elbow_flex": 0.8 * s}
        obs = arm.act(target)
        real.observe()
        if k >= 60:
            worst = max(worst, *(abs(obs[j] - v) for j, v in target.items()))
    assert worst < 0.05  # dev: 1.013 rad arm first, 2.959 rad real first
    assert _touching(arm.sim, real.twin.id) <= {"world"}
    real.close()


def test_a_twin_rests_on_the_floor_and_touches_nothing_else():
    # a floating twin still needs the floor: with no collisions at all a G1 twin falls through it
    sim_g1 = Robot("g1")  # a simulated G1 standing where the real G1's twin spawns
    real = _real_g1(FakeLowState())
    twin, touched = real.twin, set()
    for _ in range(50):  # 1 s of world time; nobody observes the real G1, so the twin's base is simulated
        sim_g1.act(sim_g1.observe().joints)
        touched |= _touching(twin.sim, twin.id)
    assert touched == {"world"} and twin.pose.position[2] > 0.05  # dev: it touches the sim G1 and itself too
    real.close()


def test_a_twin_stands_on_the_floor_as_its_plain_body_does():
    # cassie's shins, tarsi and feet carry only contact bits 2 and 4: they meet each other, never the floor
    plain = Robot("cassie", sim=Simulation(), id="plain")  # alone in a world of its own
    real = Robot("cassie", driver=_MemoryDriver("cassie"))  # cassie has no built-in driver: a third-party one
    # with the twin bit on every colliding geom, the twin also stood on its feet and tarsi, at 0.296 m, not 0.154 m
    assert _on_the_floor(real.twin) == _on_the_floor(plain)
    assert real.twin.pose.position[2] == pytest.approx(plain.pose.position[2], abs=0.01)
    real.close()
    plain.sim.close()


def test_a_nan_reading_never_resets_the_twins_world():
    # a NaN in qpos makes MuJoCo reset the whole world on its next step: its time and every robot in it
    msg = FakeLowState()
    real = _real_g1(msg)
    arm = Robot("so101", position=[0.0, 1.5, 0.0], id="sim_arm")  # in the twin's world, clear of the twin
    for _ in range(50):
        arm.act({"shoulder_pan": 0.8})
    t = arm.sim.time
    msg.motor_state[real.body.index[3]].q = float("nan")  # one bad sample on the bus
    assert np.isnan(real.observe().joints.positions).any()  # the observation still says what was measured
    assert np.isfinite(real.twin.observe().joints.positions).all()
    arm.act({"shoulder_pan": 0.8})
    assert arm.sim.time > t and arm.observe()["shoulder_pan"] == pytest.approx(0.8, abs=0.02)  # dev: 0.020 s, 0.0
    real.close()


@pytest.mark.parametrize("value", [1e11, np.inf], ids=["1e11", "inf"])
def test_a_huge_reading_never_resets_the_twins_world(value):
    # kinova_gen3's joint_1 has no limit, so nothing clips its reading; MuJoCo resets at a qpos beyond 1e10 (mjMAXVAL)
    drv = _MemoryDriver("kinova_gen3")
    real = Robot("kinova_gen3", driver=drv, position=[0.0, -1.5, 0.0])
    arm = Robot("so101", position=[0.0, 1.5, 0.0], id="sim_arm")  # in the twin's world, clear of the twin
    for _ in range(50):
        arm.act({"shoulder_pan": 0.8})
    t = arm.sim.time
    drv.q[0] = value  # one bad sample from the driver
    assert real.observe()["joint_1"] == value  # the observation still says what was read
    arm.act({"shoulder_pan": 0.8})
    assert arm.sim.time > t and arm.observe()["shoulder_pan"] == pytest.approx(0.8, abs=0.02)  # dev: 0.020 s, 0.0
    real.close()


def test_close_waits_for_the_step_another_thread_is_taking():
    sim = Simulation()
    Robot("so101", sim=sim)
    closed = threading.Event()
    with sim.lock:  # another thread is inside a step
        th = threading.Thread(target=lambda: (sim.close(), closed.set()))
        th.start()
        assert not closed.wait(0.2)  # dev: closed at once, under the step
    th.join(5)
    assert closed.is_set() and sim.backend.model is None


@pytest.mark.parametrize(
    "verb",
    [lambda sim, arm: arm.act({"shoulder_pan": 0.2}), lambda sim, arm: sim.reset()],
    ids=["act", "reset"],
)
def test_close_while_another_thread_uses_the_world_raises_a_clean_error(verb):
    seen = set()
    for _ in range(10):
        sim = Simulation()
        arm, errors = Robot("so101", sim=sim), []

        def loop(sim=sim, arm=arm, errors=errors):  # given the world too: arm.sim is None once it closes
            try:
                for _ in range(10_000):
                    verb(sim, arm)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        th = threading.Thread(target=loop)
        th.start()
        time.sleep(0.02)
        sim.close()
        th.join(10)
        seen |= {type(err) for err in errors}
    assert seen == {RuntimeError}  # dev: TypeError from mj_step() (act) or mj_forward() (reset) on the closed model


def test_a_real_robot_outlives_its_twins_world():
    real = Robot("so101", driver=make_driver()[0])
    world, out = real.twin.sim, []
    with world.lock:  # the twin's world is busy, so real.observe() waits there to mirror ...
        th = threading.Thread(target=lambda: out.append(real.observe()))
        th.start()
        time.sleep(0.2)
        world.close()  # ... and the world closes while it waits
    th.join(5)
    assert out and real.twin.sim is None  # dev: observe() raises AttributeError from the closed twin
    assert real.act({"shoulder_pan": 0.3})["shoulder_pan"] == pytest.approx(0.3, abs=1e-3)  # the bus still moves
    real.close()


@pytest.mark.parametrize(
    "use",
    [lambda s: s.camera("c"), lambda s: s.randomize(friction=1.1), lambda s: s.render()],
    ids=["camera", "randomize", "render"],
)
def test_a_closed_world_is_never_rebuilt(use):
    sim = Simulation()
    Robot("so101", sim=sim)
    sim.close()
    with pytest.raises(RuntimeError, match="closed"):
        use(sim)
    assert sim.backend.model is None  # dev: the closed world's so101 compiled again
