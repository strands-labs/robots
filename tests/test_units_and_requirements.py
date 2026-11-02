"""Units belong to the policy's dataset; a policy declares its needs (experiments/SMOLVLA.md in the reference).

The pure parts first, then what ``Robot.run`` does with them.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots import Description, JointState, Observation, Policy, Robot, Simulation
from strands_robots.policy import Requirements
from strands_robots.units import Units, declared_units, lerobot_units, resolve_units


@pytest.fixture(scope="module")
def so101():
    return Description.load("so101")


def test_lerobot_units_round_trip_on_so101(so101):
    u = Units.for_robot(so101, "lerobot")
    lo = np.array([so101.limits[k][0] for k in so101.action_keys])
    hi = np.array([so101.limits[k][1] for k in so101.action_keys])
    # body joints map their limits onto -100..100, the gripper onto 0..100
    np.testing.assert_allclose(u.from_rad(lo), [-100, -100, -100, -100, -100, 0])
    np.testing.assert_allclose(u.from_rad(hi), [100, 100, 100, 100, 100, 100])
    np.testing.assert_allclose(u.from_rad((lo + hi) / 2), [0, 0, 0, 0, 0, 50])
    q = np.linspace(lo, hi, 7)
    np.testing.assert_allclose(u.to_rad(u.from_rad(q)), q, atol=1e-12)


def test_rad_units_are_identity_and_unknown_names_refuse(so101):
    assert Units.for_robot(so101, "rad").is_identity
    with pytest.raises(ValueError, match="unknown units 'degrees'"):
        Units.for_robot(so101, "degrees")


class _Speaks:
    def __init__(self, units=None):
        if units is not None:
            self.units = units


class _InDegrees:
    """Declares its units per robot, as a checkpoint run with norm= does: LeRobot's, every arm joint in degrees."""

    def units_for(self, description):
        return lerobot_units(description, {"*": "degrees", "gripper": "range_0_100"})


def test_resolve_units_follows_the_declaration_and_a_scheme_only_speaks_for_a_policy_that_cannot(so101):
    lerobot = _Speaks("lerobot")
    assert resolve_units(lerobot, "auto", so101).name == "lerobot"
    assert resolve_units(lerobot, "lerobot", so101).name == "lerobot"  # agreeing is fine
    own = resolve_units(_InDegrees(), "auto", so101).modes
    assert resolve_units(_InDegrees(), "lerobot", so101).modes == own  # the policy's modes, not the registry's
    for other in ("rad", None):  # radians for numbers the policy says are LeRobot's: the factor-57 hazard
        with pytest.raises(ValueError, match=f"declares units 'lerobot', but units={other!r} was passed"):
            resolve_units(lerobot, other, so101)
    with pytest.raises(ValueError, match="declares units 'rad', but units='lerobot' was passed"):
        resolve_units(_Speaks("rad"), "lerobot", so101)
    assert resolve_units(_Speaks("rad"), "rad", so101) is None
    assert resolve_units(_Speaks(), "auto", so101) is None  # cannot say -> radians
    assert resolve_units(_Speaks(), "rad", so101) is None and resolve_units(_Speaks(), None, so101) is None
    assert resolve_units(_Speaks(), "lerobot", so101).name == "lerobot"  # a scheme says it for the policy
    with pytest.raises(ValueError, match="policy declares units 'furlongs'"):
        declared_units(_Speaks("furlongs"))


def _obs(n_joints: int, **cams: tuple[int, int]) -> Observation:
    names = tuple(f"j{i}" for i in range(n_joints))
    images = {c: np.zeros((h, w, 3), np.uint8) for c, (w, h) in cams.items()}
    return Observation(0.0, JointState.zeros(names), images=images)


_THREE_CAMERAS = Requirements(images={"camera1": (256, 256), "camera2": (256, 256), "camera3": None}, state_dim=6)


@pytest.mark.parametrize(
    ("obs", "refusal"),
    [
        (_obs(6), r"needs camera\(s\) \['camera1', 'camera2', 'camera3'\] but the observation has none"),
        (
            _obs(6, camera1=(256, 256), camera2=(320, 240), camera3=(64, 64)),
            "camera2: has 320x240, needs 256x256",
        ),
        (
            _obs(29, camera1=(256, 256), camera2=(256, 256), camera3=(64, 64)),
            "expects a 6-dim state, the robot reports 29 joints",
        ),
        (_obs(6, camera1=(256, 256), camera2=(256, 256), camera3=(64, 64)), None),
    ],
)
def test_requirements_refuse_naming_what_is_missing(obs, refusal):
    if refusal is None:
        _THREE_CAMERAS.check(obs)
        return
    with pytest.raises(ValueError, match=refusal):
        _THREE_CAMERAS.check(obs)


# Robot.run and Simulation.run bind a policy the same way: one table, both loops.
LOOPS = {
    "robot.run": lambda arm, pol, steps: arm.run(pol, steps=steps),
    "sim.run": lambda arm, pol, steps: arm.sim.run({arm: pol}, steps=steps),
}
loops = pytest.mark.parametrize("loop", LOOPS.values(), ids=LOOPS.keys())


@pytest.fixture
def arm():
    with Simulation() as sim:
        yield Robot("so101", sim=sim)


class _HoldInLerobotUnits(Policy):
    """A policy that speaks lerobot: it sees -100..100 and asks for the gripper half open (50)."""

    units = "lerobot"

    def __init__(self):
        self.seen = []

    def act(self, obs, *, instruction=None):
        self.seen.append(obs.joints.positions.copy())
        a = obs.joints.positions.copy()
        a[-1] = 50.0
        return a


@loops
def test_run_converts_both_directions_when_the_policy_declares_units(arm, loop):
    pol = _HoldInLerobotUnits()
    loop(arm, pol, 3)
    assert all(abs(s[:-1]).max() <= 100 for s in pol.seen) and 0 <= pol.seen[0][-1] <= 100  # it saw bus units
    loop(arm, pol, 20)
    assert abs(arm.observe()["gripper"] - 0.785) < 0.15  # the arm got radians: 50 is half of 0..1.745 rad, not 50 rad


class _Needs(Policy):
    def __init__(self, req):
        self.req = req

    def requirements(self):
        return self.req

    def act(self, obs, *, instruction=None):
        self.saw = sorted(obs.images)
        return obs.joints.positions


@loops
def test_run_refuses_before_inference_naming_the_missing_camera(arm, loop):
    with pytest.raises(ValueError, match=r"needs camera\(s\) \['camera1', 'camera2', 'camera3'\]"):
        loop(arm, _Needs(_THREE_CAMERAS), 1)
    for i, size in ((1, (256, 256)), (2, (320, 240)), (3, (64, 64))):
        arm.sim.camera(f"camera{i}", size=size, attach=arm)
    with pytest.raises(ValueError, match="camera2: has 320x240, needs 256x256"):
        loop(arm, _Needs(_THREE_CAMERAS), 1)


@loops
def test_run_passes_when_the_requirements_are_met_and_the_policy_sees_the_cameras(arm, loop):
    for i in (1, 2, 3):
        arm.sim.camera(f"camera{i}", size=(256, 256), attach=arm)
    pol = _Needs(_THREE_CAMERAS)
    loop(arm, pol, 2)
    assert pol.saw == ["camera1", "camera2", "camera3"]


def test_run_names_a_state_length_mismatch():
    with Simulation() as sim, pytest.raises(ValueError, match="expects a 6-dim state, the robot reports"):
        Robot("g1", sim=sim).run(_Needs(Requirements(state_dim=6)), steps=1)


_PAN_ELBOW = {"shoulder_pan": 60.0, "elbow_flex": 30.0}


@pytest.mark.parametrize(
    ("returned", "refusal"),
    [
        (lambda seen: dict(_PAN_ELBOW), None),  # a mapping in policy units, other joints unnamed
        (lambda seen: JointState(tuple(_PAN_ELBOW), np.array([60.0, 30.0]), np.zeros(2), np.zeros(2)), None),
        (lambda seen: {**dict(zip(seen.joints.names, seen.joints.positions, strict=True)), **_PAN_ELBOW}, None),
        (lambda seen: {"shoulder_pan.pos": 60.0}, "LeRobot '<joint>.pos' names"),
        (lambda seen: {"tail": 1.0}, r"unknown joint\(s\) \['tail'\]"),
    ],
    ids=["dict", "joint-state", "full-dict", "lerobot-key", "unknown-key"],
)
def test_a_mapping_in_policy_units_converts_per_key_and_unnamed_joints_hold(returned, refusal):
    from strands_robots.robot.run.bind import bind, decide

    arm = Robot("so101", sim=Simulation())
    u = Units.for_robot(arm.description, "lerobot")
    obs = arm.observe()
    bound = bind(arm, lambda seen: returned(seen), "lerobot")
    if refusal is not None:
        with pytest.raises(KeyError, match=refusal):
            decide(arm, bound, obs, None)
        return
    keys = list(arm.description.action_keys)
    want = u.from_rad(obs.joints.positions)
    for k, v in _PAN_ELBOW.items():
        want[keys.index(k)] = v
    np.testing.assert_allclose(
        decide(arm, bound, obs, None), arm.coerce(u.to_rad(want), measured=obs.joints.positions), atol=1e-12
    )
    held = [i for i, k in enumerate(keys) if k not in _PAN_ELBOW]  # an unnamed joint holds q exactly
    np.testing.assert_allclose(decide(arm, bound, obs, None)[held], obs.joints.positions[held], atol=1e-12)


def test_run_with_units_takes_a_mapping_policy():
    arm = Robot("so101", sim=Simulation())
    arm.run(lambda obs: dict(_PAN_ELBOW), hz=20, steps=3, units="lerobot")
    pan = Units.for_robot(arm.description, "lerobot").from_rad(arm.observe().joints.positions)[0]
    assert pan > 0  # moved toward +60 in LeRobot units
