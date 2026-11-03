import numpy as np
import pytest

from strands_robots import JointState, Observation, Pose
from strands_robots.robot.drivers import Calibration
from strands_robots.robot.drivers.calibration import JointCalibration
from strands_robots.robot.run import RunResult


def test_joint_state_is_frozen_and_indexable():
    js = JointState(("a", "b"), [1.0, 2.0], [0, 0], [0, 0])
    assert js["b"] == 2.0 and len(js) == 2
    with pytest.raises(ValueError):
        js.positions[0] = 9.0
    with pytest.raises(KeyError, match="unknown joint 'z'"):
        js["z"]


def test_joint_state_shape_checked():
    with pytest.raises(ValueError, match="positions"):
        JointState(("a",), [1.0, 2.0], [0], [0])


def test_pose_shapes():
    p = Pose.identity((1, 2, 3))
    assert p.position.tolist() == [1, 2, 3] and p.orientation[0] == 1.0
    with pytest.raises(ValueError, match="orientation"):
        Pose([0, 0, 0], [1, 0, 0])


def test_observation_to_lerobot():
    js = JointState(("a",), [0.5], [0], [0])
    img = np.zeros((4, 4, 3), np.uint8)
    o = Observation(1.0, js, images={"front": img})
    lr = o.to_lerobot()
    assert lr["observation.state"].dtype == np.float32
    assert lr["observation.images.front"] is img
    assert o["a"] == 0.5


def test_calibration_roundtrips_lerobot_json():
    data = {"shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -2047, "range_min": 758, "range_max": 3292}}
    c = Calibration.from_json(data)
    assert c.joints["shoulder_pan"] == JointCalibration(1, 0, -2047, 758, 3292)
    assert c.to_json() == data


def _js(*q):
    return JointState(
        tuple(f"j{i}" for i in range(len(q))), np.array(q, dtype=float), np.zeros(len(q)), np.zeros(len(q))
    )


def _obs(*q, img=0):
    return Observation(0.1, _js(*q), Pose.identity(), {"front": np.full((2, 2, 3), img, np.uint8)}, {"k": np.ones(2)})


def _run(final):
    return RunResult(3, 0.1, 30.0, 29.0, "steps", final=final)


@pytest.mark.parametrize(
    "make, other",
    [
        (lambda: _js(0.1, 0.2, 0.3), lambda: _js(0.1, 0.2, 0.4)),
        (lambda: _js(0.0, 0.0), lambda: _js(0.0, 1.0)),
        (lambda: Pose.identity((1, 2, 3)), lambda: Pose.identity((1, 2, 4))),
        (lambda: _obs(0.1, 0.2), lambda: _obs(0.1, 0.2, img=9)),
        (lambda: _run(_obs(0.1, 0.2)), lambda: _run(_obs(0.1, 0.3))),
    ],
)
def test_every_public_type_is_a_value(make, other):
    # == compares the arrays (any dof count), hash agrees with it, so the types work in sets and dict keys
    a, b, c = make(), make(), other()
    assert a == b and not (a != b) and hash(a) == hash(b)
    assert a != c and a != "not a value"
    assert len({a, b, c}) == 2


def test_hash_agrees_with_equality_on_negative_zero():
    assert _js(-0.0, 1.0) == _js(0.0, 1.0) and hash(_js(-0.0, 1.0)) == hash(_js(0.0, 1.0))
