"""Regression tests for the ``base_beyond_y`` floating-base lateral-progress predicate.

The predicate/reward DSL grew the FORWARD-progress success predicate
(``base_beyond_x``) alongside the velocity-tracking reward
(``base_velocity_tracking``, which already accepts a lateral ``vy`` command) and
the fall predicates (``base_tipped`` / ``base_below_z``). The LATERAL success
half - "the base actually strafed sideways" - was still inexpressible:
``base_beyond_x`` reads only ``base_pos`` x, so a strafe benchmark could reward
a ``vy`` command but had no way to SCORE reaching a lateral goal, and
``inside_region`` / ``body_above_z`` need a base body name a mobile base's
unnamed free joint does not expose.

``base_beyond_y(y, robot)`` closes that gap: TRUE when the base's world y has
passed beyond ``y`` (world +y is the robot's left for the identity spawn
orientation), reading the same ``base_pos`` signal ``base_beyond_x`` reads, so
it drops straight into a ``success`` clause next to the fall predicates in
``failure``. These tests set a KNOWN base pose directly on the sim and assert
the threshold, height/orientation independence (it is a pure y-position
predicate), that x-position does NOT trip it (the axis is distinct from
``base_beyond_x``), live tracking, fixed-base degradation, and that a real
``DeclarativeBenchmark`` whose success is ``base_beyond_y`` and failure is
``base_tipped`` + ``base_below_z`` succeeds only once the base strafes past the
line. They are GL-free (``get_observation`` with ``skip_images``) so they run in
CI without a display.
"""

import logging
import math
import os
import tempfile

import mujoco
import pytest

from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.predicates import (
    PREDICATE_REGISTRY,
    _reset_resolution_warnings,
    make_predicate,
    predicate_kind,
)

# Floating base with a NAMED free joint (a humanoid's floating_base_joint) plus
# one actuated hinge. get_observation surfaces base_pos for this robot.
NAMED_BASE_XML = """
<mujoco model="test_named_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.3 0.3 0.8 1"/>
      <body name="thigh" pos="0 0 -0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.3" rgba="0.8 0.3 0.3 1"/>
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_act" joint="hip"/>
  </actuator>
</mujoco>
"""

# Fixed-base arm: no free joint anywhere -> no base position. base_beyond_y must
# degrade to False (and warn) rather than crash or invent a value.
FIXED_ARM_XML = """
<mujoco model="test_fixed">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link" pos="0 0 0.1">
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
      <joint name="j0" type="hinge" axis="0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="j0_act" joint="j0" kp="10"/>
  </actuator>
</mujoco>
"""


def _axis_quat(axis: str, deg: float) -> list[float]:
    """Unit (w, x, y, z) quaternion for a rotation of ``deg`` about a world axis."""
    h = math.radians(deg) / 2.0
    c, sn = math.cos(h), math.sin(h)
    return {
        "x": [c, sn, 0.0, 0.0],
        "y": [c, 0.0, sn, 0.0],
        "z": [c, 0.0, 0.0, sn],
    }[axis]


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_base_beyond_y", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _set_base_pose(sim, y: float, x: float = 0.0, z: float = 0.8, quat_wxyz: list[float] | None = None) -> None:
    """Set the robot's (only) free joint to world (x, y, z) with orientation quat."""
    model, data = sim._world._model, sim._world._data
    jid = -1
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            jid = j
            break
    assert jid >= 0
    qadr = int(model.jnt_qposadr[jid])
    q = quat_wxyz if quat_wxyz is not None else [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr : qadr + 7] = [x, y, z, *q]
    mujoco.mj_forward(model, data)


def test_base_beyond_y_is_registered_as_a_bool_predicate():
    """It must classify as bool so the DSL accepts it in success/failure clauses."""
    assert "base_beyond_y" in PREDICATE_REGISTRY
    assert predicate_kind("base_beyond_y") == "bool"


def test_base_beyond_y_trips_once_the_base_passes_the_threshold(sim):
    """FALSE while the base y is at/behind the threshold, TRUE once it passes it."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_beyond_y", y=1.0)
    _set_base_pose(sim, 0.0)
    assert pred(sim) is False
    _set_base_pose(sim, 1.0)  # exactly at the threshold is not "beyond"
    assert pred(sim) is False
    _set_base_pose(sim, 1.01)
    assert pred(sim) is True
    _set_base_pose(sim, 2.5)  # strafed well left
    assert pred(sim) is True


def test_base_beyond_y_reads_the_y_axis_not_the_x_axis(sim):
    """It is distinct from base_beyond_x: forward x-progress must NOT satisfy a
    lateral y-goal, and lateral y-progress must NOT satisfy a forward x-read."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    y_pred = make_predicate("base_beyond_y", y=1.0)
    x_pred = make_predicate("base_beyond_x", x=1.0)
    # Walked far forward (x=3) but zero lateral: y-goal not met, x-goal met.
    _set_base_pose(sim, 0.0, x=3.0)
    assert y_pred(sim) is False
    assert x_pred(sim) is True
    # Strafed far left (y=3) but zero forward: y-goal met, x-goal not met.
    _set_base_pose(sim, 3.0, x=0.0)
    assert y_pred(sim) is True
    assert x_pred(sim) is False


def test_base_beyond_y_is_independent_of_height_and_orientation(sim):
    """It reads only lateral y-position: the same y at any height / orientation
    reads identically (a base that strafed but toppled or dropped still counts
    as having reached the line - the fall predicates reject that)."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_beyond_y", y=1.0)
    for quat in ([1.0, 0.0, 0.0, 0.0], _axis_quat("y", 90.0), _axis_quat("z", 120.0)):
        for z in (0.8, 0.1):
            _set_base_pose(sim, 0.0, z=z, quat_wxyz=quat)
            assert pred(sim) is False, "behind the line -> not beyond"
            _set_base_pose(sim, 2.0, z=z, quat_wxyz=quat)
            assert pred(sim) is True, "past the line -> beyond (any height/orientation)"


def test_base_beyond_y_tracks_the_live_base_position(sim):
    """The predicate reads the CURRENT base y: strafing the base left flips it."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_beyond_y", y=0.5)
    _set_base_pose(sim, 0.0)
    assert pred(sim) is False
    _set_base_pose(sim, 0.9)
    assert pred(sim) is True


def test_base_beyond_y_accepts_a_negative_threshold(sim):
    """y is an unvalidated world threshold (mirrors base_beyond_x): a negative y
    is a right-of-origin line a base spawned at the origin already reads True on."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_beyond_y", y=-0.5)
    _set_base_pose(sim, 0.0)
    assert pred(sim) is True
    _set_base_pose(sim, -1.0)
    assert pred(sim) is False


def test_base_beyond_y_degrades_to_false_on_fixed_base_arm(sim, caplog):
    """A fixed-base arm has no base position: the predicate degrades to False
    (never made lateral progress -> never spuriously succeeds) and warns once."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    _reset_resolution_warnings()
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.predicates"):
        val = make_predicate("base_beyond_y", y=1.0)(sim)
    assert val is False
    assert any("base" in r.message.lower() for r in caplog.records)


def test_declarative_strafe_benchmark_succeeds_only_past_the_line(sim):
    """End to end: a DeclarativeBenchmark whose success is base_beyond_y and whose
    failure is base_tipped + base_below_z - the lateral velocity-tracking task
    vocabulary (tracking reward with a vy command shapes HOW to strafe, fall
    predicates end a bad rollout, base_beyond_y scores the GOAL) - compiles and
    reports success only once the base strafes past the line, and never succeeds
    while it is still behind it (even if standing perfectly)."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    bench = DeclarativeBenchmark.from_dict(
        {
            "name": "strafe-left",
            "default_robot": "humanoid",
            "max_steps": 1000,
            "dense_reward": [
                {"predicate": "base_velocity_tracking", "vy": 0.5, "lin_weight": 1.0},
            ],
            "success": {"all": [{"predicate": "base_beyond_y", "y": 1.0}]},
            "failure": {
                "any": [
                    {"predicate": "base_tipped", "tol": 0.7},
                    {"predicate": "base_below_z", "z": 0.3},
                ]
            },
        }
    )
    # Standing upright at the origin: not fallen, but has NOT strafed.
    _set_base_pose(sim, 0.0, z=0.8, quat_wxyz=[1.0, 0.0, 0.0, 0.0])
    assert bench.is_failure(sim) is False
    assert bench.is_success(sim) is False, "standing still must not score the strafe goal"
    # Strafed past the line, still upright: the goal is reached.
    _set_base_pose(sim, 2.5, z=0.8, quat_wxyz=[1.0, 0.0, 0.0, 0.0])
    assert bench.is_failure(sim) is False
    assert bench.is_success(sim) is True
    # Reached the line but toppled on the way: success on y, but failure fires
    # too, so the fall predicates correctly veto a "strafed then fell" run.
    _set_base_pose(sim, 2.5, z=0.8, quat_wxyz=_axis_quat("y", 90.0))
    assert bench.is_success(sim) is True
    assert bench.is_failure(sim) is True
