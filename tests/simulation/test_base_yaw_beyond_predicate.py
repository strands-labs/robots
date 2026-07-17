"""Regression tests for the ``base_yaw_beyond`` floating-base heading predicate.

The predicate/reward DSL grew FORWARD- and LATERAL-progress success predicates
(``base_beyond_x`` / ``base_beyond_y``) alongside the velocity-tracking reward
(``base_velocity_tracking``, which already accepts a yaw-rate ``wz`` command) and
the fall predicates (``base_tipped`` / ``base_below_z``). The YAW success half -
"the base actually turned" - was still inexpressible: ``base_beyond_x`` /
``base_beyond_y`` read only ``base_pos``, so a turn-in-place benchmark could
reward a ``wz`` command but had no way to SCORE reaching a heading goal, and
``base_tipped`` fires on ANY tilt (roll/pitch), not a deliberate turn about the
vertical.

``base_yaw_beyond(yaw, robot)`` closes that gap: TRUE when the base's world yaw
heading (extracted from ``base_quat``) has passed ``yaw`` radians (positive is a
left / counter-clockwise turn from the identity spawn). These tests set a KNOWN
base pose directly on the sim and assert the threshold, that it reads the YAW
axis (a pure roll/pitch does NOT trip it, distinguishing it from ``base_tipped``),
position/height independence, that x-position does NOT trip it (distinct from
``base_beyond_x``), live tracking, fixed-base degradation, and that a real
``DeclarativeBenchmark`` whose success is ``base_yaw_beyond`` and failure is
``base_tipped`` + ``base_below_z`` succeeds once the base turns and is vetoed if
it falls. They are GL-free (``get_observation`` with ``skip_images``) so they run
in CI without a display.
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
# one actuated hinge. get_observation surfaces base_quat for this robot.
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

# Fixed-base arm: no free joint anywhere -> no base orientation. base_yaw_beyond
# must degrade to False (and warn) rather than crash or invent a heading.
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
    s = Simulation(tool_name="test_base_yaw_beyond", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _set_base_pose(sim, quat_wxyz: list[float] | None = None, x: float = 0.0, y: float = 0.0, z: float = 0.8) -> None:
    """Set the robot's (only) free joint to world (x, y, z) with orientation quat."""
    model, data = sim._world._model, sim._world._data
    jid = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    qadr = int(model.jnt_qposadr[jid])
    q = quat_wxyz if quat_wxyz is not None else [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr : qadr + 7] = [x, y, z, *q]
    mujoco.mj_forward(model, data)


def test_base_yaw_beyond_is_registered_as_a_bool_predicate():
    """It must classify as bool so the DSL accepts it in success/failure clauses."""
    assert "base_yaw_beyond" in PREDICATE_REGISTRY
    assert predicate_kind("base_yaw_beyond") == "bool"


def test_base_yaw_beyond_trips_once_the_base_turns_past_the_threshold(sim):
    """FALSE while the heading is at/behind the threshold, TRUE once it passes it."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_yaw_beyond", yaw=1.0)  # ~57 deg left
    _set_base_pose(sim, _axis_quat("z", 0.0))
    assert pred(sim) is False
    _set_base_pose(sim, _axis_quat("z", 50.0))  # ~0.87 rad, short of the 1.0 rad line
    assert pred(sim) is False
    _set_base_pose(sim, _axis_quat("z", 60.0))  # ~1.05 rad, past the line
    assert pred(sim) is True
    _set_base_pose(sim, _axis_quat("z", 120.0))  # turned well left
    assert pred(sim) is True


def test_base_yaw_beyond_reads_yaw_not_a_roll_or_pitch_tilt(sim):
    """It is a heading test, NOT a tilt test: a base that merely rolls or pitches
    (about a horizontal axis) has an unchanged yaw and must NOT satisfy a turn
    goal - the property that distinguishes it from base_tipped."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_yaw_beyond", yaw=0.1)  # a tiny turn goal
    for tilt in (_axis_quat("x", 80.0), _axis_quat("y", 80.0)):  # large roll / pitch
        _set_base_pose(sim, tilt)
        assert pred(sim) is False, "a pure roll/pitch tilt is not a yaw turn"
    # but a genuine turn about the vertical does satisfy it
    _set_base_pose(sim, _axis_quat("z", 30.0))
    assert pred(sim) is True


def test_base_yaw_beyond_reads_heading_not_position(sim):
    """It is distinct from base_beyond_x/y: linear displacement (with no turn)
    must NOT satisfy a heading goal, and a turn (at the origin) must NOT satisfy
    a forward-position read."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    yaw_pred = make_predicate("base_yaw_beyond", yaw=0.5)
    x_pred = make_predicate("base_beyond_x", x=1.0)
    # Walked far forward + left but never turned: heading-goal not met, x-goal met.
    _set_base_pose(sim, _axis_quat("z", 0.0), x=3.0, y=3.0)
    assert yaw_pred(sim) is False
    assert x_pred(sim) is True
    # Turned in place at the origin: heading-goal met, x-goal not met.
    _set_base_pose(sim, _axis_quat("z", 45.0), x=0.0, y=0.0)
    assert yaw_pred(sim) is True
    assert x_pred(sim) is False


def test_base_yaw_beyond_is_independent_of_position_and_height(sim):
    """It reads only the yaw heading: the same heading at any world x/y/z reads
    identically (a base that turned but drifted or dropped still counts as having
    reached the heading - the fall predicates reject a dropped base)."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_yaw_beyond", yaw=1.0)
    for x, y, z in ((0.0, 0.0, 0.8), (2.5, -1.5, 0.1)):
        _set_base_pose(sim, _axis_quat("z", 0.0), x=x, y=y, z=z)
        assert pred(sim) is False, "not turned -> not beyond (any position/height)"
        _set_base_pose(sim, _axis_quat("z", 90.0), x=x, y=y, z=z)
        assert pred(sim) is True, "turned 90 deg -> beyond (any position/height)"


def test_base_yaw_beyond_tracks_the_live_base_heading(sim):
    """The predicate reads the CURRENT base heading: turning the base flips it."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_yaw_beyond", yaw=0.5)
    _set_base_pose(sim, _axis_quat("z", 0.0))
    assert pred(sim) is False
    _set_base_pose(sim, _axis_quat("z", 45.0))
    assert pred(sim) is True


def test_base_yaw_beyond_accepts_a_negative_threshold(sim):
    """yaw is an unvalidated world heading (mirrors base_beyond_x/y): a negative
    threshold is a right-of-spawn heading a base at the identity spawn already
    reads True on."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_yaw_beyond", yaw=-0.5)
    _set_base_pose(sim, _axis_quat("z", 0.0))
    assert pred(sim) is True
    _set_base_pose(sim, _axis_quat("z", -45.0))  # turned right, below -0.5 rad
    assert pred(sim) is False


def test_base_yaw_beyond_wraps_at_pi_so_a_turn_past_pi_reads_below_the_goal_again(sim):
    """The heading is atan2-wrapped to (-pi, pi], so it is single-valued only for
    a sub-pi turn - the documented reason a turn goal must stay below pi. A turn
    of just under half a revolution (170 deg -> +2.97 rad) satisfies a yaw=1.0
    goal, a turn of exactly 180 deg lands on the +pi wrap edge and still reads
    True, but turning FURTHER (190 deg) wraps the heading to -2.97 rad and the
    same goal reads False again: past pi the predicate is NOT monotonic in the
    physical turn angle. This pins that documented discontinuity so a refactor to
    a cumulative / unwrapped heading (which would keep reading True past pi) is a
    visible contract change, not a silent one."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    pred = make_predicate("base_yaw_beyond", yaw=1.0)
    # Just under a half-revolution left: heading ~+2.97 rad, comfortably past 1.0.
    _set_base_pose(sim, _axis_quat("z", 170.0))
    assert pred(sim) is True
    # Exactly half a revolution: atan2(0, -1) = +pi, the top of the (-pi, pi] range.
    _set_base_pose(sim, _axis_quat("z", 180.0))
    assert pred(sim) is True
    # Past pi: the heading WRAPS to -2.97 rad, so the yaw=1.0 goal reads False
    # again even though the base turned FURTHER left - the wrap discontinuity.
    _set_base_pose(sim, _axis_quat("z", 190.0))
    assert pred(sim) is False
    _set_base_pose(sim, _axis_quat("z", 200.0))
    assert pred(sim) is False


def test_base_yaw_beyond_degrades_to_false_on_fixed_base_arm(sim, caplog):
    """A fixed-base arm has no base orientation: the predicate degrades to False
    (never turned -> never spuriously succeeds) and warns once."""
    sim.add_robot("arm", urdf_path=_write(FIXED_ARM_XML))
    _reset_resolution_warnings()
    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.predicates"):
        val = make_predicate("base_yaw_beyond", yaw=1.0)(sim)
    assert val is False
    assert any("base" in r.message.lower() for r in caplog.records)


def test_declarative_turn_benchmark_succeeds_on_turn_and_is_vetoed_by_a_fall(sim):
    """End to end: a DeclarativeBenchmark whose success is base_yaw_beyond and
    whose failure is base_tipped + base_below_z - the yaw velocity-tracking task
    vocabulary (tracking reward with a wz command shapes HOW to turn, fall
    predicates end a bad rollout, base_yaw_beyond scores the GOAL) - compiles and
    reports success only once the base turns past the heading, never while it is
    standing un-turned, and a topple vetoes the run via the failure clause."""
    sim.add_robot("humanoid", urdf_path=_write(NAMED_BASE_XML))
    bench = DeclarativeBenchmark.from_dict(
        {
            "name": "turn-left",
            "default_robot": "humanoid",
            "max_steps": 1000,
            "dense_reward": [
                {"predicate": "base_velocity_tracking", "wz": 0.5, "lin_weight": 1.0},
            ],
            "success": {"all": [{"predicate": "base_yaw_beyond", "yaw": 1.0}]},
            "failure": {
                "any": [
                    {"predicate": "base_tipped", "tol": 0.7},
                    {"predicate": "base_below_z", "z": 0.3},
                ]
            },
        }
    )
    # Standing upright, not turned: not fallen, but has NOT reached the heading.
    _set_base_pose(sim, _axis_quat("z", 0.0), z=0.8)
    assert bench.is_failure(sim) is False
    assert bench.is_success(sim) is False, "standing un-turned must not score the turn goal"
    # Turned past the line, still upright: the goal is reached.
    _set_base_pose(sim, _axis_quat("z", 90.0), z=0.8)
    assert bench.is_failure(sim) is False
    assert bench.is_success(sim) is True
    # Toppled (pitched onto its side): the fall predicate fires and vetoes the
    # rollout (a toppled base's yaw is ill-defined, so the tilt is the terminal).
    _set_base_pose(sim, _axis_quat("y", 90.0), z=0.8)
    assert bench.is_failure(sim) is True
