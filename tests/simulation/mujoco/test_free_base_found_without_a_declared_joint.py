"""Regression tests: a floating base is found even when the robot declares no joint.

A floating base that is not a named entry in ``robot.joint_names`` is recovered
by walking up the kinematic tree from a seed part of the machine. That walk used
to accept only one kind of seed -- a body carrying one of the robot's *declared
joints* -- which made the fallback unreachable for the shape it exists to serve:
a robot whose actuation is not a joint at all.

An aerial robot is that shape, and it is the stock one. Each rotor is a force
applied at a site on the airframe (``mjTRN_SITE``), so the model declares no
joint besides the unnamed ``<freejoint>`` on the airframe itself and
``robot.joint_names`` is empty. With nothing to seed from, the walk returned
"no base" for a robot that is entirely a floating base, and the surfaces derived
from it went quiet rather than wrong:

* ``get_observation`` returned no state at all -- no ``base_pos``, no
  ``base_quat``, no velocities -- for a robot that is in the scene and moving.
* ``start_recording`` declared a dataset with no ``observation.state`` column
  while still recording the ``action`` columns, so an episode captured a
  demonstration whose observations are images only and reported success.

The walk now also seeds from the bodies the robot's own actuators act on, which
is a strict superset of the previous seed and keeps the "only the robot's OWN
base" guarantee for the same reason the joint seed did: a sibling free-jointed
task object carries neither a declared joint of the robot nor any actuator of
it. That case is not hypothetical -- every Menagerie grasping scene ships a
free-jointed ``object`` body inside the gripper's own MJCF, so it arrives under
the robot's namespace, and a fixed-base gripper must still report no base.

The seed body is additionally required to carry the robot's namespace, because
one of the two actuator-ownership rules matches by the joint id an actuator
drives and a stored id does not survive a recompile. That requirement is pinned
by ``test_base_state_disappears_when_the_bare_name_misses_too`` in
``test_namespace_fallback_survives_scene_replace``, whose fixture replaces the
scene with a different machine; it needs a robot that declares a joint for the
id to collide with, which is the one shape this module deliberately does not
build.
"""

import tempfile

import numpy as np
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation

# Aerial robot: an UNNAMED <freejoint> on the airframe, NO joints at all, and
# four rotor actuators applying force at sites on that airframe (mjTRN_SITE).
# Mirrors the stock Skydio X2 / Crazyflie models.
THRUST_ONLY_XML = """
<mujoco model="test_thrust_only">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="airframe" pos="0.25 -0.5 0.75">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.02" mass="0.5" rgba="0.3 0.3 0.8 1"/>
      <site name="rotor1" pos="0.1 0.1 0.02"/>
      <site name="rotor2" pos="-0.1 0.1 0.02"/>
      <site name="rotor3" pos="-0.1 -0.1 0.02"/>
      <site name="rotor4" pos="0.1 -0.1 0.02"/>
    </body>
  </worldbody>
  <actuator>
    <general name="thrust1" site="rotor1" gear="0 0 1 0 0 0" ctrlrange="0 6"/>
    <general name="thrust2" site="rotor2" gear="0 0 1 0 0 0" ctrlrange="0 6"/>
    <general name="thrust3" site="rotor3" gear="0 0 1 0 0 0" ctrlrange="0 6"/>
    <general name="thrust4" site="rotor4" gear="0 0 1 0 0 0" ctrlrange="0 6"/>
  </actuator>
</mujoco>
"""

# Same aerial robot, plus a SIBLING free-jointed task object in the same MJCF.
# Two free joints are now in the robot's namespace and only one is its base, so
# the base must be resolved rather than guessed from namespace membership.
THRUST_WITH_TASK_OBJECT_XML = THRUST_ONLY_XML.replace(
    "  </worldbody>",
    """    <body name="object" pos="2 2 0.05">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="0.1" rgba="0.2 0.8 0.2 1"/>
    </body>
  </worldbody>""",
)

# Fixed-base tendon-driven gripper shipped with a free-jointed ``object`` to
# grasp -- the stock Robotiq 2F-85 / Allegro / Shadow Hand scene shape. The
# object is the ONLY free joint and no actuator acts on it, so this robot has no
# floating base and must gain no base state.
FIXED_GRIPPER_WITH_TASK_OBJECT_XML = """
<mujoco model="test_gripper_with_object">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="palm" pos="0 0 0.2">
      <geom type="box" size="0.05 0.05 0.02" rgba="0.4 0.4 0.4 1"/>
      <body name="finger_left" pos="0.04 0 0.02">
        <joint name="left_drive" type="hinge" axis="0 1 0" range="0 0.8"/>
        <geom type="capsule" size="0.008" fromto="0 0 0 0 0 0.06"/>
      </body>
      <body name="finger_right" pos="-0.04 0 0.02">
        <joint name="right_drive" type="hinge" axis="0 1 0" range="0 0.8"/>
        <geom type="capsule" size="0.008" fromto="0 0 0 0 0 0.06"/>
      </body>
    </body>
    <body name="object" pos="0 0 0.05">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.05" rgba="0.2 0.8 0.2 1"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="grip">
      <joint joint="left_drive" coef="0.5"/>
      <joint joint="right_drive" coef="0.5"/>
    </fixed>
  </tendon>
  <actuator>
    <position name="grip_act" tendon="grip" kp="20" ctrlrange="0 0.8"/>
  </actuator>
</mujoco>
"""

# Fixed-base arm: one hinge, no free joint anywhere.
FIXED_ARM_XML = """
<mujoco model="test_fixed_arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link" pos="0 0 0.1">
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
    </body>
  </worldbody>
  <actuator>
    <position name="j0_act" joint="j0" kp="10"/>
  </actuator>
</mujoco>
"""

# Humanoid-style NAMED free root joint -- resolved by the named-joint rung, which
# must keep answering first and unchanged.
NAMED_BASE_XML = """
<mujoco model="test_named_base">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="torso" pos="0 0 0.5">
      <joint name="floating_base_joint" type="free"/>
      <geom type="box" size="0.1 0.1 0.2"/>
      <body name="thigh" pos="0 0 -0.2">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_act" joint="hip"/>
  </actuator>
</mujoco>
"""

# Mobile base: UNNAMED freejoint reached by walking up from a declared,
# actuated hinge -- the seed that already worked, kept as a control.
UNNAMED_BASE_XML = """
<mujoco model="test_unnamed_base">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="base_plate" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.15 0.15 0.03"/>
      <body name="arm" pos="0 0 0.05">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_act" joint="shoulder"/>
  </actuator>
</mujoco>
"""

_BASE_KEYS = ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel")
_BASE_COLS = [f"{src}.{c}" for src, c in [(k, c) for k in ("base_pos",) for c in "xyz"]] + [
    f"base_quat.{c}" for c in ("w", "x", "y", "z")
]
_BASE_COLS += [f"base_lin_vel.{c}" for c in "xyz"] + [f"base_ang_vel.{c}" for c in "xyz"]


def _write(xml: str) -> str:
    import os

    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    return path


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_free_base_seed", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    try:
        s.cleanup()
    except Exception:
        # Best-effort teardown: cleanup failures must not mask the test result.
        pass


def _add(sim, name, xml):
    res = sim.add_robot(name, urdf_path=_write(xml))
    assert res["status"] == "success", res["content"][0]["text"]
    return res


def _state_names(sim):
    rec = sim._world._backend_state["dataset_recorder"]
    feat = rec.dataset.features.get("observation.state", {})
    names = feat.get("names", []) if isinstance(feat, dict) else getattr(feat, "names", [])
    return list(names)


def _start_recording(sim, name):
    root = tempfile.mkdtemp(prefix=f"freebase_{name}_")
    res = sim.start_recording(repo_id=f"local/{name}_freebase", task="t", fps=20, root=root, cameras=[])
    assert res["status"] == "success", res["content"][0]["text"]
    return root


def test_thrust_actuated_robot_reports_its_floating_base(sim):
    """A robot with no declared joints still surfaces its floating base.

    ``robot.joint_names`` is empty, so the base can only be reached from the
    rotor actuators' target sites. Pre-fix this observation was empty.
    """
    _add(sim, "drone", THRUST_ONLY_XML)
    obs = sim.get_observation(robot_name="drone", skip_images=True)
    missing = [k for k in _BASE_KEYS if k not in obs]
    assert not missing, f"floating base not surfaced: {missing} absent from {sorted(obs)}"
    # The airframe spawns at a distinctive pose, so a wrong body cannot pass.
    assert obs["base_pos"] == pytest.approx([0.25, -0.5, 0.75], abs=1e-6)
    assert obs["base_quat"] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert len(obs["base_lin_vel"]) == 3
    assert len(obs["base_ang_vel"]) == 3


def test_thrust_actuated_robot_records_an_observation_state_column(sim):
    """The recorded dataset schema carries the base columns rather than no state.

    Pre-fix the schema had no ``observation.state`` feature at all while the
    ``action`` columns were still declared, so the episode recorded a
    demonstration with images and actions and nothing to condition on.
    """
    pytest.importorskip("lerobot")  # start_recording produces a LeRobotDataset
    _add(sim, "drone", THRUST_ONLY_XML)
    _start_recording(sim, "drone")
    names = _state_names(sim)
    assert names, "recorded dataset declared no observation.state column at all"
    for col in _BASE_COLS:
        assert col in names, f"{col} missing from recorded observation.state schema: {names}"
    sim.stop_recording()


def test_recorded_base_values_are_the_airframes(sim):
    """End-to-end: the recorded state is the airframe's pose, read back after reopen."""
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _add(sim, "drone", THRUST_ONLY_XML)
    root = _start_recording(sim, "drone")
    res = sim.run_policy(robot_name="drone", policy_provider="mock", n_steps=4, control_frequency=20.0)
    assert res["status"] == "success", res["content"][0]["text"]
    stopped = sim.stop_recording()
    assert stopped["status"] == "success", stopped["content"][0]["text"]

    ds = LeRobotDataset("local/drone_freebase", root=root)
    assert ds.num_frames > 0
    state = np.asarray(ds[0]["observation.state"], dtype=float)
    assert state.shape == (len(_BASE_COLS),)
    # First frame is the spawn pose: position then identity orientation.
    assert state[:3] == pytest.approx([0.25, -0.5, 0.75], abs=1e-3)
    assert state[3:7] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-3)


def test_a_task_object_in_the_robots_namespace_is_not_taken_for_its_base(sim):
    """With two free joints under one namespace, the airframe is the base.

    ``spec.attach`` prefixes every name it merges, so a task object shipped in
    the robot's own MJCF arrives under the robot's namespace. Resolving the base
    by namespace membership would report the cube's pose as the robot's.
    """
    _add(sim, "drone", THRUST_WITH_TASK_OBJECT_XML)
    obs = sim.get_observation(robot_name="drone", skip_images=True)
    assert "base_pos" in obs, "floating base not surfaced with a task object present"
    assert obs["base_pos"] == pytest.approx([0.25, -0.5, 0.75], abs=1e-6), (
        "base_pos is the task object's pose, not the airframe's"
    )


def test_a_fixed_base_gripper_shipped_with_a_task_object_has_no_base(sim):
    """A fixed-base robot whose MJCF ships a free-jointed object gains no base state.

    The object is the only free joint in the robot's namespace and no actuator
    of the robot acts on it, so there is no base to report. This is what a
    namespace-membership rule would get wrong.
    """
    _add(sim, "gripper", FIXED_GRIPPER_WITH_TASK_OBJECT_XML)
    obs = sim.get_observation(robot_name="gripper", skip_images=True)
    claimed = [k for k in _BASE_KEYS if k in obs]
    assert not claimed, f"fixed-base gripper reported base state from a task object: {claimed}"


def test_a_fixed_base_arm_still_has_no_base(sim):
    """No free joint anywhere -> no base state, from either seed."""
    _add(sim, "arm", FIXED_ARM_XML)
    obs = sim.get_observation(robot_name="arm", skip_images=True)
    assert not [k for k in _BASE_KEYS if k in obs]


@pytest.mark.parametrize(
    ("name", "xml", "expected_pos"),
    [
        ("humanoid", NAMED_BASE_XML, [0.0, 0.0, 0.5]),
        ("mobile", UNNAMED_BASE_XML, [0.0, 0.0, 0.1]),
    ],
)
def test_the_seeds_that_already_worked_answer_unchanged(sim, name, xml, expected_pos):
    """A named free root joint and a declared-joint-seeded unnamed base are unchanged."""
    _add(sim, name, xml)
    obs = sim.get_observation(robot_name=name, skip_images=True)
    for key in _BASE_KEYS:
        assert key in obs, f"{key} missing for {name}"
    assert obs["base_pos"] == pytest.approx(expected_pos, abs=1e-6)


class TestActuatorTargetBodyIds:
    """The transmission read the base seed is built from."""

    def _model(self, xml):
        mj = pytest.importorskip("mujoco")
        return mj, mj.MjModel.from_xml_string(xml)

    def test_a_site_transmission_resolves_the_body_carrying_the_site(self):
        from strands_robots.simulation.mujoco.scene_ops import actuator_target_body_ids

        mj, model = self._model(THRUST_ONLY_XML)
        airframe = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "airframe")
        for act_id in range(model.nu):
            assert actuator_target_body_ids(model, act_id, mj) == frozenset({airframe})

    def test_a_joint_transmission_resolves_the_joints_body(self):
        from strands_robots.simulation.mujoco.scene_ops import actuator_target_body_ids

        mj, model = self._model(FIXED_ARM_XML)
        link = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link")
        assert actuator_target_body_ids(model, 0, mj) == frozenset({link})

    def test_a_tendon_transmission_resolves_the_wrapped_joints_bodies(self):
        from strands_robots.simulation.mujoco.scene_ops import actuator_target_body_ids

        mj, model = self._model(FIXED_GRIPPER_WITH_TASK_OBJECT_XML)
        expected = frozenset(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in ("finger_left", "finger_right"))
        assert actuator_target_body_ids(model, 0, mj) == expected

    def test_a_body_transmission_resolves_that_body(self):
        from strands_robots.simulation.mujoco.scene_ops import actuator_target_body_ids

        xml = """
        <mujoco model="test_adhesion">
          <compiler angle="radian" autolimits="true"/>
          <worldbody>
            <body name="pad" pos="0 0 0.1">
              <freejoint/>
              <geom type="box" size="0.05 0.05 0.01" mass="0.2"/>
            </body>
          </worldbody>
          <actuator>
            <adhesion name="suck" body="pad" ctrlrange="0 1" gain="2"/>
          </actuator>
        </mujoco>
        """
        mj, model = self._model(xml)
        pad = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pad")
        assert int(model.actuator_trntype[0]) == int(mj.mjtTrn.mjTRN_BODY)
        assert actuator_target_body_ids(model, 0, mj) == frozenset({pad})

    def test_the_joint_helpers_are_empty_for_a_site_transmission(self):
        """Why this is a third function: neither joint reader answers here."""
        from strands_robots.simulation.mujoco.scene_ops import (
            actuator_driven_joint_ids,
            actuator_joint_id,
        )

        mj, model = self._model(THRUST_ONLY_XML)
        assert actuator_joint_id(model, 0, mj) == -1
        assert actuator_driven_joint_ids(model, 0, mj) == frozenset()
