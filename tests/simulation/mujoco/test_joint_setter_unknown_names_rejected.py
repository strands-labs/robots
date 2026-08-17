"""Regression tests: the joint setters must reject joint names they cannot write.

``set_joint_positions`` / ``set_joint_velocities`` accept a
``{joint_name: value}`` mapping. The dict form used to resolve each name inside
the write loop and simply ``continue`` past the ones MuJoCo did not know,
answering ``status="success"`` afterwards::

    sim.set_joint_positions({"shoulder_pan": 0.4})   # scene names it "shoulder"
    # -> success: "Set 0/1 joint positions, FK updated (ignored: ['shoulder_pan'])"

so a typo, a name taken from the wrong robot, or a name that needed a namespace
prefix was reported as an applied pose while ``qpos`` never changed. A mapping
that mixed one good name with one bad one was worse: half the requested pose
landed in ``qpos`` and the call still reported success, leaving the scene in a
state the caller never asked for.

Both sibling contracts already reject what they cannot apply - the ordered-list
form errors on a joint-count mismatch, and ``send_action`` errors on action keys
it cannot resolve - so these pin the same all-or-nothing contract for the dict
form: unknown (or empty) input is a structured error, nothing is written, and
the message names the joints the model does have.
"""

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import (  # noqa: E402
    _PUBLISHED_ACTIONS,
    Simulation,
)

ARM_XML = """
<mujoco model="joint_setter_test">
  <compiler angle="radian"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01"/>
    <body name="link1" pos="0 0 0.1">
      <joint name="shoulder" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
      <geom name="link1_geom" type="capsule" size="0.02 0.1"/>
      <body name="link2" pos="0 0 0.2">
        <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
        <geom name="link2_geom" type="capsule" size="0.015 0.08"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_joint_setter_names", mesh=False)
    s.create_world()
    assert s.replace_scene_mjcf(ARM_XML)["status"] == "success"
    yield s
    s.cleanup()


def _state(sim):
    return sim._world._data.qpos.copy(), sim._world._data.qvel.copy()


def test_positions_unknown_joint_rejected_and_qpos_untouched(sim):
    """A typo'd joint name is an error, not a success with nothing written."""
    qpos, _ = _state(sim)
    result = sim.set_joint_positions(positions={"shouldr": 0.4})
    assert result["status"] == "error", result
    text = result["content"][0]["text"]
    assert "Joint 'shouldr' not found" in text  # consistent not-found prefix
    assert "shoulder" in text  # close match / available joints
    # Discovery pointer, and a published action. The second half is asserted
    # rather than left to the comment: a bare name check cannot tell a
    # followable pointer from an unfollowable one, which is why this passed
    # for as long as the hint named a Python-only action.
    assert "action='get_robot_state'" in text
    assert "get_robot_state" in _PUBLISHED_ACTIONS
    assert np.array_equal(sim._world._data.qpos, qpos)


def test_positions_partial_mapping_writes_nothing(sim):
    """One bad name must not leave the good half of the pose applied."""
    qpos, _ = _state(sim)
    result = sim.set_joint_positions(positions={"shoulder": 0.4, "wrist": -0.2})
    assert result["status"] == "error", result
    assert "1 of 2" in result["content"][0]["text"]
    assert np.array_equal(sim._world._data.qpos, qpos), "partial write leaked into qpos"


def test_velocities_unknown_joint_rejected_and_qvel_untouched(sim):
    _, qvel = _state(sim)
    result = sim.set_joint_velocities(velocities={"elbo": 1.5, "wrist": 0.5})
    assert result["status"] == "error", result
    text = result["content"][0]["text"]
    assert "2 of 2" in text
    assert "wrist" in text and "elbo" in text  # every unresolved key is named
    assert np.array_equal(sim._world._data.qvel, qvel)


@pytest.mark.parametrize(
    ("method", "param"),
    [("set_joint_positions", "positions"), ("set_joint_velocities", "velocities")],
)
def test_empty_mapping_rejected(sim, method, param):
    """An empty mapping writes nothing, so reporting success is misleading."""
    result = getattr(sim, method)(**{param: {}})
    assert result["status"] == "error", result
    assert "empty" in result["content"][0]["text"]


def test_known_joints_still_applied(sim):
    """The corrected guard must not reject names the model does know."""
    result = sim.set_joint_positions(positions={"shoulder": 0.5, "elbow": -0.3})
    assert result["status"] == "success", result
    assert "2/2" in result["content"][0]["text"]
    model, data = sim._world._model, sim._world._data
    for name, expected in (("shoulder", 0.5), ("elbow", -0.3)):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        assert data.qpos[model.jnt_qposadr[jid]] == pytest.approx(expected)

    vel = sim.set_joint_velocities(velocities={"shoulder": 1.0})
    assert vel["status"] == "success", vel
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder")
    assert data.qvel[model.jnt_dofadr[jid]] == pytest.approx(1.0)


def test_namespaced_robot_joint_short_name_still_resolves():
    """A short name that only resolves via a robot namespace must keep working."""
    s = Simulation(tool_name="test_joint_setter_namespace", mesh=False)
    try:
        s.create_world()
        if s.add_robot(name="so101_follower")["status"] != "success":
            pytest.skip("so101_follower assets unavailable")
        joints = s.robot_joint_names("so101_follower")
        assert joints, "fixture robot must expose joint names"
        result = s.set_joint_positions(positions={joints[0]: 0.1})
        assert result["status"] == "success", result
    finally:
        s.cleanup()
