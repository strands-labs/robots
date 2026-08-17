"""A drawer joint in a real compiled scene must be reachable by a joint predicate.

``joint_progress`` documents itself as being for "drawer/door tasks where
success is 'joint near target position'". Such a fixture is necessarily a
second entity alongside the arm being controlled, and ``get_observation`` is
robot-scoped, so an unscoped read never reports the fixture's joint. These
tests drive the real MuJoCo backend rather than a stub, so they pin the
resolution against the observation structure the engine actually produces.
"""

from __future__ import annotations

import pytest

from strands_robots.simulation import create_simulation
from strands_robots.simulation.predicates import make_predicate

CABINET_MJCF = """<mujoco model="cabinet">
  <compiler angle="radian"/>
  <worldbody>
    <body name="carcass" pos="0 0 0">
      <geom type="box" size="0.075 0.075 0.006" pos="0 0 0.006"/>
      <geom type="box" size="0.075 0.006 0.057" pos="0 0.069 0.066"/>
    </body>
    <body name="front" pos="0 -0.069 0.066">
      <joint name="drawer" type="slide" axis="0 -1 0" range="0 0.20" damping="1"/>
      <geom type="box" size="0.063 0.006 0.045" mass="0.12"/>
    </body>
  </worldbody>
</mujoco>
"""

ARM_MJCF = """<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0.05">
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def scene(tmp_path):
    """Arm plus an articulated cabinet, the arm registered first."""
    arm = tmp_path / "arm.xml"
    arm.write_text(ARM_MJCF)
    cab = tmp_path / "cabinet.xml"
    cab.write_text(CABINET_MJCF)
    sim = create_simulation(backend="mujoco", tool_name="scene_joint_pred", mesh=False)
    assert sim.create_world(ground_plane=True)["status"] == "success"
    assert sim.add_robot(name="arm", urdf_path=str(arm))["status"] == "success"
    assert sim.add_robot(name="cabinet", urdf_path=str(cab), position=[0.0, -0.3, 0.0])["status"] == "success"
    try:
        yield sim
    finally:
        sim.destroy()


def _set_drawer(sim, metres: float) -> None:
    """Place the compiled drawer joint and settle kinematics."""
    import mujoco

    model, data = sim.mj_model, sim.mj_data
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cabinet/drawer")
    assert jid >= 0, "the compiled scene must contain the namespaced drawer joint"
    data.qpos[model.jnt_qposadr[jid]] = metres
    mujoco.mj_forward(model, data)


def test_the_scene_exposes_the_drawer_joint_only_when_the_fixture_is_named(scene):
    """Premise: an unscoped observation cannot see the fixture's joint.

    This is the structure that makes the resolution necessary; if it ever
    changes, the tests below stop measuring what they claim to.
    """
    assert "drawer" not in scene.get_observation(skip_images=True)
    assert "drawer" in scene.get_observation(robot_name="cabinet", skip_images=True)


@pytest.mark.parametrize("metres", [0.06, 0.12, 0.18])
def test_an_open_drawer_satisfies_an_open_criterion(scene, metres):
    _set_drawer(scene, metres)
    assert make_predicate("joint_above", joint="drawer", value=0.05)(scene) is True


def test_a_shut_drawer_satisfies_a_close_criterion(scene):
    _set_drawer(scene, 0.0)
    assert make_predicate("joint_below", joint="drawer", value=0.005)(scene) is True


def test_an_open_drawer_never_satisfies_a_close_criterion(scene):
    _set_drawer(scene, 0.18)
    assert make_predicate("joint_below", joint="drawer", value=0.005)(scene) is False


def test_the_drawer_criterion_tracks_the_compiled_joint(scene):
    """The verdict must follow physics across the threshold, both ways."""
    pred = make_predicate("joint_above", joint="drawer", value=0.10)
    _set_drawer(scene, 0.02)
    assert pred(scene) is False
    _set_drawer(scene, 0.15)
    assert pred(scene) is True
    _set_drawer(scene, 0.02)
    assert pred(scene) is False


def test_joint_progress_gives_a_live_reward_for_the_drawer(scene):
    term = make_predicate("joint_progress", joint="drawer", target=0.20, weight=10.0)
    _set_drawer(scene, 0.05)
    far = term(scene)
    _set_drawer(scene, 0.19)
    near = term(scene)
    assert far == pytest.approx(-1.5)
    assert near == pytest.approx(-0.1)
    assert near > far, "the reward must improve as the drawer approaches its target"


def test_the_arms_own_joint_still_resolves(scene):
    """The controlled robot's joints keep their existing unscoped fast path."""
    assert make_predicate("joint_below", joint="shoulder", value=0.5)(scene) is True
