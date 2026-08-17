"""A generalized force latched before the scene GROWS is still applied after it.

``spec.recompile`` transfers ``qpos``, ``qvel``, ``ctrl``, ``act`` and the clock
but neither applied-force buffer, so both have to be carried across a grow by
name. ``xfrc_applied`` already is. ``qfrc_applied`` was not, which left it
carried when the scene SHRINKS (``eject_robot_from_scene`` snapshots it with the
rest of the joint state) and dropped when the scene GROWS -- so the same latched
force survived ``remove_robot`` and vanished on ``add_object``.

That asymmetry also made ``_SceneState``'s own claim about the two buffers
("both are part of the state ``save_state`` checkpoints, so both are carried")
true on one path and not the other, and it is the reason this is a defect rather
than a design question: ``save_state`` checkpoints ``qfrc_applied`` precisely so
a restore reinstates "latched external forces, not just positions".

The force is measured by its EFFECT, from rest. A joint that is already turning
keeps turning once its drive is gone, so an angle that merely kept increasing
would not distinguish a live force from momentum.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A hinge with no actuator and no damping: only ``qfrc_applied`` can turn it.
_HINGE_XML = """
<mujoco model="hinge_only">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0.1">
      <joint name="spin" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

_TORQUE = 0.5
_STEPS = 50


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_grow_joint_forces", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _spinner(sim: Simulation, tmp_path) -> None:
    path = tmp_path / "hinge_only.xml"
    path.write_text(_HINGE_XML)
    assert sim.create_world(gravity=[0.0, 0.0, 0.0])["status"] == "success"
    assert sim.add_robot(name="spinner", urdf_path=str(path))["status"] == "success"


def _handles(sim: Simulation):
    world = sim._world
    assert world is not None and world._model is not None and world._data is not None
    mj = sim._mj
    jid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_JOINT, "spinner/spin")
    assert jid >= 0, "joint 'spinner/spin' missing from the compiled model"
    return mj, world._model, world._data, jid


def _latched_torque(sim: Simulation) -> float:
    _, model, data, jid = _handles(sim)
    return float(data.qfrc_applied[int(model.jnt_dofadr[jid])])


def _latch(sim: Simulation, value: float = _TORQUE) -> None:
    _, model, data, jid = _handles(sim)
    data.qfrc_applied[int(model.jnt_dofadr[jid])] = value


def _turn_from_rest(sim: Simulation) -> float:
    """Angle swept in ``_STEPS`` starting from zero velocity.

    Zeroing the velocity first is what makes this a measurement of the force
    rather than of momentum the joint already carried.
    """
    mj, model, data, jid = _handles(sim)
    data.qvel[int(model.jnt_dofadr[jid])] = 0.0
    mj.mj_forward(model, data)
    start = float(data.qpos[int(model.jnt_qposadr[jid])])
    assert sim.step(_STEPS)["status"] == "success"
    _, model, data, jid = _handles(sim)
    return float(data.qpos[int(model.jnt_qposadr[jid])]) - start


class TestALatchedJointForceSurvivesAGrow:
    """The grow path must carry ``qfrc_applied`` as the eject path already does."""

    @pytest.mark.parametrize(
        "grow",
        [
            pytest.param(lambda s, t: s.add_object(name="crate", shape="box", position=[1, 0, 0.1]), id="add_object"),
            pytest.param(
                lambda s, t: s.add_camera(name="extra", position=[1, -1, 1], target=[0, 0, 0]), id="add_camera"
            ),
        ],
    )
    def test_the_joint_keeps_being_driven(self, sim, tmp_path, grow):
        _spinner(sim, tmp_path)
        _latch(sim)
        baseline = _turn_from_rest(sim)
        assert baseline > 1e-4, f"premise: the torque turns the joint before any grow (got {baseline})"
        _latch(sim)  # the sweep above consumed no force, but re-latch to be explicit

        assert grow(sim, tmp_path)["status"] == "success"

        assert _latched_torque(sim) == pytest.approx(_TORQUE), (
            "spec.recompile drops qfrc_applied, so a latched joint torque stopped "
            "driving the joint as soon as anything entered the scene"
        )
        assert _turn_from_rest(sim) == pytest.approx(baseline, rel=1e-6)

    def test_it_survives_a_grow_that_adds_a_robot(self, sim, tmp_path):
        _spinner(sim, tmp_path)
        _latch(sim)
        baseline = _turn_from_rest(sim)
        assert baseline > 1e-4, "premise: the torque turns the joint"
        _latch(sim)

        path = tmp_path / "hinge_only.xml"
        assert sim.add_robot(name="newcomer", urdf_path=str(path), position=[1, 0, 0])["status"] == "success"

        assert _latched_torque(sim) == pytest.approx(_TORQUE)
        assert _turn_from_rest(sim) == pytest.approx(baseline, rel=1e-6)

    def test_the_two_rebuild_directions_now_agree(self, sim, tmp_path):
        """The point of the fix: shrink already carried it, grow did not."""
        _spinner(sim, tmp_path)
        path = tmp_path / "hinge_only.xml"
        assert sim.add_robot(name="doomed", urdf_path=str(path), position=[1, 0, 0])["status"] == "success"
        _latch(sim)

        assert sim.remove_robot("doomed")["status"] == "success"
        after_shrink = _latched_torque(sim)
        assert sim.add_object(name="crate", shape="box", position=[1, 0, 0.1])["status"] == "success"
        after_grow = _latched_torque(sim)

        assert after_grow == pytest.approx(after_shrink), (
            f"a rebuild that shrinks the scene kept {after_shrink} but one that grows it "
            f"kept {after_grow}; the same latched force must survive both"
        )


class TestTheRestoreDoesNotReachPastTheDefect:
    """Controls: these hold before the fix as well."""

    def test_a_scene_with_no_latched_force_is_untouched(self, sim, tmp_path):
        _spinner(sim, tmp_path)

        assert sim.add_object(name="crate", shape="box", position=[1, 0, 0.1])["status"] == "success"

        _, _, data, _ = _handles(sim)
        assert not data.qfrc_applied.any(), "a scene nobody applied a force to gained one"

    def test_a_new_joint_is_not_given_someone_elses_force(self, sim, tmp_path):
        _spinner(sim, tmp_path)
        _latch(sim)
        path = tmp_path / "hinge_only.xml"

        assert sim.add_robot(name="newcomer", urdf_path=str(path), position=[1, 0, 0])["status"] == "success"

        mj, model, data, _ = _handles(sim)
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "newcomer/spin")
        assert jid >= 0
        assert float(data.qfrc_applied[int(model.jnt_dofadr[jid])]) == 0.0, (
            "the newcomer's joint inherited the force latched on another robot"
        )

    def test_a_reset_still_clears_it(self, sim, tmp_path):
        _spinner(sim, tmp_path)
        _latch(sim)

        assert sim.reset()["status"] == "success"

        assert _latched_torque(sim) == 0.0, "reset must still clear a latched joint force"
