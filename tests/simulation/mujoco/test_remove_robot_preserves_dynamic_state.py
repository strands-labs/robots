"""``remove_robot`` preserves the dynamic state ``add_robot`` already preserves.

Removing a robot rebuilds the whole scene: ``eject_robot_from_scene`` compiles a
fresh model and allocates a fresh ``MjData``, so every buffer starts at its
reset value and the survivors' state has to be carried across that gap by name.
The pose half of that carry was already pinned (``test_multi_robot_eject``,
``test_scene_ops_guardrails``). The pose is not the whole state, and the other
half is what this module pins, because each missing piece failed silently -
``remove_robot`` returned ``"success"`` and an immediate read-back looked right:

* ``ctrl`` - the setpoint a position servo holds a pose with. Fresh ``MjData``
  reads it as zero, so the next ``mj_step`` drove every actuator of every
  SURVIVING robot toward its zero configuration. An arm parked mid-air sagged to
  the floor, one step after a call that only removed some other robot. ``act``
  carries the same command for a stateful actuator.
* an unnamed ``<freejoint/>`` - the standard MJCF floating-base idiom, used by
  the Unitree Go2 and by LeKiwi. No joint-name lookup reaches it, so a surviving
  mobile base was re-seated at its spawn pose with its velocity zeroed.
* ``data.time`` - reset to 0 while ``world.sim_time`` kept counting, so the clock
  the physics reads and the clock ``get_state`` reports disagreed.

The reference for all three is the other direction of the same composition:
:meth:`~strands_robots.simulation.mujoco.MuJoCoSimEngine.add_robot` documents
that "an arm already in the world keeps the pose it is in and the actuator
setpoints holding it there ... and the clock keeps counting", and it delivers
that. Each preservation test below is paired with the add-path measurement, so
the two directions of one composition are pinned to agree rather than each to a
hand-picked constant.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A one-joint arm driven by a position servo: the setpoint in ``ctrl`` is the
# only thing holding the joint away from zero, so dropping it is visible as
# motion rather than as a changed number. The gains are chosen so the joint
# settles ON its setpoint well inside ``_SETTLE_STEPS`` and stays there: an
# underdamped servo keeps overshooting, which would turn "did it hold its pose"
# into a question about when the assertion happens to run.
_SERVO_ARM_XML = """
<mujoco model="servo_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link0" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="1.0"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="200"/>
  </actuator>
</mujoco>
"""

# A floating base whose free joint carries NO name - what go2 and LeKiwi ship.
# The body is named, and MuJoCo refuses a free joint alongside any other joint
# on one body ("more than 6 dofs in body"), so the body names the joint exactly.
_FLOATING_BASE_XML = """
<mujoco model="rover">
  <compiler angle="radian"/>
  <worldbody>
    <body name="chassis" pos="0 0 0.2">
      <freejoint/>
      <geom type="box" size="0.1 0.06 0.03"/>
    </body>
  </worldbody>
</mujoco>
"""

# A stateful actuator: ``dyntype="filter"`` gives it an internal activation in
# ``act`` that charges toward the command, so ``act`` is a second piece of live
# actuator state a rebuild has to carry - and unlike ``ctrl`` it is non-zero even
# while the command is constant.
_FILTERED_ARM_XML = """
<mujoco model="filtered_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link0" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1" damping="1.0"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <general name="pan_act" joint="pan" dyntype="filter" dynprm="0.4"
             gaintype="fixed" gainprm="8" biastype="none"/>
  </actuator>
</mujoco>
"""

_HELD_SETPOINT = 0.9
_SETTLE_STEPS = 400


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_eject_state", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _write_arm(tmp_path, name: str, xml: str) -> str:
    path = tmp_path / f"{name}.xml"
    path.write_text(xml)
    return str(path)


def _joint_angle(sim: Simulation, joint: str) -> float:
    """Read one hinge's angle straight out of the live model, by name."""
    world = sim._world
    assert world is not None and world._model is not None and world._data is not None
    mj = sim._mj
    jid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_JOINT, joint)
    assert jid >= 0, f"joint {joint!r} missing from the compiled model"
    return float(world._data.qpos[int(world._model.jnt_qposadr[jid])])


def _base_state(sim: Simulation, body: str) -> tuple[list[float], list[float]]:
    """Read the unnamed free joint of ``body`` as ``(qpos, qvel)``."""
    world = sim._world
    assert world is not None and world._model is not None and world._data is not None
    mj = sim._mj
    model, data = world._model, world._data
    bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body)
    assert bid >= 0, f"body {body!r} missing from the compiled model"
    assert int(model.body_jntnum[bid]) == 1, "premise: the base carries exactly its free joint"
    jid = int(model.body_jntadr[bid])
    assert not mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid), "premise: that free joint is unnamed"
    adr, dof = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    return (
        [float(v) for v in data.qpos[adr : adr + 7]],
        [float(v) for v in data.qvel[dof : dof + 6]],
    )


def _two_servo_arms(sim: Simulation, tmp_path) -> str:
    """Build a scene with a bystander holding a setpoint plus a doomed sibling.

    Returns the path of the shared arm XML. The bystander (``keeper``) is driven
    to ``_HELD_SETPOINT`` and left to settle, so its pose exists only because
    ``ctrl`` is holding it there.
    """
    sim.create_world()
    arm = _write_arm(tmp_path, "servo_arm", _SERVO_ARM_XML)
    assert sim.add_robot(name="keeper", urdf_path=arm)["status"] == "success"
    assert sim.add_robot(name="doomed", urdf_path=arm, position=[0.5, 0, 0])["status"] == "success"
    assert sim.send_action({"pan": _HELD_SETPOINT}, robot_name="keeper")["status"] == "success"
    sim.step(_SETTLE_STEPS)
    return arm


class TestASurvivingServoKeepsHoldingItsPose:
    """The setpoint holding a bystander's pose survives removing a sibling.

    This is the failure the whole module exists for: the pose read back correct
    immediately after the removal, so only stepping the world showed that the
    arm had been let go.
    """

    def test_a_bystander_does_not_sag_after_a_sibling_is_removed(self, sim: Simulation, tmp_path) -> None:
        _two_servo_arms(sim, tmp_path)
        held = _joint_angle(sim, "keeper/pan")
        assert held == pytest.approx(_HELD_SETPOINT, abs=0.05), "premise: the servo reached its setpoint"

        assert sim.remove_robot("doomed")["status"] == "success"
        # The pose itself was already carried over; that is not the question.
        assert _joint_angle(sim, "keeper/pan") == pytest.approx(held, abs=1e-9)

        sim.step(_SETTLE_STEPS)
        after = _joint_angle(sim, "keeper/pan")
        assert after == pytest.approx(held, abs=0.05), (
            f"the bystander was holding {held:.4f} rad and drifted to {after:.4f} rad over "
            f"{_SETTLE_STEPS} steps after an unrelated robot was removed - its setpoint was "
            f"dropped, so the servo drove it toward zero"
        )

    def test_the_add_path_preserves_the_same_setpoint(self, sim: Simulation, tmp_path) -> None:
        """The reference the removal is held to: adding a robot keeps it.

        Passing both before and after the fix, this is what makes the assertion
        above a symmetry requirement between the two directions of one scene
        composition rather than a number chosen by hand.
        """
        arm = _two_servo_arms(sim, tmp_path)
        held = _joint_angle(sim, "keeper/pan")

        assert sim.add_robot(name="newcomer", urdf_path=arm, position=[-0.5, 0, 0])["status"] == "success"
        sim.step(_SETTLE_STEPS)
        assert _joint_angle(sim, "keeper/pan") == pytest.approx(held, abs=0.05)

    def test_the_removed_robot_leaves_no_state_behind(self, sim: Simulation, tmp_path) -> None:
        """Carrying state over must not resurrect the robot that was removed.

        The snapshot is taken before the rebuild, so it still holds the ejected
        robot's entries; they have to be dropped on the way back in rather than
        written into whatever now occupies those indices.
        """
        _two_servo_arms(sim, tmp_path)
        assert sim.send_action({"pan": -0.7}, robot_name="doomed")["status"] == "success"
        sim.step(50)

        assert sim.remove_robot("doomed")["status"] == "success"
        world = sim._world
        assert world is not None and world._model is not None
        mj = sim._mj
        model = world._model
        names = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(int(model.nu))}
        assert names == {"keeper/pan_act"}
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "doomed/pan") < 0


class TestASurvivingFloatingBaseStaysWhereItIs:
    """An unnamed ``<freejoint/>`` is carried by its owning body's name."""

    def _rover_and_arm(self, sim: Simulation, tmp_path) -> None:
        sim.create_world()
        assert (
            sim.add_robot(name="rover", urdf_path=_write_arm(tmp_path, "rover", _FLOATING_BASE_XML))["status"]
            == "success"
        )
        assert (
            sim.add_robot(name="doomed", urdf_path=_write_arm(tmp_path, "servo_arm", _SERVO_ARM_XML))["status"]
            == "success"
        )

    def _drive_base_away_from_spawn(self, sim: Simulation) -> tuple[list[float], list[float]]:
        world = sim._world
        assert world is not None and world._model is not None and world._data is not None
        mj = sim._mj
        model, data = world._model, world._data
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "rover/chassis")
        jid = int(model.body_jntadr[bid])
        adr, dof = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        data.qpos[adr : adr + 3] = [0.8, -0.4, 0.35]
        data.qvel[dof : dof + 3] = [0.25, 0.1, 0.0]
        mj.mj_forward(model, data)
        return _base_state(sim, "rover/chassis")

    def test_the_base_keeps_its_pose_and_velocity(self, sim: Simulation, tmp_path) -> None:
        self._rover_and_arm(sim, tmp_path)
        spawn_pos, _ = _base_state(sim, "rover/chassis")
        driven_pos, driven_vel = self._drive_base_away_from_spawn(sim)
        assert driven_pos[:3] != pytest.approx(spawn_pos[:3]), "premise: the base moved off its spawn pose"

        assert sim.remove_robot("doomed")["status"] == "success"
        pos, vel = _base_state(sim, "rover/chassis")
        assert pos == pytest.approx(driven_pos, abs=1e-9), (
            f"the base was at {driven_pos[:3]} and came back at {pos[:3]} "
            f"(its spawn pose is {spawn_pos[:3]}) - an unnamed free joint is invisible "
            f"to a joint-name lookup, so it was re-seated where it was authored"
        )
        assert vel == pytest.approx(driven_vel, abs=1e-9)

    def test_the_add_path_preserves_the_same_base_pose(self, sim: Simulation, tmp_path) -> None:
        """The add-path reference for the floating base."""
        self._rover_and_arm(sim, tmp_path)
        driven_pos, driven_vel = self._drive_base_away_from_spawn(sim)

        assert (
            sim.add_robot(name="newcomer", urdf_path=_write_arm(tmp_path, "servo_arm", _SERVO_ARM_XML))["status"]
            == "success"
        )
        pos, vel = _base_state(sim, "rover/chassis")
        assert pos == pytest.approx(driven_pos, abs=1e-9)
        assert vel == pytest.approx(driven_vel, abs=1e-9)


class TestTheClockKeepsCounting:
    """``data.time`` continues across the rebuild, matching ``world.sim_time``."""

    def test_the_physics_clock_matches_the_reported_clock(self, sim: Simulation, tmp_path) -> None:
        _two_servo_arms(sim, tmp_path)
        world = sim._world
        assert world is not None and world._data is not None
        before = float(world._data.time)
        assert before > 0.0, "premise: the world has been stepped"
        assert world.sim_time == pytest.approx(before, abs=1e-9)

        assert sim.remove_robot("doomed")["status"] == "success"
        assert world._data is not None
        assert float(world._data.time) == pytest.approx(world.sim_time, abs=1e-9), (
            f"the physics clock read {float(world._data.time):.4f}s after the rebuild while "
            f"get_state reports {world.sim_time:.4f}s"
        )


class TestTheKeyNamespacesDoNotCollide:
    """A joint named exactly like some body is not confused with a base.

    MuJoCo's joint and body names are independent namespaces, so a single flat
    string key could match an unnamed free joint's body-derived handle against a
    real joint of the same name and write one element's state into the other.
    """

    _COLLIDING_XML = """
<mujoco model="collide">
  <compiler angle="radian"/>
  <worldbody>
    <body name="chassis" pos="0 0 0.2">
      <freejoint/>
      <geom type="box" size="0.1 0.06 0.03"/>
      <body name="arm" pos="0 0 0.05">
        <joint name="chassis" type="hinge" axis="0 0 1"/>
        <geom type="cylinder" size="0.02 0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

    def test_a_hinge_sharing_a_bodys_name_keeps_its_own_state(self, sim: Simulation, tmp_path) -> None:
        sim.create_world()
        assert (
            sim.add_robot(name="rover", urdf_path=_write_arm(tmp_path, "collide", self._COLLIDING_XML))["status"]
            == "success"
        )
        assert (
            sim.add_robot(name="doomed", urdf_path=_write_arm(tmp_path, "servo_arm", _SERVO_ARM_XML))["status"]
            == "success"
        )

        world = sim._world
        assert world is not None and world._model is not None and world._data is not None
        mj = sim._mj
        model, data = world._model, world._data
        # Premise: one name, two namespaces - body ``rover/chassis`` carries the
        # unnamed free joint, and hinge ``rover/chassis`` is a different element.
        assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "rover/chassis") >= 0
        hinge = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "rover/chassis")
        assert hinge >= 0

        data.qpos[int(model.jnt_qposadr[hinge])] = 0.4
        driven_pos, driven_vel = self._seat_base(sim, mj, model, data)

        assert sim.remove_robot("doomed")["status"] == "success"
        assert _joint_angle(sim, "rover/chassis") == pytest.approx(0.4, abs=1e-9)
        pos, vel = _base_state(sim, "rover/chassis")
        assert pos == pytest.approx(driven_pos, abs=1e-9)
        assert vel == pytest.approx(driven_vel, abs=1e-9)

    @staticmethod
    def _seat_base(sim: Simulation, mj, model, data) -> tuple[list[float], list[float]]:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "rover/chassis")
        jid = int(model.body_jntadr[bid])
        adr, dof = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        data.qpos[adr : adr + 3] = [0.6, -0.3, 0.4]
        data.qvel[dof : dof + 3] = [0.15, 0.0, 0.0]
        mj.mj_forward(model, data)
        return (
            [float(v) for v in data.qpos[adr : adr + 7]],
            [float(v) for v in data.qvel[dof : dof + 6]],
        )


class TestAStatefulActuatorKeepsItsActivation:
    """``act`` - a stateful actuator's internal activation - survives the rebuild.

    A ``dyntype="filter"`` actuator's effective command lives in ``act``, not in
    ``ctrl``, so restoring ``ctrl`` alone still discharges it to zero and the
    actuator has to charge back up from nothing.
    """

    def _activation(self, sim: Simulation) -> float:
        world = sim._world
        assert world is not None and world._model is not None and world._data is not None
        mj = sim._mj
        model = world._model
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "filt/pan_act")
        assert aid >= 0, "actuator missing from the compiled model"
        adr = int(model.actuator_actadr[aid])
        assert adr >= 0, "premise: the actuator is stateful, so it has an activation slot"
        return float(world._data.act[adr])

    def test_the_activation_is_carried_across_the_rebuild(self, sim: Simulation, tmp_path) -> None:
        sim.create_world()
        filtered = _write_arm(tmp_path, "filtered_arm", _FILTERED_ARM_XML)
        assert sim.add_robot(name="filt", urdf_path=filtered)["status"] == "success"
        assert (
            sim.add_robot(name="doomed", urdf_path=_write_arm(tmp_path, "servo_arm", _SERVO_ARM_XML))["status"]
            == "success"
        )
        assert sim.send_action({"pan": 0.8}, robot_name="filt")["status"] == "success"
        sim.step(150)

        charged = self._activation(sim)
        assert abs(charged) > 1e-6, "premise: the filter has charged up"

        assert sim.remove_robot("doomed")["status"] == "success"
        assert self._activation(sim) == pytest.approx(charged, abs=1e-9), (
            f"the actuator's activation was {charged:.6f} and came back "
            f"{self._activation(sim):.6f} after an unrelated robot was removed"
        )
