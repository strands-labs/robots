"""A motion primitive writes a joint pose only where ``ctrl`` IS a joint pose.

``move_to``, ``rotate_wrist`` and ``set_gripper`` all drive a joint the same way:
they write a joint coordinate into ``data.ctrl`` and step. That is the joint's
target only on a position servo. On a ``<velocity>`` drive the same number is a
rate, on a ``<motor>`` a torque - so writing a joint angle there commands a
different physical quantity that happens to be numerically equal to an angle.

``joint_drive_map`` already owns that split, and already guards the one other
surface that writes a pose into ``ctrl``
(``set_joint_positions(hold=True)``): "writing a joint angle into its ``ctrl``
would command a torque numerically equal to an angle in radians". The primitives
resolved their actuators through ``_joint_actuator_map`` alone, which asks only
whether the *transmission* is the joint, so every non-servo joint drive was
written to as if it were a pose. Two things followed, both reported as success:

* the set-point was never held. Measured on Menagerie's ``pal_tiago``
  ``tiago_velocity.xml`` - the stock asset ``joint_drive_map`` names as reaching
  this path - ``rotate_wrist`` reported ``reached: True`` for ``arm_7_joint`` and
  the joint then travelled a further 0.16 rad over the next 200 steps with
  nothing else commanded, because ``ctrl`` still held 0.4 as a rate. The
  convergence test was met by the joint sweeping *past* the number while
  accelerating.
* "holds every other joint at its current position" moved them instead. In the
  same call the joints it promised to hold travelled 0.04 - 0.28 rad, while
  ``torso_lift_joint`` - the one position servo among them - held to 0.004 rad.
  A wheel drive is the worst of these because a wheel angle accumulates, so the
  "hold" grows with every turn the wheel has already made; on the scene below it
  diverges the integrator outright (``mj_step`` reports "Nan, Inf or huge value
  in QACC" and the state resets mid-rollout, so the wrist never reaches either).

19 of the loadable registry robots have at least one joint-transmission actuator
that is not a position servo, and they split two ways, which is why the fix is
not a blanket refusal:

* the drive the primitive *targets* is not a servo (``tiago_velocity``'s arm,
  ``openarm``'s 16 motors) - the set-point cannot be commanded at all, so the
  primitive refuses and names the actuator;
* only *other* joints are non-servo - the wheels of ``lekiwi``, ``stretch3`` and
  ``tiago_dual``, whose arms are fully servo-driven. Those calls keep working
  with their residuals unchanged; the wheel drives are left uncommanded and named
  in the payload rather than written to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from strands_robots import create_simulation

mujoco = pytest.importorskip("mujoco")


# A velocity drive is the sharpest case: its ctrl clears the bias-type term that
# a naive servo check would look at (biasprm = [0, 0, -kv]), so only the position
# feedback slot distinguishes it from a servo.
_VELOCITY_ARM = """<mujoco model="varm">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="base" pos="0 0 0.2">
      <geom type="box" size="0.05 0.05 0.05" mass="2"/>
      <body name="link1" pos="0 0 0.06">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.02" mass="0.5"/>
        <body name="link2" pos="0 0 0.12">
          <joint name="wrist_roll" type="hinge" axis="0 0 1" range="-2 2" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.015" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="shoulder" kv="5" name="shoulder_vel"/>
    <velocity joint="wrist_roll" kv="5" name="wrist_vel"/>
  </actuator>
</mujoco>"""

# Servo arm + one velocity drive: the shape every stock mobile manipulator has.
_MIXED_ARM = """<mujoco model="marm">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="base" pos="0 0 0.2">
      <geom type="box" size="0.05 0.05 0.05" mass="2"/>
      <body name="wheel" pos="0.07 0 0">
        <joint name="drive_wheel" type="hinge" axis="1 0 0" damping="0.1"/>
        <geom type="cylinder" size="0.03 0.01" quat="0.7071 0 0.7071 0" mass="0.3"/>
      </body>
      <body name="link1" pos="0 0 0.06">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.12" size="0.02" mass="0.5"/>
        <body name="link2" pos="0 0 0.12">
          <joint name="wrist_roll" type="hinge" axis="0 0 1" range="-2 2" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.015" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="shoulder" kp="60" name="shoulder_pos" ctrlrange="-2 2"/>
    <position joint="wrist_roll" kp="60" name="wrist_pos" ctrlrange="-2 2"/>
    <velocity joint="drive_wheel" kv="4" name="wheel_vel" ctrlrange="-5 5"/>
  </actuator>
</mujoco>"""

_JAW_TEMPLATE = """<mujoco model="jaw">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="base" pos="0 0 0.2">
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="gripper" pos="0 0 0.06">
        <joint name="jaw" type="slide" axis="1 0 0" range="0 0.04" damping="0.5"/>
        <geom type="box" size="0.01 0.01 0.03" mass="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>{actuator}</actuator>
</mujoco>"""


def _world(tmp_path: Path, xml: str, name: str) -> Any:
    """A world holding one robot loaded from *xml*, or a hard failure saying why."""
    path = tmp_path / f"{name}.xml"
    path.write_text(xml, encoding="utf-8")
    sim = create_simulation(backend="mujoco", tool_name="t", mesh=False)
    created = sim.create_world()
    if created["status"] != "success":
        raise AssertionError(f"create_world: {created['content'][0]['text']}")
    added = sim.add_robot(name=name, urdf_path=str(path))
    if added["status"] != "success":
        raise AssertionError(f"add_robot: {added['content'][0]['text']}")
    return sim


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    blocks = [b for b in result["content"] if "json" in b]
    return dict(blocks[-1]["json"]) if blocks else {}


def _qpos(sim: Any, joint: str) -> float:
    model, data = sim.mj_model, sim._world._data
    jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if jnt < 0:
        raise AssertionError(f"no joint {joint!r}")
    return float(data.qpos[int(model.jnt_qposadr[jnt])])


def _bad_qacc(sim: Any) -> int:
    """How many times ``mj_step`` reported a diverging acceleration."""
    return int(sim._world._data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)


def _ctrl(sim: Any, actuator: str) -> float:
    model, data = sim.mj_model, sim._world._data
    act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
    if act < 0:
        raise AssertionError(f"no actuator {actuator!r}")
    return float(data.ctrl[act])


class TestATargetedDriveThatIsNotAPoseIsRefused:
    """A set-point can only be commanded where ``ctrl`` is the joint target."""

    def test_rotate_wrist_refuses_a_velocity_driven_wrist(self, tmp_path: Path) -> None:
        sim = _world(tmp_path, _VELOCITY_ARM, "varm")

        result = sim.rotate_wrist(robot_name="varm", target_yaw=0.4, max_steps=200)

        assert result["status"] == "error", (
            "a velocity drive reads target_yaw as a rate, so the servo loop stops when the joint "
            f"sweeps past the number while still accelerating, not when it settles: {_text(result)}"
        )
        message = _text(result)
        assert "wrist_vel" in message, f"the refusal must name the drive that cannot take a pose: {message}"
        assert "send_action" in message, f"the refusal must quote a way to drive it anyway: {message}"

    def test_the_refusal_commands_nothing_at_all(self, tmp_path: Path) -> None:
        """Refusing before the servo loop is what keeps the primitive atomic."""
        sim = _world(tmp_path, _VELOCITY_ARM, "varm")
        pose = {"shoulder": 0.30, "wrist_roll": 0.25}
        seeded = sim.set_joint_positions(pose, robot_name="varm")
        assert seeded["status"] == "success", _text(seeded)
        before = {j: _qpos(sim, f"varm/{j}") for j in pose}

        result = sim.rotate_wrist(robot_name="varm", target_yaw=0.4, max_steps=200)

        assert result["status"] == "error"
        for joint, was in before.items():
            assert _qpos(sim, f"varm/{joint}") == pytest.approx(was, abs=1e-12), f"{joint} moved during a refused call"
        for actuator in ("shoulder_vel", "wrist_vel"):
            assert _ctrl(sim, f"varm/{actuator}") == pytest.approx(0.0, abs=1e-12), (
                f"{actuator} was commanded by a call that refused"
            )


class TestANonPoseDriveIsLeftUncommandedRatherThanWritten:
    """Holding "every other joint" is reported for the joints it can hold."""

    def test_rotate_wrist_does_not_command_a_velocity_drive_while_holding(self, tmp_path: Path) -> None:
        sim = _world(tmp_path, _MIXED_ARM, "marm")
        # A wheel angle accumulates, so a live reading of it is a large number in
        # rate units: writing it as a "hold" spins the wheel faster the further
        # it has already turned.
        spun = sim.set_joint_positions({"drive_wheel": 0.8}, robot_name="marm")
        assert spun["status"] == "success", _text(spun)
        wheel_before = _qpos(sim, "marm/drive_wheel")

        result = sim.rotate_wrist(robot_name="marm", target_yaw=0.5, max_steps=250)

        assert _ctrl(sim, "marm/wheel_vel") == pytest.approx(0.0, abs=1e-12), (
            f"the wheel's live joint angle ({wheel_before}) was written into its ctrl, which this drive reads as a rate"
        )
        assert _bad_qacc(sim) == 0, (
            "commanding a rate drive with a joint coordinate diverged the integrator: mj_step "
            "reported 'Nan, Inf or huge value in QACC', which resets the state mid-rollout"
        )
        assert result["status"] == "success", _text(result)
        assert _payload(result)["reached"] is True, "the servo wrist still converges"
        assert _qpos(sim, "marm/drive_wheel") == pytest.approx(wheel_before, abs=5e-3), (
            "the wheel was driven by a call that only meant to hold it"
        )

    def test_the_payload_names_the_drives_it_left_alone(self, tmp_path: Path) -> None:
        sim = _world(tmp_path, _MIXED_ARM, "marm")

        result = sim.rotate_wrist(robot_name="marm", target_yaw=0.5, max_steps=250)

        assert result["status"] == "success", _text(result)
        assert _payload(result).get("uncommanded_drives") == ["wheel_vel"], (
            "a joint the primitive could not hold has to be named, not silently skipped"
        )
        assert "send_action" in _text(result)


class TestTheGripperSetpointSubstitutionNeedsAPoseDrive:
    """The driven-joint-range substitution asserts that ``ctrl`` is the joint target."""

    def test_a_velocity_jaw_with_an_unset_ctrlrange_is_refused(self, tmp_path: Path) -> None:
        xml = _JAW_TEMPLATE.format(actuator='<velocity joint="jaw" kv="5" name="jaw_vel"/>')
        sim = _world(tmp_path, xml, "vj")

        result = sim.set_gripper(robot_name="vj", state="open", steps=40)

        assert result["status"] == "error", (
            "the joint's 0..0.04 m limits were commanded as 0.04 m/s, so 'commanded open' left the "
            f"jaw 1% open and still creeping: {_text(result)}"
        )
        assert "jaw_vel" in _text(result)
        assert _qpos(sim, "vj/jaw") == pytest.approx(0.0, abs=1e-9)

    def test_a_servo_jaw_with_an_unset_ctrlrange_still_substitutes(self, tmp_path: Path) -> None:
        """The substitution itself is untouched - only its premise is now checked."""
        xml = _JAW_TEMPLATE.format(actuator='<position joint="jaw" kp="80" name="jaw_pos"/>')
        sim = _world(tmp_path, xml, "sj")

        result = sim.set_gripper(robot_name="sj", state="open", steps=60)

        assert result["status"] == "success", _text(result)
        payload = _payload(result)
        assert payload["setpoint_sources"] == {"jaw_pos": "driven joint range"}
        assert payload["targets"] == {"jaw_pos": pytest.approx(0.04)}

    def test_a_velocity_jaw_with_its_own_ctrlrange_is_still_commanded(self, tmp_path: Path) -> None:
        """A usable ctrlrange is authoritative and is already in the drive's own units.

        Only the *substitution* claims ``ctrl`` is a joint coordinate, so only the
        substitution needs the drive to be a servo. Whether an open/close
        set-point is the right thing to ask of a rate command is a separate
        question about ``set_gripper``'s vocabulary, deliberately unchanged here.
        """
        xml = _JAW_TEMPLATE.format(actuator='<velocity joint="jaw" kv="5" name="jaw_vel" ctrlrange="-1 1"/>')
        sim = _world(tmp_path, xml, "cj")

        result = sim.set_gripper(robot_name="cj", state="open", steps=20)

        assert result["status"] == "success", _text(result)
        assert _payload(result)["setpoint_sources"] == {"jaw_vel": "actuator ctrlrange"}


class TestAFullyServoRobotIsUnaffected:
    """The envelope a position-servo robot gets back does not change."""

    def test_rotate_wrist_reports_no_uncommanded_drives(self, tmp_path: Path) -> None:
        xml = _MIXED_ARM.replace(
            '<velocity joint="drive_wheel" kv="4" name="wheel_vel" ctrlrange="-5 5"/>',
            '<position joint="drive_wheel" kp="30" name="wheel_pos" ctrlrange="-5 5"/>',
        )
        sim = _world(tmp_path, xml, "sarm")

        result = sim.rotate_wrist(robot_name="sarm", target_yaw=0.5, max_steps=250)

        assert result["status"] == "success", _text(result)
        assert "uncommanded_drives" not in _payload(result), (
            "a robot with nothing to skip must get the historical payload"
        )
        assert "uncommanded" not in _text(result)

    def test_every_servo_joint_is_still_held(self, tmp_path: Path) -> None:
        xml = _MIXED_ARM.replace(
            '<velocity joint="drive_wheel" kv="4" name="wheel_vel" ctrlrange="-5 5"/>',
            '<position joint="drive_wheel" kp="30" name="wheel_pos" ctrlrange="-5 5"/>',
        )
        sim = _world(tmp_path, xml, "sarm")
        seeded = sim.set_joint_positions({"shoulder": 0.4}, robot_name="sarm", hold=True)
        assert seeded["status"] == "success", _text(seeded)

        result = sim.rotate_wrist(robot_name="sarm", target_yaw=0.5, max_steps=250)

        assert result["status"] == "success", _text(result)
        assert _ctrl(sim, "sarm/shoulder_pos") == pytest.approx(0.4, abs=2e-2), (
            "a servo joint the primitive holds must still be commanded to its live position"
        )
