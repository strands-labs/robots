"""The joint damping floor must land on either MuJoCo layout of that field.

``actuate_robot`` converts an actuator-less (URDF-loaded) arm into a
position-servo arm, and part of that surgery is flooring each driven joint's
``damping``: bare URDFs ship none, and stiff position servos on an undamped
chain diverge. The floor is written onto the ``MjSpec``, through
``MjsJoint.damping``.

That field has two layouts across the MuJoCo range this package declares
(``mujoco>=3.2.0,<4.0.0``):

* a **per-DOF sequence** on builds from 3.10, where the floor is written to
  element 0,
* a plain **float** on the older builds, where the field itself is assigned.

Reading or writing through the wrong one raises ``TypeError``. That exception is
caught by :func:`strands_robots.simulation.mujoco.scene_ops.actuate_robot_in_scene`
as a refused spec surgery, so the whole actuation is rolled back and
``actuate_robot`` reports ``status="error"`` - i.e. on the layout the code does
not handle, no URDF arm can be actuated at all, and the failure surfaces as a
generic refusal that names neither ``damping`` nor MuJoCo.

:func:`strands_robots.simulation.mujoco.scene_ops._raise_spec_joint_damping` owns
both layouts. The tests below drive it with each layout explicitly rather than
only through the installed build, because the installed build exercises exactly
one of the two - a test that only used the real spec would pass on whichever
layout happens to be installed and pin nothing about the other, which is the
coupling this file exists to prevent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.scene_ops import _raise_spec_joint_damping  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

from .test_actuate_robot import MINI_ARM_URDF  # noqa: E402


class _PerDofDampingJoint:
    """A spec joint whose ``damping`` is a per-DOF sequence (mujoco >= 3.10)."""

    def __init__(self, *dofs: float) -> None:
        self.damping = np.array(dofs, dtype=np.float64)


class _ScalarDampingJoint:
    """A spec joint whose ``damping`` is a plain float (mujoco < 3.10)."""

    def __init__(self, damping: float) -> None:
        self.damping = float(damping)


def _damping_of(joint: Any) -> float:
    value = joint.damping
    try:
        return float(value[0])
    except TypeError:
        return float(value)


class TestTheFloorIsAppliedOnBothLayouts:
    """A joint below the floor is raised to it, whichever layout it uses."""

    def test_a_per_dof_joint_below_the_floor_is_raised(self) -> None:
        joint = _PerDofDampingJoint(0.0)
        _raise_spec_joint_damping(joint, 2.0)
        assert _damping_of(joint) == pytest.approx(2.0)

    def test_a_scalar_joint_below_the_floor_is_raised(self) -> None:
        joint = _ScalarDampingJoint(0.0)
        _raise_spec_joint_damping(joint, 2.0)
        assert _damping_of(joint) == pytest.approx(2.0)


class TestAStifferJointKeepsItsOwnDamping:
    """It is a floor, not an assignment: a larger existing value survives."""

    def test_a_per_dof_joint_above_the_floor_is_left_alone(self) -> None:
        joint = _PerDofDampingJoint(7.5)
        _raise_spec_joint_damping(joint, 2.0)
        assert _damping_of(joint) == pytest.approx(7.5)

    def test_a_scalar_joint_above_the_floor_is_left_alone(self) -> None:
        joint = _ScalarDampingJoint(7.5)
        _raise_spec_joint_damping(joint, 2.0)
        assert _damping_of(joint) == pytest.approx(7.5)


class TestFurtherDofsAreNotTouched:
    """Only DOF 0 is floored, matching the pre-existing per-DOF behaviour.

    A ball or free joint carries several DOFs in the same field. Flooring all of
    them would be a silent behaviour change for those joints, so the sequence
    layout keeps writing element 0 only.
    """

    def test_the_remaining_dofs_keep_their_values(self) -> None:
        joint = _PerDofDampingJoint(0.0, 0.25, 0.5)
        _raise_spec_joint_damping(joint, 2.0)
        assert joint.damping.tolist() == pytest.approx([2.0, 0.25, 0.5])


class TestActuationSucceedsOnTheInstalledBuild:
    """End to end: the arm actuates and its joints carry the requested floor.

    This is the surface the layout bug actually broke - ``actuate_robot``
    returning ``status="error"`` for every URDF arm - so it is asserted through
    the public facade on whatever MuJoCo is installed.
    """

    def test_actuate_robot_applies_the_damping_floor(self, tmp_path) -> None:
        urdf = tmp_path / "mini_arm.urdf"
        urdf.write_text(MINI_ARM_URDF)
        sim = Simulation(backend="mujoco")
        try:
            sim.create_world(timestep=0.002, gravity=[0, 0, -9.81], ground_plane=True)
            assert sim.add_robot(name="arm", urdf_path=str(urdf))["status"] == "success"

            result = sim.actuate_robot("arm", kp=300.0, damping=3.5)
            assert result["status"] == "success", result

            world = sim._world
            assert world is not None and world._model is not None
            model = world._model
            driven = [
                int(model.jnt_dofadr[j])
                for j in range(model.njnt)
                if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or "").startswith("arm/")
            ]
            assert driven
            assert all(float(model.dof_damping[adr]) >= 3.5 for adr in driven)
        finally:
            sim.cleanup()
