"""An uncommanded DOF is constrained inside the QP, not zeroed after it.

``MinkIKBridge(commanded_dofs=...)`` used to zero the velocity of every
uncommanded DOF *after* ``mink.solve_ik`` returned. The QP therefore solved as
if those DOFs were free: on a robot whose chassis rides slide or hinge joints -
a kinematic planar base - moving the base is the cheapest way to move the
end-effector, so the solution spent itself on the base, the mask threw that
share away, and the arm alone never converged. The reachable target below solved
to a residual of the order of the base excursion, with the arm joints pinned at
their limits, on the shipped ``move_to`` path.

The fix hands ``mink.solve_ik`` a zero ``VelocityLimit`` on every fully
uncommanded non-free joint (and a ``ConfigurationLimit`` so the commanded joints
stay inside their ranges), so the QP itself never allocates motion to a DOF the
caller cannot realise. A free joint cannot carry a velocity bound, so a floating
base keeps the post-solve mask - the behaviour it always had.

The model is synthetic and self-contained: a planar ``x``/``y``/``yaw`` base
carrying a two-link arm with a ``tcp`` site. Targets are produced by forward
kinematics of the *arm alone*, so they are reachable by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mink")

from strands_robots.simulation.ik import MinkIKBridge  # noqa: E402

_PLANAR_BASE_ARM = """
<mujoco model="planar_base_arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="base_x">
      <joint name="base_x" type="slide" axis="1 0 0"/>
      <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
      <body name="base_y">
        <joint name="base_y" type="slide" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
        <body name="base">
          <joint name="base_yaw" type="hinge" axis="0 0 1"/>
          <geom type="box" size="0.1 0.1 0.02" mass="1"/>
          <body name="link1" pos="0 0 0.02">
            <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.01" mass="0.1"/>
            <body name="link2" pos="0 0 0.2">
              <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.01" mass="0.1"/>
              <site name="tcp" pos="0 0 0.2" size="0.005"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="base_x" joint="base_x" kv="1"/>
    <velocity name="base_y" joint="base_y" kv="1"/>
    <velocity name="base_yaw" joint="base_yaw" kv="1"/>
    <position name="shoulder" joint="shoulder" kp="10"/>
    <position name="elbow" joint="elbow" kp="10"/>
  </actuator>
</mujoco>
"""

ARM_JOINTS = ("shoulder", "elbow")
BASE_JOINTS = ("base_x", "base_y", "base_yaw")


@pytest.fixture
def model() -> Any:
    return mujoco.MjModel.from_xml_string(_PLANAR_BASE_ARM)


def _arm_reachable_target(model: Any, q_arm: tuple[float, float]) -> np.ndarray:
    """The tcp position the arm alone reaches at ``q_arm``, with the base at rest."""
    data = mujoco.MjData(model)
    for name, value in zip(ARM_JOINTS, q_arm, strict=True):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_kinematics(model, data)
    return data.site("tcp").xpos.copy()


def _solve(model: Any, target: np.ndarray, commanded: tuple[str, ...]) -> np.ndarray:
    dofs = [int(model.jnt_dofadr[model.joint(name).id]) for name in commanded]
    bridge = MinkIKBridge(model, "tcp", "site", orientation_cost=0.0, max_iters=200, commanded_dofs=dofs)
    q0 = np.zeros(model.nq)
    pose = np.eye(4)
    pose[:3, 3] = target
    pose[:3, :3] = bridge.ee_pose(q0)[:3, :3]
    return bridge.solve(pose, q0)


class TestUncommandedBaseDofsDoNotAbsorbTheTask:
    @pytest.mark.parametrize("q_arm", [(0.6, -0.9), (-0.8, 1.2), (0.3, 0.4)])
    def test_the_arm_alone_reaches_a_target_the_arm_can_reach(self, model, q_arm) -> None:
        target = _arm_reachable_target(model, q_arm)
        q = _solve(model, target, ARM_JOINTS)
        data = mujoco.MjData(model)
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        residual = float(np.linalg.norm(data.site("tcp").xpos - target))
        assert residual < 5e-3, f"arm-only solve missed a reachable target by {residual:.4f} m"

    @pytest.mark.parametrize("q_arm", [(0.6, -0.9), (-0.8, 1.2)])
    def test_the_base_holds_its_seed_exactly(self, model, q_arm) -> None:
        """The documented contract: every DOF outside ``commanded_dofs`` keeps its seed."""
        q = _solve(model, _arm_reachable_target(model, q_arm), ARM_JOINTS)
        for name in BASE_JOINTS:
            assert q[model.joint(name).qposadr[0]] == pytest.approx(0.0, abs=1e-12), name

    def test_the_commanded_joints_stay_inside_their_ranges(self, model) -> None:
        """A target past the arm's reach is answered from inside the joint ranges,
        which is the configuration the position servos can actually realise."""
        q = _solve(model, np.array([0.9, 0.0, 0.1]), ARM_JOINTS)
        for name in ARM_JOINTS:
            lo, hi = model.jnt_range[model.joint(name).id]
            assert lo - 1e-9 <= q[model.joint(name).qposadr[0]] <= hi + 1e-9, name


class TestTheUnrestrictedSolveIsUnchanged:
    def test_without_a_mask_every_dof_may_move(self, model) -> None:
        """No ``commanded_dofs`` means no velocity bound - the base is free to help."""
        target = np.array([0.6, 0.0, 0.25])  # beyond the arm's 0.4 m reach: needs the base
        bridge = MinkIKBridge(model, "tcp", "site", orientation_cost=0.0, max_iters=300)
        q0 = np.zeros(model.nq)
        pose = np.eye(4)
        pose[:3, 3] = target
        pose[:3, :3] = bridge.ee_pose(q0)[:3, :3]
        q = bridge.solve(pose, q0)
        assert abs(q[model.joint("base_x").qposadr[0]]) > 0.05, "the unrestricted solve should use the base"
        assert float(np.linalg.norm(bridge.ee_pose(q)[:3, 3] - target)) < 1e-2
