"""G1 arm manipulation: world-frame differential IK + cloth grasp.

The G1 pelvis spawns rotated 90 deg about Z and 0.75 m up, so its root frame is
neither the world frame nor the identity. Isaac Lab's ``DifferentialIKController``
must therefore be driven **entirely in the world frame** — world EE pose, the
world-frame PhysX Jacobian, and a world-frame target. (Running it in the root
frame, as the stock ``run_diff_ik.py`` tutorial does, diverges for the G1
because that tutorial's robots have their root at the world origin.) Verified on
the DGX Spark: in-workspace targets converge to ~1 cm.

On a FLOATING base (the walking G1 — issue #2 item #4) the PhysX jacobian gains 6
leading root-DOF columns, so the arm-joint columns shift by +6 (``jac_cols``) and
the end-effector body index is no longer offset by -1 (``ej``). With those two
floating-base corrections the same world-frame controller converges to <1 mm on a
pinned base and within a few cm while the locomotion policy actively balances.

Grasping is a PhysX auto-attachment between the cloth and the hand's palm link;
releasing deletes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
# Kinematics-only G1 URDF (Inspire hand) for the Pink IK solver. Built from
# unitree_ros' g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf with the visual/collision
# mesh geometry stripped (Pinocchio needs only the kinematic tree, and bundling the
# STL meshes is unnecessary). Its 53 actuated joint names match the Isaac Lab Inspire
# USD exactly, so the controller maps cleanly onto our sim robot.
PINK_URDF = str(Path(__file__).with_name("assets") / "g1_inspire_kin.urdf")

ARM_JOINTS = {
    "left": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
             "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
             "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
              "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
}
EE_LINK = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
PALM_LINK = {"left": "left_hand_palm_link", "right": "right_hand_palm_link"}


class ArmIK:
    """World-frame differential-IK driver for one G1 arm."""

    def __init__(self, robot, scene, side: str, device: str, max_step: float = 0.04):
        import torch
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        self._torch = torch
        self.robot = robot
        self.side = side
        self.device = device
        # Max per-step joint change (rad). The arm eases toward the target instead
        # of whipping; a whip on the policy-balanced FLOATING base jolts the body
        # and makes it stagger (verified: unclamped IK walked the robot off its
        # feet). Set to None to disable.
        self.max_step = max_step
        # Resolve joint/body indices directly off THIS robot's articulation rather
        # than via SceneEntityCfg("robot"): the bed-making scene holds two robots
        # ("robot_0"/"robot_1"), so there is no entity literally named "robot".
        # find_joints(preserve_order=True) keeps ARM_JOINTS order, which the IK
        # uses consistently for the Jacobian columns, joint_pos, and the target.
        self.joint_ids, _ = robot.find_joints(ARM_JOINTS[side], preserve_order=True)
        body_ids, _ = robot.find_bodies([EE_LINK[side]], preserve_order=True)
        self.ee_body_id = int(body_ids[0])
        self.ej = self.ee_body_id - 1 if robot.is_fixed_base else self.ee_body_id
        # PhysX jacobian columns: a FLOATING base prepends 6 root DOFs, so the arm
        # joint columns are shifted by 6 (a fixed base has no such prefix). Without
        # this the IK drives the wrong columns and diverges.
        self.jac_cols = (list(self.joint_ids) if robot.is_fixed_base
                         else [j + 6 for j in self.joint_ids])
        # Palm link name varies by hand (Dex3 vs Inspire); fall back to the wrist
        # EE link if the named palm link isn't present.
        try:
            palm_ids, _ = robot.find_bodies([PALM_LINK[side]], preserve_order=True)
        except Exception:
            palm_ids = []
        self.palm_body_id = int(palm_ids[0]) if palm_ids else self.ee_body_id
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
            num_envs=robot.num_instances if hasattr(robot, "num_instances") else 1,
            device=device,
        )
        self.target = None

    def _ee(self):
        p = self.robot.data.body_pose_w[:, self.ee_body_id]
        return p[:, 0:3], p[:, 3:7]

    def set_target(self, pos_w) -> None:
        """Command a world-frame EE position (orientation held free)."""
        t = self._torch.tensor([list(pos_w)], device=self.device, dtype=self._torch.float32)
        self.target = t
        _, qw = self._ee()
        self.ik.reset()
        self.ik.set_command(t, ee_quat=qw)

    def clear(self) -> None:
        self.target = None

    def tick(self) -> Optional[float]:
        """Advance one IK step toward the target. Returns distance-to-target (m).

        Reads the WORLD-frame jacobian columns for this arm's joints via
        ``self.jac_cols``: on a FLOATING base the 6 root DOFs come first, so the arm
        columns are offset by +6. Indexing with the raw ``joint_ids`` (as before)
        drives the wrong columns and the arm diverges — verified on the bedside G1
        (27-37 cm error vs <1 mm with the offset). The per-step joint change is then
        clamped (:attr:`max_step`) so the arm eases in without whipping the base."""
        if self.target is None:
            return None
        torch = self._torch
        jac = self.robot.root_physx_view.get_jacobians()[:, self.ej, :, self.jac_cols]
        pw, qw = self._ee()
        cur = self.robot.data.joint_pos[:, self.joint_ids]
        jpd = self.ik.compute(pw, qw, jac, cur)
        if self.max_step is not None:
            jpd = cur + torch.clamp(jpd - cur, -self.max_step, self.max_step)
        self.robot.set_joint_position_target(jpd, joint_ids=self.joint_ids)
        return float(torch.norm(pw[0] - self.target[0]).item())

    def ee_pos(self) -> List[float]:
        return [float(x) for x in self._ee()[0][0].tolist()]

    def palm_pos(self) -> List[float]:
        """World position of the hand palm link (where the cloth grasp binds)."""
        p = self.robot.data.body_pose_w[:, self.palm_body_id]
        return [float(x) for x in p[0, 0:3].tolist()]

    def palm_path(self, robot_prim_path: str) -> str:
        return f"{robot_prim_path}/{PALM_LINK[self.side]}"


class PinkArmIK:
    """NVIDIA Pink (Pinocchio) IK driver for one G1 arm **plus the waist**.

    This is the established G1 manipulation IK from Isaac Lab's locomanipulation
    pick-place env (``isaaclab.controllers.pink_ik``). Unlike the single-arm
    differential-IK :class:`ArmIK`, it solves a weighted QP over the arm *and the
    3-DOF waist*, so the torso **leans into** a low reach the way a person bends to
    make a bed — and it respects joint limits and damps near singularities.

    Drop-in for :class:`ArmIK`: ``set_target(world_xyz)`` / ``tick()`` / ``ee_pos()``.
    Orientation is left near the wrist's current pose (position-dominant), which is
    what the friction grasp needs. The base is held (pinned) during manipulation, so
    the pelvis frame the IK solves against is stable.

    Requires ``eigenpy``/``pinocchio`` to be imported **before** the Isaac app (see
    demo.py) or Pinocchio's ``StdVec_StdString`` converter is shadowed and the
    controller fails to build.
    """

    def __init__(self, robot, side: str, device: str, control_dt: float, max_step: float = 0.05,
                 with_waist: bool = True):
        import numpy as np
        import torch
        from isaaclab.controllers.pink_ik import PinkIKController
        from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
        from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
        from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
        from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

        self._np, self._torch = np, torch
        self.robot = robot
        self.side = side
        self.device = device
        self.dt = control_dt
        self.max_step = max_step
        # The two hands run as two independent single-arm Pink solvers. Only ONE of them
        # may own the shared 3-DOF waist or they fight over those joints — so the pulling
        # (right) arm controls the waist (it leans into the reach) and the assisting
        # (left) arm is built with_waist=False (it controls only its 7 arm joints).
        self.with_waist = with_waist
        waist = WAIST_JOINTS if with_waist else []
        ee = EE_LINK[side]
        posture_joints = [ARM_JOINTS[side][0], ARM_JOINTS[side][1], ARM_JOINTS[side][2]] + waist
        cfg = PinkIKControllerCfg(
            urdf_path=PINK_URDF,
            num_hand_joints=0,
            base_link_name="pelvis",
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                LocalFrameTask(ee, base_link_frame_name="pelvis",
                               position_cost=8.0, orientation_cost=0.5, lm_damping=10, gain=0.5),
                NullSpacePostureTask(cost=0.3, lm_damping=1, controlled_frames=[ee],
                                     controlled_joints=posture_joints, gain=0.2),
            ],
            fixed_input_tasks=[],
        )
        controlled = ARM_JOINTS[side] + waist
        self.cj_ids, cj_names = robot.find_joints(controlled, preserve_order=True)
        cfg.joint_names = cj_names
        cfg.all_joint_names = list(robot.data.joint_names)
        self._tasks = cfg.variable_input_tasks
        self.ctrl = PinkIKController(cfg=cfg, robot_cfg=G1_INSPIRE_FTP_CFG, device=str(device),
                                     controlled_joint_indices=list(self.cj_ids))
        from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask as _LFT
        self._LFT = _LFT
        wid, _ = robot.find_bodies([ee], preserve_order=True)
        pid, _ = robot.find_bodies(["pelvis"], preserve_order=True)
        self.wrist_id, self.pelvis_id = int(wid[0]), int(pid[0])
        self.target = None

    def _wrist_world(self):
        p = self.robot.data.body_link_pose_w[0, self.wrist_id, :3]
        return self._np.array([float(p[0]), float(p[1]), float(p[2])])

    def set_target(self, pos_w) -> None:
        """Command a world-frame wrist position (orientation held near current)."""
        self.target = self._np.array([float(v) for v in pos_w])

    def clear(self) -> None:
        self.target = None

    def tick(self) -> Optional[float]:
        """Solve the Pink QP for the current target and apply the joint targets.
        Returns the wrist-to-target distance (m). Call once per control step."""
        if self.target is None:
            return None
        import pinocchio as pin
        np, torch = self._np, self._torch
        import isaaclab.utils.math as math_utils

        ppose = self.robot.data.body_link_pose_w[0, self.pelvis_id, :7]
        p_p = ppose[:3].cpu().numpy()
        R_p = math_utils.matrix_from_quat(ppose[3:7].unsqueeze(0))[0].cpu().numpy()
        # keep the wrist's current orientation (position-dominant reach)
        wq = self.robot.data.body_link_pose_w[0, self.wrist_id, 3:7]
        R_t = math_utils.matrix_from_quat(wq.unsqueeze(0))[0].cpu().numpy()
        trans_pelvis = R_p.T @ (self.target - p_p)
        rot_pelvis = R_p.T @ R_t
        se3 = pin.SE3(np.ascontiguousarray(rot_pelvis), np.ascontiguousarray(trans_pelvis))
        for task in self._tasks:
            if isinstance(task, self._LFT):
                task.set_target(se3)
        curr = self.robot.data.joint_pos.cpu().numpy()[0]
        des = self.ctrl.compute(curr, self.dt)  # (n_controlled,) torch on device
        cur_ctrl = self.robot.data.joint_pos[0, self.cj_ids]
        if self.max_step is not None:
            des = cur_ctrl + torch.clamp(des - cur_ctrl, -self.max_step, self.max_step)
        self.robot.set_joint_position_target(des.unsqueeze(0), joint_ids=list(self.cj_ids))
        return float(np.linalg.norm(self._wrist_world() - self.target))

    def ee_pos(self) -> List[float]:
        return [float(v) for v in self._wrist_world()]


class FingerGrip:
    """Close / open one G1 Inspire hand to physically grip a handful of cloth.

    The Inspire hand joints are ``R_``/``L_``-prefixed. Closing curls every finger
    (and the thumb across the palm) from its open default toward whichever joint
    limit is farther from the default — so the axis sign does not matter, ``set(1.0)``
    always makes a fist and ``set(0.0)`` opens it. Combined with high hand friction
    (:func:`apply_hand_friction`) the closed fingers cage and grip the cloth so it can
    be dragged — the friction grasp the demo relies on (no kinematic attachment)."""

    def __init__(self, robot, side: str, device: str):
        import torch

        self._torch = torch
        self.robot = robot
        prefix = "R_" if side == "right" else "L_"
        jnames = list(robot.data.joint_names)
        lo = robot.data.joint_pos_limits[0, :, 0]
        hi = robot.data.joint_pos_limits[0, :, 1]
        dft = robot.data.default_joint_pos[0]
        ids, base, span = [], [], []
        for ji, nm in enumerate(jnames):
            if not nm.startswith(prefix):
                continue
            d = float(dft[ji])
            closed = float(lo[ji]) if abs(float(lo[ji]) - d) > abs(float(hi[ji]) - d) else float(hi[ji])
            ids.append(ji)
            base.append(d)
            span.append(closed - d)
        self.ids = ids
        self.base = torch.tensor(base, device=device, dtype=torch.float32)
        self.span = torch.tensor(span, device=device, dtype=torch.float32)

    def set(self, frac: float) -> None:
        """frac 0 = open, 1 = fully closed (fist)."""
        if not self.ids:
            return
        tgt = (self.base + float(frac) * self.span).unsqueeze(0)
        self.robot.set_joint_position_target(tgt, joint_ids=self.ids)


def apply_hand_friction(stage, robot_prim_path: str, side: str,
                        static_friction: float = 2.5, dynamic_friction: float = 2.5) -> int:
    """Give the hand's colliders a high-friction ("rubberized palm/fingertips")
    material so the hand grips the cloth by *friction* — no kinematic attachment.
    Binds the material to every collider under the hand/wrist subtree. Returns the
    number of colliders bound."""
    import isaaclab.sim as sim_utils
    from omni.physx.scripts import physicsUtils
    from pxr import UsdPhysics

    mat_path = f"{robot_prim_path}/grip_material_{side}"
    if not stage.GetPrimAtPath(mat_path):
        cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=static_friction, dynamic_friction=dynamic_friction, restitution=0.0)
        cfg.func(mat_path, cfg)

    bound = []
    # Inspire fingers use an R_/L_ prefix (e.g. "R_index_proximal", "R_thumb_distal");
    # Dex3 uses "{side}_hand_*". Match the wrist + both naming styles so the whole
    # gripping surface (palm + fingertips) gets the rubberized material.
    sp = "R_" if side == "right" else "L_"
    want = (f"{side}_hand", f"{side}_wrist",
            f"{sp}index", f"{sp}middle", f"{sp}thumb", f"{sp}ring", f"{sp}pinky")
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith(robot_prim_path):
            continue
        if not any(w in path for w in want):
            continue
        # Bind to the actual collider geometry (collision meshes) and to any prim
        # carrying a collision API — covers both how the G1 hand colliders appear.
        if prim.HasAPI(UsdPhysics.CollisionAPI) or "collision" in path.lower() or prim.GetTypeName() == "Mesh":
            try:
                physicsUtils.add_physics_material_to_prim(stage, prim, mat_path)
                bound.append(path)
            except Exception:
                pass
    return bound
