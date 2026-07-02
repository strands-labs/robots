"""Learned approach-walk for the G1 using Isaac Lab's pretrained **agile**
locomotion policy.

Instead of teleporting the robots to the bedside, each G1 walks there under a
learned RL policy. We reuse the pretrained leg policy shipped with Isaac Lab
(``{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt``) — the same one the
``Isaac-PickPlace-Locomanipulation-G1`` env uses. It is a *leg* policy: it
**observes the whole body** (29 joints) and **outputs 12 leg-joint targets** to
track a base velocity command while balancing, so the upper body is free for
manipulation.

The policy is run by hand (outside the RL env), reproducing the exact observation
the env builds (see ``configs/agile_locomotion_observation_cfg.py`` and
``mdp/actions.py`` in Isaac Lab)::

    policy_input = [ vx, vy, wz, hip_height,           # command (4)
                     base_lin_vel(3), base_ang_vel(3), # body frame
                     projected_gravity(3),
                     joint_pos_rel(29),                # all body joints
                     joint_vel_rel(29) * 0.1,
                     last_leg_action(12) ]             # previous policy output
    leg_target = policy(policy_input) * 0.25 + default_leg_pos

This is the policy's native robot (``G1_29DOF_CFG``); the same 29-joint ordering
is reproduced here by resolving the joints with the same name patterns.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# ── unitree_rl_lab G1 velocity-walk policy (Unitree official, Isaac-Lab-native) ──
# Unitree's official Isaac-Lab training repo (github.com/unitreerobotics/unitree_rl_lab)
# ships a *pretrained, exported* G1-29DOF velocity-tracking walk policy. It is a plain
# 4-layer MLP actor (obs 480 → 512 → 256 → 128 → 29, ELU), trained in Isaac Lab on the
# SAME G1 articulation we use here, so its observation/action joint order is exactly our
# robot's Isaac articulation order — no SDK remap needed in sim.
#
# We run it through **torch on the GPU** (cuda) rather than onnxruntime: the DGX Spark
# (aarch64 / GB10, sm_121) has no onnxruntime CUDA execution provider available, and the
# project rule is to never run inference on the CPU. The ONNX is a pure MLP, so we lift
# its Gemm weights straight into ``torch.nn.Linear`` layers on the GPU (see VelocityWalker).
#
# Spec is taken verbatim from the policy's own C++ deploy runtime + deploy.yaml
# (policies/g1_velocity_walk.deploy.yaml):
#   obs = concat over terms of each term's 5-step history (oldest→newest), terms in order
#         base_ang_vel·0.2 (3), projected_gravity (3), velocity_commands (3),
#         joint_pos_rel·1.0 (29), joint_vel_rel·0.05 (29), last_action·1.0 (29)
#       = (3+3+3+29+29+29)·5 = 480
#   action = raw·0.25 + default_joint_pos        (all 29 joints, Isaac order)
#   command = [vx, vy, wz], ranges vx[-0.5,1.0] vy[-0.3,0.3] wz[-0.2,0.2]; step_dt 0.02 (50 Hz)
VEL_ONNX_PATH = str(Path(__file__).with_name("policies") / "g1_velocity_walk.onnx")
# The 29 body joints in the policy's (Isaac articulation) order — the first 29 joints of
# the Inspire-hand G1 (verified against deploy.yaml default_joint_pos).
VEL_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
VEL_DEFAULT_POS = [-0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3,
                   -0.2, -0.2, 0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.97, 0.97, 0.15, -0.15,
                   0.0, 0.0, 0.0, 0.0]
VEL_ANG_VEL_SCALE = 0.2
VEL_DOF_VEL_SCALE = 0.05
VEL_ACTION_SCALE = 0.25
VEL_HISTORY = 5
VEL_CMD_RANGES = ((-0.5, 1.0), (-0.3, 0.3), (-0.2, 0.2))  # vx, vy, wz

# ── unitree_rl_gym G1 flat-walk policy (open/public, BSD-3) ──────────────────
# The Isaac Lab agile policy above is a tabletop-balance *teacher* — it holds a
# stand but never strides. Unitree's open ``unitree_rl_gym`` ships a real G1
# walking policy (an LSTM actor + a gait clock); driven with its exact 47-dim
# observation it walks across the floor (issue #2 item #4, verified on video).
# Constants below are from the policy's own deploy config (policies/g1_rlgym_walk.yaml).
RLGYM_POLICY_PATH = str(Path(__file__).with_name("policies") / "g1_rlgym_walk.pt")
# 12 leg joints, LEFT then RIGHT, each [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
RLGYM_LEG_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
RLGYM_DEFAULT_ANGLES = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
# PD gains the policy was trained with — MUST be set on the Isaac leg actuators
# (see scene.build_scene_cfg). kp/kd ordered like RLGYM_LEG_ORDER.
RLGYM_KPS = [100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40]
RLGYM_KDS = [2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2]
RLGYM_ANG_VEL_SCALE = 0.25
RLGYM_DOF_VEL_SCALE = 0.05
RLGYM_ACTION_SCALE = 0.25
RLGYM_CMD_SCALE = [2.0, 2.0, 0.25]   # [vx, vy, vyaw]
RLGYM_GAIT_PERIOD = 0.8              # gait-clock period (s)

# Leg joints the policy *controls* (output), and the body joints it *observes*.
LEG_RE = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]
OBS_RE = [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint",
          ".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint"]
POLICY_NUCLEUS_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt"
STAND_HIP_HEIGHT = 0.72       # nominal command hip height (from the standing retargeter)
OUTPUT_SCALE = 0.25           # policy_output_scale from the locomanip action cfg


def make_floating_base(stage, robot_prim_path: str) -> bool:
    """Turn a fixed-base-authored G1 USD into a **floating-base** articulation so a
    locomotion policy can move its root.

    The Inspire-hand G1 USD bakes a ``root_joint`` (a ``PhysicsFixedJoint`` that
    pins the pelvis to the world) and puts the ``ArticulationRootAPI`` on *that
    joint*. With ``fix_root_link=False`` Isaac still finds the root on the joint and
    fails to build a floating articulation. We **deactivate** the ``root_joint``
    (it lives in a referenced layer, so it can't be deleted — deactivating prunes
    it, and its world-pin + root API, from composition) and move the articulation
    root onto the ``pelvis`` body. Must run BEFORE ``sim.reset()``.
    """
    from pxr import PhysxSchema, UsdPhysics

    rj = stage.GetPrimAtPath(robot_prim_path + "/root_joint")
    pelvis = stage.GetPrimAtPath(robot_prim_path + "/pelvis")
    if not pelvis:
        return False
    if rj and rj.IsValid():
        rj.SetActive(False)
    UsdPhysics.ArticulationRootAPI.Apply(pelvis)
    PhysxSchema.PhysxArticulationAPI.Apply(pelvis)
    return True


def load_agile_policy(device: str, path: Optional[str] = None):
    """Fetch + load the pretrained agile locomotion TorchScript policy (shareable
    across robots)."""
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.io.torchscript import load_torchscript_model

    return load_torchscript_model(retrieve_file_path(path or POLICY_NUCLEUS_PATH), device=device)


class LocomotionPolicy:
    """Drives one G1's legs from the shared agile policy to track a base command."""

    def __init__(self, robot, device: str, policy):
        import torch

        self._torch = torch
        self.robot = robot
        self.device = device
        self.policy = policy
        self.leg_ids, self.leg_names = robot.find_joints(LEG_RE)
        self.obs_ids, self.obs_names = robot.find_joints(OBS_RE)
        self.default_leg = robot.data.default_joint_pos[:, self.leg_ids].clone()
        self.last_action = torch.zeros(1, len(self.leg_ids), device=device)

    def reset(self) -> None:
        self.last_action = self._torch.zeros(1, len(self.leg_ids), device=self.device)

    def _obs(self, cmd):
        torch = self._torch
        d = self.robot.data
        base_lin = d.root_lin_vel_b[:, :3]
        base_ang = d.root_ang_vel_b[:, :3]
        grav = d.projected_gravity_b[:, :3]
        jpr = (d.joint_pos - d.default_joint_pos)[:, self.obs_ids]
        jvr = (d.joint_vel - d.default_joint_vel)[:, self.obs_ids] * 0.1
        cmd_t = torch.tensor([cmd], device=self.device, dtype=torch.float32)
        return torch.cat([cmd_t, base_lin, base_ang, grav, jpr, jvr, self.last_action], dim=-1)

    def act(self, cmd) -> None:
        """Run the policy for command ``[vx, vy, wz, hip_height]`` and set leg targets.
        Call once per control step (every ``decimation`` sim steps, ~50 Hz)."""
        out = self.policy.forward(self._obs(cmd))
        self.last_action = out.detach()
        tgt = out * OUTPUT_SCALE + self.default_leg
        self.robot.set_joint_position_target(tgt, joint_ids=self.leg_ids)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _base_yaw(self) -> float:
        q = self.robot.data.root_quat_w[0]  # (w, x, y, z)
        w, x, y, z = (float(v) for v in q.tolist())
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def base_xy(self):
        p = self.robot.data.root_pos_w[0, :2]
        return float(p[0]), float(p[1])

    def command_to(self, target_xy, *, stop_radius: float = 0.12,
                   max_vx: float = 0.45, turn_gain: float = 1.5):
        """Velocity command that walks the base toward ``target_xy`` (world). Returns
        ``(cmd, arrived)``. Walks forward in the heading direction and steers by yaw
        error; stops (zero command) within ``stop_radius``."""
        px, py = self.base_xy()
        dx, dy = target_xy[0] - px, target_xy[1] - py
        dist = math.hypot(dx, dy)
        if dist < stop_radius:
            return [0.0, 0.0, 0.0, STAND_HIP_HEIGHT], True
        yaw = self._base_yaw()
        heading = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        # Walk forward only when roughly facing the target; otherwise turn in place.
        vx = max_vx * max(0.0, math.cos(yaw_err)) * min(1.0, dist / 0.5)
        wz = max(-1.0, min(1.0, turn_gain * yaw_err))
        return [vx, 0.0, wz, STAND_HIP_HEIGHT], False

    def stand(self):
        """A pure standing command (balance in place)."""
        return [0.0, 0.0, 0.0, STAND_HIP_HEIGHT]


class RLGymWalker:
    """Drives one G1's legs with Unitree's open ``unitree_rl_gym`` walk policy.

    Reproduces the policy's reference deploy exactly: the 47-dim observation
    ``[ang_vel·0.25, gravity_orientation, cmd·[2,2,0.25], (q−default), dq·0.05,
    last_action, sin/cos gait phase]``, ``action·0.25 + default`` leg targets, and a
    gait clock advanced one control step per :meth:`act`. The policy is a **stateful
    LSTM**, so each robot gets its **own** module instance (interleaving two robots
    on one module would corrupt both hidden states). The matching leg PD gains must
    be set on the robot's actuators (see :func:`scene.build_scene_cfg`).

    Same call surface as :class:`LocomotionPolicy` (``act``/``command_to``/``stand``/
    ``base_xy``) so the demo's control loop is unchanged — except the command is
    ``[vx, vy, vyaw]`` (3-dim, body frame) instead of the agile ``[vx, vy, wz, h]``.
    """

    def __init__(self, robot, device: str, control_dt: float, policy_path: Optional[str] = None):
        import torch

        self._torch = torch
        self.robot = robot
        self.device = device
        self.control_dt = control_dt
        # Per-robot instance: the LSTM hidden state lives inside the module.
        self.policy = torch.jit.load(policy_path or RLGYM_POLICY_PATH, map_location=device)
        self.policy.eval()
        self.leg_ids, self.leg_names = robot.find_joints(RLGYM_LEG_ORDER, preserve_order=True)
        self.default_angles = torch.tensor(RLGYM_DEFAULT_ANGLES, device=device)
        self.cmd_scale = torch.tensor(RLGYM_CMD_SCALE, device=device)
        self.last_action = torch.zeros(12, device=device)
        self.t = 0.0  # gait clock

    def reset(self) -> None:
        self.last_action = self._torch.zeros(12, device=self.device)
        self.t = 0.0
        if hasattr(self.policy, "reset_memory"):
            self.policy.reset_memory()

    @staticmethod
    def _gravity_orientation(q):
        """Projected gravity in the base frame, by the deploy's formula. q=(w,x,y,z)."""
        import torch
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        return torch.stack([2 * (-qz * qx + qw * qy),
                            -2 * (qz * qy + qw * qx),
                            1 - 2 * (qw * qw + qz * qz)])

    def act(self, cmd) -> None:
        """Run the policy for command ``[vx, vy, vyaw]`` and set leg targets. Call
        once per control step (advances the gait clock by one ``control_dt``)."""
        torch = self._torch
        d = self.robot.data
        omega = d.root_ang_vel_b[0] * RLGYM_ANG_VEL_SCALE
        grav = self._gravity_orientation(d.root_quat_w[0])
        cmd_t = torch.tensor(cmd, device=self.device, dtype=torch.float32) * self.cmd_scale
        qj = d.joint_pos[0, self.leg_ids] - self.default_angles
        dqj = d.joint_vel[0, self.leg_ids] * RLGYM_DOF_VEL_SCALE
        phase = (self.t % RLGYM_GAIT_PERIOD) / RLGYM_GAIT_PERIOD
        sc = torch.tensor([math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)],
                          device=self.device)
        obs = torch.cat([omega, grav, cmd_t, qj, dqj, self.last_action, sc]).unsqueeze(0).float()
        action = self.policy(obs).detach()[0]
        self.last_action = action
        tgt = (action * RLGYM_ACTION_SCALE + self.default_angles).unsqueeze(0)
        self.robot.set_joint_position_target(tgt, joint_ids=self.leg_ids)
        self.t += self.control_dt

    # ── helpers (mirror LocomotionPolicy) ───────────────────────────────────────
    def _base_yaw(self) -> float:
        q = self.robot.data.root_quat_w[0]  # (w, x, y, z)
        w, x, y, z = (float(v) for v in q.tolist())
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def base_xy(self):
        p = self.robot.data.root_pos_w[0, :2]
        return float(p[0]), float(p[1])

    def command_to(self, target_xy, *, stop_radius: float = 0.14,
                   max_vx: float = 0.5, turn_gain: float = 1.5):
        """Velocity command ``[vx, vy, vyaw]`` that walks the base toward
        ``target_xy`` (world). Returns ``(cmd, arrived)``: walks forward in the
        heading direction, steers by yaw error, stops within ``stop_radius``."""
        px, py = self.base_xy()
        dx, dy = target_xy[0] - px, target_xy[1] - py
        dist = math.hypot(dx, dy)
        if dist < stop_radius:
            return [0.0, 0.0, 0.0], True
        yaw = self._base_yaw()
        heading = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        vx = max_vx * max(0.0, math.cos(yaw_err)) * min(1.0, dist / 0.5)
        vyaw = max(-1.0, min(1.0, turn_gain * yaw_err))
        return [vx, 0.0, vyaw], False

    def stand(self):
        """A pure standing command (zero velocity; the gait clock still ticks)."""
        return [0.0, 0.0, 0.0]


def _load_onnx_mlp_to_torch(onnx_path: str, device: str):
    """Lift an exported MLP-actor ONNX (Gemm/ELU stack) into a torch ``Sequential`` on
    ``device`` (the GPU). The Spark has no onnxruntime GPU execution provider, so we run
    the policy through torch.cuda instead of onnxruntime — the project rule is no CPU
    inference. Weights map 1:1: ONNX Gemm ``transB=1`` stores weight as (out, in), exactly
    ``torch.nn.Linear`` layout."""
    import onnx
    import torch
    from onnx import numpy_helper

    init = {w.name: numpy_helper.to_array(w) for w in onnx.load(onnx_path).graph.initializer}
    pairs = [("actor.0.weight", "actor.0.bias"), ("actor.2.weight", "actor.2.bias"),
             ("actor.4.weight", "actor.4.bias"), ("actor.6.weight", "actor.6.bias")]
    layers = []
    for k, (wn, bn) in enumerate(pairs):
        W = torch.tensor(init[wn], dtype=torch.float32, device=device)  # (out, in)
        b = torch.tensor(init[bn], dtype=torch.float32, device=device)
        lin = torch.nn.Linear(W.shape[1], W.shape[0]).to(device)
        with torch.no_grad():
            lin.weight.copy_(W)
            lin.bias.copy_(b)
        layers.append(lin)
        if k < len(pairs) - 1:
            layers.append(torch.nn.ELU(alpha=1.0))
    return torch.nn.Sequential(*layers).to(device).eval()


class VelocityWalker:
    """Drives one G1 with Unitree's **official** ``unitree_rl_lab`` velocity-walk policy.

    A whole-body (29-joint) velocity-tracking policy: at a non-zero command it strides to
    track ``[vx, vy, wz]``; at zero command it balances in place (so the same instance
    serves the settle-stand AND the approach-walk). The MLP runs on the GPU via torch (no
    onnxruntime — see :func:`_load_onnx_mlp_to_torch`). The 480-dim observation, the
    5-step term-major history, the ``·0.25 + default`` action map and the Isaac joint order
    are all reproduced from the policy's own deploy runtime (see VEL_* constants).

    Drop-in call surface matching the other walkers: ``act(cmd)`` / ``command_to`` /
    ``stand`` / ``base_xy`` / ``leg_ids`` — except ``cmd`` is ``[vx, vy, wz]`` (3-dim)."""

    def __init__(self, robot, device: str, control_dt: float, onnx_path: Optional[str] = None):
        import torch

        self._torch = torch
        self.robot = robot
        self.device = device
        self.control_dt = control_dt
        self.policy = _load_onnx_mlp_to_torch(onnx_path or VEL_ONNX_PATH, device)
        # 29 body joints in the policy's (Isaac articulation) order.
        self.joint_ids, _ = robot.find_joints(VEL_JOINT_NAMES, preserve_order=True)
        self.leg_ids, _ = robot.find_joints(LEG_RE)
        # Positions within the 29-vector that are leg joints (for legs-only balance mode),
        # and the matching articulation ids in that same order.
        legset = set(int(j) for j in self.leg_ids)
        self._leg_cols = [k for k, j in enumerate(self.joint_ids) if int(j) in legset]
        self._leg_apply_ids = [int(self.joint_ids[k]) for k in self._leg_cols]
        # The policy's default pose IS the real G1's natural standing/walking stance — arms at the
        # sides (shoulder_pitch 0.3, roll ±0.25, a relaxed elbow), the canonical Unitree pose.
        self.default = torch.tensor(VEL_DEFAULT_POS, device=device)        # (29,) Isaac order
        self.ang_scale = VEL_ANG_VEL_SCALE
        self.dof_vel_scale = VEL_DOF_VEL_SCALE
        self.last_action = torch.zeros(29, device=device)                  # raw policy output
        self._hist = None  # list[6] of deques(maxlen=5), oldest at index 0

    def reset(self) -> None:
        self.last_action = self._torch.zeros(29, device=self.device)
        self._hist = None

    def default_pose(self):
        """The policy's default 29-joint pose (Isaac order) + the joint ids to write it to —
        used to set the robot into the walking stance before the policy takes over."""
        return self.default.unsqueeze(0), self.joint_ids

    def _terms(self, cmd):
        torch = self._torch
        d = self.robot.data
        ang = d.root_ang_vel_b[0] * self.ang_scale                         # (3)
        grav = d.projected_gravity_b[0]                                     # (3)
        cmd_t = torch.tensor(cmd, device=self.device, dtype=torch.float32)  # (3)
        q = d.joint_pos[0, self.joint_ids]
        jpos = q - self.default                                            # (29)
        jvel = d.joint_vel[0, self.joint_ids] * self.dof_vel_scale          # (29)
        return [ang, grav, cmd_t, jpos, jvel, self.last_action]            # 6 terms

    def act(self, cmd, legs_only: bool = False) -> None:
        """Run the policy for command ``[vx, vy, wz]`` and set joint targets. Call once per
        control step (~50 Hz). With ``legs_only=True`` only the 12 LEG targets are applied —
        the policy keeps balancing the legs while another controller (Pink IK) owns the arms
        + waist for manipulation. The policy still OBSERVES the true whole-body state, so its
        leg balance accounts for wherever the arms reach."""
        from collections import deque

        torch = self._torch
        terms = self._terms(cmd)
        if self._hist is None:  # fill the 5-step history with the first observation
            self._hist = [deque([t.clone() for _ in range(VEL_HISTORY)], maxlen=VEL_HISTORY)
                          for t in terms]
        else:
            for k, t in enumerate(terms):
                self._hist[k].append(t)
        # term-major: for each term, its history oldest→newest, then concatenate terms.
        obs = torch.cat([h for term_hist in self._hist for h in term_hist]).unsqueeze(0)
        raw = self.policy(obs.float()).detach()[0]                         # (29,)
        self.last_action = raw
        tgt = raw * VEL_ACTION_SCALE + self.default                        # (29,)
        if legs_only:
            self.robot.set_joint_position_target(
                tgt[self._leg_cols].unsqueeze(0), joint_ids=self._leg_apply_ids)
        else:
            self.robot.set_joint_position_target(tgt.unsqueeze(0), joint_ids=self.joint_ids)

    # ── helpers (mirror the other walkers) ──────────────────────────────────────
    def _base_yaw(self) -> float:
        q = self.robot.data.root_quat_w[0]  # (w, x, y, z)
        w, x, y, z = (float(v) for v in q.tolist())
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def base_xy(self):
        p = self.robot.data.root_pos_w[0, :2]
        return float(p[0]), float(p[1])

    def command_to(self, target_xy, *, stop_radius: float = 0.15,
                   max_vx: float = 0.6, turn_gain: float = 1.5):
        """Velocity command ``[vx, vy, wz]`` that walks the base toward ``target_xy``
        (world), clamped to the policy's trained command ranges. Returns ``(cmd, arrived)``."""
        px, py = self.base_xy()
        dx, dy = target_xy[0] - px, target_xy[1] - py
        dist = math.hypot(dx, dy)
        if dist < stop_radius:
            return [0.0, 0.0, 0.0], True
        yaw = self._base_yaw()
        heading = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        (vxlo, vxhi), _, (wzlo, wzhi) = VEL_CMD_RANGES
        vx = min(vxhi, max(vxlo, max_vx * max(0.0, math.cos(yaw_err)) * min(1.0, dist / 0.5)))
        wz = min(wzhi, max(wzlo, turn_gain * yaw_err))
        return [vx, 0.0, wz], False

    def stand(self):
        """A pure standing command (zero velocity → balance in place)."""
        return [0.0, 0.0, 0.0]


# ── our own whole-body bed-reach policy (Isaac Lab RL, issue #2 RL session) ──
# A free-base Inspire-hand G1 that BALANCES on its own two feet while leaning + squatting
# to reach a commanded hand target — the loco-manipulation skill a *walking*-balance policy
# can't hold (the deep bed-making lean throws the CoM past the feet, so a walking policy
# step-recovers and topples). Trained in Isaac Lab (rl/bed_reach_env_cfg.py) with rsl_rl
# PPO; deployed here via the exported TorchScript actor (rl/policy/policy.pt). Verified by
# eye: the G1 squats/leans to forward+down targets without toppling (rl/eval/).
#
# Unlike VelocityWalker.legs_only, this ONE policy owns ALL 29 body joints — legs (balance)
# AND waist + arms (reach). There is no separate arm controller: the policy reaches AND
# balances in a single forward pass. The hand target is given in the robot's BASE frame (so
# the policy is yaw-invariant), so the behavior layer hands us a WORLD point and we transform
# it in with the robot's current root pose.
#
# Observation (151), term order = bed_reach_env_cfg.ObservationsCfg.PolicyCfg (concatenated):
#   base_lin_vel(3) base_ang_vel(3) projected_gravity(3) hand_target(7 = base-frame pos+quat)
#   joint_pos_rel(ALL 53 joints) joint_vel_rel(ALL 53) last_action(29)
# (no per-term scaling — the env uses the raw mdp terms). Action = raw·0.5 + default_joint_pos,
# applied to the 29 body joints in Isaac articulation order (matching the training
# JointPositionAction). The exported actor has no obs normalization (agents.py sets it off),
# so it's a pure obs→action MLP; loaded with torch.jit (verified obs 151 → action 29).
BEDREACH_POLICY_PT = str(Path(__file__).with_name("rl") / "policy" / "policy.pt")
BEDREACH_ACTION_SCALE = 0.5
# The policy is ambidextrous: it reaches with whichever hand is on the target's side (left for a
# +y target, right for −y). We resolve both wrists and pick the active one from the live command.
BEDREACH_RIGHT_EE_BODY = "right_wrist_yaw_link"
BEDREACH_LEFT_EE_BODY = "left_wrist_yaw_link"
# The base-frame command ranges the policy was TRAINED on (rl/bed_reach_env_cfg CommandsCfg).
# At deploy we clamp the commanded target into this box: if the robot steps back off its spot,
# the fixed world target would otherwise land OUTSIDE the trained reach cone (base_x > 0.5),
# and the policy chases that out-of-distribution command by leaning harder — stepping back even
# more, a runaway that walks it off its feet. Clamping keeps every command in-domain, so the
# hand reaches only as far forward as the policy can balance (an honest physical limit) instead
# of toppling. Pulled in slightly from the training edges for margin.
BEDREACH_CMD_CLAMP = ((0.20, 0.53), (-0.38, 0.38), (-0.14, 0.08))  # (x_lo,x_hi),(y),(z) base frame
# The 29 body joints the policy controls (legs + waist + arms; Inspire fingers excluded).
# Mirrors rl/robot_cfg.BODY_JOINTS — find_joints resolves these to the SAME articulation-order
# ids the training action used, so the policy's 29 outputs map 1:1.
BEDREACH_BODY_JOINTS = [
    ".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint",
    ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    ".*_elbow_joint", ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint",
]


class BedReachPolicy:
    """Drives one G1 with our whole-body bed-reach policy: it reaches a commanded WORLD hand
    target while balancing on its own feet (no pin/teleport/freeze), with whichever hand is on the
    target's side (ambidextrous). Call :meth:`set_world_target` to aim it, then :meth:`act` once per
    control step (~50 Hz, every ``decimation`` sim steps). The exported actor is a stateless MLP, so
    the only per-robot state is the ``last_action`` history term — hence one instance per robot."""

    def __init__(self, robot, device: str, policy_path: Optional[str] = None):
        import torch

        self._torch = torch
        self.robot = robot
        self.device = device
        self.policy = torch.jit.load(policy_path or BEDREACH_POLICY_PT, map_location=device)
        self.policy.eval()
        # 29 body joints in articulation order (default preserve_order=False → ascending ids,
        # exactly as the training JointPositionAction resolved them).
        self.body_ids, _ = robot.find_joints(BEDREACH_BODY_JOINTS)
        rid, _ = robot.find_bodies([BEDREACH_RIGHT_EE_BODY])
        lid, _ = robot.find_bodies([BEDREACH_LEFT_EE_BODY])
        self.right_ee_id, self.left_ee_id = int(rid[0]), int(lid[0])
        self.last_action = torch.zeros(len(self.body_ids), device=device)  # raw policy output (29)
        self._target_w = None  # world-frame hand target (3,)
        self._target_b = None  # base-frame hand target (3,) — takes precedence over _target_w
        self._cmd_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # identity (training fixed rpy=0)
        self.hand_lock = None  # None | "left" | "right": commit to one hand for the whole grab+pull

    def reset(self) -> None:
        self.last_action = self._torch.zeros(len(self.body_ids), device=self.device)

    def set_world_target(self, xyz) -> None:
        """Aim the reaching hand at a WORLD point (x, y, z). Transformed into the base frame each
        step (so it stays correct as the robot leans/turns); the policy reaches with whichever hand
        is on the target's side — aim it headward and each flanking robot uses its natural hand."""
        self._target_w = self._torch.tensor(xyz, device=self.device, dtype=self._torch.float32)
        self._target_b = None

    def set_base_target(self, pos_b) -> None:
        """Aim the reaching hand at a point in the robot's BASE frame (the frame the policy was
        TRAINED on: UniformPoseCommand samples the target in base frame). Held constant in the base
        frame, so (a) the active hand can't flip as the base drifts — base_y's sign is fixed, so one
        hand owns the whole grab+pull — and (b) the policy sees the stationary, in-distribution
        command it trained with, which holds station better than chasing a fixed world point. The
        sign of pos_b[1] selects the hand: +y → left, −y → right (same-side ambidexterity)."""
        self._target_b = self._torch.tensor(pos_b, device=self.device, dtype=self._torch.float32)
        self._target_w = None

    def _hand_target_b(self):
        """The 7-dim base-frame command [pos_b(3), quat(4)] the obs term expects. A base-frame
        target is used directly; a world target is mapped into the base frame via the inverse root
        rotation (training: des_pos_w = root_pos_w + R(root_quat)·pos_b)."""
        from isaaclab.utils.math import quat_rotate_inverse

        torch = self._torch
        d = self.robot.data
        root_p = d.root_pos_w[0]
        root_q = d.root_quat_w[0]
        if self._target_b is not None:
            pos_b = self._target_b
        else:
            tgt = self._target_w if self._target_w is not None else root_p
            pos_b = quat_rotate_inverse(root_q.unsqueeze(0), (tgt - root_p).unsqueeze(0))[0]
        # Clamp into the trained command box so a receding base can't drive an out-of-distribution
        # command (which the policy chases by leaning harder → steps back → runaway → topples).
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = BEDREACH_CMD_CLAMP
        # Hand lock: commit to ONE hand for the whole grab+pull by forcing base_y to that hand's
        # side of centre (the policy reaches with the hand on the command's side, so a one-sided
        # base_y can't flip mid-task — the failure where the gripping hand went idle and the empty
        # hand swept headward). +y side = left hand, −y side = right hand.
        if self.hand_lock == "left":
            ylo = max(ylo, 0.06)
        elif self.hand_lock == "right":
            yhi = min(yhi, -0.06)
        pos_b = torch.stack([pos_b[0].clamp(xlo, xhi), pos_b[1].clamp(ylo, yhi), pos_b[2].clamp(zlo, zhi)])
        return torch.cat([pos_b, self._cmd_quat])

    def _obs(self):
        torch = self._torch
        d = self.robot.data
        base_lin = d.root_lin_vel_b[0]                       # (3)
        base_ang = d.root_ang_vel_b[0]                       # (3)
        grav = d.projected_gravity_b[0]                      # (3)
        hand_t = self._hand_target_b()                       # (7)
        jpr = d.joint_pos[0] - d.default_joint_pos[0]        # (53) ALL joints, articulation order
        jvr = d.joint_vel[0] - d.default_joint_vel[0]        # (53)
        return torch.cat([base_lin, base_ang, grav, hand_t, jpr, jvr, self.last_action]).unsqueeze(0)

    def act(self) -> None:
        """Run the policy once and set the 29 body-joint position targets. The policy owns the
        whole body (legs balance, waist + arms reach), so this is the ONLY control call per step
        — no separate balance()/arm controller."""
        raw = self.policy(self._obs().float()).detach()[0]   # (29) raw action
        self.last_action = raw
        tgt = raw * BEDREACH_ACTION_SCALE + self.robot.data.default_joint_pos[0, self.body_ids]
        self.robot.set_joint_position_target(tgt.unsqueeze(0), joint_ids=self.body_ids)

    # ── ambidexterity helpers ────────────────────────────────────────────────────
    def _active_is_left(self) -> bool:
        """True when the policy is reaching with the LEFT hand — i.e. the commanded target is on
        the robot's left (base +y). With a hand_lock set, the answer is fixed (the clamp keeps the
        command on that side); otherwise read straight from the live (clamped) base-frame command."""
        if self.hand_lock in ("left", "right"):
            return self.hand_lock == "left"
        return bool(self._hand_target_b()[1] >= 0.0)

    def active_wrist_link(self) -> str:
        """The wrist link the active (same-side) hand uses — where the cloth grasp attaches."""
        return BEDREACH_LEFT_EE_BODY if self._active_is_left() else BEDREACH_RIGHT_EE_BODY

    def active_wrist_pose_w(self):
        """World pose (pos (3,), quat wxyz (4,)) of the active reaching hand's wrist link — used to
        anchor the spring-grip joint's local frame (cloth.add_spring_grip) at grab time."""
        ee_id = self.left_ee_id if self._active_is_left() else self.right_ee_id
        pose = self.robot.data.body_pose_w[0, ee_id]
        return pose[:3], pose[3:7]

    # ── helpers (mirror the walkers) ─────────────────────────────────────────────
    def ee_pos(self):
        """World position of the active (same-side) reaching hand."""
        ee_id = self.left_ee_id if self._active_is_left() else self.right_ee_id
        p = self.robot.data.body_pose_w[0, ee_id, :3]
        return float(p[0]), float(p[1]), float(p[2])

    def base_xy(self):
        p = self.robot.data.root_pos_w[0, :2]
        return float(p[0]), float(p[1])
