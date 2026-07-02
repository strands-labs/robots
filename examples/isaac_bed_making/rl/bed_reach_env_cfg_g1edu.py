"""Whole-body bed-reach RL env for the 23-DOF Unitree G1 EDU (issue #3).

This is the transfer-valid sibling of `bed_reach_env_cfg.py` (issue #2's 29-DOF env). Same
physically-valid free-base balance-while-reach task — NO base pinning / teleporting / joint
freezing — but configured for the joint set the REAL G1 EDU actually has (robotics-connect
`unitree_g1_edu.json`, verified on hardware):

  * ACTION = the 23 real EDU joints (12 legs + waist_yaw + 10 arms), NOT 29. The 6 joints the
    EDU lacks (waist roll/pitch, both wrist pitch/yaw) are excluded from the action set and held
    rigid by the robot cfg's "locked" actuator group — so the policy never relies on a DOF the
    hardware doesn't have. (The issue #2 policy leaned on waist pitch and is therefore not
    transfer-valid; this fixes that — see the descriptor's `sim_real_reconciliation`.)
  * OBSERVATION joint_pos / joint_vel are restricted to those same 23 joints, so the policy's
    input is exactly what the real robot can report. Keeping this obs/action contract explicit is
    the #1 sim-to-real lever for G1 (the public failures are obs-layout / joint-order mismatches,
    not the locking itself — IsaacLab #4037, NVIDIA forum 325592).
  * EE = wrist_roll_link (the distal ACTUATED arm link on the EDU; robotics-connect `ee_links`).

Everything else — the bed obstacle, station-keeping, grip-slip force, ambidextrous same-side
reach, the DR, the rewards — is identical to issue #2, reused as-is. Only the joint set and the
end-effector change. The shared scene + terminations are imported from `bed_reach_env_cfg`.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide as feet_slide_fn

from .bed_reach_env_cfg import BedReachSceneCfg, TerminationsCfg
from .mdp import (
    base_xy_anchor_l2,
    idle_arm_deviation_l1,
    randomize_ee_load,
    same_side_position_error,
    same_side_position_error_tanh,
)
from .robot_cfg_g1edu import (
    BODY_JOINTS,
    FOOT_BODIES,
    LEFT_EE_BODY,
    PELVIS_BODY,
    RIGHT_EE_BODY,
    WAIST_JOINTS,
    make_bed_g1edu_cfg,
)

# Nominal command body (a base-frame target pose; the reward tracks whichever hand is on the
# target's side). Both hands in [right, left] order so the ambidextrous mdp terms read
# body_ids[0]=right, body_ids[1]=left (preserve_order). EDU EE = wrist_roll_link.
REACH_BODY = RIGHT_EE_BODY
HANDS = SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY, LEFT_EE_BODY], preserve_order=True)

# The 23 real joints restricted obs/action set — the obs/action contract that must match deploy.
JOINTS_23 = SceneEntityCfg("robot", joint_names=BODY_JOINTS)


@configclass
class BedReachEduSceneCfg(BedReachSceneCfg):
    """Identical to issue #2's scene (flat ground + bed obstacle + lights + eval cam) but with the
    23-DOF G1 EDU robot in place of the 29-DOF one."""

    robot = make_bed_g1edu_cfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class CommandsCfg:
    """A hand-target pose command sampled in the robot's base frame (position-dominant)."""

    hand_target = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=REACH_BODY,
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # base frame +x fwd / +y left / +z up (pelvis-local), on/above the bed surface: forward
            # onto the near bed, BOTH lateral sides (ambidextrous — a +y target is reached with the
            # LEFT hand, −y with the RIGHT, so the headward drag is a natural abduction), and from
            # just below the cover to a bit above.
            pos_x=(0.18, 0.55),
            pos_y=(-0.40, 0.40),
            pos_z=(-0.16, 0.10),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Position targets on the 23 real EDU body joints (legs + waist_yaw + arms). The 6 absent
    joints are NOT actioned (held rigid by the robot cfg's `locked` actuator); fingers stay at
    default."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=BODY_JOINTS, scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # base_lin_vel is DELIBERATELY excluded: the real G1 cannot observe its base linear
        # velocity reliably (the deploy used a noisy leg-kinematics odom estimate), and a policy
        # that leans on it mis-balances on hardware. Train without it — the #1 sim-to-real fix.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        hand_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_target"})
        # joint_pos / joint_vel restricted to the 23 real EDU joints — the obs the hardware can
        # actually report (transfer parity). NOT all 29 (and not the Inspire finger joints).
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": JOINTS_23}, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": JOINTS_23}, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """PRIVILEGED critic observation (runs in sim ONLY — never deployed). Asymmetric
        actor-critic: the critic keeps base_lin_vel (the term the actor drops for hardware
        deployability), so the value function still sees the true base velocity. Dropping it from
        BOTH nets starved the critic and destabilised training (the 0.39->0.29 regression); this is
        the standard sim-to-real fix (privileged critic / teacher-style obs). Clean (no noise)."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        hand_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_target"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": JOINTS_23})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": JOINTS_23})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.1),
            "dynamic_friction_range": (0.5, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # Sim-to-real domain randomization: the whole-body transfer failed on the real G1, so make
    # the policy robust to the dynamics gap. Per-env (startup) variation of the actuator PD gains
    # and the base mass, alongside the friction DR above and the push_robot perturbation below.
    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.26, 0.26)},
            "velocity_range": {
                "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 7.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )
    # Grip-slip / sheet-tension load on the ACTIVE (gripping) hand, toggling on/off (FALCON-style
    # force disturbance for force-adaptive whole-body control).
    ee_load = EventTerm(
        func=randomize_ee_load,
        mode="interval",
        interval_range_s=(1.0, 2.5),
        params={"command_name": "hand_target", "asset_cfg": HANDS, "force_range": (0.0, 35.0), "slip_prob": 0.4},
    )


@configclass
class RewardsCfg:
    # -- task: reach the SAME-SIDE hand to the commanded target (coarse shaping + sharp bonus)
    reach_coarse = RewTerm(
        func=same_side_position_error_tanh, weight=2.0,
        params={"std": 0.20, "command_name": "hand_target", "asset_cfg": HANDS},
    )
    reach_fine = RewTerm(
        func=same_side_position_error_tanh, weight=1.5,
        params={"std": 0.06, "command_name": "hand_target", "asset_cfg": HANDS},
    )
    reach_l2 = RewTerm(
        func=same_side_position_error, weight=-0.3,
        params={"command_name": "hand_target", "asset_cfg": HANDS},
    )
    # -- balance / staying alive
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_anchor = RewTerm(
        func=base_xy_anchor_l2, weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=PELVIS_BODY)},
    )
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-0.5,
        params={"target_height": 0.70, "asset_cfg": SceneEntityCfg("robot", body_names=PELVIS_BODY)},
    )
    feet_slide = RewTerm(
        func=feet_slide_fn, weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
        },
    )
    # -- regularizers (smooth, sim-to-real-able motion)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_.*_joint", ".*_knee_joint"])},
    )
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINTS)},
    )
    # Keep the IDLE arm natural. On the EDU the arm's distal joint is wrist_ROLL only (no
    # wrist pitch/yaw), so the arm joint sets are shoulder + elbow + wrist_roll.
    idle_arm = RewTerm(
        func=idle_arm_deviation_l1, weight=-0.2,
        params={
            "command_name": "hand_target",
            "right_arm_cfg": SceneEntityCfg(
                "robot", joint_names=["right_shoulder_.*_joint", "right_elbow_joint", "right_wrist_roll_joint"]
            ),
            "left_arm_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_.*_joint", "left_elbow_joint", "left_wrist_roll_joint"]
            ),
        },
    )


@configclass
class BedReachEduEnvCfg(ManagerBasedRLEnvCfg):
    scene: BedReachEduSceneCfg = BedReachEduSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005  # 200 Hz physics, 50 Hz control
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class BedReachEduEnvCfg_PLAY(BedReachEduEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.hand_target.resampling_time_range = (4.0, 4.0)
        self.scene.eval_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/eval_cam",
            update_period=0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=22.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(2.4, 2.4, 1.7), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        )
