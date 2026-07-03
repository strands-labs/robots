"""Whole-body bed-reach RL env — a free-base Inspire-hand G1 learns to balance while
reaching a commanded hand target in a forward+down workspace (the loco-manipulation skill
a pure walking policy can't hold: the deep bed-making lean throws the CoM past the feet).

Manager-based RL env on Isaac Lab's native rails (trains with the bundled rsl-rl-lib 5.x).
Physically valid for sim-to-real: NO base pinning / teleporting / joint freezing. The legs
balance the free base through the normal actuator path; the waist + arms reach. A deep reach
that would topple the robot must be handled by stepping / counter-leaning — like real hardware.

The reach target is sampled in the robot's BASE frame (UniformPoseCommand), so the policy is
yaw-invariant; at deploy the behavior layer commands the bed-corner positions as base-frame
targets. Tracking is same-side/ambidextrous: the reward follows whichever `wrist_yaw_link` EE
is on the target's side (a +y target uses the LEFT hand, a −y target the RIGHT), so a sideways
drag is a natural abduction rather than a cross-body sweep one hand can't hold (see HANDS and
the `same_side_*` mdp terms this file wires up).
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide as feet_slide_fn

from .mdp import (
    base_xy_anchor_l2,
    idle_arm_deviation_l1,
    randomize_ee_load,
    same_side_position_error,
    same_side_position_error_tanh,
)
from .robot_cfg import (
    BODY_JOINTS,
    FOOT_BODIES,
    LEFT_EE_BODY,
    PELVIS_BODY,
    RIGHT_EE_BODY,
    WAIST_JOINTS,
    make_bed_g1_cfg,
)

# Nominal command body (the target is a base-frame pose; the reward tracks whichever hand is on the
# target's side, see HANDS).
REACH_BODY = RIGHT_EE_BODY
# Both hands in [right, left] order (preserve_order so the ambidextrous mdp terms see
# body_ids[0]=right, body_ids[1]=left). The reach is solved with whichever hand is on the target's
# side, so a sideways drag is a natural abduction rather than a cross-body sweep one hand can't hold.
HANDS = SceneEntityCfg("robot", body_names=[RIGHT_EE_BODY, LEFT_EE_BODY], preserve_order=True)


##
# Scene
##
@configclass
class BedReachSceneCfg(InteractiveSceneCfg):
    """Flat ground + the free-base Inspire-hand G1 + a BED in front of it. The robot faces +x
    (small yaw noise) and the bed sits just ahead, so it must bend OVER the bedside — feet
    outside, knees/shins against the bed's near face — to reach the sheet, the real constraint a
    free-space reach policy never feels (and why that policy walks itself off its spot beside a
    real bed). The reach target lives on/above the bed surface; the bed is a static collision
    obstacle, NOT terminated on contact (only a fall = pelvis/torso contact ends the episode), so
    the policy learns to work against the bedside, not avoid all contact."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot = make_bed_g1_cfg(prim_path="{ENV_REGEX_NS}/Robot")

    # The bed (mattress) as a static collision obstacle directly in front of the robot: near face
    # at x≈0.20 (just ahead of the feet), top at z=0.66, wide in y so lateral reaches stay over it.
    bed = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bed",
        spawn=sim_utils.CuboidCfg(
            size=(1.4, 2.4, 0.66),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.6, dynamic_friction=0.6, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.30, 0.22)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.90, 0.0, 0.33)),
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # Eval-only overview camera (populated in BedReachEnvCfg_PLAY; None during training so it
    # adds no render cost). play.py reads its rgb each step and ffmpeg-encodes the mp4.
    eval_cam: CameraCfg | None = None

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP
##
@configclass
class CommandsCfg:
    """A hand-target pose command sampled in the robot's base frame (position-dominant)."""

    hand_target = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=REACH_BODY,
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # base frame: +x forward, +y left, +z up (pelvis-local). On/above the bed SURFACE (the
            # bed obstacle blocks anything lower): forward onto the near part of the bed, BOTH
            # lateral sides, and from just below the cover to a bit above (raise/spread). pos_y
            # spanning both sides is what makes the reach ambidextrous — a +y (leftward) target is
            # reached with the LEFT hand, a −y target with the RIGHT hand (see HANDS), so the
            # headward drag (sideways in the base frame) is a natural abduction for whichever hand
            # leads. Deeper/lower targets are unreachable through the mattress, so we don't sample.
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
    """Position targets on the 29 body joints (legs + waist + arms). Inspire fingers are not
    part of the policy — they stay at default, held by their own actuator group."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=BODY_JOINTS, scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        hand_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_target"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


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

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # No xy spawn offset: the base spawns AT the env origin, which is the anchor the
            # station-keeping reward (base_xy_anchor_l2) measures drift from. Yaw noise is now
            # SMALL (±15°) because the bed is a fixed obstacle in front (+x): the robot must face
            # it, the way it faces the bed at the bedside. (Full yaw-invariance is unnecessary
            # here — at deploy the robot always squares up to the bed; ±15° gives orientation
            # robustness without letting the bed fall outside the reach direction.)
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.26, 0.26)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )

    # Light perturbations for balance robustness (sim-to-real). Gentle so early learning isn't
    # swamped; a real robot nudged mid-reach must recover rather than topple.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 7.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )

    # Grip-slip / sheet-tension load on the ACTIVE (gripping) hand: a random horizontal force that
    # toggles on and OFF every 1-2.5 s (slip_prob of the time it's zero). The on↔off steps are sudden
    # load changes the policy must absorb without toppling — "don't fall when the sheet slips or you
    # let go." (Force-adaptive whole-body control, FALCON / Isaac Lab 2.3 force-disturbance curriculum.)
    ee_load = EventTerm(
        func=randomize_ee_load,
        mode="interval",
        interval_range_s=(1.0, 2.5),
        params={
            "command_name": "hand_target",
            "asset_cfg": HANDS,
            "force_range": (0.0, 35.0),
            "slip_prob": 0.4,
        },
    )


@configclass
class RewardsCfg:
    # -- task: reach the SAME-SIDE hand to the commanded target (coarse shaping + sharp bonus near it)
    reach_coarse = RewTerm(
        func=same_side_position_error_tanh,
        weight=2.0,
        params={"std": 0.20, "command_name": "hand_target", "asset_cfg": HANDS},
    )
    reach_fine = RewTerm(
        func=same_side_position_error_tanh,
        weight=1.5,
        params={"std": 0.06, "command_name": "hand_target", "asset_cfg": HANDS},
    )
    reach_l2 = RewTerm(
        func=same_side_position_error,
        weight=-0.3,
        params={"command_name": "hand_target", "asset_cfg": HANDS},
    )

    # -- balance / staying alive (the hard part: hold balance through the lean)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    # Stay PLANTED: penalize the base drifting (xy) off its spawn spot. This is THE fix for the
    # deploy topple — without it the free base reaches forward by stepping BACKWARD, walking
    # itself off a bedside stance and toppling. Quadratic, so a small lean shift (~0.1 m) is
    # nearly free while a backward step (~0.5-1 m) is heavily penalized: the policy learns to
    # reach + drag by leaning/squatting with its feet planted ("without losing balance").
    base_anchor = RewTerm(
        func=base_xy_anchor_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=PELVIS_BODY)},
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={"target_height": 0.70, "asset_cfg": SceneEntityCfg("robot", body_names=PELVIS_BODY)},
    )
    feet_slide = RewTerm(
        func=feet_slide_fn,
        weight=-0.2,
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
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_.*_joint", ".*_knee_joint"])},
    )
    # keep the lower body from splaying; let the WAIST stay near neutral but free to lean
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINTS)},
    )
    # Keep the IDLE arm (whichever one is NOT reaching) natural: penalize it drifting from its
    # at-side default so it hangs instead of contorting into an uncanny counter-pose. The active
    # (same-side as the target) arm is deliberately free. Light weight — small counter-balancing is
    # still free, only awkward bends are penalized.
    idle_arm = RewTerm(
        func=idle_arm_deviation_l1,
        weight=-0.2,
        params={
            "command_name": "hand_target",
            "right_arm_cfg": SceneEntityCfg(
                "robot", joint_names=["right_shoulder_.*_joint", "right_elbow_joint", "right_wrist_.*_joint"]
            ),
            "left_arm_cfg": SceneEntityCfg(
                "robot", joint_names=["left_shoulder_.*_joint", "left_elbow_joint", "left_wrist_.*_joint"]
            ),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})  # ~57 deg pelvis tilt
    torso_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link"]), "threshold": 1.0},
    )


@configclass
class BedReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: BedReachSceneCfg = BedReachSceneCfg(num_envs=2048, env_spacing=2.5)
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
class BedReachEnvCfg_PLAY(BedReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # hold a fixed target band so the played behavior is easy to read by eye
        self.commands.hand_target.resampling_time_range = (4.0, 4.0)
        # per-env overview camera framing each robot (play.py sets the exact look-at pose)
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
