# Free-base Inspire-hand G1 articulation config for the whole-body bed-reach RL env.
#
# This is the SAME robot the demo deploys (Inspire 5-finger hands, 29 body DOF), configured
# for a physically-valid free-base balance task:
#   * Inspire 5-finger hands (hard requirement — needed to grasp the sheet).
#   * Free base + gravity ON. The stock G1_INSPIRE_FTP_CFG ships fixed-base + gravity-off for
#     tabletop manipulation; the Inspire USD also bakes a `/Robot/root_joint` PhysicsFixedJoint.
#     Isaac Lab 2.3's `modify_articulation_root_properties` DISABLES that baked joint when
#     `fix_root_link=False` (schemas.py: it finds the existing fixed joint and sets
#     JointEnabled=False), so the manager-based spawn path yields a true floating base —
#     no per-prim SetActive() surgery (verified by rl/check_spawn.py).
#   * Deploy PD gains on all 29 body joints (legs/feet/waist/arms), matching the demo's
#     `scene.walker()` and the velocity-walk deploy.yaml. The stock Inspire config is far
#     stiffer (waist kp 5000, arms kp 3000); fed those gains a whole-body policy collapses
#     (NVIDIA forums: gain misconfiguration -> falling). The Inspire "hands" group is left
#     as shipped — fingers are NOT part of the policy (held at default during the reach).

from __future__ import annotations

import os

# Mobile-base Inspire-hand G1 USD (stock USD + the baked root-joint disabled and the
# articulation root moved to the pelvis). Generated once by rl/make_inspire_mobile_usd.py.
MOBILE_USD = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets", "g1_inspire_mobile.usd")

# The 29 controllable body joints = the policy's action set (Inspire fingers excluded).
LEG_JOINTS = [
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM_JOINTS = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
]
BODY_JOINTS = LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS  # 12 + 3 + 14 = 29

# End-effector body links (the reach target tracks the right hand by default; left is
# available for a future bimanual reward). `*_wrist_yaw_link` is the EE convention used by
# NVIDIA's own locomanipulation G1 env.
RIGHT_EE_BODY = "right_wrist_yaw_link"
LEFT_EE_BODY = "left_wrist_yaw_link"
PELVIS_BODY = "pelvis"
FOOT_BODIES = ".*_ankle_roll_link"

# Standing stance the policy resets into + uses as its action/observation neutral (matches the
# demo's scene.WALK_DEFAULT_JOINTS). Legs bent-knee; arms at the G1's natural AT-SIDES pose — the
# same pose the velocity-walk policy holds (locomotion.VEL_DEFAULT_POS) — so the deployed walk→reach
# handoff lands IN-DISTRIBUTION (no OOD launch) and the idle arm rests naturally at the side.
DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.10,
    ".*_knee_joint": 0.30,
    ".*_ankle_pitch_joint": -0.20,
    "left_shoulder_pitch_joint": 0.30, "right_shoulder_pitch_joint": 0.30,
    "left_shoulder_roll_joint": 0.25, "right_shoulder_roll_joint": -0.25,
    "left_elbow_joint": 0.97, "right_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15, "right_wrist_roll_joint": -0.15,
}

# Spawn height: pelvis ~0.78 m with bent knees plants the feet on the ground without
# punching through (0.75 sinks feet through the floor on this build — see memory).
SPAWN_Z = 0.80


def make_bed_g1_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot"):
    """Return the free-base Inspire-hand G1 ArticulationCfg for the bed-reach RL env."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

    g = G1_INSPIRE_FTP_CFG.copy()
    g.prim_path = prim_path

    # Use the mobile-base Inspire USD (baked world pin removed, articulation root on pelvis).
    g.spawn.usd_path = os.path.abspath(MOBILE_USD)

    # Free base + gravity on (physically-valid balance task; no kinematic cheats).
    g.spawn.rigid_props.disable_gravity = False
    g.spawn.articulation_props.fix_root_link = False
    # Contact sensors on for foot-contact rewards / fall termination.
    g.spawn.activate_contact_sensors = True
    # Allow self-collision off (stock) — keeps the solver cheap at 4096 envs; the reward
    # shaping (not collision) keeps the arms out of the torso.

    # Deploy PD gains on the 29 body joints (legs/feet/waist/arms). Leave "hands" as shipped.
    g.actuators = dict(g.actuators)
    g.actuators["legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"],
        stiffness={".*_hip_.*_joint": 100.0, ".*_knee_joint": 150.0},
        damping={".*_hip_.*_joint": 2.0, ".*_knee_joint": 4.0},
        armature=0.03,
        effort_limit_sim=300.0,
        velocity_limit_sim=100.0,
    )
    g.actuators["feet"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        stiffness=40.0,
        damping=2.0,
        armature=0.03,
        effort_limit_sim=100.0,
        velocity_limit_sim=100.0,
    )
    g.actuators["waist"] = ImplicitActuatorCfg(
        joint_names_expr=["waist_.*_joint"],
        stiffness=200.0,
        damping=5.0,
        armature=0.001,
        effort_limit_sim=300.0,
        velocity_limit_sim=100.0,
    )
    g.actuators["arms"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
        stiffness=40.0,
        damping=10.0,
        armature=0.001,
        effort_limit_sim=300.0,
        velocity_limit_sim=100.0,
    )

    g.init_state = g.init_state.replace(
        pos=(0.0, 0.0, SPAWN_Z),
        # Identity yaw = facing +x (the demo's yaw_to_quat(0)); the bed-reach env puts the bed
        # obstacle in front at +x, so the robot squares up to it the way it faces a real bedside.
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=dict(DEFAULT_JOINT_POS),
        joint_vel={".*": 0.0},
    )
    return g
