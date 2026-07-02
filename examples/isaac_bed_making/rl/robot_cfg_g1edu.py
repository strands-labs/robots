# 23-DOF Unitree G1 EDU articulation config for the whole-body bed-reach RL env (issue #3).
#
# The REAL robot is a 23-DOF G1 EDU (robotics-connect `unitree_g1_edu.json`, verified on
# hardware over SSH). Its actuated joints are:
#       12 legs + waist_yaw (1) + 10 arms (shoulder p/r/y, elbow, wrist_roll per side).
# It LACKS the 6 joints the 29-DOF Isaac asset has: waist_roll, waist_pitch, and both
# wrist_pitch + wrist_yaw — the Unitree SDK enum marks those "INVALID for g1 23dof".
#
# The issue #2 bed-reach policy was trained on the FULL 29 DOF; it leans using the waist PITCH
# the EDU does not have, so it is a valid SIM result but NOT transfer-valid to this robot (the
# descriptor's `sim_real_reconciliation.notes` says exactly this: "retrain with these joints
# locked for hardware transfer"). This config does that:
#   * the ACTION set is the 23 real joints (the 6 absent joints are excluded), and
#   * the 6 absent joints are held RIGID at their neutral by a dedicated high-stiffness "locked"
#     actuator group — the sim stand-in for "this DOF does not exist."
# So the policy reaches/balances using only the DOF the hardware actually has.
#
# Everything else matches the real EDU from the descriptor: the walk-neutral default pose, the
# EE = wrist_roll_link (the distal ACTUATED link on the 23-DOF arm — there is no wrist_yaw_link
# to move), and the unitree_rl_lab walk-policy PD gains (the controller the deployed walk→reach
# handoff runs under; the descriptor sanctions "walk-policy gains to match the RL walk").
#
# The sim asset is still the 29-DOF Inspire-hand mobile USD (same free-base fix as issue #2;
# §8 of RL_WHOLE_BODY_REACH.md). Sim hands are Inspire, real hands are Brainco — irrelevant to
# the body policy, which never actions the fingers (descriptor `hand_substitution.rationale`).

from __future__ import annotations

import os

# Mobile-base Inspire-hand G1 USD (stock USD + the baked root-joint disabled and the
# articulation root moved to the pelvis). Generated once by rl/make_inspire_mobile_usd.py.
MOBILE_USD = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets", "g1_inspire_mobile.usd")

# The 23 controllable joints = the policy's action set (the EDU's real actuated DOF).
LEG_JOINTS = [
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint"]  # the EDU waist is YAW-ONLY (no roll/pitch)
ARM_JOINTS = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",  # the distal actuated arm joint on the 23-DOF arm
]
BODY_JOINTS = LEG_JOINTS + WAIST_JOINTS + ARM_JOINTS  # 12 + 1 + 10 = 23

# The 6 joints the 29-DOF asset has but the 23-DOF EDU does NOT — held RIGID at default.
LOCKED_JOINTS = [
    "waist_roll_joint",
    "waist_pitch_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
]

# End-effector body links. On the 23-DOF arm the distal ACTUATED link is the forearm
# (wrist_roll_link); there is no wrist_yaw_link motion to track. The palm is a fixed offset from
# this link (arm_fk reports a correct palm XYZ), and robotics-connect `ee_links` names exactly
# these — so train and the real-robot deploy agree on what "the hand" is.
RIGHT_EE_BODY = "right_wrist_roll_link"
LEFT_EE_BODY = "left_wrist_roll_link"
PELVIS_BODY = "pelvis"
FOOT_BODIES = ".*_ankle_roll_link"

# Standing stance the policy resets into + uses as its action/observation neutral: the EDU
# walk-neutral (robotics-connect descriptor `default_pose`) — legs bent-knee, arms at the G1's
# natural AT-SIDES pose (the velocity-walk policy's own default), so the deployed walk→reach
# handoff lands IN-DISTRIBUTION. The 6 absent joints are pinned at 0 (they do not exist).
DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.10,
    ".*_knee_joint": 0.30,
    ".*_ankle_pitch_joint": -0.20,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.30, "right_shoulder_pitch_joint": 0.30,
    "left_shoulder_roll_joint": 0.25, "right_shoulder_roll_joint": -0.25,
    "left_elbow_joint": 0.97, "right_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15, "right_wrist_roll_joint": -0.15,
    # the 6 absent joints pinned at neutral (the hardware has no motor here):
    "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    "left_wrist_pitch_joint": 0.0, "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}

# Spawn height: pelvis ~0.80 m with bent knees plants the feet without punching the floor.
SPAWN_Z = 0.80


def make_bed_g1edu_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot"):
    """Return the free-base 23-DOF G1 EDU ArticulationCfg for the bed-reach RL env.

    Same free-base Inspire-hand mobile USD as issue #2, but the action set is the 23 real EDU
    joints and the 6 absent joints are locked rigid — so the trained policy is transfer-valid to
    the physical 23-DOF EDU."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

    g = G1_INSPIRE_FTP_CFG.copy()
    g.prim_path = prim_path

    # Use the mobile-base Inspire USD (baked world pin removed, articulation root on pelvis).
    g.spawn.usd_path = os.path.abspath(MOBILE_USD)

    # Free base + gravity on (physically-valid balance task; no kinematic cheats).
    g.spawn.rigid_props.disable_gravity = False
    g.spawn.articulation_props.fix_root_link = False
    g.spawn.activate_contact_sensors = True

    # PD gains. Legs/feet/waist-yaw/arms get the unitree_rl_lab walk-policy training gains (the
    # controller the deployed walk runs under — descriptor "walk-policy gains to match the RL
    # walk"; the stock Inspire config's waist kp 5000 / arms kp 3000 collapses a balance policy).
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
    # Waist YAW only (the EDU has no waist roll/pitch — those go in the locked group below).
    g.actuators["waist"] = ImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint"],
        stiffness=200.0,
        damping=5.0,
        armature=0.001,
        effort_limit_sim=300.0,
        velocity_limit_sim=100.0,
    )
    # Arms = shoulder p/r/y + elbow + wrist_ROLL only (no wrist pitch/yaw — locked below).
    g.actuators["arms"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_roll_joint"],
        stiffness=40.0,
        damping=10.0,
        armature=0.001,
        effort_limit_sim=300.0,
        velocity_limit_sim=100.0,
    )
    # LOCKED group: the 6 joints absent on the EDU, held rigid at their neutral with a stiff PD —
    # the sim stand-in for "this DOF does not exist." They are NOT in the action set, so the
    # policy never commands them; this group only holds them put against gravity/inertia.
    g.actuators["locked"] = ImplicitActuatorCfg(
        joint_names_expr=list(LOCKED_JOINTS),
        stiffness=500.0,
        damping=20.0,
        armature=0.001,
        effort_limit_sim=300.0,
        velocity_limit_sim=100.0,
    )

    g.init_state = g.init_state.replace(
        pos=(0.0, 0.0, SPAWN_Z),
        rot=(1.0, 0.0, 0.0, 0.0),  # identity yaw = facing +x (toward the bed obstacle)
        joint_pos=dict(DEFAULT_JOINT_POS),
        joint_vel={".*": 0.0},
    )
    return g
