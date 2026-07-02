"""Scene geometry + builders for the two-G1 Isaac Sim bed-making demo.

Layout (world frame, metres). The bed's long axis is **x**: the **head** is at
-x (headboard + pillows) and the **foot** is at +x.

* Bed (mattress): 2.0 (x) x 1.8 (y) x 0.66 (z) box, top at z=0.66.
* Headboard: a board standing at the head (-x), rising above the mattress.
* Two pillows: resting at the head, in place (the robots never touch them).
* Bedsheet: a particle-cloth sheet that **starts flat, lying on the foot half of
  the bed and draping off the foot** — as if it had been pulled down toward the
  foot. The head half of the mattress is bare. The robots make the bed by grabbing
  the two **forward (head-side) corners** and pulling them toward the head, drawing
  the sheet up over the mattress. (This replaced an accordion fold-back start, which
  was too hard for the simulated G1s to unfold — issue #2.)
* Robot 0 starts ~0.5 m off the -y long side and **walks in** under a learned
  policy; robot 1 mirrors on the +y side. (They used to be planted in reach;
  now they step in — issue #2 item #4.)

The G1 USD spawns rotated +90 deg about Z (facing +y); robot 1 gets -90 deg.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Tuple


def _envf(name: str, default: float) -> float:
    """Read a float tuning knob from the environment (default off → the code value).
    DIAGNOSTIC: lets a run sweep geometry without re-editing; strip before a ship."""
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default

# ── Bed (mattress) ──────────────────────────────────────────────────────────
BED_SIZE = (2.0, 1.8, 0.61)   # height lowered 5 cm (0.66→0.61) so the cover's grab edge drops into the
BED_CENTER = (0.0, 0.0, 0.305)  # balancing robots' reach; bottom stays on the floor (center = height/2)
BED_TOP_Z = BED_CENTER[2] + BED_SIZE[2] / 2.0  # 0.61
HEAD_X = -BED_SIZE[0] / 2.0  # -1.0
FOOT_X = +BED_SIZE[0] / 2.0  # +1.0
SIDE_Y = BED_SIZE[1] / 2.0   # 0.9

OVERHANG = 0.2286  # 9 inches: how far the made sheet hangs past each side + foot

# Collision rounding for the cloth-facing props (bed + pillows). A rest offset inflates
# and rounds the collider's sharp corners and leaves a small air gap; the contact offset
# (must be > rest) widens the band where contacts are generated so the cloth catches the
# collider early instead of tunnelling. This is the "round the mattress corners" fix done
# at the collision level — see scene.static_box and .isaac/fix_tune.py.
COLLIDER_REST_OFFSET = 0.015
COLLIDER_CONTACT_OFFSET = 0.035

# ── Headboard + pillows (static props at the head) ──────────────────────────
HEADBOARD_SIZE = (0.12, 1.9, 0.95)
# Centred so the headboard front face OVERLAPS the mattress head by ~6 cm (front at
# x=-0.94 vs bed back at -1.0) — it sits flush against the bed, no gap. (Was HEAD_X-0.06,
# which left a visible gap.) It also gets a rounded collider (static_box rounded=True).
HEADBOARD_CENTER = (HEAD_X, 0.0, 0.55)  # touches the bed, rises to ~1.0
PILLOW_SIZE = (0.5, 0.72, 0.16)
# Pillows render as rounded superellipsoid meshes (props.superellipsoid_mesh built in
# demo.py), not square boxes; 1.0 = ellipsoid, ~0.2 = barely rounded. Box collider stays.
PILLOW_ROUNDNESS = 0.35
# Prop the pillows UP against the headboard (almost vertical), not lying flat on the mattress.
# Tipping the 0.5 m-long pillow about the y (bed-width) axis stands it on its foot-side edge with
# its head-side edge leaning back onto the headboard front (x≈-0.94): the base rests at the
# mattress-headboard junction and the top pokes just above the headboard, like a real propped
# pillow. PILLOW_PROP_DEG is the tip angle from flat (≈70° → ~20° off vertical, leaning back).
PILLOW_PROP_DEG = _envf("BEDDEMO_PILLOW_PROP_DEG", 70.0)
# Centre of the propped pillow: pushed toward the headboard and raised to its standing mid-height.
PILLOWS = {
    "left": (HEAD_X + 0.22, -0.43, BED_TOP_Z + 0.22),
    "right": (HEAD_X + 0.22, 0.43, BED_TOP_Z + 0.22),
}

# ── Robots: walk in from ~1 m off their long side, flank the cover's head edge ──
# (pos, yaw_deg). yaw about Z: +90 faces +y, -90 faces -y. They flank opposite long sides at
# the cover's head edge (~mid-bed), facing the bed, then DRAW the cover headward up toward the
# pillows — walking headward as they pull, so they end up flanking the head (the Figure Helix
# approach, .isaac/FIGURE_ANALYSIS.md). The geometry that dissolves the grab/pull hand-conflict:
# facing the bed, world −x (headward) is base +y for robot 0 (left hand) and base −y for robot 1
# (right hand); with the grab point HEADWARD-inboard of the robot (the head edge sits just
# headward of MANIP_X), the grab and the headward pull keep the same base-y sign, so one hand
# owns the whole grab+pull (no mid-task hand flip). The reach policy translates the base toward
# the headward target, so the robots WALK the cover up rather than pull it taut from a planted
# stance (the accordion slack feeds out). See run_bedmaking_rl.
ROBOT_Z = 0.80      # spawn pelvis height: the velocity-walk policy's natural standing
                    # height (feet just touch the floor). 0.75 punched the feet ~4 cm
                    # through the floor at spawn and the policy couldn't recover.
STAND_PELVIS_Z = 0.80  # clean upright standing pelvis height for manipulation (feet on floor)
# Start just FOOTWARD of the cover's head edge (~x=0.15) so the head edge is a short headward
# reach in front of each robot (keeping grab+pull on one hand), then walk headward dragging it.
MANIP_X = _envf("BEDDEMO_MANIP_X", 0.40)   # flank ~0.2 m footward of the cover's settled head edge (~x=0.20); walk headward to draw it up
MANIP_Y = 1.05      # walk-in target at the bedside (just off the y=+/-0.9 bed side)
APPROACH_Y = MANIP_Y + 1.0  # spawn ~1 m out from the bedside mark, so the G1s WALK in under the
                            # policy (not teleport); the walk doubles as the sheet's drape time.
ROBOTS = {
    0: {"approach": (MANIP_X, -APPROACH_Y, ROBOT_Z), "manip": (MANIP_X, -MANIP_Y),
        "yaw_deg": 90.0, "side": "-y"},
    1: {"approach": (MANIP_X, APPROACH_Y, ROBOT_Z), "manip": (MANIP_X, MANIP_Y),
        "yaw_deg": -90.0, "side": "+y"},
}
# Spawn / default stance for the demo G1s. Bent-knee legs (the rl_gym/velocity walk default, an
# offset from this pose). Arms at the G1's natural AT-SIDES pose — the same pose the velocity-walk
# policy holds (locomotion.VEL_DEFAULT_POS) and the bed-reach policy trains with (rl.robot_cfg
# DEFAULT_JOINT_POS) — so the robot's observation neutral matches the trained policy and the
# walk→reach handoff lands in-distribution. (These two dicts must stay in sync.)
WALK_DEFAULT_JOINTS = {
    ".*_hip_pitch_joint": -0.10,
    ".*_knee_joint": 0.30,
    ".*_ankle_pitch_joint": -0.20,
    "left_shoulder_pitch_joint": 0.30, "right_shoulder_pitch_joint": 0.30,
    "left_shoulder_roll_joint": 0.25, "right_shoulder_roll_joint": -0.25,
    "left_elbow_joint": 0.97, "right_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15, "right_wrist_roll_joint": -0.15,
}

# Bed corners in the world (compass labels match the swarm driver). +x=foot/East.
BED_CORNERS: Dict[str, Tuple[float, float, float]] = {
    "NE": (FOOT_X, SIDE_Y, BED_TOP_Z),
    "NW": (HEAD_X, SIDE_Y, BED_TOP_Z),
    "SE": (FOOT_X, -SIDE_Y, BED_TOP_Z),
    "SW": (HEAD_X, -SIDE_Y, BED_TOP_Z),
}

# ── Bedsheet ────────────────────────────────────────────────────────────────
# ACCORDION-AT-THE-FOOT start (issue #2): the cover is gathered toward the FOOT of the bed,
# its head edge at ~mid-bed (well foot-of-the-pillows so there is a real draw to make), the
# rest pleated into a ruffle of slack over the foot half and draping off the foot. The head
# half up to the pillows is bare mattress. The robots grab the head edge and draw it HEADWARD
# up toward the pillows; the foot accordion UNSPOOLS that slack as they pull, so the cover
# feeds up instead of dragging taut (a taut sheet yanks the gripping hand and launches the
# balancing robot — issue #2). See cloth.build_bedsheet's accordion_* args.
#
# FULL SIDE OVERHANG (~9 in each long side): the cover hangs over the mattress sides — visually
# important for the demo. It is a FINE, thin, drapey sheet so the overhang folds DOWN over the
# mattress edges (below the robots' reach height) and stays calm, rather than juting out stiff.
# The walk-in is the drape time: by the time the robots arrive the side overhang has folded
# down off the edges, clearing their arm space (so it does not dump on their arms and topple
# them — the reason it was briefly trimmed; restored here per the demo's visual need).
SHEET_LEN = _envf("BEDDEMO_SHEET_LEN", 1.55)                  # x length: head edge → off the foot
SHEET_W = _envf("BEDDEMO_SHEET_W", BED_SIZE[1] + 2 * OVERHANG)  # y width incl. 9-in side overhang (~2.26)
SHEET_SIZE = (SHEET_LEN, SHEET_W)
# MuJoCo recipe (issue #2, 2026-06-06): COARSE + FEATHERLIGHT + THICK is what made the
# MuJoCo flex sheet look good and hang still — not "triangles". MuJoCo used 143 verts /
# 0.18 kg / ~4.4 cm. Match it: a coarse grid (with cloth.py's light particle_mass this
# totals ~0.2 kg) settles instead of sloshing. NOTE: the *physics* is thick, but the
# RENDER is still a single-layer membrane (looks thin) — giving it visual thickness
# needs a render-side shell (extrude / double-layer the mesh). That is the next task.
SHEET_RES = (14, 12)                            # coarse like MuJoCo's 13×11 → stable drape
SHEET_THICKNESS = 0.05                          # thick physics/collision profile
# Visual thickness: the particle cloth is a single-layer membrane (renders thin), so
# cloth.build_shell_mesh extrudes a closed double-layer SLAB of this depth along the
# surface normal each frame — purely render-side, physics unchanged. 0.07 m reads as a
# substantial folded cover (validated in isolation 2026-06-08, .isaac/shell_tune).
SHEET_SHELL_THICKNESS = 0.07
# Centre placed from the desired HEAD EDGE x: the head edge (the grab line the robots draw
# headward) sits at ~mid-bed, foot-of-the-pillows, so there is a real draw to make; the cover
# extends footward from there over the foot half and drapes off the foot. Rests just above
# the mattress top so it settles onto it (not floating in the air).
SHEET_HEAD_X = _envf("BEDDEMO_SHEET_HEAD_X", 0.15)    # world x of the cover's head edge (the grab line)
SHEET_ORIGIN = (SHEET_HEAD_X + SHEET_LEN / 2.0, 0.0, BED_TOP_Z + 0.04)
# Accordion (crumple) the WHOLE cover, head edge included, so it sits as a short bunched band with
# only a little overhang over the FOOT of the bed — not a long flat sheet draping far off the foot.
# That drape's weight + the foot-edge friction was anchoring the cover and fighting the pull (issue
# #2; user feedback on the head_v6/v7 mp4). With START≈0 the whole sheet is gathered (not just the
# foot), its x-extent compressed to GATHER and the slack taken up by WAVES deep z-folds of height
# AMP that rest on each other (self-collision) so the crumple holds instead of flattening out.
SHEET_ACCORDION_START = _envf("BEDDEMO_ACC_START", 0.0)    # 0 → accordion the entire length (incl. the head end)
SHEET_ACCORDION_GATHER = _envf("BEDDEMO_ACC_GATHER", 0.5)  # compress the gathered x-extent to this fraction
SHEET_ACCORDION_WAVES = int(_envf("BEDDEMO_ACC_WAVES", 8))  # number of deep crumple folds
SHEET_ACCORDION_AMP = _envf("BEDDEMO_ACC_AMP", 0.14)       # fold height (taller folds stack + hold the crumple)
SHEET_PARTICLE_MASS = _envf("BEDDEMO_SHEET_MASS", 0.0007)  # per particle; a lighter cover is easier to draw up
# Spring damping for the particle cloth. The deep 8-wave accordion starts with a lot of fold potential
# energy; at the old default (1.5) the soft-spring lattice kept trading it back and forth — a visible
# wobble/jiggle that never settled (user feedback on the pull_v* mp4). Strong damping bleeds that energy
# so the cover drapes and SETTLES to a still rest state. Env-tunable to retune the drape↔settle balance.
SHEET_DAMPING = _envf("BEDDEMO_SHEET_DAMPING", 10.0)

# Camera: 3/4 view framing the whole bed (head + foot), both robots.
CAM_EYE = (3.9, -3.5, 2.7)
CAM_TARGET = (0.0, 0.0, 0.5)
CAM_RES = (854, 480)


def yaw_to_quat(yaw_deg: float) -> Tuple[float, float, float, float]:
    """Quaternion (w,x,y,z) for a rotation of yaw_deg about +Z."""
    h = math.radians(yaw_deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def roty_to_quat(deg: float) -> Tuple[float, float, float, float]:
    """Quaternion (w,x,y,z) for a rotation of ``deg`` about +Y (tips a prop up/back).
    Matches props.superellipsoid_mesh's rot_y_deg so a pillow's collider and visual align."""
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def build_scene_cfg(g1_cfg, spawn_at_manip: bool = False):
    """Return an InteractiveSceneCfg subclass: two floating (walk-capable) G1s +
    bed + headboard + pillows. The G1s spawn at their approach positions with
    gravity on and a free base so the locomotion policy can walk them in.

    ``spawn_at_manip`` spawns the G1s directly at their bedside marks instead of the
    approach positions (the ``--no-walk`` debug shortcut) — a legitimate start pose, not
    a mid-sim teleport."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    def walker(idx: int):
        g = g1_cfg.copy()
        # Free base + gravity on so the policy can move the root (the Inspire USD
        # ships fixed + gravity-off for manipulation; demo.py also deactivates its
        # baked root_joint before reset — see locomotion.make_floating_base).
        g.spawn.rigid_props.disable_gravity = False
        g.spawn.articulation_props.fix_root_link = False
        # The velocity-walk policy is a WHOLE-BODY controller trained with specific PD
        # gains on ALL 29 joints (its deploy.yaml). The stock G1_29DOF/Inspire config is
        # far stiffer for manipulation (waist kp 5000, arms kp 3000) — fed those gains the
        # locomotion policy can't balance and collapses (NVIDIA forums: gain
        # misconfiguration → falling). So set the EXACT trained gains on legs, feet, waist
        # and arms (from policies/g1_velocity_walk.deploy.yaml); only the Inspire "hands"
        # group is left as-is. (demo.py re-stiffens waist+arms at the plant for the firm
        # manipulation lean — see run_bedmaking; the legs are kinematically frozen by then.)
        g.actuators = dict(g.actuators)
        g.actuators["legs"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint",
                              ".*_hip_yaw_joint", ".*_knee_joint"],
            stiffness={".*_hip_.*_joint": 100.0, ".*_knee_joint": 150.0},
            damping={".*_hip_.*_joint": 2.0, ".*_knee_joint": 4.0},
            armature=0.03, effort_limit_sim=300.0, velocity_limit_sim=100.0)
        g.actuators["feet"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0, damping=2.0, armature=0.03,
            effort_limit_sim=100.0, velocity_limit_sim=100.0)
        g.actuators["waist"] = ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            stiffness=200.0, damping=5.0, armature=0.001,
            effort_limit_sim=300.0, velocity_limit_sim=100.0)
        g.actuators["arms"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
            stiffness=40.0, damping=10.0, armature=0.001,
            effort_limit_sim=300.0, velocity_limit_sim=100.0)
        if spawn_at_manip:
            mx, my = ROBOTS[idx]["manip"]
            spawn = (mx, my, ROBOT_Z)
        else:
            spawn = ROBOTS[idx]["approach"]
        g.init_state = g.init_state.replace(
            pos=spawn,
            rot=yaw_to_quat(ROBOTS[idx]["yaw_deg"]),
            joint_pos=dict(WALK_DEFAULT_JOINTS),
        )
        return g.replace(prim_path=f"/World/envs/env_0/Robot_{idx}")

    def static_box(size, center, color, friction=1.0, rounded=False, rot=None):
        # A rest/contact offset INFLATES + ROUNDS the collision corners and leaves a
        # small air gap, so the particle cloth drapes over a rounded edge instead of
        # catching on a sharp 90° corner (which pokes through the sheet — the "mattress
        # edge clips the bedsheet" bug) and so it reliably catches thin colliders like
        # the pillows instead of tunnelling through them. PhysX requires contact > rest.
        # (NVIDIA forums: tune contact/rest offset on both the cloth and the colliders.)
        # ``rot`` (w,x,y,z) tips the collider to match a propped visual (e.g. the pillows).
        cp = (sim_utils.CollisionPropertiesCfg(
                  contact_offset=COLLIDER_CONTACT_OFFSET, rest_offset=COLLIDER_REST_OFFSET)
              if rounded else sim_utils.CollisionPropertiesCfg())
        init = (AssetBaseCfg.InitialStateCfg(pos=center, rot=rot) if rot is not None
                else AssetBaseCfg.InitialStateCfg(pos=center))
        return AssetBaseCfg(
            spawn=sim_utils.CuboidCfg(
                size=size,
                collision_props=cp,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=friction, dynamic_friction=friction, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            init_state=init,
        )

    @configclass
    class BedMakingSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        dome = AssetBaseCfg(prim_path="/World/light",
                            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.95)))
        key = AssetBaseCfg(prim_path="/World/key",
                           spawn=sim_utils.DistantLightCfg(intensity=2000.0, angle=2.0))
        # Mattress — LOW friction so the robots can actually drag the cover across it.
        # A high-friction bed anchors the cover, and the drag force then exceeds the
        # hand's friction grip and tears the cover out of it (the cover "slips"). With a
        # slick mattress the cover slides headward under even a marginal grip. (Trade-off:
        # too slick and the foot-draped cover slides off on its own — 0.4 holds it.)
        bed = static_box(BED_SIZE, BED_CENTER, (0.42, 0.30, 0.22), friction=0.4, rounded=True).replace(
            prim_path="/World/Bed")
        headboard = static_box(HEADBOARD_SIZE, HEADBOARD_CENTER, (0.35, 0.24, 0.17),
                               rounded=True).replace(prim_path="/World/Headboard")
        # Pillows are rounded colliders (tipped up to match the propped visual) so the cover
        # drapes against them instead of passing through. The tip is about +Y (roty_to_quat).
        pillow_l = static_box(PILLOW_SIZE, PILLOWS["left"], (0.93, 0.93, 0.96),
                              rounded=True, rot=roty_to_quat(PILLOW_PROP_DEG)).replace(prim_path="/World/PillowL")
        pillow_r = static_box(PILLOW_SIZE, PILLOWS["right"], (0.93, 0.93, 0.96),
                              rounded=True, rot=roty_to_quat(PILLOW_PROP_DEG)).replace(prim_path="/World/PillowR")
        robot_0 = walker(0)
        robot_1 = walker(1)

    return BedMakingSceneCfg
