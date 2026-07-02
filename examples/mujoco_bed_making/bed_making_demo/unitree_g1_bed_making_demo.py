#!/usr/bin/env python3
"""Unitree G1 bed-making robotic system in MuJoCo.

The demo builds a scene with:
- a control G1 task planner,
- a worker G1 assisting as an effector,
- a 2.0 m x 1.8 m x 0.5 m bed object,
- a triangulated bed mesh,
- a MuJoCo flex-grid bedsheet dropped on the floor at the foot of the bed.

Physical realism choices:
- Robot pelvises are driven by mocap bodies connected via stiff weld
  equalities, so the floating bases do not fight dynamics and do not warp
  through the bed. Limb joints are still PD-actuated.
- Bedsheet corners are grasped by toggling per-corner weld equalities
  between the right hand wrist of the relevant robot and the corresponding
  flex node body. The sheet's physics handles collision with the bed, and
  the placement is intentionally imperfect after the welds are released.
- Walking routes never traverse the bed footprint.

Usage:
    .venv/bin/python examples/mujoco_bed_making/bed_making_demo/unitree_g1_bed_making_demo.py
    .venv/bin/python examples/mujoco_bed_making/bed_making_demo/unitree_g1_bed_making_demo.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("STRANDS_ASSETS_DIR", str(REPO_ROOT / ".strands_robots" / "assets"))


def _maybe_reexec_with_mjpython() -> None:
    """MuJoCo passive viewer on macOS should run under mjpython."""

    if sys.platform != "darwin":
        return
    if not sys.stdin.isatty():
        return
    if "--dry-run" in sys.argv:
        return
    if "--no-viewer" in sys.argv and "--render-frames" not in sys.argv:
        return
    if os.path.basename(sys.executable) == "mjpython" or os.environ.get("MJPYTHON_BIN"):
        return
    if os.environ.get("STRANDS_MJPYTHON_REEXEC") == "1":
        return

    venv_mjpython = Path(sys.executable).with_name("mjpython")
    mjpython = str(venv_mjpython) if venv_mjpython.exists() else shutil.which("mjpython")
    if not mjpython:
        print(
            "Warning: MuJoCo viewer on macOS usually requires mjpython, but no mjpython executable was found. "
            "The simulation will continue without the automatic viewer handoff.",
            file=sys.stderr,
        )
        return

    env = dict(os.environ)
    env["STRANDS_MJPYTHON_REEXEC"] = "1"
    os.execve(mjpython, [mjpython, *sys.argv], env)


_maybe_reexec_with_mjpython()

from strands_robots.simulation import _configure_gl_backend  # noqa: E402
from strands_robots.simulation.model_registry import resolve_model  # noqa: E402
from strands_robots.tools.download_assets import download_robots  # noqa: E402

_configure_gl_backend()


BED_LENGTH_M = 2.0
BED_WIDTH_M = 1.8
BED_HEIGHT_M = 0.5
SHEET_LENGTH_M = 2.2
SHEET_WIDTH_M = 2.0
SHEET_GRID_X = 13
SHEET_GRID_Y = 11
SHEET_SPACING_X_M = SHEET_LENGTH_M / (SHEET_GRID_X - 1)
SHEET_SPACING_Y_M = SHEET_WIDTH_M / (SHEET_GRID_Y - 1)

CONTROL_PREFIX = "control_"
WORKER_PREFIX = "worker_"
PELVIS_HEIGHT_M = 0.793
BED_STAND_OFFSET_M = 0.55
SHEET_DROP_POS = (-(BED_LENGTH_M / 2.0 + 0.65), 0.0, 0.45)

# Right hand grasp anchor relative to the wrist_yaw body origin (palm front).
HAND_GRASP_OFFSET = (0.13, -0.005, 0.0)

SheetLayout = Dict[int, Tuple[float, float, float]]


@dataclass(frozen=True)
class MeshAssetPaths:
    bed_obj: Path
    bedsheet_obj: Path


@dataclass(frozen=True)
class RobotSceneSpec:
    role: str
    prefix: str
    position: Tuple[float, float, float]
    yaw_radians: float


@dataclass(frozen=True)
class SceneBuildResult:
    xml: str
    scene_path: Path
    mesh_assets: MeshAssetPaths
    robot_joint_names: Dict[str, List[str]]
    robot_actuator_names: Dict[str, List[str]]
    sheet_corner_nodes: Dict[str, int]
    bed_corner_targets: Dict[str, Tuple[float, float, float]]
    base_weld_names: Dict[str, str] = field(default_factory=dict)
    base_mocap_names: Dict[str, str] = field(default_factory=dict)
    # Per-corner kinematic anchor: a mocap body that, when its weld is
    # activated, pins the cloth corner to the mocap position. This decouples
    # cloth motion from the robot arm dynamics during a grasp.
    corner_anchor_mocaps: Dict[str, str] = field(default_factory=dict)
    corner_anchor_welds: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacementStep:
    sheet_corner: str
    bed_corner: str
    label: str
    ask_worker_after: bool = False


@dataclass(frozen=True)
class BasePose:
    x: float
    y: float
    z: float
    yaw: float


def default_robot_specs() -> Tuple[RobotSceneSpec, RobotSceneSpec]:
    side_offset = BED_WIDTH_M / 2.0 + BED_STAND_OFFSET_M
    return (
        RobotSceneSpec("control", CONTROL_PREFIX, (0.0, side_offset, 0.0), -math.pi / 2.0),
        RobotSceneSpec("worker", WORKER_PREFIX, (0.0, -side_offset, 0.0), math.pi / 2.0),
    )


def placement_plan() -> Tuple[PlacementStep, ...]:
    return (
        PlacementStep("sheet_foot_left", "bed_head_right", "control robot locates a loose sheet corner"),
        PlacementStep("sheet_foot_right", "bed_head_left", "control robot traces the sheet edge to the next corner"),
        PlacementStep(
            "sheet_head_right",
            "bed_foot_left",
            "worker robot holds the partly placed sheet while the controller continues",
            ask_worker_after=True,
        ),
        PlacementStep(
            "sheet_head_left",
            "bed_foot_right",
            "worker robot assists while the controller places the final corner",
            ask_worker_after=True,
        ),
    )


def sheet_corner_nodes() -> Dict[str, int]:
    return {
        "sheet_foot_left": _sheet_node_index(0, 0),
        "sheet_foot_right": _sheet_node_index(0, SHEET_GRID_Y - 1),
        "sheet_head_left": _sheet_node_index(SHEET_GRID_X - 1, 0),
        "sheet_head_right": _sheet_node_index(SHEET_GRID_X - 1, SHEET_GRID_Y - 1),
    }


def bed_corner_targets(z_lift: float = 0.055) -> Dict[str, Tuple[float, float, float]]:
    x = BED_LENGTH_M / 2.0
    y = BED_WIDTH_M / 2.0
    z = BED_HEIGHT_M + z_lift
    return {
        "bed_foot_left": (-x, -y, z),
        "bed_foot_right": (-x, y, z),
        "bed_head_left": (x, -y, z),
        "bed_head_right": (x, y, z),
    }


def write_mesh_assets(mesh_dir: Path) -> MeshAssetPaths:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    bed_obj = mesh_dir / "bed_2m_x_1p8m_x_0p5m.obj"
    bedsheet_obj = mesh_dir / "bedsheet_2p2m_x_2m_triangles.obj"
    _write_obj(bed_obj, *_bed_mesh())
    _write_obj(bedsheet_obj, *_bedsheet_mesh())
    return MeshAssetPaths(bed_obj=bed_obj, bedsheet_obj=bedsheet_obj)


def build_scene(
    *,
    g1_xml_path: Path,
    output_dir: Path,
    robot_specs: Sequence[RobotSceneSpec] | None = None,
) -> SceneBuildResult:
    g1_xml_path = g1_xml_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_assets = write_mesh_assets(output_dir / "meshes")
    robot_specs = tuple(robot_specs or default_robot_specs())

    g1_root = ET.parse(g1_xml_path).getroot()
    g1_body = _required(g1_root.find("worldbody"), "worldbody").find("body")
    if g1_body is None:
        raise ValueError(f"No root body found in {g1_xml_path}")

    base_joint_names = [node.attrib["name"] for node in g1_body.iter("joint") if "name" in node.attrib]
    base_actuator_names = [
        node.attrib["name"] for node in _required(g1_root.find("actuator"), "actuator") if "name" in node.attrib
    ]

    scene = ET.Element("mujoco", {"model": "unitree_g1_two_humanoids_bed_making"})
    ET.SubElement(
        scene,
        "compiler",
        {
            "angle": "radian",
            "autolimits": "true",
            "meshdir": str((g1_xml_path.parent / "assets").resolve()),
        },
    )
    ET.SubElement(
        scene,
        "option",
        {
            "timestep": "0.002",
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
            "cone": "elliptic",
            "impratio": "4",
        },
    )

    default = g1_root.find("default")
    if default is not None:
        scene.append(copy.deepcopy(default))

    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "960", "azimuth": "140", "elevation": "-25"})
    # Pulling the headlight forward + softening ambient keeps the bed/cloth
    # visible at the wide camera distance below.
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.55 0.55 0.55", "ambient": "0.25 0.25 0.25", "specular": "0.3 0.3 0.3"},
    )

    # The default MuJoCo viewer auto-fits to model statistics. With the cloth
    # flexcomp + tall G1 humanoids the auto-fit zooms in too close. Use a
    # large extent (≈10x the bed long edge) so the viewer opens with the
    # whole scene comfortably in frame.
    ET.SubElement(
        scene,
        "statistic",
        {"center": "0 0 1.0", "extent": "20.0"},
    )

    asset = copy.deepcopy(_required(g1_root.find("asset"), "asset"))
    ET.SubElement(asset, "material", {"name": "bed_fabric", "rgba": "0.48 0.42 0.36 1"})
    ET.SubElement(asset, "material", {"name": "bedsheet_blue", "rgba": "0.16 0.38 0.88 0.92"})
    ET.SubElement(asset, "material", {"name": "floor_matte", "rgba": "0.78 0.78 0.74 1"})
    ET.SubElement(asset, "mesh", {"name": "bed_mesh", "file": str(mesh_assets.bed_obj.resolve())})
    scene.append(asset)

    worldbody = ET.SubElement(scene, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"name": "key_light", "pos": "0 -2.5 3.5", "dir": "0 0 -1", "directional": "true"},
    )
    ET.SubElement(
        worldbody,
        "light",
        {"name": "fill_light", "pos": "-2 2 2.5", "dir": "1 -1 -1", "diffuse": "0.4 0.4 0.4"},
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": "-3.4 -3.4 2.4",
            "xyaxes": "0.74 -0.67 0 0.36 0.40 0.84",
            "fovy": "45",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "5 5 0.05",
            "material": "floor_matte",
            "condim": "3",
            "friction": "1 0.3 0.001",
        },
    )
    _add_bed(worldbody)
    _add_bedsheet_flex(worldbody)

    # Mocap bodies for each robot pelvis; the base weld below makes the pelvis
    # track the mocap pose without writing freejoint qpos during the run.
    base_mocap_names: Dict[str, str] = {}
    for spec in robot_specs:
        mocap_name = f"{spec.prefix}base_mocap"
        base_mocap_names[spec.role] = mocap_name
        ET.SubElement(
            worldbody,
            "body",
            {
                "name": mocap_name,
                "mocap": "true",
                "pos": _fmt((spec.position[0], spec.position[1], PELVIS_HEIGHT_M)),
                "quat": _fmt(_yaw_quat(spec.yaw_radians)),
            },
        )

    # One mocap "anchor" per sheet corner. The driver sets the mocap position
    # before activating the corresponding weld so the cloth corner is pinned to
    # the mocap without an initial yank, and then moves the mocap smoothly to
    # drive the cloth — independent of the robot arm dynamics.
    sheet_corners = sheet_corner_nodes()
    corner_anchor_mocaps: Dict[str, str] = {}
    for corner_name in sheet_corners:
        anchor_name = f"cloth_anchor_{corner_name}"
        corner_anchor_mocaps[corner_name] = anchor_name
        ET.SubElement(
            worldbody,
            "body",
            {
                "name": anchor_name,
                "mocap": "true",
                "pos": _fmt(SHEET_DROP_POS),
            },
        )

    robot_joint_names: Dict[str, List[str]] = {}
    robot_actuator_names: Dict[str, List[str]] = {}
    for spec in robot_specs:
        worldbody.append(_clone_robot_body(g1_body, spec))
        robot_joint_names[spec.role] = [f"{spec.prefix}{name}" for name in base_joint_names]
        robot_actuator_names[spec.role] = [f"{spec.prefix}{name}" for name in base_actuator_names]

    actuator = ET.SubElement(scene, "actuator")
    source_actuator = _required(g1_root.find("actuator"), "actuator")
    for spec in robot_specs:
        for actuator_node in source_actuator:
            copied = copy.deepcopy(actuator_node)
            _prefix_reference_attrs(copied, spec.prefix, {"name", "joint"})
            actuator.append(copied)

    equality = ET.SubElement(scene, "equality")
    base_weld_names: Dict[str, str] = {}
    corner_anchor_welds: Dict[str, str] = {}
    for spec in robot_specs:
        base_weld = f"{spec.prefix}base_weld"
        base_weld_names[spec.role] = base_weld
        ET.SubElement(
            equality,
            "weld",
            {
                "name": base_weld,
                "body1": base_mocap_names[spec.role],
                "body2": f"{spec.prefix}pelvis",
                "relpose": "0 0 0 1 0 0 0",
                "solref": "0.005 1",
                "solimp": "0.97 0.99 0.001",
            },
        )

    # Cloth-anchor welds: kinematic mocap ↔ cloth corner node. Inactive at
    # compile time. The driver enables them at grasp and writes the mocap
    # position to drive the corner along a planned trajectory.
    for corner_name, node_index in sheet_corners.items():
        weld_name = f"cloth_anchor_{corner_name}_weld"
        corner_anchor_welds[corner_name] = weld_name
        ET.SubElement(
            equality,
            "weld",
            {
                "name": weld_name,
                "body1": corner_anchor_mocaps[corner_name],
                "body2": f"bedsheet_{node_index}",
                "relpose": "0 0 0 1 0 0 0",
                "active": "false",
                "solref": "0.01 1",
                "solimp": "0.95 0.99 0.002",
            },
        )

    ET.indent(scene, space="  ")
    xml = ET.tostring(scene, encoding="unicode")
    scene_path = output_dir / "unitree_g1_bed_making_scene.xml"
    scene_path.write_text(xml)

    return SceneBuildResult(
        xml=xml,
        scene_path=scene_path,
        mesh_assets=mesh_assets,
        robot_joint_names=robot_joint_names,
        robot_actuator_names=robot_actuator_names,
        sheet_corner_nodes=sheet_corners,
        bed_corner_targets=bed_corner_targets(),
        base_weld_names=base_weld_names,
        base_mocap_names=base_mocap_names,
        corner_anchor_mocaps=corner_anchor_mocaps,
        corner_anchor_welds=corner_anchor_welds,
    )


# ---------------------------------------------------------------------------
# Pose targets
# ---------------------------------------------------------------------------


def pose_for_phase(prefix: str, side: str, phase: str) -> Dict[str, float]:
    active_sign = 1 if side == "left" else -1
    inactive_side = "right" if side == "left" else "left"
    inactive_sign = -active_sign
    waist_pitch = {
        "home": 0.02,
        "walk": 0.06,
        "reach": 0.34,
        "grasp": 0.42,
        "carry": 0.22,
        "trace": 0.28,
        "place": 0.38,
        "hold": 0.30,
    }.get(phase, 0.02)
    pose: Dict[str, float] = {
        f"{prefix}waist_yaw_joint": 0.10 * active_sign if phase in {"reach", "grasp", "place", "trace"} else 0.0,
        f"{prefix}waist_roll_joint": 0.04 * active_sign if phase in {"reach", "grasp", "place"} else 0.0,
        f"{prefix}waist_pitch_joint": waist_pitch,
    }
    _apply_leg_pose(pose, prefix, phase)
    _apply_arm_pose(pose, prefix, side, active_sign, phase)
    _apply_arm_pose(pose, prefix, inactive_side, inactive_sign, "balance" if phase == "walk" else "home")
    return pose


def worker_hold_pose(prefix: str = WORKER_PREFIX) -> Dict[str, float]:
    pose = {
        f"{prefix}waist_yaw_joint": 0.0,
        f"{prefix}waist_roll_joint": 0.0,
        f"{prefix}waist_pitch_joint": 0.28,
    }
    _apply_leg_pose(pose, prefix, "hold")
    _apply_arm_pose(pose, prefix, "left", 1, "hold")
    _apply_arm_pose(pose, prefix, "right", -1, "hold")
    return pose


def home_pose(prefix: str) -> Dict[str, float]:
    pose = {
        f"{prefix}waist_yaw_joint": 0.0,
        f"{prefix}waist_roll_joint": 0.0,
        f"{prefix}waist_pitch_joint": 0.02,
    }
    _apply_leg_pose(pose, prefix, "home")
    _apply_arm_pose(pose, prefix, "left", 1, "home")
    _apply_arm_pose(pose, prefix, "right", -1, "home")
    return pose


def blend_pose(start: Dict[str, float], target: Dict[str, float], alpha: float) -> Dict[str, float]:
    keys = set(start) | set(target)
    return {key: start.get(key, 0.0) + (target.get(key, 0.0) - start.get(key, 0.0)) * alpha for key in keys}


def default_base_pose(role: str) -> BasePose:
    for spec in default_robot_specs():
        if spec.role == role:
            return BasePose(spec.position[0], spec.position[1], PELVIS_HEIGHT_M, spec.yaw_radians)
    raise ValueError(f"Unknown robot role: {role}")


def base_pose_for_corner(role: str, bed_corner: str) -> BasePose:
    target = bed_corner_targets()[bed_corner]
    side_y = math.copysign(BED_WIDTH_M / 2.0 + BED_STAND_OFFSET_M, target[1])
    stand_x = max(-(BED_LENGTH_M / 2.0 - 0.05), min(BED_LENGTH_M / 2.0 - 0.05, target[0]))
    if role == "worker":
        side_y = -side_y
    yaw = -math.pi / 2.0 if side_y > 0 else math.pi / 2.0
    return BasePose(stand_x, side_y, PELVIS_HEIGHT_M, yaw)


def base_pose_for_sheet_pickup(role: str) -> BasePose:
    """Stand beside the dropped sheet, just past the foot of the bed."""

    side_y = math.copysign(BED_WIDTH_M / 2.0 + BED_STAND_OFFSET_M, 1.0)
    if role == "worker":
        side_y = -side_y
    stand_x = SHEET_DROP_POS[0] + 0.1
    yaw = -math.pi / 2.0 if side_y > 0 else math.pi / 2.0
    return BasePose(stand_x, side_y, PELVIS_HEIGHT_M, yaw)


def waypoints_around_bed(
    start: BasePose, target: BasePose, role: str
) -> List[BasePose]:
    """Return intermediate waypoints between two stations that skirt the bed.

    Two stations on the same long side of the bed connect directly. Stations
    on opposite long sides detour around the foot end so the walk path never
    crosses the bed footprint.
    """

    same_side = math.copysign(1.0, start.y) == math.copysign(1.0, target.y)
    if same_side:
        return [target]
    foot_x = -(BED_LENGTH_M / 2.0 + BED_STAND_OFFSET_M + 0.05)
    waypoint_yaw = math.atan2(target.y - start.y, target.x - start.x)
    return [
        BasePose(foot_x, start.y, PELVIS_HEIGHT_M, waypoint_yaw),
        BasePose(foot_x, target.y, PELVIS_HEIGHT_M, target.yaw),
        target,
    ]


def blend_base(start: BasePose, target: BasePose, alpha: float) -> BasePose:
    yaw_delta = math.atan2(math.sin(target.yaw - start.yaw), math.cos(target.yaw - start.yaw))
    return BasePose(
        start.x + (target.x - start.x) * alpha,
        start.y + (target.y - start.y) * alpha,
        start.z + (target.z - start.z) * alpha,
        start.yaw + yaw_delta * alpha,
    )


def crumpled_sheet_layout(seed: int) -> SheetLayout:
    rng = random.Random(seed)
    layout: SheetLayout = {}
    for ix, iy, node in _sheet_nodes():
        u = ix / (SHEET_GRID_X - 1)
        v = iy / (SHEET_GRID_Y - 1)
        fold_a = math.sin(5.0 * math.pi * u + 1.7 * math.sin(3.0 * math.pi * v))
        fold_b = math.sin(4.0 * math.pi * v + 0.8 * math.cos(2.0 * math.pi * u))
        x = SHEET_DROP_POS[0] - 0.55 + 0.65 * u + 0.08 * fold_b + rng.uniform(-0.018, 0.018)
        y = -0.35 + 0.70 * v + 0.10 * fold_a + rng.uniform(-0.018, 0.018)
        z = 0.45 + 0.20 * abs(fold_a) + 0.08 * abs(fold_b) + rng.uniform(-0.02, 0.035)
        layout[node] = (x, y, z)
    return layout


def final_sheet_layout(seed: int) -> SheetLayout:
    rng = random.Random(seed + 31)
    origin = (1.02, 0.87, BED_HEIGHT_M + 0.045)
    u_axis = (-SHEET_LENGTH_M * 0.985, -0.08, 0.0)
    v_axis = (-0.05, -SHEET_WIDTH_M * 0.965, 0.0)
    layout: SheetLayout = {}
    for ix, iy, node in _sheet_nodes():
        u = ix / (SHEET_GRID_X - 1)
        v = iy / (SHEET_GRID_Y - 1)
        wrinkle = 0.024 * math.sin(4.0 * math.pi * u + 0.7) * math.sin(3.0 * math.pi * v + 0.2)
        edge_sag = 0.0
        if ix in {0, SHEET_GRID_X - 1} or iy in {0, SHEET_GRID_Y - 1}:
            edge_sag = -0.035 * (0.45 + rng.random() * 0.55)
        x = origin[0] + u_axis[0] * u + v_axis[0] * v + 0.015 * math.sin(5.0 * math.pi * v)
        y = origin[1] + u_axis[1] * u + v_axis[1] * v + 0.018 * math.sin(4.0 * math.pi * u)
        z = origin[2] + wrinkle + edge_sag
        layout[node] = (x, y, max(BED_HEIGHT_M + 0.012, z))
    return layout


def _apply_leg_pose(pose: Dict[str, float], prefix: str, phase: str) -> None:
    # The pelvis is mocap-welded, so dramatic leg poses fight floor contact and
    # destabilise the solver. Use only small leg motions for visual variety;
    # the bend toward the bed/floor is taken by the waist pitch and arms.
    hip, knee, ankle = {
        "home": (0.00, 0.10, -0.05),
        "walk": (0.05, 0.14, -0.07),
        "reach": (0.10, 0.20, -0.10),
        "grasp": (0.12, 0.24, -0.12),
        "carry": (0.06, 0.16, -0.08),
        "trace": (0.10, 0.18, -0.09),
        "place": (0.12, 0.22, -0.11),
        "hold": (0.10, 0.20, -0.10),
    }.get(phase, (0.0, 0.10, -0.05))
    for side, sign in (("left", 1), ("right", -1)):
        pose[f"{prefix}{side}_hip_pitch_joint"] = hip
        pose[f"{prefix}{side}_hip_roll_joint"] = 0.0
        pose[f"{prefix}{side}_hip_yaw_joint"] = 0.0
        pose[f"{prefix}{side}_knee_joint"] = knee
        pose[f"{prefix}{side}_ankle_pitch_joint"] = ankle
        pose[f"{prefix}{side}_ankle_roll_joint"] = 0.0


def _apply_gait_pose(pose: Dict[str, float], prefix: str, gait_phase: float, intensity: float) -> None:
    """Visible step cycle so the legs lift/plant as the mocap base glides.

    Each leg follows a sinusoid offset 180° from the other: when one hip
    pitches forward and the knee bends (foot lifts), the other extends
    (foot plants). Amplitude is proportional to ``intensity`` so transitions
    in and out of the walk look natural.
    """

    if intensity <= 0.0:
        return
    for side, sign in (("left", 1), ("right", -1)):
        phase = gait_phase if side == "left" else gait_phase + math.pi
        swing = math.sin(phase)
        lift = max(0.0, swing)
        # Hip pitches forward when the foot is in swing phase (lift > 0) and
        # backward as the foot pushes off (swing < 0). Knee bends only on
        # lift so the foot clears the ground.
        pose[f"{prefix}{side}_hip_pitch_joint"] += 0.55 * intensity * swing
        pose[f"{prefix}{side}_knee_joint"] += 0.95 * intensity * lift
        pose[f"{prefix}{side}_ankle_pitch_joint"] -= 0.35 * intensity * swing
        # Slight hip roll so the body weight shifts onto the planted foot.
        pose[f"{prefix}{side}_hip_roll_joint"] = pose.get(
            f"{prefix}{side}_hip_roll_joint", 0.0
        ) + 0.08 * intensity * sign * math.cos(phase)


def _apply_arm_pose(pose: Dict[str, float], prefix: str, side: str, sign: int, phase: str) -> None:
    # Tuples: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
    #         wrist_roll, wrist_pitch, wrist_yaw, hand_closed.
    # Note: shoulder_pitch axis is +y; for the G1, positive pitch rotates the
    # arm downward (positive angle ≈ arm pointing down/forward).
    values = {
        # Match the G1 "stand" keyframe so the robot arms hang naturally at rest.
        "home": (0.20, 0.20, 0.0, 1.28, 0.0, 0.0, 0.0, False),
        # Counterbalance arm swung slightly forward while walking.
        "balance": (0.28, 0.16, 0.0, 1.10, 0.0, -0.08, 0.04, False),
        # Reach forward-down toward the sheet on the floor; arm extends down and
        # the wrist tucks slightly inward.
        "reach": (1.35, 0.18, 0.0, 0.45, 0.0, -0.10, 0.12, False),
        # Grasp: arm down and slightly forward, elbow partly bent, hand closed.
        "grasp": (1.30, 0.20, 0.0, 0.55, 0.0, -0.15, 0.16, True),
        # Carrying with hand closed at hip/waist height.
        "carry": (0.85, 0.18, 0.05, 0.95, 0.0, -0.12, 0.10, True),
        # Tracing along the sheet edge over the bed.
        "trace": (1.05, 0.20, 0.05, 0.70, 0.05, -0.18, 0.14, True),
        # Place corner over a bed corner: hand at the bed-top height in front
        # of the robot.
        "place": (1.10, 0.22, 0.08, 0.55, 0.05, -0.20, 0.18, True),
        # Holding a corner steady, arm extended toward bed.
        "hold": (1.05, 0.30, 0.08, 0.72, 0.0, -0.18, 0.14, True),
    }.get(phase, (0.20, 0.20, 0.0, 1.28, 0.0, 0.0, 0.0, False))
    shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw, closed = values
    side_prefix = f"{prefix}{side}_"
    pose[f"{side_prefix}shoulder_pitch_joint"] = shoulder_pitch
    pose[f"{side_prefix}shoulder_roll_joint"] = shoulder_roll * sign
    pose[f"{side_prefix}shoulder_yaw_joint"] = shoulder_yaw * sign
    pose[f"{side_prefix}elbow_joint"] = elbow
    pose[f"{side_prefix}wrist_roll_joint"] = wrist_roll * sign
    pose[f"{side_prefix}wrist_pitch_joint"] = wrist_pitch
    pose[f"{side_prefix}wrist_yaw_joint"] = wrist_yaw * sign
    _apply_hand_pose(pose, side_prefix, side, closed)


def _apply_hand_pose(pose: Dict[str, float], side_prefix: str, side: str, closed: bool) -> None:
    if not closed:
        curl_index = 0.0
        curl_middle = 0.0
        thumb0 = 0.20 if side == "left" else -0.20
        thumb1 = 0.0
        thumb2 = 0.0
    elif side == "left":
        # Joint signs for the left hand: index/middle 0/1 have range that closes negative.
        curl_index = -1.40
        curl_middle = -1.40
        thumb0 = 0.95
        thumb1 = 0.72
        thumb2 = 1.60
    else:
        # Right hand mirror signs (range opens positive).
        curl_index = 1.40
        curl_middle = 1.40
        thumb0 = -0.95
        thumb1 = -0.72
        thumb2 = -1.60
    pose[f"{side_prefix}hand_thumb_0_joint"] = thumb0
    pose[f"{side_prefix}hand_thumb_1_joint"] = thumb1
    pose[f"{side_prefix}hand_thumb_2_joint"] = thumb2
    pose[f"{side_prefix}hand_index_0_joint"] = curl_index
    pose[f"{side_prefix}hand_index_1_joint"] = curl_index * 0.85
    pose[f"{side_prefix}hand_middle_0_joint"] = curl_middle
    pose[f"{side_prefix}hand_middle_1_joint"] = curl_middle * 0.85


# ---------------------------------------------------------------------------
# Mesh emission and scene helpers
# ---------------------------------------------------------------------------


def _bed_mesh() -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    x = BED_LENGTH_M / 2.0
    y = BED_WIDTH_M / 2.0
    z = BED_HEIGHT_M / 2.0
    vertices = [
        (-x, -y, -z),
        (x, -y, -z),
        (x, y, -z),
        (-x, y, -z),
        (-x, -y, z),
        (x, -y, z),
        (x, y, z),
        (-x, y, z),
    ]
    triangles = [
        (1, 2, 3),
        (1, 3, 4),
        (5, 8, 7),
        (5, 7, 6),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 8),
        (3, 8, 4),
        (4, 8, 5),
        (4, 5, 1),
    ]
    return vertices, triangles


def _bedsheet_mesh() -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    vertices: List[Tuple[float, float, float]] = []
    for ix in range(SHEET_GRID_X):
        x = -SHEET_LENGTH_M / 2.0 + ix * SHEET_SPACING_X_M
        for iy in range(SHEET_GRID_Y):
            y = -SHEET_WIDTH_M / 2.0 + iy * SHEET_SPACING_Y_M
            vertices.append((x, y, 0.0))

    triangles: List[Tuple[int, int, int]] = []
    for ix in range(SHEET_GRID_X - 1):
        for iy in range(SHEET_GRID_Y - 1):
            a = ix * SHEET_GRID_Y + iy + 1
            b = (ix + 1) * SHEET_GRID_Y + iy + 1
            c = (ix + 1) * SHEET_GRID_Y + iy + 2
            d = ix * SHEET_GRID_Y + iy + 2
            triangles.append((a, b, c))
            triangles.append((a, c, d))
    return vertices, triangles


def _write_obj(
    path: Path,
    vertices: Iterable[Tuple[float, float, float]],
    triangles: Iterable[Tuple[int, int, int]],
) -> None:
    lines = [f"# Generated for Strands Robots Unitree G1 bed-making demo: {path.name}"]
    for vertex in vertices:
        lines.append("v " + _fmt(vertex))
    for tri in triangles:
        lines.append(f"f {tri[0]} {tri[1]} {tri[2]}")
    path.write_text("\n".join(lines) + "\n")


def _add_bed(worldbody: ET.Element) -> None:
    body = ET.SubElement(worldbody, "body", {"name": "bed", "pos": "0 0 0.25"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": "bed_collision",
            "type": "box",
            "size": f"{BED_LENGTH_M / 2.0} {BED_WIDTH_M / 2.0} {BED_HEIGHT_M / 2.0}",
            "rgba": "0.48 0.42 0.36 1",
            "condim": "6",
            # High tangential + torsional friction so the cloth grips the bed
            # top and does not slide off when corners are tugged.
            "friction": "3.0 1.5 0.01",
            "solref": "0.02 1",
            "solimp": "0.9 0.95 0.005",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "bed_mesh_visual",
            "type": "mesh",
            "mesh": "bed_mesh",
            "material": "bed_fabric",
            "contype": "0",
            "conaffinity": "0",
        },
    )


def _add_bedsheet_flex(worldbody: ET.Element) -> None:
    flex = ET.SubElement(
        worldbody,
        "flexcomp",
        {
            "name": "bedsheet",
            "type": "grid",
            "count": f"{SHEET_GRID_X} {SHEET_GRID_Y} 1",
            "spacing": f"{SHEET_SPACING_X_M:.6f} {SHEET_SPACING_Y_M:.6f} 0.05",
            "pos": _fmt(SHEET_DROP_POS),
            "dim": "2",
            "mass": "0.18",
            "radius": "0.022",
            "rgba": "0.16 0.38 0.88 0.92",
        },
    )
    ET.SubElement(
        flex,
        "contact",
        {
            "contype": "1",
            "conaffinity": "1",
            "condim": "6",
            # High friction on the cloth surface so it grips the bed top and
            # does not slide off when a corner is pulled.
            "friction": "3.0 1.0 0.01",
            "solref": "0.01 1",
            "solimp": "0.9 0.97 0.005",
            "selfcollide": "none",
            "internal": "false",
        },
    )
    # Edge equality with significant damping keeps the sheet nearly
    # inextensible while crumpling/draping; heavy damping reduces the high
    # constraint forces that appear when one corner is mocap-anchored and
    # the rest of the cloth is being dragged along.
    ET.SubElement(flex, "edge", {"equality": "true", "damping": "0.6"})


def _clone_robot_body(source_body: ET.Element, spec: RobotSceneSpec) -> ET.Element:
    body = copy.deepcopy(source_body)
    source_pos = _parse_floats(body.attrib.get("pos", "0 0 0"), expected=3)
    placed_pos = (
        spec.position[0] + source_pos[0],
        spec.position[1] + source_pos[1],
        PELVIS_HEIGHT_M,
    )
    body.set("pos", _fmt(placed_pos))
    body.set("quat", _fmt(_yaw_quat(spec.yaw_radians)))
    _prefix_reference_attrs(body, spec.prefix, {"name"})
    return body


def _prefix_reference_attrs(node: ET.Element, prefix: str, attrs: set[str]) -> None:
    for elem in node.iter():
        for attr in attrs:
            if attr in elem.attrib:
                elem.set(attr, f"{prefix}{elem.attrib[attr]}")


def _required(value: ET.Element | None, name: str) -> ET.Element:
    if value is None:
        raise ValueError(f"Required MJCF element missing: {name}")
    return value


def _sheet_node_index(ix: int, iy: int) -> int:
    return ix * SHEET_GRID_Y + iy


def _sheet_node_grid(node: int) -> Tuple[int, int]:
    return divmod(node, SHEET_GRID_Y)


def _sheet_nodes():
    for ix in range(SHEET_GRID_X):
        for iy in range(SHEET_GRID_Y):
            yield ix, iy, _sheet_node_index(ix, iy)


def _yaw_quat(yaw: float) -> Tuple[float, float, float, float]:
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _parse_floats(value: str, expected: int) -> Tuple[float, ...]:
    parsed = tuple(float(item) for item in value.split())
    if len(parsed) != expected:
        raise ValueError(f"Expected {expected} floats, got {value!r}")
    return parsed


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


# ---------------------------------------------------------------------------
# Device Connect-style coordination layer
# ---------------------------------------------------------------------------


class _ClothAnchorRegistry:
    """Shared registry of per-corner cloth anchor mocaps and welds.

    Both the control planner and the worker effector route their grasps
    through this registry: ``acquire`` snaps the mocap onto the current cloth
    corner position before activating the weld (so there is no yank), and
    ``update`` writes the mocap position each frame to drive the corner.
    Only one role can hold a corner at a time.

    The driver can either ``follow_hand`` (anchor tracks a robot wrist body
    each frame so the cloth stays visually in the hand) or write an arbitrary
    target via ``update`` (for the final placement onto a bed corner).
    """

    def __init__(self, mujoco, model, data, scene: SceneBuildResult) -> None:
        self._mujoco = mujoco
        self._model = model
        self._data = data
        self._scene = scene
        self._anchor_mocap_ids: Dict[str, int] = {}
        for corner, body_name in scene.corner_anchor_mocaps.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise RuntimeError(f"Missing cloth anchor body {body_name}")
            self._anchor_mocap_ids[corner] = int(model.body_mocapid[body_id])
        self._anchor_weld_ids: Dict[str, int] = {}
        for corner, weld_name in scene.corner_anchor_welds.items():
            eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
            if eq_id < 0:
                raise RuntimeError(f"Missing cloth anchor weld {weld_name}")
            self._anchor_weld_ids[corner] = eq_id
        self._owner: Dict[str, Optional[str]] = {corner: None for corner in scene.corner_anchor_mocaps}

    def held_by(self, corner: str) -> Optional[str]:
        return self._owner.get(corner)

    def acquire(self, corner: str, role: str) -> Tuple[float, float, float]:
        if corner not in self._anchor_mocap_ids:
            raise ValueError(f"Unknown sheet corner {corner}")
        if self._owner[corner] not in (None, role):
            raise RuntimeError(f"Corner {corner} already held by {self._owner[corner]}")
        node_index = self._scene.sheet_corner_nodes[corner]
        body_id = self._mujoco.mj_name2id(
            self._model, self._mujoco.mjtObj.mjOBJ_BODY, f"bedsheet_{node_index}"
        )
        corner_pos = tuple(float(v) for v in self._data.xpos[body_id])
        # Snap the anchor mocap to the corner so activating the weld is a no-op.
        self._data.mocap_pos[self._anchor_mocap_ids[corner]] = corner_pos
        self._data.eq_active[self._anchor_weld_ids[corner]] = 1
        self._owner[corner] = role
        return corner_pos

    def update(self, corner: str, role: str, target: Tuple[float, float, float]) -> None:
        if self._owner.get(corner) != role:
            return
        self._data.mocap_pos[self._anchor_mocap_ids[corner]] = target

    def follow_hand(self, corner: str, role: str, hand_body_name: str) -> Tuple[float, float, float]:
        """Snap the anchor mocap to the current world position of the hand body.

        Returns the world position written. Caller calls this every frame
        during carry/lift phases so the cloth corner visually stays in the
        robot's hand.
        """

        if self._owner.get(corner) != role:
            return (0.0, 0.0, 0.0)
        body_id = self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_BODY, hand_body_name)
        if body_id < 0:
            raise RuntimeError(f"Missing hand body {hand_body_name}")
        pos = tuple(float(v) for v in self._data.xpos[body_id])
        self._data.mocap_pos[self._anchor_mocap_ids[corner]] = pos
        return pos

    def release(self, corner: str, role: str) -> None:
        if self._owner.get(corner) != role:
            return
        self._data.eq_active[self._anchor_weld_ids[corner]] = 0
        self._owner[corner] = None

    def position(self, corner: str) -> Tuple[float, float, float]:
        return tuple(float(v) for v in self._data.mocap_pos[self._anchor_mocap_ids[corner]])


def _resolve_dc_drivers(bg_runtime) -> Tuple[Any, Any]:
    """Return the two swarm-peer drivers from a BackgroundRuntime, or (None, None).

    The peers are equal, so this is role-agnostic: it prefers the legacy
    ``control``/``worker`` keys if present, then falls back to the
    ``peer-0``/``peer-1`` aliases, then to the first two distinct
    ``DeviceRuntime`` instances registered (keyed by device_id). The first
    returned driver is the one the demo's planner emits from; the second is
    the assisting peer.
    """

    if bg_runtime is None:
        return None, None
    runtimes = getattr(bg_runtime, "runtimes", None) or {}

    def _driver(runtime):
        if runtime is None:
            return None
        # DeviceRuntime stores driver as _driver in current SDK.
        return getattr(runtime, "_driver", None)

    for first_key, second_key in (("control", "worker"), ("peer-0", "peer-1")):
        first, second = runtimes.get(first_key), runtimes.get(second_key)
        if first is not None or second is not None:
            return _driver(first), _driver(second)

    # Fall back to the first two distinct runtimes (de-duplicated, since the
    # BackgroundRuntime stores each runtime under both its device_id and a
    # role alias).
    distinct: List[Any] = []
    for runtime in runtimes.values():
        if runtime is not None and runtime not in distinct:
            distinct.append(runtime)
    first = _driver(distinct[0]) if len(distinct) >= 1 else None
    second = _driver(distinct[1]) if len(distinct) >= 2 else None
    return first, second


def _emit_dc_event(dc_driver, bg_runtime, event_name: str, **payload) -> None:
    """Schedule a DC event emit from the simulation (main) thread.

    Routes through the background asyncio loop where the DC runtime lives.
    All inter-peer coordination in the bed-making swarm flows through these
    NATS-delivered events; the dashboard sees every transition.
    """

    if dc_driver is None or bg_runtime is None:
        return
    loop = getattr(bg_runtime, "_loop", None)
    if loop is None:
        return
    method = getattr(dc_driver, event_name, None)
    if method is None:
        return
    try:
        import asyncio

        coro = method(**payload)
        asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception:
        # The dashboard heartbeat thread can lag during shutdown; never let
        # an emit failure crash the simulation main loop.
        pass


class WorkerEffector:
    """Worker robot side of the Device Connect split.

    Wraps the worker's mocap base and per-corner cloth anchors behind
    RPC-style methods (``holdCorner``, ``releaseCorner``, ``assistPlace``,
    ``setIdle``, ``getStatus``). The control planner calls into this object.
    In a multi-process Device Connect deployment these would be remote RPCs
    against a separate ``device-connect-edge`` participant.
    """

    def __init__(
        self,
        *,
        mujoco,
        model,
        data,
        scene: SceneBuildResult,
        anchors: "_ClothAnchorRegistry | None" = None,
        prefix: str = WORKER_PREFIX,
        role: str = "worker",
        idle_pose_fn: Callable[[], Dict[str, float]] | None = None,
        hold_pose_fn: Callable[[], Dict[str, float]] | None = None,
        dc_driver: Any = None,
        dc_runtime: Any = None,
    ) -> None:
        self._mujoco = mujoco
        self._model = model
        self._data = data
        self._scene = scene
        self._prefix = prefix
        self._role = role
        self._idle_pose_fn = idle_pose_fn or (lambda: home_pose(prefix))
        self._hold_pose_fn = hold_pose_fn or (lambda: worker_hold_pose(prefix))
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene.base_mocap_names[role])
        if body_id < 0:
            raise RuntimeError(f"Missing mocap body {scene.base_mocap_names[role]}")
        self._mocap_id = int(model.body_mocapid[body_id])
        self._anchors = anchors if anchors is not None else _ClothAnchorRegistry(
            mujoco, model, data, scene
        )
        self._dc_driver = dc_driver
        self._dc_runtime = dc_runtime
        self.status: str = "online"
        self.held_corner: Optional[str] = None
        self.target_base: BasePose = default_base_pose(role)
        self.pose: Dict[str, float] = self._idle_pose_fn()

    # RPC-style methods. Naming uses camelCase to match Device Connect schema.
    # Every state change also fires a DC event so the swarm dashboard reflects
    # what this peer is doing in real time.

    def setIdle(self) -> dict:
        self.status = "online"
        self.target_base = default_base_pose(self._role)
        self.pose = self._idle_pose_fn()
        _emit_dc_event(self._dc_driver, self._dc_runtime, "heartbeat", status="online", holding="")
        return {"status": self.status}

    def holdCorner(self, corner: str, station: BasePose) -> dict:
        pos = self._anchors.acquire(corner, self._role)
        self.status = "holding"
        self.held_corner = corner
        self.target_base = station
        self.pose = self._hold_pose_fn()
        _emit_dc_event(
            self._dc_driver, self._dc_runtime, "cornerHeld",
            corner=corner, x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
        )
        return {"status": self.status, "corner": corner}

    def releaseCorner(self) -> dict:
        released = self.held_corner
        if released is not None:
            self._anchors.release(released, self._role)
            _emit_dc_event(self._dc_driver, self._dc_runtime, "cornerReleased", corner=released)
        self.held_corner = None
        self.status = "online"
        return {"released": released}

    def assistPlace(self, corner: str, station: BasePose) -> dict:
        return self.holdCorner(corner, station)

    def getStatus(self) -> dict:
        return {"status": self.status, "corner": self.held_corner}

    # Driver-side helpers ---------------------------------------------------

    def apply(self, gait_phase: float, gait_intensity: float) -> Dict[str, float]:
        pose = dict(self.pose)
        if gait_intensity > 0.0:
            _apply_gait_pose(pose, self._prefix, gait_phase, gait_intensity)
        return pose

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def mocap_id(self) -> int:
        return self._mocap_id

    @property
    def anchors(self) -> "_ClothAnchorRegistry":
        return self._anchors


class ControlPlanner:
    """Control robot side of the Device Connect split.

    Owns the high-level bed-making plan and drives its own mocap base. Grasps
    are routed through the shared :class:`_ClothAnchorRegistry`, and worker
    actions are invoked via the :class:`WorkerEffector` RPC surface.
    """

    def __init__(
        self,
        *,
        mujoco,
        model,
        data,
        scene: SceneBuildResult,
        worker: WorkerEffector,
        prefix: str = CONTROL_PREFIX,
        role: str = "control",
        dc_driver: Any = None,
        dc_runtime: Any = None,
    ) -> None:
        self._mujoco = mujoco
        self._model = model
        self._data = data
        self._scene = scene
        self._worker = worker
        self._prefix = prefix
        self._role = role
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene.base_mocap_names[role])
        if body_id < 0:
            raise RuntimeError(f"Missing mocap body {scene.base_mocap_names[role]}")
        self._mocap_id = int(model.body_mocapid[body_id])
        self._anchors = worker.anchors
        self._hand_body = f"{prefix}right_wrist_yaw_link"
        self._dc_driver = dc_driver
        self._dc_runtime = dc_runtime
        self.target_base: BasePose = default_base_pose(role)
        self.pose: Dict[str, float] = home_pose(prefix)
        self.held_corner: Optional[str] = None
        self.placed_corners: Dict[str, str] = {}
        self.worker_assist_calls: int = 0

    def claim(self, corner: str) -> None:
        """Announce a claim before walking to the pickup. Visible on the dashboard."""
        _emit_dc_event(self._dc_driver, self._dc_runtime, "claimCorner", corner=corner)

    def grab(self, corner: str) -> Tuple[float, float, float]:
        pos = self._anchors.acquire(corner, self._role)
        self.held_corner = corner
        _emit_dc_event(
            self._dc_driver, self._dc_runtime, "cornerHeld",
            corner=corner, x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
        )
        return pos

    def steer_anchor(self, target: Tuple[float, float, float]) -> None:
        if self.held_corner is not None:
            self._anchors.update(self.held_corner, self._role, target)

    def track_hand(self) -> Tuple[float, float, float]:
        """Snap the held corner's anchor mocap to the current hand position."""
        if self.held_corner is None:
            return (0.0, 0.0, 0.0)
        return self._anchors.follow_hand(self.held_corner, self._role, self._hand_body)

    def release(self, bed_corner: str = "") -> None:
        if self.held_corner is not None:
            released = self.held_corner
            self._anchors.release(released, self._role)
            self.placed_corners[released] = "placed"
            if bed_corner:
                _emit_dc_event(
                    self._dc_driver, self._dc_runtime, "cornerPlaced",
                    corner=released, bed_corner=bed_corner,
                )
            _emit_dc_event(self._dc_driver, self._dc_runtime, "cornerReleased", corner=released)
            self.held_corner = None

    def request_worker_hold(self, corner: str, station: BasePose) -> None:
        """Ask the peer to assist via a DC askAssist event, then locally trigger the worker.

        The askAssist event is what the dashboard sees — it shows the two G1s
        coordinating over the network. The local worker call is the
        sub-millisecond fast path that drives the MuJoCo simulation while the
        event propagates.
        """
        self.worker_assist_calls += 1
        _emit_dc_event(
            self._dc_driver, self._dc_runtime, "askAssist",
            corner=corner, reason="hold_placed_corner",
        )
        self._worker.holdCorner(corner, station)

    def request_worker_release(self) -> None:
        _emit_dc_event(self._dc_driver, self._dc_runtime, "askAssist", corner="", reason="release")
        self._worker.releaseCorner()

    def request_worker_idle(self) -> None:
        _emit_dc_event(self._dc_driver, self._dc_runtime, "askAssist", corner="", reason="idle")
        self._worker.setIdle()

    @property
    def prefix(self) -> str:
        return self._prefix

    @property
    def mocap_id(self) -> int:
        return self._mocap_id

    def apply(self, gait_phase: float, gait_intensity: float) -> Dict[str, float]:
        pose = dict(self.pose)
        if gait_intensity > 0.0:
            _apply_gait_pose(pose, self._prefix, gait_phase, gait_intensity)
        return pose


# ---------------------------------------------------------------------------
# CLI plumbing and headless / viewer detection
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="make the bed", help="High-level task label to record in the summary.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "mujoco_bed_making" / "unitree_g1_bed_making"))
    parser.add_argument("--width", type=int, default=960, help="Render width.")
    parser.add_argument("--height", type=int, default=640, help="Render height.")
    parser.add_argument(
        "--render-every",
        type=int,
        default=18,
        help="Save every Nth control frame with --render-frames.",
    )
    parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per control frame.")
    parser.add_argument("--settle-steps", type=int, default=400, help="Physics steps for the initial sheet drop.")
    parser.add_argument("--walk-steps", type=int, default=180, help="Control frames per walk segment.")
    parser.add_argument("--phase-steps", type=int, default=120, help="Control frames per manipulation phase.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed (currently unused for physics state).")
    parser.add_argument("--no-viewer", action="store_true", help="Skip opening the MuJoCo passive viewer.")
    parser.add_argument("--render-frames", action="store_true", help="Export PNG frames during the scripted run.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Do not export PNG frames. Viewer still opens by default.",
    )
    parser.add_argument("--strict-render", action="store_true", help="Fail if rendering is unavailable.")
    parser.add_argument("--dry-run", action="store_true", help="Build and compile the scene, but do not simulate.")
    parser.add_argument(
        "--no-device-connect",
        action="store_true",
        help=(
            "Skip starting the Device Connect runtimes that register the two G1s "
            "with the dashboard. By default the demo registers the robots from "
            "the JWT credentials under .credentials/ if they are present."
        ),
    )
    parser.add_argument(
        "--device-connect-nats-url",
        default=os.environ.get("DEVICE_CONNECT_NATS_URL", "nats://fabric.deviceconnect.dev:4222"),
        help="Device Connect NATS broker URL (default: %(default)s).",
    )
    return parser.parse_args()


def _render_troubleshooting_hint() -> str:
    if sys.platform == "darwin":
        return (
            "On macOS, render from a normal Terminal session with `.venv/bin/mjpython "
            "examples/mujoco_bed_making/bed_making_demo/unitree_g1_bed_making_demo.py`, or run with `--no-render` in non-interactive shells."
        )
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return (
                "On headless Linux, use `--no-render` or install an offscreen MuJoCo GL backend. "
                "For CPU rendering, install OSMesa, for example `sudo apt-get install libosmesa6-dev`, "
                "then run with `MUJOCO_GL=osmesa`. For GPU rendering, install EGL/NVIDIA libraries and "
                "run with `MUJOCO_GL=egl`."
            )
        return (
            "On Linux desktop sessions, check that DISPLAY or WAYLAND_DISPLAY points to a working display "
            "and that OpenGL/GLFW system libraries are installed. Use `--no-render` for simulation-only runs."
        )
    return "Use `--no-render` for simulation-only runs, or configure a MuJoCo-compatible OpenGL backend."


def _viewer_troubleshooting_hint() -> str:
    if sys.platform == "darwin":
        return (
            "On macOS, launch the viewer from a normal Terminal session. The script will re-exec through "
            "`.venv/bin/mjpython` when needed. Use `--no-viewer` for non-interactive runs."
        )
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return (
                "On headless Linux there is no desktop display for the MuJoCo viewer. Use `--no-viewer`, "
                "or run under a desktop/X11/Wayland session."
            )
        return (
            "On Linux, check that DISPLAY or WAYLAND_DISPLAY points to a working display and that GLFW/OpenGL "
            "system libraries are installed. Use `--no-viewer` for simulation-only runs."
        )
    return "Use `--no-viewer` for simulation-only runs."


def _preflight_render_environment() -> None:
    if _is_darwin_noninteractive():
        raise RuntimeError(
            "MuJoCo rendering on macOS requires a foreground Terminal TTY or mjpython. "
            + _render_troubleshooting_hint()
        )
    if sys.platform.startswith("linux"):
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        has_gl_backend = bool(os.environ.get("MUJOCO_GL"))
        if not has_display and not has_gl_backend:
            raise RuntimeError(
                "MuJoCo rendering is not configured for this headless Linux session. "
                + _render_troubleshooting_hint()
            )


def _is_darwin_noninteractive() -> bool:
    return (
        sys.platform == "darwin"
        and not sys.stdin.isatty()
        and os.environ.get("STRANDS_MJPYTHON_REEXEC") != "1"
        and not os.environ.get("MJPYTHON_BIN")
    )


def _should_export_frames(args: argparse.Namespace) -> bool:
    return args.render_frames and not args.no_render


def _should_open_viewer(args: argparse.Namespace) -> bool:
    if args.no_viewer:
        return False
    if not sys.stdin.isatty():
        print(f"Skipping MuJoCo viewer in this non-interactive shell. {_viewer_troubleshooting_hint()}")
        return False
    return True


def _open_viewer(mujoco, model, data, args: argparse.Namespace):
    if not _should_open_viewer(args):
        return None
    if _is_darwin_noninteractive():
        raise RuntimeError("MuJoCo viewer on macOS requires mjpython. " + _viewer_troubleshooting_hint())

    try:
        import mujoco.viewer as viewer

        handle = viewer.launch_passive(model, data)
        print("Interactive MuJoCo viewer opened.")
        return handle
    except Exception as exc:
        raise RuntimeError(f"Could not open MuJoCo viewer: {exc}. {_viewer_troubleshooting_hint()}") from exc


def _sync_viewer(viewer_handle) -> None:
    if viewer_handle is None:
        return
    try:
        if hasattr(viewer_handle, "is_running") and not viewer_handle.is_running():
            return
        viewer_handle.sync()
    except Exception:
        pass


def _close_viewer(viewer_handle) -> None:
    if viewer_handle is None:
        return
    try:
        viewer_handle.close()
    except Exception:
        pass


def _ensure_g1_xml() -> Path:
    result = download_robots(names=["unitree_g1"])
    failures = result.get("failed_details", {})
    if failures:
        raise RuntimeError(f"Failed to prepare Unitree G1 assets: {failures}")

    resolved = resolve_model("unitree_g1")
    if not resolved:
        raise RuntimeError("Could not resolve the Unitree G1 MuJoCo scene.")

    resolved_path = Path(resolved)
    sibling_g1_with_hands = resolved_path.parent / "g1_with_hands.xml"
    if sibling_g1_with_hands.exists():
        return sibling_g1_with_hands
    sibling_g1 = resolved_path.parent / "g1.xml"
    return sibling_g1 if sibling_g1.exists() else resolved_path


def _install_quiet_mujoco_warning_handler(mujoco) -> None:
    """Silence the cosmetic MuJoCo NaN/QACC startup warnings.

    The flex bedsheet + multiple weld equalities produce a small batch of
    "Nan, Inf or huge value in QACC/QPOS" lines as the constraint solver
    settles in the first few steps. They are recoverable and the simulation
    runs fine afterwards, but they spam the user's terminal. Install a
    user-warning callback that swallows those specific messages while still
    letting real fatal errors propagate via mj_step exceptions.
    """

    try:
        def _handler(message: str) -> None:
            lowered = message.lower() if isinstance(message, str) else ""
            if "nan" in lowered or "huge value" in lowered or "unstable" in lowered:
                return
            sys.stderr.write(message.rstrip() + "\n")

        mujoco.set_mju_user_warning(_handler)
    except Exception:
        # Older mujoco bindings may not expose set_mju_user_warning.
        pass


def _load_mujoco_scene(scene_path: Path):
    import mujoco

    _install_quiet_mujoco_warning_handler(mujoco)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def _set_controls(mujoco, model, data, actuator_cache: Dict[str, int], pose: Dict[str, float]) -> None:
    for name, value in pose.items():
        act_id = actuator_cache.get(name)
        if act_id is None:
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            actuator_cache[name] = act_id
        if act_id >= 0:
            data.ctrl[act_id] = float(value)


def _set_mocap(data, mocap_id: int, base: BasePose) -> None:
    data.mocap_pos[mocap_id] = (base.x, base.y, base.z)
    data.mocap_quat[mocap_id] = _yaw_quat(base.yaw)


def _initialize_robot_qpos(mujoco, model, data, spec: RobotSceneSpec) -> None:
    """Place the pelvis freejoint at the mocap initial pose to avoid jumps."""

    joint_name = f"{spec.prefix}floating_base_joint"
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return
    qpos_addr = model.jnt_qposadr[joint_id]
    qvel_addr = model.jnt_dofadr[joint_id]
    data.qpos[qpos_addr : qpos_addr + 3] = (spec.position[0], spec.position[1], PELVIS_HEIGHT_M)
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = _yaw_quat(spec.yaw_radians)
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def _sheet_body_id(mujoco, model, node_index: int) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bedsheet_{node_index}")
    if body_id < 0:
        raise RuntimeError(f"Could not find bedsheet node body bedsheet_{node_index}")
    return body_id


def _sheet_corner_position(
    mujoco, model, data, scene: SceneBuildResult, corner: str
) -> Tuple[float, float, float]:
    node_index = scene.sheet_corner_nodes[corner]
    body_id = _sheet_body_id(mujoco, model, node_index)
    return tuple(float(v) for v in data.xpos[body_id])


def _count_drifted_corners(
    mujoco,
    model,
    data,
    scene: SceneBuildResult,
    placed: Dict[str, Tuple[float, float, float]],
) -> int:
    drifted = 0
    for corner, target in placed.items():
        node = scene.sheet_corner_nodes[corner]
        body_id = _sheet_body_id(mujoco, model, node)
        if math.dist(data.xpos[body_id], target) > 0.12:
            drifted += 1
    return drifted


def _render_frame(mujoco, model, data, output_path: Path, width: int, height: int) -> None:
    _preflight_render_environment()

    from PIL import Image

    renderer = None
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        renderer.update_scene(data, camera=camera_id if camera_id >= 0 else None)
        img = renderer.render()
    except Exception as exc:
        raise RuntimeError(f"{exc}. {_render_troubleshooting_hint()}") from exc
    finally:
        if renderer is not None:
            renderer.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(output_path)


# ---------------------------------------------------------------------------
# Scripted bed-making run loop
# ---------------------------------------------------------------------------


@dataclass
class FrameState:
    control_frame: int = 0
    saved_frame: int = 0
    render_failed: bool = False
    solver_recoveries: int = 0


def _step_run(
    *,
    mujoco,
    model,
    data,
    scene: SceneBuildResult,
    args: argparse.Namespace,
    frame_state: FrameState,
    actuator_cache: Dict[str, int],
    planner: ControlPlanner,
    worker: WorkerEffector,
    duration_steps: int,
    label: str,
    control_target: BasePose,
    worker_target: BasePose,
    control_start: BasePose,
    worker_start: BasePose,
    control_pose_target: Dict[str, float],
    worker_pose_target: Dict[str, float],
    control_pose_start: Dict[str, float],
    worker_pose_start: Dict[str, float],
    placed: Dict[str, Tuple[float, float, float]],
    viewer_handle,
    control_gait: bool = False,
    worker_gait: bool = False,
    anchor_mode: str = "none",
    anchor_target: Optional[Tuple[float, float, float]] = None,
) -> Tuple[BasePose, BasePose, Dict[str, float], Dict[str, float], int, Optional[str]]:
    """Run one phase, returning the final control/worker bases and poses.

    ``anchor_mode`` controls how the held cloth corner is driven:
        - ``"none"``: anchor is not steered this phase (cloth stays where it is
          relative to the world or the hand from the prior frame).
        - ``"hand"``: each frame the held corner's anchor is snapped to the
          robot's right wrist body, so the cloth visually stays in the hand.
        - ``"path"``: each frame the anchor interpolates from the current
          hand position toward ``anchor_target`` (the bed corner). Useful for
          the place phase so the cloth lands on the bed.
    """

    render_error: Optional[str] = None
    drifted = 0

    # If we'll need a starting point for the planned path, capture it from the
    # current hand world position at the start of the phase.
    path_start: Optional[Tuple[float, float, float]] = None
    if anchor_mode == "path" and planner.held_corner is not None and anchor_target is not None:
        path_start = planner.track_hand()

    for step in range(duration_steps):
        alpha = (step + 1) / max(1, duration_steps)
        eased = 0.5 - 0.5 * math.cos(math.pi * alpha)
        control_base = blend_base(control_start, control_target, eased)
        worker_base = blend_base(worker_start, worker_target, eased)
        control_pose = blend_pose(control_pose_start, control_pose_target, eased)
        worker_pose = blend_pose(worker_pose_start, worker_pose_target, eased)

        # Update planner/worker pose memory each frame so gait modulation is visible.
        planner.pose = control_pose
        worker.pose = worker_pose

        gait_phase = frame_state.control_frame * 0.55
        full_control = planner.apply(gait_phase, 1.0 if control_gait else 0.0)
        full_worker = worker.apply(gait_phase + math.pi, 0.75 if worker_gait else 0.0)
        _set_controls(mujoco, model, data, actuator_cache, {**full_control, **full_worker})

        _set_mocap(data, planner.mocap_id, control_base)
        _set_mocap(data, worker.mocap_id, worker_base)

        if anchor_mode == "hand":
            planner.track_hand()
        elif anchor_mode == "path" and path_start is not None and anchor_target is not None:
            planner.steer_anchor(
                (
                    path_start[0] + (anchor_target[0] - path_start[0]) * eased,
                    path_start[1] + (anchor_target[1] - path_start[1]) * eased,
                    path_start[2] + (anchor_target[2] - path_start[2]) * eased,
                )
            )

        # Snapshot the physics state so a transient solver failure (e.g. a
        # rank-deficient Hessian when the stiff cloth weld engages during a
        # grasp/carry) can be recovered instead of aborting the whole run.
        _qpos0 = data.qpos.copy()
        _qvel0 = data.qvel.copy()
        _act0 = data.act.copy() if data.act.size else None
        for _ in range(args.substeps):
            try:
                mujoco.mj_step(model, data)
            except Exception:
                # Restore the pre-substep state, damp out the velocity blow-up,
                # and carry on. The scripted kinematic targets keep advancing,
                # so the run continues to completion; we just skip the rest of
                # this frame's substeps.
                data.qpos[:] = _qpos0
                data.qvel[:] = 0.0
                if _act0 is not None:
                    data.act[:] = _act0
                mujoco.mj_forward(model, data)
                frame_state.solver_recoveries += 1
                break

        drifted = max(drifted, _count_drifted_corners(mujoco, model, data, scene, placed))
        _sync_viewer(viewer_handle)
        frame_state.control_frame += 1
        if (
            _should_export_frames(args)
            and not frame_state.render_failed
            and frame_state.control_frame % args.render_every == 0
            and render_error is None
        ):
            frame_state.saved_frame += 1
            frame_path = Path(args.output_dir) / f"frame_{frame_state.saved_frame:04d}_{label}.png"
            try:
                _render_frame(mujoco, model, data, frame_path, args.width, args.height)
            except Exception as exc:
                render_error = str(exc)
                frame_state.render_failed = True
                if args.strict_render:
                    raise

    return control_target, worker_target, control_pose_target, worker_pose_target, drifted, render_error


def _initial_settle(
    mujoco,
    model,
    data,
    scene: SceneBuildResult,
    args: argparse.Namespace,
    planner: ControlPlanner,
    worker: WorkerEffector,
    actuator_cache: Dict[str, int],
    viewer_handle,
) -> None:
    """Initialize qpos and drop the sheet onto the floor before the run."""

    for spec in default_robot_specs():
        _initialize_robot_qpos(mujoco, model, data, spec)
    _set_mocap(data, planner.mocap_id, planner.target_base)
    _set_mocap(data, worker.mocap_id, worker.target_base)
    _set_controls(mujoco, model, data, actuator_cache, {**planner.pose, **worker.pose})
    mujoco.mj_forward(model, data)

    for _ in range(args.settle_steps):
        mujoco.mj_step(model, data)
        _sync_viewer(viewer_handle)


def _drive_make_bed(
    args: argparse.Namespace,
    scene: SceneBuildResult,
    mujoco,
    model,
    data,
    viewer_handle=None,
    dc_runtime=None,
) -> Tuple[int, Optional[str], int]:
    actuator_cache: Dict[str, int] = {}
    frame_state = FrameState()
    placed: Dict[str, Tuple[float, float, float]] = {}
    render_error: Optional[str] = None
    drifted_total = 0

    control_driver, worker_driver = _resolve_dc_drivers(dc_runtime)
    worker = WorkerEffector(
        mujoco=mujoco,
        model=model,
        data=data,
        scene=scene,
        dc_driver=worker_driver,
        dc_runtime=dc_runtime,
    )
    planner = ControlPlanner(
        mujoco=mujoco,
        model=model,
        data=data,
        scene=scene,
        worker=worker,
        dc_driver=control_driver,
        dc_runtime=dc_runtime,
    )

    _initial_settle(mujoco, model, data, scene, args, planner, worker, actuator_cache, viewer_handle)

    plan = placement_plan()
    pickup_base = base_pose_for_sheet_pickup("control")

    current_control = planner.target_base
    current_worker = worker.target_base
    current_control_pose = planner.pose
    current_worker_pose = worker.pose

    for index, step in enumerate(plan, start=1):
        # Announce the upcoming claim over Device Connect so the peer (and
        # the dashboard) sees the swarm intent before any physical motion.
        planner.claim(step.sheet_corner)
        # 1. Walk to the sheet pickup station, then squat to grasp.
        for waypoint in waypoints_around_bed(current_control, pickup_base, "control"):
            (
                current_control,
                current_worker,
                current_control_pose,
                current_worker_pose,
                drifted,
                error,
            ) = _step_run(
                mujoco=mujoco,
                model=model,
                data=data,
                scene=scene,
                args=args,
                frame_state=frame_state,
                actuator_cache=actuator_cache,
                planner=planner,
                worker=worker,
                duration_steps=args.walk_steps,
                label=f"{index:02d}_walk_pickup",
                control_target=waypoint,
                worker_target=current_worker,
                control_start=current_control,
                worker_start=current_worker,
                control_pose_target=pose_for_phase(CONTROL_PREFIX, "right", "walk"),
                worker_pose_target=current_worker_pose,
                control_pose_start=current_control_pose,
                worker_pose_start=current_worker_pose,
                placed=placed,
                viewer_handle=viewer_handle,
                control_gait=True,
            )
            drifted_total = max(drifted_total, drifted)
            render_error = render_error or error

        (
            current_control,
            current_worker,
            current_control_pose,
            current_worker_pose,
            drifted,
            error,
        ) = _step_run(
            mujoco=mujoco,
            model=model,
            data=data,
            scene=scene,
            args=args,
            frame_state=frame_state,
            actuator_cache=actuator_cache,
            planner=planner,
            worker=worker,
            duration_steps=args.phase_steps,
            label=f"{index:02d}_squat",
            control_target=current_control,
            worker_target=current_worker,
            control_start=current_control,
            worker_start=current_worker,
            control_pose_target=pose_for_phase(CONTROL_PREFIX, "right", "reach"),
            worker_pose_target=current_worker_pose,
            control_pose_start=current_control_pose,
            worker_pose_start=current_worker_pose,
            placed=placed,
            viewer_handle=viewer_handle,
        )
        drifted_total = max(drifted_total, drifted)
        render_error = render_error or error

        # 2. Grasp: snap the cloth anchor to the corner's current position and
        # activate the weld. From this point on the cloth corner tracks the
        # robot's right wrist body each frame ("anchor_mode='hand'") so it is
        # visually in the hand instead of being yanked along a precomputed
        # path.
        planner.grab(step.sheet_corner)
        (
            current_control,
            current_worker,
            current_control_pose,
            current_worker_pose,
            drifted,
            error,
        ) = _step_run(
            mujoco=mujoco,
            model=model,
            data=data,
            scene=scene,
            args=args,
            frame_state=frame_state,
            actuator_cache=actuator_cache,
            planner=planner,
            worker=worker,
            duration_steps=args.phase_steps,
            label=f"{index:02d}_grasp",
            control_target=current_control,
            worker_target=current_worker,
            control_start=current_control,
            worker_start=current_worker,
            control_pose_target=pose_for_phase(CONTROL_PREFIX, "right", "grasp"),
            worker_pose_target=current_worker_pose,
            control_pose_start=current_control_pose,
            worker_pose_start=current_worker_pose,
            placed=placed,
            viewer_handle=viewer_handle,
            anchor_mode="hand",
        )
        drifted_total = max(drifted_total, drifted)
        render_error = render_error or error

        # 3. Walk to the bed corner station while carrying the corner. The
        # cloth follows the right wrist via the per-frame hand-tracking anchor.
        bed_target = scene.bed_corner_targets[step.bed_corner]
        waypoints = waypoints_around_bed(current_control, base_pose_for_corner("control", step.bed_corner), "control")
        for waypoint in waypoints:
            (
                current_control,
                current_worker,
                current_control_pose,
                current_worker_pose,
                drifted,
                error,
            ) = _step_run(
                mujoco=mujoco,
                model=model,
                data=data,
                scene=scene,
                args=args,
                frame_state=frame_state,
                actuator_cache=actuator_cache,
                planner=planner,
                worker=worker,
                duration_steps=args.walk_steps,
                label=f"{index:02d}_carry",
                control_target=waypoint,
                worker_target=current_worker,
                control_start=current_control,
                worker_start=current_worker,
                control_pose_target=pose_for_phase(CONTROL_PREFIX, "right", "carry"),
                worker_pose_target=current_worker_pose,
                control_pose_start=current_control_pose,
                worker_pose_start=current_worker_pose,
                placed=placed,
                viewer_handle=viewer_handle,
                anchor_mode="hand",
            )
            drifted_total = max(drifted_total, drifted)
            render_error = render_error or error

        # 4. Optionally ask the worker to walk over and hold an already placed corner.
        if step.ask_worker_after and placed:
            sheet_corner_to_hold = next(iter(placed))
            bed_corner_to_hold = _sheet_to_bed_corner(sheet_corner_to_hold)
            worker_station = base_pose_for_corner("worker", bed_corner_to_hold)
            for waypoint in waypoints_around_bed(current_worker, worker_station, "worker"):
                (
                    current_control,
                    current_worker,
                    current_control_pose,
                    current_worker_pose,
                    drifted,
                    error,
                ) = _step_run(
                    mujoco=mujoco,
                    model=model,
                    data=data,
                    scene=scene,
                    args=args,
                    frame_state=frame_state,
                    actuator_cache=actuator_cache,
                    planner=planner,
                    worker=worker,
                    duration_steps=args.walk_steps,
                    label=f"{index:02d}_worker_assist",
                    control_target=current_control,
                    worker_target=waypoint,
                    control_start=current_control,
                    worker_start=current_worker,
                    control_pose_target=current_control_pose,
                    worker_pose_target=pose_for_phase(WORKER_PREFIX, "left", "walk"),
                    control_pose_start=current_control_pose,
                    worker_pose_start=current_worker_pose,
                    placed=placed,
                    viewer_handle=viewer_handle,
                    worker_gait=True,
                )
                drifted_total = max(drifted_total, drifted)
                render_error = render_error or error
            planner.request_worker_hold(sheet_corner_to_hold, worker_station)
            (
                current_control,
                current_worker,
                current_control_pose,
                current_worker_pose,
                drifted,
                error,
            ) = _step_run(
                mujoco=mujoco,
                model=model,
                data=data,
                scene=scene,
                args=args,
                frame_state=frame_state,
                actuator_cache=actuator_cache,
                planner=planner,
                worker=worker,
                duration_steps=args.phase_steps,
                label=f"{index:02d}_worker_grasp",
                control_target=current_control,
                worker_target=current_worker,
                control_start=current_control,
                worker_start=current_worker,
                control_pose_target=current_control_pose,
                worker_pose_target=worker_hold_pose(),
                control_pose_start=current_control_pose,
                worker_pose_start=current_worker_pose,
                placed=placed,
                viewer_handle=viewer_handle,
            )
            drifted_total = max(drifted_total, drifted)
            render_error = render_error or error

        # 5. Place the carried corner over the bed corner and release. The
        # anchor follows a planned path from the current hand position down to
        # the bed top (slightly off-corner for an imperfect placement). The
        # arm pose simultaneously moves to "place".
        place_pos = (
            bed_target[0] + (0.04 if "right" in step.bed_corner else -0.04),
            bed_target[1] + (-0.03 if "right" in step.bed_corner else 0.03),
            BED_HEIGHT_M + 0.06,
        )
        (
            current_control,
            current_worker,
            current_control_pose,
            current_worker_pose,
            drifted,
            error,
        ) = _step_run(
            mujoco=mujoco,
            model=model,
            data=data,
            scene=scene,
            args=args,
            frame_state=frame_state,
            actuator_cache=actuator_cache,
            planner=planner,
            worker=worker,
            duration_steps=args.phase_steps,
            label=f"{index:02d}_place",
            control_target=current_control,
            worker_target=current_worker,
            control_start=current_control,
            worker_start=current_worker,
            control_pose_target=pose_for_phase(CONTROL_PREFIX, "right", "place"),
            worker_pose_target=current_worker_pose,
            control_pose_start=current_control_pose,
            worker_pose_start=current_worker_pose,
            placed=placed,
            viewer_handle=viewer_handle,
            anchor_mode="path",
            anchor_target=place_pos,
        )
        placed[step.sheet_corner] = _sheet_corner_position(mujoco, model, data, scene, step.sheet_corner)
        planner.release(bed_corner=step.bed_corner)
        drifted_total = max(drifted_total, drifted)
        render_error = render_error or error

        # 6. Lift hand, then trace toward home to clear the bed before the next pickup.
        (
            current_control,
            current_worker,
            current_control_pose,
            current_worker_pose,
            drifted,
            error,
        ) = _step_run(
            mujoco=mujoco,
            model=model,
            data=data,
            scene=scene,
            args=args,
            frame_state=frame_state,
            actuator_cache=actuator_cache,
            planner=planner,
            worker=worker,
            duration_steps=args.phase_steps,
            label=f"{index:02d}_trace",
            control_target=current_control,
            worker_target=current_worker,
            control_start=current_control,
            worker_start=current_worker,
            control_pose_target=pose_for_phase(CONTROL_PREFIX, "right", "trace"),
            worker_pose_target=current_worker_pose,
            control_pose_start=current_control_pose,
            worker_pose_start=current_worker_pose,
            placed=placed,
            viewer_handle=viewer_handle,
        )
        drifted_total = max(drifted_total, drifted)
        render_error = render_error or error

    # Final: worker releases any held corner and both return home.
    if worker.held_corner is not None:
        planner.request_worker_release()
    control_home = default_base_pose("control")
    worker_home = default_base_pose("worker")
    for waypoint in waypoints_around_bed(current_control, control_home, "control"):
        (
            current_control,
            current_worker,
            current_control_pose,
            current_worker_pose,
            drifted,
            error,
        ) = _step_run(
            mujoco=mujoco,
            model=model,
            data=data,
            scene=scene,
            args=args,
            frame_state=frame_state,
            actuator_cache=actuator_cache,
            planner=planner,
            worker=worker,
            duration_steps=args.walk_steps,
            label="99_return",
            control_target=waypoint,
            worker_target=worker_home,
            control_start=current_control,
            worker_start=current_worker,
            control_pose_target=home_pose(CONTROL_PREFIX),
            worker_pose_target=home_pose(WORKER_PREFIX),
            control_pose_start=current_control_pose,
            worker_pose_start=current_worker_pose,
            placed=placed,
            viewer_handle=viewer_handle,
            control_gait=True,
            worker_gait=True,
        )
        drifted_total = max(drifted_total, drifted)
        render_error = render_error or error

    return frame_state.saved_frame, render_error, planner.worker_assist_calls + drifted_total


def _opposite_bed_corner(corner: str) -> str:
    mapping = {
        "bed_head_right": "bed_foot_left",
        "bed_head_left": "bed_foot_right",
        "bed_foot_right": "bed_head_left",
        "bed_foot_left": "bed_head_right",
    }
    return mapping.get(corner, corner)


def _sheet_to_bed_corner(sheet_corner: str) -> str:
    for step in placement_plan():
        if step.sheet_corner == sheet_corner:
            return step.bed_corner
    raise KeyError(f"No bed target known for sheet corner {sheet_corner}")


def _write_summary(
    args: argparse.Namespace,
    scene: SceneBuildResult,
    model,
    saved_frames: int,
    render_error: Optional[str],
    worker_assists: int,
) -> None:
    lines = [
        "Unitree G1 two-humanoid bed-making demo complete.",
        f"Task: {args.task}",
        f"Scene: {scene.scene_path}",
        f"Bed mesh: {scene.mesh_assets.bed_obj}",
        f"Bedsheet mesh: {scene.mesh_assets.bedsheet_obj}",
        f"Robots: {', '.join(scene.robot_joint_names)}",
        f"Sheet flex corners: {scene.sheet_corner_nodes}",
        f"Bed targets: {bed_corner_targets()}",
        f"MuJoCo bodies: {model.nbody}, joints: {model.njnt}, actuators: {model.nu}, "
        f"equalities: {model.neq}, flexes: {model.nflex}",
        f"Frame export: {'enabled' if _should_export_frames(args) else 'disabled'}",
        f"Frames exported: {saved_frames}",
        f"Worker hold corrections: {worker_assists}",
        f"Render status: {render_error or 'ok'}",
    ]
    lines.extend(f"Plan {index}: {step.label}" for index, step in enumerate(placement_plan(), start=1))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "summary.txt").write_text("\n".join(lines) + "\n")


def _maybe_start_device_connect(args: argparse.Namespace):
    """Register the two G1s with Device Connect in a background thread.

    Returns the running :class:`BackgroundRuntime` instance, or ``None`` if
    Device Connect registration is disabled or the credentials are absent.
    """

    if args.no_device_connect:
        return None
    try:
        from examples.mujoco_bed_making.bed_making_swarm.unitree_g1_bed_making_device_connect_sidecar import (
            BackgroundRuntime,
            default_specs,
        )
    except Exception as exc:
        print(f"Device Connect import failed; skipping registration: {exc}", file=sys.stderr)
        return None
    specs = default_specs(repo_root=REPO_ROOT)
    if not specs:
        print(
            "No Device Connect credentials found under .credentials/ — skipping. "
            "Run examples/mujoco_bed_making/bed_making_swarm/unitree_g1_bed_making_device_connect_sidecar.py with explicit "
            "--credentials to register manually.",
            file=sys.stderr,
        )
        return None
    try:
        bg = BackgroundRuntime.start(specs, nats_url=args.device_connect_nats_url, tenant=None)
    except Exception as exc:
        print(f"Failed to start Device Connect runtimes: {exc}", file=sys.stderr)
        return None
    print(
        "Registered with Device Connect: "
        + ", ".join(f"{s.role}={s.device_id}" for s in specs)
        + f"  (broker: {args.device_connect_nats_url})",
    )
    return bg


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    g1_xml = _ensure_g1_xml()
    scene = build_scene(g1_xml_path=g1_xml, output_dir=output_dir)
    mujoco, model, data = _load_mujoco_scene(scene.scene_path)

    if args.dry_run:
        print(f"Built scene: {scene.scene_path}")
        print(
            f"Bodies={model.nbody} joints={model.njnt} actuators={model.nu} "
            f"equalities={model.neq} flexes={model.nflex}"
        )
        print(f"Bed mesh: {scene.mesh_assets.bed_obj}")
        print(f"Bedsheet mesh: {scene.mesh_assets.bedsheet_obj}")
        return 0

    device_connect_runtime = _maybe_start_device_connect(args)
    viewer_handle = None
    try:
        viewer_handle = _open_viewer(mujoco, model, data, args)
        saved_frames, render_error, worker_assists = _drive_make_bed(
            args,
            scene,
            mujoco,
            model,
            data,
            viewer_handle=viewer_handle,
            dc_runtime=device_connect_runtime,
        )
    finally:
        _close_viewer(viewer_handle)
        if device_connect_runtime is not None:
            device_connect_runtime.stop()

    _write_summary(args, scene, model, saved_frames, render_error, worker_assists)

    if render_error:
        print(f"Simulation completed, but rendering stopped after: {render_error}")
        print(f"Wrote summary to {output_dir / 'summary.txt'}")
    else:
        frame_text = f"Saved {saved_frames} frames and " if _should_export_frames(args) else ""
        print(f"{frame_text}Wrote summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
