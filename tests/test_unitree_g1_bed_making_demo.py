"""Tests for the Unitree G1 two-humanoid bed-making demo scene."""

import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from examples.mujoco_bed_making.bed_making_demo import unitree_g1_bed_making_demo
from examples.mujoco_bed_making.bed_making_demo.unitree_g1_bed_making_demo import (
    SHEET_GRID_X,
    SHEET_GRID_Y,
    base_pose_for_corner,
    base_pose_for_sheet_pickup,
    build_scene,
    final_sheet_layout,
    placement_plan,
    pose_for_phase,
    sheet_corner_nodes,
    waypoints_around_bed,
    write_mesh_assets,
)


def test_mesh_assets_have_requested_dimensions_and_triangles(tmp_path):
    assets = write_mesh_assets(tmp_path)

    bed_lines = assets.bed_obj.read_text().splitlines()
    sheet_lines = assets.bedsheet_obj.read_text().splitlines()

    assert sum(line.startswith("v ") for line in bed_lines) == 8
    assert sum(line.startswith("f ") for line in bed_lines) == 12
    assert sum(line.startswith("v ") for line in sheet_lines) == SHEET_GRID_X * SHEET_GRID_Y
    assert sum(line.startswith("f ") for line in sheet_lines) == (SHEET_GRID_X - 1) * (SHEET_GRID_Y - 1) * 2


def test_scene_builder_creates_two_prefixed_g1_instances(tmp_path):
    g1_xml = _write_minimal_g1_xml(tmp_path)

    scene = build_scene(g1_xml_path=g1_xml, output_dir=tmp_path / "out")
    root = ET.fromstring(scene.xml)

    body_names = {node.attrib["name"] for node in root.findall(".//body") if "name" in node.attrib}
    actuator_names = {node.attrib["name"] for node in root.findall(".//actuator/*") if "name" in node.attrib}

    assert "control_pelvis" in body_names
    assert "worker_pelvis" in body_names
    assert "control_right_shoulder_pitch_joint" in actuator_names
    assert "worker_right_shoulder_pitch_joint" in actuator_names
    assert root.find(".//flexcomp[@name='bedsheet']") is not None
    assert root.find(".//body[@name='bed']/geom[@name='bed_mesh_visual']") is not None
    assert scene.sheet_corner_nodes["sheet_head_right"] == SHEET_GRID_X * SHEET_GRID_Y - 1


def test_scene_builder_emits_mocap_bodies_and_equality_welds(tmp_path):
    g1_xml = _write_minimal_g1_xml(tmp_path)

    scene = build_scene(g1_xml_path=g1_xml, output_dir=tmp_path / "out")
    root = ET.fromstring(scene.xml)

    body_names = {node.attrib["name"] for node in root.findall(".//body") if "name" in node.attrib}
    equality_names = {node.attrib["name"] for node in root.findall(".//equality/*") if "name" in node.attrib}

    # Each robot has a mocap body that the base weld follows.
    assert "control_base_mocap" in body_names
    assert "worker_base_mocap" in body_names
    assert scene.base_mocap_names == {"control": "control_base_mocap", "worker": "worker_base_mocap"}
    for body_name in scene.base_mocap_names.values():
        node = root.find(f".//body[@name='{body_name}']")
        assert node is not None
        assert node.attrib.get("mocap") == "true"

    # Base weld between each mocap and the pelvis.
    for role, weld_name in scene.base_weld_names.items():
        assert weld_name in equality_names
        weld = root.find(f".//equality/weld[@name='{weld_name}']")
        assert weld is not None
        assert weld.attrib["body1"] == scene.base_mocap_names[role]
        assert weld.attrib["body2"].endswith("pelvis")

    # One kinematic anchor mocap + weld per sheet corner. Welds are inactive
    # at compile time so the driver can toggle them at runtime via eq_active.
    assert set(scene.corner_anchor_mocaps) == set(scene.sheet_corner_nodes)
    assert set(scene.corner_anchor_welds) == set(scene.sheet_corner_nodes)
    for corner, anchor_body in scene.corner_anchor_mocaps.items():
        anchor = root.find(f".//body[@name='{anchor_body}']")
        assert anchor is not None and anchor.attrib.get("mocap") == "true"
        weld_name = scene.corner_anchor_welds[corner]
        weld = root.find(f".//equality/weld[@name='{weld_name}']")
        assert weld is not None
        assert weld.attrib.get("active") == "false"
        assert weld.attrib["body1"] == anchor_body
        assert weld.attrib["body2"] == f"bedsheet_{scene.sheet_corner_nodes[corner]}"


def test_base_pose_helpers_keep_robots_outside_bed_footprint():
    side_half = unitree_g1_bed_making_demo.BED_WIDTH_M / 2.0
    for corner in unitree_g1_bed_making_demo.bed_corner_targets():
        ctrl = base_pose_for_corner("control", corner)
        worker = base_pose_for_corner("worker", corner)
        assert abs(ctrl.y) >= side_half, f"control station {corner} too close to bed edge"
        assert abs(worker.y) >= side_half, f"worker station {corner} too close to bed edge"
        # Robots should stand on opposite long edges of the bed.
        assert math.copysign(1.0, ctrl.y) == -math.copysign(1.0, worker.y)

    pickup = base_pose_for_sheet_pickup("control")
    assert pickup.x < -unitree_g1_bed_making_demo.BED_LENGTH_M / 2.0
    assert pickup.z == unitree_g1_bed_making_demo.PELVIS_HEIGHT_M


def test_waypoints_skip_bed_when_switching_long_sides():
    start = base_pose_for_corner("control", "bed_head_right")
    target = base_pose_for_corner("control", "bed_head_left")
    waypoints = waypoints_around_bed(start, target, "control")

    assert len(waypoints) >= 3, "switching sides should detour around the bed foot"
    foot_x = -(unitree_g1_bed_making_demo.BED_LENGTH_M / 2.0 + unitree_g1_bed_making_demo.BED_STAND_OFFSET_M)
    # All but the last waypoint should be at the foot detour x.
    for wp in waypoints[:-1]:
        assert wp.x <= foot_x + 1e-6, "detour waypoint should be at the foot of the bed"


def test_same_side_waypoints_skip_detour():
    start = base_pose_for_corner("control", "bed_head_right")
    target = base_pose_for_corner("control", "bed_foot_right")
    assert waypoints_around_bed(start, target, "control") == [target]


def test_placement_plan_and_pose_targets_are_prefixed():
    plan = placement_plan()
    pose = pose_for_phase("control_", "right", "grasp")

    assert len(plan) == 4
    assert plan[0].sheet_corner == "sheet_foot_left"
    assert plan[-1].ask_worker_after is True
    assert "control_right_shoulder_pitch_joint" in pose
    assert "control_right_knee_joint" in pose
    assert "control_right_hand_index_0_joint" in pose
    assert "right_shoulder_pitch_joint" not in pose


def test_final_sheet_layout_keeps_adjacent_grid_spacing_realistic():
    layout = final_sheet_layout(seed=7)
    x_edges = []
    y_edges = []
    for ix in range(SHEET_GRID_X):
        for iy in range(SHEET_GRID_Y):
            node = ix * SHEET_GRID_Y + iy
            if ix + 1 < SHEET_GRID_X:
                x_edges.append(math.dist(layout[node], layout[(ix + 1) * SHEET_GRID_Y + iy]))
            if iy + 1 < SHEET_GRID_Y:
                y_edges.append(math.dist(layout[node], layout[ix * SHEET_GRID_Y + iy + 1]))

    assert abs(sum(x_edges) / len(x_edges) - unitree_g1_bed_making_demo.SHEET_SPACING_X_M) < 0.03
    assert abs(sum(y_edges) / len(y_edges) - unitree_g1_bed_making_demo.SHEET_SPACING_Y_M) < 0.03

    corners = sheet_corner_nodes()
    assert layout[corners["sheet_foot_left"]][0] > 0.95
    assert layout[corners["sheet_head_right"]][0] < -1.0


class _StubData:
    """Minimal stand-in for mujoco.MjData carrying eq_active, mocap_pos, xpos."""

    def __init__(self, neq: int, nbody: int, nmocap: int):
        self.eq_active = [0] * neq
        self.mocap_pos = [[0.0, 0.0, 0.0] for _ in range(nmocap)]
        self.xpos = [[0.0, 0.0, 0.0] for _ in range(nbody)]


class _StubModel:
    """Stand-in for mujoco.MjModel exposing name-to-id lookups via dicts."""

    def __init__(self, body_ids, mocap_ids):
        self._body_ids = body_ids
        max_body = max(body_ids.values()) if body_ids else 0
        self.body_mocapid = [-1] * (max_body + 1)
        for body_name, body_id in body_ids.items():
            if body_name in mocap_ids:
                self.body_mocapid[body_id] = mocap_ids[body_name]


class _StubMujoco:
    """Stand-in for the mujoco module's name lookups used by the planner."""

    class mjtObj:
        mjOBJ_BODY = 0
        mjOBJ_EQUALITY = 1
        mjOBJ_ACTUATOR = 2

    def __init__(self, body_ids, equality_ids):
        self._body_ids = body_ids
        self._equality_ids = equality_ids

    def mj_name2id(self, model, obj_type, name):
        if obj_type == self.mjtObj.mjOBJ_BODY:
            return self._body_ids.get(name, -1)
        if obj_type == self.mjtObj.mjOBJ_EQUALITY:
            return self._equality_ids.get(name, -1)
        return -1


def test_worker_effector_toggles_corner_anchor_on_hold_and_release():
    sheet_corners = sheet_corner_nodes()
    scene = unitree_g1_bed_making_demo.SceneBuildResult(
        xml="",
        scene_path=Path("/tmp/scene.xml"),
        mesh_assets=unitree_g1_bed_making_demo.MeshAssetPaths(
            bed_obj=Path("/tmp/bed.obj"), bedsheet_obj=Path("/tmp/sheet.obj")
        ),
        robot_joint_names={"control": [], "worker": []},
        robot_actuator_names={"control": [], "worker": []},
        sheet_corner_nodes=sheet_corners,
        bed_corner_targets=unitree_g1_bed_making_demo.bed_corner_targets(),
        base_weld_names={"control": "control_base_weld", "worker": "worker_base_weld"},
        base_mocap_names={"control": "control_base_mocap", "worker": "worker_base_mocap"},
        corner_anchor_mocaps={c: f"cloth_anchor_{c}" for c in sheet_corners},
        corner_anchor_welds={c: f"cloth_anchor_{c}_weld" for c in sheet_corners},
    )
    # Body IDs: base mocaps + corner anchor mocaps + cloth corner bodies.
    body_ids = {"control_base_mocap": 1, "worker_base_mocap": 2}
    mocap_ids = {"control_base_mocap": 0, "worker_base_mocap": 1}
    next_body_id = 3
    next_mocap_id = 2
    for corner, name in scene.corner_anchor_mocaps.items():
        body_ids[name] = next_body_id
        mocap_ids[name] = next_mocap_id
        next_body_id += 1
        next_mocap_id += 1
    for corner, node_index in sheet_corners.items():
        body_ids[f"bedsheet_{node_index}"] = next_body_id
        next_body_id += 1
    equality_ids = {name: i for i, name in enumerate(scene.corner_anchor_welds.values())}

    model = _StubModel(body_ids, mocap_ids)
    data = _StubData(neq=len(equality_ids), nbody=next_body_id, nmocap=next_mocap_id)
    mujoco_stub = _StubMujoco(body_ids, equality_ids)
    # Pretend the foot_left corner has settled at a particular world position.
    data.xpos[body_ids[f"bedsheet_{sheet_corners['sheet_foot_left']}"]] = [-1.2, -0.8, 0.05]

    worker = unitree_g1_bed_making_demo.WorkerEffector(
        mujoco=mujoco_stub,
        model=model,
        data=data,
        scene=scene,
    )
    planner = unitree_g1_bed_making_demo.ControlPlanner(
        mujoco=mujoco_stub,
        model=model,
        data=data,
        scene=scene,
        worker=worker,
    )

    station = unitree_g1_bed_making_demo.BasePose(0.0, -1.4, 0.793, 0.0)
    # Worker holds the cloth corner. Mocap snaps to cloth position, weld activates.
    planner.request_worker_hold("sheet_foot_left", station)
    assert worker.status == "holding"
    weld_id = equality_ids["cloth_anchor_sheet_foot_left_weld"]
    mocap_id = mocap_ids["cloth_anchor_sheet_foot_left"]
    assert data.eq_active[weld_id] == 1
    assert tuple(data.mocap_pos[mocap_id]) == (-1.2, -0.8, 0.05)
    assert planner.worker_assist_calls == 1

    planner.request_worker_release()
    assert worker.held_corner is None
    assert data.eq_active[weld_id] == 0

    # Planner can pick up a different corner and steer the anchor along a path.
    data.xpos[body_ids[f"bedsheet_{sheet_corners['sheet_head_left']}"]] = [-0.3, -0.8, 0.04]
    planner.grab("sheet_head_left")
    assert planner.held_corner == "sheet_head_left"
    head_left_mocap = mocap_ids["cloth_anchor_sheet_head_left"]
    assert data.eq_active[equality_ids["cloth_anchor_sheet_head_left_weld"]] == 1
    assert tuple(data.mocap_pos[head_left_mocap]) == (-0.3, -0.8, 0.04)
    planner.steer_anchor((1.0, 0.9, 0.6))
    assert tuple(data.mocap_pos[head_left_mocap]) == (1.0, 0.9, 0.6)
    planner.release()
    assert data.eq_active[equality_ids["cloth_anchor_sheet_head_left_weld"]] == 0
    assert "sheet_head_left" in planner.placed_corners


def test_linux_render_hint_does_not_reference_mjpython():
    original_platform = sys.platform
    original_display = os.environ.pop("DISPLAY", None)
    original_wayland = os.environ.pop("WAYLAND_DISPLAY", None)
    try:
        sys.platform = "linux"
        hint = unitree_g1_bed_making_demo._render_troubleshooting_hint()
    finally:
        sys.platform = original_platform
        if original_display is not None:
            os.environ["DISPLAY"] = original_display
        if original_wayland is not None:
            os.environ["WAYLAND_DISPLAY"] = original_wayland

    assert "mjpython" not in hint
    assert "MUJOCO_GL=osmesa" in hint


def _write_minimal_g1_xml(tmp_path: Path) -> Path:
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    g1_xml = tmp_path / "g1.xml"
    g1_xml.write_text(
        """<mujoco model="minimal_g1">
  <compiler angle="radian" meshdir="assets"/>
  <default>
    <default class="g1">
      <joint armature="0.01"/>
      <position kp="100"/>
    </default>
  </default>
  <asset>
    <material name="metal" rgba="0.7 0.7 0.7 1"/>
  </asset>
  <worldbody>
    <body name="pelvis" pos="0 0 0.793" childclass="g1">
      <freejoint name="floating_base_joint"/>
      <body name="torso_link">
        <joint name="waist_yaw_joint" axis="0 0 1"/>
        <body name="right_shoulder_pitch_link">
          <joint name="right_shoulder_pitch_joint" axis="0 1 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position class="g1" name="waist_yaw_joint" joint="waist_yaw_joint"/>
    <position class="g1" name="right_shoulder_pitch_joint" joint="right_shoulder_pitch_joint"/>
  </actuator>
</mujoco>
"""
    )
    return g1_xml
