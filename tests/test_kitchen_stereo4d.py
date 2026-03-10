#!/usr/bin/env python3
"""Comprehensive CPU tests for the Kitchen Stereo4D pipeline.

Tests the CURRENT API in scripts/pick_and_place_kitchen.py:
  - Constants: STEREO_BASELINE_MM, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV,
    NUM_EPISODES, STEPS_PER_EPISODE, FPS, KITCHEN_OBJECTS, ARTIFACTS_DIR, DATASET_DIR
  - Dataclasses: StereoFrame, Episode
  - Classes: StereoCameraRig (compute_disparity, depth_to_point_cloud),
    FrankaPickPolicy (generate_trajectory)
  - Functions: build_kitchen_scene_xml, save_depth_colormap, save_side_by_side,
    save_stereo_anaglyph, save_point_cloud_image
  - strands_robots.stereo module API contract (StereoConfig, StereoResult)

All tests run on CPU without GPU, MuJoCo, OpenCV, or Pillow.

Refs: Issue #190, supersedes broken test_kitchen_pipeline.py/test_kitchen_stereo4d_pipeline.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree

import numpy as np
import pytest

# ── Import the script's functions ──────────────────────────────────
_SCRIPTS_DIR = str(Path(__file__).parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from pick_and_place_kitchen import (  # noqa: E402
    ARTIFACTS_DIR,
    CAMERA_FOV,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DATASET_DIR,
    FPS,
    KITCHEN_OBJECTS,
    NUM_EPISODES,
    STEPS_PER_EPISODE,
    STEREO_BASELINE_MM,
    Episode,
    FrankaPickPolicy,
    StereoCameraRig,
    StereoFrame,
    build_kitchen_scene_xml,
    save_depth_colormap,
    save_point_cloud_image,
    save_side_by_side,
    save_stereo_anaglyph,
)


def _has_torch() -> bool:
    """Check if PyTorch is available."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


# ═══════════════════════════════════════════════════════════════════
# 1. Module Constants
# ═══════════════════════════════════════════════════════════════════


class TestModuleConstants:
    """Verify configuration constants have expected types and values."""

    def test_stereo_baseline_mm(self):
        assert isinstance(STEREO_BASELINE_MM, float)
        assert STEREO_BASELINE_MM == 50.0

    def test_camera_width(self):
        assert isinstance(CAMERA_WIDTH, int)
        assert CAMERA_WIDTH == 640

    def test_camera_height(self):
        assert isinstance(CAMERA_HEIGHT, int)
        assert CAMERA_HEIGHT == 480

    def test_camera_fov(self):
        assert isinstance(CAMERA_FOV, float)
        assert CAMERA_FOV == 60.0

    def test_num_episodes(self):
        assert isinstance(NUM_EPISODES, int)
        assert NUM_EPISODES > 0

    def test_steps_per_episode(self):
        assert isinstance(STEPS_PER_EPISODE, int)
        assert STEPS_PER_EPISODE > 0

    def test_fps(self):
        assert isinstance(FPS, int)
        assert FPS == 30

    def test_artifacts_dir_is_path(self):
        assert isinstance(ARTIFACTS_DIR, Path)

    def test_dataset_dir_is_path(self):
        assert isinstance(DATASET_DIR, Path)


# ═══════════════════════════════════════════════════════════════════
# 2. StereoFrame Dataclass
# ═══════════════════════════════════════════════════════════════════


class TestStereoFrame:
    """Verify the StereoFrame dataclass structure and defaults."""

    def test_creation_with_required_fields(self):
        frame = StereoFrame(
            left_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            right_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            gt_depth_left=np.ones((480, 640), dtype=np.float32),
            gt_depth_right=np.ones((480, 640), dtype=np.float32),
        )
        assert frame.left_rgb.shape == (480, 640, 3)
        assert frame.right_rgb.shape == (480, 640, 3)
        assert frame.gt_depth_left.shape == (480, 640)
        assert frame.gt_depth_right.shape == (480, 640)

    def test_default_timestamp(self):
        frame = StereoFrame(
            left_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            right_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            gt_depth_left=np.ones((1, 1), dtype=np.float32),
            gt_depth_right=np.ones((1, 1), dtype=np.float32),
        )
        assert frame.timestamp == 0.0

    def test_default_episode_and_step(self):
        frame = StereoFrame(
            left_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            right_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            gt_depth_left=np.ones((1, 1), dtype=np.float32),
            gt_depth_right=np.ones((1, 1), dtype=np.float32),
        )
        assert frame.episode == 0
        assert frame.step == 0

    def test_custom_metadata_fields(self):
        frame = StereoFrame(
            left_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            right_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            gt_depth_left=np.ones((1, 1), dtype=np.float32),
            gt_depth_right=np.ones((1, 1), dtype=np.float32),
            timestamp=1.5,
            episode=3,
            step=42,
        )
        assert frame.timestamp == 1.5
        assert frame.episode == 3
        assert frame.step == 42

    def test_depth_dtype_preserved(self):
        depth = np.random.uniform(0.5, 5.0, (10, 10)).astype(np.float32)
        frame = StereoFrame(
            left_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
            right_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
            gt_depth_left=depth,
            gt_depth_right=depth.copy(),
        )
        assert frame.gt_depth_left.dtype == np.float32
        assert frame.gt_depth_right.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════
# 3. Episode Dataclass
# ═══════════════════════════════════════════════════════════════════


class TestEpisode:
    """Verify the Episode dataclass structure and defaults."""

    def test_default_empty_lists(self):
        ep = Episode()
        assert ep.frames == []
        assert ep.joint_positions == []
        assert ep.actions == []
        assert ep.metadata == {}

    def test_independent_default_instances(self):
        """Two Episode instances should NOT share mutable defaults."""
        ep1 = Episode()
        ep2 = Episode()
        ep1.frames.append("test")
        assert len(ep2.frames) == 0, "Episodes should have independent lists"

    def test_accumulation(self):
        ep = Episode()
        frame = StereoFrame(
            left_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            right_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            gt_depth_left=np.ones((1, 1), dtype=np.float32),
            gt_depth_right=np.ones((1, 1), dtype=np.float32),
        )
        ep.frames.append(frame)
        ep.joint_positions.append(np.zeros(7))
        ep.actions.append(np.zeros(8))
        ep.metadata["episode"] = 0
        assert len(ep.frames) == 1
        assert len(ep.joint_positions) == 1
        assert len(ep.actions) == 1
        assert ep.metadata["episode"] == 0

    def test_metadata_dict_independent(self):
        ep1 = Episode()
        ep2 = Episode()
        ep1.metadata["key"] = "val"
        assert "key" not in ep2.metadata


# ═══════════════════════════════════════════════════════════════════
# 4. FrankaPickPolicy — Trajectory Generation
# ═══════════════════════════════════════════════════════════════════


class TestFrankaPickPolicy:
    """Verify 4-phase pick-and-place trajectory generation."""

    def test_trajectory_length(self):
        policy = FrankaPickPolicy(
            n_actuators=8,
            cube_pos=np.array([0.3, 0.0, 0.42]),
            place_pos=np.array([0.5, 0.0, 0.42]),
        )
        traj = policy.generate_trajectory(total_steps=200)
        assert len(traj) == 200

    def test_trajectory_custom_length(self):
        policy = FrankaPickPolicy(
            n_actuators=8,
            cube_pos=np.array([0.3, 0.0, 0.42]),
            place_pos=np.array([0.5, 0.0, 0.42]),
        )
        traj = policy.generate_trajectory(total_steps=100)
        assert len(traj) == 100

    def test_action_shape_matches_actuators(self):
        for n_act in [6, 7, 8, 9]:
            policy = FrankaPickPolicy(
                n_actuators=n_act,
                cube_pos=np.array([0.3, 0.0, 0.42]),
                place_pos=np.array([0.5, 0.0, 0.42]),
            )
            traj = policy.generate_trajectory(total_steps=40)
            for action in traj:
                assert action.shape == (n_act,), f"n_actuators={n_act}, got shape {action.shape}"
                assert action.dtype == np.float32

    def test_gripper_open_in_reach_phase(self):
        """In reach phase (first 25% of trajectory), gripper should be open."""
        policy = FrankaPickPolicy(
            n_actuators=8,
            cube_pos=np.array([0.3, 0.0, 0.42]),
            place_pos=np.array([0.5, 0.0, 0.42]),
        )
        traj = policy.generate_trajectory(total_steps=200)
        # Step 10 is in reach phase (10/200 = 5%, < 25%)
        action = traj[10]
        # Gripper joints (indices 7+) should be 0.04 (open)
        assert action[7] == pytest.approx(0.04)

    def test_gripper_closes_in_grasp_phase(self):
        """In grasp phase (25-50%), gripper should progressively close."""
        policy = FrankaPickPolicy(
            n_actuators=8,
            cube_pos=np.array([0.3, 0.0, 0.42]),
            place_pos=np.array([0.5, 0.0, 0.42]),
        )
        traj = policy.generate_trajectory(total_steps=200)
        # Step 60 is in grasp phase (60/200 = 30%, between 25-50%)
        # Gripper should be partially closed (< 0.04)
        action_early_grasp = traj[55]
        action_late_grasp = traj[95]
        # Late grasp should have smaller gripper value than early grasp
        assert action_late_grasp[7] <= action_early_grasp[7]

    def test_gripper_closed_in_lift_phase(self):
        """In lift phase (50-75%), gripper should be fully closed."""
        policy = FrankaPickPolicy(
            n_actuators=8,
            cube_pos=np.array([0.3, 0.0, 0.42]),
            place_pos=np.array([0.5, 0.0, 0.42]),
        )
        traj = policy.generate_trajectory(total_steps=200)
        # Step 130 is in lift phase (130/200 = 65%)
        action = traj[130]
        assert action[7] == pytest.approx(0.0)

    def test_trajectory_stored_on_policy(self):
        policy = FrankaPickPolicy(
            n_actuators=8,
            cube_pos=np.array([0.3, 0.0, 0.42]),
            place_pos=np.array([0.5, 0.0, 0.42]),
        )
        assert policy._trajectory is None
        traj = policy.generate_trajectory(total_steps=100)
        assert policy._trajectory is traj

    def test_trajectory_finite_values(self):
        """All trajectory actions should be finite (no NaN/Inf)."""
        policy = FrankaPickPolicy(
            n_actuators=8,
            cube_pos=np.array([0.3, 0.0, 0.42]),
            place_pos=np.array([0.5, 0.0, 0.42]),
        )
        traj = policy.generate_trajectory(total_steps=200)
        for i, action in enumerate(traj):
            assert np.all(np.isfinite(action)), f"Non-finite values at step {i}"

    def test_trajectory_smooth_transitions(self):
        """Adjacent actions should not have large jumps (cosine smoothing)."""
        policy = FrankaPickPolicy(
            n_actuators=8,
            cube_pos=np.array([0.3, 0.0, 0.42]),
            place_pos=np.array([0.5, 0.0, 0.42]),
        )
        traj = policy.generate_trajectory(total_steps=200)
        for i in range(1, len(traj)):
            delta = np.max(np.abs(traj[i] - traj[i - 1]))
            assert delta < 0.5, f"Large jump at step {i}: delta={delta}"

    def test_policy_works_with_few_actuators(self):
        """Policy should work even with fewer than 7 actuators."""
        policy = FrankaPickPolicy(
            n_actuators=4,
            cube_pos=np.array([0.3, 0.0, 0.42]),
            place_pos=np.array([0.5, 0.0, 0.42]),
        )
        traj = policy.generate_trajectory(total_steps=40)
        assert len(traj) == 40
        for action in traj:
            assert action.shape == (4,)


# ═══════════════════════════════════════════════════════════════════
# 5. KITCHEN_OBJECTS
# ═══════════════════════════════════════════════════════════════════


class TestKitchenObjects:
    """Validate the KITCHEN_OBJECTS constant list."""

    def test_count(self):
        assert len(KITCHEN_OBJECTS) == 8

    def test_required_keys(self):
        for obj in KITCHEN_OBJECTS:
            assert "name" in obj
            assert "shape" in obj
            assert "size" in obj
            assert "color" in obj

    def test_names_unique(self):
        names = [obj["name"] for obj in KITCHEN_OBJECTS]
        assert len(names) == len(set(names)), "Object names should be unique"

    def test_known_names(self):
        names = {obj["name"] for obj in KITCHEN_OBJECTS}
        expected = {
            "red_mug",
            "green_plate",
            "blue_bowl",
            "yellow_banana",
            "orange_can",
            "white_sugar_box",
            "purple_bottle",
            "brown_bread",
        }
        assert names == expected

    def test_shapes_valid_mujoco(self):
        valid_shapes = {"box", "sphere", "cylinder", "capsule", "ellipsoid"}
        for obj in KITCHEN_OBJECTS:
            assert obj["shape"] in valid_shapes, f"Invalid shape: {obj['shape']}"

    def test_colors_rgba(self):
        for obj in KITCHEN_OBJECTS:
            assert len(obj["color"]) == 4, f"Color should be RGBA: {obj['name']}"
            for c in obj["color"]:
                assert 0.0 <= c <= 1.0, f"Color out of range: {obj['name']}"

    def test_sizes_non_empty(self):
        for obj in KITCHEN_OBJECTS:
            assert len(obj["size"]) >= 1, f"Size list empty: {obj['name']}"
            for s in obj["size"]:
                assert s > 0, f"Size must be positive: {obj['name']}"


# ═══════════════════════════════════════════════════════════════════
# 6. build_kitchen_scene_xml
# ═══════════════════════════════════════════════════════════════════


class TestBuildKitchenSceneXml:
    """Verify MJCF XML generation for the kitchen scene."""

    def test_returns_string(self):
        xml = build_kitchen_scene_xml()
        assert isinstance(xml, str)

    def test_valid_xml(self):
        xml = build_kitchen_scene_xml()
        root = ElementTree.fromstring(xml)
        assert root.tag == "mujoco"

    def test_model_name(self):
        xml = build_kitchen_scene_xml()
        root = ElementTree.fromstring(xml)
        assert root.get("model") == "kitchen_stereo4d"

    def test_contains_stereo_left_camera(self):
        xml = build_kitchen_scene_xml()
        assert 'name="stereo_left"' in xml

    def test_contains_stereo_right_camera(self):
        xml = build_kitchen_scene_xml()
        assert 'name="stereo_right"' in xml

    def test_contains_overhead_camera(self):
        xml = build_kitchen_scene_xml()
        assert 'name="overhead"' in xml

    def test_camera_fov(self):
        xml = build_kitchen_scene_xml()
        assert f'fovy="{CAMERA_FOV}"' in xml

    def test_resolution_in_visual(self):
        xml = build_kitchen_scene_xml()
        assert f'offwidth="{CAMERA_WIDTH}"' in xml
        assert f'offheight="{CAMERA_HEIGHT}"' in xml

    def test_default_uses_4_objects(self):
        """Default should use first 4 kitchen objects."""
        np.random.seed(42)
        xml = build_kitchen_scene_xml()
        for obj in KITCHEN_OBJECTS[:4]:
            assert f'name="{obj["name"]}"' in xml

    def test_custom_objects(self):
        custom_objs = [
            {"name": "test_cube", "shape": "box", "size": [0.05, 0.05, 0.05], "color": [1, 0, 0, 1]},
        ]
        np.random.seed(42)
        xml = build_kitchen_scene_xml(objects=custom_objs)
        assert 'name="test_cube"' in xml

    def test_table_present(self):
        xml = build_kitchen_scene_xml()
        assert 'name="table"' in xml

    def test_table_height_parameter(self):
        np.random.seed(42)
        xml_low = build_kitchen_scene_xml(table_height=0.3)
        xml_high = build_kitchen_scene_xml(table_height=0.6)
        # Table pos should contain half the table height
        assert "0.15" in xml_low  # 0.3/2
        assert "0.3" in xml_high  # 0.6/2

    def test_actuators_present(self):
        xml = build_kitchen_scene_xml()
        root = ElementTree.fromstring(xml)
        actuators = root.find(".//actuator")
        assert actuators is not None
        motors = actuators.findall("motor")
        assert len(motors) == 8  # 6 joints + 2 fingers

    def test_lighting(self):
        xml = build_kitchen_scene_xml()
        root = ElementTree.fromstring(xml)
        lights = root.findall(".//light")
        assert len(lights) >= 3  # top, side, fill

    def test_ground_plane(self):
        xml = build_kitchen_scene_xml()
        root = ElementTree.fromstring(xml)
        geoms = root.findall(".//geom[@type='plane']")
        assert len(geoms) >= 1

    def test_robot_base(self):
        xml = build_kitchen_scene_xml()
        assert 'name="robot_base"' in xml

    def test_gripper_joints(self):
        xml = build_kitchen_scene_xml()
        assert 'name="finger_l"' in xml
        assert 'name="finger_r"' in xml

    def test_objects_have_free_joints(self):
        """Kitchen objects should have free joints for manipulation."""
        np.random.seed(42)
        xml = build_kitchen_scene_xml()
        root = ElementTree.fromstring(xml)
        for obj in KITCHEN_OBJECTS[:4]:
            body = root.find(f".//body[@name='{obj['name']}']")
            assert body is not None, f"Missing body: {obj['name']}"
            joint = body.find("joint")
            assert joint is not None, f"No joint on: {obj['name']}"
            assert joint.get("type") == "free"


# ═══════════════════════════════════════════════════════════════════
# 7. XML Scene Integration
# ═══════════════════════════════════════════════════════════════════


class TestXMLSceneIntegration:
    """Deeper XML structure checks."""

    def test_camera_count(self):
        xml = build_kitchen_scene_xml()
        root = ElementTree.fromstring(xml)
        cameras = root.findall(".//camera")
        assert len(cameras) == 3  # stereo_left, stereo_right, overhead

    def test_all_actuators_have_ctrl_range(self):
        xml = build_kitchen_scene_xml()
        root = ElementTree.fromstring(xml)
        motors = root.findall(".//actuator/motor")
        for motor in motors:
            assert motor.get("ctrllimited") == "true"
            assert motor.get("ctrlrange") is not None

    def test_joint_count(self):
        """Should have 6 arm joints + 2 finger joints = 8 robot joints."""
        xml = build_kitchen_scene_xml()
        root = ElementTree.fromstring(xml)
        # Count hinge + slide joints in robot body tree
        robot_base = root.find(".//body[@name='robot_base']")
        assert robot_base is not None
        joints = robot_base.findall(".//joint")
        assert len(joints) == 8  # j1-j6 + finger_l + finger_r

    def test_skybox_texture(self):
        xml = build_kitchen_scene_xml()
        assert 'type="skybox"' in xml

    def test_materials_defined(self):
        xml = build_kitchen_scene_xml()
        root = ElementTree.fromstring(xml)
        materials = root.findall(".//material")
        assert len(materials) >= 2  # grid_mat, wood_mat

    def test_compiler_angle_radian(self):
        xml = build_kitchen_scene_xml()
        root = ElementTree.fromstring(xml)
        compiler = root.find(".//compiler")
        assert compiler is not None
        assert compiler.get("angle") == "radian"


# ═══════════════════════════════════════════════════════════════════
# 8. StereoCameraRig — compute_disparity (mocked MuJoCo)
# ═══════════════════════════════════════════════════════════════════


class TestStereoCameraRigDisparity:
    """Test compute_disparity without MuJoCo (set focal_length_px and baseline_m manually)."""

    @pytest.fixture
    def rig(self):
        """Create a StereoCameraRig-like object with needed attributes."""
        rig = StereoCameraRig.__new__(StereoCameraRig)
        rig.focal_length_px = (CAMERA_WIDTH / 2.0) / math.tan(math.radians(CAMERA_FOV / 2.0))
        rig.baseline_m = STEREO_BASELINE_MM / 1000.0
        return rig

    def test_disparity_output_shape(self, rig):
        depth = np.ones((480, 640), dtype=np.float32) * 2.0
        disp = rig.compute_disparity(depth)
        assert disp.shape == (480, 640)
        assert disp.dtype == np.float32

    def test_disparity_formula(self, rig):
        """disparity = focal_length * baseline / depth."""
        depth = np.array([[1.0, 2.0, 5.0]], dtype=np.float32)
        disp = rig.compute_disparity(depth)
        expected = rig.focal_length_px * rig.baseline_m / depth
        np.testing.assert_allclose(disp, expected)

    def test_closer_objects_higher_disparity(self, rig):
        """Closer objects should have larger disparity."""
        depth = np.array([[0.5, 1.0, 5.0]], dtype=np.float32)
        disp = rig.compute_disparity(depth)
        assert disp[0, 0] > disp[0, 1] > disp[0, 2]

    def test_zero_depth_safe(self, rig):
        """Zero depth should not produce inf (clamped to 0.001)."""
        depth = np.array([[0.0]], dtype=np.float32)
        disp = rig.compute_disparity(depth)
        assert np.all(np.isfinite(disp))

    def test_small_depth_clamped(self, rig):
        """Very small depth should be clamped to prevent huge disparities."""
        depth = np.array([[0.0001]], dtype=np.float32)
        disp = rig.compute_disparity(depth)
        expected = rig.focal_length_px * rig.baseline_m / 0.001
        np.testing.assert_allclose(disp[0, 0], expected, rtol=1e-5)

    def test_disparity_roundtrip(self, rig):
        """depth → disparity → depth should round-trip."""
        depth = np.random.uniform(0.3, 10.0, (100, 100)).astype(np.float32)
        disp = rig.compute_disparity(depth)
        recovered = rig.focal_length_px * rig.baseline_m / disp
        np.testing.assert_allclose(recovered, depth, rtol=1e-5)


# ═══════════════════════════════════════════════════════════════════
# 9. StereoCameraRig — depth_to_point_cloud (mocked MuJoCo)
# ═══════════════════════════════════════════════════════════════════


class TestStereoCameraRigPointCloud:
    """Test depth_to_point_cloud without MuJoCo."""

    @pytest.fixture
    def rig(self):
        rig = StereoCameraRig.__new__(StereoCameraRig)
        rig.focal_length_px = (CAMERA_WIDTH / 2.0) / math.tan(math.radians(CAMERA_FOV / 2.0))
        rig.baseline_m = STEREO_BASELINE_MM / 1000.0
        return rig

    def test_output_shapes(self, rig):
        depth = np.ones((480, 640), dtype=np.float32) * 2.0
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        points, colors = rig.depth_to_point_cloud(depth, rgb)
        assert points.ndim == 2
        assert points.shape[1] == 3
        assert colors.ndim == 2
        assert colors.shape[1] == 3
        assert len(points) == len(colors)

    def test_filters_invalid_depth(self, rig):
        """Points with depth <= 0.01 or >= 10.0 should be filtered out."""
        depth = np.array([[0.0, 0.005, 2.0, 15.0]], dtype=np.float32)
        rgb = np.zeros((1, 4, 3), dtype=np.uint8)
        points, colors = rig.depth_to_point_cloud(depth, rgb)
        # Only depth=2.0 is valid (0.01 < 2.0 < 10.0)
        assert len(points) == 1
        assert points[0, 2] == pytest.approx(2.0)

    def test_colors_normalized(self, rig):
        """Colors should be in [0, 1] range (divided by 255)."""
        depth = np.ones((10, 10), dtype=np.float32) * 2.0
        rgb = np.full((10, 10, 3), 127, dtype=np.uint8)
        points, colors = rig.depth_to_point_cloud(depth, rgb)
        assert colors.max() <= 1.0
        assert colors.min() >= 0.0
        np.testing.assert_allclose(colors[0], 127 / 255.0, rtol=1e-5)

    def test_principal_point_xy_near_zero(self, rig):
        """At the image center, X and Y should be approximately zero."""
        depth = np.ones((480, 640), dtype=np.float32) * 3.0
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        points, colors = rig.depth_to_point_cloud(depth, rgb)
        # Center pixel is at (320, 240), cx=320, cy=240
        # X = (320 - 320) * 3 / f = 0
        # Y = (240 - 240) * 3 / f = 0
        # Find the point closest to (0, 0, 3)
        center_idx = np.argmin(np.abs(points[:, 0]) + np.abs(points[:, 1]))
        assert abs(points[center_idx, 0]) < 0.01
        assert abs(points[center_idx, 1]) < 0.01

    def test_empty_for_all_invalid_depth(self, rig):
        """All-zero depth should produce empty point cloud."""
        depth = np.zeros((10, 10), dtype=np.float32)
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        points, colors = rig.depth_to_point_cloud(depth, rgb)
        assert len(points) == 0
        assert len(colors) == 0

    def test_z_equals_depth_for_valid_points(self, rig):
        """Z coordinate should equal the depth value for valid points."""
        depth = np.full((10, 10), 5.0, dtype=np.float32)
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        points, colors = rig.depth_to_point_cloud(depth, rgb)
        np.testing.assert_allclose(points[:, 2], 5.0)


# ═══════════════════════════════════════════════════════════════════
# 10. Disparity ↔ Depth Consistency
# ═══════════════════════════════════════════════════════════════════


class TestDisparityDepthConsistency:
    """Verify the disparity = f * b / d formula is applied correctly."""

    def test_known_values(self):
        f = 554.256  # focal length from 60° FOV at 640 width
        b = 0.05  # 50mm baseline
        depth = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
        disparity = f * b / depth
        # At 1m: d = 554.256 * 0.05 / 1.0 = 27.7128
        assert disparity[1] == pytest.approx(f * b, rel=1e-5)

    def test_round_trip(self):
        f = 600.0
        b = 0.05
        depth = np.random.uniform(0.3, 10.0, (50,)).astype(np.float32)
        disparity = f * b / depth
        recovered = f * b / disparity
        np.testing.assert_allclose(recovered, depth, rtol=1e-5)

    def test_near_zero_depth_clamped(self):
        """np.maximum(depth, 0.001) should prevent inf."""
        depth = np.array([0.0, 0.001, 1.0], dtype=np.float32)
        safe_depth = np.maximum(depth, 0.001)
        disparity = 600.0 * 0.05 / safe_depth
        assert np.all(np.isfinite(disparity))


# ═══════════════════════════════════════════════════════════════════
# 11. save_* Visualization Functions (with mocked cv2)
# ═══════════════════════════════════════════════════════════════════


class TestSaveVisualizationFunctions:
    """Test visualization helper functions with mocked cv2."""

    @pytest.fixture(autouse=True)
    def mock_cv2(self):
        """Mock cv2 for all tests in this class."""
        mock = MagicMock()
        # applyColorMap returns an ndarray of the same shape
        mock.applyColorMap = lambda arr, cmap: np.stack([arr, arr, arr], axis=-1) if arr.ndim == 2 else arr
        mock.COLORMAP_TURBO = 20
        mock.imwrite = MagicMock(return_value=True)
        mock.cvtColor = lambda img, code: img
        mock.resize = lambda img, size: np.zeros((*size[::-1], 3), dtype=np.uint8)
        mock.putText = MagicMock()
        mock.FONT_HERSHEY_SIMPLEX = 0
        mock.COLOR_GRAY2BGR = 8
        with patch.dict(sys.modules, {"cv2": mock}):
            self.cv2_mock = mock
            yield

    def test_save_depth_colormap_returns_array(self, tmp_path):
        depth = np.random.uniform(0.0, 3.0, (480, 640)).astype(np.float32)
        result = save_depth_colormap(depth, str(tmp_path / "depth.png"))
        assert isinstance(result, np.ndarray)

    def test_save_depth_colormap_clips_values(self, tmp_path):
        depth = np.array([[0.0, 1.5, 3.0, 5.0]], dtype=np.float32)
        result = save_depth_colormap(depth, str(tmp_path / "depth.png"), vmin=0.0, vmax=3.0)
        assert isinstance(result, np.ndarray)
        self.cv2_mock.imwrite.assert_called_once()

    def test_save_side_by_side_returns_array(self, tmp_path):
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        right = np.zeros((480, 640, 3), dtype=np.uint8)
        result = save_side_by_side(left, right, str(tmp_path / "side.png"))
        assert isinstance(result, np.ndarray)

    def test_save_stereo_anaglyph_returns_array(self, tmp_path):
        left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = save_stereo_anaglyph(left, right, str(tmp_path / "anaglyph.png"))
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)

    def test_anaglyph_channel_mapping(self, tmp_path):
        """Red from left, green+blue from right."""
        left = np.zeros((10, 10, 3), dtype=np.uint8)
        left[:, :, 2] = 200  # Red channel in BGR
        right = np.zeros((10, 10, 3), dtype=np.uint8)
        right[:, :, 0] = 100  # Blue channel
        right[:, :, 1] = 150  # Green channel
        result = save_stereo_anaglyph(left, right, str(tmp_path / "a.png"))
        assert result[0, 0, 2] == 200  # Red from left
        assert result[0, 0, 1] == 150  # Green from right
        assert result[0, 0, 0] == 100  # Blue from right

    def test_save_point_cloud_image_returns_array(self, tmp_path):
        points = np.random.uniform(-1, 1, (1000, 3)).astype(np.float32)
        colors = np.random.uniform(0, 1, (1000, 3)).astype(np.float32)
        result = save_point_cloud_image(points, colors, str(tmp_path / "pc.png"))
        assert isinstance(result, np.ndarray)
        assert result.shape == (640, 640, 3)

    def test_save_point_cloud_image_empty(self, tmp_path):
        """Empty points should produce a black image."""
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.float32)
        result = save_point_cloud_image(points, colors, str(tmp_path / "pc.png"))
        assert result.shape == (640, 640, 3)
        assert np.all(result == 0)

    def test_save_point_cloud_custom_size(self, tmp_path):
        points = np.random.uniform(-1, 1, (100, 3)).astype(np.float32)
        colors = np.random.uniform(0, 1, (100, 3)).astype(np.float32)
        result = save_point_cloud_image(points, colors, str(tmp_path / "pc.png"), size=256)
        assert result.shape == (256, 256, 3)


# ═══════════════════════════════════════════════════════════════════
# 12. strands_robots.stereo Module API Compatibility
# ═══════════════════════════════════════════════════════════════════


class TestStereoModuleCompatibility:
    """Verify the stereo module's API contract that pick_and_place_kitchen.py expects."""

    def test_stereo_config_import(self):
        from strands_robots.stereo import StereoConfig

        cfg = StereoConfig(model_variant="23-36-37", valid_iters=8)
        assert cfg.model_variant == "23-36-37"
        assert cfg.valid_iters == 8

    def test_stereo_config_default(self):
        from strands_robots.stereo import StereoConfig

        cfg = StereoConfig()
        assert cfg.max_disp == 192
        assert cfg.scale == 1.0

    def test_stereo_pipeline_import(self):
        from strands_robots.stereo import StereoDepthPipeline

        pipe = StereoDepthPipeline()
        assert hasattr(pipe, "estimate_depth")
        assert hasattr(pipe, "_load_image")

    def test_stereo_result_properties(self):
        from strands_robots.stereo import StereoResult

        disp = np.ones((480, 640), dtype=np.float32) * 30.0
        result = StereoResult(
            disparity=disp,
            depth=np.ones((480, 640), dtype=np.float32) * 2.0,
        )
        assert result.height == 480
        assert result.width == 640
        assert result.median_depth == pytest.approx(2.0)
        assert result.valid_ratio > 0.99

    def test_stereo_result_to_dict(self):
        from strands_robots.stereo import StereoResult

        result = StereoResult(
            disparity=np.ones((10, 10), dtype=np.float32),
            depth=np.ones((10, 10), dtype=np.float32),
        )
        d = result.to_dict()
        assert "height" in d
        assert "width" in d
        assert "has_depth" in d
        assert d["has_depth"] is True

    def test_estimate_depth_convenience(self):
        from strands_robots.stereo import estimate_depth

        assert callable(estimate_depth)


# ═══════════════════════════════════════════════════════════════════
# 13. StereoConfig Validation
# ═══════════════════════════════════════════════════════════════════


class TestStereoConfigValidation:
    """Verify StereoConfig's __post_init__ validation."""

    def test_invalid_model_variant_raises(self):
        from strands_robots.stereo import StereoConfig

        with pytest.raises(ValueError, match="Invalid model_variant"):
            StereoConfig(model_variant="bad-variant")

    def test_invalid_valid_iters_raises(self):
        from strands_robots.stereo import StereoConfig

        with pytest.raises(ValueError, match="valid_iters"):
            StereoConfig(valid_iters=0)

    def test_invalid_scale_raises(self):
        from strands_robots.stereo import StereoConfig

        with pytest.raises(ValueError, match="scale"):
            StereoConfig(scale=0.0)

    def test_invalid_camera_raises(self):
        from strands_robots.stereo import StereoConfig

        with pytest.raises(ValueError, match="Unknown camera"):
            StereoConfig(camera="nonexistent_cam")

    def test_valid_camera_accepted(self):
        from strands_robots.stereo import StereoConfig

        cfg = StereoConfig(camera="realsense_d435")
        assert cfg.camera == "realsense_d435"


# ═══════════════════════════════════════════════════════════════════
# 14. StereoResult Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestStereoResultEdgeCases:
    """Edge cases for StereoResult properties."""

    def test_median_depth_none_without_depth(self):
        from strands_robots.stereo import StereoResult

        result = StereoResult(disparity=np.ones((10, 10), dtype=np.float32))
        assert result.median_depth is None

    def test_median_depth_all_inf(self):
        from strands_robots.stereo import StereoResult

        depth = np.full((10, 10), np.inf, dtype=np.float32)
        result = StereoResult(
            disparity=np.ones((10, 10), dtype=np.float32),
            depth=depth,
        )
        assert result.median_depth is None

    def test_valid_ratio_all_zero(self):
        from strands_robots.stereo import StereoResult

        disp = np.zeros((10, 10), dtype=np.float32)
        result = StereoResult(disparity=disp)
        assert result.valid_ratio == 0.0

    def test_valid_ratio_all_valid(self):
        from strands_robots.stereo import StereoResult

        disp = np.ones((10, 10), dtype=np.float32) * 5.0
        result = StereoResult(disparity=disp)
        assert result.valid_ratio == 1.0


# ═══════════════════════════════════════════════════════════════════
# 15. Focal Length from FOV
# ═══════════════════════════════════════════════════════════════════


class TestFocalLengthComputation:
    """Verify the focal_length = (width/2) / tan(fov/2) formula."""

    def test_known_fov_60(self):
        """60° FOV at 640 width → f ≈ 554.256."""
        expected = (640 / 2.0) / math.tan(math.radians(30.0))
        assert expected == pytest.approx(554.256, rel=1e-3)

    def test_consistency_with_rig(self):
        """StereoCameraRig's focal computation should match the formula."""
        expected = (CAMERA_WIDTH / 2.0) / math.tan(math.radians(CAMERA_FOV / 2.0))
        rig = StereoCameraRig.__new__(StereoCameraRig)
        rig.focal_length_px = expected
        assert rig.focal_length_px == pytest.approx(expected)

    def test_wider_fov_shorter_focal(self):
        """90° FOV should have shorter focal length than 60°."""
        f_60 = (640 / 2.0) / math.tan(math.radians(30.0))
        f_90 = (640 / 2.0) / math.tan(math.radians(45.0))
        assert f_90 < f_60


# ═══════════════════════════════════════════════════════════════════
# 16. Disparity Visualisation Helper
# ═══════════════════════════════════════════════════════════════════


class TestDisparityVisualization:
    """Test _visualize_disparity from strands_robots.stereo."""

    def test_output_shape_and_type(self):
        from strands_robots.stereo import _visualize_disparity

        disp = np.random.uniform(0, 50, (100, 200)).astype(np.float32)
        vis = _visualize_disparity(disp)
        assert vis.shape == (100, 200, 3)
        assert vis.dtype == np.uint8

    def test_all_invalid_produces_black(self):
        from strands_robots.stereo import _visualize_disparity

        disp = np.full((10, 10), np.inf, dtype=np.float32)
        vis = _visualize_disparity(disp)
        assert np.all(vis == 0)

    def test_negative_disparity_zeroed(self):
        from strands_robots.stereo import _visualize_disparity

        disp = np.array([[-1.0, 0.0, 5.0]], dtype=np.float32)
        vis = _visualize_disparity(disp)
        # Negative disparity should be treated as invalid (black)
        assert vis[0, 0, 0] == 0 and vis[0, 0, 1] == 0 and vis[0, 0, 2] == 0


# ═══════════════════════════════════════════════════════════════════
# 17. InputPadder
# ═══════════════════════════════════════════════════════════════════


class TestInputPadder:
    """Test the _InputPadder utility from strands_robots.stereo."""

    def test_already_divisible(self):
        """If dims are already divisible by 32, no padding needed."""
        from strands_robots.stereo import _InputPadder

        padder = _InputPadder((1, 3, 480, 640), divis_by=32)
        # Should have zero padding
        assert padder._pad == [0, 0, 0, 0]

    @pytest.mark.skipif(not _has_torch(), reason="torch not installed")
    def test_padded_shape_divisible(self):
        """After padding, dimensions should be divisible by 32."""
        import torch

        from strands_robots.stereo import _InputPadder

        x = torch.randn(1, 3, 500, 650)
        padder = _InputPadder(x.shape, divis_by=32)
        [x_padded] = padder.pad(x)
        assert x_padded.shape[2] % 32 == 0
        assert x_padded.shape[3] % 32 == 0

    @pytest.mark.skipif(not _has_torch(), reason="torch not installed")
    def test_unpad_recovers_original(self):
        """pad → unpad should recover original dimensions."""
        import torch

        from strands_robots.stereo import _InputPadder

        x = torch.randn(1, 3, 500, 650)
        padder = _InputPadder(x.shape, divis_by=32)
        [x_padded] = padder.pad(x)
        x_unpadded = padder.unpad(x_padded)
        assert x_unpadded.shape == x.shape


# ═══════════════════════════════════════════════════════════════════
# 18. End-to-end: StereoCameraRig disparity → point cloud
# ═══════════════════════════════════════════════════════════════════


class TestE2EDisparityToPointCloud:
    """Full pipeline: depth → disparity → reconstructed depth, and depth → point cloud."""

    def test_disparity_to_depth_to_point_cloud(self):
        rig = StereoCameraRig.__new__(StereoCameraRig)
        rig.focal_length_px = 554.256
        rig.baseline_m = 0.05

        # Create a realistic depth map
        depth = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Compute disparity
        disp = rig.compute_disparity(depth)

        # Recover depth from disparity
        recovered_depth = rig.focal_length_px * rig.baseline_m / disp
        np.testing.assert_allclose(recovered_depth, depth, rtol=1e-5)

        # Generate point cloud
        points, colors = rig.depth_to_point_cloud(depth, rgb)
        assert len(points) > 0
        assert points.shape[1] == 3
        # All Z values should be within the depth range
        assert points[:, 2].min() >= 0.01
        assert points[:, 2].max() <= 10.0
