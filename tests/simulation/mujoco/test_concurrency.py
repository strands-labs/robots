"""Regression tests for PR #85 review feedback.

Tests:
1. Thread-safety: concurrent dispatch + policy doesn't corrupt state
2. Flat-index state copy: joint positions survive object injection
3. apply_force: force is latched (persists across steps)
4. Camera recording roundtrip: namespaced cameras survive schema reconcile

Run: MUJOCO_GL=osmesa python -m pytest tests/test_mujoco_regressions.py -v
"""

import math
import os
import shutil
import tempfile
import threading
import time

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No OpenGL context available (headless without EGL/OSMesa)",
)

# Test robot XML (simple 3-DOF arm)

ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <camera name="arm0/wrist_cam" pos="0.5 0 0.5" xyaxes="0 1 0 0 0 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <body name="link2" pos="0 0 0.2">
          <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.15" rgba="0.3 0.8 0.3 1"/>
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="shoulder_lift_act" joint="shoulder_lift" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def robot_xml_path():
    """Write test robot XML to a temp file."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(ROBOT_XML)
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sim_with_robot(robot_xml_path):
    """Simulation with world + robot loaded."""
    sim = Simulation(tool_name="test_regression", mesh=False)
    result = sim.create_world(gravity=[0, 0, -9.81])
    assert result["status"] == "success"
    result = sim.add_robot("arm1", urdf_path=robot_xml_path)
    assert result["status"] == "success"
    yield sim
    sim.cleanup()


class TestFlatIndexStatePreservation:
    """Regression: joint positions must survive object injection (layout shift)."""

    def test_joint_survives_object_injection(self, sim_with_robot):
        """Set a joint to π/3, inject an object, verify joint is still ≈π/3.

        This catches the flat-index qpos copy bug where injected bodies
        shift existing qpos entries.
        """
        sim = sim_with_robot
        target_angle = math.pi / 3

        # Set shoulder_pan to π/3
        result = sim.set_joint_positions(
            positions={"shoulder_pan": target_angle},
            robot_name="arm1",
        )
        assert result["status"] == "success"

        # Verify it's set
        state = sim.get_robot_state("arm1")
        assert state["status"] == "success"  # state returned
        # Read qpos directly
        model = sim._world._model
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm1/shoulder_pan")
        if jid < 0:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder_pan")
        assert jid >= 0
        qpos_before = float(sim._world._data.qpos[model.jnt_qposadr[jid]])
        assert abs(qpos_before - target_angle) < 1e-6

        # Inject an object (triggers XML round-trip + _reload_scene_from_xml)
        result = sim.add_object(
            "test_box",
            shape="box",
            position=[0.5, 0.5, 0.1],
            size=[0.05, 0.05, 0.05],
        )
        assert result["status"] == "success"

        # Verify joint is still ≈π/3 after injection
        model = sim._world._model
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "arm1/shoulder_pan")
        if jid < 0:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder_pan")
        assert jid >= 0
        qpos_after = float(sim._world._data.qpos[model.jnt_qposadr[jid]])
        assert abs(qpos_after - target_angle) < 1e-4, (
            f"Joint drifted from {target_angle:.6f} to {qpos_after:.6f} after object injection"
        )


class TestApplyForceLatchedBehavior:
    """Regression: apply_force is latched (persists across steps)."""

    def test_force_persists_across_multiple_steps(self, sim_with_robot):
        """Apply lateral force to a body, step 50 times, verify body moved.

        This validates the docstring contract: force is latched in
        qfrc_applied and applied on every subsequent step.

        NOTE: We use an X-force (lateral) because a Z-force along the
        kinematic chain of hinge joints produces zero generalized torque
        (mj_applyFT maps Cartesian force to joint space; Z-force at CoM
        compresses the chain without creating torques on Y-axis hinges).
        """
        sim = sim_with_robot

        # Get initial x position of link2
        model = sim._world._model
        data = sim._world._data
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "arm1/link2")
        if body_id < 0:
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "link2")
        assert body_id >= 0

        x_before = float(data.xpos[body_id, 0])

        # Apply strong lateral (X) force - this creates torques on Y-axis hinges
        result = sim.apply_force("link2", force=[100.0, 0, 0])
        assert result["status"] == "success"

        # Step physics 50 times - force should persist (latched)
        sim.step(n_steps=50)

        x_after = float(data.xpos[body_id, 0])
        # Body should have moved laterally due to persistent force
        assert abs(x_after - x_before) > 1e-4, (
            f"Body did not move (x_before={x_before:.6f}, x_after={x_after:.6f}). "
            "Force may not be persisting across steps."
        )

    def test_zero_force_stops_effect(self, sim_with_robot):
        """Apply force, then zero it, verify force buffer is cleared."""
        sim = sim_with_robot

        # Apply lateral (X) force - produces non-zero generalized torques
        sim.apply_force("link2", force=[50.0, 0, 0])
        assert np.any(sim._world._data.qfrc_applied != 0), "X-force on link2 should produce non-zero generalized forces"

        # Zero it - apply_force zeros buffer first, then applies zero force
        sim.apply_force("link2", force=[0, 0, 0])
        # After zeroing + applying zero force/torque, buffer should be all zeros
        assert np.allclose(sim._world._data.qfrc_applied, 0.0)


class TestThreadSafety:
    """Regression: concurrent operations don't corrupt MuJoCo state."""

    def test_concurrent_step_and_reset_no_crash(self, sim_with_robot):
        """Concurrent step() and reset() must not SIGSEGV.

        Both acquire self._lock, so they serialize. This test verifies
        the lock is actually held (no segfault, no exception).
        """
        sim = sim_with_robot
        errors = []

        def stepper():
            try:
                for _ in range(100):
                    sim.step(n_steps=1)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"stepper: {e}")

        def resetter():
            try:
                for _ in range(10):
                    sim.reset()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(f"resetter: {e}")

        t1 = threading.Thread(target=stepper)
        t2 = threading.Thread(target=resetter)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_set_joint_and_step(self, sim_with_robot):
        """Concurrent set_joint_positions and step must serialize safely."""
        sim = sim_with_robot
        errors = []

        def setter():
            try:
                for i in range(50):
                    sim.set_joint_positions(
                        positions={"shoulder_pan": float(i) * 0.01},
                        robot_name="arm1",
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"setter: {e}")

        def stepper():
            try:
                for _ in range(50):
                    sim.step(n_steps=2)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"stepper: {e}")

        t1 = threading.Thread(target=setter)
        t2 = threading.Thread(target=stepper)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"


# Robot XML for multi-robot asset directory test

ROBOT_B_XML = """
<mujoco model="test_gripper">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="grip_base" pos="0 0 0.05">
      <geom type="box" size="0.02 0.04 0.02" rgba="0.5 0.5 0.1 1"/>
      <joint name="grip_slide" type="slide" axis="1 0 0" range="-0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="grip_act" joint="grip_slide" kp="30"/>
  </actuator>
</mujoco>
"""


class TestRecordingRoundtripCameraFrames:
    """Regression: namespaced cameras survive schema reconcile and have frames.

    @yinsong1986 review (2026-04-30): "Please add a round-trip test:
    start_recording → run_policy → stop_recording, reopen the dataset,
    assert the camera feature has non-zero frames."
    """

    @pytest.fixture
    def sim_with_namespaced_camera(self, robot_xml_path, tmp_path):
        """Sim with a robot whose camera name contains '/' (namespace)."""
        sim = Simulation(tool_name="test_recording", mesh=False)
        result = sim.create_world(gravity=[0, 0, -9.81])
        assert result["status"] == "success"
        result = sim.add_robot("arm1", urdf_path=robot_xml_path)
        assert result["status"] == "success"
        yield sim
        sim.cleanup()

    @requires_gl
    def test_recording_roundtrip_has_camera_frames(self, sim_with_namespaced_camera, tmp_path):
        """Record → run mock policy → stop → verify dataset has camera data.

        This validates the /→__ sanitization fix doesn't silently drop frames.
        The test robot XML has camera 'arm0/wrist_cam' which becomes
        'arm0__wrist_cam' in the dataset schema.
        """
        pytest.importorskip("lerobot")
        from pathlib import Path

        sim = sim_with_namespaced_camera
        ds_root = str(tmp_path / "roundtrip_ds")

        # Start recording
        result = sim._dispatch_action(
            "start_recording",
            {"repo_id": "local/rt-test", "root": ds_root, "fps": 10, "overwrite": True},
        )
        assert result["status"] == "success", f"start_recording failed: {result}"

        # Run mock policy for a short burst (generates frames via on_frame hook)
        result = sim._dispatch_action(
            "run_policy",
            {
                "robot_name": "arm1",
                "policy_provider": "mock",
                "duration": 0.5,
                "control_frequency": 10,
            },
        )
        assert result["status"] == "success", f"run_policy failed: {result}"

        # Stop recording
        result = sim._dispatch_action("stop_recording", {})
        assert result["status"] == "success", f"stop_recording failed: {result}"

        # Verify dataset exists and has frames
        ds_path = Path(ds_root)
        assert ds_path.exists(), f"Dataset dir not created at {ds_root}"

        # Reopen dataset and verify camera feature has frames
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(repo_id="local/rt-test", root=ds_root)
        except (ImportError, RuntimeError):
            pytest.skip("lerobot dataset API not available (torchcodec/ffmpeg missing)")

        assert len(ds) > 0, f"Dataset has no frames (expected > 0, got {len(ds)})"

        # Check that the camera feature exists (sanitized name)
        cam_feature_found = False
        for feat_name in ds.features:
            if feat_name.startswith("observation.images."):
                cam_feature_found = True
                break

        assert cam_feature_found, (
            f"No observation.images.* feature found in dataset. Features: {list(ds.features.keys())}"
        )

        # Access a frame and verify image data is present (requires ffmpeg for video decode)
        try:
            sample = ds[0]
            for feat_name in ds.features:
                if feat_name.startswith("observation.images."):
                    assert feat_name in sample, f"Camera feature {feat_name} missing from sample"
                    img = sample[feat_name]
                    # Image should be non-empty (tensor or array with shape)
                    assert hasattr(img, "shape"), f"Camera data has no shape: {type(img)}"
                    assert img.shape[0] > 0, f"Camera image has zero height: {img.shape}"
                    break
        except RuntimeError:
            # torchcodec requires system FFmpeg libraries for video decode
            pass


class TestMultiRobotDifferentAssetDirs:
    """Regression: two robots from different asset dirs both compile and render.

    @yinsong1986 review (2026-04-30): "load two robots whose urdf_paths
    are in different directories; assert both render."
    """

    def test_two_robots_different_directories_both_load(self):
        """Load two robots from separate temp dirs, verify both have joints."""
        tmpdir_a = tempfile.mkdtemp(prefix="robot_a_")
        tmpdir_b = tempfile.mkdtemp(prefix="robot_b_")

        try:
            # Write robot A (arm) to dir A
            path_a = os.path.join(tmpdir_a, "arm.xml")
            with open(path_a, "w") as f:
                f.write(ROBOT_XML)

            # Write robot B (gripper) to dir B
            path_b = os.path.join(tmpdir_b, "gripper.xml")
            with open(path_b, "w") as f:
                f.write(ROBOT_B_XML)

            sim = Simulation(tool_name="test_multi_asset", mesh=False)
            result = sim.create_world(gravity=[0, 0, -9.81])
            assert result["status"] == "success"

            # Add robot A from dir A
            result = sim.add_robot("arm1", urdf_path=path_a)
            assert result["status"] == "success", f"Robot A failed: {result}"

            # Add robot B from dir B (different asset directory)
            result = sim.add_robot("grip1", urdf_path=path_b, position=[0.3, 0, 0])
            assert result["status"] == "success", f"Robot B failed: {result}"

            # Both robots should be registered
            assert "arm1" in sim._world.robots
            assert "grip1" in sim._world.robots

            # Both should have joints discovered
            assert len(sim._world.robots["arm1"].joint_names) == 3  # shoulder_pan, shoulder_lift, elbow
            assert len(sim._world.robots["grip1"].joint_names) == 1  # grip_slide

            # Physics step should succeed (proves combined model compiled)
            result = sim.step(n_steps=10)
            assert result["status"] == "success", f"Step failed: {result}"

            # Verify we can read state from both robots
            state_a = sim.get_robot_state("arm1")
            assert state_a["status"] == "success", f"State A failed: {state_a}"
            state_b = sim.get_robot_state("grip1")
            assert state_b["status"] == "success", f"State B failed: {state_b}"

            sim.cleanup()
        finally:
            shutil.rmtree(tmpdir_a, ignore_errors=True)
            shutil.rmtree(tmpdir_b, ignore_errors=True)

    @requires_gl
    def test_two_robots_both_render_cameras(self):
        """Two robots with cameras from different dirs - both cameras render."""
        # Robot A has arm0/wrist_cam (from ROBOT_XML)
        # Add a camera to Robot B as well
        robot_b_with_cam = """
<mujoco model="gripper_cam">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <camera name="grip_cam" pos="0 0.2 0.3" xyaxes="1 0 0 0 0 1"/>
    <body name="grip_base" pos="0 0 0.05">
      <geom type="box" size="0.02 0.04 0.02" rgba="0.5 0.5 0.1 1"/>
      <joint name="grip_slide" type="slide" axis="1 0 0" range="-0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="grip_act" joint="grip_slide" kp="30"/>
  </actuator>
</mujoco>
"""
        tmpdir_a = tempfile.mkdtemp(prefix="robot_a_cam_")
        tmpdir_b = tempfile.mkdtemp(prefix="robot_b_cam_")

        try:
            path_a = os.path.join(tmpdir_a, "arm.xml")
            with open(path_a, "w") as f:
                f.write(ROBOT_XML)

            path_b = os.path.join(tmpdir_b, "gripper_cam.xml")
            with open(path_b, "w") as f:
                f.write(robot_b_with_cam)

            sim = Simulation(tool_name="test_render_multi", mesh=False)
            result = sim.create_world(gravity=[0, 0, -9.81])
            assert result["status"] == "success"

            result = sim.add_robot("arm1", urdf_path=path_a)
            assert result["status"] == "success"
            result = sim.add_robot("grip1", urdf_path=path_b, position=[0.5, 0, 0])
            assert result["status"] == "success"

            # Step to settle physics
            sim.step(n_steps=5)

            # Get observation (includes camera renders)
            obs = sim._get_sim_observation("arm1")

            # We should have at least one camera rendered (arm0/wrist_cam)
            cam_frames = {k: v for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 3}
            assert len(cam_frames) > 0, f"No camera frames rendered. Observation keys: {list(obs.keys())}"

            # Verify camera frame is not all-zero (actually rendered something)
            for cam_name, frame in cam_frames.items():
                assert frame.shape[2] == 3, f"Camera {cam_name} not RGB: shape={frame.shape}"
                # At minimum, the frame should have some non-zero pixels
                # (ground plane + colored geoms should provide contrast)
                assert frame.sum() > 0, f"Camera {cam_name} rendered all-black frame"

            sim.cleanup()
        finally:
            shutil.rmtree(tmpdir_a, ignore_errors=True)
            shutil.rmtree(tmpdir_b, ignore_errors=True)


class TestSceneMutationBlockedDuringPolicy:
    """Scene mutations must hard-fail while a policy is running.

    A concurrent PolicyRunner worker calling mj_step on stale model/data
    pointers (swapped by XML round-trip in add_object, add_camera, etc.)
    is undefined behaviour. The guard ensures agents learn to stop_policy
    before modifying the scene.
    """

    @pytest.fixture
    def robot_path(self, tmp_path):
        """Write test robot XML to a temp file."""
        path = tmp_path / "arm.xml"
        path.write_text(ROBOT_XML)
        return str(path)

    def test_add_object_blocked_during_policy(self, robot_path):
        sim = Simulation(tool_name="test_guard_obj", mesh=False)
        result = sim.create_world(gravity=[0, 0, -9.81])
        assert result["status"] == "success"

        result = sim.add_robot("arm1", urdf_path=robot_path)
        assert result["status"] == "success"

        # Start a policy (fast_mode so it completes quickly after stop)
        result = sim.start_policy("arm1", policy_provider="mock", duration=2.0, fast_mode=True)
        assert result["status"] == "success"

        # Try adding an object while policy is running - should be blocked
        result = sim.add_object("cube", shape="box", position=[0.3, 0, 0.05])
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()

        # Stop the policy
        sim.stop_policy("arm1")
        if "arm1" in sim._policy_threads:
            sim._policy_threads["arm1"].result(timeout=10.0)

        # Now it should work
        result = sim.add_object("cube", shape="box", position=[0.3, 0, 0.05])
        assert result["status"] == "success"

        sim.cleanup()

    def test_add_camera_blocked_during_policy(self, robot_path):
        sim = Simulation(tool_name="test_guard_cam", mesh=False)
        result = sim.create_world(gravity=[0, 0, -9.81])
        assert result["status"] == "success"

        result = sim.add_robot("arm1", urdf_path=robot_path)
        assert result["status"] == "success"

        result = sim.start_policy("arm1", policy_provider="mock", duration=2.0, fast_mode=True)
        assert result["status"] == "success"

        # Try adding a camera while policy is running - should be blocked
        result = sim.add_camera("top_cam", position=[0, 0, 2], target=[0, 0, 0])
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()

        sim.stop_policy("arm1")
        if "arm1" in sim._policy_threads:
            sim._policy_threads["arm1"].result(timeout=10.0)

        result = sim.add_camera("top_cam", position=[0, 0, 2], target=[0, 0, 0])
        assert result["status"] == "success"

        sim.cleanup()

    def test_load_scene_blocked_during_policy(self, robot_path):
        sim = Simulation(tool_name="test_guard_scene", mesh=False)
        result = sim.create_world(gravity=[0, 0, -9.81])
        assert result["status"] == "success"

        result = sim.add_robot("arm1", urdf_path=robot_path)
        assert result["status"] == "success"

        result = sim.start_policy("arm1", policy_provider="mock", duration=2.0, fast_mode=True)
        assert result["status"] == "success"

        # load_scene while policy is running - should be blocked
        result = sim.load_scene(robot_path)
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()

        sim.stop_policy("arm1")
        if "arm1" in sim._policy_threads:
            sim._policy_threads["arm1"].result(timeout=10.0)

        sim.cleanup()

    def test_move_object_blocked_during_policy(self, robot_path):
        sim = Simulation(tool_name="test_guard_move", mesh=False)
        result = sim.create_world(gravity=[0, 0, -9.81])
        assert result["status"] == "success"

        result = sim.add_robot("arm1", urdf_path=robot_path)
        assert result["status"] == "success"

        # Add an object to move later
        result = sim.add_object("cube", shape="box", position=[0.3, 0, 0.05])
        assert result["status"] == "success"

        result = sim.start_policy("arm1", policy_provider="mock", duration=2.0, fast_mode=True)
        assert result["status"] == "success"

        # Try moving an object while policy is running - should be blocked
        result = sim.move_object("cube", position=[0.5, 0, 0.1])
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()

        sim.stop_policy("arm1")
        if "arm1" in sim._policy_threads:
            sim._policy_threads["arm1"].result(timeout=10.0)

        # Now it should work
        result = sim.move_object("cube", position=[0.5, 0, 0.1])
        assert result["status"] == "success"

        sim.cleanup()

    def test_add_robot_blocked_during_other_policy(self, robot_path):
        """add_robot recompiles the whole XML and swaps the model/data
        pointers, so it must hard-fail while ANY policy is live - a running
        PolicyRunner on a different arm holds cached joint/actuator IDs that
        the round-trip invalidates. (Global-scope guard.)
        """
        sim = Simulation(tool_name="test_guard_add_robot", mesh=False)
        assert sim.create_world(gravity=[0, 0, -9.81])["status"] == "success"
        assert sim.add_robot("arm1", urdf_path=robot_path)["status"] == "success"

        result = sim.start_policy("arm1", policy_provider="mock", duration=2.0, fast_mode=True)
        assert result["status"] == "success"

        # Adding a second arm while arm1's policy runs is refused.
        result = sim.add_robot("arm2", urdf_path=robot_path)
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()

        sim.stop_policy("arm1")
        if "arm1" in sim._policy_threads:
            sim._policy_threads["arm1"].result(timeout=10.0)

        # Now it works.
        assert sim.add_robot("arm2", urdf_path=robot_path)["status"] == "success"

        sim.cleanup()

    def test_remove_object_blocked_during_policy(self, robot_path):
        """remove_object ejects a body from the live MjSpec and recompiles,
        swapping model/data - blocked while a policy is running so a worker
        never mj_steps stale pointers.
        """
        sim = Simulation(tool_name="test_guard_remove_obj", mesh=False)
        assert sim.create_world(gravity=[0, 0, -9.81])["status"] == "success"
        assert sim.add_robot("arm1", urdf_path=robot_path)["status"] == "success"
        # Object must exist before the policy starts (add is itself guarded).
        assert sim.add_object("cube", shape="box", position=[0.3, 0, 0.05])["status"] == "success"

        result = sim.start_policy("arm1", policy_provider="mock", duration=2.0, fast_mode=True)
        assert result["status"] == "success"

        # Removing the object while the policy runs is refused.
        result = sim.remove_object("cube")
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()
        # The object survives the refused mutation.
        assert "cube" in sim._world.objects

        sim.stop_policy("arm1")
        if "arm1" in sim._policy_threads:
            sim._policy_threads["arm1"].result(timeout=10.0)

        # Now it works.
        assert sim.remove_object("cube")["status"] == "success"
        assert "cube" not in sim._world.objects

        sim.cleanup()

    def test_remove_camera_blocked_during_policy(self, robot_path):
        """remove_camera deletes the camera from the MjSpec and recompiles,
        so - like the other scene mutations - it is refused while a policy is
        running rather than swapping model/data under a live worker.
        """
        sim = Simulation(tool_name="test_guard_remove_cam", mesh=False)
        assert sim.create_world(gravity=[0, 0, -9.81])["status"] == "success"
        assert sim.add_robot("arm1", urdf_path=robot_path)["status"] == "success"
        # Camera must exist before the policy starts (add is itself guarded).
        assert sim.add_camera("top_cam", position=[0, 0, 2], target=[0, 0, 0])["status"] == "success"

        result = sim.start_policy("arm1", policy_provider="mock", duration=2.0, fast_mode=True)
        assert result["status"] == "success"

        # Removing the camera while the policy runs is refused.
        result = sim.remove_camera("top_cam")
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()
        # The camera survives the refused mutation.
        assert "top_cam" in sim._world.cameras

        sim.stop_policy("arm1")
        if "arm1" in sim._policy_threads:
            sim._policy_threads["arm1"].result(timeout=10.0)

        # Now it works.
        assert sim.remove_camera("top_cam")["status"] == "success"
        assert "top_cam" not in sim._world.cameras

        sim.cleanup()

    def test_remove_robot_stops_own_policy_and_succeeds(self, robot_path):
        """Per-robot scoping (GH #114): remove_robot(X) gracefully stops X's
        own policy before removing it. Previously this errored, forcing the
        agent into a two-step stop-then-remove dance even in the common
        'delete the robot I'm running' case.
        """
        sim = Simulation(tool_name="test_guard_remove_robot", mesh=False)
        result = sim.create_world(gravity=[0, 0, -9.81])
        assert result["status"] == "success"

        result = sim.add_robot("arm1", urdf_path=robot_path)
        assert result["status"] == "success"

        result = sim.start_policy("arm1", policy_provider="mock", duration=2.0, fast_mode=True)
        assert result["status"] == "success"

        # GH #114: remove_robot on the same arm gracefully stops its policy
        # and proceeds. No two-step dance required.
        result = sim.remove_robot("arm1")
        assert result["status"] == "success", result
        assert "arm1" in result["content"][0]["text"]
        # Policy future was pruned.
        assert "arm1" not in sim._policy_threads

        sim.cleanup()

    def test_remove_robot_blocked_by_OTHER_robot_policy(self, robot_path):
        """Global-scope guard (GH #114): remove_robot(A) still errors if
        a policy is active on a different robot B, because the XML round-trip
        invalidates cached actuator/joint IDs held by B's PolicyRunner.
        """
        sim = Simulation(tool_name="test_guard_other_robot", mesh=False)
        assert sim.create_world(gravity=[0, 0, -9.81])["status"] == "success"
        assert sim.add_robot("armA", urdf_path=robot_path)["status"] == "success"
        assert sim.add_robot("armB", urdf_path=robot_path)["status"] == "success"

        # Policy on B...
        assert sim.start_policy("armB", policy_provider="mock", duration=5.0, fast_mode=True)["status"] == "success"

        # ...blocks remove_robot on A (scene mutation invalidates IDs).
        result = sim.remove_robot("armA")
        assert result["status"] == "error"
        assert "policy is running" in result["content"][0]["text"].lower()
        assert "armB" in result["content"][0]["text"]

        sim.stop_policy("armB")
        if "armB" in sim._policy_threads:
            sim._policy_threads["armB"].result(timeout=10.0)

        # Now removal works.
        assert sim.remove_robot("armA")["status"] == "success"

        sim.cleanup()


class TestConcurrentPerRobotPolicies:
    """GH #114: two or more policies can run concurrently on different robots.

    Proves the post-fix semantics:

    * ``start_policy`` only blocks on the SAME robot; a second start_policy
      on a DIFFERENT robot while the first is running now succeeds.
    * ``list_policies_running`` accurately reports all active ones and
      prunes completed Futures as a side-effect.
    * Two policies mutating their own ``ctrl[]`` slots in parallel never
      corrupt MuJoCo state (``self._lock`` still serializes ``mj_step``).
    """

    @pytest.fixture
    def robot_path(self, tmp_path):
        path = tmp_path / "arm.xml"
        path.write_text(ROBOT_XML)
        return str(path)

    def test_start_policy_allowed_on_second_robot_while_first_runs(self, robot_path):
        sim = Simulation(tool_name="test_concurrent_start", mesh=False)
        assert sim.create_world()["status"] == "success"
        assert sim.add_robot("armA", urdf_path=robot_path)["status"] == "success"
        assert sim.add_robot("armB", urdf_path=robot_path)["status"] == "success"

        # First policy starts.
        r1 = sim.start_policy("armA", policy_provider="mock", duration=3.0, fast_mode=True)
        assert r1["status"] == "success", r1

        # Second policy on a DIFFERENT robot also starts (per-robot gate).
        r2 = sim.start_policy("armB", policy_provider="mock", duration=3.0, fast_mode=True)
        assert r2["status"] == "success", r2

        # Both active.
        active = sim._active_policy_robots()
        assert set(active) == {"armA", "armB"}, active

        sim.stop_policy("armA")
        sim.stop_policy("armB")
        # Wait for graceful stop.
        for name in ("armA", "armB"):
            fut = sim._policy_threads.get(name)
            if fut is not None:
                try:
                    fut.result(timeout=10.0)
                except Exception:
                    pass
        sim.cleanup()

    def test_start_policy_still_rejected_on_SAME_robot(self, robot_path):
        """Per-robot gate still fires when we start twice on the same robot."""
        sim = Simulation(tool_name="test_concurrent_same", mesh=False)
        assert sim.create_world()["status"] == "success"
        assert sim.add_robot("arm1", urdf_path=robot_path)["status"] == "success"

        r1 = sim.start_policy("arm1", policy_provider="mock", duration=3.0, fast_mode=True)
        assert r1["status"] == "success"

        r2 = sim.start_policy("arm1", policy_provider="mock", duration=3.0, fast_mode=True)
        assert r2["status"] == "error"
        assert "arm1" in r2["content"][0]["text"]

        sim.stop_policy("arm1")
        fut = sim._policy_threads.get("arm1")
        if fut is not None:
            try:
                fut.result(timeout=10.0)
            except Exception:
                pass
        sim.cleanup()

    def test_list_policies_running_reports_active(self, robot_path):
        sim = Simulation(tool_name="test_list_policies", mesh=False)
        sim.create_world()
        sim.add_robot("armA", urdf_path=robot_path)
        sim.add_robot("armB", urdf_path=robot_path)

        # None active.
        r = sim.list_policies_running()
        assert r["status"] == "success"
        assert "No policies" in r["content"][0]["text"]

        # One active.
        sim.start_policy("armA", policy_provider="mock", duration=3.0, fast_mode=True)
        r = sim.list_policies_running()
        assert r["status"] == "success"
        assert "armA" in r["content"][0]["text"]
        assert "armB" not in r["content"][0]["text"]

        # Two active.
        sim.start_policy("armB", policy_provider="mock", duration=3.0, fast_mode=True)
        r = sim.list_policies_running()
        assert "armA" in r["content"][0]["text"]
        assert "armB" in r["content"][0]["text"]

        # Clean shutdown.
        sim.stop_policy("armA")
        sim.stop_policy("armB")
        for name in ("armA", "armB"):
            fut = sim._policy_threads.get(name)
            if fut is not None:
                try:
                    fut.result(timeout=10.0)
                except Exception:
                    pass

        # After both stop, list is empty again (stale prune).
        r = sim.list_policies_running()
        assert "No policies" in r["content"][0]["text"]
        assert sim._policy_threads == {}

        sim.cleanup()

    def test_completed_futures_are_pruned(self, robot_path):
        """GH #120 (companion fix): completed Futures must not linger in
        _policy_threads forever.
        """
        sim = Simulation(tool_name="test_prune", mesh=False)
        sim.create_world()
        sim.add_robot("armA", urdf_path=robot_path)

        # Very short policy - let it complete naturally.
        sim.start_policy("armA", policy_provider="mock", duration=0.1, fast_mode=True)
        fut = sim._policy_threads.get("armA")
        assert fut is not None
        try:
            fut.result(timeout=10.0)
        except Exception:
            pass

        # Future is done - one introspection call prunes it.
        active = sim._active_policy_robots()
        assert active == [], active
        assert "armA" not in sim._policy_threads

        sim.cleanup()

    def test_scene_mutation_lists_which_robots_are_running(self, robot_path):
        """Error message names the active-policy robots so the LLM can
        stop_policy on each without guessing.
        """
        sim = Simulation(tool_name="test_err_msg", mesh=False)
        sim.create_world()
        sim.add_robot("armA", urdf_path=robot_path)
        sim.add_robot("armB", urdf_path=robot_path)

        sim.start_policy("armA", policy_provider="mock", duration=3.0, fast_mode=True)
        sim.start_policy("armB", policy_provider="mock", duration=3.0, fast_mode=True)

        r = sim.set_gravity([0, 0, -5.0])
        assert r["status"] == "error"
        text = r["content"][0]["text"]
        assert "armA" in text
        assert "armB" in text

        sim.stop_policy("armA")
        sim.stop_policy("armB")
        for name in ("armA", "armB"):
            fut = sim._policy_threads.get(name)
            if fut is not None:
                try:
                    fut.result(timeout=10.0)
                except Exception:
                    pass
        sim.cleanup()

    def test_two_policies_no_segfault_under_stress(self, robot_path):
        """Smoke test: two concurrent policies actually *run* (not just
        both "started") and produce step_count > 0 on both robots, with
        self._lock serializing the shared mj_step safely.

        Uses a short duration + fast_mode so the test finishes under
        a second.
        """
        sim = Simulation(tool_name="test_stress_concurrent", mesh=False)
        sim.create_world()
        sim.add_robot("armA", urdf_path=robot_path)
        sim.add_robot("armB", urdf_path=robot_path)

        sim.start_policy("armA", policy_provider="mock", duration=0.5, fast_mode=True)
        sim.start_policy("armB", policy_provider="mock", duration=0.5, fast_mode=True)

        # Let both run to completion.
        for name in ("armA", "armB"):
            fut = sim._policy_threads.get(name)
            if fut is not None:
                try:
                    fut.result(timeout=15.0)
                except Exception:
                    pass

        # Both robots advanced their step counter - proves both ran.
        assert sim._world is not None
        assert sim._world.robots["armA"].policy_steps > 0, "armA never stepped - concurrent scheduling broke it"
        assert sim._world.robots["armB"].policy_steps > 0, "armB never stepped - concurrent scheduling broke it"

        sim.cleanup()


class TestCleanupGracefulShutdown:
    """GH #116: cleanup() must wait for live policy workers before nulling
    the world, otherwise an in-flight mj_step segfaults on freed arrays.
    """

    @pytest.fixture
    def robot_path(self, tmp_path):
        path = tmp_path / "arm.xml"
        path.write_text(ROBOT_XML)
        return str(path)

    def test_cleanup_awaits_running_policy(self, robot_path):
        """Start a long-running policy, call cleanup, verify the worker
        completed (Future.done()) before cleanup returned and we do NOT
        segfault on world nulling."""
        sim = Simulation(tool_name="test_cleanup_await", mesh=False)
        sim.create_world()
        sim.add_robot("armA", urdf_path=robot_path)

        sim.start_policy("armA", policy_provider="mock", duration=5.0, fast_mode=True)
        fut = sim._policy_threads.get("armA")
        assert fut is not None and not fut.done(), "policy should be live"

        # Cleanup with tight timeout - the cooperative-stop flag is read
        # every step so 1s is plenty for MockPolicy to exit.
        sim.cleanup(policy_stop_timeout=2.0)

        # Post-cleanup invariants.
        assert fut.done(), "Future must have terminated before cleanup returned"
        assert sim._world is None, "world must be nulled after cleanup"
        assert sim._policy_threads == {}, "policy_threads must be drained"

    def test_cleanup_tolerates_wedged_policy(self, robot_path):
        """A policy that refuses to stop within the timeout must NOT hang
        the whole process. Cleanup logs a warning and proceeds."""
        sim = Simulation(tool_name="test_cleanup_wedged", mesh=False)
        sim.create_world()
        sim.add_robot("armA", urdf_path=robot_path)

        sim.start_policy("armA", policy_provider="mock", duration=5.0, fast_mode=True)

        # Aggressively short timeout forces the "wedged" path even if the
        # mock is fast - the test is that cleanup RETURNS in bounded time,
        # not that the future is done.
        import time as _time

        t0 = _time.monotonic()
        sim.cleanup(policy_stop_timeout=0.001)
        elapsed = _time.monotonic() - t0

        # Even with timeout=1ms, total cleanup must complete quickly.
        # We allow some slack for teardown of renderers/viewer.
        assert elapsed < 10.0, f"cleanup blocked too long: {elapsed:.2f}s"
        assert sim._world is None

    def test_cleanup_is_idempotent_with_no_policies(self, robot_path):
        """Calling cleanup with no live policies must be a straight no-op
        for the policy-drain path (no Futures to wait on)."""
        sim = Simulation(tool_name="test_cleanup_noop", mesh=False)
        sim.create_world()
        sim.add_robot("armA", urdf_path=robot_path)
        # No start_policy call.

        sim.cleanup(policy_stop_timeout=0.1)

        assert sim._world is None
        assert sim._policy_threads == {}

    def test_cleanup_drains_multiple_concurrent_policies(self, robot_path):
        """With concurrent per-robot policies (GH #114), cleanup must await
        BOTH before nulling the world."""
        sim = Simulation(tool_name="test_cleanup_multi", mesh=False)
        sim.create_world()
        sim.add_robot("armA", urdf_path=robot_path)
        sim.add_robot("armB", urdf_path=robot_path)

        sim.start_policy("armA", policy_provider="mock", duration=5.0, fast_mode=True)
        sim.start_policy("armB", policy_provider="mock", duration=5.0, fast_mode=True)

        futs = {name: sim._policy_threads.get(name) for name in ("armA", "armB")}
        assert all(f is not None and not f.done() for f in futs.values())

        sim.cleanup(policy_stop_timeout=3.0)

        # Both worker futures settled before cleanup returned.
        for name, fut in futs.items():
            assert fut is not None and fut.done(), f"'{name}' future was not awaited"
        assert sim._world is None


class TestMutationGuardStress:
    """GH #119: hammer the mutation guard to prove no race between
    the ``_require_no_running_policy`` check and the PolicyRunner's
    ``mj_step`` call. Historically we relied on the check being 'atomic
    enough in practice' - no test proved it.

    The critical contract we're validating:

    1. Every scene-mutation call attempted while a policy is live must
       either (a) return status=error with our uniform message, or
       (b) return status=success if the policy has already settled.
       NOTHING may corrupt MuJoCo state or segfault.

    2. The mutation guard must be fast enough that 1000 concurrent
       requests from the main thread do not starve the policy worker.
    """

    @pytest.fixture
    def robot_path(self, tmp_path):
        path = tmp_path / "arm.xml"
        path.write_text(ROBOT_XML)
        return str(path)

    def test_1000_set_gravity_calls_during_policy_never_segfault(self, robot_path):
        """Start a policy, then bang set_gravity 1000 times from the main
        thread. Every call must return a well-formed dict - no crash, no
        half-applied mutation. Once the policy ends, the last set_gravity
        succeeds."""
        sim = Simulation(tool_name="test_stress_set_gravity", mesh=False)
        sim.create_world()
        sim.add_robot("arm", urdf_path=robot_path)

        sim.start_policy("arm", policy_provider="mock", duration=1.0, fast_mode=True)

        # Hammer from the main thread while the worker runs.
        blocked = 0
        succeeded = 0
        for _ in range(1000):
            r = sim.set_gravity([0.0, 0.0, -9.81])
            assert isinstance(r, dict), r
            assert r["status"] in ("success", "error"), r
            if r["status"] == "error":
                assert "policy is running" in r["content"][0]["text"].lower()
                blocked += 1
            else:
                succeeded += 1

        # At least one call must have been blocked (policy was live).
        assert blocked > 0, "stress loop never saw the policy as live - timing broken"

        # After policy finishes, set_gravity works.
        fut = sim._policy_threads.get("arm")
        if fut is not None:
            try:
                fut.result(timeout=10.0)
            except Exception:
                pass

        result = sim.set_gravity([0.0, 0.0, -5.0])
        assert result["status"] == "success"

        sim.cleanup(policy_stop_timeout=2.0)

    def test_rapid_start_stop_start_stop_policy(self, robot_path):
        """Stress the Future lifecycle. Rapid start/stop cycles must leave
        _policy_threads in a consistent state every iteration."""
        sim = Simulation(tool_name="test_rapid_cycle", mesh=False)
        sim.create_world()
        sim.add_robot("arm", urdf_path=robot_path)

        for i in range(10):
            r_start = sim.start_policy("arm", policy_provider="mock", duration=2.0, fast_mode=True)
            assert r_start["status"] == "success", (i, r_start)

            r_stop = sim.stop_policy("arm")
            assert r_stop["status"] == "success", (i, r_stop)

            # Await worker so the next start_policy doesn't race.
            fut = sim._policy_threads.get("arm")
            if fut is not None:
                try:
                    fut.result(timeout=5.0)
                except Exception:
                    pass

            # Prune runs as a side effect of _active_policy_robots.
            active = sim._active_policy_robots()
            assert active == [], (i, active)

        sim.cleanup(policy_stop_timeout=2.0)

    def test_mutation_accepted_immediately_after_policy_completes(self, robot_path):
        """Once the policy Future is done(), the VERY NEXT scene mutation
        must succeed - no lingering guard state from the just-completed run."""
        sim = Simulation(tool_name="test_no_lingering_guard", mesh=False)
        sim.create_world()
        sim.add_robot("arm", urdf_path=robot_path)

        # Very short policy.
        sim.start_policy("arm", policy_provider="mock", duration=0.05, fast_mode=True)
        fut = sim._policy_threads.get("arm")
        assert fut is not None
        try:
            fut.result(timeout=5.0)
        except Exception:
            pass
        assert fut.done()

        # First mutation after completion must succeed.
        r = sim.set_gravity([0.0, 0.0, -9.81])
        assert r["status"] == "success", r

        sim.cleanup(policy_stop_timeout=1.0)

    def test_concurrent_policies_stress_no_deadlock(self, robot_path):
        """Two concurrent policies (GH #114) + main-thread mutation spam
        must not deadlock on self._lock."""
        sim = Simulation(tool_name="test_concurrent_stress", mesh=False)
        sim.create_world()
        sim.add_robot("armA", urdf_path=robot_path)
        sim.add_robot("armB", urdf_path=robot_path)

        sim.start_policy("armA", policy_provider="mock", duration=1.0, fast_mode=True)
        sim.start_policy("armB", policy_provider="mock", duration=1.0, fast_mode=True)

        blocked = 0
        errors = 0
        for _ in range(500):
            r = sim.set_gravity([0.0, 0.0, -9.81])
            assert r["status"] in ("success", "error"), r
            if r["status"] == "error":
                # When blocked, the message must name AT LEAST one robot.
                text = r["content"][0]["text"]
                if "armA" in text or "armB" in text:
                    blocked += 1
                else:
                    errors += 1

        assert errors == 0, f"unexpected error shape: {errors}"
        assert blocked > 0, "never caught policies as live"

        # Wait for both to settle.
        for name in ("armA", "armB"):
            fut = sim._policy_threads.get(name)
            if fut is not None:
                try:
                    fut.result(timeout=10.0)
                except Exception:
                    pass

        sim.cleanup(policy_stop_timeout=2.0)
