#!/usr/bin/env python3
"""
Audit Areas D, E, F — URDF resolution, Dataset recording E2E, Policy dispatch.

D. URDF Resolution — resolve_urdf() for all registered robots, verify
   _URDF_REGISTRY consistency with factory _UNIFIED_ROBOTS.
E. Dataset Recording E2E — sim + robot + mock policy 20 steps, verify
   actual frame data (not empty episodes).
F. Policy Dispatch — run_policy, start_policy, stop_policy through
   Simulation dispatch mechanism.
"""

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, ".")
os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

try:
    import mujoco  # noqa: F401

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

requires_mujoco = pytest.mark.skipif(not HAS_MUJOCO, reason="mujoco not installed")

# =====================================================================
# Area D — URDF Resolution
# =====================================================================


class TestURDFResolution:
    """D. Verify resolve_urdf() for every _URDF_REGISTRY entry,
    and that _URDF_REGISTRY is consistent with _UNIFIED_ROBOTS in factory.
    """

    def test_urdf_registry_is_dict(self):
        """_URDF_REGISTRY must be a non-empty dict."""
        from strands_robots.simulation import _URDF_REGISTRY

        assert isinstance(_URDF_REGISTRY, dict)
        assert len(_URDF_REGISTRY) > 0, "_URDF_REGISTRY should not be empty"

    def test_resolve_urdf_returns_none_for_unknown(self):
        """resolve_urdf for unknown name → None (not crash)."""
        from strands_robots.simulation import resolve_urdf

        result = resolve_urdf("does_not_exist_robot_xyz")
        assert result is None

    def test_resolve_urdf_for_all_registered(self):
        """Every key in _URDF_REGISTRY either resolves to a path or to None
        (files may not be downloaded), but must not raise."""
        from strands_robots.simulation import _URDF_REGISTRY, resolve_urdf

        for name in _URDF_REGISTRY:
            result = resolve_urdf(name)
            # result is either a string path or None — never raises
            assert result is None or isinstance(result, str), f"resolve_urdf('{name}') returned {type(result)}"

    def test_resolve_model_prefers_menagerie(self):
        """resolve_model() should prefer Menagerie MJCF files when available."""
        from strands_robots.simulation import resolve_model

        # so100 is in both registries — Menagerie should win
        path = resolve_model("so100")
        if path is not None:
            assert isinstance(path, str)
            # Menagerie models are XML files
            assert path.endswith(".xml"), f"Expected .xml but got: {path}"

    def test_resolve_model_for_common_robots(self):
        """Common robots (so100, panda, unitree_g1) should resolve."""
        from strands_robots.simulation import resolve_model

        common = ["so100", "panda", "unitree_g1"]
        for name in common:
            path = resolve_model(name)
            # May be None if assets not downloaded, but should not crash
            assert path is None or os.path.exists(path), f"resolve_model('{name}') returned non-existent: {path}"

    def test_asset_manager_robot_list(self):
        """Asset manager should list 30+ robots."""
        from strands_robots.assets import list_available_robots

        robots = list_available_robots()
        assert isinstance(robots, list)
        assert len(robots) >= 30, f"Expected ≥30 robots, got {len(robots)}"
        # Each entry has expected fields
        for r in robots:
            assert "name" in r
            assert "description" in r
            assert "category" in r

    def test_asset_aliases_resolve(self):
        """All aliases in assets._ALIASES resolve to a canonical name in _ROBOT_MODELS."""
        from strands_robots.assets import _ALIASES, _ROBOT_MODELS, resolve_robot_name

        for alias, canonical in _ALIASES.items():
            resolved = resolve_robot_name(alias)
            assert resolved in _ROBOT_MODELS, f"Alias '{alias}' → '{resolved}' not in _ROBOT_MODELS"

    def test_factory_robots_have_sim_entries(self):
        """Every _UNIFIED_ROBOTS entry with sim!=None should resolve via
        resolve_model() (either to Menagerie or URDF registry)."""
        from strands_robots.factory import _UNIFIED_ROBOTS
        from strands_robots.simulation import resolve_model

        for name, info in _UNIFIED_ROBOTS.items():
            sim_name = info.get("sim")
            if sim_name is None:
                continue
            # Should at least attempt resolution without crashing
            path = resolve_model(sim_name)
            # We don't require all files to exist (some need download),
            # but the function should return str or None
            assert path is None or isinstance(path, str), (
                f"resolve_model('{sim_name}') for factory robot '{name}' " f"returned unexpected type: {type(path)}"
            )

    def test_urdf_registry_keys_are_strings(self):
        """All keys and values in _URDF_REGISTRY are strings."""
        from strands_robots.simulation import _URDF_REGISTRY

        for k, v in _URDF_REGISTRY.items():
            assert isinstance(k, str), f"Key {k!r} is not str"
            assert isinstance(v, str), f"Value for key '{k}' is not str: {v!r}"

    def test_list_registered_urdfs(self):
        """list_registered_urdfs() returns a dict without crashing."""
        from strands_robots.simulation import list_registered_urdfs

        result = list_registered_urdfs()
        assert isinstance(result, dict)
        # Every value is either None or a string path
        for k, v in result.items():
            assert v is None or isinstance(v, str)

    def test_register_urdf_custom(self):
        """register_urdf() should allow registering a custom path."""
        from strands_robots.simulation import (
            _URDF_REGISTRY,
            register_urdf,
        )

        test_name = "_test_robot_custom_xyz"
        register_urdf(test_name, "/tmp/fake_robot.urdf")
        assert test_name in _URDF_REGISTRY
        assert _URDF_REGISTRY[test_name] == "/tmp/fake_robot.urdf"
        # Cleanup
        del _URDF_REGISTRY[test_name]


# =====================================================================
# Area E — Dataset Recording E2E
# =====================================================================


@requires_mujoco
class TestDatasetRecordingE2E:
    """E. Create sim, add robot, run mock policy for 20 steps,
    verify actual frame data (not empty episodes).

    The key insight from the audit: we need to trace WHY add_frame
    might not get called during step(). We test both:
    1. DatasetRecorder.add_frame directly
    2. Full E2E through Simulation.run_policy with recording
    """

    def test_dataset_recorder_add_frame_directly(self):
        """DatasetRecorder.add_frame writes actual data to the dataset."""
        try:
            from strands_robots.dataset_recorder import (
                HAS_LEROBOT_DATASET,
                DatasetRecorder,
            )
        except ImportError:
            pytest.skip("dataset_recorder not importable")

        if not HAS_LEROBOT_DATASET:
            pytest.skip("lerobot not installed")

        tmpdir = tempfile.mkdtemp(prefix="test_ds_")
        repo_id = "local/test_add_frame"
        ds_root = os.path.join(tmpdir, repo_id)

        try:
            recorder = DatasetRecorder.create(
                repo_id=repo_id,
                fps=10,
                robot_type="test_bot",
                joint_names=["j1", "j2", "j3"],
                camera_keys=[],
                task="test_task",
                root=ds_root,
                use_videos=False,
            )

            # Add 20 frames
            for i in range(20):
                obs = {"j1": float(i * 0.1), "j2": float(i * 0.2), "j3": float(i * 0.3)}
                act = {"j1": float(i * 0.01), "j2": float(i * 0.02), "j3": float(i * 0.03)}
                recorder.add_frame(observation=obs, action=act, task="test_task")

            assert recorder.frame_count == 20, f"Expected 20 frames, got {recorder.frame_count}"

            # Save episode
            result = recorder.save_episode()
            assert result["status"] == "success", f"save_episode failed: {result}"
            assert recorder.episode_count == 1

            recorder.finalize()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_sim_recording_e2e_with_mock_policy(self):
        """Full E2E: sim → add_robot → start_recording → run_policy(mock)
        → stop_recording → verify frames > 0.

        This is the critical test: during run_policy, every step should
        call DatasetRecorder.add_frame. If the episode is empty, the
        recording path in run_policy is broken.
        """
        try:
            from strands_robots.dataset_recorder import HAS_LEROBOT_DATASET
        except ImportError:
            pytest.skip("dataset_recorder not importable")

        if not HAS_LEROBOT_DATASET:
            pytest.skip("lerobot not installed")

        from strands_robots.simulation import Simulation

        tmpdir = tempfile.mkdtemp(prefix="test_sim_rec_")
        repo_id = f"local/test_sim_e2e_{int(time.time())}"
        ds_root = os.path.join(tmpdir, repo_id)

        sim = Simulation(tool_name="test_rec_sim", mesh=False)

        try:
            # Create world and add robot
            result = sim.create_world()
            assert result["status"] == "success", f"create_world failed: {result}"

            result = sim.add_robot(name="arm", data_config="so100")
            if result["status"] == "error":
                pytest.skip(f"Could not add so100 robot: {result}")

            # Start recording
            rec_result = sim.start_recording(
                repo_id=repo_id,
                task="test pick up",
                fps=10,
                root=ds_root,
            )
            assert rec_result["status"] == "success", f"start_recording failed: {rec_result}"

            # Verify recorder is set
            assert sim._world._recording is True
            assert sim._world._dataset_recorder is not None

            # Run mock policy for a short duration
            policy_result = sim.run_policy(
                robot_name="arm",
                policy_provider="mock",
                instruction="test pick up",
                duration=2.0,
                control_frequency=10.0,
            )
            assert policy_result["status"] == "success", f"run_policy failed: {policy_result}"

            # Check that frames were actually recorded
            recorder = sim._world._dataset_recorder
            frame_count = recorder.frame_count if recorder else 0
            assert frame_count > 0, (
                f"BUG: Dataset has 0 frames after run_policy! "
                f"add_frame is not being called during policy step. "
                f"Recording flag: {sim._world._recording}, "
                f"Recorder: {sim._world._dataset_recorder}"
            )

            # Stop recording
            stop_result = sim.stop_recording()
            assert stop_result["status"] == "success", f"stop_recording failed: {stop_result}"

            # Verify final state
            assert "frames" in str(stop_result) or frame_count > 0

        finally:
            sim.cleanup()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_sim_manual_add_frame_during_step(self):
        """Manually replicate what run_policy should do:
        get_observation → send_action → add_frame to recorder.

        This isolates whether the issue is in the policy loop or
        in add_frame itself.
        """
        try:
            from strands_robots.dataset_recorder import (
                HAS_LEROBOT_DATASET,
                DatasetRecorder,
            )
        except ImportError:
            pytest.skip("dataset_recorder not importable")

        if not HAS_LEROBOT_DATASET:
            pytest.skip("lerobot not installed")

        from strands_robots.simulation import Simulation

        tmpdir = tempfile.mkdtemp(prefix="test_manual_rec_")
        repo_id = f"local/test_manual_{int(time.time())}"
        ds_root = os.path.join(tmpdir, repo_id)

        sim = Simulation(tool_name="test_manual_sim", mesh=False)

        try:
            sim.create_world()
            result = sim.add_robot(name="arm", data_config="so100")
            if result["status"] == "error":
                pytest.skip(f"Could not add so100: {result}")

            robot = sim._world.robots["arm"]
            joint_names = robot.joint_names
            assert len(joint_names) > 0, "Robot has no joints"

            # Create recorder manually (same as start_recording does)
            recorder = DatasetRecorder.create(
                repo_id=repo_id,
                fps=10,
                robot_type="so100",
                joint_names=joint_names,
                camera_keys=[],
                task="test_manual",
                root=ds_root,
                use_videos=False,
            )

            # Manual step loop (20 steps)
            for step in range(20):
                obs = sim.get_observation("arm")
                assert isinstance(obs, dict), f"get_observation returned {type(obs)}"
                assert len(obs) > 0, "Observation is empty"

                # Create action (zero action)
                action = {jn: 0.0 for jn in joint_names}
                sim.send_action(action, "arm")

                # Record frame manually
                recorder.add_frame(
                    observation=obs,
                    action=action,
                    task="test_manual",
                )

            assert recorder.frame_count == 20, f"Expected 20 frames but got {recorder.frame_count}"

            # Save and verify
            result = recorder.save_episode()
            assert result["status"] == "success"
            assert recorder.episode_count == 1
            recorder.finalize()

        finally:
            sim.cleanup()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_observation_returns_joint_data(self):
        """get_observation() must return joint position data for the robot."""
        from strands_robots.simulation import Simulation

        sim = Simulation(tool_name="test_obs_sim", mesh=False)
        try:
            sim.create_world()
            result = sim.add_robot(name="arm", data_config="so100")
            if result["status"] == "error":
                pytest.skip(f"Could not add so100: {result}")

            obs = sim.get_observation("arm")
            assert isinstance(obs, dict)
            assert len(obs) > 0, "get_observation returned empty dict"

            # Check that joint values are floats
            robot = sim._world.robots["arm"]
            for jn in robot.joint_names:
                if jn in obs:
                    assert isinstance(obs[jn], float), f"Joint {jn} value is {type(obs[jn])}, expected float"
        finally:
            sim.cleanup()


# =====================================================================
# Area F — Policy Dispatch in Simulation
# =====================================================================


@requires_mujoco
class TestPolicyDispatch:
    """F. Test tool dispatch: run_policy, start_policy, stop_policy
    through Simulation.__call__ or the dispatch mechanism.
    """

    @pytest.fixture(autouse=True)
    def setup_sim(self):
        """Create a sim with a robot for each test."""
        from strands_robots.simulation import Simulation

        self.sim = Simulation(tool_name="test_dispatch_sim", mesh=False)
        self.sim.create_world()
        result = self.sim.add_robot(name="arm", data_config="so100")
        self.robot_available = result["status"] == "success"
        yield
        self.sim.cleanup()

    def test_dispatch_run_policy_mock(self):
        """run_policy with mock provider should succeed and report steps."""
        if not self.robot_available:
            pytest.skip("so100 robot not available")

        result = self.sim._dispatch_action(
            "run_policy",
            {
                "robot_name": "arm",
                "policy_provider": "mock",
                "instruction": "test",
                "duration": 1.0,
                "control_frequency": 10.0,
            },
        )
        assert result["status"] == "success", f"run_policy failed: {result}"
        # Should mention completion
        text = result["content"][0]["text"]
        assert "complete" in text.lower() or "✅" in text

    def test_dispatch_start_policy_async(self):
        """start_policy should return immediately (non-blocking)."""
        if not self.robot_available:
            pytest.skip("so100 robot not available")

        start_time = time.time()
        result = self.sim._dispatch_action(
            "start_policy",
            {
                "robot_name": "arm",
                "policy_provider": "mock",
                "instruction": "test async",
                "duration": 5.0,
            },
        )
        elapsed = time.time() - start_time

        assert result["status"] == "success", f"start_policy failed: {result}"
        # Should return quickly (within 1 second), not block for 5s
        assert elapsed < 2.0, f"start_policy blocked for {elapsed:.1f}s (should be async)"

    def test_dispatch_stop_policy(self):
        """stop_policy should stop a running policy."""
        if not self.robot_available:
            pytest.skip("so100 robot not available")

        # Start async policy
        self.sim._dispatch_action(
            "start_policy",
            {
                "robot_name": "arm",
                "policy_provider": "mock",
                "instruction": "test stop",
                "duration": 30.0,
            },
        )

        # Give it a moment to start
        time.sleep(0.5)

        # Stop it
        result = self.sim._dispatch_action(
            "stop_policy",
            {
                "robot_name": "arm",
            },
        )
        assert result["status"] == "success", f"stop_policy failed: {result}"

        # Verify the robot's policy_running flag is False
        robot = self.sim._world.robots["arm"]
        assert robot.policy_running is False, "Robot still running after stop"

    def test_dispatch_run_policy_missing_robot(self):
        """run_policy for non-existent robot should return error."""
        result = self.sim._dispatch_action(
            "run_policy",
            {
                "robot_name": "nonexistent",
                "policy_provider": "mock",
                "instruction": "test",
                "duration": 1.0,
            },
        )
        assert result["status"] == "error"

    def test_dispatch_stop_policy_missing_robot(self):
        """stop_policy for non-existent robot should return error."""
        result = self.sim._dispatch_action(
            "stop_policy",
            {
                "robot_name": "nonexistent",
            },
        )
        assert result["status"] == "error"

    def test_dispatch_run_policy_no_simulation(self):
        """run_policy when no world exists should return error."""
        from strands_robots.simulation import Simulation

        empty_sim = Simulation(tool_name="empty", mesh=False)
        result = empty_sim._dispatch_action(
            "run_policy",
            {
                "robot_name": "arm",
                "policy_provider": "mock",
                "instruction": "test",
            },
        )
        assert result["status"] == "error"
        empty_sim.cleanup()

    def test_dispatch_full_lifecycle(self):
        """Full lifecycle: create_world → add_robot → run_policy → reset → destroy."""
        if not self.robot_available:
            pytest.skip("so100 robot not available")

        # Run policy
        result = self.sim._dispatch_action(
            "run_policy",
            {
                "robot_name": "arm",
                "policy_provider": "mock",
                "instruction": "pick up",
                "duration": 0.5,
                "control_frequency": 10.0,
            },
        )
        assert result["status"] == "success"

        # Get state
        result = self.sim._dispatch_action("get_state", {})
        assert result["status"] == "success"

        # Get robot state
        result = self.sim._dispatch_action("get_robot_state", {"robot_name": "arm"})
        assert result["status"] == "success"

        # Step
        result = self.sim._dispatch_action("step", {"n_steps": 10})
        assert result["status"] == "success"

        # Reset
        result = self.sim._dispatch_action("reset", {})
        assert result["status"] == "success"

        # Destroy
        result = self.sim._dispatch_action("destroy", {})
        assert result["status"] == "success"

    def test_dispatch_unknown_action(self):
        """Unknown action should return error (not crash)."""
        result = self.sim._dispatch_action("totally_unknown_action_xyz", {})
        assert result["status"] == "error"

    def test_dispatch_get_features(self):
        """get_features should return model introspection."""
        if not self.robot_available:
            pytest.skip("so100 robot not available")

        result = self.sim._dispatch_action("get_features", {})
        assert result["status"] == "success"
        # Should have both text and json content
        assert len(result["content"]) >= 1
        text = result["content"][0]["text"]
        assert "joint" in text.lower() or "actuator" in text.lower()

    def test_run_policy_advances_sim_time(self):
        """run_policy should advance simulation time (physics steps happen)."""
        if not self.robot_available:
            pytest.skip("so100 robot not available")

        initial_time = self.sim._world.sim_time

        self.sim.run_policy(
            robot_name="arm",
            policy_provider="mock",
            instruction="test",
            duration=1.0,
            control_frequency=10.0,
        )

        final_time = self.sim._world.sim_time
        assert final_time > initial_time, f"Sim time did not advance: {initial_time} → {final_time}"

    def test_mock_policy_produces_actions(self):
        """Mock policy should produce valid actions (not empty)."""
        import asyncio

        from strands_robots.policies import create_policy

        policy = create_policy("mock")
        policy.set_robot_state_keys(["j1", "j2", "j3"])

        obs = {"j1": 0.0, "j2": 0.0, "j3": 0.0}
        actions = asyncio.run(policy.get_actions(obs, "test"))

        assert isinstance(actions, list)
        assert len(actions) > 0, "Mock policy returned empty actions"
        assert isinstance(actions[0], dict)
        assert len(actions[0]) > 0, "Mock policy returned empty action dict"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
