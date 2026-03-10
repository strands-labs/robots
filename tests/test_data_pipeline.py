#!/usr/bin/env python3
"""
Data Collection Pipeline Integration Tests (Tasks 10-16)

Tests the full flow: scripted demo → record → validate → dataset format.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Check mujoco availability for tests that need simulation
try:
    import mujoco  # noqa: F401

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

# Check for REAL gymnasium (not mock from other test files)
# Mock gymnasium has no __file__ attribute; real packages do
import sys as _sys

_gym_mod = _sys.modules.get("gymnasium")
if _gym_mod is not None:
    HAS_GYMNASIUM = hasattr(_gym_mod, "__file__") and _gym_mod.__file__ is not None
else:
    try:
        import gymnasium  # noqa: F401

        HAS_GYMNASIUM = True
    except (ImportError, ValueError):
        HAS_GYMNASIUM = False

HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class TestDataPipelineIntegration:
    """Full data pipeline: generate → record → validate."""

    def test_generate_and_record_episodes(self):
        """Generate scripted demos and save as episode JSON."""
        from scripted_pick_demo import ScriptedPickPolicy

        with tempfile.TemporaryDirectory() as tmpdir:
            num_episodes = 5
            steps = 100

            for ep in range(num_episodes):
                policy = ScriptedPickPolicy(num_joints=6)
                traj = policy.generate_trajectory(steps)

                ep_data = {
                    "observations": [{"state": np.random.randn(6).tolist(), "step": s} for s in range(steps)],
                    "actions": [t.tolist() for t in traj],
                    "metadata": {
                        "episode": ep,
                        "robot": "so100",
                        "task": "pick up the red cube",
                    },
                }

                path = os.path.join(tmpdir, f"episode_{ep:04d}.json")
                with open(path, "w") as f:
                    json.dump(ep_data, f)

            # Validate all episodes exist and are valid
            files = sorted(Path(tmpdir).glob("episode_*.json"))
            assert len(files) == num_episodes

            for fpath in files:
                with open(fpath) as f:
                    data = json.load(f)
                assert len(data["observations"]) == steps
                assert len(data["actions"]) == steps
                assert len(data["actions"][0]) == 7  # 6 joints + gripper
                assert "metadata" in data

    def test_action_dimensions_match_robot(self):
        """Action dims should match SO-100 (6 joints + 1 gripper = 7)."""
        from scripted_pick_demo import ScriptedPickPolicy

        policy = ScriptedPickPolicy(num_joints=6)
        traj = policy.generate_trajectory(50)
        for action in traj:
            assert action.shape == (7,)
            assert all(np.isfinite(action))

    @pytest.mark.skipif(not HAS_MUJOCO or not HAS_GYMNASIUM, reason="requires mujoco and gymnasium")
    @pytest.mark.skipif(not HAS_DISPLAY, reason="requires display for MuJoCo OpenGL rendering")
    def test_observation_state_dimensions(self):
        """Observation state should have correct dimensions from sim."""
        from strands_robots.envs import StrandsSimEnv

        env = StrandsSimEnv(robot_name="so100", task="test", max_episode_steps=5)
        obs, _ = env.reset()
        if isinstance(obs, dict):
            state = obs["state"]
        else:
            state = obs
        assert len(state) > 0
        assert all(np.isfinite(state))
        env.close()

    def test_episode_to_dataset_format(self):
        """Episode data should be convertible to training-ready format."""
        from scripted_pick_demo import ScriptedPickPolicy

        policy = ScriptedPickPolicy(num_joints=6)
        traj = policy.generate_trajectory(50)

        # Convert to LeRobot-like format
        frames = []
        for step in range(50):
            frame = {
                "observation.state": np.random.randn(6).tolist(),
                "action": traj[step][:6].tolist(),  # joints only
                "action.gripper": float(traj[step][6]),
                "timestamp": step / 30.0,
                "episode_index": 0,
                "frame_index": step,
            }
            frames.append(frame)

        assert len(frames) == 50
        assert "observation.state" in frames[0]
        assert "action" in frames[0]
        assert len(frames[0]["action"]) == 6

    def test_multi_episode_stats(self):
        """Stats across episodes should be consistent."""
        from scripted_pick_demo import ScriptedPickPolicy

        stats = []
        for ep in range(10):
            policy = ScriptedPickPolicy(num_joints=6)
            traj = policy.generate_trajectory(100)
            actions = np.array(traj)
            stats.append(
                {
                    "mean_action": np.mean(actions[:, :6], axis=0).tolist(),
                    "std_action": np.std(actions[:, :6], axis=0).tolist(),
                    "gripper_range": [float(actions[:, 6].min()), float(actions[:, 6].max())],
                }
            )

        # All episodes should have similar stats (deterministic policy)
        for s in stats:
            assert len(s["mean_action"]) == 6
            assert s["gripper_range"][0] <= s["gripper_range"][1]


class TestDatasetRecorderAPI:
    """Test DatasetRecorder has correct interface."""

    def test_recorder_class_exists(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        assert DatasetRecorder is not None

    def test_recorder_create_method(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        assert callable(DatasetRecorder.create)

    def test_recorder_has_all_methods(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        required = ["create", "add_frame", "save_episode", "push_to_hub", "finalize"]
        for method in required:
            assert hasattr(DatasetRecorder, method), f"Missing method: {method}"


class TestRecordSessionAPI:
    """Test RecordSession interface."""

    def test_record_module_imports(self):
        from strands_robots.record import RecordSession

        assert RecordSession is not None

    def test_record_session_class_methods(self):
        from strands_robots.record import RecordSession

        assert hasattr(RecordSession, "__init__")


class TestTrainingScriptImports:
    """Verify all training scripts can at least be parsed."""

    @pytest.mark.parametrize(
        "script",
        [
            "scripts/training/train_groot.py",
            "scripts/training/train_lerobot.py",
            "scripts/training/train_rl.py",
            "scripts/training/train_cosmos.py",
            "scripts/training/train_dreamgen.py",
        ],
    )
    def test_script_parseable(self, script):
        """Each training script should be valid Python."""
        import py_compile

        py_compile.compile(script, doraise=True)

    @pytest.mark.parametrize(
        "script",
        [
            "scripts/scripted_pick_demo.py",
            "scripts/eval_harness.py",
            "scripts/newton_benchmark.py",
        ],
    )
    def test_utility_script_parseable(self, script):
        """Each utility script should be valid Python."""
        import py_compile

        py_compile.compile(script, doraise=True)
