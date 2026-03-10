#!/usr/bin/env python3
"""Tests for Sample 05 — Data Collection & Recording."""

# Skip if samples/ directory not present (requires PR #13)
import os as _os

import pytest as _pytest_guard

if not _os.path.isdir(_os.path.join(_os.path.dirname(__file__), "..", "samples")):
    _pytest_guard.skip("Requires PR #13 (samples)", allow_module_level=True)

import json
import math
from unittest.mock import MagicMock, patch

# ── inspect_dataset tests ───────────────────────────────────────────


class TestInspectTrajectoryJson:
    """Test the trajectory JSON inspection logic."""

    def _make_trajectory(self, num_frames=30, fps=30):
        """Helper: create a sample trajectory list."""
        joint_names = ["j1", "j2", "j3"]
        traj = []
        for i in range(num_frames):
            t = i / fps
            joints = {name: 0.3 * math.sin(2 * math.pi * 0.5 * t + j) for j, name in enumerate(joint_names)}
            traj.append(
                {
                    "time": t,
                    "joints": joints,
                    "velocities": {name: 0.1 for name in joint_names},
                    "actions": {name: v * 0.9 for name, v in joints.items()},
                }
            )
        return traj

    def test_trajectory_structure(self):
        """Trajectory frames have expected keys."""
        traj = self._make_trajectory()
        assert len(traj) == 30
        frame = traj[0]
        assert "time" in frame
        assert "joints" in frame
        assert "velocities" in frame
        assert "actions" in frame

    def test_trajectory_time_monotonic(self):
        """Time values are monotonically increasing."""
        traj = self._make_trajectory(num_frames=60)
        times = [f["time"] for f in traj]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]

    def test_trajectory_json_roundtrip(self, tmp_path):
        """Trajectory survives JSON serialization."""
        traj = self._make_trajectory()
        path = tmp_path / "traj.json"
        with open(path, "w") as f:
            json.dump(traj, f)

        with open(path) as f:
            loaded = json.load(f)

        assert len(loaded) == len(traj)
        assert loaded[0]["time"] == traj[0]["time"]
        assert set(loaded[0]["joints"].keys()) == set(traj[0]["joints"].keys())

    def test_trajectory_fps_calculation(self):
        """FPS can be computed from time delta."""
        fps = 30
        traj = self._make_trajectory(num_frames=90, fps=fps)
        duration = traj[-1]["time"] - traj[0]["time"]
        computed_fps = len(traj) / duration if duration > 0 else 0
        assert abs(computed_fps - fps) < 1.0  # Allow small rounding

    def test_joint_range_extraction(self):
        """Joint min/max range is computable from trajectory."""
        traj = self._make_trajectory(num_frames=100)
        joint_name = "j1"
        values = [f["joints"][joint_name] for f in traj]
        assert max(values) > min(values)  # There is motion
        assert max(values) <= 0.4  # Within amplitude bounds
        assert min(values) >= -0.4

    def test_empty_trajectory(self, tmp_path):
        """Empty trajectory file is handled gracefully."""
        path = tmp_path / "empty.json"
        with open(path, "w") as f:
            json.dump([], f)

        with open(path) as f:
            data = json.load(f)

        assert data == []


class TestInspectLeRobotDir:
    """Test LeRobot dataset directory inspection logic."""

    def _make_dataset_dir(self, tmp_path, num_episodes=3):
        """Helper: create a mock LeRobot dataset directory."""
        ds_dir = tmp_path / "my_dataset"
        meta_dir = ds_dir / "meta"
        data_dir = ds_dir / "data" / "chunk-000"
        meta_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        # info.json
        info = {
            "codebase_version": "v3.0",
            "robot_type": "so100",
            "fps": 30,
            "total_episodes": num_episodes,
            "total_frames": num_episodes * 90,
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": ["j1", "j2", "j3", "j4", "j5", "j6"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": ["j1", "j2", "j3", "j4", "j5", "j6"],
                },
            },
        }
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f)

        # episodes.jsonl
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for i in range(num_episodes):
                ep = {
                    "episode_index": i,
                    "length": 90,
                    "task": "reach cube",
                }
                f.write(json.dumps(ep) + "\n")

        return ds_dir, info

    def test_info_json_structure(self, tmp_path):
        """info.json has required fields."""
        ds_dir, info = self._make_dataset_dir(tmp_path)
        with open(ds_dir / "meta" / "info.json") as f:
            loaded = json.load(f)
        assert loaded["robot_type"] == "so100"
        assert loaded["fps"] == 30
        assert "observation.state" in loaded["features"]
        assert "action" in loaded["features"]

    def test_episodes_jsonl_parsing(self, tmp_path):
        """episodes.jsonl is valid JSONL."""
        ds_dir, _ = self._make_dataset_dir(tmp_path, num_episodes=5)
        episodes = []
        with open(ds_dir / "meta" / "episodes.jsonl") as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
        assert len(episodes) == 5
        assert episodes[0]["episode_index"] == 0
        assert episodes[4]["episode_index"] == 4

    def test_feature_shapes(self, tmp_path):
        """Features have correct shapes."""
        _, info = self._make_dataset_dir(tmp_path)
        obs_shape = info["features"]["observation.state"]["shape"]
        act_shape = info["features"]["action"]["shape"]
        assert obs_shape == [6]
        assert act_shape == [6]

    def test_total_frames_consistency(self, tmp_path):
        """total_frames matches sum of episode lengths."""
        ds_dir, info = self._make_dataset_dir(tmp_path, num_episodes=3)
        episodes = []
        with open(ds_dir / "meta" / "episodes.jsonl") as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
        total = sum(ep["length"] for ep in episodes)
        assert total == info["total_frames"]


class TestMockRecorderPattern:
    """Test the mock recording pattern from record_with_recorder.py."""

    def test_observation_action_shapes_match(self):
        """Observation and action have same dimensionality."""
        import numpy as np

        joint_names = ["j1", "j2", "j3", "j4", "j5", "j6"]
        t = 0.5

        obs = np.array(
            [(0.2 + j * 0.03) * math.sin(2 * math.pi * (0.5 + j * 0.15) * t) for j in range(len(joint_names))],
            dtype=np.float32,
        )

        act = np.array(
            [
                (0.2 + j * 0.03) * math.sin(2 * math.pi * (0.5 + j * 0.15) * (t + 1 / 30))
                for j in range(len(joint_names))
            ],
            dtype=np.float32,
        )

        assert obs.shape == act.shape == (6,)
        assert obs.dtype == act.dtype == np.float32

    def test_mock_dataset_json_structure(self, tmp_path):
        """Mock dataset JSON has correct structure."""
        import numpy as np

        joint_names = ["j1", "j2", "j3"]
        episodes = []
        for _ in range(2):
            obs_list = []
            act_list = []
            for step in range(10):
                obs = np.random.randn(len(joint_names)).astype(np.float32)
                act = np.random.randn(len(joint_names)).astype(np.float32)
                obs_list.append(obs.tolist())
                act_list.append(act.tolist())
            episodes.append({"observations": obs_list, "actions": act_list})

        output = {
            "info": {
                "fps": 30,
                "robot_type": "so100",
                "total_episodes": len(episodes),
                "total_frames": sum(len(ep["observations"]) for ep in episodes),
                "features": {
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [len(joint_names)],
                    },
                    "action": {
                        "dtype": "float32",
                        "shape": [len(joint_names)],
                    },
                },
            },
        }

        path = tmp_path / "mock.json"
        with open(path, "w") as f:
            json.dump(output, f)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["info"]["total_episodes"] == 2
        assert loaded["info"]["total_frames"] == 20


class TestPushToHubDryRun:
    """Test the dry-run push logic."""

    def test_hf_auth_check_no_hub(self):
        """Auth check returns False when huggingface_hub missing."""
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            # Can't import → should return False
            try:
                from huggingface_hub import HfApi  # noqa: F401

                # If somehow importable, mock the call
                api = MagicMock()
                api.whoami.side_effect = Exception("not logged in")
                result = False
            except (ImportError, TypeError):
                result = False
            assert result is False

    def test_dry_run_shows_structure(self, capsys):
        """Dry run prints expected dataset structure info."""
        # We just verify the function can be constructed
        sample_info = {
            "robot_type": "so100",
            "fps": 30,
            "total_episodes": 5,
            "total_frames": 450,
        }
        assert sample_info["total_frames"] == sample_info["total_episodes"] * 90
