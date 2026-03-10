#!/usr/bin/env python3
"""Comprehensive tests for strands_robots.dataset_recorder module.

Tests cover:
1. Syntax validation
2. _numpy_ify() utility function
3. DatasetRecorder — init, properties, repr
4. DatasetRecorder._build_features() — all feature combinations
5. DatasetRecorder.create() — mocked LeRobotDataset
6. DatasetRecorder.add_frame() — state, action, camera key detection
7. DatasetRecorder.save_episode() and finalize()
8. DatasetRecorder.push_to_hub()
9. Edge cases: closed recorder, empty dicts, error handling

All tests run on CPU without LeRobot installed.
"""

import ast
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────
# 0. Syntax Validation
# ─────────────────────────────────────────────────────────────────────


class TestDatasetRecorderSyntax:
    """Validate dataset_recorder.py parses correctly."""

    MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "strands_robots", "dataset_recorder.py")

    def test_file_exists(self):
        assert os.path.isfile(self.MODULE_PATH)

    def test_syntax_valid(self):
        with open(self.MODULE_PATH) as f:
            source = f.read()
        ast.parse(source, filename=self.MODULE_PATH)

    def test_module_imports(self):
        from strands_robots import dataset_recorder

        assert hasattr(dataset_recorder, "DatasetRecorder")
        assert hasattr(dataset_recorder, "_numpy_ify")


# ─────────────────────────────────────────────────────────────────────
# 1. _numpy_ify() utility
# ─────────────────────────────────────────────────────────────────────


class TestNumpyIfy:
    """Test the _numpy_ify helper function."""

    def test_numpy_array_passthrough(self):
        from strands_robots.dataset_recorder import _numpy_ify

        arr = np.array([1.0, 2.0, 3.0])
        result = _numpy_ify(arr)
        np.testing.assert_array_equal(result, arr)

    def test_int_to_array(self):
        from strands_robots.dataset_recorder import _numpy_ify

        result = _numpy_ify(5)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [5])
        assert result.dtype == np.float32

    def test_float_to_array(self):
        from strands_robots.dataset_recorder import _numpy_ify

        result = _numpy_ify(3.14)
        assert isinstance(result, np.ndarray)
        assert abs(result[0] - 3.14) < 1e-5

    def test_list_to_array(self):
        from strands_robots.dataset_recorder import _numpy_ify

        result = _numpy_ify([1.0, 2.0, 3.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_tensor_with_numpy_method(self):
        """Objects with .numpy() method should be converted."""
        from strands_robots.dataset_recorder import _numpy_ify

        mock_tensor = MagicMock()
        mock_tensor.numpy.return_value = np.array([1.0, 2.0])
        result = _numpy_ify(mock_tensor)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_string_passthrough(self):
        from strands_robots.dataset_recorder import _numpy_ify

        result = _numpy_ify("hello")
        assert result == "hello"

    def test_none_passthrough(self):
        from strands_robots.dataset_recorder import _numpy_ify

        result = _numpy_ify(None)
        assert result is None


# ─────────────────────────────────────────────────────────────────────
# 2. DatasetRecorder Init & Properties
# ─────────────────────────────────────────────────────────────────────


class TestDatasetRecorderInit:
    """Test DatasetRecorder construction and properties."""

    def test_init(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        mock_dataset = MagicMock()
        mock_dataset.repo_id = "user/my_dataset"
        mock_dataset.root = "/tmp/data"

        recorder = DatasetRecorder(dataset=mock_dataset, task="pick up cube")
        assert recorder.dataset is mock_dataset
        assert recorder.default_task == "pick up cube"
        assert recorder.frame_count == 0
        assert recorder.episode_count == 0
        assert recorder._closed is False

    def test_repo_id_property(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        mock_dataset = MagicMock()
        mock_dataset.repo_id = "user/test"
        recorder = DatasetRecorder(dataset=mock_dataset)
        assert recorder.repo_id == "user/test"

    def test_root_property(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        mock_dataset = MagicMock()
        mock_dataset.root = "/tmp/data"
        recorder = DatasetRecorder(dataset=mock_dataset)
        assert recorder.root == "/tmp/data"

    def test_repr(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        mock_dataset = MagicMock()
        mock_dataset.repo_id = "user/test"
        recorder = DatasetRecorder(dataset=mock_dataset)
        recorder.episode_count = 3
        recorder.frame_count = 150
        r = repr(recorder)
        assert "user/test" in r
        assert "episodes=3" in r
        assert "frames=150" in r


# ─────────────────────────────────────────────────────────────────────
# 3. _build_features()
# ─────────────────────────────────────────────────────────────────────


class TestBuildFeatures:
    """Test DatasetRecorder._build_features() class method."""

    def test_empty_features(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        features = DatasetRecorder._build_features()
        assert features == {}

    def test_camera_keys_video(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        features = DatasetRecorder._build_features(
            camera_keys=["wrist", "front"],
            use_videos=True,
        )
        assert "observation.images.wrist" in features
        assert "observation.images.front" in features
        assert features["observation.images.wrist"]["dtype"] == "video"

    def test_camera_keys_image(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        features = DatasetRecorder._build_features(
            camera_keys=["wrist"],
            use_videos=False,
        )
        assert features["observation.images.wrist"]["dtype"] == "image"

    def test_joint_names_state_and_action(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        features = DatasetRecorder._build_features(
            joint_names=["shoulder", "elbow", "wrist"],
        )
        assert "observation.state" in features
        assert features["observation.state"]["shape"] == (3,)
        assert "action" in features
        assert features["action"]["shape"] == (3,)

    def test_robot_features_state(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        features = DatasetRecorder._build_features(
            robot_features={"j0": "float32", "j1": "float32", "j2": "float32"},
        )
        assert "observation.state" in features
        assert features["observation.state"]["shape"] == (3,)

    def test_action_features(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        features = DatasetRecorder._build_features(
            action_features={"act0": "float32", "act1": "float32"},
        )
        assert "action" in features
        assert features["action"]["shape"] == (2,)

    def test_action_dim_defaults_to_state_dim(self):
        """If no action_features and no joint_names, action dim matches state dim."""
        from strands_robots.dataset_recorder import DatasetRecorder

        features = DatasetRecorder._build_features(
            robot_features={"j0": "float32", "j1": "float32"},
        )
        assert features["action"]["shape"] == (2,)

    def test_camera_keys_excluded_from_state_count(self):
        """Camera features in robot_features dict should not count as state dims."""
        from strands_robots.dataset_recorder import DatasetRecorder

        features = DatasetRecorder._build_features(
            robot_features={
                "j0": "float32",
                "j1": "float32",
                "cam": {"dtype": "image"},
            },
        )
        assert features["observation.state"]["shape"] == (2,)

    def test_full_feature_set(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        features = DatasetRecorder._build_features(
            camera_keys=["wrist"],
            joint_names=["j0", "j1"],
            use_videos=True,
        )
        assert "observation.images.wrist" in features
        assert "observation.state" in features
        assert "action" in features


# ─────────────────────────────────────────────────────────────────────
# 4. DatasetRecorder.create() (mocked)
# ─────────────────────────────────────────────────────────────────────


class TestDatasetRecorderCreate:
    """Test create() factory method with mocked LeRobotDataset."""

    def test_create_without_lerobot_raises(self):
        """Without lerobot, create() should raise ImportError."""
        from strands_robots.dataset_recorder import DatasetRecorder

        with patch.object(DatasetRecorder, "__init__", return_value=None):
            import strands_robots.dataset_recorder as dr

            original = dr.HAS_LEROBOT_DATASET
            dr.HAS_LEROBOT_DATASET = False
            try:
                with pytest.raises(ImportError, match="lerobot not installed"):
                    DatasetRecorder.create(repo_id="user/test")
            finally:
                dr.HAS_LEROBOT_DATASET = original

    def test_create_with_mocked_lerobot(self):
        import strands_robots.dataset_recorder as dr

        mock_dataset_class = MagicMock()
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.repo_id = "user/test"
        mock_dataset_class.create.return_value = mock_dataset_instance

        original_has = dr.HAS_LEROBOT_DATASET
        original_cls = dr.LeRobotDataset if hasattr(dr, "LeRobotDataset") else None
        dr.HAS_LEROBOT_DATASET = True
        dr.LeRobotDataset = mock_dataset_class
        try:
            recorder = dr.DatasetRecorder.create(
                repo_id="user/test",
                fps=30,
                joint_names=["j0", "j1"],
                task="test task",
            )
            assert isinstance(recorder, dr.DatasetRecorder)
            assert recorder.default_task == "test task"
            mock_dataset_class.create.assert_called_once()
        finally:
            dr.HAS_LEROBOT_DATASET = original_has
            if original_cls is not None:
                dr.LeRobotDataset = original_cls


# ─────────────────────────────────────────────────────────────────────
# 5. DatasetRecorder.add_frame()
# ─────────────────────────────────────────────────────────────────────


class TestDatasetRecorderAddFrame:
    """Test add_frame() method."""

    def _make_recorder(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        mock_dataset = MagicMock()
        mock_dataset.repo_id = "user/test"
        mock_dataset.root = "/tmp"
        return DatasetRecorder(dataset=mock_dataset, task="default task")

    def test_add_frame_basic(self):
        recorder = self._make_recorder()
        obs = {"j0": 0.1, "j1": 0.2}
        action = {"j0": 0.5, "j1": 0.6}
        recorder.add_frame(obs, action)
        assert recorder.frame_count == 1
        recorder.dataset.add_frame.assert_called_once()

    def test_add_frame_with_camera(self):
        recorder = self._make_recorder()
        obs = {
            "j0": 0.1,
            "cam": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        action = {"j0": 0.5}
        recorder.add_frame(obs, action)

        call_args = recorder.dataset.add_frame.call_args[0][0]
        assert "observation.images.cam" in call_args
        assert "observation.state" in call_args

    def test_add_frame_custom_camera_keys(self):
        recorder = self._make_recorder()
        obs = {
            "rgb": np.zeros((240, 320, 3), dtype=np.uint8),
            "depth": np.zeros((240, 320), dtype=np.float32),
            "j0": 0.1,
        }
        action = {"j0": 0.5}
        recorder.add_frame(obs, action, camera_keys=["rgb"])

        call_args = recorder.dataset.add_frame.call_args[0][0]
        assert "observation.images.rgb" in call_args

    def test_add_frame_task_override(self):
        recorder = self._make_recorder()
        recorder.add_frame({"j0": 0.1}, {"j0": 0.5}, task="custom task")
        call_args = recorder.dataset.add_frame.call_args[0][0]
        assert call_args["task"] == "custom task"

    def test_add_frame_default_task(self):
        recorder = self._make_recorder()
        recorder.add_frame({"j0": 0.1}, {"j0": 0.5})
        call_args = recorder.dataset.add_frame.call_args[0][0]
        assert call_args["task"] == "default task"

    def test_add_frame_empty_task_fallback(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        mock_dataset = MagicMock()
        mock_dataset.repo_id = "test"
        mock_dataset.root = "/tmp"
        recorder = DatasetRecorder(dataset=mock_dataset, task="")
        recorder.add_frame({"j0": 0.1}, {"j0": 0.5})
        call_args = recorder.dataset.add_frame.call_args[0][0]
        assert call_args["task"] == "untitled"

    def test_add_frame_when_closed(self):
        recorder = self._make_recorder()
        recorder._closed = True
        recorder.add_frame({"j0": 0.1}, {"j0": 0.5})
        assert recorder.frame_count == 0
        recorder.dataset.add_frame.assert_not_called()

    def test_add_frame_exception_handled(self):
        recorder = self._make_recorder()
        recorder.dataset.add_frame.side_effect = RuntimeError("Write error")
        recorder.add_frame({"j0": 0.1}, {"j0": 0.5})
        # Frame count should not increment on error
        assert recorder.frame_count == 0

    def test_add_frame_multiple(self):
        recorder = self._make_recorder()
        for i in range(10):
            recorder.add_frame({"j0": float(i)}, {"j0": float(i)})
        assert recorder.frame_count == 10

    def test_add_frame_state_as_list(self):
        recorder = self._make_recorder()
        obs = {"joint_pos": [0.1, 0.2, 0.3]}
        action = {"ctrl": [0.5, 0.6, 0.7]}
        recorder.add_frame(obs, action)
        call_args = recorder.dataset.add_frame.call_args[0][0]
        assert "observation.state" in call_args
        assert len(call_args["observation.state"]) == 3

    def test_add_frame_state_as_numpy(self):
        recorder = self._make_recorder()
        obs = {"state_vec": np.array([0.1, 0.2])}
        action = {"act": np.array([0.5])}
        recorder.add_frame(obs, action)
        call_args = recorder.dataset.add_frame.call_args[0][0]
        assert "observation.state" in call_args

    def test_add_frame_float_image_conversion(self):
        """Float images (0-1) should be converted to uint8 (0-255)."""
        recorder = self._make_recorder()
        float_img = np.random.rand(240, 320, 3).astype(np.float32)
        obs = {"cam": float_img, "j0": 0.1}
        action = {"j0": 0.5}
        recorder.add_frame(obs, action)
        call_args = recorder.dataset.add_frame.call_args[0][0]
        img = call_args["observation.images.cam"]
        assert img.dtype == np.uint8

    def test_add_frame_empty_action(self):
        recorder = self._make_recorder()
        recorder.add_frame({"j0": 0.1}, {})
        call_args = recorder.dataset.add_frame.call_args[0][0]
        assert "action" not in call_args


# ─────────────────────────────────────────────────────────────────────
# 6. save_episode() and finalize()
# ─────────────────────────────────────────────────────────────────────


class TestSaveEpisodeFinalize:
    """Test save_episode() and finalize() methods."""

    def _make_recorder(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        mock_dataset = MagicMock()
        mock_dataset.repo_id = "user/test"
        mock_dataset.root = "/tmp"
        return DatasetRecorder(dataset=mock_dataset, task="test")

    def test_save_episode_success(self):
        recorder = self._make_recorder()
        recorder.frame_count = 100
        result = recorder.save_episode()
        assert result["status"] == "success"
        assert result["episode"] == 1
        assert recorder.episode_count == 1
        recorder.dataset.save_episode.assert_called_once()

    def test_save_episode_multiple(self):
        recorder = self._make_recorder()
        recorder.save_episode()
        recorder.save_episode()
        assert recorder.episode_count == 2

    def test_save_episode_when_closed(self):
        recorder = self._make_recorder()
        recorder._closed = True
        result = recorder.save_episode()
        assert result["status"] == "error"

    def test_save_episode_exception(self):
        recorder = self._make_recorder()
        recorder.dataset.save_episode.side_effect = RuntimeError("Disk full")
        result = recorder.save_episode()
        assert result["status"] == "error"
        assert "Disk full" in result["message"]

    def test_finalize(self):
        recorder = self._make_recorder()
        recorder.finalize()
        assert recorder._closed is True
        recorder.dataset.finalize.assert_called_once()

    def test_finalize_when_closed(self):
        recorder = self._make_recorder()
        recorder._closed = True
        recorder.finalize()
        recorder.dataset.finalize.assert_not_called()

    def test_finalize_exception_handled(self):
        recorder = self._make_recorder()
        recorder.dataset.finalize.side_effect = RuntimeError("Error")
        recorder.finalize()  # Should not raise
        assert recorder._closed is True


# ─────────────────────────────────────────────────────────────────────
# 7. push_to_hub()
# ─────────────────────────────────────────────────────────────────────


class TestPushToHub:
    """Test push_to_hub() method."""

    def _make_recorder(self):
        from strands_robots.dataset_recorder import DatasetRecorder

        mock_dataset = MagicMock()
        mock_dataset.repo_id = "user/test"
        mock_dataset.root = "/tmp"
        return DatasetRecorder(dataset=mock_dataset, task="test")

    def test_push_success(self):
        recorder = self._make_recorder()
        recorder.episode_count = 5
        recorder.frame_count = 500
        result = recorder.push_to_hub()
        assert result["status"] == "success"
        assert result["repo_id"] == "user/test"
        assert result["episodes"] == 5
        assert result["frames"] == 500

    def test_push_with_tags(self):
        recorder = self._make_recorder()
        recorder.push_to_hub(tags=["so100", "pick"], private=True)
        recorder.dataset.push_to_hub.assert_called_once_with(
            tags=["so100", "pick"],
            private=True,
        )

    def test_push_failure(self):
        recorder = self._make_recorder()
        recorder.dataset.push_to_hub.side_effect = RuntimeError("Auth failed")
        result = recorder.push_to_hub()
        assert result["status"] == "error"
        assert "Auth failed" in result["message"]


# ─────────────────────────────────────────────────────────────────────
# 8. Edge Cases
# ─────────────────────────────────────────────────────────────────────


class TestDatasetRecorderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_scalar_observation_values(self):
        """Scalar 0-dim numpy arrays in observations."""
        from strands_robots.dataset_recorder import DatasetRecorder

        mock_dataset = MagicMock()
        mock_dataset.repo_id = "test"
        mock_dataset.root = "/tmp"
        recorder = DatasetRecorder(dataset=mock_dataset)

        obs = {"j0": np.float32(0.5), "j1": np.array(0.3)}
        action = {"a0": 1.0}
        recorder.add_frame(obs, action)
        assert recorder.frame_count == 1

    def test_mixed_state_types(self):
        """Mix of int, float, list, ndarray in observation."""
        from strands_robots.dataset_recorder import DatasetRecorder

        mock_dataset = MagicMock()
        mock_dataset.repo_id = "test"
        mock_dataset.root = "/tmp"
        recorder = DatasetRecorder(dataset=mock_dataset)

        obs = {
            "position": [0.1, 0.2],
            "velocity": np.array([0.3, 0.4]),
            "gripper": 0.5,
            "count": 3,
        }
        action = {"a0": 0.1}
        recorder.add_frame(obs, action)
        assert recorder.frame_count == 1

        call_args = recorder.dataset.add_frame.call_args[0][0]
        state = call_args["observation.state"]
        assert len(state) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
