#!/usr/bin/env python3
"""Comprehensive tests for strands_robots.motion_library module.

Tests cover:
1. Syntax validation
2. Motion data class — construction, get_action_at, as_numpy, repr
3. MotionLibrary — init, register, get, list, search, count
4. MotionLibrary — load_file, load_directory (with temp files)
5. MotionLibrary — preload (mocked huggingface_hub)
6. MotionLibrary — save, play_in_sim (mocked mujoco)
7. Edge cases: empty data, missing keys, type variations

All tests run on CPU without any heavy dependencies.
"""

import ast
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────
# 0. Syntax Validation
# ─────────────────────────────────────────────────────────────────────


class TestMotionLibrarySyntax:
    """Validate motion_library.py parses correctly."""

    MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "strands_robots", "motion_library.py")

    def test_file_exists(self):
        assert os.path.isfile(self.MODULE_PATH)

    def test_syntax_valid(self):
        with open(self.MODULE_PATH) as f:
            source = f.read()
        ast.parse(source, filename=self.MODULE_PATH)

    def test_module_imports(self):
        from strands_robots import motion_library

        assert hasattr(motion_library, "Motion")
        assert hasattr(motion_library, "MotionLibrary")


# ─────────────────────────────────────────────────────────────────────
# 1. Motion Data Class
# ─────────────────────────────────────────────────────────────────────


class TestMotion:
    """Test the Motion data class."""

    def test_basic_construction(self):
        from strands_robots.motion_library import Motion

        data = {
            "time": [0.0, 0.5, 1.0],
            "joint_positions": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            "description": "wave",
        }
        m = Motion("wave", data, source="test")
        assert m.name == "wave"
        assert m.source == "test"
        assert m.n_steps == 3
        assert m.duration == 1.0
        assert m.description == "wave"

    def test_timestamps_key(self):
        """Support 'timestamps' as well as 'time'."""
        from strands_robots.motion_library import Motion

        data = {"timestamps": [0.0, 1.0, 2.0], "joint_positions": [[1], [2], [3]]}
        m = Motion("test", data)
        assert m.n_steps == 3
        assert m.duration == 2.0

    def test_trajectory_key(self):
        """Support 'trajectory' as well as 'joint_positions'."""
        from strands_robots.motion_library import Motion

        data = {"time": [0.0, 0.5], "trajectory": [[0.1], [0.2]]}
        m = Motion("test", data)
        assert len(m.joint_positions) == 2

    def test_actions_key(self):
        """Support 'actions' as well as 'joint_positions'."""
        from strands_robots.motion_library import Motion

        data = {"time": [0.0], "actions": [[0.1, 0.2, 0.3]]}
        m = Motion("test", data)
        assert len(m.joint_positions) == 1

    def test_empty_data(self):
        from strands_robots.motion_library import Motion

        m = Motion("empty", {})
        assert m.n_steps == 0
        assert m.duration == 0.0
        assert m.joint_positions == []

    def test_description_defaults_to_name(self):
        from strands_robots.motion_library import Motion

        m = Motion("my_motion", {})
        assert m.description == "my_motion"

    def test_repr(self):
        from strands_robots.motion_library import Motion

        data = {"time": [0.0, 1.0], "joint_positions": [[1], [2]]}
        m = Motion("wave", data)
        r = repr(m)
        assert "wave" in r
        assert "steps=2" in r
        assert "1.0s" in r

    def test_get_action_at_empty(self):
        from strands_robots.motion_library import Motion

        m = Motion("empty", {})
        assert m.get_action_at(0.5) is None

    def test_get_action_at_dict_positions(self):
        from strands_robots.motion_library import Motion

        data = {
            "time": [0.0, 1.0, 2.0],
            "joint_positions": [
                {"j0": 0.0, "j1": 0.0},
                {"j0": 1.0, "j1": 1.0},
                {"j0": 2.0, "j1": 2.0},
            ],
        }
        m = Motion("test", data)
        action = m.get_action_at(0.5)
        assert isinstance(action, dict)
        assert "j0" in action

    def test_get_action_at_list_positions(self):
        from strands_robots.motion_library import Motion

        data = {
            "time": [0.0, 1.0],
            "joint_positions": [[0.1, 0.2], [0.3, 0.4]],
        }
        m = Motion("test", data)
        action = m.get_action_at(0.0)
        assert isinstance(action, dict)
        assert "joint_0" in action

    def test_get_action_at_clamped_to_duration(self):
        from strands_robots.motion_library import Motion

        data = {
            "time": [0.0, 1.0],
            "joint_positions": [[0.1], [0.2]],
        }
        m = Motion("test", data)
        # Beyond duration, should clamp
        action = m.get_action_at(5.0)
        assert action is not None

    def test_get_action_at_negative_time(self):
        from strands_robots.motion_library import Motion

        data = {
            "time": [0.0, 1.0],
            "joint_positions": [[0.1], [0.2]],
        }
        m = Motion("test", data)
        action = m.get_action_at(-1.0)
        assert action is not None

    def test_as_numpy_list_positions(self):
        from strands_robots.motion_library import Motion

        data = {
            "time": [0.0, 0.5, 1.0],
            "joint_positions": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        }
        m = Motion("test", data)
        arr = m.as_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 2)
        np.testing.assert_array_equal(arr[0], [1.0, 2.0])

    def test_as_numpy_dict_positions(self):
        from strands_robots.motion_library import Motion

        data = {
            "time": [0.0, 1.0],
            "joint_positions": [
                {"a": 1.0, "b": 2.0},
                {"a": 3.0, "b": 4.0},
            ],
        }
        m = Motion("test", data)
        arr = m.as_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)

    def test_as_numpy_empty(self):
        from strands_robots.motion_library import Motion

        m = Motion("empty", {})
        assert m.as_numpy() is None

    def test_as_numpy_non_list_non_dict_returns_none(self):
        """If joint_positions contains unexpected types, as_numpy returns None."""
        from strands_robots.motion_library import Motion

        data = {"time": [0.0], "joint_positions": ["string_data"]}
        m = Motion("test", data)
        # first element is string → neither dict nor list
        assert m.as_numpy() is None


# ─────────────────────────────────────────────────────────────────────
# 2. MotionLibrary — Basic Operations
# ─────────────────────────────────────────────────────────────────────


class TestMotionLibraryBasic:
    """Test MotionLibrary init, register, get, list, search, count."""

    def test_init_default(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        assert lib.count == 0
        assert lib._motions == {}
        assert lib._loaded_repos == set()

    def test_init_custom_cache_dir(self):
        from strands_robots.motion_library import MotionLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            lib = MotionLibrary(cache_dir=tmpdir)
            assert str(lib._cache_dir) == tmpdir

    def test_register(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        data = {"time": [0.0, 1.0], "joint_positions": [[0.1], [0.2]]}
        motion = lib.register("wave", data)
        assert motion.name == "wave"
        assert motion.source == "in-memory"
        assert lib.count == 1

    def test_get_existing(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib.register("wave", {"time": [0.0], "joint_positions": [[0.1]]})
        m = lib.get("wave")
        assert m is not None
        assert m.name == "wave"

    def test_get_missing(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        assert lib.get("nonexistent") is None

    def test_list_motions(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib.register("wave", {"time": [0.0, 1.0], "joint_positions": [[0.1], [0.2]]})
        lib.register("nod", {"time": [0.0, 0.5], "joint_positions": [[0.3], [0.4]]})
        motions = lib.list_motions()
        assert len(motions) == 2
        names = {m["name"] for m in motions}
        assert names == {"wave", "nod"}
        # Each entry should have expected keys
        for m in motions:
            assert "name" in m
            assert "duration" in m
            assert "steps" in m

    def test_search_by_name(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib.register("happy_wave", {"time": [0.0], "joint_positions": [[0.1]]})
        lib.register("sad_nod", {"time": [0.0], "joint_positions": [[0.2]]})
        lib.register("happy_dance", {"time": [0.0], "joint_positions": [[0.3]]})

        results = lib.search("happy")
        assert len(results) == 2
        names = {m.name for m in results}
        assert names == {"happy_wave", "happy_dance"}

    def test_search_case_insensitive(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib.register("WAVE", {"time": [0.0], "joint_positions": [[0.1]]})
        results = lib.search("wave")
        assert len(results) == 1

    def test_search_by_description(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib.register("m1", {"time": [0.0], "joint_positions": [[0.1]], "description": "happy greeting"})
        lib.register("m2", {"time": [0.0], "joint_positions": [[0.2]], "description": "sad farewell"})
        results = lib.search("greeting")
        assert len(results) == 1
        assert results[0].name == "m1"

    def test_search_empty_query(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib.register("wave", {"time": [0.0], "joint_positions": [[0.1]]})
        results = lib.search("")
        assert len(results) == 1  # Empty string matches everything

    def test_count_property(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        assert lib.count == 0
        lib.register("a", {})
        assert lib.count == 1
        lib.register("b", {})
        assert lib.count == 2

    def test_repr(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib.register("wave", {})
        r = repr(lib)
        assert "MotionLibrary" in r
        assert "1 motions" in r


# ─────────────────────────────────────────────────────────────────────
# 3. MotionLibrary — File I/O
# ─────────────────────────────────────────────────────────────────────


class TestMotionLibraryFileIO:
    """Test load_file, load_directory, save."""

    def test_load_file_single_motion(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()

        data = {"name": "wave", "time": [0.0, 1.0], "joint_positions": [[0.1], [0.2]]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            motion = lib.load_file(path)
            assert motion is not None
            assert motion.name == "wave"
            assert lib.count == 1
        finally:
            os.unlink(path)

    def test_load_file_custom_name(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()

        data = {"time": [0.0], "joint_positions": [[0.1]]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            motion = lib.load_file(path, name="custom_name")
            assert motion.name == "custom_name"
        finally:
            os.unlink(path)

    def test_load_file_invalid_json(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json{{{")
            path = f.name

        try:
            motion = lib.load_file(path)
            assert motion is None
        finally:
            os.unlink(path)

    def test_load_file_nonexistent(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        motion = lib.load_file("/nonexistent/path.json")
        assert motion is None

    def test_load_directory_single_motion_files(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["wave", "nod", "bow"]:
                data = {"name": name, "time": [0.0, 1.0], "joint_positions": [[0.1], [0.2]]}
                with open(os.path.join(tmpdir, f"{name}.json"), "w") as f:
                    json.dump(data, f)

            count = lib.load_directory(tmpdir, source="test_dir")
            assert count == 3
            assert lib.count == 3

    def test_load_directory_list_format(self):
        """JSON file containing a list of motions."""
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            data = [
                {"name": "wave", "time": [0.0], "joint_positions": [[0.1]]},
                {"name": "nod", "time": [0.0], "joint_positions": [[0.2]]},
            ]
            with open(os.path.join(tmpdir, "motions.json"), "w") as f:
                json.dump(data, f)

            count = lib.load_directory(tmpdir)
            assert count == 2

    def test_load_directory_dict_collection(self):
        """JSON file containing a dict of named motions."""
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "wave": {"time": [0.0], "joint_positions": [[0.1]]},
                "nod": {"time": [0.0], "joint_positions": [[0.2]]},
            }
            with open(os.path.join(tmpdir, "collection.json"), "w") as f:
                json.dump(data, f)

            count = lib.load_directory(tmpdir)
            assert count == 2

    def test_load_directory_recursive(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "sub", "dir")
            os.makedirs(subdir)
            data = {"name": "deep", "time": [0.0], "joint_positions": [[0.1]]}
            with open(os.path.join(subdir, "motion.json"), "w") as f:
                json.dump(data, f)

            count = lib.load_directory(tmpdir)
            assert count == 1

    def test_load_directory_skips_invalid_files(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid file
            with open(os.path.join(tmpdir, "good.json"), "w") as f:
                json.dump({"name": "good", "time": [0.0], "joint_positions": [[0.1]]}, f)
            # Invalid file
            with open(os.path.join(tmpdir, "bad.json"), "w") as f:
                f.write("not json")

            count = lib.load_directory(tmpdir)
            assert count == 1

    def test_save_motion(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib.register("wave", {"time": [0.0, 1.0], "joint_positions": [[0.1], [0.2]]})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = lib.save("wave", path=os.path.join(tmpdir, "wave.json"))
            assert os.path.isfile(path)

            with open(path) as f:
                saved = json.load(f)
            assert "time" in saved

    def test_save_creates_directory(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib.register("wave", {"time": [0.0], "joint_positions": [[0.1]]})

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "sub", "dir", "wave.json")
            path = lib.save("wave", path=nested)
            assert os.path.isfile(path)

    def test_save_nonexistent_raises(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        with pytest.raises(ValueError, match="not found"):
            lib.save("nonexistent")

    def test_save_default_path(self):
        from strands_robots.motion_library import MotionLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            lib = MotionLibrary(cache_dir=tmpdir)
            lib.register("wave", {"time": [0.0], "joint_positions": [[0.1]]})
            path = lib.save("wave")
            assert os.path.isfile(path)
            assert tmpdir in path


# ─────────────────────────────────────────────────────────────────────
# 4. MotionLibrary — preload (mocked HF hub)
# ─────────────────────────────────────────────────────────────────────


class TestMotionLibraryPreload:
    """Test preload from HuggingFace (mocked)."""

    def test_preload_success(self):
        from strands_robots.motion_library import MotionLibrary

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a motion file in the "downloaded" directory
            data = {"name": "hf_wave", "time": [0.0], "joint_positions": [[0.1]]}
            with open(os.path.join(tmpdir, "motion.json"), "w") as f:
                json.dump(data, f)

            with patch("strands_robots.motion_library.snapshot_download", create=True):
                # We need to mock the import inside the method
                mock_sd = MagicMock(return_value=tmpdir)

                lib = MotionLibrary()
                # Patch the import inside preload
                with patch.dict("sys.modules", {"huggingface_hub": MagicMock(snapshot_download=mock_sd)}):
                    count = lib.preload("pollen-robotics/test-library")
                    assert count == 1
                    assert "pollen-robotics/test-library" in lib._loaded_repos

    def test_preload_already_loaded(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib._loaded_repos.add("already/loaded")
        count = lib.preload("already/loaded")
        assert count == 0

    def test_preload_no_huggingface_hub(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()

        with patch.dict("sys.modules", {"huggingface_hub": None}):
            count = lib.preload("some/repo")
            assert count == 0

    def test_preload_download_failure(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()

        mock_hf = MagicMock()
        mock_hf.snapshot_download.side_effect = RuntimeError("Network error")

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            count = lib.preload("failing/repo")
            assert count == 0


# ─────────────────────────────────────────────────────────────────────
# 5. MotionLibrary — play_in_sim (mocked)
# ─────────────────────────────────────────────────────────────────────


class TestMotionLibraryPlayInSim:
    """Test play_in_sim with mocked simulation."""

    def test_play_motion_not_found(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        result = lib.play_in_sim(MagicMock(), "robot", "nonexistent")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_play_no_trajectory(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib.register("empty", {})
        result = lib.play_in_sim(MagicMock(), "robot", "empty")
        assert result["status"] == "error"
        assert "no valid trajectory" in result["message"].lower()

    def test_play_success_mocked(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib.register(
            "wave",
            {
                "time": [0.0, 0.5, 1.0],
                "joint_positions": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            },
        )

        mock_mj = MagicMock()
        mock_model = MagicMock()
        mock_model.nu = 2
        mock_model.opt.timestep = 0.002
        mock_data = MagicMock()
        mock_data.ctrl = np.zeros(2)

        mock_world = MagicMock()
        mock_world._model = mock_model
        mock_world._data = mock_data

        mock_sim = MagicMock()
        mock_sim._world = mock_world

        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            result = lib.play_in_sim(mock_sim, "robot", "wave")

        assert result["status"] == "success"

    def test_play_no_world(self):
        from strands_robots.motion_library import MotionLibrary

        lib = MotionLibrary()
        lib.register(
            "wave",
            {
                "time": [0.0, 1.0],
                "joint_positions": [[0.1], [0.2]],
            },
        )

        mock_sim = MagicMock()
        mock_sim._world = None

        mock_mj = MagicMock()

        with patch.dict("sys.modules", {"mujoco": mock_mj}):
            result = lib.play_in_sim(mock_sim, "robot", "wave")
        assert result["status"] == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
