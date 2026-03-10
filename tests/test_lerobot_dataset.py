"""Tests for strands_robots/tools/lerobot_dataset.py — dataset management tool.

All LeRobot and HuggingFace Hub operations are mocked. CPU-only.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock strands if not installed so tests can run without the full SDK
try:
    import strands

    HAS_STRANDS = hasattr(strands, "Agent")
except ImportError:
    import types

    _mock_strands = types.ModuleType("strands")
    _mock_strands.tool = lambda f: f  # @tool decorator becomes identity
    sys.modules["strands"] = _mock_strands
    HAS_STRANDS = False

from strands_robots.tools.lerobot_dataset import (
    _ACTIVE_RECORDINGS,
    _RECORDING_LOCK,
    _get_lerobot_dataset,
    lerobot_dataset,
)

_requires = pytest.mark.skipif(not HAS_STRANDS, reason="requires strands-agents SDK")


# ═════════════════════════════════════════════════════════════════════════════
# Action Dispatch Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestActionDispatch:
    """Test top-level action routing."""

    def test_unknown_action(self):
        result = lerobot_dataset(action="bogus")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]
        assert "bogus" in result["content"][0]["text"]

    def test_create_no_repo_id(self):
        result = lerobot_dataset(action="create")
        assert result["status"] == "error"
        assert "repo_id" in result["content"][0]["text"]

    def test_info_no_repo_id(self):
        result = lerobot_dataset(action="info")
        assert result["status"] == "error"

    def test_record_no_repo_id(self):
        result = lerobot_dataset(action="record")
        assert result["status"] == "error"

    def test_push_no_repo_id(self):
        result = lerobot_dataset(action="push")
        assert result["status"] == "error"

    def test_pull_no_repo_id(self):
        result = lerobot_dataset(action="pull")
        assert result["status"] == "error"

    def test_browse_no_repo_id(self):
        result = lerobot_dataset(action="browse")
        assert result["status"] == "error"

    def test_replay_no_repo_id(self):
        result = lerobot_dataset(action="replay")
        assert result["status"] == "error"

    def test_compute_stats_no_repo_id(self):
        result = lerobot_dataset(action="compute_stats")
        assert result["status"] == "error"


# ═════════════════════════════════════════════════════════════════════════════
# Create Action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestCreateAction:
    """Test dataset creation."""

    @patch("strands_robots.tools.lerobot_dataset.LeRobotDataset", create=True)
    def test_create_with_lerobot(self, mock_ds_cls):
        """Create via lerobot when available."""
        mock_ds = MagicMock()
        mock_ds.root = "/tmp/datasets/test"

        # Mock the import path
        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(
                    LeRobotDataset=MagicMock(create=MagicMock(return_value=mock_ds))
                ),
            },
        ):
            result = lerobot_dataset(action="create", repo_id="user/test_data", fps=30)
            assert result["status"] == "success"
            assert "user/test_data" in result["content"][0]["text"]

    def test_create_fallback_no_lerobot(self, tmp_path):
        """Create minimal structure when lerobot not installed."""
        # Compute paths BEFORE entering the mock to avoid pathlib import issues
        root_str = str(tmp_path / "ds")
        info_path = tmp_path / "ds" / "info.json"
        # Patch the import to fail inside the create branch
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def selective_import(name, *args, **kwargs):
            if name in ("lerobot", "lerobot.datasets", "lerobot.datasets.lerobot_dataset"):
                raise ImportError("no lerobot")
            return real_import(name, *args, **kwargs)

        with patch.dict("sys.modules", {"lerobot.datasets.lerobot_dataset": None}):
            with patch("builtins.__import__", side_effect=selective_import):
                # The tool handles ImportError internally
                result = lerobot_dataset(action="create", repo_id="user/test_data", root=root_str)
                # Should succeed with fallback
                assert result["status"] == "success"
                assert "structure created" in result["content"][0]["text"]
                # Verify info.json was created
                assert info_path.exists()
                info = json.loads(info_path.read_text())
                assert info["repo_id"] == "user/test_data"
                assert info["codebase_version"] == "v3.0"


# ═════════════════════════════════════════════════════════════════════════════
# Info Action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestInfoAction:
    """Test dataset info retrieval."""

    def test_info_with_lerobot(self):
        mock_meta = MagicMock()
        mock_meta.total_episodes = 100
        mock_meta.total_frames = 50000
        mock_meta.fps = 30
        mock_meta.features = {"observation.state": "float32[6]", "action": "float32[6]"}
        mock_meta.tasks = ["pick up cube"]

        mock_ds = MagicMock()
        mock_ds.meta = mock_meta

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=MagicMock(return_value=mock_ds)),
            },
        ):
            result = lerobot_dataset(action="info", repo_id="user/my_data")
            assert result["status"] == "success"
            text = result["content"][0]["text"]
            assert "100" in text  # episodes
            assert "50000" in text  # frames
            # Check JSON content
            json_data = result["content"][1]["json"]
            assert json_data["total_episodes"] == 100
            assert json_data["fps"] == 30

    def test_info_fallback_with_info_json(self, tmp_path):
        """Read info.json when lerobot not installed."""
        ds_dir = tmp_path / "user" / "test"
        ds_dir.mkdir(parents=True)
        info = {"repo_id": "user/test", "total_episodes": 5, "fps": 30}
        (ds_dir / "info.json").write_text(json.dumps(info))
        # Compute root path string BEFORE entering mock to avoid pathlib import issues
        root_str = str(tmp_path / "user" / "test")

        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def selective_import(name, *args, **kwargs):
            if name in ("lerobot", "lerobot.datasets", "lerobot.datasets.lerobot_dataset"):
                raise ImportError("no lerobot")
            return real_import(name, *args, **kwargs)

        with patch.dict("sys.modules", {"lerobot.datasets.lerobot_dataset": None}):
            with patch("builtins.__import__", side_effect=selective_import):
                result = lerobot_dataset(action="info", repo_id="user/test", root=root_str)
                assert result["status"] == "success"

    def test_info_fallback_not_found(self, tmp_path):
        # Compute root path string BEFORE entering mock to avoid pathlib import issues
        root_str = str(tmp_path / "nope")
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def selective_import(name, *args, **kwargs):
            if name in ("lerobot", "lerobot.datasets", "lerobot.datasets.lerobot_dataset"):
                raise ImportError("no lerobot")
            return real_import(name, *args, **kwargs)

        with patch.dict("sys.modules", {"lerobot.datasets.lerobot_dataset": None}):
            with patch("builtins.__import__", side_effect=selective_import):
                result = lerobot_dataset(action="info", repo_id="nonexistent/ds", root=root_str)
                assert result["status"] == "error"
                assert "not found" in result["content"][0]["text"].lower()


# ═════════════════════════════════════════════════════════════════════════════
# Record Action Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestRecordAction:
    """Test recording functionality."""

    def setup_method(self):
        """Clean global recording state."""
        with _RECORDING_LOCK:
            _ACTIVE_RECORDINGS.clear()

    def test_record_lerobot_import_error(self):
        """Record fails gracefully when lerobot not installed."""
        with patch.dict("sys.modules", {"lerobot.datasets.lerobot_dataset": None}):
            with patch("builtins.__import__", side_effect=ImportError("no lerobot")):
                result = lerobot_dataset(action="record", repo_id="user/test")
                assert result["status"] == "error"
                assert "lerobot" in result["content"][0]["text"].lower()

    def test_record_creates_session(self):
        """Recording creates entry in _ACTIVE_RECORDINGS then cleans up."""
        mock_ds = MagicMock()
        mock_ds_cls = MagicMock(return_value=mock_ds)
        mock_ds.add_frame = MagicMock()
        mock_ds.save_episode = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=mock_ds_cls),
            },
        ):
            result = lerobot_dataset(
                action="record",
                repo_id="user/test",
                num_episodes=1,
                episode_time_s=0.05,
                fps=10,  # very short for fast test
            )
            assert result["status"] == "success"
            # Verify cleanup
            with _RECORDING_LOCK:
                assert len(_ACTIVE_RECORDINGS) == 0


# ═════════════════════════════════════════════════════════════════════════════
# Stop Recording Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestStopRecording:
    """Test stop_recording action."""

    def setup_method(self):
        with _RECORDING_LOCK:
            _ACTIVE_RECORDINGS.clear()

    def test_stop_no_active(self):
        result = lerobot_dataset(action="stop_recording")
        assert result["status"] == "success"
        assert "No active" in result["content"][0]["text"]

    def test_stop_active_recording(self):
        with _RECORDING_LOCK:
            _ACTIVE_RECORDINGS["rec_123"] = {
                "repo_id": "user/test",
                "episodes_recorded": 3,
            }
        result = lerobot_dataset(action="stop_recording")
        assert result["status"] == "success"
        assert "Stopped" in result["content"][0]["text"]
        with _RECORDING_LOCK:
            assert len(_ACTIVE_RECORDINGS) == 0


# ═════════════════════════════════════════════════════════════════════════════
# Push / Pull Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestPushPull:
    """Test push and pull actions."""

    def test_push(self):
        mock_ds = MagicMock()
        mock_ds.push_to_hub = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=MagicMock(return_value=mock_ds)),
            },
        ):
            result = lerobot_dataset(action="push", repo_id="user/test")
            assert result["status"] == "success"
            mock_ds.push_to_hub.assert_called_once()

    def test_pull(self):
        mock_meta = MagicMock()
        mock_meta.total_episodes = 10
        mock_meta.total_frames = 5000
        mock_ds = MagicMock()
        mock_ds.meta = mock_meta
        mock_ds.root = "/tmp/datasets"

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=MagicMock(return_value=mock_ds)),
            },
        ):
            result = lerobot_dataset(action="pull", repo_id="user/test")
            assert result["status"] == "success"
            assert "10" in result["content"][0]["text"]


# ═════════════════════════════════════════════════════════════════════════════
# Browse Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestBrowse:
    """Test browse action."""

    def test_browse(self):
        mock_meta = MagicMock()
        mock_meta.total_episodes = 3
        mock_meta.episodes = [
            {"length": 100, "task": "pick"},
            {"length": 200, "task": "place"},
            {"length": 150, "task": "wipe"},
        ]
        mock_ds = MagicMock()
        mock_ds.meta = mock_meta

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=MagicMock(return_value=mock_ds)),
            },
        ):
            result = lerobot_dataset(action="browse", repo_id="user/test", max_episodes=3)
            assert result["status"] == "success"
            assert "3 episodes" in result["content"][0]["text"]
            episodes = result["content"][1]["json"]["episodes"]
            assert len(episodes) == 3


# ═════════════════════════════════════════════════════════════════════════════
# Replay Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestReplay:
    """Test replay action."""

    def test_replay_success(self):
        pytest.importorskip("torch", reason="torch required for replay test")
        mock_meta = MagicMock()
        mock_meta.total_episodes = 1
        mock_meta.episodes = [{"length": 5}]
        mock_ds = MagicMock()
        mock_ds.meta = mock_meta
        # Simulate frame access
        mock_ds.__getitem__ = MagicMock(
            return_value={
                "action.joint_0": MagicMock(tolist=MagicMock(return_value=0.5)),
                "observation.state": MagicMock(tolist=MagicMock(return_value=[0.1, 0.2])),
            }
        )

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=MagicMock(return_value=mock_ds)),
                "lerobot.datasets.utils": MagicMock(),
            },
        ):
            result = lerobot_dataset(action="replay", repo_id="user/test", episode=0)
            assert result["status"] == "success"
            assert result["content"][1]["json"]["episode"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# Compute Stats Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestComputeStats:
    """Test compute_stats action."""

    def test_compute_stats_success(self):
        mock_meta = MagicMock()
        mock_meta.stats = {"observation.state.mean": [0.1, 0.2]}
        mock_ds = MagicMock()
        mock_ds.meta = mock_meta
        mock_ds.consolidate = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=MagicMock(return_value=mock_ds)),
            },
        ):
            result = lerobot_dataset(action="compute_stats", repo_id="user/test")
            assert result["status"] == "success"
            mock_ds.consolidate.assert_called_once()

    def test_compute_stats_error(self):
        mock_ds = MagicMock()
        mock_ds.consolidate.side_effect = RuntimeError("Failed")

        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=MagicMock(return_value=mock_ds)),
            },
        ):
            result = lerobot_dataset(action="compute_stats", repo_id="user/test")
            assert result["status"] == "error"


# ═════════════════════════════════════════════════════════════════════════════
# List Hub Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestListHub:
    """Test list_hub action."""

    def test_list_hub_success(self):
        mock_ds1 = MagicMock()
        mock_ds1.id = "lerobot/aloha_sim"
        mock_ds1.downloads = 500

        mock_ds2 = MagicMock()
        mock_ds2.id = "lerobot/pusht"
        mock_ds2.downloads = 300

        mock_api = MagicMock()
        mock_api.list_datasets.return_value = [mock_ds1, mock_ds2]

        with patch.dict(
            "sys.modules",
            {
                "huggingface_hub": MagicMock(HfApi=MagicMock(return_value=mock_api)),
            },
        ):
            result = lerobot_dataset(action="list_hub")
            assert result["status"] == "success"
            assert "lerobot/aloha_sim" in result["content"][0]["text"]

    def test_list_hub_error(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with patch("builtins.__import__", side_effect=ImportError("no hf_hub")):
                result = lerobot_dataset(action="list_hub")
                assert result["status"] == "error"


# ═════════════════════════════════════════════════════════════════════════════
# _get_lerobot_dataset Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestGetLerobotDataset:
    """Test helper function."""

    def test_import_error(self):
        with patch.dict("sys.modules", {"lerobot.datasets.lerobot_dataset": None}):
            with patch("builtins.__import__", side_effect=ImportError("no lerobot")):
                with pytest.raises(ImportError):
                    _get_lerobot_dataset("user/test")

    def test_create_mode(self):
        mock_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=mock_cls),
            },
        ):
            _get_lerobot_dataset("user/test", create=True)
            mock_cls.create.assert_called_once()

    def test_load_mode(self):
        mock_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=mock_cls),
            },
        ):
            _get_lerobot_dataset("user/test", create=False)
            mock_cls.assert_called_once()
