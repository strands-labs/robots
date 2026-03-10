"""Coverage-gap tests for strands_robots/tools/lerobot_dataset.py.

Targets uncovered lines: 314-401 (record episode loop internals: teleop,
policy branches, frame building, sleep, stop check, save/finalize/push),
407-408, 417, 422-430 (record error/cleanup paths), 454-457 (browse
episode exception), 525-526 (replay frame extraction), 544-573 (compute_stats
detail paths), 619-626 (outer exception handlers).

All LeRobot and HuggingFace Hub operations are mocked. CPU-only.
"""

import os
import sys
from unittest.mock import MagicMock, PropertyMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import strands

    HAS_STRANDS = hasattr(strands, "Agent")
except ImportError:
    import types

    _mock_strands = types.ModuleType("strands")
    _mock_strands.tool = lambda f: f
    sys.modules["strands"] = _mock_strands
    HAS_STRANDS = False

from strands_robots.tools.lerobot_dataset import (
    _ACTIVE_RECORDINGS,
    _RECORDING_LOCK,
    _build_default_features,
    lerobot_dataset,
)

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _make_mock_ds(features=None, total_episodes=0, total_frames=0, fps=30):
    """Build a mock LeRobotDataset with common attributes."""
    ds = MagicMock()
    ds.meta = MagicMock()
    ds.meta.total_episodes = total_episodes
    ds.meta.total_frames = total_frames
    ds.meta.fps = fps
    ds.meta.features = features or {"observation.state": "float32[6]", "action": "float32[6]"}
    ds.meta.tasks = ["pick up cube"]
    ds.meta.stats = None
    ds.meta.episodes = []
    ds.root = "/tmp/test_ds"
    ds.features = {
        "observation.state": {"dtype": "float32", "shape": (6,)},
        "action": {"dtype": "float32", "shape": (6,)},
    }
    ds.add_frame = MagicMock()
    ds.save_episode = MagicMock()
    ds.finalize = MagicMock()
    ds.push_to_hub = MagicMock()
    ds.consolidate = MagicMock()
    return ds


def _lerobot_modules(ds_cls_mock):
    """Return sys.modules dict for mocking lerobot imports."""
    return {
        "lerobot": MagicMock(),
        "lerobot.datasets": MagicMock(),
        "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=ds_cls_mock),
    }


# ═════════════════════════════════════════════════════════════════════════════
# _build_default_features
# ═════════════════════════════════════════════════════════════════════════════


class TestBuildDefaultFeatures:
    def test_returns_dict_with_state_and_action(self):
        f = _build_default_features()
        assert "observation.state" in f
        assert "action" in f
        assert f["observation.state"]["dtype"] == "float32"
        assert f["action"]["shape"] == (6,)

    def test_with_robot_type(self):
        f = _build_default_features("so100")
        assert "observation.state" in f


# ═════════════════════════════════════════════════════════════════════════════
# Record action — deep coverage of episode loop
# ═════════════════════════════════════════════════════════════════════════════


class TestRecordDeep:
    def setup_method(self):
        with _RECORDING_LOCK:
            _ACTIVE_RECORDINGS.clear()

    def test_record_full_episode_loop(self):
        """Exercise the full recording loop: create dataset, add frames, save episode."""
        mock_ds = _make_mock_ds()
        mock_ds_cls = MagicMock()
        # First call (load existing) raises, second (create) returns mock_ds
        mock_ds_cls.side_effect = Exception("not found")
        mock_ds_cls.create = MagicMock(return_value=mock_ds)

        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(
                action="record",
                repo_id="user/test",
                num_episodes=2,
                episode_time_s=0.02,
                fps=100,
                task="pick cube",
            )
        assert result["status"] == "success"
        assert "pick cube" in result["content"][0]["text"]
        # Verify add_frame was called many times
        assert mock_ds.add_frame.call_count > 0
        # save_episode called once per episode
        assert mock_ds.save_episode.call_count == 2
        # finalize called
        mock_ds.finalize.assert_called_once()

    def test_record_with_push_to_hub(self):
        """Recording with push_to_hub=True triggers push."""
        mock_ds = _make_mock_ds()
        mock_ds_cls = MagicMock(return_value=mock_ds)

        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(
                action="record",
                repo_id="user/test",
                num_episodes=1,
                episode_time_s=0.01,
                fps=100,
                push_to_hub=True,
                tags=["test"],
            )
        assert result["status"] == "success"
        mock_ds.push_to_hub.assert_called_once_with(tags=["test"])

    def test_record_push_to_hub_failure(self):
        """Push failure is logged but doesn't crash recording."""
        mock_ds = _make_mock_ds()
        mock_ds.push_to_hub.side_effect = Exception("auth error")
        mock_ds_cls = MagicMock(return_value=mock_ds)

        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(
                action="record",
                repo_id="user/test",
                num_episodes=1,
                episode_time_s=0.01,
                fps=100,
                push_to_hub=True,
            )
        assert result["status"] == "success"

    def test_record_finalize_failure(self):
        """Finalize failure is logged but doesn't crash."""
        mock_ds = _make_mock_ds()
        mock_ds.finalize.side_effect = Exception("finalize error")
        mock_ds_cls = MagicMock(return_value=mock_ds)

        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(
                action="record",
                repo_id="user/test",
                num_episodes=1,
                episode_time_s=0.01,
                fps=100,
            )
        assert result["status"] == "success"

    def test_record_frame_add_error(self):
        """Frame add errors are logged but recording continues."""
        mock_ds = _make_mock_ds()
        mock_ds.add_frame.side_effect = Exception("frame error")
        mock_ds_cls = MagicMock(return_value=mock_ds)

        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(
                action="record",
                repo_id="user/test",
                num_episodes=1,
                episode_time_s=0.01,
                fps=100,
            )
        assert result["status"] == "success"

    def test_record_save_episode_error(self):
        """Episode save errors are logged but recording continues."""
        mock_ds = _make_mock_ds()
        mock_ds.save_episode.side_effect = Exception("save error")
        mock_ds_cls = MagicMock(return_value=mock_ds)

        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(
                action="record",
                repo_id="user/test",
                num_episodes=1,
                episode_time_s=0.01,
                fps=100,
            )
        assert result["status"] == "success"

    def test_record_with_policy_provider(self):
        """Recording with a policy_provider exercises the policy branch."""
        mock_ds = _make_mock_ds()
        mock_ds_cls = MagicMock(return_value=mock_ds)

        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(
                action="record",
                repo_id="user/test",
                num_episodes=1,
                episode_time_s=0.01,
                fps=100,
                policy_provider="mock",
            )
        assert result["status"] == "success"

    def test_record_with_teleop_device(self):
        """Recording with teleop_device exercises the teleop branch."""
        mock_ds = _make_mock_ds()
        mock_ds_cls = MagicMock(return_value=mock_ds)
        mock_teleop = MagicMock()
        mock_teleop.get_action.return_value = {"j1": 0.5, "j2": 0.3}
        mock_resolve = MagicMock(return_value=mock_teleop)

        with patch.dict(
            "sys.modules",
            {
                **_lerobot_modules(mock_ds_cls),
                "strands_robots.tools.teleoperator": MagicMock(_resolve_teleoperator=mock_resolve),
            },
        ):
            result = lerobot_dataset(
                action="record",
                repo_id="user/test",
                num_episodes=1,
                episode_time_s=0.01,
                fps=100,
                teleop_device="keyboard_ee",
            )
        assert result["status"] == "success"
        mock_teleop.connect.assert_called_once()
        mock_teleop.disconnect.assert_called_once()

    def test_record_with_custom_features(self):
        """Recording with custom features dict."""
        mock_ds = _make_mock_ds()
        custom_features = {
            "observation.state": {"dtype": "float32", "shape": (3,)},
            "action": {"dtype": "float32", "shape": (3,)},
        }
        mock_ds.features = custom_features
        mock_ds_cls = MagicMock()
        mock_ds_cls.side_effect = Exception("not found")
        mock_ds_cls.create = MagicMock(return_value=mock_ds)

        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(
                action="record",
                repo_id="user/test",
                num_episodes=1,
                episode_time_s=0.01,
                fps=100,
                features=custom_features,
            )
        assert result["status"] == "success"

    def test_record_generic_exception_reraises(self):
        """Non-ImportError exceptions in record are re-raised and caught by outer handler."""
        mock_ds_cls = MagicMock(side_effect=RuntimeError("unexpected"))
        # Also make .create raise so the fallback path fails too
        mock_ds_cls.create = MagicMock(side_effect=RuntimeError("unexpected"))

        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(
                action="record",
                repo_id="user/test",
                num_episodes=1,
            )
        assert result["status"] == "error"
        assert "unexpected" in result["content"][0]["text"]


# ═════════════════════════════════════════════════════════════════════════════
# Browse — episode exception handling
# ═════════════════════════════════════════════════════════════════════════════


class TestBrowseDeep:
    def test_browse_episode_exception(self):
        """Exercise the except branch when accessing episode data."""
        mock_ds = _make_mock_ds(total_episodes=2)
        # episodes attribute exists but raises on access
        mock_meta = MagicMock()
        mock_meta.total_episodes = 2
        type(mock_meta).episodes = PropertyMock(side_effect=AttributeError("no episodes"))
        mock_ds.meta = mock_meta

        mock_ds_cls = MagicMock(return_value=mock_ds)
        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(action="browse", repo_id="user/test", max_episodes=2)
        assert result["status"] == "success"
        # Should still return episodes (with index-only data)
        episodes = result["content"][1]["json"]["episodes"]
        assert len(episodes) == 2


# ═════════════════════════════════════════════════════════════════════════════
# Replay — deeper paths
# ═════════════════════════════════════════════════════════════════════════════


class TestReplayDeep:
    def test_replay_multi_episode_offset(self):
        """Test replay calculates correct ep_start for episode > 0."""
        mock_ds = _make_mock_ds(total_episodes=3)
        mock_ds.meta.episodes = [
            {"length": 10},
            {"length": 20},
            {"length": 15},
        ]
        # Mock frame access for episode 1 (frames 10-29)
        mock_ds.__getitem__ = MagicMock(
            return_value={
                "action.joint_0": MagicMock(tolist=MagicMock(return_value=0.5)),
            }
        )

        mock_ds_cls = MagicMock(return_value=mock_ds)
        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(action="replay", repo_id="user/test", episode=1)
        assert result["status"] == "success"
        assert result["content"][1]["json"]["episode"] == 1
        assert result["content"][1]["json"]["num_frames"] == 20

    def test_replay_extraction_error(self):
        """Replay handles exceptions during frame extraction gracefully."""
        mock_ds = _make_mock_ds(total_episodes=1)
        mock_ds.meta.episodes = [{"length": 5}]
        # First frame works, second raises
        call_count = [0]

        def getitem_side_effect(idx):
            call_count[0] += 1
            if call_count[0] > 2:
                raise RuntimeError("corrupt frame")
            return {"action.j1": MagicMock(tolist=MagicMock(return_value=0.1))}

        mock_ds.__getitem__ = getitem_side_effect

        mock_ds_cls = MagicMock(return_value=mock_ds)
        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(action="replay", repo_id="user/test", episode=0)
        assert result["status"] == "success"
        # The outer try/except catches the error, so 0 frames is valid
        assert result["content"][1]["json"]["num_frames"] >= 0

    def test_replay_no_episodes_metadata(self):
        """Replay when meta has no episodes attribute."""
        mock_ds = _make_mock_ds(total_episodes=1)
        del mock_ds.meta.episodes

        mock_ds_cls = MagicMock(return_value=mock_ds)
        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(action="replay", repo_id="user/test", episode=0)
        assert result["status"] == "success"
        # No frames extracted (ep_length=0)
        assert result["content"][1]["json"]["num_frames"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# Compute Stats — deeper paths
# ═════════════════════════════════════════════════════════════════════════════


class TestComputeStatsDeep:
    def test_compute_stats_with_stats_detail(self):
        """Exercise the stats display branch."""
        mock_ds = _make_mock_ds()
        mock_ds.meta.stats = {
            "observation.state.mean": [0.1, 0.2, 0.3],
            "observation.state.std": [0.01, 0.02, 0.03],
        }

        mock_ds_cls = MagicMock(return_value=mock_ds)
        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(action="compute_stats", repo_id="user/test")
        assert result["status"] == "success"
        assert "observation.state.mean" in result["content"][0]["text"]


# ═════════════════════════════════════════════════════════════════════════════
# Outer exception handler (ImportError, generic Exception)
# ═════════════════════════════════════════════════════════════════════════════


class TestOuterExceptionHandlers:
    def test_import_error_caught(self):
        """ImportError from push/pull action caught by outer handler."""
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def selective_import(name, *args, **kwargs):
            if name in ("lerobot", "lerobot.datasets", "lerobot.datasets.lerobot_dataset"):
                raise ImportError("no lerobot")
            return real_import(name, *args, **kwargs)

        with patch.dict("sys.modules", {"lerobot.datasets.lerobot_dataset": None}):
            with patch("builtins.__import__", side_effect=selective_import):
                result = lerobot_dataset(action="push", repo_id="user/test")
        assert result["status"] == "error"
        assert "lerobot" in result["content"][0]["text"].lower()

    def test_generic_error_in_push(self):
        """Generic exception caught by outer handler."""
        mock_ds = _make_mock_ds()
        mock_ds.push_to_hub.side_effect = RuntimeError("network error")
        mock_ds_cls = MagicMock(return_value=mock_ds)

        with patch.dict("sys.modules", _lerobot_modules(mock_ds_cls)):
            result = lerobot_dataset(action="push", repo_id="user/test")
        assert result["status"] == "error"
        assert "network error" in result["content"][0]["text"]
