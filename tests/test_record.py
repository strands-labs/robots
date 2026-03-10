#!/usr/bin/env python3
"""Comprehensive tests for strands_robots.record — Native Recording Pipeline.

Tests RecordSession, EpisodeStats, RecordMode using mocked robot/teleop/policy.
Target: 14% → 80%+ coverage (243 uncovered statements).
"""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strands_robots.record import EpisodeStats, RecordMode, RecordSession

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_robot():
    """Create a mock LeRobot Robot."""
    robot = MagicMock()
    robot.is_connected = False
    robot.name = "mock_robot"
    robot.robot_type = "so100"
    robot.observation_features = {
        "shoulder_pan": float,
        "shoulder_lift": float,
        "elbow_flex": float,
        "wrist_cam": (480, 640, 3),
    }
    robot.action_features = {
        "shoulder_pan": float,
        "shoulder_lift": float,
        "elbow_flex": float,
    }
    robot.get_observation.return_value = {
        "shoulder_pan": 0.5,
        "shoulder_lift": -0.3,
        "elbow_flex": 1.2,
        "wrist_cam": np.zeros((480, 640, 3), dtype=np.uint8),
    }
    robot.send_action.return_value = None
    return robot


@pytest.fixture
def mock_teleop():
    """Create a mock LeRobot Teleoperator."""
    teleop = MagicMock()
    teleop.is_connected = False
    teleop.get_action.return_value = {
        "shoulder_pan": 0.1,
        "shoulder_lift": 0.2,
        "elbow_flex": 0.3,
    }
    return teleop


@pytest.fixture
def mock_policy():
    """Create a mock strands_robots Policy."""
    policy = MagicMock()

    async def fake_get_actions(obs, task):
        return [{"shoulder_pan": 0.1, "shoulder_lift": 0.2, "elbow_flex": 0.3}]

    policy.get_actions = fake_get_actions
    return policy


@pytest.fixture
def mock_dataset():
    """Create a mock LeRobotDataset."""
    dataset = MagicMock()
    dataset.root = "/tmp/test_dataset"
    dataset.add_frame = MagicMock()
    dataset.save_episode = MagicMock()
    dataset.consolidate = MagicMock()
    dataset.push_to_hub = MagicMock()
    dataset.clear_episode_buffer = MagicMock()
    return dataset


def _make_session(robot, teleop=None, policy=None, **kwargs):
    """Helper to create a RecordSession with defaults."""
    defaults = dict(
        repo_id="test/recording",
        task="pick up cube",
        fps=30,
        episode_time_s=0.1,  # Short for fast tests
        num_episodes=2,
    )
    defaults.update(kwargs)
    return RecordSession(
        robot=robot,
        teleop=teleop,
        policy=policy,
        **defaults,
    )


# ── RecordMode Tests ─────────────────────────────────────────────────────────


class TestRecordMode:
    def test_enum_values(self):
        assert RecordMode.TELEOP.value == "teleop"
        assert RecordMode.POLICY.value == "policy"
        assert RecordMode.IDLE.value == "idle"

    def test_from_string(self):
        assert RecordMode("teleop") == RecordMode.TELEOP
        assert RecordMode("policy") == RecordMode.POLICY
        assert RecordMode("idle") == RecordMode.IDLE


# ── EpisodeStats Tests ───────────────────────────────────────────────────────


class TestEpisodeStats:
    def test_defaults(self):
        stats = EpisodeStats()
        assert stats.index == 0
        assert stats.frames == 0
        assert stats.duration_s == 0.0
        assert stats.task == ""
        assert stats.discarded is False

    def test_custom_values(self):
        stats = EpisodeStats(index=5, frames=300, duration_s=10.0, task="grasp", discarded=True)
        assert stats.index == 5
        assert stats.frames == 300
        assert stats.duration_s == 10.0
        assert stats.task == "grasp"
        assert stats.discarded is True


# ── RecordSession Init Tests ─────────────────────────────────────────────────


class TestRecordSessionInit:
    def test_basic_init(self, mock_robot):
        session = _make_session(mock_robot)
        assert session.robot is mock_robot
        assert session.teleop is None
        assert session.policy is None
        assert session.repo_id == "test/recording"
        assert session.default_task == "pick up cube"
        assert session.fps == 30
        assert session._connected is False
        assert session._recording is False
        assert session._episodes == []

    def test_init_with_teleop(self, mock_robot, mock_teleop):
        session = _make_session(mock_robot, teleop=mock_teleop)
        assert session.teleop is mock_teleop

    def test_init_with_policy(self, mock_robot, mock_policy):
        session = _make_session(mock_robot, policy=mock_policy)
        assert session.policy is mock_policy

    def test_init_all_params(self, mock_robot, mock_teleop):
        session = RecordSession(
            robot=mock_robot,
            teleop=mock_teleop,
            repo_id="user/data",
            task="stack blocks",
            fps=60,
            root="/tmp/data",
            use_videos=False,
            vcodec="h264",
            streaming_encoding=False,
            image_writer_threads=8,
            push_to_hub=True,
            num_episodes=100,
            episode_time_s=120.0,
            reset_time_s=5.0,
            use_processor=False,
            visualize="",
        )
        assert session.fps == 60
        assert session.push_to_hub is True
        assert session.use_videos is False
        assert session.num_episodes == 100

    def test_init_with_visualize_no_visualizer(self, mock_robot):
        """With visualize set but no actual visualizer available, _visualizer may be set."""
        _make_session(mock_robot, visualize="terminal")
        # The visualizer is optional — may or may not be created


# ── RecordSession Connect Tests ──────────────────────────────────────────────


class TestRecordSessionConnect:
    @patch("strands_robots.record.RecordSession._create_dataset")
    def test_connect_robot_not_connected(self, mock_create, mock_robot):
        mock_robot.is_connected = False
        session = _make_session(mock_robot, use_processor=False)
        session.connect()
        mock_robot.connect.assert_called_once()
        assert session._connected is True

    @patch("strands_robots.record.RecordSession._create_dataset")
    def test_connect_robot_already_connected(self, mock_create, mock_robot):
        mock_robot.is_connected = True
        session = _make_session(mock_robot, use_processor=False)
        session.connect()
        mock_robot.connect.assert_not_called()
        assert session._connected is True

    @patch("strands_robots.record.RecordSession._create_dataset")
    def test_connect_with_teleop(self, mock_create, mock_robot, mock_teleop):
        mock_robot.is_connected = False
        mock_teleop.is_connected = False
        session = _make_session(mock_robot, teleop=mock_teleop, use_processor=False)
        session.connect()
        mock_teleop.connect.assert_called_once()

    @patch("strands_robots.record.RecordSession._create_dataset")
    def test_connect_teleop_already_connected(self, mock_create, mock_robot, mock_teleop):
        mock_teleop.is_connected = True
        session = _make_session(mock_robot, teleop=mock_teleop, use_processor=False)
        session.connect()
        mock_teleop.connect.assert_not_called()

    @patch("strands_robots.record.RecordSession._create_dataset")
    def test_connect_idempotent(self, mock_create, mock_robot):
        session = _make_session(mock_robot, use_processor=False)
        session.connect()
        session.connect()  # Second call should be no-op
        assert mock_create.call_count == 1

    @patch("strands_robots.record.RecordSession._create_dataset")
    def test_connect_initializes_processors(self, mock_create, mock_robot):
        """When use_processor=True, _init_processors should be called."""
        session = _make_session(mock_robot, use_processor=True)
        with patch.object(session, "_init_processors") as mock_init:
            session.connect()
            mock_init.assert_called_once()


# ── _init_processors Tests ───────────────────────────────────────────────────


class TestInitProcessors:
    def test_processors_import_error(self, mock_robot):
        session = _make_session(mock_robot)
        # Simulate lerobot.processor not installed
        with patch.dict(sys.modules, {"lerobot.processor": None}):
            session._init_processors()
        assert session._action_processor is None
        assert session._observation_processor is None

    def test_processors_exception(self, mock_robot):
        session = _make_session(mock_robot)
        mock_module = MagicMock()
        mock_module.make_default_robot_action_processor.side_effect = RuntimeError("boom")
        with patch.dict(sys.modules, {"lerobot.processor": mock_module}):
            session._init_processors()
        # Should gracefully handle the error

    def test_processors_success(self, mock_robot):
        session = _make_session(mock_robot)
        mock_module = MagicMock()
        mock_action_proc = MagicMock()
        mock_obs_proc = MagicMock()
        mock_module.make_default_robot_action_processor.return_value = mock_action_proc
        mock_module.make_default_robot_observation_processor.return_value = mock_obs_proc
        with patch.dict(sys.modules, {"lerobot.processor": mock_module}):
            session._init_processors()
        assert session._action_processor is mock_action_proc
        assert session._observation_processor is mock_obs_proc


# ── _create_dataset Tests ────────────────────────────────────────────────────


class TestCreateDataset:
    def test_creates_dataset_with_features(self, mock_robot, mock_dataset):
        session = _make_session(mock_robot, use_processor=False)

        mock_lerobot_dataset_cls = MagicMock()
        mock_lerobot_dataset_cls.create.return_value = mock_dataset

        with patch.dict(
            sys.modules,
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=mock_lerobot_dataset_cls),
            },
        ):
            session._create_dataset()

        mock_lerobot_dataset_cls.create.assert_called_once()
        call_kwargs = mock_lerobot_dataset_cls.create.call_args[1]
        assert call_kwargs["repo_id"] == "test/recording"
        assert call_kwargs["fps"] == 30
        assert "observation.images.wrist_cam" in call_kwargs["features"]
        assert "observation.state" in call_kwargs["features"]
        assert "action" in call_kwargs["features"]
        assert session._dataset is mock_dataset

    def test_create_dataset_exception(self, mock_robot):
        session = _make_session(mock_robot, use_processor=False)

        mock_lerobot_dataset_cls = MagicMock()
        mock_lerobot_dataset_cls.create.side_effect = RuntimeError("disk full")

        with patch.dict(
            sys.modules,
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=mock_lerobot_dataset_cls),
            },
        ):
            with pytest.raises(RuntimeError, match="disk full"):
                session._create_dataset()

    def test_create_dataset_no_observation_features(self, mock_robot, mock_dataset):
        """Robot without observation_features should still work."""
        del mock_robot.observation_features
        del mock_robot.action_features
        session = _make_session(mock_robot, use_processor=False)

        mock_lerobot_dataset_cls = MagicMock()
        mock_lerobot_dataset_cls.create.return_value = mock_dataset

        with patch.dict(
            sys.modules,
            {
                "lerobot": MagicMock(),
                "lerobot.datasets": MagicMock(),
                "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=mock_lerobot_dataset_cls),
            },
        ):
            session._create_dataset()
        assert session._dataset is mock_dataset


# ── Record Episode Tests ─────────────────────────────────────────────────────


class TestRecordEpisode:
    def _setup_session(self, mock_robot, mock_teleop=None, mock_policy=None, **kwargs):
        """Create a connected session with mocked dataset."""
        session = _make_session(mock_robot, teleop=mock_teleop, policy=mock_policy, episode_time_s=0.05, **kwargs)
        session._connected = True
        session._dataset = MagicMock()
        session._dataset.add_frame = MagicMock()
        session._dataset.save_episode = MagicMock()
        return session

    def test_teleop_mode_autodetect(self, mock_robot, mock_teleop):
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        stats = session.record_episode()
        assert stats.frames > 0
        assert stats.duration_s > 0
        assert stats.task == "pick up cube"
        mock_teleop.get_action.assert_called()
        mock_robot.send_action.assert_called()
        session._dataset.save_episode.assert_called_once()

    def test_policy_mode_autodetect(self, mock_robot, mock_policy):
        session = self._setup_session(mock_robot, mock_policy=mock_policy)
        stats = session.record_episode()
        assert stats.frames > 0
        session._dataset.save_episode.assert_called_once()

    def test_idle_mode(self, mock_robot):
        """No teleop, no policy → idle mode (just observes)."""
        session = self._setup_session(mock_robot)
        stats = session.record_episode()
        assert stats.frames > 0
        mock_robot.get_observation.assert_called()
        mock_robot.send_action.assert_not_called()

    def test_explicit_mode(self, mock_robot, mock_teleop):
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        stats = session.record_episode(mode=RecordMode.TELEOP)
        assert stats.frames > 0

    def test_custom_task(self, mock_robot, mock_teleop):
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        stats = session.record_episode(task="stack blocks")
        assert stats.task == "stack blocks"

    def test_custom_duration(self, mock_robot, mock_teleop):
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        stats = session.record_episode(duration=0.02)
        assert stats.duration_s < 1.0

    def test_on_frame_callback(self, mock_robot, mock_teleop):
        callback = MagicMock()
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        stats = session.record_episode(on_frame=callback)
        assert callback.call_count == stats.frames

    def test_on_frame_callback_exception(self, mock_robot, mock_teleop):
        """Callback exception should not crash recording."""
        callback = MagicMock(side_effect=RuntimeError("callback failed"))
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        stats = session.record_episode(on_frame=callback)
        assert stats.frames > 0  # Still recorded

    def test_connect_if_not_connected(self, mock_robot, mock_teleop, mock_dataset):
        """record_episode should auto-connect."""
        session = _make_session(mock_robot, teleop=mock_teleop, episode_time_s=0.05)
        with patch.object(session, "connect") as mock_connect:
            mock_connect.side_effect = lambda: setattr(session, "_connected", True)
            session._dataset = mock_dataset
            session.record_episode()
            mock_connect.assert_called_once()

    def test_dataset_add_frame_exception(self, mock_robot, mock_teleop):
        """Frame write failure should be logged but not crash."""
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        session._dataset.add_frame.side_effect = RuntimeError("write error")
        stats = session.record_episode()
        assert stats.frames > 0

    def test_dataset_save_episode_failure(self, mock_robot, mock_teleop):
        """Save failure should mark episode as discarded."""
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        session._dataset.save_episode.side_effect = RuntimeError("save error")
        stats = session.record_episode()
        assert stats.discarded is True

    def test_observation_processor(self, mock_robot, mock_teleop):
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        session._observation_processor = MagicMock(return_value={"processed": True})
        session.record_episode()
        session._observation_processor.assert_called()

    def test_observation_processor_exception(self, mock_robot, mock_teleop):
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        session._observation_processor = MagicMock(side_effect=RuntimeError("proc fail"))
        stats = session.record_episode()
        assert stats.frames > 0  # Falls back to raw obs

    def test_action_processor_teleop(self, mock_robot, mock_teleop):
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        session._action_processor = MagicMock(return_value={"processed_action": 1.0})
        session.record_episode(mode=RecordMode.TELEOP)
        session._action_processor.assert_called()

    def test_action_processor_exception(self, mock_robot, mock_teleop):
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        session._action_processor = MagicMock(side_effect=RuntimeError("proc fail"))
        stats = session.record_episode(mode=RecordMode.TELEOP)
        assert stats.frames > 0  # Falls back to raw action

    def test_no_dataset(self, mock_robot, mock_teleop):
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        session._dataset = None
        stats = session.record_episode()
        assert stats.frames > 0

    def test_episodes_accumulate(self, mock_robot, mock_teleop):
        session = self._setup_session(mock_robot, mock_teleop=mock_teleop)
        s1 = session.record_episode()
        s2 = session.record_episode()
        assert s1.index == 0
        assert s2.index == 1
        assert len(session._episodes) == 2

    def test_stop_flag(self, mock_robot, mock_teleop):
        """Setting _stop_flag should end recording early."""
        session = _make_session(mock_robot, teleop=mock_teleop, episode_time_s=60.0)
        session._connected = True
        session._dataset = MagicMock()
        # Set stop flag after a brief delay — use 0.3s for CI robustness
        import threading

        def stop_later():
            time.sleep(0.3)
            session._stop_flag = True

        t = threading.Thread(target=stop_later)
        t.start()
        stats = session.record_episode()
        t.join()
        assert stats.duration_s < 30.0


# ── _build_dataset_frame Tests ───────────────────────────────────────────────


class TestBuildDatasetFrame:
    @pytest.fixture
    def session_with_torch(self, mock_robot):
        """Session that can import torch."""
        pytest.importorskip("torch")
        session = _make_session(mock_robot, use_processor=False)
        session._connected = True
        return session

    def test_build_frame_with_images_and_state(self, session_with_torch, mock_robot):
        import torch

        obs = {
            "shoulder_pan": 0.5,
            "shoulder_lift": -0.3,
            "elbow_flex": 1.2,
            "wrist_cam": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        action = {"shoulder_pan": 0.1, "shoulder_lift": 0.2, "elbow_flex": 0.3}
        frame = session_with_torch._build_dataset_frame(obs, action, "pick up cube")
        assert "observation.images.wrist_cam" in frame
        assert "observation.state" in frame
        assert "action" in frame
        assert isinstance(frame["observation.images.wrist_cam"], torch.Tensor)
        assert frame["observation.state"].shape == (3,)
        assert frame["action"].shape == (3,)

    def test_build_frame_with_torch_tensor_image(self, session_with_torch, mock_robot):
        import torch

        obs = {
            "shoulder_pan": 0.5,
            "shoulder_lift": -0.3,
            "elbow_flex": 1.2,
            "wrist_cam": torch.zeros(480, 640, 3, dtype=torch.uint8),
        }
        frame = session_with_torch._build_dataset_frame(obs, {}, "task")
        assert "observation.images.wrist_cam" in frame

    def test_build_frame_empty_action(self, session_with_torch, mock_robot):
        obs = {"shoulder_pan": 0.5, "shoulder_lift": -0.3, "elbow_flex": 1.2}
        frame = session_with_torch._build_dataset_frame(obs, {}, "task")
        assert "action" not in frame

    def test_build_frame_list_action(self, session_with_torch, mock_robot):
        obs = {"shoulder_pan": 0.5, "shoulder_lift": -0.3, "elbow_flex": 1.2}
        frame = session_with_torch._build_dataset_frame(obs, [0.1, 0.2, 0.3], "task")
        assert "action" in frame
        assert frame["action"].shape == (3,)

    def test_build_frame_numpy_action(self, session_with_torch, mock_robot):
        obs = {"shoulder_pan": 0.5, "shoulder_lift": -0.3, "elbow_flex": 1.2}
        frame = session_with_torch._build_dataset_frame(obs, np.array([0.1, 0.2]), "task")
        assert "action" in frame

    def test_build_frame_torch_tensor_action(self, session_with_torch, mock_robot):
        import torch

        obs = {"shoulder_pan": 0.5, "shoulder_lift": -0.3, "elbow_flex": 1.2}
        frame = session_with_torch._build_dataset_frame(obs, torch.tensor([0.1, 0.2]), "task")
        assert "action" in frame

    def test_build_frame_no_observation_features(self, session_with_torch, mock_robot):
        """Robot without observation_features should use fallback logic."""
        del mock_robot.observation_features
        # But add config with cameras
        mock_robot.config = MagicMock()
        mock_robot.config.cameras = {"wrist_cam": {}}
        obs = {
            "wrist_cam": np.zeros((480, 640, 3), dtype=np.uint8),
            "motor_pos": 1.5,
        }
        frame = session_with_torch._build_dataset_frame(obs, {}, "task")
        assert "observation.images.wrist_cam" in frame
        assert "observation.state" in frame


# ── Discard Episode Tests ────────────────────────────────────────────────────


class TestDiscardEpisode:
    def test_discard_current_episode(self, mock_robot):
        session = _make_session(mock_robot)
        session._current_episode = EpisodeStats(index=0, task="test")
        session.discard_episode()
        assert session._stop_flag is True
        assert session._current_episode.discarded is True

    def test_discard_last_episode(self, mock_robot):
        session = _make_session(mock_robot)
        session._episodes = [EpisodeStats(index=0, frames=100, task="test")]
        session._dataset = MagicMock()
        session._dataset.clear_episode_buffer = MagicMock()
        session.discard_episode()
        assert session._episodes[0].discarded is True
        session._dataset.clear_episode_buffer.assert_called_once()

    def test_discard_last_episode_no_dataset(self, mock_robot):
        session = _make_session(mock_robot)
        session._episodes = [EpisodeStats(index=0)]
        session._dataset = None
        session.discard_episode()
        assert session._episodes[0].discarded is True

    def test_discard_last_episode_buffer_clear_exception(self, mock_robot):
        session = _make_session(mock_robot)
        session._episodes = [EpisodeStats(index=0)]
        session._dataset = MagicMock()
        session._dataset.clear_episode_buffer.side_effect = RuntimeError("no buffer")
        session.discard_episode()  # Should not raise
        assert session._episodes[0].discarded is True

    def test_discard_already_discarded(self, mock_robot):
        session = _make_session(mock_robot)
        session._episodes = [EpisodeStats(index=0, discarded=True)]
        session.discard_episode()  # No-op for already discarded


# ── Stop Tests ───────────────────────────────────────────────────────────────


class TestStop:
    def test_stop_sets_flag(self, mock_robot):
        session = _make_session(mock_robot)
        session.stop()
        assert session._stop_flag is True


# ── Save and Push Tests ──────────────────────────────────────────────────────


class TestSaveAndPush:
    def test_basic_save(self, mock_robot, mock_dataset):
        session = _make_session(mock_robot)
        session._dataset = mock_dataset
        session._episodes = [
            EpisodeStats(index=0, frames=100, task="pick"),
            EpisodeStats(index=1, frames=50, task="pick", discarded=True),
            EpisodeStats(index=2, frames=200, task="pick"),
        ]
        result = session.save_and_push()
        assert result["episodes"] == 2  # Excludes discarded
        assert result["total_frames"] == 300
        assert result["discarded"] == 1
        mock_dataset.consolidate.assert_called_once()

    def test_save_with_push_success(self, mock_robot, mock_dataset):
        session = _make_session(mock_robot, push_to_hub=True)
        session._dataset = mock_dataset
        session._episodes = [EpisodeStats(index=0, frames=100)]
        result = session.save_and_push()
        mock_dataset.push_to_hub.assert_called_once()
        assert result["pushed"] is True

    def test_save_with_push_failure(self, mock_robot, mock_dataset):
        session = _make_session(mock_robot, push_to_hub=True)
        session._dataset = mock_dataset
        mock_dataset.push_to_hub.side_effect = RuntimeError("auth error")
        session._episodes = [EpisodeStats(index=0, frames=100)]
        result = session.save_and_push()
        assert result["pushed"] is False
        assert "auth error" in result["push_error"]

    def test_save_consolidation_failure(self, mock_robot, mock_dataset):
        session = _make_session(mock_robot)
        session._dataset = mock_dataset
        mock_dataset.consolidate.side_effect = RuntimeError("consolidation error")
        session._episodes = []
        result = session.save_and_push()
        assert result["episodes"] == 0

    def test_save_no_dataset(self, mock_robot):
        session = _make_session(mock_robot)
        session._episodes = [EpisodeStats(index=0, frames=100)]
        result = session.save_and_push()
        assert result["episodes"] == 1
        assert result["total_frames"] == 100

    def test_save_dataset_without_consolidate(self, mock_robot):
        session = _make_session(mock_robot)
        dataset = MagicMock(spec=[])  # No consolidate
        session._dataset = dataset
        session._episodes = []
        result = session.save_and_push()
        assert result["episodes"] == 0


# ── Disconnect Tests ─────────────────────────────────────────────────────────


class TestDisconnect:
    def test_disconnect_robot_and_teleop(self, mock_robot, mock_teleop):
        session = _make_session(mock_robot, teleop=mock_teleop)
        session._connected = True
        session.disconnect()
        mock_robot.disconnect.assert_called_once()
        mock_teleop.disconnect.assert_called_once()
        assert session._connected is False

    def test_disconnect_no_teleop(self, mock_robot):
        session = _make_session(mock_robot)
        session._connected = True
        session.disconnect()
        mock_robot.disconnect.assert_called_once()
        assert session._connected is False

    def test_disconnect_exception(self, mock_robot, mock_teleop):
        mock_robot.disconnect.side_effect = RuntimeError("usb error")
        mock_teleop.disconnect.side_effect = RuntimeError("usb error")
        session = _make_session(mock_robot, teleop=mock_teleop)
        session._connected = True
        session.disconnect()  # Should not raise
        assert session._connected is False

    def test_disconnect_no_disconnect_method(self, mock_robot):
        """Robot without disconnect method should not crash."""
        del mock_robot.disconnect
        session = _make_session(mock_robot)
        session._connected = True
        session.disconnect()
        assert session._connected is False


# ── Status Tests ─────────────────────────────────────────────────────────────


class TestGetStatus:
    def test_initial_status(self, mock_robot, mock_teleop, mock_policy):
        session = _make_session(mock_robot, teleop=mock_teleop, policy=mock_policy)
        status = session.get_status()
        assert status["repo_id"] == "test/recording"
        assert status["connected"] is False
        assert status["recording"] is False
        assert status["episodes_recorded"] == 0
        assert status["has_teleop"] is True
        assert status["has_policy"] is True
        assert status["task"] == "pick up cube"

    def test_status_after_recording(self, mock_robot, mock_teleop):
        session = _make_session(mock_robot, teleop=mock_teleop, episode_time_s=0.05)
        session._connected = True
        session._dataset = MagicMock()
        session.record_episode()
        status = session.get_status()
        assert status["episodes_recorded"] == 1
        assert status["total_frames"] > 0


# ── Context Manager Tests ────────────────────────────────────────────────────


class TestContextManager:
    @patch("strands_robots.record.RecordSession.save_and_push")
    @patch("strands_robots.record.RecordSession.disconnect")
    @patch("strands_robots.record.RecordSession.connect")
    def test_context_manager(self, mock_connect, mock_disconnect, mock_save, mock_robot):
        with RecordSession(robot=mock_robot, repo_id="test/ctx") as session:
            assert session is not None
        mock_connect.assert_called_once()
        mock_save.assert_called_once()
        mock_disconnect.assert_called_once()

    def test_del(self, mock_robot):
        session = _make_session(mock_robot)
        session._connected = True
        session.__del__()
        mock_robot.disconnect.assert_called()

    def test_del_exception(self, mock_robot):
        mock_robot.disconnect.side_effect = RuntimeError("err")
        session = _make_session(mock_robot)
        session.__del__()  # Should not raise
