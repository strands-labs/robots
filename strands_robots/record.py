"""
Native Recording Pipeline — Teleop + Policy recording to LeRobotDataset.

Equivalent to lerobot-record but callable programmatically from agent tools or scripts.

This module provides a native Python recording flow equivalent to `lerobot-record`,
but callable programmatically from agent tools or scripts. It supports:

1. **Teleop recording**: Human demos via leader→follower with LeRobot Teleoperator
2. **Policy recording**: Autonomous policy rollouts (same as robot.py record_task)
3. **Hybrid**: Teleop with policy-assisted control

Uses LeRobot's native:
- `Teleoperator` ABC for leader device
- `Robot` ABC for follower
- `LeRobotDataset` for storage
- `make_default_robot_action_processor` for action processing
- `make_default_robot_observation_processor` for observation processing

Usage:
    from strands_robots.record import RecordSession

    session = RecordSession(
        robot=my_robot,           # LeRobot Robot instance
        teleop=my_teleop,         # LeRobot Teleoperator instance (optional)
        repo_id="user/my_data",
        task="Pick up the red cube",
        fps=30,
    )
    session.connect()
    session.record_episode()      # Records one episode (teleop or policy)
    session.record_episode()      # Record another
    session.save_and_push()       # Finalize + optionally push to Hub
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Optional live visualizer
try:
    from strands_robots.visualizer import RecordingVisualizer

    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


class RecordMode(Enum):
    TELEOP = "teleop"
    POLICY = "policy"
    IDLE = "idle"


@dataclass
class EpisodeStats:
    """Stats for a single recorded episode."""

    index: int = 0
    frames: int = 0
    duration_s: float = 0.0
    task: str = ""
    discarded: bool = False


class RecordSession:
    """Native recording session using LeRobot Robot + Teleoperator + Dataset.

    Equivalent to `lerobot-record` but callable from Python / agent tools.
    Supports both teleop demos and policy rollouts.
    """

    def __init__(
        self,
        robot,  # LeRobot Robot instance
        teleop=None,  # LeRobot Teleoperator instance (optional)
        policy=None,  # strands_robots Policy instance (optional)
        repo_id: str = "local/recording",
        task: str = "",
        fps: int = 30,
        root: Optional[str] = None,
        use_videos: bool = True,
        vcodec: str = "libsvtav1",
        streaming_encoding: bool = True,
        image_writer_threads: int = 4,
        push_to_hub: bool = False,
        num_episodes: int = 50,
        episode_time_s: float = 60.0,
        reset_time_s: float = 10.0,
        use_processor: bool = True,
        visualize: str = "",
    ):
        self.robot = robot
        self.teleop = teleop
        self.policy = policy
        self.repo_id = repo_id
        self.default_task = task
        self.fps = fps
        self.root = root
        self.use_videos = use_videos
        self.vcodec = vcodec
        self.streaming_encoding = streaming_encoding
        self.image_writer_threads = image_writer_threads
        self.push_to_hub = push_to_hub
        self.num_episodes = num_episodes
        self.episode_time_s = episode_time_s
        self.reset_time_s = reset_time_s
        self.use_processor = use_processor
        self.visualize = visualize

        # Live visualizer
        self._visualizer = None
        if visualize and HAS_VISUALIZER:
            self._visualizer = RecordingVisualizer(
                mode=visualize,
                refresh_rate=2.0,
            )

        # State
        self._dataset = None
        self._connected = False
        self._recording = False
        self._episodes: List[EpisodeStats] = []
        self._current_episode: Optional[EpisodeStats] = None
        self._stop_flag = False

        # Processors (from lerobot.processor)
        self._action_processor = None
        self._observation_processor = None

        logger.info("RecordSession: %s @ %sfps, task='%s'", repo_id, fps, task)

    def connect(self):
        """Connect robot and teleop devices."""
        if self._connected:
            return

        # Connect robot
        if not self.robot.is_connected:
            self.robot.connect()
            logger.info("Robot connected: %s", self.robot)

        # Connect teleop if provided
        if self.teleop and not self.teleop.is_connected:
            self.teleop.connect()
            logger.info("Teleoperator connected: %s", self.teleop)

        # Initialize processors if available
        if self.use_processor:
            self._init_processors()

        # Create dataset
        self._create_dataset()
        # Start live visualizer if configured
        if self._visualizer:
            self._visualizer.stats.repo_id = self.repo_id
            self._visualizer.stats.fps_target = self.fps
            self._visualizer.stats.total_episodes = self.num_episodes
            self._visualizer.start()

        self._connected = True

    def _init_processors(self):
        """Initialize LeRobot action/observation processors."""
        try:
            from lerobot.processor import (
                make_default_robot_action_processor,
                make_default_robot_observation_processor,
            )

            self._action_processor = make_default_robot_action_processor()
            self._observation_processor = make_default_robot_observation_processor()
            logger.info("LeRobot processors initialized")
        except ImportError:
            logger.debug("LeRobot processor not available, using raw obs/action")
        except Exception as e:
            logger.debug("Processor init failed: %s, using raw obs/action", e)

    def _create_dataset(self):
        """Create LeRobotDataset for recording."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            # Get robot features
            features = {}
            if hasattr(self.robot, "observation_features"):
                obs_feats = self.robot.observation_features
                for key, val in obs_feats.items():
                    if isinstance(val, tuple):
                        # Image feature (h, w, c)
                        features[f"observation.images.{key}"] = {
                            "dtype": "video" if self.use_videos else "image",
                            "shape": val,
                            "names": ["height", "width", "channels"],
                        }
                    elif val is float:
                        # Scalar state feature — accumulate into observation.state
                        pass  # Handled below

                # Build observation.state from scalar features
                state_keys = [k for k, v in obs_feats.items() if v is float]
                if state_keys:
                    features["observation.state"] = {
                        "dtype": "float32",
                        "shape": (len(state_keys),),
                        "names": state_keys,
                    }

            if hasattr(self.robot, "action_features"):
                act_feats = self.robot.action_features
                action_keys = [k for k, v in act_feats.items() if v is float]
                if action_keys:
                    features["action"] = {
                        "dtype": "float32",
                        "shape": (len(action_keys),),
                        "names": action_keys,
                    }

            robot_type = getattr(
                self.robot, "robot_type", getattr(self.robot, "name", "unknown")
            )

            self._dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.root,
                robot_type=robot_type,
                features=features,
                use_videos=self.use_videos,
                image_writer_threads=self.image_writer_threads,
                vcodec=self.vcodec,
            )
            logger.info("Dataset created: %s (%s features)", self.repo_id, len(features))

        except Exception as e:
            logger.error("Dataset creation failed: %s", e)
            raise

    def record_episode(
        self,
        task: Optional[str] = None,
        mode: Optional[RecordMode] = None,
        duration: Optional[float] = None,
        on_frame: Optional[Callable] = None,
    ) -> EpisodeStats:
        """Record a single episode.

        Args:
            task: Task description (defaults to session task)
            mode: TELEOP or POLICY (auto-detected if not specified)
            duration: Episode duration in seconds
            on_frame: Callback(frame_idx, obs, action) called each frame

        Returns:
            EpisodeStats for the recorded episode
        """
        if not self._connected:
            self.connect()

        # Auto-detect mode
        if mode is None:
            mode = RecordMode.TELEOP if self.teleop else RecordMode.POLICY

        task = task or self.default_task
        duration = duration or self.episode_time_s
        frame_interval = 1.0 / self.fps

        ep_idx = len(self._episodes)
        stats = EpisodeStats(index=ep_idx, task=task)
        self._current_episode = stats
        self._stop_flag = False

        logger.info(
            f"Recording episode {ep_idx} ({mode.value}): '{task}' for {duration}s"
        )
        self._recording = True

        start = time.time()
        frame_idx = 0

        try:
            while time.time() - start < duration and not self._stop_flag:
                step_start = time.time()

                # Get observation from robot
                obs = self.robot.get_observation()

                # Process observation if processor available
                if self._observation_processor:
                    try:
                        obs = self._observation_processor(obs)
                    except Exception:
                        pass  # Fallback to raw obs

                # Get action based on mode
                if mode == RecordMode.TELEOP and self.teleop:
                    action = self.teleop.get_action()

                    # Process action
                    if self._action_processor:
                        try:
                            action = self._action_processor((action, obs))
                        except Exception:
                            pass

                    # Send action to follower robot
                    self.robot.send_action(action)

                elif mode == RecordMode.POLICY and self.policy:
                    # Get action from policy
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None

                    if loop and loop.is_running():
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as ex:
                            actions = ex.submit(
                                lambda: asyncio.run(self.policy.get_actions(obs, task))
                            ).result()
                    else:
                        actions = asyncio.run(self.policy.get_actions(obs, task))

                    if actions:
                        action = actions[0]
                        self.robot.send_action(action)
                    else:
                        action = {}
                else:
                    # Idle mode — just observe
                    action = {}

                # Write frame to dataset
                if self._dataset is not None:
                    try:
                        frame = self._build_dataset_frame(obs, action, task)
                        self._dataset.add_frame(frame)
                    except Exception as e:
                        logger.debug("Frame write failed: %s", e)

                # Callback
                if on_frame:
                    try:
                        on_frame(frame_idx, obs, action)
                    except Exception:
                        pass

                frame_idx += 1

                # Maintain frame rate
                elapsed = time.time() - step_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Episode interrupted by user")
        except Exception as e:
            logger.error("Episode recording error: %s", e)

        # Finalize episode stats
        self._recording = False
        stats.frames = frame_idx
        stats.duration_s = time.time() - start
        self._current_episode = None

        # Save episode to dataset
        # LeRobot v3: save_episode() takes no task arg — tasks are embedded
        # per-frame via add_frame() which includes the "task" key.
        if self._dataset is not None and frame_idx > 0:
            try:
                self._dataset.save_episode()
                logger.info(
                    f"Episode {ep_idx} saved: {frame_idx} frames, "
                    f"{stats.duration_s:.1f}s"
                )
            except Exception as e:
                logger.error("Episode save failed: %s", e)
                stats.discarded = True

        self._episodes.append(stats)
        return stats

    def discard_episode(self):
        """Discard the current or last episode."""
        if self._current_episode:
            self._stop_flag = True
            self._current_episode.discarded = True
            logger.info("Episode discarded")
        elif self._episodes and not self._episodes[-1].discarded:
            self._episodes[-1].discarded = True
            # Try to clear last episode from dataset
            if self._dataset and hasattr(self._dataset, "clear_episode_buffer"):
                try:
                    self._dataset.clear_episode_buffer()
                except Exception:
                    pass
            logger.info("Last episode discarded")

    def stop(self):
        """Stop current recording."""
        self._stop_flag = True

    def _build_dataset_frame(
        self,
        obs: Dict[str, Any],
        action: Dict[str, Any],
        task: str,
    ) -> Dict[str, Any]:
        """Build a frame dict in LeRobotDataset v3 format from raw obs/action.

        LeRobot v3 add_frame() requires:
        - "task": str (mandatory, popped by add_frame and stored per-frame)
        - Feature keys matching features dict exactly (e.g. "observation.state", "action")
        - Values as numpy arrays or torch tensors (auto-converted by add_frame)
        """
        import torch

        frame = {}

        # Task is mandatory for LeRobot v3 — stored per-frame in episode buffer
        frame["task"] = task or self.default_task or "untitled"

        # Observation: separate images from state
        state_values = []
        if hasattr(self.robot, "observation_features"):
            obs_feats = self.robot.observation_features
            for key, val in obs_feats.items():
                if isinstance(val, tuple) and key in obs:
                    # Image
                    img = obs[key]
                    if isinstance(img, np.ndarray):
                        frame[f"observation.images.{key}"] = torch.from_numpy(img)
                    elif isinstance(img, torch.Tensor):
                        frame[f"observation.images.{key}"] = img
                elif (val is float) and key in obs:
                    state_values.append(float(obs[key]))
        else:
            # Fallback: treat non-array values as state
            camera_keys = []
            if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
                camera_keys = list(self.robot.config.cameras.keys())

            for key, val in obs.items():
                if key in camera_keys and isinstance(val, np.ndarray):
                    frame[f"observation.images.{key}"] = torch.from_numpy(val)
                elif isinstance(val, (int, float)):
                    state_values.append(float(val))

        if state_values:
            frame["observation.state"] = torch.tensor(state_values, dtype=torch.float32)

        # Action — avoid bare truthiness check for numpy/torch arrays
        has_action = action is not None and not (
            isinstance(action, dict) and len(action) == 0
        )
        if has_action:
            if isinstance(action, dict):
                action_values = list(action.values())
                frame["action"] = torch.tensor(
                    [float(v) for v in action_values], dtype=torch.float32
                )
            elif isinstance(action, (np.ndarray, list)):
                frame["action"] = torch.tensor(action, dtype=torch.float32)
            elif isinstance(action, torch.Tensor):
                frame["action"] = action

        return frame

    def save_and_push(self) -> Dict[str, Any]:
        """Finalize dataset and optionally push to Hub.

        Returns:
            Dict with dataset info
        """
        result = {
            "repo_id": self.repo_id,
            "episodes": len([e for e in self._episodes if not e.discarded]),
            "total_frames": sum(e.frames for e in self._episodes if not e.discarded),
            "discarded": sum(1 for e in self._episodes if e.discarded),
        }

        if self._dataset:
            try:
                if hasattr(self._dataset, "consolidate"):
                    self._dataset.consolidate()
                result["root"] = (
                    str(self._dataset.root) if hasattr(self._dataset, "root") else None
                )
            except Exception as e:
                logger.error("Dataset consolidation failed: %s", e)

            if self.push_to_hub:
                try:
                    self._dataset.push_to_hub(tags=["strands-robots"])
                    result["pushed"] = True
                    logger.info("Dataset pushed to Hub: %s", self.repo_id)
                except Exception as e:
                    result["pushed"] = False
                    result["push_error"] = str(e)
                    logger.error("Hub push failed: %s", e)

        return result

    def disconnect(self):
        """Disconnect robot and teleop."""
        if self.teleop and hasattr(self.teleop, "disconnect"):
            try:
                self.teleop.disconnect()
            except Exception:
                pass
        if self.robot and hasattr(self.robot, "disconnect"):
            try:
                self.robot.disconnect()
            except Exception:
                pass
        self._connected = False

    def get_status(self) -> Dict[str, Any]:
        """Get recording session status."""
        return {
            "repo_id": self.repo_id,
            "connected": self._connected,
            "recording": self._recording,
            "episodes_recorded": len(self._episodes),
            "episodes_discarded": sum(1 for e in self._episodes if e.discarded),
            "total_frames": sum(e.frames for e in self._episodes if not e.discarded),
            "has_teleop": self.teleop is not None,
            "has_policy": self.policy is not None,
            "task": self.default_task,
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.save_and_push()
        self.disconnect()

    def __del__(self):
        try:
            if self._connected:
                self.disconnect()
        except Exception:
            pass
