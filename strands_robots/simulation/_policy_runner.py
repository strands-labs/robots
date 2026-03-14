"""Policy execution mixin — run_policy, start_policy, record_video, replay_episode, eval_policy."""

import logging
import os
import time
from typing import Any, Dict

import numpy as np

from strands_robots._async_utils import _resolve_coroutine
from strands_robots.simulation._backend import _ensure_mujoco
from strands_robots.simulation._models import TrajectoryStep

logger = logging.getLogger(__name__)


class PolicyRunnerMixin:
    """Policy execution for Simulation. Expects self._world, self._executor, self._policy_threads."""

    def run_policy(
        self,
        robot_name: str,
        policy_provider: str = "mock",
        instruction: str = "",
        duration: float = 10.0,
        action_horizon: int = 8,
        control_frequency: float = 50.0,
        fast_mode: bool = False,
        record_video: str = None,
        video_fps: int = 30,
        video_camera: str = None,
        video_width: int = 640,
        video_height: int = 480,
        **policy_kwargs,
    ) -> Dict[str, Any]:
        """Run a policy on a simulated robot (blocking).

        Args:
            record_video: If set, path to save an MP4 recording of the run.
            video_fps: Frames per second for the recording (default 30).
            video_camera: Camera name for recording (default: first scene camera).
            video_width: Recording width in pixels.
            video_height: Recording height in pixels.
        """
        if self._world is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}
        if robot_name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"❌ Robot '{robot_name}' not found."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data
        robot = self._world.robots[robot_name]

        # Video recording setup
        writer = None
        frame_count = 0
        cam_id = -1
        if record_video:
            import imageio

            os.makedirs(os.path.dirname(os.path.abspath(record_video)), exist_ok=True)
            writer = imageio.get_writer(record_video, fps=video_fps, quality=8, macro_block_size=1)
            if video_camera:
                cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, video_camera)
            elif model.ncam > 0:
                cam_id = 0
            frame_interval = control_frequency / video_fps  # fractional steps per frame

        try:
            from strands_robots.policies import create_policy as _create_policy

            policy = _create_policy(policy_provider, **policy_kwargs)
            policy.set_robot_state_keys(robot.joint_names)

            robot.policy_running = True
            robot.policy_instruction = instruction
            robot.policy_steps = 0
            next_frame_step = 0.0

            sim_duration = duration * control_frequency  # target number of control steps
            start_time = time.time()
            action_sleep = 1.0 / control_frequency

            while robot.policy_steps < sim_duration and robot.policy_running:
                observation = self._get_sim_observation(robot_name)

                coro_or_result = policy.get_actions(observation, instruction)
                actions = _resolve_coroutine(coro_or_result)

                for action_dict in actions[:action_horizon]:
                    if not robot.policy_running:
                        break

                    if self._world._recording:
                        self._world._trajectory.append(
                            TrajectoryStep(
                                timestamp=time.time(),
                                sim_time=self._world.sim_time,
                                robot_name=robot_name,
                                observation={k: v for k, v in observation.items() if not isinstance(v, np.ndarray)},
                                action=action_dict,
                                instruction=instruction,
                            )
                        )
                        if self._world._dataset_recorder is not None:
                            self._world._dataset_recorder.add_frame(
                                observation=observation,
                                action=action_dict,
                                task=instruction,
                            )

                    self._apply_sim_action(robot_name, action_dict)
                    robot.policy_steps += 1

                    if writer and robot.policy_steps >= next_frame_step:
                        renderer = self._get_renderer(video_width, video_height)
                        if cam_id >= 0:
                            renderer.update_scene(data, camera=cam_id)
                        else:
                            renderer.update_scene(data)
                        writer.append_data(renderer.render().copy())
                        frame_count += 1
                        next_frame_step += frame_interval

                    if not fast_mode:
                        time.sleep(action_sleep)

            elapsed = time.time() - start_time
            robot.policy_running = False

            result_text = (
                f"✅ Policy complete on '{robot_name}'\n"
                f"🧠 {policy_provider} | 🎯 {instruction}\n"
                f"⏱️ {elapsed:.1f}s | 📊 {robot.policy_steps} steps | "
                f"🕐 sim_t={self._world.sim_time:.3f}s"
            )

            if writer:
                writer.close()
                file_kb = os.path.getsize(record_video) / 1024
                result_text += (
                    f"\n🎬 Video: {record_video}\n"
                    f"📹 {frame_count} frames, {video_fps}fps, {video_width}x{video_height} | 💾 {file_kb:.0f} KB"
                )

            return {"status": "success", "content": [{"text": result_text}]}

        except Exception as e:
            robot.policy_running = False
            if writer:
                writer.close()
            return {"status": "error", "content": [{"text": f"❌ Policy failed: {e}"}]}

    def start_policy(
        self,
        robot_name: str,
        policy_provider: str = "mock",
        instruction: str = "",
        duration: float = 10.0,
        fast_mode: bool = False,
        **policy_kwargs,
    ) -> Dict[str, Any]:
        """Start policy execution in background (non-blocking)."""
        if self._world is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}
        if robot_name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"❌ Robot '{robot_name}' not found."}]}

        future = self._executor.submit(
            self.run_policy,
            robot_name,
            policy_provider,
            instruction,
            duration,
            fast_mode=fast_mode,
            **policy_kwargs,
        )
        self._policy_threads[robot_name] = future

        return {
            "status": "success",
            "content": [{"text": f"🚀 Policy started on '{robot_name}' (async)"}],
        }

    def replay_episode(
        self,
        repo_id: str,
        robot_name: str = None,
        episode: int = 0,
        root: str = None,
        speed: float = 1.0,
    ) -> Dict[str, Any]:
        """Replay actions from a LeRobotDataset episode in simulation."""
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world. Call create_world first."}]}

        if robot_name is None:
            if not self._world.robots:
                return {"status": "error", "content": [{"text": "❌ No robots in sim. Add one first."}]}
            robot_name = next(iter(self._world.robots))

        robot = self._world.robots.get(robot_name)
        if robot is None:
            return {"status": "error", "content": [{"text": f"❌ Robot '{robot_name}' not found"}]}

        try:
            from strands_robots.dataset_recorder import load_lerobot_episode

            ds, episode_start, episode_length = load_lerobot_episode(repo_id, episode, root)
        except ImportError:
            return {"status": "error", "content": [{"text": "❌ lerobot not installed"}]}
        except (ValueError, Exception) as e:
            return {"status": "error", "content": [{"text": f"❌ {e}"}]}

        mj = _ensure_mujoco()
        dataset_fps = getattr(ds, "fps", 30)
        frame_interval = 1.0 / (dataset_fps * speed)
        model = self._world._model
        data = self._world._data
        n_actuators = model.nu
        frames_applied = 0
        start_time = time.time()

        for frame_idx in range(episode_length):
            step_start = time.time()
            frame = ds[episode_start + frame_idx]

            if "action" in frame:
                action_vals = frame["action"]
                if hasattr(action_vals, "numpy"):
                    action_vals = action_vals.numpy()
                if hasattr(action_vals, "tolist"):
                    action_vals = action_vals.tolist()
                for i in range(min(len(action_vals), n_actuators)):
                    data.ctrl[i] = float(action_vals[i])

            mj.mj_step(model, data)
            frames_applied += 1

            elapsed = time.time() - step_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        duration = time.time() - start_time
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"▶️ Replayed episode {episode} from {repo_id} on '{robot_name}'\n"
                        f"Frames: {frames_applied}/{episode_length} | Duration: {duration:.1f}s | Speed: {speed}x"
                    )
                },
                {
                    "json": {
                        "episode": episode,
                        "robot_name": robot_name,
                        "frames_applied": frames_applied,
                        "total_frames": episode_length,
                        "duration_s": round(duration, 2),
                        "speed": speed,
                    }
                },
            ],
        }

    def eval_policy(
        self,
        robot_name: str = None,
        policy_provider: str = "mock",
        instruction: str = "",
        n_episodes: int = 10,
        max_steps: int = 300,
        success_fn: str = None,
        **policy_kwargs,
    ) -> Dict[str, Any]:
        """Evaluate a policy over multiple episodes with success metrics."""
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world. Call create_world first."}]}

        if robot_name is None:
            if not self._world.robots:
                return {"status": "error", "content": [{"text": "❌ No robots"}]}
            robot_name = next(iter(self._world.robots))

        robot = self._world.robots.get(robot_name)
        if robot is None:
            return {"status": "error", "content": [{"text": f"❌ Robot '{robot_name}' not found"}]}

        from strands_robots.policies import create_policy

        mj = _ensure_mujoco()
        policy_instance = create_policy(policy_provider, **policy_kwargs)
        policy_instance.set_robot_state_keys(robot.joint_names)

        model = self._world._model
        data = self._world._data

        results = []
        for ep in range(n_episodes):
            mj.mj_resetData(model, data)
            mj.mj_forward(model, data)

            total_reward = 0.0
            success = False
            steps = 0

            for step in range(max_steps):
                obs = self._get_sim_observation(robot_name=robot_name)
                coro_or_result = policy_instance.get_actions(obs, instruction)
                actions = _resolve_coroutine(coro_or_result)

                if actions:
                    self._apply_sim_action(robot_name, actions[0])

                mj.mj_step(model, data)
                steps += 1

                if success_fn == "contact":
                    for i in range(data.ncon):
                        if data.contact[i].dist < 0:
                            success = True
                            break
                    if success:
                        break

            results.append({"episode": ep, "steps": steps, "success": success, "reward": total_reward})

        n_success = sum(1 for r in results if r["success"])
        success_rate = n_success / max(n_episodes, 1)
        avg_steps = sum(r["steps"] for r in results) / max(n_episodes, 1)

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"📊 Evaluation: {policy_provider} on '{robot_name}'\n"
                        f"Episodes: {n_episodes} | Success: {n_success}/{n_episodes} ({success_rate:.1%})\n"
                        f"Avg steps: {avg_steps:.0f}/{max_steps}"
                    )
                },
                {
                    "json": {
                        "success_rate": round(success_rate, 4),
                        "n_episodes": n_episodes,
                        "n_success": n_success,
                        "avg_steps": round(avg_steps, 1),
                        "max_steps": max_steps,
                        "episodes": results,
                    }
                },
            ],
        }
