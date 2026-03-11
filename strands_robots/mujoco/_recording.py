"""Recording — start/stop recording, video capture, replay, evaluation."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from ._registry import _ensure_mujoco
from ._rendering import apply_sim_action, get_renderer, get_sim_observation

if TYPE_CHECKING:
    from ._core import MujocoBackend

logger = logging.getLogger(__name__)


def start_recording(
    sim: MujocoBackend,
    repo_id: str = "local/sim_recording",
    task: str = "",
    fps: int = 30,
    root: str = None,
    push_to_hub: bool = False,
    vcodec: str = "libsvtav1",
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Start recording to LeRobotDataset format (parquet + video)."""
    if sim._world is None:
        return {"status": "error", "content": [{"text": "No world."}]}
    # Re-check at call time to avoid stale module-level state from test mocks
    try:
        from strands_robots.dataset_recorder import DatasetRecorder as _DatasetRecorder
        from strands_robots.dataset_recorder import has_lerobot_dataset as _has_lerobot
    except ImportError:
        def _has_lerobot():
            return False

        _DatasetRecorder = None
    if not _has_lerobot() or _DatasetRecorder is None:
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        "lerobot not installed. Install with: pip install lerobot\n"
                        "Required for dataset recording."
                    )
                }
            ],
        }

    sim._world._recording = True
    sim._world._trajectory = []
    sim._world._push_to_hub = push_to_hub

    try:
        # Clean up existing dataset directory if overwrite is True
        if overwrite:
            if root:
                dataset_dir = Path(root)
            elif (
                "/" not in repo_id
                or repo_id.startswith("/")
                or repo_id.startswith("./")
            ):
                dataset_dir = Path(repo_id)
            else:
                dataset_dir = (
                    Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
                )
            if dataset_dir.exists() and dataset_dir.is_dir():
                shutil.rmtree(dataset_dir)
                logger.info("Removed existing dataset dir: %s", dataset_dir)

        joint_names = []
        camera_keys = []
        robot_type = "unknown"
        for rname, robot in sim._world.robots.items():
            joint_names.extend(robot.joint_names)
            robot_type = robot.data_config or rname
        mj = _ensure_mujoco()
        for i in range(sim._world._model.ncam):
            cam_name = mj.mj_id2name(sim._world._model, mj.mjtObj.mjOBJ_CAMERA, i)
            if cam_name:
                camera_keys.append(cam_name)

        sim._world._dataset_recorder = _DatasetRecorder.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            joint_names=joint_names,
            camera_keys=camera_keys,
            task=task,
            root=root,
            vcodec=vcodec,
        )
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Recording to LeRobotDataset: {repo_id}\n"
                        f"{len(joint_names)} joints, {len(camera_keys)} cameras @ {fps}fps\n"
                        f"Codec: {vcodec} | Task: {task or '(set per policy)'}\n"
                        f"Run policies to capture frames, then stop_recording to save episode"
                    )
                }
            ],
        }
    except Exception as e:
        sim._world._recording = False
        logger.error("Dataset recorder init failed: %s", e)
        return {
            "status": "error",
            "content": [{"text": f"Dataset init failed: {e}"}],
        }


def stop_recording(sim: MujocoBackend, output_path: str = None) -> Dict[str, Any]:
    """Stop recording and save episode to LeRobotDataset."""
    if sim._world is None or not sim._world._recording:
        return {"status": "error", "content": [{"text": "Not recording."}]}

    sim._world._recording = False
    recorder = sim._world._dataset_recorder

    if recorder is None:
        return {
            "status": "error",
            "content": [{"text": "No dataset recorder active."}],
        }

    recorder.save_episode()
    push_result = None
    if getattr(sim._world, "_push_to_hub", False):
        push_result = recorder.push_to_hub(tags=["strands-robots", "sim"])

    repo_id = recorder.repo_id
    frame_count = recorder.frame_count
    episode_count = recorder.episode_count
    root = recorder.root

    recorder.finalize()
    sim._world._dataset_recorder = None
    sim._world._trajectory = []

    text = (
        f"Episode saved to LeRobotDataset\n"
        f"{repo_id} -- {frame_count} frames, {episode_count} episode(s)\n"
        f"Local: {root}"
    )
    if push_result and push_result.get("status") == "success":
        text += "\nPushed to HuggingFace Hub"

    return {"status": "success", "content": [{"text": text}]}


def get_recording_status(sim: MujocoBackend) -> Dict[str, Any]:
    if sim._world is None:
        return {"status": "error", "content": [{"text": "❌ No world."}]}

    recording = sim._world._recording
    steps = len(sim._world._trajectory)

    return {
        "status": "success",
        "content": [
            {
                "text": f"{'🔴 Recording' if recording else '⚪ Not recording'}: {steps} steps captured"
            }
        ],
    }


def record_video(
    sim: MujocoBackend,
    robot_name: str,
    policy_provider: str = "lerobot_local",
    instruction: str = "",
    duration: float = 10.0,
    fps: int = 30,
    camera_name: str = None,
    width: int = 640,
    height: int = 480,
    output_path: str = None,
    cosmos_transfer: bool = False,
    cosmos_prompt: str = None,
    cosmos_control: str = "depth",
    **policy_kwargs,
) -> Dict[str, Any]:
    """Run a policy and record video simultaneously."""
    if sim._world is None or sim._world._model is None:
        return {"status": "error", "content": [{"text": "❌ No simulation."}]}
    if robot_name not in sim._world.robots:
        return {
            "status": "error",
            "content": [{"text": f"❌ Robot '{robot_name}' not found."}],
        }

    mj = _ensure_mujoco()
    model, data = sim._world._model, sim._world._data
    robot = sim._world.robots[robot_name]

    # Auto-generate output path
    if output_path is None:
        output_dir = os.path.join(tempfile.gettempdir(), "strands_sim")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"video_{robot_name}_{int(time.time())}.mp4"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Resolve camera
    cam_id = -1
    if camera_name:
        cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
    else:
        if model.ncam > 0:
            cam_id = 0
            camera_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, 0)

    try:
        import imageio

        from strands_robots._async_utils import _resolve_coroutine
        from strands_robots.policies import create_policy as _create_policy

        policy = _create_policy(policy_provider, **policy_kwargs)
        policy.set_robot_state_keys(robot.joint_names)

        total_frames = int(duration * fps)
        dt = 1.0 / fps
        phys_per_frame = max(1, int(dt / model.opt.timestep))

        logger.info(
            "Recording %s frames at %sfps, %s physics steps/frame",
            total_frames, fps, phys_per_frame,
        )

        writer = imageio.get_writer(
            output_path, fps=fps, quality=8, macro_block_size=1
        )
        robot.policy_running = True
        robot.policy_instruction = instruction
        robot.policy_steps = 0

        start_time = time.time()

        for frame_idx in range(total_frames):
            if not robot.policy_running:
                break

            observation = get_sim_observation(sim, robot_name, cam_name=camera_name)

            try:
                coro_or_result = policy.get_actions(observation, instruction)
                actions = _resolve_coroutine(coro_or_result)
            except Exception as e:
                logger.warning("Policy inference failed at frame %s: %s", frame_idx, e)
                actions = [{key: 0.0 for key in robot.joint_names}]

            if actions:
                apply_sim_action(sim, robot_name, actions[0], n_substeps=phys_per_frame)
                robot.policy_steps += 1
            else:
                for _ in range(phys_per_frame):
                    mj.mj_step(model, data)
                sim._world.sim_time = data.time

            renderer = get_renderer(sim, width, height)
            if cam_id >= 0:
                renderer.update_scene(data, camera=cam_id)
            else:
                renderer.update_scene(data)
            frame = renderer.render().copy()

            writer.append_data(frame)

            if (frame_idx + 1) % 60 == 0:
                elapsed = time.time() - start_time
                real_fps = (frame_idx + 1) / elapsed
                logger.info(
                    "  frame %s/%s | %.1fs | %.1f fps",
                    frame_idx + 1, total_frames, elapsed, real_fps,
                )

        writer.close()
        robot.policy_running = False

        elapsed = time.time() - start_time
        file_kb = os.path.getsize(output_path) / 1024

        # Cosmos Transfer 2.5: Convert sim video to photorealistic
        cosmos_output = None
        if cosmos_transfer:
            try:
                from strands_robots.cosmos_transfer import (
                    CosmosTransferConfig,
                    CosmosTransferPipeline,
                )

                prompt = (
                    cosmos_prompt
                    or instruction
                    or f"Robot {robot_name} performing task in modern workspace"
                )
                cosmos_out_path = output_path.replace(".mp4", "_photorealistic.mp4")

                logger.info(
                    "🎬 Running Cosmos Transfer 2.5 (%s) → %s",
                    cosmos_control, cosmos_out_path,
                )

                config = CosmosTransferConfig(control_type=cosmos_control)
                pipeline = CosmosTransferPipeline(config=config)
                pipeline.transfer_video(
                    input_video=output_path,
                    prompt=prompt,
                    output_path=cosmos_out_path,
                )

                cosmos_output = cosmos_out_path
                cosmos_kb = (
                    os.path.getsize(cosmos_out_path) / 1024
                    if os.path.exists(cosmos_out_path)
                    else 0
                )
                logger.info("🎬 Cosmos Transfer complete: %.0f KB", cosmos_kb)
            except ImportError:
                logger.warning(
                    "Cosmos Transfer not available (missing deps). Skipping."
                )
            except Exception as e:
                logger.warning(
                    "Cosmos Transfer failed: %s. Sim video still available.", e
                )

        result_text = (
            f"🎬 Video recorded: {output_path}\n"
            f"📹 {total_frames} frames, {fps}fps, {width}x{height}\n"
            f"🤖 Robot: {robot_name} | 🧠 Policy: {policy_provider}\n"
            f"⏱️ {elapsed:.1f}s real time | 💾 {file_kb:.0f} KB\n"
            f"📊 {robot.policy_steps} policy steps"
        )
        if cosmos_output:
            result_text += f"\n🌌 Photorealistic: {cosmos_output}"

        return {
            "status": "success",
            "content": [{"text": result_text}],
        }

    except Exception as e:
        robot.policy_running = False
        logger.error("Video recording failed: %s", e)
        return {
            "status": "error",
            "content": [{"text": f"❌ Video recording failed: {e}"}],
        }


def replay_episode(
    sim: MujocoBackend,
    repo_id: str,
    robot_name: str = None,
    episode: int = 0,
    root: str = None,
    speed: float = 1.0,
) -> Dict[str, Any]:
    """Replay actions from a LeRobotDataset episode in simulation."""
    if sim._world is None:
        return {
            "status": "error",
            "content": [{"text": "❌ No world. Call create_world first."}],
        }

    if robot_name is None:
        if not sim._world.robots:
            return {
                "status": "error",
                "content": [{"text": "❌ No robots in sim. Add one first."}],
            }
        robot_name = next(iter(sim._world.robots))

    robot = sim._world.robots.get(robot_name)
    if robot is None:
        return {
            "status": "error",
            "content": [{"text": f"❌ Robot '{robot_name}' not found"}],
        }

    try:
        from strands_robots.dataset_recorder import load_lerobot_episode

        ds, episode_start, episode_length = load_lerobot_episode(
            repo_id, episode, root
        )
    except ImportError:
        return {
            "status": "error",
            "content": [{"text": "❌ lerobot not installed"}],
        }
    except (ValueError, Exception) as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ {e}"}],
        }

    mj = _ensure_mujoco()
    dataset_fps = getattr(ds, "fps", 30)
    frame_interval = 1.0 / (dataset_fps * speed)
    model = sim._world._model
    data = sim._world._data
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
    sim: MujocoBackend,
    robot_name: str = None,
    policy_provider: str = "mock",
    instruction: str = "",
    n_episodes: int = 10,
    max_steps: int = 300,
    success_fn: str = None,
    **policy_kwargs,
) -> Dict[str, Any]:
    """Evaluate a policy over multiple episodes with success metrics."""
    if sim._world is None:
        return {
            "status": "error",
            "content": [{"text": "❌ No world. Call create_world first."}],
        }

    if robot_name is None:
        if not sim._world.robots:
            return {"status": "error", "content": [{"text": "❌ No robots"}]}
        robot_name = next(iter(sim._world.robots))

    robot = sim._world.robots.get(robot_name)
    if robot is None:
        return {
            "status": "error",
            "content": [{"text": f"❌ Robot '{robot_name}' not found"}],
        }

    from strands_robots._async_utils import _resolve_coroutine
    from strands_robots.policies import create_policy

    policy_instance = create_policy(policy_provider, **policy_kwargs)
    policy_instance.set_robot_state_keys(robot.joint_names)

    mj = _ensure_mujoco()
    model = sim._world._model
    data = sim._world._data

    results = []
    for ep in range(n_episodes):
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

        total_reward = 0.0
        success = False
        steps = 0

        for step_idx in range(max_steps):
            obs = get_sim_observation(sim, robot_name=robot_name)

            coro_or_result = policy_instance.get_actions(obs, instruction)
            actions = _resolve_coroutine(coro_or_result)

            if actions:
                apply_sim_action(sim, robot_name, actions[0])

            mj.mj_step(model, data)
            steps += 1

            if success_fn == "contact":
                contacts = []
                for i in range(data.ncon):
                    c = data.contact[i]
                    if c.dist < 0:
                        contacts.append(i)
                if contacts:
                    success = True
                    break

        results.append(
            {
                "episode": ep,
                "steps": steps,
                "success": success,
                "reward": total_reward,
            }
        )

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
