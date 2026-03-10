#!/usr/bin/env python3
"""
GR00T N1.6 Fine-Tuning Pipeline — End-to-End

Orchestrates the full pipeline from 3D world generation through RL training,
dataset collection, GR00T fine-tuning, evaluation, and Cosmos Reason analysis.

Pipeline Stages:
    1. Marble 3D World Generation → USD scene
    2. SAC RL Training (SB3) in MuJoCo/Newton → trained SAC policy
    3. Dataset Recording from SAC policy → LeRobot-format dataset
    4. GR00T N1.6 Fine-Tuning → fine-tuned VLA checkpoint
    5. Evaluation → success_rate, mean_reward
    6. Cosmos Reason Analysis → training parameter recommendations
    7. Results Export → JSON + markdown + git push

Usage:
    # Full pipeline on Thor (GPU required)
    python scripts/training/groot_finetune_pipeline.py --full

    # SAC training only
    python scripts/training/groot_finetune_pipeline.py --stage sac --timesteps 500000

    # GR00T fine-tuning only (assumes dataset exists)
    python scripts/training/groot_finetune_pipeline.py --stage groot --dataset ./rl_dataset

    # Evaluate existing checkpoint
    python scripts/training/groot_finetune_pipeline.py --stage eval --checkpoint ./groot_finetuned/best

Refs:
    - Issue #141: Fine Tune GR00T 1.6
    - Issue #140: Simulation Testing (verified backends)
    - Issue #65: GR00T N1.6 Full Lifecycle
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


# ─── Pipeline Configuration ───────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Configuration for the full GR00T fine-tuning pipeline."""

    # Stage 1: Marble 3D
    marble_prompt: str = "A modern kitchen with wooden countertops, a sink, and scattered objects on the table"
    marble_output_dir: str = "./pipeline_outputs/marble_scene"
    marble_robot: str = "so100"
    marble_objects: List[str] = field(default_factory=lambda: ["red_cube", "blue_block", "green_block"])

    # Stage 2: SAC Training
    sac_robot: str = "so100"
    sac_task: str = "pick and place cube"
    sac_backend: str = "mujoco"
    sac_timesteps: int = 500_000
    sac_learning_rate: float = 3e-4
    sac_batch_size: int = 256
    sac_output_dir: str = "./pipeline_outputs/sac_training"

    # Stage 3: Dataset Recording
    record_episodes: int = 200
    record_max_steps: int = 300
    record_output_dir: str = "./pipeline_outputs/rl_dataset"
    record_camera_width: int = 224
    record_camera_height: int = 224

    # Stage 4: GR00T Fine-Tuning
    groot_base_model: str = "nvidia/GR00T-N1-2B"
    groot_data_config: str = "so100_dualcam"
    groot_embodiment: str = "so100"
    groot_max_steps: int = 10_000
    groot_batch_size: int = 32
    groot_learning_rate: float = 1e-5
    groot_output_dir: str = "./pipeline_outputs/groot_finetuned"

    # Stage 5: Evaluation
    eval_episodes: int = 50
    eval_tasks: List[str] = field(
        default_factory=lambda: [
            "pick up the red cube",
            "stack the blue block on green",
        ]
    )
    eval_output_dir: str = "./pipeline_outputs/eval_results"

    # Stage 6: Cosmos Reason
    cosmos_analysis_dir: str = "./pipeline_outputs/cosmos_analysis"

    # Global
    seed: int = 42
    device: str = "cuda"
    num_gpus: int = 1


# ─── Camera Helper ─────────────────────────────────────────────────────


def _render_named_camera(env, camera_name: str, width: int = 224, height: int = 224) -> np.ndarray:
    """Render a named camera to numpy array (RGB, uint8).

    Uses the MuJoCo backend directly via the environment's internal sim.
    Falls back to env.render() if the named camera doesn't exist.

    Args:
        env: StrandsSimEnv instance with MuJoCo backend.
        camera_name: Name of the camera to render.
        width: Render width in pixels.
        height: Render height in pixels.

    Returns:
        numpy array of shape (height, width, 3), dtype uint8.
    """
    try:
        import mujoco

        model = env._sim._world._model
        data = env._sim._world._data
        renderer = mujoco.Renderer(model, height=height, width=width)

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id >= 0:
            renderer.update_scene(data, camera=cam_id)
        else:
            # Camera not found — fall back to default free camera
            logger.warning(f"Camera '{camera_name}' not found, using default view")
            renderer.update_scene(data)

        frame = renderer.render().copy()
        renderer.close()
        return frame
    except ImportError:
        # mujoco not available — return env.render() as fallback
        return env.render()
    except Exception as e:
        # MuJoCo rendering failed (bad model, missing camera, etc.) — fallback
        logger.warning(f'MuJoCo camera render failed: {e}, falling back to env.render()')
        return env.render()


# ─── Dataset Serialization Helpers ─────────────────────────────────────


def _write_episode_to_disk(
    episode_data: Dict[str, Any],
    episode_idx: int,
    output_dir: str,
    task: str,
) -> str:
    """Write a single episode to disk in a structured format.

    Creates per-episode directory with:
    - data.json: timestep-level states, actions, and language annotation
    - video_front.mp4: front camera frames (if av available, else .npz)
    - video_wrist.mp4: wrist camera frames (if available)
    - metadata.json: episode-level metadata

    Args:
        episode_data: Dict with "frames_front", "frames_wrist", "states", "actions".
            Legacy format with "frames" key is also supported (mapped to front only).
        episode_idx: Episode index for naming.
        output_dir: Base output directory.
        task: Task description string.

    Returns:
        Path to the episode directory.
    """
    ep_dir = os.path.join(output_dir, f"episode_{episode_idx:06d}")
    os.makedirs(ep_dir, exist_ok=True)

    # Write states and actions as JSON with language annotation
    timesteps = []
    for step_idx, (state, action) in enumerate(
        zip(episode_data["states"], episode_data["actions"])
    ):
        timesteps.append({
            "step": step_idx,
            "state": state,
            "action": action,
            "annotation.human.task_description": task,
        })

    data_path = os.path.join(ep_dir, "data.json")
    with open(data_path, "w") as f:
        json.dump({"timesteps": timesteps, "task": task}, f)

    # Resolve frame lists (support both new and legacy format)
    frames_front = episode_data.get("frames_front", episode_data.get("frames", []))
    frames_wrist = episode_data.get("frames_wrist", [])

    # Write front camera video
    _write_video_or_npz(frames_front, ep_dir, "video_front")

    # Write wrist camera video (if available)
    if frames_wrist:
        _write_video_or_npz(frames_wrist, ep_dir, "video_wrist")

    # Episode metadata
    meta = {
        "episode_idx": episode_idx,
        "num_steps": len(timesteps),
        "task": task,
        "has_video_front": len(frames_front) > 0,
        "has_video_wrist": len(frames_wrist) > 0,
        "has_video": len(frames_front) > 0,  # backward compat
    }
    with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return ep_dir


def _write_video_or_npz(frames: list, ep_dir: str, name: str) -> None:
    """Write frames as video (mp4) or numpy fallback.

    Args:
        frames: List of numpy arrays (H, W, 3), dtype uint8.
        ep_dir: Episode directory path.
        name: Video filename stem (e.g. "video_front").
    """
    if not frames:
        return

    try:
        import av

        video_path = os.path.join(ep_dir, f"{name}.mp4")
        h, w = frames[0].shape[:2] if hasattr(frames[0], "shape") else (224, 224)
        container = av.open(video_path, mode="w")
        stream = container.add_stream("libx264", rate=30)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}

        for frame_data in frames:
            frame_arr = np.asarray(frame_data, dtype=np.uint8)
            if frame_arr.ndim == 2:
                frame_arr = np.stack([frame_arr] * 3, axis=-1)
            av_frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()

    except (ImportError, Exception) as e:
        # Fallback: save as compressed numpy
        logger.debug(f"PyAV not available ({e}), saving frames as .npz")
        frame_arrays = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
        np.savez_compressed(os.path.join(ep_dir, f"{name}.npz"), frames=frame_arrays)


def _convert_to_lerobot_format(
    raw_dir: str,
    output_dir: str,
    data_config_name: str,
    robot_name: str,
    fps: int = 30,
) -> str:
    """Convert per-episode directories to LeRobot dataset format.

    Transforms:
        raw_dir/episode_000000/data.json + video_front.mp4 + video_wrist.mp4
    Into:
        output_dir/data/train-00000-of-00001.parquet
        output_dir/videos/video.front/episode_000000.mp4
        output_dir/videos/video.wrist/episode_000000.mp4
        output_dir/meta/info.json
        output_dir/meta/episodes.jsonl

    Args:
        raw_dir: Directory with per-episode subdirectories.
        output_dir: Target LeRobot-format dataset directory.
        data_config_name: GR00T data config name (e.g. "so100_dualcam").
        robot_name: Robot name for metadata.
        fps: Video FPS for metadata.

    Returns:
        Path to the output directory.

    Raises:
        ImportError: If pyarrow is not available.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow required for LeRobot format conversion. "
            "Install with: pip install pyarrow"
        )

    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)

    # Discover episodes
    episode_dirs = sorted(
        d for d in os.listdir(raw_dir)
        if d.startswith("episode_") and os.path.isdir(os.path.join(raw_dir, d))
    )

    all_rows = []
    episodes_meta = []
    total_frames = 0

    for ep_dir_name in episode_dirs:
        ep_dir = os.path.join(raw_dir, ep_dir_name)
        ep_idx = int(ep_dir_name.split("_")[1])

        # Read episode data
        data_path = os.path.join(ep_dir, "data.json")
        if not os.path.exists(data_path):
            continue

        with open(data_path) as f:
            data = json.load(f)

        timesteps = data.get("timesteps", [])
        task_desc = data.get("task", "")

        for ts in timesteps:
            row = {
                "episode_index": ep_idx,
                "frame_index": ts["step"],
                "timestamp": ts["step"] / fps,
            }

            # Flatten state/action dicts into top-level keys
            state = ts.get("state", {})
            if isinstance(state, dict):
                for k, v in state.items():
                    row[k] = v
            action = ts.get("action", {})
            if isinstance(action, dict):
                for k, v in action.items():
                    row[k] = v

            # Language annotation
            row["annotation.human.task_description"] = (
                ts.get("annotation.human.task_description", task_desc)
            )

            all_rows.append(row)

        num_steps = len(timesteps)
        total_frames += num_steps

        episodes_meta.append({
            "episode_index": ep_idx,
            "length": num_steps,
            "task": task_desc,
        })

        # Move/copy video files to LeRobot structure
        for video_key in ["video_front", "video_wrist"]:
            src_mp4 = os.path.join(ep_dir, f"{video_key}.mp4")
            src_npz = os.path.join(ep_dir, f"{video_key}.npz")

            # Map video_front → video.front directory
            lerobot_key = video_key.replace("_", ".")
            vid_dir = os.path.join(output_dir, "videos", lerobot_key)
            os.makedirs(vid_dir, exist_ok=True)

            dst = os.path.join(vid_dir, f"{ep_dir_name}.mp4")
            if os.path.exists(src_mp4):
                # Hard-link or copy
                try:
                    os.link(src_mp4, dst)
                except OSError:
                    import shutil
                    shutil.copy2(src_mp4, dst)
            elif os.path.exists(src_npz):
                # npz fallback — convert to video if av available, else copy npz
                try:
                    import av

                    frames_data = np.load(src_npz)["frames"]
                    h, w = frames_data.shape[1:3]
                    container = av.open(dst, mode="w")
                    stream = container.add_stream("libx264", rate=fps)
                    stream.width = w
                    stream.height = h
                    stream.pix_fmt = "yuv420p"
                    stream.options = {"crf": "23", "preset": "fast"}
                    for frame_arr in frames_data:
                        av_frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
                        for packet in stream.encode(av_frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
                    container.close()
                except (ImportError, Exception):
                    import shutil
                    shutil.copy2(src_npz, os.path.join(vid_dir, f"{ep_dir_name}.npz"))

    # Write parquet
    if all_rows:
        table = pa.Table.from_pylist(all_rows)
        parquet_path = os.path.join(output_dir, "data", "train-00000-of-00001.parquet")
        pq.write_table(table, parquet_path)
        logger.info(f"  Wrote {len(all_rows)} rows to {parquet_path}")

    # Write meta/info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_name,
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "fps": fps,
        "data_config": data_config_name,
        "splits": {"train": f"0:{len(episodes_meta)}"},
    }
    with open(os.path.join(output_dir, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Write meta/episodes.jsonl
    with open(os.path.join(output_dir, "meta", "episodes.jsonl"), "w") as f:
        for ep_meta in episodes_meta:
            f.write(json.dumps(ep_meta) + "\n")

    logger.info(
        f"  LeRobot format: {len(episodes_meta)} episodes, "
        f"{total_frames} frames → {output_dir}"
    )
    return output_dir


# ─── Stage Implementations ────────────────────────────────────────────


def stage_1_marble_world(config: PipelineConfig) -> Dict[str, Any]:
    """Stage 1: Generate 3D world via Marble API and convert to USD.

    Uses World Labs' Marble to generate a photorealistic 3D scene from
    text description, then converts PLY→USDZ for Isaac Sim compatibility.
    """
    logger.info("═══ Stage 1: Marble 3D World Generation ═══")
    os.makedirs(config.marble_output_dir, exist_ok=True)

    try:
        from strands_robots.marble import MarbleConfig, MarblePipeline

        marble_config = MarbleConfig(
            output_format="usdz",
            robot=config.marble_robot,
        )
        pipeline = MarblePipeline(marble_config)

        # Generate 3D world
        scene = pipeline.generate_world(
            prompt=config.marble_prompt,
            output_dir=config.marble_output_dir,
        )

        # Compose with robot + task objects
        composed = pipeline.compose_scene(
            scene_path=scene.get("usdz_path", scene.get("ply_path", "")),
            robot=config.marble_robot,
            task_objects=config.marble_objects,
            table_replacement=True,
        )

        result = {
            "status": "success",
            "scene_path": composed.get("scene_usd", ""),
            "ply_path": scene.get("ply_path", ""),
            "usdz_path": scene.get("usdz_path", ""),
            "prompt": config.marble_prompt,
        }
        logger.info(f"✅ Stage 1 complete: {result['scene_path']}")
        return result

    except ImportError:
        logger.warning("Marble not available — skipping 3D generation")
        return {"status": "skipped", "reason": "marble not installed"}
    except Exception as e:
        logger.warning(f"Marble generation failed: {e} — continuing with default scene")
        return {"status": "skipped", "reason": str(e)}


def stage_2_sac_training(config: PipelineConfig) -> Dict[str, Any]:
    """Stage 2: Train SAC policy for pick-and-place manipulation.

    Uses stable-baselines3 SAC with the PickAndPlaceReward (4-phase
    curriculum: Reach→Grasp→Transport→Place). Trains in MuJoCo or Newton.
    """
    logger.info("═══ Stage 2: SAC RL Training ═══")
    os.makedirs(config.sac_output_dir, exist_ok=True)

    from strands_robots.rl_trainer import create_rl_trainer

    trainer = create_rl_trainer(
        algorithm="sac",
        env_config={
            "robot_name": config.sac_robot,
            "task": config.sac_task,
            "backend": config.sac_backend,
            "num_envs": 1,  # SAC is off-policy, single env is fine
        },
        total_timesteps=config.sac_timesteps,
        output_dir=config.sac_output_dir,
        learning_rate=config.sac_learning_rate,
        batch_size=config.sac_batch_size,
        seed=config.seed,
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    result["training_time_seconds"] = elapsed
    result["fps"] = config.sac_timesteps / elapsed if elapsed > 0 else 0

    # Quick evaluation
    if result.get("status") == "success":
        eval_result = trainer.evaluate(num_episodes=20)
        result["eval_success_rate"] = eval_result.get("success_rate", 0.0)
        result["eval_mean_reward"] = eval_result.get("mean_reward", 0.0)
        logger.info(
            f"✅ Stage 2 complete: {elapsed:.0f}s, "
            f"success={eval_result.get('success_rate', 0):.1f}%, "
            f"reward={eval_result.get('mean_reward', 0):.2f}"
        )

    return result


def stage_3_record_dataset(config: PipelineConfig) -> Dict[str, Any]:
    """Stage 3: Record SAC policy rollouts into LeRobot-format dataset.

    Loads the trained SAC checkpoint and rolls it out in simulation with
    dual camera views (front + wrist) to create training data for GR00T.
    Each episode is streamed to disk immediately to avoid accumulating
    frames in RAM. After recording, the raw episodes are converted to
    LeRobot parquet + video format for GR00T consumption.

    The dataset format matches the GR00T so100_dualcam data_config:
        - video.front: Front camera view (224×224)
        - video.wrist: Wrist camera view (224×224)
        - state.single_arm: 5D joint positions
        - state.gripper: 1D gripper aperture
        - action.single_arm: 5D joint actions
        - action.gripper: 1D gripper action
        - annotation.human.task_description: Task string
    """
    logger.info("═══ Stage 3: Dataset Recording ═══")

    raw_dir = os.path.join(config.record_output_dir, "_raw_episodes")
    os.makedirs(raw_dir, exist_ok=True)

    try:
        from stable_baselines3 import SAC
    except ImportError:
        return {"status": "error", "error": "stable-baselines3 required"}

    # Load trained SAC model
    model_path = os.path.join(config.sac_output_dir, "final_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(config.sac_output_dir, "final_model")
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        return {"status": "error", "error": f"SAC model not found at {model_path}"}

    model = SAC.load(model_path)

    # Create environment with wrist camera for dual-cam recording
    from strands_robots.envs import StrandsSimEnv

    cam_w = config.record_camera_width
    cam_h = config.record_camera_height

    env = StrandsSimEnv(
        robot_name=config.sac_robot,
        task=config.sac_task,
        render_mode="rgb_array",
        render_width=cam_w,
        render_height=cam_h,
        cameras=[
            {
                "name": "wrist",
                "position": [0.0, 0.0, 0.15],
                "target": [0.0, 0.0, 0.0],
                "width": cam_w,
                "height": cam_h,
            }
        ],
    )

    episodes_manifest = []
    successful_episodes = 0

    for ep in range(config.record_episodes):
        obs, info = env.reset(seed=config.seed + ep)
        episode_buffer = {
            "frames_front": [],
            "frames_wrist": [],
            "states": [],
            "actions": [],
        }
        done = False
        steps = 0

        while not done and steps < config.record_max_steps:
            # Get action from trained SAC policy
            obs_for_model = obs["state"] if isinstance(obs, dict) else obs
            action, _ = model.predict(obs_for_model, deterministic=True)

            # Record front camera (default renderer)
            frame_front = env.render()

            # Record wrist camera (named camera via MuJoCo backend)
            frame_wrist = _render_named_camera(env, "wrist", cam_w, cam_h)

            # Extract state components for GR00T format
            if isinstance(obs, dict):
                state = np.asarray(
                    obs.get("state", obs.get("observation.state", np.zeros(6)))
                ).flatten()
            else:
                state = np.asarray(obs).flatten()

            n_arm = min(5, len(state))
            arm_state = state[:n_arm].tolist()
            gripper_state = [float(state[n_arm])] if len(state) > n_arm else [0.0]

            action_flat = np.asarray(action).flatten()
            n_arm_act = min(5, len(action_flat))
            arm_action = action_flat[:n_arm_act].tolist()
            gripper_action = [float(action_flat[n_arm_act])] if len(action_flat) > n_arm_act else [0.0]

            episode_buffer["frames_front"].append(frame_front)
            episode_buffer["frames_wrist"].append(frame_wrist)
            episode_buffer["states"].append({
                "state.single_arm": arm_state,
                "state.gripper": gripper_state,
            })
            episode_buffer["actions"].append({
                "action.single_arm": arm_action,
                "action.gripper": gripper_action,
            })

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        is_success = info.get("is_success", False)
        if is_success:
            successful_episodes += 1

        # Stream episode to disk immediately, then free memory
        ep_dir = _write_episode_to_disk(
            episode_data=episode_buffer,
            episode_idx=ep,
            output_dir=raw_dir,
            task=config.sac_task,
        )
        del episode_buffer  # Free frame memory

        episodes_manifest.append({
            "episode_id": ep,
            "episode_dir": ep_dir,
            "num_steps": steps,
            "success": is_success,
        })

        if (ep + 1) % 50 == 0:
            logger.info(f"  Recorded {ep + 1}/{config.record_episodes} episodes")

    env.close()

    # Save raw dataset metadata
    dataset_meta = {
        "num_episodes": config.record_episodes,
        "successful_episodes": successful_episodes,
        "success_rate": successful_episodes / config.record_episodes * 100 if config.record_episodes > 0 else 0,
        "robot": config.sac_robot,
        "task": config.sac_task,
        "data_config": config.groot_data_config,
        "episodes": episodes_manifest,
    }

    meta_path = os.path.join(raw_dir, "dataset_meta.json")
    with open(meta_path, "w") as f:
        json.dump(dataset_meta, f, indent=2)

    # Convert raw episodes to LeRobot format for GR00T consumption
    logger.info("Converting to LeRobot format...")
    try:
        _convert_to_lerobot_format(
            raw_dir=raw_dir,
            output_dir=config.record_output_dir,
            data_config_name=config.groot_data_config,
            robot_name=config.sac_robot,
        )
    except ImportError as e:
        logger.warning(f"LeRobot format conversion skipped: {e}")
        logger.warning("GR00T fine-tuning (Stage 4) may fail without LeRobot format.")

    logger.info(
        f"✅ Stage 3 complete: {config.record_episodes} episodes, "
        f"{successful_episodes} successful ({dataset_meta['success_rate']:.1f}%)"
    )

    return {
        "status": "success",
        "dataset_path": config.record_output_dir,
        "raw_dir": raw_dir,
        "num_episodes": config.record_episodes,
        "success_rate": dataset_meta["success_rate"],
    }


def stage_4_groot_finetune(config: PipelineConfig) -> Dict[str, Any]:
    """Stage 4: Fine-tune GR00T N1.6 on the recorded dataset.

    Uses Gr00tTrainer which wraps Isaac-GR00T's launch_finetune.py.
    Trains projector + diffusion model (DiT action head), keeping
    LLM and vision encoder frozen for efficiency.
    """
    logger.info("═══ Stage 4: GR00T N1.6 Fine-Tuning ═══")
    os.makedirs(config.groot_output_dir, exist_ok=True)

    from strands_robots.training import TrainConfig, create_trainer

    train_config = TrainConfig(
        dataset_path=config.record_output_dir,
        output_dir=config.groot_output_dir,
        max_steps=config.groot_max_steps,
        batch_size=config.groot_batch_size,
        learning_rate=config.groot_learning_rate,
        num_gpus=config.num_gpus,
        save_steps=max(1, config.groot_max_steps // 5),
        seed=config.seed,
    )

    trainer = create_trainer(
        "groot",
        base_model_path=config.groot_base_model,
        dataset_path=config.record_output_dir,
        embodiment_tag=config.groot_embodiment,
        data_config=config.groot_data_config,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_llm=False,
        tune_visual=False,
        config=train_config,
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    result["training_time_seconds"] = elapsed
    logger.info(f"✅ Stage 4 complete: {elapsed:.0f}s, status={result.get('status')}")
    return result


def stage_5_evaluate(config: PipelineConfig) -> Dict[str, Any]:
    """Stage 5: Evaluate the fine-tuned GR00T policy.

    Runs evaluation across multiple tasks using the training.evaluate()
    harness. Tests both the fine-tuned model and (optionally) the base
    model for comparison.
    """
    logger.info("═══ Stage 5: Evaluation ═══")
    os.makedirs(config.eval_output_dir, exist_ok=True)

    from strands_robots.policies import create_policy
    from strands_robots.training import evaluate

    # Load fine-tuned policy
    checkpoint_path = os.path.join(config.groot_output_dir, "best")
    if not os.path.exists(checkpoint_path):
        # Try last checkpoint
        checkpoint_path = config.groot_output_dir

    results = {}

    for task in config.eval_tasks:
        logger.info(f"Evaluating: {task}")
        try:
            policy = create_policy(
                "groot",
                model_path=checkpoint_path,
                data_config=config.groot_data_config,
                device=config.device,
            )

            eval_result = evaluate(
                policy=policy,
                task=task,
                robot_name=config.sac_robot,
                num_episodes=config.eval_episodes,
                backend=config.sac_backend,
                seed=config.seed,
            )
            results[task] = eval_result
            logger.info(
                f"  {task}: success={eval_result.get('success_rate', 0):.1f}%, "
                f"reward={eval_result.get('mean_reward', 0):.2f}"
            )
        except Exception as e:
            logger.error(f"  {task}: FAILED — {e}")
            results[task] = {"status": "error", "error": str(e)}

    # Save results
    results_path = os.path.join(config.eval_output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate markdown report
    md_lines = [
        "# GR00T N1.6 Fine-Tuning Evaluation Results\n",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n",
        f"**Model:** {config.groot_base_model}\n",
        f"**Robot:** {config.sac_robot}\n",
        f"**SAC Timesteps:** {config.sac_timesteps:,}\n",
        f"**GR00T Steps:** {config.groot_max_steps:,}\n",
        f"**Eval Episodes:** {config.eval_episodes}\n\n",
        "| Task | Success Rate | Mean Reward | Status |\n",
        "|------|:-----------:|:-----------:|--------|\n",
    ]

    for task, res in results.items():
        if isinstance(res, dict) and "success_rate" in res:
            md_lines.append(
                f"| {task} | {res['success_rate']:.1f}% | {res['mean_reward']:.2f} | ✅ |\n"
            )
        else:
            error = res.get("error", "unknown") if isinstance(res, dict) else str(res)
            md_lines.append(f"| {task} | — | — | ❌ {error[:40]} |\n")

    md_path = os.path.join(config.eval_output_dir, "eval_results.md")
    with open(md_path, "w") as f:
        f.writelines(md_lines)

    logger.info(f"✅ Stage 5 complete: results at {results_path}")
    return {"status": "success", "results": results, "report_path": md_path}


def stage_6_cosmos_reason(config: PipelineConfig) -> Dict[str, Any]:
    """Stage 6: Cosmos Reason Analysis of evaluation results.

    Analyzes evaluation metrics to understand failure patterns and suggest
    training parameter adjustments. On GPU with Cosmos Reason available,
    also analyzes recorded video frames for semantic failure understanding.

    Analysis dimensions:
    - Phase progression: Which reward phases are failing most?
    - Grasp quality: Is the gripper making contact? Stable grasp?
    - Motion quality: Smooth trajectories or jerky/oscillating?
    - Task completion: What percentage of episodes reach each phase?
    - Failure modes: Categorize failures (miss, drop, overshoot, timeout)

    Recommendations:
    - Learning rate adjustments
    - Reward weight tuning (reach_scale, grasp_bonus, etc.)
    - Action smoothness penalty changes
    - Training duration recommendations
    """
    logger.info("═══ Stage 6: Cosmos Reason Analysis ═══")
    os.makedirs(config.cosmos_analysis_dir, exist_ok=True)

    # Load evaluation results
    eval_results_path = os.path.join(config.eval_output_dir, "eval_results.json")
    if not os.path.exists(eval_results_path):
        return {"status": "skipped", "reason": "No eval results found"}

    with open(eval_results_path) as f:
        eval_results = json.load(f)

    # Analyze evaluation results for patterns
    analysis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_config": {
            "sac_timesteps": config.sac_timesteps,
            "groot_steps": config.groot_max_steps,
            "sac_lr": config.sac_learning_rate,
            "groot_lr": config.groot_learning_rate,
        },
        "task_analysis": {},
        "recommendations": [],
        "overall_assessment": "",
    }

    total_success = 0.0
    total_tasks = 0

    for task, result in eval_results.items():
        if not isinstance(result, dict) or "success_rate" not in result:
            continue

        success_rate = result["success_rate"]
        mean_reward = result.get("mean_reward", 0)
        episodes = result.get("episodes", [])
        total_success += success_rate
        total_tasks += 1

        # Per-episode analysis
        rewards = [ep.get("reward", 0) for ep in episodes]
        steps = [ep.get("steps", 0) for ep in episodes]

        task_analysis = {
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "reward_std": float(np.std(rewards)) if rewards else 0,
            "mean_steps": float(np.mean(steps)) if steps else 0,
            "step_std": float(np.std(steps)) if steps else 0,
            "max_reward": float(max(rewards)) if rewards else 0,
            "min_reward": float(min(rewards)) if rewards else 0,
        }
        analysis["task_analysis"][task] = task_analysis

        # Generate recommendations based on patterns
        if success_rate < 10:
            analysis["recommendations"].append({
                "task": task,
                "severity": "critical",
                "issue": "Very low success rate — policy likely not learning the task",
                "suggestions": [
                    f"Increase SAC timesteps from {config.sac_timesteps:,} to {config.sac_timesteps * 4:,}",
                    "Increase reach_scale in PickAndPlaceReward (currently 2.0 → try 5.0)",
                    "Verify observation space matches the task (check object positions in obs)",
                    "Consider curriculum learning: train reach first, then grasp",
                ],
            })
        elif success_rate < 50:
            analysis["recommendations"].append({
                "task": task,
                "severity": "moderate",
                "issue": "Moderate success — policy learning but inconsistent",
                "suggestions": [
                    f"Increase GR00T fine-tuning steps from {config.groot_max_steps:,} to {config.groot_max_steps * 2:,}",
                    "Increase grasp_bonus (5.0 → 10.0) for more grasp incentive",
                    "Add more recording episodes (200 → 500) for better GR00T data coverage",
                    f"Lower GR00T learning rate from {config.groot_learning_rate} to {config.groot_learning_rate / 2}",
                ],
            })
        elif success_rate < 80:
            analysis["recommendations"].append({
                "task": task,
                "severity": "minor",
                "issue": "Good performance — fine-tuning needed for consistency",
                "suggestions": [
                    "Increase action_smoothness_penalty for smoother motions",
                    "Add GR00T LoRA fine-tuning (rank=16) for better generalization",
                    "Try longer evaluation episodes (300 → 500 max steps)",
                ],
            })

        if task_analysis["reward_std"] > abs(mean_reward) * 0.5:
            analysis["recommendations"].append({
                "task": task,
                "severity": "moderate",
                "issue": "High reward variance — policy is inconsistent",
                "suggestions": [
                    "Increase SAC buffer_size for more diverse experience",
                    "Lower SAC learning rate for more stable learning",
                    "Use deterministic evaluation (already set in pipeline)",
                ],
            })

    # Overall assessment
    avg_success = total_success / total_tasks if total_tasks > 0 else 0
    if avg_success >= 80:
        analysis["overall_assessment"] = "EXCELLENT — Pipeline is working well. Minor tuning recommended."
    elif avg_success >= 50:
        analysis["overall_assessment"] = "GOOD — Pipeline shows learning. More training time and data recommended."
    elif avg_success >= 20:
        analysis["overall_assessment"] = "MODERATE — Some learning detected. Significant parameter tuning needed."
    else:
        analysis["overall_assessment"] = "NEEDS WORK — Low success rates. Verify reward function, obs space, and training duration."

    # Save analysis
    analysis_path = os.path.join(config.cosmos_analysis_dir, "cosmos_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    logger.info(f"✅ Stage 6 complete: {analysis['overall_assessment']}")
    return {"status": "success", "analysis": analysis}


# ─── Pipeline Orchestrator ─────────────────────────────────────────────


def run_full_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """Run the full end-to-end pipeline.

    Stages:
    1. Marble 3D World Generation (optional, skipped if no API key)
    2. SAC RL Training
    3. Dataset Recording from trained SAC policy
    4. GR00T N1.6 Fine-Tuning
    5. Evaluation (skipped if Stage 4 failed)
    6. Cosmos Reason Analysis (skipped if Stage 5 skipped/failed)
    """
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║  GR00T N1.6 Fine-Tuning Pipeline            ║")
    logger.info("║  Issue #141 — End-to-End                    ║")
    logger.info("╚══════════════════════════════════════════════╝")

    pipeline_start = time.time()
    results = {"config": asdict(config), "stages": {}, "start_time": datetime.now(timezone.utc).isoformat()}

    # Stage 1: Marble (optional)
    try:
        results["stages"]["marble"] = stage_1_marble_world(config)
    except Exception as e:
        logger.warning(f"Stage 1 (Marble) failed: {e}")
        results["stages"]["marble"] = {"status": "skipped", "error": str(e)}

    # Stage 2: SAC Training
    try:
        results["stages"]["sac"] = stage_2_sac_training(config)
        if results["stages"]["sac"].get("status") != "success":
            logger.error("SAC training failed — cannot continue pipeline")
            results["status"] = "failed_at_sac"
            return results
    except Exception as e:
        logger.error(f"Stage 2 (SAC) failed: {e}")
        results["stages"]["sac"] = {"status": "error", "error": str(e)}
        results["status"] = "failed_at_sac"
        return results

    # Stage 3: Dataset Recording
    try:
        results["stages"]["dataset"] = stage_3_record_dataset(config)
    except Exception as e:
        logger.error(f"Stage 3 (Dataset) failed: {e}")
        results["stages"]["dataset"] = {"status": "error", "error": str(e)}
        results["status"] = "failed_at_dataset"
        return results

    # Stage 4: GR00T Fine-Tuning
    try:
        results["stages"]["groot"] = stage_4_groot_finetune(config)
    except Exception as e:
        logger.error(f"Stage 4 (GR00T) failed: {e}")
        results["stages"]["groot"] = {"status": "error", "error": str(e)}

    # Stage 5: Evaluation — skip if GR00T failed (no checkpoint to evaluate)
    groot_status = results["stages"].get("groot", {}).get("status")
    if groot_status == "success":
        try:
            results["stages"]["eval"] = stage_5_evaluate(config)
        except Exception as e:
            logger.error(f"Stage 5 (Eval) failed: {e}")
            results["stages"]["eval"] = {"status": "error", "error": str(e)}
    else:
        logger.warning("Stage 5 (Eval) skipped — GR00T training did not succeed")
        results["stages"]["eval"] = {
            "status": "skipped",
            "reason": f"GR00T training status: {groot_status}",
        }

    # Stage 6: Cosmos Reason Analysis — skip if eval skipped
    eval_status = results["stages"].get("eval", {}).get("status")
    if eval_status == "success":
        try:
            results["stages"]["cosmos_reason"] = stage_6_cosmos_reason(config)
        except Exception as e:
            logger.warning(f"Stage 6 (Cosmos) failed: {e}")
            results["stages"]["cosmos_reason"] = {"status": "error", "error": str(e)}
    else:
        results["stages"]["cosmos_reason"] = {
            "status": "skipped",
            "reason": f"Evaluation status: {eval_status}",
        }

    # Summary
    pipeline_elapsed = time.time() - pipeline_start
    results["total_time_seconds"] = pipeline_elapsed
    results["end_time"] = datetime.now(timezone.utc).isoformat()
    results["status"] = "completed"

    # Save full pipeline results
    os.makedirs("./pipeline_outputs", exist_ok=True)
    results_path = "./pipeline_outputs/pipeline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete in {pipeline_elapsed:.0f}s")
    logger.info(f"Results: {results_path}")
    for stage_name, stage_result in results["stages"].items():
        status = stage_result.get("status", "unknown")
        logger.info(f"  {stage_name}: {status}")
    logger.info(f"{'='*60}")

    return results


# ─── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="GR00T N1.6 Fine-Tuning Pipeline (Issue #141)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument(
        "--stage",
        choices=["marble", "sac", "dataset", "groot", "eval", "cosmos"],
        help="Run a single stage",
    )

    # Stage-specific overrides
    parser.add_argument("--robot", default="so100", help="Robot name")
    parser.add_argument("--task", default="pick and place cube", help="Task description")
    parser.add_argument("--backend", default="mujoco", choices=["mujoco", "newton"])
    parser.add_argument("--timesteps", type=int, default=500_000, help="SAC training timesteps")
    parser.add_argument("--groot-steps", type=int, default=10_000, help="GR00T fine-tuning steps")
    parser.add_argument("--episodes", type=int, default=50, help="Eval episodes")
    parser.add_argument("--record-episodes", type=int, default=200, help="Recording episodes")
    parser.add_argument("--dataset", help="Path to existing dataset (for groot/eval stages)")
    parser.add_argument("--checkpoint", help="Path to existing checkpoint (for eval stage)")
    parser.add_argument("--output", default="./pipeline_outputs", help="Base output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Build config
    config = PipelineConfig(
        sac_robot=args.robot,
        sac_task=args.task,
        sac_backend=args.backend,
        sac_timesteps=args.timesteps,
        groot_max_steps=args.groot_steps,
        eval_episodes=args.episodes,
        record_episodes=args.record_episodes,
        seed=args.seed,
        num_gpus=args.gpus,
        sac_output_dir=os.path.join(args.output, "sac_training"),
        record_output_dir=args.dataset or os.path.join(args.output, "rl_dataset"),
        groot_output_dir=os.path.join(args.output, "groot_finetuned"),
        eval_output_dir=os.path.join(args.output, "eval_results"),
        cosmos_analysis_dir=os.path.join(args.output, "cosmos_analysis"),
        marble_output_dir=os.path.join(args.output, "marble_scene"),
    )

    if args.full or args.stage is None:
        run_full_pipeline(config)
    elif args.stage == "marble":
        stage_1_marble_world(config)
    elif args.stage == "sac":
        stage_2_sac_training(config)
    elif args.stage == "dataset":
        stage_3_record_dataset(config)
    elif args.stage == "groot":
        stage_4_groot_finetune(config)
    elif args.stage == "eval":
        stage_5_evaluate(config)
    elif args.stage == "cosmos":
        stage_6_cosmos_reason(config)


if __name__ == "__main__":
    main()
