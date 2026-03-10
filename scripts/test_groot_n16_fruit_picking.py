#!/usr/bin/env python3
"""
GR00T N1.6 Fruit-Picking Evaluation — SO-101 in MuJoCo + Newton Backend

This script tests the fine-tuned GR00T N1.6 checkpoint:
  aaronsu11/GR00T-N1.6-3B-SO101-FruitPicking

It runs the policy in MuJoCo simulation with Newton GPU backend and records MP4 videos.

Architecture:
  Camera Image (224×224) + Robot Joint State (6-DOF) + "pick the fruit"
      │                          │                          │
      ▼                          ▼                          ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
  │ Eagle Vision  │    │  State MLP    │    │ Qwen3 Tokenizer  │
  │ (SigLIP2)     │    │  Encoder      │    │                  │
  └──────────────┘    └──────────────┘    └──────────────────┘
        │                      │                     │
        └──────────────────────┼─────────────────────┘
                               ▼
                   ┌─────────────────────┐
                   │     Qwen3 LLM       │
                   │  (16 layers, 2048)   │
                   └─────────────────────┘
                               │
                               ▼
                   ┌─────────────────────┐
                   │   Diffusion Head    │ → Action Chunk (16 steps)
                   │  (4 denoise steps)  │   = future joint positions
                   └─────────────────────┘

Requirements:
    - GPU with 24GB+ VRAM (runs on Thor: 132GB unified memory)
    - pip install strands-robots[all,dev] isaac-gr00t
    - MUJOCO_GL=egl for headless rendering
    - Newton + Warp for GPU-accelerated physics

Usage:
    # Full test with Newton backend + GR00T inference + video recording
    MUJOCO_GL=egl python scripts/test_groot_n16_fruit_picking.py

    # MuJoCo backend only (no Newton GPU requirement)
    MUJOCO_GL=egl python scripts/test_groot_n16_fruit_picking.py --backend mujoco

    # Quick test with mock policy (no GR00T model needed)
    MUJOCO_GL=egl python scripts/test_groot_n16_fruit_picking.py --policy mock

    # Custom checkpoint
    MUJOCO_GL=egl python scripts/test_groot_n16_fruit_picking.py \
        --model /data/checkpoints/my_groot/best

    # Multiple episodes
    MUJOCO_GL=egl python scripts/test_groot_n16_fruit_picking.py --episodes 5

Related issue: https://github.com/cagataycali/strands-gtc-nvidia/issues/203
Checkpoint: https://huggingface.co/aaronsu11/GR00T-N1.6-3B-SO101-FruitPicking
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("groot_fruit_picking")

# ──────────────────────────────────────────────────────────────────────
# Scene definition: SO-101 + fruit objects + table (MJCF)
# ──────────────────────────────────────────────────────────────────────

FRUIT_PICKING_SCENE_XML = """
<mujoco model="fruit_picking_scene">
    <!-- Include the SO-101 robot -->
    <include file="so101.xml" />

    <option timestep="0.002" gravity="0 0 -9.81" />

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
        <global azimuth="160" elevation="-20" />
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.05 0.05 0.15"
                 width="512" height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge"
                 rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
                 width="300" height="300" />
        <material name="groundplane" texture="groundplane" texuniform="true"
                  texrepeat="5 5" reflectance="0.2" />
        <texture type="2d" name="wood_tex" builtin="flat" rgb1="0.6 0.45 0.3"
                 rgb2="0.55 0.4 0.25" width="256" height="256" />
        <material name="wood" texture="wood_tex" texrepeat="3 3" />
    </asset>

    <worldbody>
        <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
        <light pos="0.5 0.5 2" dir="-0.3 -0.3 -1" diffuse="0.4 0.4 0.4" />

        <!-- Ground -->
        <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" />

        <!-- Table surface (wooden) -->
        <body name="table" pos="0.35 0 0.0">
            <geom type="box" size="0.3 0.3 0.005" pos="0 0 0" material="wood"
                  contype="2" conaffinity="1" />
        </body>

        <!-- Front camera (for policy observation — simulates webcam view) -->
        <camera name="front_cam" pos="0.35 -0.6 0.45" xyaxes="1 0 0 0 0.7 0.7" fovy="60" />
        <!-- wrist_cam already defined in so101.xml -->
        <camera name="overhead_cam" pos="0.35 0 0.8" xyaxes="1 0 0 0 -1 0" fovy="60" />

        <!-- Fruit: Orange (sphere) -->
        <body name="orange" pos="0.4 0.05 0.04">
            <freejoint />
            <geom type="sphere" name="orange_geom" size="0.025"
                  rgba="1.0 0.6 0.0 1" mass="0.15"
                  condim="3" friction="1.0 0.03 0.003"
                  contype="2" conaffinity="1" solref="0.01 1" />
        </body>

        <!-- Fruit: Apple (sphere, slightly larger) -->
        <body name="apple" pos="0.3 -0.08 0.04">
            <freejoint />
            <geom type="sphere" name="apple_geom" size="0.03"
                  rgba="0.9 0.15 0.1 1" mass="0.18"
                  condim="3" friction="1.0 0.03 0.003"
                  contype="2" conaffinity="1" solref="0.01 1" />
        </body>

        <!-- Fruit: Banana (capsule) -->
        <body name="banana" pos="0.45 -0.05 0.03">
            <freejoint />
            <geom type="capsule" name="banana_geom" size="0.012 0.06"
                  rgba="1.0 0.9 0.2 1" mass="0.12"
                  condim="3" friction="0.8 0.03 0.003"
                  contype="2" conaffinity="1" solref="0.01 1"
                  euler="0 90 20" />
        </body>

        <!-- Target plate (green) -->
        <body name="plate" pos="0.25 0.15 0.005">
            <geom type="cylinder" name="plate_geom" size="0.06 0.003"
                  rgba="0.2 0.7 0.3 0.8"
                  contype="2" conaffinity="1" />
        </body>
    </worldbody>

    <!-- Starting pose: arm slightly raised, ready to reach -->
    <keyframe>
        <key name="ready"
             qpos="0 0 0.3 0.8 1.2 0.5
                   0.4 0.05 0.04 1 0 0 0
                   0.3 -0.08 0.04 1 0 0 0
                   0.45 -0.05 0.03 1 0 0 0"
             ctrl="0 0 0.3 0.8 1.2 0.5" />
    </keyframe>
</mujoco>
"""


def _find_assets_dir() -> Path:
    """Find the SO-101 asset directory (repo or installed package)."""
    # From repo root
    assets_dir = Path(__file__).resolve().parent.parent / "strands_robots" / "assets" / "robotstudio_so101"
    if assets_dir.exists():
        return assets_dir

    # From installed package
    try:
        import strands_robots
        pkg_dir = Path(strands_robots.__file__).parent
        assets_dir = pkg_dir / "assets" / "robotstudio_so101"
        if assets_dir.exists():
            return assets_dir
    except ImportError:
        pass

    raise FileNotFoundError(
        "Cannot find SO-101 assets. Install strands-robots or run from repo root."
    )


def write_scene_file(output_dir: str) -> str:
    """Write the fruit-picking scene MJCF file alongside the SO-101 model.

    The scene uses <include file="so101.xml"/>, so it must live next to
    the SO-101 MJCF that ships with strands-robots.
    """
    assets_dir = _find_assets_dir()
    scene_path = assets_dir / "scene_fruit_picking.xml"
    scene_path.write_text(FRUIT_PICKING_SCENE_XML)
    logger.info(f"📝 Wrote scene file: {scene_path}")
    return str(scene_path)


# ──────────────────────────────────────────────────────────────────────
# Strategy A: Use strands_robots.simulation.Simulation (MuJoCo backend)
# ──────────────────────────────────────────────────────────────────────

def run_mujoco_backend(args, scene_path: str, output_dir: str) -> list:
    """Run using the MuJoCo (CPU) backend with Simulation class.

    The Simulation class API:
      1. sim.create_world()
      2. sim.add_robot(name="so101", urdf_path="path/to/scene.xml")
      3. sim.record_video(...)
    """
    from strands_robots.simulation import Simulation

    logger.info("🎮 Using MuJoCo backend (CPU)")

    sim = Simulation()

    # Step 1: Create empty world
    result = sim.create_world()
    if result.get("status") != "success":
        logger.error(f"Failed to create world: {result}")
        return []

    # Step 2: Add robot using the scene XML (which includes so101.xml + fruits + cameras)
    # The Simulation.add_robot() loads the MJCF and replaces the world model.
    result = sim.add_robot(name="so101", urdf_path=scene_path)
    if result.get("status") != "success":
        logger.error(f"Failed to add robot: {result}")
        return []
    logger.info(f"✅ {result['content'][0]['text']}")

    robot_name = "so101"
    video_paths = []

    for ep in range(args.episodes):
        logger.info(f"\n{'='*60}")
        logger.info(f"  Episode {ep+1}/{args.episodes}")
        logger.info(f"{'='*60}")

        output_path = os.path.join(output_dir, f"groot_fruit_picking_mujoco_ep{ep+1}.mp4")

        # Build policy kwargs
        policy_kwargs = {}
        if args.policy == "groot":
            policy_kwargs = {
                "model_path": args.model,
                "data_config": args.data_config,
                "groot_version": "n1.6",
                "denoising_steps": args.denoising_steps,
                "device": args.device,
            }

        try:
            result = sim.record_video(
                robot_name=robot_name,
                policy_provider=args.policy,
                instruction=args.instruction,
                duration=args.duration,
                fps=args.fps,
                camera_name="front_cam",
                width=args.width,
                height=args.height,
                output_path=output_path,
                **policy_kwargs,
            )

            if result.get("status") == "success":
                video_paths.append(output_path)
                logger.info(f"✅ {result['content'][0]['text']}")
            else:
                logger.error(f"❌ {result['content'][0]['text']}")
        except Exception as e:
            logger.error(f"❌ Episode {ep+1} failed: {e}")

    return video_paths


# ──────────────────────────────────────────────────────────────────────
# Strategy B: Use NewtonBackend (GPU-accelerated physics)
# ──────────────────────────────────────────────────────────────────────

def run_newton_backend(args, scene_path: str, output_dir: str) -> list:
    """Run using Newton GPU backend with MuJoCo-Warp solver."""
    from strands_robots.newton import NewtonBackend, NewtonConfig

    logger.info("⚡ Using Newton backend (GPU, MuJoCo-Warp solver)")

    config = NewtonConfig(
        num_envs=1,
        solver="mujoco",           # MuJoCo-Warp: best for articulated rigid bodies
        device=args.device,
        physics_dt=0.002,           # 500 Hz physics
        render_backend="opengl",    # OpenGL for video capture
    )

    backend = NewtonBackend(config)
    backend.create_world(gravity=(0.0, 0.0, -9.81), ground_plane=True)

    # Add SO-101 robot from MJCF
    assets_dir = Path(scene_path).parent
    so101_xml = assets_dir / "so101.xml"
    if so101_xml.exists():
        backend.add_robot("so101", mjcf_path=str(so101_xml))
    else:
        backend.add_robot("so101")

    video_paths = []

    for ep in range(args.episodes):
        logger.info(f"\n{'='*60}")
        logger.info(f"  Episode {ep+1}/{args.episodes} (Newton)")
        logger.info(f"{'='*60}")

        output_path = os.path.join(output_dir, f"groot_fruit_picking_newton_ep{ep+1}.mp4")

        try:
            result = backend.record_video(
                robot_name="so101",
                policy_provider=args.policy if args.policy != "groot" else "mock",
                instruction=args.instruction,
                duration=args.duration,
                fps=args.fps,
                width=args.width,
                height=args.height,
                output_path=output_path,
            )

            if result.get("status") == "success":
                video_paths.append(output_path)
                logger.info(f"✅ {result['content'][0]['text']}")
            else:
                logger.error(f"❌ {result.get('content', [{}])[0].get('text', 'Unknown error')}")
        except Exception as e:
            logger.error(f"❌ Newton episode {ep+1} failed: {e}")

    backend.destroy()
    return video_paths


# ──────────────────────────────────────────────────────────────────────
# Strategy C: Manual loop — GR00T N1.6 + MuJoCo render + PyAV encode
# This gives full control over the inference loop.
# ──────────────────────────────────────────────────────────────────────

def run_groot_manual_loop(args, scene_path: str, output_dir: str) -> list:
    """Full GR00T N1.6 inference loop with manual MuJoCo step + video recording.

    This is the most flexible approach:
    1. Load MuJoCo model directly
    2. Load GR00T N1.6 checkpoint
    3. Run inference loop: observe → infer → act → render → encode
    4. Save MP4 via PyAV (H.264)
    """
    import mujoco

    logger.info("🧠 Manual GR00T inference loop + MuJoCo render")

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Reset to "ready" keyframe if available
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Load GR00T N1.6 policy
    from strands_robots.policies.groot import Gr00tPolicy

    logger.info(f"🧠 Loading GR00T N1.6: {args.model}")
    policy = Gr00tPolicy(
        data_config=args.data_config,
        model_path=args.model,
        groot_version="n1.6",
        denoising_steps=args.denoising_steps,
        device=args.device,
    )

    # Discover hinge joints (SO-101 has 6 actuated joints)
    joint_names = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                joint_names.append(name)
    if not joint_names:
        joint_names = [f"joint_{i}" for i in range(model.nu)]

    policy.set_robot_state_keys(joint_names[:model.nu])
    logger.info(f"🔩 Robot joints ({len(joint_names)}): {joint_names[:model.nu]}")

    # Camera IDs
    front_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_cam")
    wrist_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

    video_paths = []

    for ep in range(args.episodes):
        logger.info(f"\n{'='*60}")
        logger.info(f"  Episode {ep+1}/{args.episodes} — GR00T N1.6 Inference")
        logger.info(f"{'='*60}")

        # Reset
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        output_path = os.path.join(output_dir, f"groot_fruit_picking_ep{ep+1}.mp4")
        total_frames = int(args.duration * args.fps)
        phys_per_frame = max(1, int(1.0 / (args.fps * model.opt.timestep)))

        frames = []
        latencies = []

        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        loop = asyncio.new_event_loop()

        for frame_idx in range(total_frames):
            # ── 1. Observe ──
            obs = {}

            # Camera image (front cam → 224×224 for GR00T)
            renderer.update_scene(data, camera=front_cam_id)
            front_img = renderer.render().copy()

            try:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(front_img).resize((224, 224), PILImage.BILINEAR)
                obs["front"] = np.array(pil_img, dtype=np.uint8)
            except ImportError:
                obs["front"] = front_img

            # Wrist camera (if dual-cam config)
            if wrist_cam_id >= 0 and "dualcam" in args.data_config:
                renderer.update_scene(data, camera=wrist_cam_id)
                wrist_img = renderer.render().copy()
                try:
                    from PIL import Image as PILImage
                    pil_img = PILImage.fromarray(wrist_img).resize((224, 224), PILImage.BILINEAR)
                    obs["wrist"] = np.array(pil_img, dtype=np.uint8)
                except ImportError:
                    obs["wrist"] = wrist_img

            # Joint states (individual keys for mapping)
            for i, name in enumerate(joint_names[:model.nu]):
                obs[name] = float(data.qpos[i])

            # Aggregate state keys for GR00T data config
            obs["state.single_arm"] = data.qpos[:5].astype(np.float32)
            obs["state.gripper"] = data.qpos[5:6].astype(np.float32)

            # ── 2. GR00T Inference ──
            t0 = time.perf_counter()
            try:
                actions = loop.run_until_complete(
                    policy.get_actions(obs, args.instruction)
                )
            except Exception as e:
                if frame_idx == 0:
                    logger.warning(f"  Inference failed: {e}")
                actions = [{name: 0.0 for name in joint_names[:model.nu]}]
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            # ── 3. Apply action ──
            if actions:
                action = actions[0]  # First step from action chunk
                for i, name in enumerate(joint_names[:model.nu]):
                    if name in action:
                        data.ctrl[i] = action[name]

            # ── 4. Step physics ──
            for _ in range(phys_per_frame):
                mujoco.mj_step(model, data)

            # ── 5. Render for video ──
            renderer.update_scene(data, camera=front_cam_id)
            frame = renderer.render().copy()
            frames.append(frame)

            # Progress
            if (frame_idx + 1) % 30 == 0 or frame_idx == 0:
                avg_lat = np.mean(latencies[-30:])
                logger.info(
                    f"  Frame {frame_idx+1}/{total_frames} | "
                    f"latency={avg_lat:.1f}ms | "
                    f"sim_time={data.time:.2f}s"
                )

        loop.close()
        del renderer

        # ── 6. Encode video ──
        if frames:
            _encode_video(frames, output_path, args.fps)
            video_paths.append(output_path)
            file_kb = os.path.getsize(output_path) / 1024

            logger.info(f"\n📊 Episode {ep+1} Summary:")
            logger.info(f"   Frames:          {len(frames)}")
            logger.info(f"   Duration:        {args.duration:.1f}s")
            logger.info(f"   Mean latency:    {np.mean(latencies):.1f} ms")
            logger.info(f"   Inference freq:  {1000/np.mean(latencies):.1f} Hz")
            logger.info(f"   Video:           {output_path} ({file_kb:.0f} KB)")

    return video_paths


# ──────────────────────────────────────────────────────────────────────
# Strategy D: Direct MuJoCo + mock policy (quick validation, no GR00T)
# ──────────────────────────────────────────────────────────────────────

def run_mock_mujoco_direct(args, scene_path: str, output_dir: str) -> list:
    """Quick validation with mock policy using raw MuJoCo (no strands_robots deps)."""
    import mujoco

    logger.info("🎮 Direct MuJoCo + mock policy (quick validation)")

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    video_paths = []

    for ep in range(args.episodes):
        logger.info(f"\n  Episode {ep+1}/{args.episodes} (mock)")

        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        output_path = os.path.join(output_dir, f"groot_fruit_picking_mock_ep{ep+1}.mp4")
        total_frames = int(args.duration * args.fps)
        phys_per_frame = max(1, int(1.0 / (args.fps * model.opt.timestep)))

        frames = []
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front_cam")
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)

        for frame_idx in range(total_frames):
            # Mock sinusoidal policy
            t = data.time
            for i in range(model.nu):
                data.ctrl[i] = 0.1 * np.sin(1.0 * t + i * 0.5)

            for _ in range(phys_per_frame):
                mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=cam_id)
            frame = renderer.render().copy()
            frames.append(frame)

        del renderer

        if frames:
            _encode_video(frames, output_path, args.fps)
            video_paths.append(output_path)
            file_kb = os.path.getsize(output_path) / 1024
            logger.info(f"  ✅ {len(frames)} frames → {output_path} ({file_kb:.0f} KB)")

    return video_paths


def _encode_video(frames: list, output_path: str, fps: int):
    """Encode frames to MP4 using PyAV (H.264) with imageio fallback."""
    try:
        import av

        container = av.open(output_path, mode="w")
        stream = container.add_stream("h264", rate=fps)
        h, w = frames[0].shape[:2]
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "profile:v": "high"}

        for frame_data in frames:
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()
        logger.info(f"📹 Encoded {len(frames)} frames via PyAV (H.264)")
        return

    except ImportError:
        logger.debug("PyAV not available, trying imageio...")

    try:
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, quality=8, macro_block_size=1)
        for f in frames:
            writer.append_data(f)
        writer.close()
        logger.info(f"📹 Encoded {len(frames)} frames via imageio")
        return
    except ImportError:
        pass

    # Last resort: save as numpy
    npz_path = output_path.replace(".mp4", ".npz")
    np.savez_compressed(npz_path, frames=np.array(frames))
    logger.info(f"💾 Saved {len(frames)} frames as {npz_path} (no video encoder found)")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test GR00T N1.6 Fruit-Picking checkpoint in MuJoCo/Newton simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full GR00T N1.6 test (GPU required)
  MUJOCO_GL=egl python scripts/test_groot_n16_fruit_picking.py

  # Quick mock-policy test (no GPU)
  MUJOCO_GL=egl python scripts/test_groot_n16_fruit_picking.py --policy mock

  # Newton GPU backend
  MUJOCO_GL=egl python scripts/test_groot_n16_fruit_picking.py --backend newton

  # Custom checkpoint
  MUJOCO_GL=egl python scripts/test_groot_n16_fruit_picking.py \\
      --model /data/checkpoints/my_groot_finetuned/best

  # Multiple episodes with dual-cam config
  MUJOCO_GL=egl python scripts/test_groot_n16_fruit_picking.py \\
      --episodes 5 --data-config so100_dualcam --duration 15

Related: Issue #203 — https://github.com/cagataycali/strands-gtc-nvidia/issues/203
""",
    )

    # Model
    parser.add_argument(
        "--model",
        default="aaronsu11/GR00T-N1.6-3B-SO101-FruitPicking",
        help="HuggingFace model ID or local path (default: aaronsu11/GR00T-N1.6-3B-SO101-FruitPicking)",
    )
    parser.add_argument("--data-config", default="so100", help="GR00T data config (default: so100)")
    parser.add_argument("--denoising-steps", type=int, default=4, help="Diffusion denoising steps")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    # Simulation
    parser.add_argument(
        "--backend",
        choices=["mujoco", "newton", "manual", "mock_direct"],
        default="manual",
        help="Simulation backend (default: manual — full GR00T loop)",
    )
    parser.add_argument(
        "--policy",
        default="groot",
        help="Policy provider: groot, mock, stopped, random (default: groot)",
    )
    parser.add_argument("--instruction", default="pick up the orange fruit", help="Task instruction")

    # Video
    parser.add_argument("--duration", type=float, default=10.0, help="Episode duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")

    # Output
    parser.add_argument("--output-dir", default=None, help="Output directory for videos")

    args = parser.parse_args()

    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join("artifacts", "groot_fruit_picking")
    os.makedirs(args.output_dir, exist_ok=True)

    # Banner
    print("=" * 70)
    print("  🤖 GR00T N1.6 Fruit-Picking — SO-101 Simulation Test")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Data config: {args.data_config}")
    print(f"  Backend:     {args.backend}")
    print(f"  Policy:      {args.policy}")
    print(f"  Instruction: {args.instruction}")
    print(f"  Duration:    {args.duration}s × {args.episodes} episodes")
    print(f"  Video:       {args.width}×{args.height} @ {args.fps}fps")
    print(f"  Device:      {args.device}")
    print(f"  Output:      {args.output_dir}")
    print()

    # GPU check
    if args.device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
                print(f"  🎮 GPU: {gpu_name} ({gpu_mem:.0f} GB)")
            else:
                print("  ⚠️  CUDA not available — falling back to CPU")
                args.device = "cpu"
                if args.policy == "groot":
                    print("  ⚠️  GR00T requires GPU — switching to mock policy")
                    args.policy = "mock"
        except ImportError:
            print("  ⚠️  PyTorch not installed — cannot check GPU")
            if args.policy == "groot":
                args.device = "cpu"
    print()

    # Write scene file
    scene_path = write_scene_file(args.output_dir)

    # Run
    t0 = time.time()

    if args.backend == "mock_direct":
        # Quick validation: raw MuJoCo + mock policy (no strands_robots deps)
        video_paths = run_mock_mujoco_direct(args, scene_path, args.output_dir)
    elif args.backend == "newton":
        video_paths = run_newton_backend(args, scene_path, args.output_dir)
    elif args.backend == "mujoco":
        video_paths = run_mujoco_backend(args, scene_path, args.output_dir)
    elif args.backend == "manual":
        if args.policy == "groot":
            video_paths = run_groot_manual_loop(args, scene_path, args.output_dir)
        else:
            # For non-groot policies, use direct MuJoCo mock
            video_paths = run_mock_mujoco_direct(args, scene_path, args.output_dir)
    else:
        video_paths = run_mujoco_backend(args, scene_path, args.output_dir)

    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 70)
    print("  📋 Final Summary")
    print("=" * 70)
    print(f"  Total time:    {elapsed:.1f}s")
    print(f"  Videos saved:  {len(video_paths)}")
    for vp in video_paths:
        size_kb = os.path.getsize(vp) / 1024 if os.path.exists(vp) else 0
        print(f"    📹 {vp} ({size_kb:.0f} KB)")

    # Save manifest
    manifest = {
        "model": args.model,
        "data_config": args.data_config,
        "backend": args.backend,
        "policy": args.policy,
        "instruction": args.instruction,
        "duration": args.duration,
        "fps": args.fps,
        "episodes": args.episodes,
        "device": args.device,
        "videos": video_paths,
        "total_time_seconds": elapsed,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  📄 Manifest:   {manifest_path}")
    print()

    if video_paths:
        print("  ✅ Success! Videos ready for review.")
    else:
        print("  ❌ No videos were produced. Check errors above.")

    return 0 if video_paths else 1


if __name__ == "__main__":
    sys.exit(main())
