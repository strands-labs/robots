#!/usr/bin/env python3
"""
🦆 Thor Data Collection: Newton GPU Sim + Ego Camera → LeRobot v2.1 Dataset
===========================================================================
Issue #204 — Designed for NVIDIA AGX Thor (132GB unified GPU)

Collects 256 episodes of G1 43-DOF locomotion with ego-view camera rendering
at 1280×720, producing a LeRobot v2.1 compatible dataset with MP4 videos.

Usage:
    MUJOCO_GL=egl python3 scripts/newton_groot/thor_collect_data.py [--episodes 256] [--resume]

Produces:
    /home/cagatay/e2e_pipeline/gr00t_dataset/
    ├── data/chunk-000/episode_NNNNNN.parquet
    ├── videos/chunk-000/observation.images.ego_view/episode_NNNNNN.mp4
    └── meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Force headless EGL rendering
os.environ.setdefault("MUJOCO_GL", "egl")

# ============================================================
# Config
# ============================================================
OUTPUT_DIR = Path(os.environ.get("THOR_DATASET_DIR", "/home/cagatay/e2e_pipeline/gr00t_dataset"))
NUM_EPISODES = int(os.environ.get("THOR_NUM_EPISODES", "256"))
STEPS_PER_EPISODE = 400       # At 50Hz control = 8 seconds
BATCH_SIZE = 16               # Parallel worlds
DT = 1.0 / 200.0             # Physics dt (200Hz)
CONTROL_DT = 4                # Control every 4 physics steps → 50Hz
IMG_W, IMG_H = 1280, 720     # 720p (Thor has memory)
RENDER_EVERY = 4              # Render camera every 4 physics steps → 50Hz
CAMERA_FOV = math.radians(60.0)

# Camera placement relative to torso
CAM_OFFSET_FORWARD = 0.3
CAM_OFFSET_UP = 0.4

STATUS_FILE = Path("/home/cagatay/e2e_pipeline/status.json")


def update_status(step_name, status, extra=None):
    """Update pipeline status file."""
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE) as f:
                data = json.load(f)
        else:
            data = {"pipeline": "issue_204_e2e", "started_at": datetime.now(tz=timezone.utc).isoformat(), "steps": {}}

        step_data = {"status": status, "updated_at": datetime.now(tz=timezone.utc).isoformat()}
        if extra:
            step_data.update(extra)
        data["steps"][step_name] = step_data
        data["current_step"] = step_name

        with open(STATUS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"  Warning: Could not update status: {e}")


def main():
    parser = argparse.ArgumentParser(description="Thor Data Collection")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-width", type=int, default=IMG_W)
    parser.add_argument("--img-height", type=int, default=IMG_H)
    parser.add_argument("--resume", action="store_true", help="Skip already-collected episodes")
    parser.add_argument("--no-camera", action="store_true", help="Skip camera rendering (faster)")
    args = parser.parse_args()

    num_episodes = args.episodes
    batch_size = args.batch_size
    img_w, img_h = args.img_width, args.img_height
    use_camera = not args.no_camera

    print("=" * 70)
    print("  🦆 Thor Data Collection — Newton GPU Sim + Ego Camera")
    print(f"  Episodes: {num_episodes} | Batch: {batch_size} | Image: {img_w}×{img_h}")
    print(f"  Camera: {'ON' if use_camera else 'OFF'} | Resume: {args.resume}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    update_status("step2_newton_sim", "running", {"episodes": num_episodes, "batch_size": batch_size})

    # ============================================================
    # Import heavy deps
    # ============================================================
    import newton
    import newton.utils
    import torch
    import warp as wp
    import yaml
    from newton import JointTargetMode
    from newton.sensors import SensorTiledCamera

    # ============================================================
    # Load G1 + RL Policy
    # ============================================================
    print("\n[1/5] Loading G1 asset and RL policy...")
    asset_path = newton.utils.download_asset("unitree_g1")
    mjcf_file = str(asset_path / "mjcf" / "g1_29dof_with_hand_rev_1_0.xml")

    # Load policy weights
    _jit_model = torch.jit.load(str(asset_path / "rl_policies" / "mjw_g1_29DOF.pt"), map_location="cpu")
    policy_ckpt = {}
    for name, param in _jit_model.named_parameters():
        policy_ckpt[name] = param.data
    for name, buf in _jit_model.named_buffers():
        policy_ckpt[name] = buf

    with open(str(asset_path / "rl_policies" / "g1_29dof.yaml")) as f:
        policy_cfg = yaml.safe_load(f)

    num_dofs = policy_cfg["num_dofs"]
    action_scale = policy_cfg.get("action_scale", 0.5)
    obs_dim = 141  # gravity(3) + ang_vel(3) + cmd(3) + q(43) + qd(43) + prev_action(43) + extras(3)
    joint_names = policy_cfg.get("mjw_joint_names", [])
    print(f"  Policy: {num_dofs} DOFs, scale={action_scale}")
    print(f"  Image: {img_w}x{img_h}, FOV={math.degrees(CAMERA_FOV):.0f}°")

    # ============================================================
    # Build Model
    # ============================================================
    print("\n[2/5] Building Newton model...")
    g1 = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(g1)
    g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1e3, limit_kd=1e1, friction=1e-5)
    g1.default_shape_cfg.ke = 2e3
    g1.default_shape_cfg.kd = 1e2
    g1.default_shape_cfg.kf = 1e3
    g1.default_shape_cfg.mu = 0.75
    g1.add_mjcf(mjcf_file, xform=wp.transform(wp.vec3(0, 0, 0.8)),
                collapse_fixed_joints=True, enable_self_collisions=False)

    for i in range(6, g1.joint_dof_count):
        g1.joint_target_ke[i] = 200.0
        g1.joint_target_kd[i] = 5.0
        g1.joint_target_mode[i] = int(JointTargetMode.POSITION)
    g1.approximate_meshes("bounding_box")

    builder = newton.ModelBuilder()
    builder.replicate(g1, batch_size)
    builder.default_shape_cfg.ke = 1e3
    builder.default_shape_cfg.kd = 1e2
    builder.add_ground_plane()
    model = builder.finalize()
    dofs_per_world = model.joint_dof_count // batch_size
    bodies_per_world = model.body_count // batch_size
    print(f"  Model: {model.body_count} bodies, {model.joint_dof_count} DOFs, {batch_size} worlds")

    # ============================================================
    # Camera Setup
    # ============================================================
    cam = None
    color_output = None
    camera_rays = None

    if use_camera:
        print("\n[3/5] Setting up tiled camera...")
        try:
            cam = SensorTiledCamera(model)
            cam.assign_random_colors_per_world(seed=42)
            cam.create_default_light()
            color_output = cam.create_color_image_output(img_w, img_h, camera_count=1)
            camera_rays = cam.compute_pinhole_camera_rays(img_w, img_h, camera_fovs=CAMERA_FOV)
            print(f"  Camera: rays={camera_rays.shape}, output={color_output.shape}")
        except Exception as e:
            print(f"  ⚠️  Camera setup failed: {e}")
            print("  Continuing without camera rendering")
            use_camera = False
            cam = None
    else:
        print("\n[3/5] Camera disabled (--no-camera)")

    # ============================================================
    # Solver
    # ============================================================
    print("\n[4/5] Creating MuJoCo solver...")
    solver = newton.solvers.SolverMuJoCo(
        model, use_mujoco_cpu=False,
        solver="newton", integrator="implicitfast",
        njmax=300 * batch_size, nconmax=150 * batch_size,
        cone="elliptic", impratio=100, iterations=100, ls_iterations=50,
    )

    # ============================================================
    # RL Policy Network (simple MLP)
    # ============================================================
    class PolicyMLP:
        def __init__(self, ckpt, obs_dim, act_dim, action_scale):
            def get_param(names):
                for n in names:
                    if n in ckpt:
                        return ckpt[n].numpy() if hasattr(ckpt[n], 'numpy') else ckpt[n]
                raise KeyError(f"None of {names} found")

            self.w1 = get_param(["actor.0.weight"])
            self.b1 = get_param(["actor.0.bias"])
            self.w2 = get_param(["actor.2.weight"])
            self.b2 = get_param(["actor.2.bias"])
            self.w3 = get_param(["actor.4.weight"])
            self.b3 = get_param(["actor.4.bias"])
            self.w4 = get_param(["actor.6.weight"])
            self.b4 = get_param(["actor.6.bias"])
            self.action_scale = action_scale
            print(f"  Policy MLP: {self.w1.shape[1]}→{self.w1.shape[0]}→{self.w2.shape[0]}→{self.w3.shape[0]}→{self.w4.shape[0]}")

        def __call__(self, obs):
            obs = np.clip(obs, -5.0, 5.0)
            x = np.tanh(obs @ self.w1.T + self.b1)
            x = np.tanh(x @ self.w2.T + self.b2)
            x = np.tanh(x @ self.w3.T + self.b3)
            x = x @ self.w4.T + self.b4
            return np.clip(x * self.action_scale, -1.0, 1.0)

    policy = PolicyMLP(policy_ckpt, obs_dim, num_dofs, action_scale)

    # ============================================================
    # Create output directories
    # ============================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "videos" / "chunk-000" / "observation.images.ego_view").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "meta").mkdir(parents=True, exist_ok=True)

    # Check which episodes already exist (for resume)
    existing_episodes = set()
    if args.resume:
        for p in (OUTPUT_DIR / "data" / "chunk-000").glob("episode_*.parquet"):
            ep_num = int(p.stem.split("_")[1])
            existing_episodes.add(ep_num)
        print(f"  Resume mode: {len(existing_episodes)} episodes already collected")

    # ============================================================
    # Warmup / compile kernels
    # ============================================================
    print("\n[5/5] Compiling CUDA kernels (first step)...")
    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
    model.collide(state0, contacts)
    state0.clear_forces()
    solver.step(state0, state1, control, contacts, DT)
    state0, state1 = state1, state0
    wp.synchronize()
    print("  Kernels compiled ✅")

    # ============================================================
    # Collect Episodes
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  Collecting {num_episodes} episodes ({STEPS_PER_EPISODE} steps each)")
    print(f"{'='*70}\n")

    num_batches = math.ceil(num_episodes / batch_size)
    all_episode_lengths = []
    total_video_frames = 0
    t_start = time.time()

    for batch_idx in range(num_batches):
        batch_start = time.time()
        ep_start = batch_idx * batch_size
        current_batch_size = min(batch_size, num_episodes - ep_start)

        # Check if all episodes in this batch already exist
        if args.resume:
            batch_eps = set(range(ep_start, ep_start + current_batch_size))
            if batch_eps.issubset(existing_episodes):
                print(f"  Batch {batch_idx}: episodes {ep_start}-{ep_start+current_batch_size-1} already exist, skipping")
                for w in range(current_batch_size):
                    all_episode_lengths.append(STEPS_PER_EPISODE // CONTROL_DT)
                continue

        # Reset states
        state0 = model.state()
        state1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state0)

        # Storage per episode in batch
        ep_states = [[] for _ in range(current_batch_size)]
        ep_actions = [[] for _ in range(current_batch_size)]
        ep_frames = [[] for _ in range(current_batch_size)]
        prev_actions = [np.zeros(num_dofs, dtype=np.float32) for _ in range(current_batch_size)]

        for step in range(STEPS_PER_EPISODE):
            is_control_step = (step % CONTROL_DT == 0)
            is_render_step = use_camera and (step % RENDER_EVERY == 0)

            if is_control_step:
                joint_q = model.joint_q.numpy()
                joint_qd = model.joint_qd.numpy()

                for w in range(current_batch_size):
                    start = w * dofs_per_world
                    q = joint_q[start + 6:start + 6 + num_dofs]
                    qd = joint_qd[start + 6:start + 6 + num_dofs]

                    # Build observation (141 dims)
                    obs = np.zeros(obs_dim, dtype=np.float32)
                    obs[:3] = [0.0, 0.0, -9.81]
                    obs[3:6] = [0.0, 0.0, 0.0]
                    obs[6:9] = [1.0, 0.0, 0.0]  # Forward command
                    obs[9:9 + num_dofs] = q
                    obs[9 + num_dofs:9 + 2 * num_dofs] = qd
                    obs[9 + 2 * num_dofs:9 + 3 * num_dofs] = prev_actions[w]
                    # Extra dims
                    body_q_np = state0.body_q.numpy()
                    torso_z = body_q_np[w * bodies_per_world, 2]
                    obs[9 + 3 * num_dofs] = torso_z
                    obs[9 + 3 * num_dofs + 1] = 0.0
                    obs[9 + 3 * num_dofs + 2] = 0.0

                    action = policy(obs)
                    prev_actions[w] = action

                    ep_states[w].append(q.copy())
                    ep_actions[w].append(action.copy())

                # Apply targets
                target_arr = control.joint_target_pos.numpy()
                for w in range(current_batch_size):
                    start_dof = w * dofs_per_world
                    action = prev_actions[w]
                    for i in range(min(num_dofs, dofs_per_world - 6)):
                        target_arr[start_dof + 6 + i] = action[i]
                control.joint_target_pos.assign(wp.array(target_arr, dtype=wp.float32, device='cuda:0'))

            # Render camera frames
            if is_render_step and cam is not None:
                try:
                    body_q = state0.body_q.numpy()
                    cam_xforms = []
                    for w in range(current_batch_size):
                        torso_idx = w * bodies_per_world
                        torso_pos = body_q[torso_idx, :3]

                        cam_pos = wp.vec3(
                            torso_pos[0] + CAM_OFFSET_FORWARD,
                            torso_pos[1],
                            torso_pos[2] + CAM_OFFSET_UP
                        )
                        target = wp.vec3(torso_pos[0], torso_pos[1], torso_pos[2])
                        d = wp.normalize(wp.vec3(
                            float(target[0]) - float(cam_pos[0]),
                            float(target[1]) - float(cam_pos[1]),
                            float(target[2]) - float(cam_pos[2])
                        ))
                        yaw = math.atan2(float(d[1]), float(d[0]))
                        pitch = math.asin(max(-1.0, min(1.0, -float(d[2]))))
                        q = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw) * wp.quat_from_axis_angle(wp.vec3(0, 1, 0), pitch)
                        cam_xforms.append(wp.transform(cam_pos, q))

                    # Pad to full batch_size if current_batch_size < batch_size
                    while len(cam_xforms) < batch_size:
                        cam_xforms.append(cam_xforms[-1])

                    cam_transforms = wp.array([cam_xforms], dtype=wp.transformf, device="cuda:0")
                    cam.update(state0, cam_transforms, camera_rays, color_image=color_output)
                    wp.synchronize()

                    # Extract per-world images from tiled output
                    flat = cam.flatten_color_image_to_rgba(color_output)
                    flat_np = flat.numpy()

                    grid_cols = int(math.ceil(math.sqrt(batch_size)))
                    for w in range(current_batch_size):
                        row = w // grid_cols
                        col = w % grid_cols
                        y_start = row * img_h
                        x_start = col * img_w
                        frame = flat_np[y_start:y_start + img_h, x_start:x_start + img_w, :3].copy()
                        ep_frames[w].append(frame)
                        total_video_frames += 1

                except Exception as e:
                    if step == 0 and batch_idx == 0:
                        print(f"  ⚠️  Camera rendering error (will continue without): {e}")
                        use_camera = False
                        cam = None

            # Physics step
            model.collide(state0, contacts)
            state0.clear_forces()
            solver.step(state0, state1, control, contacts, DT)
            state0, state1 = state1, state0

        # Save episodes
        import pyarrow as pa
        import pyarrow.parquet as pq

        for w in range(current_batch_size):
            ep_idx = ep_start + w
            num_steps = len(ep_states[w])

            # Save parquet
            table = pa.table({
                "observation.state": [ep_states[w][i].tolist() for i in range(num_steps)],
                "action": [ep_actions[w][i].tolist() for i in range(num_steps)],
                "episode_index": [ep_idx] * num_steps,
                "frame_index": list(range(num_steps)),
                "timestamp": [i * CONTROL_DT * DT for i in range(num_steps)],
            })
            pq.write_table(table, OUTPUT_DIR / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")

            # Save video as MP4 via ffmpeg pipe
            num_frames = len(ep_frames[w])
            if num_frames > 0:
                video_path = OUTPUT_DIR / 'videos' / 'chunk-000' / 'observation.images.ego_view' / f'episode_{ep_idx:06d}.mp4'
                try:
                    cmd = [
                        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                        '-s', f'{img_w}x{img_h}', '-pix_fmt', 'rgb24', '-r', '50',
                        '-i', '-', '-an', '-vcodec', 'mpeg4', '-q:v', '5', '-pix_fmt', 'yuv420p',
                        str(video_path),
                    ]
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    for frame in ep_frames[w]:
                        raw = np.ascontiguousarray(frame, dtype=np.uint8)
                        proc.stdin.write(raw.tobytes())
                    proc.stdin.close()
                    proc.wait()
                    if proc.returncode != 0:
                        stderr_out = proc.stderr.read().decode() if proc.stderr else ''
                        if batch_idx == 0 and w == 0:
                            print(f'  ffmpeg error: {stderr_out[:300]}')
                except BrokenPipeError:
                    if batch_idx == 0 and w == 0:
                        print(f'  BrokenPipe on video {ep_idx} — frame size may mismatch ffmpeg expectation')
                except Exception as ve:
                    if batch_idx == 0 and w == 0:
                        print(f'  Video encoding error (ep {ep_idx}): {ve}')
            all_episode_lengths.append(num_steps)

        avg_z = np.mean([state0.body_q.numpy()[w * bodies_per_world, 2] for w in range(current_batch_size)])
        batch_time = time.time() - batch_start
        vid_info = f", video_frames={len(ep_frames[0])}" if ep_frames[0] else ""
        print(f"  Batch {batch_idx}/{num_batches}: eps {ep_start}-{ep_start+current_batch_size-1}, "
              f"avg_z={avg_z:.3f}m, steps/ep={len(ep_states[0])}{vid_info}, {batch_time:.1f}s")

    total_time = time.time() - t_start
    total_steps = sum(all_episode_lengths)

    print(f"\n  ✅ Collected {num_episodes} episodes, {total_steps} steps, {total_video_frames} video frames in {total_time:.1f}s")

    # ============================================================
    # Save Metadata
    # ============================================================
    print(f"\nSaving metadata to {OUTPUT_DIR}...")

    control_hz = int(1.0 / (CONTROL_DT * DT))

    info = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1",
        "total_episodes": num_episodes,
        "total_frames": total_steps,
        "total_video_frames": total_video_frames,
        "fps": control_hz,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [num_dofs], "names": joint_names},
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [img_h, img_w, 3],
                "video_info": {
                    "video.fps": 50, "video.codec": "mpeg4",
                    "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False
                }
            },
            "action": {"dtype": "float32", "shape": [num_dofs], "names": joint_names},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
        "splits": {"train": f"0:{int(num_episodes*0.8)}", "test": f"{int(num_episodes*0.8)}:{num_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": num_episodes,
    }
    with open(OUTPUT_DIR / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    with open(OUTPUT_DIR / "meta" / "episodes.jsonl", "w") as f:
        for ep in range(num_episodes):
            length = all_episode_lengths[ep] if ep < len(all_episode_lengths) else STEPS_PER_EPISODE // CONTROL_DT
            f.write(json.dumps({"episode_index": ep, "tasks": ["walk_forward"], "length": length}) + "\n")

    with open(OUTPUT_DIR / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": "walk_forward"}) + "\n")

    modality = {
        "video": ["observation.images.ego_view"],
        "state": ["observation.state"],
        "action": ["action"],
        "language": "walk forward"
    }
    with open(OUTPUT_DIR / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # Collection summary
    summary = {
        "total_episodes": num_episodes,
        "total_steps": total_steps,
        "total_video_frames": total_video_frames,
        "has_video": total_video_frames > 0,
        "sim_hz": 200,
        "control_hz": control_hz,
        "num_dofs": num_dofs,
        "image_size": f"{img_w}x{img_h}",
        "batch_size": batch_size,
        "gpu": "NVIDIA Thor",
        "collection_time_s": round(total_time, 1),
        "pipeline": "Newton + MuJoCo Warp + SensorTiledCamera → LeRobot v2.1 → GR00T N1.6",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    with open(OUTPUT_DIR / "collection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Dataset saved to {OUTPUT_DIR}")
    print(f"   Episodes: {num_episodes}, Steps: {total_steps}, Video frames: {total_video_frames}")
    print("   Format: LeRobot v2.1 + GR00T N1.6 compatible")

    update_status("step2_newton_sim", "completed", {
        "episodes": num_episodes,
        "total_steps": total_steps,
        "video_frames": total_video_frames,
        "duration_s": round(total_time, 1),
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
