"""
🦆 GR00T Dataset Collection with Camera — Newton G1 Locomotion
Step 2: Adds ego-view camera rendering + MP4 video output

Produces LeRobot v2.1 compatible dataset with:
- observation.state: 43 DOF joint positions
- observation.images.ego_view: 320x240 ego camera frames → MP4
- action: 43 DOF position targets
"""
import json
import math
import subprocess
import time
from pathlib import Path

import newton
import newton.utils
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import warp as wp
import yaml
from newton import JointTargetMode
from newton.sensors import SensorTiledCamera

# ============================================================
# Config
# ============================================================
OUTPUT_DIR = Path("/home/ubuntu/room_sim/gr00t_dataset_v2")
NUM_EPISODES = 64
STEPS_PER_EPISODE = 400
BATCH_SIZE = 16       # Parallel worlds
DT = 1.0 / 200.0     # Physics dt
CONTROL_DT = 4        # Control every 4 physics steps → 50Hz
IMG_W, IMG_H = 320, 240
RENDER_EVERY = 4      # Render camera every 4 physics steps → 50Hz (matches control)
CAMERA_FOV = math.radians(60.0)

# Camera placement (behind and above robot, looking forward)
CAM_OFFSET = wp.vec3(0.3, 0.0, 0.4)  # Relative to torso: forward, 0 lateral, up
# We'll compute camera transforms per-world based on torso position

print("=" * 60)
print("🦆 GR00T Dataset Collection v2 — With Camera")
print("=" * 60)

# ============================================================
# Load G1 + RL Policy
# ============================================================
asset_path = newton.utils.download_asset("unitree_g1")
mjcf_file = str(asset_path / "mjcf" / "g1_29dof_with_hand_rev_1_0.xml")

# Load RL policy
_jit_model = torch.jit.load(str(asset_path / "rl_policies" / "mjw_g1_29DOF.pt"), map_location="cpu")
# Extract weights from TorchScript model
policy_ckpt = {}
for name, param in _jit_model.named_parameters():
    policy_ckpt[name] = param.data
# Get running mean/var from buffers
for name, buf in _jit_model.named_buffers():
    policy_ckpt[name] = buf

with open(str(asset_path / "rl_policies" / "g1_29dof.yaml")) as f:
    policy_cfg = yaml.safe_load(f)

num_dofs = policy_cfg["num_dofs"]
action_scale = policy_cfg.get("action_scale", 0.5)
obs_dim = 141  # gravity(3) + ang_vel(3) + cmd(3) + q(43) + qd(43) + prev_action(43) + base_height_yaw_pitch(3)
joint_names = policy_cfg.get("mjw_joint_names", [])
print(f"  Policy: {num_dofs} DOFs, scale={action_scale}")
print(f"  Image: {IMG_W}x{IMG_H}, FOV={math.degrees(CAMERA_FOV):.0f}°")

# ============================================================
# Build Model
# ============================================================
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
builder.replicate(g1, BATCH_SIZE)
builder.default_shape_cfg.ke = 1e3
builder.default_shape_cfg.kd = 1e2
builder.add_ground_plane()
model = builder.finalize()
dofs_per_world = model.joint_dof_count // BATCH_SIZE
bodies_per_world = model.body_count // BATCH_SIZE

print(f"  Model: {model.body_count} bodies, {model.joint_dof_count} DOFs, {BATCH_SIZE} worlds")

# ============================================================
# Camera Setup
# ============================================================
cam = SensorTiledCamera(model)
cam.assign_random_colors_per_world(seed=42)
cam.create_default_light()
color_output = cam.create_color_image_output(IMG_W, IMG_H, camera_count=1)
camera_rays = cam.compute_pinhole_camera_rays(IMG_W, IMG_H, camera_fovs=CAMERA_FOV)
print(f"  Camera: rays={camera_rays.shape}, output={color_output.shape}")

# ============================================================
# Solver
# ============================================================
solver = newton.solvers.SolverMuJoCo(
    model, use_mujoco_cpu=False,
    solver="newton", integrator="implicitfast",
    njmax=300 * BATCH_SIZE, nconmax=150 * BATCH_SIZE,
    cone="elliptic", impratio=100, iterations=100, ls_iterations=50
)

# ============================================================
# RL Policy Network (simple MLP)
# ============================================================
class PolicyMLP:
    def __init__(self, ckpt, obs_dim, act_dim, action_scale):
        # Try both naming conventions (state_dict vs TorchScript)
        def get_param(names):
            for n in names:
                if n in ckpt:
                    return ckpt[n].numpy() if hasattr(ckpt[n], 'numpy') else ckpt[n]
            raise KeyError(f"None of {names} found in checkpoint. Keys: {list(ckpt.keys())[:10]}")

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
# Collect Episodes
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "videos" / "chunk-000" / "observation.images.ego_view").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "meta").mkdir(parents=True, exist_ok=True)

print(f"\nCollecting {NUM_EPISODES} episodes ({STEPS_PER_EPISODE} steps each)...")
print("-" * 60)

num_batches = NUM_EPISODES // BATCH_SIZE
all_episode_lengths = []
t_start = time.time()

for batch_idx in range(num_batches):
    batch_start = time.time()
    ep_start = batch_idx * BATCH_SIZE

    # Reset states
    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)

    # Storage per episode in batch
    ep_states = [[] for _ in range(BATCH_SIZE)]
    ep_actions = [[] for _ in range(BATCH_SIZE)]
    ep_frames = [[] for _ in range(BATCH_SIZE)]  # Camera frames

    prev_actions = [np.zeros(num_dofs) for _ in range(BATCH_SIZE)]

    for step in range(STEPS_PER_EPISODE):
        is_control_step = (step % CONTROL_DT == 0)
        is_render_step = (step % RENDER_EVERY == 0)

        if is_control_step:
            # Get joint states for all worlds
            joint_q = model.joint_q.numpy()
            joint_qd = model.joint_qd.numpy()

            for w in range(BATCH_SIZE):
                start = w * dofs_per_world
                q = joint_q[start + 6:start + 6 + num_dofs]
                qd = joint_qd[start + 6:start + 6 + num_dofs]

                # Build observation (141 dims)
                obs = np.zeros(obs_dim, dtype=np.float32)
                obs[:3] = [0.0, 0.0, -9.81]  # gravity in body frame
                obs[3:6] = [0.0, 0.0, 0.0]   # angular velocity
                obs[6:9] = [1.0, 0.0, 0.0]    # command velocity (forward)
                obs[9:9 + num_dofs] = q
                obs[9 + num_dofs:9 + 2 * num_dofs] = qd
                obs[9 + 2 * num_dofs:9 + 3 * num_dofs] = prev_actions[w]
                # Extra 3 dims: base height, yaw, pitch
                body_q_np = state0.body_q.numpy()
                torso_z = body_q_np[w * bodies_per_world, 2]
                obs[9 + 3 * num_dofs] = torso_z  # base height
                obs[9 + 3 * num_dofs + 1] = 0.0  # yaw
                obs[9 + 3 * num_dofs + 2] = 0.0  # pitch

                action = policy(obs)
                prev_actions[w] = action

                ep_states[w].append(q.copy())
                ep_actions[w].append(action.copy())

            # Apply targets via control
            target_arr = control.joint_target_pos.numpy()
            for w in range(BATCH_SIZE):
                start_dof = w * dofs_per_world
                action = prev_actions[w]
                for i in range(min(num_dofs, dofs_per_world - 6)):
                    target_arr[start_dof + 6 + i] = action[i]
            control.joint_target_pos.assign(wp.array(target_arr, dtype=wp.float32, device='cuda:0'))

        # Render camera frames
        if is_render_step:
            # Compute camera transforms based on torso positions
            body_q = state0.body_q.numpy()  # [num_bodies, 7] (pos + quat)

            cam_xforms = []
            for w in range(BATCH_SIZE):
                torso_idx = w * bodies_per_world  # First body = torso
                torso_pos = body_q[torso_idx, :3]
                torso_quat = body_q[torso_idx, 3:7]

                # Camera behind and above torso
                cam_pos = wp.vec3(
                    torso_pos[0] + CAM_OFFSET[0],
                    torso_pos[1] + CAM_OFFSET[1],
                    torso_pos[2] + CAM_OFFSET[2]
                )
                # Look at torso
                target = wp.vec3(torso_pos[0], torso_pos[1], torso_pos[2])
                d = wp.normalize(wp.vec3(target[0] - cam_pos[0], target[1] - cam_pos[1], target[2] - cam_pos[2]))
                yaw = math.atan2(float(d[1]), float(d[0]))
                pitch = math.asin(max(-1.0, min(1.0, -float(d[2]))))
                q = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw) * wp.quat_from_axis_angle(wp.vec3(0, 1, 0), pitch)
                cam_xforms.append(wp.transform(cam_pos, q))

            cam_transforms = wp.array([cam_xforms], dtype=wp.transformf, device="cuda:0")
            cam.update(state0, cam_transforms, camera_rays, color_image=color_output)
            wp.synchronize()

            # Extract per-world images from tiled camera output
            flat = cam.flatten_color_image_to_rgba(color_output)
            flat_np = flat.numpy()  # [tiled_h, tiled_w, 4]

            # Newton arranges tiled camera as grid: ceil(sqrt(n)) x ceil(sqrt(n))
            grid_cols = int(math.ceil(math.sqrt(BATCH_SIZE)))
            grid_rows = int(math.ceil(BATCH_SIZE / grid_cols))

            for w in range(BATCH_SIZE):
                row = w // grid_cols
                col = w % grid_cols
                y_start = row * IMG_H
                y_end = y_start + IMG_H
                x_start = col * IMG_W
                x_end = x_start + IMG_W
                frame = flat_np[y_start:y_end, x_start:x_end, :3].copy()
                ep_frames[w].append(frame)

        # Physics step
        model.collide(state0, contacts)
        state0.clear_forces()
        solver.step(state0, state1, control, contacts, DT)
        state0, state1 = state1, state0

    # Save episodes
    for w in range(BATCH_SIZE):
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

        # Save video as MP4 using ffmpeg (more reliable than cv2 in headless)
        num_frames = len(ep_frames[w])
        if num_frames > 0:
            video_path = OUTPUT_DIR / "videos" / "chunk-000" / "observation.images.ego_view" / f"episode_{ep_idx:06d}.mp4"
            # Use ffmpeg with raw video pipe - works reliably in headless mode
            cmd = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{IMG_W}x{IMG_H}', '-pix_fmt', 'rgb24', '-r', '50',
                '-i', '-', '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '23', '-preset', 'fast', str(video_path)
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for frame in ep_frames[w]:
                proc.stdin.write(frame.astype(np.uint8).tobytes())
            proc.stdin.close()
            proc.wait()

        all_episode_lengths.append(num_steps)

    avg_z = np.mean([state0.body_q.numpy()[w * bodies_per_world, 2] for w in range(BATCH_SIZE)])
    batch_time = time.time() - batch_start
    print(f"  Batch {batch_idx}: episodes {ep_start}-{ep_start + BATCH_SIZE - 1}, "
          f"avg_z={avg_z:.3f}m, frames/ep={len(ep_frames[0])}, {batch_time:.1f}s")

total_time = time.time() - t_start
total_frames = sum(all_episode_lengths)
print(f"\n  ✅ Collected {NUM_EPISODES} episodes, {total_frames} frames in {total_time:.1f}s")

# ============================================================
# Save Metadata
# ============================================================
print(f"\nSaving dataset to {OUTPUT_DIR}...")

# info.json
info = {
    "codebase_version": "v2.1",
    "robot_type": "unitree_g1",
    "total_episodes": NUM_EPISODES,
    "total_frames": total_frames,
    "fps": int(1.0 / (CONTROL_DT * DT)),
    "features": {
        "observation.state": {"dtype": "float32", "shape": [num_dofs], "names": joint_names},
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [IMG_H, IMG_W, 3],
            "names": None,
            "video_info": {"video.fps": 50, "video.codec": "mp4v", "video.pix_fmt": "bgr24",
                          "video.is_depth_map": False, "has_audio": False}
        },
        "action": {"dtype": "float32", "shape": [num_dofs], "names": joint_names},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
    },
    "splits": {"train": f"0:{NUM_EPISODES}"},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "chunks_size": NUM_EPISODES,
}
with open(OUTPUT_DIR / "meta" / "info.json", "w") as f:
    json.dump(info, f, indent=2, default=str)

# episodes.jsonl
with open(OUTPUT_DIR / "meta" / "episodes.jsonl", "w") as f:
    for ep in range(NUM_EPISODES):
        f.write(json.dumps({"episode_index": ep, "tasks": ["walk_forward"],
                            "length": all_episode_lengths[ep]}) + "\n")

# tasks.jsonl
with open(OUTPUT_DIR / "meta" / "tasks.jsonl", "w") as f:
    f.write(json.dumps({"task_index": 0, "task": "walk_forward"}) + "\n")

# modality.json (GR00T N1 format)
modality = {
    "video": ["observation.images.ego_view"],
    "state": ["observation.state"],
    "action": ["action"],
    "language": "walk forward"
}
with open(OUTPUT_DIR / "meta" / "modality.json", "w") as f:
    json.dump(modality, f, indent=2)

# Summary
summary = {
    "total_episodes": NUM_EPISODES,
    "total_frames": total_frames,
    "sim_hz": 200,
    "control_hz": 50,
    "num_dofs": num_dofs,
    "image_size": f"{IMG_W}x{IMG_H}",
    "has_video": True,
    "gpu": "NVIDIA L40S",
    "collection_time_s": round(total_time, 1),
    "pipeline": "Newton + MuJoCo Warp + SensorTiledCamera → LeRobot v2.1 → GR00T N1.6",
}
with open(OUTPUT_DIR / "collection_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Dataset saved to {OUTPUT_DIR}")
print(f"   Episodes: {NUM_EPISODES}, Frames: {total_frames}")
print(f"   Videos: {OUTPUT_DIR / 'videos'}")
print("   Format: LeRobot v2.1 + GR00T N1.6 compatible")
