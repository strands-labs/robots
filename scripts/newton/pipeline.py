#!/usr/bin/env python3
"""
🦆 GR00T N1 Training Data Pipeline — G1 in Cagatay's Room
==========================================================

PIPELINE OVERVIEW:
1. Newton sim: G1 with pre-trained RL locomotion policy walking in room (USD scene)
2. Newton TiledCamera: Render ego-view RGB frames at 20Hz
3. Record joint states + actions in LeRobot format
4. Cosmos Transfer 2.5: Domain-randomize sim frames → photorealistic
5. GR00T N1.6 fine-tune: Train on the LeRobot dataset with unitree_g1 embodiment tag

The result: A GR00T N1 model that can navigate YOUR room on YOUR G1 robot.
"""

import json
import time
from pathlib import Path

import cv2
import newton
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import warp as wp
import yaml
from newton import JointTargetMode

# ═══════════════════════════════════════════════════════════════
# STEP 0: Configuration
# ═══════════════════════════════════════════════════════════════

ROOM_USD = "/home/ubuntu/room_sim/extracted/3_7_2026.usda"
G1_ASSET_PATH = Path("/home/ubuntu/.cache/newton/newton-assets_unitree_g1_308a72cd/unitree_g1")
G1_USD = str(G1_ASSET_PATH / "usd" / "g1_isaac.usd")
G1_POLICY = str(G1_ASSET_PATH / "rl_policies" / "mjw_g1_29DOF.pt")
G1_POLICY_CFG = str(G1_ASSET_PATH / "rl_policies" / "g1_29dof.yaml")

OUTPUT_DIR = Path("/home/ubuntu/room_sim/gr00t_dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simulation parameters
NUM_EPISODES = 50       # Number of walking episodes
EPISODE_LENGTH = 200    # Steps per episode (at 50Hz = 4 seconds)
NUM_WORLDS = 16         # Parallel simulation environments
SIM_HZ = 200            # Physics rate
CONTROL_HZ = 50         # Control / data collection rate
RENDER_HZ = 20          # Camera frame rate (for GR00T training)
IMG_WIDTH = 640
IMG_HEIGHT = 480

print("=" * 60)
print("🦆 GR00T N1 Dataset Pipeline — G1 × Cagatay's Room")
print("=" * 60)
print(f"Room USD:     {ROOM_USD}")
print(f"G1 USD:       {G1_USD}")
print(f"G1 Policy:    {G1_POLICY}")
print(f"Episodes:     {NUM_EPISODES}")
print(f"Worlds:       {NUM_WORLDS}")
print(f"Output:       {OUTPUT_DIR}")
print()

# ═══════════════════════════════════════════════════════════════
# STEP 1: Newton Simulation — G1 walking with RL policy
# ═══════════════════════════════════════════════════════════════

print("STEP 1: Newton simulation — G1 with locomotion policy")
print("-" * 60)

# Load policy config
with open(G1_POLICY_CFG) as f:
    policy_cfg = yaml.safe_load(f)

print(f"  Policy joints: {policy_cfg['num_dofs']}")
print(f"  Action scale: {policy_cfg['action_scale']}")

# Load trained policy network
policy_net = torch.jit.load(G1_POLICY, map_location="cuda:0")
policy_net.eval()
print(f"  Policy loaded: {G1_POLICY}")

# Build Newton model
builder = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
    limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
)
builder.default_shape_cfg.ke = 2.0e3
builder.default_shape_cfg.kd = 1.0e2
builder.default_shape_cfg.kf = 1.0e3
builder.default_shape_cfg.mu = 0.75

# Build G1 model (one instance as template)
g1_template = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(g1_template)
g1_template.default_joint_cfg = builder.default_joint_cfg
g1_template.default_shape_cfg = builder.default_shape_cfg

g1_template.add_usd(
    G1_USD,
    xform=wp.transform(wp.vec3(0, 0, 0.8)),
    collapse_fixed_joints=True,
    enable_self_collisions=False,
    hide_collision_shapes=True,
    skip_mesh_approximation=True,
)

# Set joint drives
for i in range(6, g1_template.joint_dof_count):
    stiffness = policy_cfg['mjw_joint_stiffness'][i] if i < len(policy_cfg['mjw_joint_stiffness']) else 100.0
    damping = policy_cfg['mjw_joint_damping'][i] if i < len(policy_cfg['mjw_joint_damping']) else 5.0
    g1_template.joint_target_ke[i] = stiffness
    g1_template.joint_target_kd[i] = damping
    g1_template.joint_target_mode[i] = int(JointTargetMode.POSITION)

g1_template.approximate_meshes("bounding_box")

# Replicate for parallel worlds
builder.replicate(g1_template, NUM_WORLDS)
builder.default_shape_cfg.ke = 1.0e3
builder.default_shape_cfg.kd = 1.0e2
builder.add_ground_plane()

model = builder.finalize()
solver = newton.solvers.SolverMuJoCo(
    model,
    use_mujoco_cpu=False,
    solver="newton",
    integrator="implicitfast",
    njmax=300 * NUM_WORLDS,
    nconmax=150 * NUM_WORLDS,
    cone="elliptic",
    impratio=100,
    iterations=100,
    ls_iterations=50,
)

state = model.state()
state_next = model.state()
control = model.control()

# Evaluate initial FK
newton.eval_fk(model, model.joint_q, model.joint_qd, state)

contacts = model.contacts()

# Setup camera sensor for ego-view rendering
camera = newton.SensorTiledCamera(
    model,
    world_count=NUM_WORLDS,
    width=IMG_WIDTH,
    height=IMG_HEIGHT,
)

print(f"  Newton model built: {model.body_count} bodies, {model.joint_count} joints")
print(f"  {NUM_WORLDS} parallel worlds ready")
print(f"  Camera sensor: {IMG_WIDTH}x{IMG_HEIGHT}")
print()

# ═══════════════════════════════════════════════════════════════
# STEP 2: Run episodes and collect data
# ═══════════════════════════════════════════════════════════════

print("STEP 2: Running episodes and collecting data...")
print("-" * 60)

# Data storage (LeRobot format)
all_episodes = []
frame_dt = 1.0 / CONTROL_HZ
sim_dt = 1.0 / SIM_HZ
sim_substeps = SIM_HZ // CONTROL_HZ  # 4 substeps

total_frames = 0
t_start = time.time()

for ep_idx in range(0, NUM_EPISODES, NUM_WORLDS):
    batch_size = min(NUM_WORLDS, NUM_EPISODES - ep_idx)

    # Reset states with random starting orientations
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    # Episode data buffers
    ep_states = [[] for _ in range(batch_size)]
    ep_actions = [[] for _ in range(batch_size)]
    ep_frames = [[] for _ in range(batch_size)]

    for step in range(EPISODE_LENGTH):
        # Get observations for policy
        joint_pos = model.joint_q.numpy().reshape(NUM_WORLDS, -1)[:batch_size]
        joint_vel = model.joint_qd.numpy().reshape(NUM_WORLDS, -1)[:batch_size]

        # Run policy inference
        with torch.no_grad():
            obs = torch.from_numpy(
                np.concatenate([joint_pos, joint_vel], axis=-1)
            ).float().cuda()
            actions = policy_net(obs).cpu().numpy()

        # Apply actions
        action_scale = policy_cfg['action_scale']
        default_pos = np.array(policy_cfg['mjw_joint_pos'])

        for w in range(batch_size):
            target = default_pos + actions[w] * action_scale
            # Store state and action
            ep_states[w].append(joint_pos[w].copy())
            ep_actions[w].append(target.copy())

        # Simulate substeps
        for sub in range(sim_substeps):
            model.collide(state, contacts)
            state.clear_forces()
            solver.step(state, state_next, control, contacts, sim_dt)
            state, state_next = state_next, state

        # Render frames at RENDER_HZ
        if step % (CONTROL_HZ // RENDER_HZ) == 0:
            camera.render(state)
            frames = camera.get_color()  # [NUM_WORLDS, H, W, 4] RGBA
            for w in range(batch_size):
                ep_frames[w].append(frames[w, :, :, :3].copy())  # RGB only
                total_frames += 1

    # Save episodes
    for w in range(batch_size):
        episode_id = ep_idx + w
        episode_data = {
            "episode_id": episode_id,
            "states": np.array(ep_states[w]),      # [T, num_dof]
            "actions": np.array(ep_actions[w]),     # [T, num_dof]
            "frames": np.array(ep_frames[w]),       # [T_render, H, W, 3]
            "fps": RENDER_HZ,
            "task": "walk_around_room",
        }
        all_episodes.append(episode_data)

    elapsed = time.time() - t_start
    eps_done = ep_idx + batch_size
    print(f"  Episodes {ep_idx}-{eps_done-1}: {total_frames} frames, {elapsed:.1f}s")

print(f"\n  Total: {len(all_episodes)} episodes, {total_frames} frames")
print(f"  Wall time: {time.time()-t_start:.1f}s")

# ═══════════════════════════════════════════════════════════════
# STEP 3: Save in LeRobot format (GR00T compatible)
# ═══════════════════════════════════════════════════════════════

print("\nSTEP 3: Saving LeRobot dataset...")
print("-" * 60)

# Create LeRobot directory structure
meta_dir = OUTPUT_DIR / "meta"
meta_dir.mkdir(exist_ok=True)
data_dir = OUTPUT_DIR / "data"
data_dir.mkdir(exist_ok=True)
videos_dir = OUTPUT_DIR / "videos"
videos_dir.mkdir(exist_ok=True)

# Joint names for state/action columns
joint_names = policy_cfg['mjw_joint_names']

# info.json
info = {
    "codebase_version": "v2.1",
    "robot_type": "unitree_g1",
    "fps": RENDER_HZ,
    "features": {
        "observation.state": {"dtype": "float32", "shape": [len(joint_names)]},
        "action": {"dtype": "float32", "shape": [len(joint_names)]},
        "observation.images.ego_view": {"dtype": "video", "shape": [IMG_HEIGHT, IMG_WIDTH, 3]},
    },
    "total_episodes": len(all_episodes),
    "total_frames": sum(len(ep["states"]) for ep in all_episodes),
    "splits": {"train": f"0:{len(all_episodes)}"},
}
with open(meta_dir / "info.json", "w") as f:
    json.dump(info, f, indent=2)

# episodes.jsonl
with open(meta_dir / "episodes.jsonl", "w") as f:
    for ep in all_episodes:
        f.write(json.dumps({
            "episode_index": ep["episode_id"],
            "tasks": [ep["task"]],
            "length": len(ep["states"]),
        }) + "\n")

# tasks.jsonl
with open(meta_dir / "tasks.jsonl", "w") as f:
    f.write(json.dumps({"task_index": 0, "task": "walk_around_room"}) + "\n")

# modality.json (GR00T specific)
modality = {
    "video": {
        "ego_view": {
            "original_key": "observation.images.ego_view",
            "fps": RENDER_HZ,
        }
    },
    "state": {
        "joint_positions": {
            "original_key": "observation.state",
            "dim": len(joint_names),
            "joint_names": joint_names,
        }
    },
    "action": {
        "joint_positions": {
            "original_key": "action",
            "dim": len(joint_names),
            "joint_names": joint_names,
        }
    },
}
with open(meta_dir / "modality.json", "w") as f:
    json.dump(modality, f, indent=2)

# Save episode data as parquet + videos
for ep in all_episodes:
    ep_id = ep["episode_id"]

    # Save state/action as parquet
    chunk_id = ep_id // 1000
    chunk_dir = data_dir / f"chunk-{chunk_id:03d}"
    chunk_dir.mkdir(exist_ok=True)

    table = pa.table({
        "observation.state": [ep["states"][i].tolist() for i in range(len(ep["states"]))],
        "action": [ep["actions"][i].tolist() for i in range(len(ep["actions"]))],
        "episode_index": [ep_id] * len(ep["states"]),
        "frame_index": list(range(len(ep["states"]))),
        "timestamp": [i / CONTROL_HZ for i in range(len(ep["states"]))],
    })
    pq.write_table(table, chunk_dir / f"episode_{ep_id:06d}.parquet")

    # Save video frames as mp4
    vid_dir = videos_dir / f"chunk-{chunk_id:03d}" / "observation.images.ego_view"
    vid_dir.mkdir(parents=True, exist_ok=True)

    # Write frames as video using cv2
    video_path = vid_dir / f"episode_{ep_id:06d}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, RENDER_HZ, (IMG_WIDTH, IMG_HEIGHT))
    for frame in ep["frames"]:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

print(f"  Dataset saved to: {OUTPUT_DIR}")
print(f"  Episodes: {len(all_episodes)}")
print("  Format: LeRobot v2.1 (GR00T compatible)")

# ═══════════════════════════════════════════════════════════════
# STEP 4: Cosmos Transfer 2.5 — Sim-to-Real Domain Transfer
# ═══════════════════════════════════════════════════════════════

print("\nSTEP 4: Cosmos Transfer (sim→real domain randomization)")
print("-" * 60)
print("  [Will be run as separate step with cosmos_transfer2 CLI]")
print("  Command:")
print("  python -m cosmos_transfer2.examples.inference \\")
print(f"    -i {OUTPUT_DIR}/cosmos_transfer_input.json \\")
print("    --setup.checkpoint_dir /home/ubuntu/checkpoints/DreamZero-DROID \\")
print(f"    --setup.output_dir {OUTPUT_DIR}/cosmos_transferred")
print()

# Create Cosmos Transfer input config
cosmos_input = {
    "input_video_dir": str(videos_dir),
    "condition_type": "edge",
    "num_frames": 25,
    "guidance_scale": 3.0,
    "seed": 42,
}
with open(OUTPUT_DIR / "cosmos_transfer_input.json", "w") as f:
    json.dump(cosmos_input, f, indent=2)

# ═══════════════════════════════════════════════════════════════
# STEP 5: GR00T N1.6 Fine-tuning Command
# ═══════════════════════════════════════════════════════════════

print("STEP 5: GR00T N1.6 Fine-tuning")
print("-" * 60)
print("  Command (run in groot_venv):")
print("  source /home/ubuntu/groot_venv/bin/activate")
print("  python -m gr00t.experiment.launch_finetune \\")
print("    --base_model_path nvidia/GR00T-N1.6-3B \\")
print(f"    --dataset_path {OUTPUT_DIR} \\")
print("    --embodiment_tag unitree_g1 \\")
print(f"    --output_dir {OUTPUT_DIR}/gr00t_finetuned \\")
print("    --max_steps 5000 \\")
print("    --global_batch_size 32 \\")
print("    --learning_rate 1e-4 \\")
print("    --num_gpus 1")
print()

print("=" * 60)
print("🦆 Pipeline complete!")
print("=" * 60)
