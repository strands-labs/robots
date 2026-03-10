#!/usr/bin/env python3
"""
Generate a test dataset using MuJoCo StrandsSimEnv.

Produces 200 transitions (observation, action, reward) from the SO-100 arm
to validate the full simulation → observation → action pipeline.

Saves as NPZ to ~/isaac_sim_datasets/
"""

import os
import time

import numpy as np

# Ensure MUJOCO_GL=egl for headless rendering
os.environ.setdefault("MUJOCO_GL", "egl")


def generate_dataset():
    """Generate synthetic dataset from MuJoCo simulation."""
    import mujoco

    from strands_robots.simulation import Simulation

    print("=" * 60)
    print("🤖 Generating Test Dataset from MuJoCo Simulation")
    print("=" * 60)

    # Create simulation
    sim = Simulation()
    result = sim.create_world()
    print(f"  World: {result['content'][0]['text'][:80]}")

    # Add SO-100 robot
    result = sim.add_robot(name="arm", data_config="so100")
    print(f"  Robot: {result['content'][0]['text'][:80]}")

    # Get model info
    model = sim._world._model
    data = sim._world._data
    n_qpos = model.nq
    n_qvel = model.nv
    n_ctrl = model.nu
    n_cam = model.ncam

    print(f"  Model: nq={n_qpos}, nv={n_qvel}, nu={n_ctrl}, ncam={n_cam}")

    # Collect transitions
    n_transitions = 200
    observations = []
    actions_list = []
    rewards = []
    rgb_frames = []

    print(f"\n  Collecting {n_transitions} transitions...")
    start_time = time.time()

    for i in range(n_transitions):
        # Current observation
        obs = np.concatenate([
            data.qpos.copy().astype(np.float32),
            data.qvel.copy().astype(np.float32),
        ])
        observations.append(obs)

        # Random action (within actuator limits)
        action = np.zeros(n_ctrl, dtype=np.float32)
        for j in range(n_ctrl):
            if model.actuator_ctrllimited[j]:
                lo = float(model.actuator_ctrlrange[j, 0])
                hi = float(model.actuator_ctrlrange[j, 1])
                action[j] = np.random.uniform(lo, hi)
            else:
                action[j] = np.random.uniform(-1.0, 1.0)
        actions_list.append(action)

        # Apply action
        np.copyto(data.ctrl[:n_ctrl], action)

        # Step physics (10 substeps at 500Hz = 0.02s control)
        for _ in range(10):
            mujoco.mj_step(model, data)

        # Simple reward: negative distance of joints from zero (home position)
        reward = -float(np.sum(data.qpos[:n_ctrl] ** 2))
        rewards.append(reward)

        # Render RGB frame every 10 steps
        if i % 10 == 0 and n_cam > 0:
            try:
                renderer = mujoco.Renderer(model, height=240, width=320)
                renderer.update_scene(data, camera=0)
                frame = renderer.render().copy()
                renderer.close()
                rgb_frames.append(frame)
            except Exception as e:
                print(f"    Render warning at step {i}: {e}")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"    Step {i+1}/{n_transitions} ({elapsed:.1f}s)")

    elapsed = time.time() - start_time

    # Convert to arrays
    observations = np.array(observations)
    actions_arr = np.array(actions_list)
    rewards_arr = np.array(rewards, dtype=np.float32)

    print(f"\n  Collection complete in {elapsed:.1f}s")
    print(f"  Observations shape: {observations.shape}")
    print(f"  Actions shape: {actions_arr.shape}")
    print(f"  Rewards shape: {rewards_arr.shape}")
    print(f"  RGB frames: {len(rgb_frames)}")

    # Save as NPZ
    output_dir = os.path.expanduser("~/isaac_sim_datasets")
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, "so100_test_dataset.npz")
    save_dict = {
        "observations": observations,
        "actions": actions_arr,
        "rewards": rewards_arr,
        "n_qpos": np.array(n_qpos),
        "n_qvel": np.array(n_qvel),
        "n_ctrl": np.array(n_ctrl),
    }

    if rgb_frames:
        rgb_arr = np.array(rgb_frames)
        save_dict["rgb_frames"] = rgb_arr
        print(f"  RGB frames shape: {rgb_arr.shape}")

    np.savez_compressed(npz_path, **save_dict)
    size_kb = os.path.getsize(npz_path) / 1024

    print(f"\n  ✅ Dataset saved: {npz_path}")
    print(f"     Size: {size_kb:.1f} KB")
    print(f"     Transitions: {n_transitions}")

    # Verify the saved file
    loaded = np.load(npz_path)
    assert loaded["observations"].shape == observations.shape
    assert loaded["actions"].shape == actions_arr.shape
    assert loaded["rewards"].shape == rewards_arr.shape
    print("     ✅ Verification passed")

    # Clean up sim
    try:
        sim.destroy()
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("📊 Dataset Generation Summary")
    print("  Robot: SO-100 (6-DOF arm)")
    print(f"  Transitions: {n_transitions}")
    print(f"  Obs dim: {observations.shape[1]} (qpos={n_qpos} + qvel={n_qvel})")
    print(f"  Action dim: {actions_arr.shape[1]}")
    print(f"  RGB frames: {len(rgb_frames)} @ 320x240")
    print(f"  Output: {npz_path} ({size_kb:.1f} KB)")
    print(f"  Duration: {elapsed:.1f}s")
    print("=" * 60)

    return npz_path


if __name__ == "__main__":
    generate_dataset()
