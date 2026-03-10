#!/usr/bin/env python3
"""
Generate a synthetic test dataset using MuJoCo StrandsSimEnv.
Collects 200 transitions (obs, action, reward) and saves as NPZ.
Also generates RGB frames from the simulation.
"""
import os
import time

import numpy as np

# Set up rendering
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("DISPLAY", ":1")

from strands_robots.envs import StrandsSimEnv


def generate_dataset(robot_name: str, num_transitions: int = 200, output_dir: str = "/home/ubuntu/isaac_sim_datasets"):
    """Generate a synthetic dataset from a MuJoCo simulation."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating StrandsSimEnv for {robot_name}...")
    env = StrandsSimEnv(
        robot_name=robot_name,
        render_mode="rgb_array",
        max_episode_steps=num_transitions + 10,
        render_width=640,
        render_height=480,
    )

    obs, info = env.reset()
    print(f"Observation keys: {list(obs.keys()) if isinstance(obs, dict) else obs.shape}")
    print(f"Action space: {env.action_space}")

    # Storage
    all_states = []
    all_actions = []
    all_rewards = []
    all_frames = []
    all_next_states = []

    t0 = time.time()
    for step in range(num_transitions):
        # Random action in action space
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Extract state
        if isinstance(obs, dict):
            state = obs.get("state", obs.get("observation", np.zeros(6)))
        else:
            state = obs

        if isinstance(next_obs, dict):
            next_state = next_obs.get("state", next_obs.get("observation", np.zeros(6)))
        else:
            next_state = next_obs

        all_states.append(np.array(state, dtype=np.float32))
        all_actions.append(np.array(action, dtype=np.float32))
        all_rewards.append(float(reward))
        all_next_states.append(np.array(next_state, dtype=np.float32))

        # Render RGB frame every 10 steps
        if step % 10 == 0:
            frame = env.render()
            if frame is not None and hasattr(frame, 'shape'):
                all_frames.append(frame)

        obs = next_obs

        if terminated or truncated:
            obs, info = env.reset()

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Step {step}/{num_transitions} ({elapsed:.1f}s)")

    env.close()
    elapsed = time.time() - t0

    # Convert to arrays
    states = np.array(all_states)
    actions = np.array(all_actions)
    rewards = np.array(all_rewards)
    next_states = np.array(all_next_states)
    frames = np.array(all_frames) if all_frames else np.array([])

    # Save as NPZ
    npz_path = os.path.join(output_dir, f"{robot_name}_dataset.npz")
    np.savez_compressed(
        npz_path,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        frames=frames,
    )

    file_size = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Dataset saved: {npz_path}")
    print(f"  Robot: {robot_name}")
    print(f"  Transitions: {len(states)}")
    print(f"  State shape: {states.shape}")
    print(f"  Action shape: {actions.shape}")
    print(f"  Rewards: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
    print(f"  RGB frames: {frames.shape}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}")

    return npz_path


def main():
    robots = ["so_arm100", "panda"]
    for robot_name in robots:
        try:
            generate_dataset(robot_name, num_transitions=200)
        except Exception as e:
            print(f"ERROR generating dataset for {robot_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
