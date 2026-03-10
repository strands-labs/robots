#!/usr/bin/env python3
"""
Generate a synthetic test dataset using MuJoCo StrandsSimEnv.
Collects observations, actions, and rewards from Franka + SO-100 robots.
Saves as NPZ files to ~/isaac_sim_datasets/.
"""
import os
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def generate_dataset(robot_name: str, num_transitions: int = 200, output_dir: str = None):
    """Generate a dataset for a given robot."""
    if output_dir is None:
        output_dir = os.path.expanduser("~/isaac_sim_datasets")
    os.makedirs(output_dir, exist_ok=True)

    from strands_robots.envs import StrandsSimEnv

    print(f"  Creating StrandsSimEnv for {robot_name}")

    env = StrandsSimEnv(
        robot_name=robot_name,
        max_episode_steps=num_transitions + 10,
        render_mode="rgb_array",
        render_width=128,
        render_height=128,
    )

    obs, env_info = env.reset()

    observations = []
    actions = []
    rewards = []
    rgb_frames = []

    action_dim = env.action_space.shape[0]
    print(f"  Action dim: {action_dim}, Obs keys: {list(obs.keys()) if isinstance(obs, dict) else obs.shape}")

    t0 = time.time()
    for step in range(num_transitions):
        t = step / 30.0
        action = np.array([
            0.1 * np.sin(t + i * 0.5) for i in range(action_dim)
        ], dtype=np.float32)

        obs_next, reward, terminated, truncated, step_info = env.step(action)

        # Capture RGB frame periodically
        if step % 4 == 0:
            try:
                frame = env.render()
                if frame is not None:
                    rgb_frames.append(frame)
            except Exception:
                pass

        if isinstance(obs_next, dict):
            state = obs_next.get("state", obs_next.get("observation", np.zeros(action_dim)))
        else:
            state = obs_next

        observations.append(np.asarray(state, dtype=np.float32).flatten())
        actions.append(action)
        rewards.append(float(reward))

        if terminated or truncated:
            obs_next, env_info = env.reset()

    elapsed = time.time() - t0
    env.close()

    obs_array = np.array(observations)
    act_array = np.array(actions)
    rew_array = np.array(rewards)

    dataset_path = os.path.join(output_dir, f"{robot_name}_qa_dataset.npz")
    save_dict = {
        "observations": obs_array,
        "actions": act_array,
        "rewards": rew_array,
        "robot_name": np.array(robot_name),
        "num_transitions": np.array(num_transitions),
        "action_dim": np.array(action_dim),
    }

    if rgb_frames:
        max_frames = min(50, len(rgb_frames))
        indices = np.linspace(0, len(rgb_frames) - 1, max_frames, dtype=int)
        save_dict["rgb_frames"] = np.array([rgb_frames[i] for i in indices])

    np.savez_compressed(dataset_path, **save_dict)
    file_size = os.path.getsize(dataset_path) / (1024 * 1024)

    print(f"  ✅ Saved {dataset_path}")
    print(f"     Transitions: {num_transitions}")
    print(f"     Obs shape:   {obs_array.shape}")
    print(f"     Act shape:   {act_array.shape}")
    print(f"     RGB frames:  {len(rgb_frames)} collected, {save_dict.get('rgb_frames', np.array([])).shape[0] if 'rgb_frames' in save_dict else 0} saved")
    print(f"     File size:   {file_size:.2f} MB")
    print(f"     Time:        {elapsed:.1f}s ({num_transitions/elapsed:.0f} transitions/sec)")

    return dataset_path


def main():
    print("=" * 60)
    print("QA Agent - Synthetic Test Dataset Generator")
    print("=" * 60)

    output_dir = os.path.expanduser("~/isaac_sim_datasets")
    robots = ["franka", "so100"]
    results = {}

    for robot in robots:
        print(f"\n{'─' * 40}")
        print(f"Robot: {robot}")
        print(f"{'─' * 40}")
        try:
            path = generate_dataset(robot, num_transitions=200, output_dir=output_dir)
            results[robot] = {"status": "success", "path": path}
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results[robot] = {"status": "error", "error": str(e)}

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for robot, result in results.items():
        status = "✅" if result["status"] == "success" else "❌"
        print(f"  {status} {robot}: {result.get('path', result.get('error', 'unknown'))}")

    print("\nDone! ✅")


if __name__ == "__main__":
    main()
