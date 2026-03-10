#!/usr/bin/env python3
"""
Generate a synthetic test dataset using MuJoCo StrandsSimEnv.
Collects 200 transitions (obs, action, reward) with Franka and SO-100 robots.
Also generates RGB+depth frames.
"""

import os
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def generate_dataset():
    """Generate dataset using Simulation with Franka robot."""
    import mujoco

    from strands_robots.simulation import Simulation

    print("=" * 60)
    print("Phase 5: Generating Test Dataset")
    print("=" * 60)

    robot_name = "franka"
    print(f"\n🤖 Creating simulation with {robot_name}...")

    # Create sim world via public API
    sim = Simulation()
    result = sim.create_world(timestep=0.002, gravity=-9.81)
    print(f"   Create world: {result.get('status', 'unknown')}")

    result = sim.add_robot(name="robot0", data_config=robot_name)
    print(f"   Add robot: {result.get('status', 'unknown')}")

    # Add some objects
    sim.add_object(name="red_cube", shape="box", size=[0.03, 0.03, 0.03],
                   position=[0.4, 0.0, 0.03], color=[1.0, 0.0, 0.0, 1.0])
    sim.add_object(name="green_sphere", shape="sphere", size=[0.025, 0.025, 0.025],
                   position=[0.3, 0.15, 0.025], color=[0.0, 1.0, 0.0, 1.0])

    print("✅ Simulation world created with robot and objects")

    # Access MuJoCo model/data via internal attribute
    model = sim.mj_model
    data = sim.mj_data
    if model is None or data is None:
        print("❌ Model/data not available, trying _world directly...")
        model = sim._world._model
        data = sim._world._data

    n_joints = model.nu
    n_qpos = model.nq
    print(f"   Actuators: {n_joints}, DOF: {n_qpos}")

    # Storage
    num_transitions = 200
    observations = []
    actions_list = []
    rewards = []
    rgb_frames = []
    depth_frames = []

    print(f"\n📊 Collecting {num_transitions} transitions...")
    start_time = time.time()

    for step_i in range(num_transitions):
        # Get observation
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        obs = np.concatenate([qpos, qvel]).astype(np.float32)
        observations.append(obs)

        # Generate random action
        action = np.random.uniform(-0.1, 0.1, size=n_joints).astype(np.float32)
        actions_list.append(action)

        # Apply action and step
        data.ctrl[:n_joints] = action
        for _ in range(5):
            mujoco.mj_step(model, data)

        # Simple reward
        if len(qpos) >= 3:
            target = np.array([0.4, 0.0, 0.3])
            reward = -np.linalg.norm(qpos[:3] - target)
        else:
            reward = 0.0
        rewards.append(reward)

        # Render every 10 steps
        if step_i % 10 == 0:
            rgb_result = sim.render(width=320, height=240)
            if rgb_result.get("success"):
                rgb_frame = rgb_result.get("image")
                if rgb_frame is not None:
                    rgb_frames.append(rgb_frame)

            depth_result = sim.render_depth(width=320, height=240)
            if depth_result.get("success"):
                depth_frame = depth_result.get("depth")
                if depth_frame is not None:
                    depth_frames.append(depth_frame)

        if (step_i + 1) % 50 == 0:
            print(f"   Step {step_i + 1}/{num_transitions} (reward: {reward:.4f})")

    elapsed = time.time() - start_time
    print(f"\n✅ Collected {num_transitions} transitions in {elapsed:.2f}s")
    print(f"   FPS: {num_transitions / elapsed:.1f}")

    # Convert to arrays
    observations = np.array(observations)
    actions_arr = np.array(actions_list)
    rewards = np.array(rewards)

    # Save
    output_dir = os.path.expanduser("~/isaac_sim_datasets")
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, "franka_transitions_200.npz")
    np.savez_compressed(
        npz_path,
        observations=observations,
        actions=actions_arr,
        rewards=rewards,
        robot_name=robot_name,
        num_transitions=num_transitions,
        observation_dim=observations.shape[1],
        action_dim=actions_arr.shape[1],
    )
    npz_size = os.path.getsize(npz_path)
    print(f"\n💾 Saved transitions: {npz_path}")
    print(f"   Size: {npz_size / 1024:.1f} KB")
    print(f"   Observations shape: {observations.shape}")
    print(f"   Actions shape: {actions_arr.shape}")
    print(f"   Rewards shape: {rewards.shape}")
    print(f"   Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]")

    # Save RGB frames
    if rgb_frames:
        rgb_array = np.array(rgb_frames)
        rgb_path = os.path.join(output_dir, "franka_rgb_frames.npz")
        np.savez_compressed(rgb_path, frames=rgb_array)
        rgb_size = os.path.getsize(rgb_path)
        print(f"\n🎨 Saved RGB frames: {rgb_path}")
        print(f"   Frames: {rgb_array.shape[0]}, Shape: {rgb_array.shape[1:]}")
        print(f"   Size: {rgb_size / 1024:.1f} KB")

    # Save depth frames
    if depth_frames:
        depth_array = np.array(depth_frames)
        depth_path = os.path.join(output_dir, "franka_depth_frames.npz")
        np.savez_compressed(depth_path, frames=depth_array)
        depth_size = os.path.getsize(depth_path)
        print(f"\n🌊 Saved depth frames: {depth_path}")
        print(f"   Frames: {depth_array.shape[0]}, Shape: {depth_array.shape[1:]}")
        print(f"   Size: {depth_size / 1024:.1f} KB")

    # Cleanup
    sim.destroy()

    # SO-100 dataset
    try:
        print("\n\n" + "=" * 60)
        print("Generating SO-100 dataset...")
        print("=" * 60)

        sim2 = Simulation()
        sim2.create_world(timestep=0.002, gravity=-9.81)
        sim2.add_robot(name="so100_robot", data_config="so100")

        model2 = sim2.mj_model
        data2 = sim2.mj_data
        if model2 is None:
            model2 = sim2._world._model
            data2 = sim2._world._data

        n_joints2 = model2.nu
        n_qpos2 = model2.nq
        print(f"   SO-100 Actuators: {n_joints2}, DOF: {n_qpos2}")

        so100_obs = []
        so100_actions = []
        so100_rewards = []

        for step_i in range(num_transitions):
            qpos = data2.qpos.copy()
            qvel = data2.qvel.copy()
            obs = np.concatenate([qpos, qvel]).astype(np.float32)
            so100_obs.append(obs)

            action = np.random.uniform(-0.1, 0.1, size=n_joints2).astype(np.float32)
            so100_actions.append(action)

            data2.ctrl[:n_joints2] = action
            for _ in range(5):
                mujoco.mj_step(model2, data2)

            reward = -np.sum(np.square(qpos[:min(3, len(qpos))]))
            so100_rewards.append(reward)

        so100_obs = np.array(so100_obs)
        so100_actions = np.array(so100_actions)
        so100_rewards = np.array(so100_rewards)

        so100_path = os.path.join(output_dir, "so100_transitions_200.npz")
        np.savez_compressed(
            so100_path,
            observations=so100_obs,
            actions=so100_actions,
            rewards=so100_rewards,
            robot_name="so100",
            num_transitions=num_transitions,
            observation_dim=so100_obs.shape[1],
            action_dim=so100_actions.shape[1],
        )
        so100_size = os.path.getsize(so100_path)
        print(f"💾 Saved SO-100 transitions: {so100_path}")
        print(f"   Size: {so100_size / 1024:.1f} KB")
        print(f"   Observations shape: {so100_obs.shape}")
        print(f"   Actions shape: {so100_actions.shape}")

        sim2.destroy()
    except Exception as e:
        print(f"⚠ SO-100 dataset generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print("\n" + "=" * 60)
    print("Dataset Generation Summary")
    print("=" * 60)
    files = os.listdir(output_dir)
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in files)
    print(f"Output directory: {output_dir}")
    print(f"Files: {len(files)}")
    for f in sorted(files):
        fpath = os.path.join(output_dir, f)
        print(f"  {f}: {os.path.getsize(fpath) / 1024:.1f} KB")
    print(f"Total size: {total_size / 1024:.1f} KB")
    print("✅ Dataset generation complete!")


if __name__ == "__main__":
    generate_dataset()
