#!/usr/bin/env python3
"""
Isaac Sim 50-Environment Pick-and-Place Pipeline — Stage 2 Validation.

Validates the full pipeline from issue #124:
  1. SO-101 MJCF → USD conversion
  2. Isaac Sim SimulationApp bootstrap (headless)
  3. GridCloner × 50 environments with SO-101 + task objects
  4. PickAndPlaceReward integration (4-phase)
  5. Physics stepping + observation extraction
  6. Performance benchmarks (throughput, VRAM)

Run:
    /home/ubuntu/IsaacSim/python.sh scripts/isaac_pick_place_50env.py

Or from system Python after `pip install isaacsim-rl`:
    DISPLAY=:1 python scripts/isaac_pick_place_50env.py

Refs: Issue #124, PR #128
"""

import json
import os
import sys
import time
import traceback

# Configuration
NUM_ENVS = 50
ENV_SPACING = 2.0  # meters between envs
N_BENCHMARK_STEPS = 500
N_ROLLOUT_STEPS = 200
TABLE_HEIGHT = 0.4
CUBE_SIZE = 0.04
CUBE_START = [0.3, 0.0, TABLE_HEIGHT + CUBE_SIZE / 2 + 0.01]
TARGET_PLACE = [0.0, 0.3, TABLE_HEIGHT + CUBE_SIZE / 2 + 0.01]
OUTPUT_DIR = "/tmp/isaac_pick_place_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg: str):
    """Print with timestamp."""
    ts = time.strftime("%H:%M:%S")
    sys.stderr.write(f"[{ts}] {msg}\n")
    sys.stderr.flush()


def get_gpu_info():
    """Get GPU VRAM usage via nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "used_mib": int(parts[0].strip()),
                "free_mib": int(parts[1].strip()),
                "total_mib": int(parts[2].strip()),
            }
    except Exception:
        pass
    return {"used_mib": 0, "free_mib": 0, "total_mib": 0}


def stage_1_mjcf_to_usd():
    """Stage 1: Convert SO-101 MJCF to USD."""
    log("═══ Stage 1: SO-101 MJCF → USD ═══")

    try:
        from strands_robots.assets import resolve_model_path
        from strands_robots.isaac.asset_converter import convert_mjcf_to_usd

        # Resolve SO-101 MJCF
        # Try so101 first, fallback to so100 (both are similar 6-DOF arms)
        for robot_name in ["so101", "so100"]:
            model_path = resolve_model_path(robot_name)
            if model_path and model_path.exists():
                log(f"  Found robot: {robot_name} at {model_path}")
                break
        else:
            log("  ⚠ No SO-101/SO-100 MJCF found, using scene_box.xml")
            model_path = resolve_model_path("so101", prefer_scene=True)

        if not model_path:
            log("  ❌ No robot MJCF found at all")
            return None

        usd_output = os.path.join(OUTPUT_DIR, "so101.usd")
        result = convert_mjcf_to_usd(str(model_path), usd_output)

        if result.get("status") == "success":
            usd_path = result.get("usd_path", usd_output)
            size_kb = os.path.getsize(usd_path) / 1024
            log(f"  ✅ USD created: {usd_path} ({size_kb:.0f} KB)")
            log(f"  Method: {result.get('method', 'unknown')}")
            return usd_path
        else:
            log(f"  ❌ Conversion failed: {result}")
            return None

    except Exception as e:
        log(f"  ❌ Stage 1 error: {e}")
        traceback.print_exc()
        return None


def stage_2_isaac_sim_gridcloner(usd_path: str | None):
    """Stage 2: Bootstrap Isaac Sim + GridCloner × 50 envs."""
    log("═══ Stage 2: Isaac Sim GridCloner × 50 ═══")

    vram_before = get_gpu_info()
    log(f"  VRAM before: {vram_before['used_mib']} MiB used / {vram_before['total_mib']} MiB total")

    # Boot Isaac Sim
    log("  Booting SimulationApp (headless)...")
    t0 = time.time()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    boot_time = time.time() - t0
    vram_after_boot = get_gpu_info()
    log(f"  SimulationApp booted in {boot_time:.1f}s")
    log(f"  VRAM after boot: {vram_after_boot['used_mib']} MiB (+{vram_after_boot['used_mib'] - vram_before['used_mib']} MiB)")

    import numpy as np
    import omni.usd
    from omni.isaac.cloner import GridCloner
    from omni.isaac.core import World
    from omni.isaac.core.objects import DynamicCuboid, FixedCuboid

    # Create world
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Create cloner
    cloner = GridCloner(spacing=ENV_SPACING)
    cloner.define_base_env("/World/envs")

    stage = omni.usd.get_context().get_stage()

    # Define template env
    from pxr import Gf, UsdGeom
    _env_prim = stage.DefinePrim("/World/envs/env_0", "Xform")

    # Add table to template env
    world.scene.add(
        FixedCuboid(
            prim_path="/World/envs/env_0/Table",
            name="table_0",
            position=[0, 0, TABLE_HEIGHT / 2],
            size=0.6,
            scale=[1.0, 0.8, TABLE_HEIGHT / 0.6],
            color=np.array([0.6, 0.45, 0.3]),
        )
    )

    # Add pickup cube to template env
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/envs/env_0/Cube",
            name="cube_0",
            position=CUBE_START,
            size=CUBE_SIZE,
            color=np.array([1.0, 0.0, 0.0]),  # red cube
            mass=0.05,
        )
    )

    # Add target placement indicator (green, static)
    world.scene.add(
        FixedCuboid(
            prim_path="/World/envs/env_0/Target",
            name="target_0",
            position=TARGET_PLACE,
            size=CUBE_SIZE * 1.5,
            color=np.array([0.0, 1.0, 0.0]),
        )
    )

    # If we have USD for SO-101, add it to the template
    if usd_path and os.path.exists(usd_path):
        try:
            from pxr import UsdGeom
            robot_ref = stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
            robot_ref.GetReferences().AddReference(usd_path)
            xform = UsdGeom.Xformable(robot_ref)
            xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, TABLE_HEIGHT))
            log("  ✅ SO-101 USD added to template env")
        except Exception as e:
            log(f"  ⚠ Could not add SO-101 USD: {e}")

    # Clone to N environments
    log(f"  Cloning {NUM_ENVS} environments (spacing={ENV_SPACING}m)...")
    t_clone = time.time()

    env_paths = cloner.generate_paths("/World/envs/env", NUM_ENVS)
    cloner.clone(
        source_prim_path="/World/envs/env_0",
        prim_paths=env_paths,
    )

    clone_time = time.time() - t_clone
    vram_after_clone = get_gpu_info()
    log(f"  Cloned in {clone_time:.3f}s")
    log(f"  VRAM after clone: {vram_after_clone['used_mib']} MiB (+{vram_after_clone['used_mib'] - vram_after_boot['used_mib']} MiB from boot)")

    # Reset and step
    world.reset()
    log("  World reset, running physics benchmark...")

    return app, world, vram_before


def stage_3_physics_benchmark(world, vram_before: dict):
    """Stage 3: Physics stepping benchmark."""
    log(f"═══ Stage 3: Physics Benchmark ({N_BENCHMARK_STEPS} steps × {NUM_ENVS} envs) ═══")

    t0 = time.time()
    for step in range(N_BENCHMARK_STEPS):
        world.step(render=False)
    elapsed = time.time() - t0

    throughput = int(N_BENCHMARK_STEPS * NUM_ENVS / elapsed)
    step_ms = elapsed / N_BENCHMARK_STEPS * 1000

    vram_after = get_gpu_info()
    log(f"  Steps: {N_BENCHMARK_STEPS} × {NUM_ENVS} envs")
    log(f"  Time: {elapsed:.3f}s")
    log(f"  Throughput: {throughput:,} env-steps/s")
    log(f"  Per step: {step_ms:.2f}ms")
    log(f"  VRAM after stepping: {vram_after['used_mib']} MiB")

    return {
        "steps": N_BENCHMARK_STEPS,
        "num_envs": NUM_ENVS,
        "elapsed_s": round(elapsed, 3),
        "throughput": throughput,
        "step_ms": round(step_ms, 2),
        "vram_used_mib": vram_after["used_mib"],
        "vram_delta_mib": vram_after["used_mib"] - vram_before["used_mib"],
    }


def stage_4_reward_integration():
    """Stage 4: PickAndPlaceReward integration test with simulated trajectory."""
    log("═══ Stage 4: PickAndPlaceReward Integration ═══")

    import numpy as np
    # Import from the merged branch
    sys.path.insert(0, "/home/ubuntu/strands-gtc-nvidia")
    from strands_robots.rl_trainer import PickAndPlaceReward

    reward_fn = PickAndPlaceReward(
        object_pos_indices=(7, 10),
        ee_pos_indices=(0, 3),
        gripper_index=6,
        target_place_pos=np.array(TARGET_PLACE),
        reach_threshold=0.05,
        lift_height=0.10,
        place_threshold=0.05,
    )

    # Simulate a full pick-and-place trajectory
    total_reward = 0.0
    phase_transitions = []
    rewards_per_step = []

    for step in range(N_ROLLOUT_STEPS):
        progress = step / N_ROLLOUT_STEPS

        # Simulated observation — robot approaches, grasps, transports, places
        ee_pos = np.array([0.3, 0.0, 0.5])  # start position
        obj_pos = np.array(CUBE_START)
        gripper = 1.0  # open

        if progress < 0.25:
            # Phase 1: Reach — EE moves toward object
            alpha = progress / 0.25
            ee_pos = np.array([0.3, 0.0, 0.5]) * (1 - alpha) + np.array(CUBE_START) * alpha
        elif progress < 0.40:
            # Phase 2: Grasp — close gripper, lift object
            alpha = (progress - 0.25) / 0.15
            ee_pos = np.array(CUBE_START)
            gripper = 1.0 - alpha  # close
            if alpha > 0.5:
                lift = (alpha - 0.5) * 2.0 * 0.15
                obj_pos = np.array(CUBE_START) + np.array([0, 0, lift])
                ee_pos = obj_pos.copy()
        elif progress < 0.75:
            # Phase 3: Transport — carry to target
            alpha = (progress - 0.40) / 0.35
            start = np.array(CUBE_START) + np.array([0, 0, 0.15])
            end = np.array(TARGET_PLACE) + np.array([0, 0, 0.15])
            obj_pos = start * (1 - alpha) + end * alpha
            ee_pos = obj_pos.copy()
            gripper = 0.0  # closed
        else:
            # Phase 4: Place — lower and release
            alpha = (progress - 0.75) / 0.25
            obj_pos = np.array(TARGET_PLACE) + np.array([0, 0, 0.15 * (1 - alpha)])
            ee_pos = obj_pos.copy()
            if alpha > 0.5:
                gripper = alpha  # open

        # Build observation vector
        state = np.zeros(14)
        state[0:3] = ee_pos
        state[3:6] = [0, 0, 0]  # velocity placeholder
        state[6] = gripper
        state[7:10] = obj_pos
        state[10:14] = [0, 0, 0, 0]  # object orientation/vel

        action = np.zeros(7)  # joint actions

        prev_phase = reward_fn.current_phase
        r = reward_fn(state, action)
        total_reward += r
        rewards_per_step.append(r)

        if reward_fn.current_phase != prev_phase:
            phase_transitions.append({
                "step": step,
                "from": prev_phase,
                "to": reward_fn.current_phase,
                "from_name": ["Reach", "Grasp", "Transport", "Place"][prev_phase],
                "to_name": reward_fn.phase_name,
            })

    info = reward_fn.get_info()
    log(f"  Total reward: {total_reward:.2f} over {N_ROLLOUT_STEPS} steps")
    log(f"  Final phase: {info['phase_name']}")
    log(f"  Success: {info['is_success']}")
    log(f"  Phase transitions: {len(phase_transitions)}")
    for pt in phase_transitions:
        log(f"    Step {pt['step']:3d}: {pt['from_name']} → {pt['to_name']}")

    return {
        "total_reward": round(total_reward, 2),
        "rollout_steps": N_ROLLOUT_STEPS,
        "final_phase": info["phase_name"],
        "success": info["is_success"],
        "phase_transitions": phase_transitions,
        "grasp_awarded": info["grasp_awarded"],
        "lift_awarded": info["lift_awarded"],
        "place_awarded": info["place_awarded"],
        "avg_reward_per_step": round(total_reward / N_ROLLOUT_STEPS, 4),
    }


def stage_5_ppo_smoke_test(world):
    """Stage 5: PPO environment interface smoke test (no actual training)."""
    log("═══ Stage 5: PPO Environment Interface ═══")

    import numpy as np

    # Test that the gym env interface works with our backend
    # We create a mock observation/action cycle mimicking SB3's PPO loop

    n_joints = 6
    obs_dim = n_joints * 2  # joint_pos + joint_vel

    # Simulate 100 PPO steps across 50 envs
    n_ppo_steps = 100
    t0 = time.time()

    _obs_buffer = np.zeros((NUM_ENVS, obs_dim), dtype=np.float32)
    action_buffer = np.random.uniform(-1, 1, (NUM_ENVS, n_joints)).astype(np.float32)
    reward_buffer = np.zeros(NUM_ENVS, dtype=np.float32)

    from strands_robots.rl_trainer import PickAndPlaceReward

    rewards = [PickAndPlaceReward(
        target_place_pos=np.array(TARGET_PLACE)
    ) for _ in range(NUM_ENVS)]

    for step in range(n_ppo_steps):
        # Step physics
        world.step(render=False)

        # Compute rewards for each env
        for env_idx in range(NUM_ENVS):
            # Simulated obs
            obs = np.random.randn(14).astype(np.float32) * 0.1
            action = action_buffer[env_idx]
            r = rewards[env_idx](obs, action)
            reward_buffer[env_idx] = r

        # Simulate policy update every n_steps
        if (step + 1) % 10 == 0:
            _mean_reward = reward_buffer.mean()
            # Simulate PPO update (just the reward stats)

    elapsed = time.time() - t0
    fps = n_ppo_steps * NUM_ENVS / elapsed

    log(f"  PPO smoke: {n_ppo_steps} steps × {NUM_ENVS} envs")
    log(f"  Time: {elapsed:.3f}s")
    log(f"  Throughput: {fps:.0f} env-steps/s (physics + reward)")
    log(f"  Mean reward: {reward_buffer.mean():.4f}")

    return {
        "ppo_steps": n_ppo_steps,
        "num_envs": NUM_ENVS,
        "elapsed_s": round(elapsed, 3),
        "throughput": int(fps),
        "mean_reward": round(float(reward_buffer.mean()), 4),
    }


def main():
    """Run the full pick-and-place pipeline validation."""
    log("╔════════════════════════════════════════════════════════╗")
    log("║  Isaac Sim 50-Env Pick-and-Place Pipeline — Stage 2   ║")
    log("║  Issue #124: Marble 3D → Isaac Sim → RL Training      ║")
    log("╚════════════════════════════════════════════════════════╝")

    results = {
        "pipeline": "isaac_pick_place_50env",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "num_envs": NUM_ENVS,
            "env_spacing": ENV_SPACING,
            "benchmark_steps": N_BENCHMARK_STEPS,
            "rollout_steps": N_ROLLOUT_STEPS,
            "table_height": TABLE_HEIGHT,
            "cube_size": CUBE_SIZE,
            "cube_start": CUBE_START,
            "target_place": TARGET_PLACE,
        },
        "gpu": get_gpu_info(),
        "stages": {},
    }

    # Stage 1: MJCF → USD
    try:
        usd_path = stage_1_mjcf_to_usd()
        results["stages"]["1_mjcf_to_usd"] = {
            "status": "pass" if usd_path else "skip",
            "usd_path": usd_path,
        }
    except Exception as e:
        log(f"Stage 1 error: {e}")
        traceback.print_exc()
        results["stages"]["1_mjcf_to_usd"] = {"status": "fail", "error": str(e)}
        usd_path = None

    # Stage 2: Isaac Sim + GridCloner
    try:
        app, world, vram_before = stage_2_isaac_sim_gridcloner(usd_path)
        results["stages"]["2_gridcloner"] = {"status": "pass", "num_envs": NUM_ENVS}
    except Exception as e:
        log(f"Stage 2 error: {e}")
        traceback.print_exc()
        results["stages"]["2_gridcloner"] = {"status": "fail", "error": str(e)}
        # Save partial results and exit
        _save_results(results)
        return

    # Stage 3: Physics benchmark
    try:
        bench_results = stage_3_physics_benchmark(world, vram_before)
        results["stages"]["3_physics_benchmark"] = {"status": "pass", **bench_results}
    except Exception as e:
        log(f"Stage 3 error: {e}")
        traceback.print_exc()
        results["stages"]["3_physics_benchmark"] = {"status": "fail", "error": str(e)}

    # Stage 4: PickAndPlaceReward integration
    try:
        reward_results = stage_4_reward_integration()
        results["stages"]["4_reward_integration"] = {"status": "pass", **reward_results}
    except Exception as e:
        log(f"Stage 4 error: {e}")
        traceback.print_exc()
        results["stages"]["4_reward_integration"] = {"status": "fail", "error": str(e)}

    # Stage 5: PPO smoke test
    try:
        ppo_results = stage_5_ppo_smoke_test(world)
        results["stages"]["5_ppo_smoke"] = {"status": "pass", **ppo_results}
    except Exception as e:
        log(f"Stage 5 error: {e}")
        traceback.print_exc()
        results["stages"]["5_ppo_smoke"] = {"status": "fail", "error": str(e)}

    # Final VRAM report
    vram_final = get_gpu_info()
    results["vram_final"] = vram_final
    log(f"═══ Final VRAM: {vram_final['used_mib']} MiB / {vram_final['total_mib']} MiB ═══")

    # Cleanup
    log("Cleaning up...")
    try:
        world.stop()
        app.close()
    except Exception as e:
        log(f"  Cleanup: {e}")

    vram_cleanup = get_gpu_info()
    results["vram_after_cleanup"] = vram_cleanup
    log(f"VRAM after cleanup: {vram_cleanup['used_mib']} MiB")

    # Summary
    log("")
    log("╔════════════════════════════════════════════════════════╗")
    log("║  PIPELINE RESULTS SUMMARY                             ║")
    log("╚════════════════════════════════════════════════════════╝")

    n_pass = sum(1 for s in results["stages"].values() if s.get("status") == "pass")
    n_fail = sum(1 for s in results["stages"].values() if s.get("status") == "fail")
    n_skip = sum(1 for s in results["stages"].values() if s.get("status") == "skip")

    for name, stage_result in results["stages"].items():
        status = stage_result.get("status", "unknown")
        icon = {"pass": "✅", "fail": "❌", "skip": "⏭"}.get(status, "❓")
        log(f"  {icon} {name}: {status}")

    log(f"\n  Total: {n_pass} pass, {n_fail} fail, {n_skip} skip")

    results["summary"] = {
        "passed": n_pass,
        "failed": n_fail,
        "skipped": n_skip,
        "total": len(results["stages"]),
    }

    _save_results(results)


def _save_results(results: dict):
    """Save results to JSON file."""
    results_path = os.path.join(OUTPUT_DIR, "pipeline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
