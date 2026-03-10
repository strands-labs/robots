#!/usr/bin/env python3
"""
Newton GPU-Accelerated Physics Simulation Tool for Strands Agents.

This AgentTool wraps NewtonBackend to give AI agents full control over
GPU-accelerated physics simulation via natural language. Like simulation.py
but powered by Newton (7 solvers, 4096+ parallel envs, differentiable).

Usage via Agent:
    "Create a Newton world with 4096 parallel quadrupeds using MuJoCo solver"
    "Step the simulation 100 times and show me the joint positions"
    "Run differentiable optimization to find the best initial velocity"
    "Add a cloth mesh and simulate it with VBD solver"

Actions:
    create_world    — Initialize simulation world
    add_robot       — Add robot from URDF/MJCF/USD
    add_cloth       — Add cloth mesh for soft-body sim
    add_cable       — Add cable/rope
    add_particles   — Add particles for MPM/granular
    replicate       — Clone scene to N parallel environments
    step            — Advance simulation
    observe         — Get robot observations
    reset           — Reset to initial state
    get_state       — Full state snapshot
    run_policy      — Run policy loop
    run_diffsim     — Differentiable sim optimization
    add_sensor      — Add contact/IMU/camera sensor
    read_sensor     — Read sensor data
    solve_ik        — Inverse kinematics
    destroy         — Tear down simulation
    list_assets     — List available Newton bundled assets
    benchmark       — Run performance benchmark
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from strands import tool

logger = logging.getLogger(__name__)

# Global backend instance (one per agent session)
_backend = None
_backend_config = None


def _get_backend(config_overrides: Optional[Dict] = None):
    """Get or create the global NewtonBackend instance."""
    global _backend, _backend_config
    if _backend is not None:
        return _backend

    from strands_robots.newton import NewtonBackend, NewtonConfig

    kwargs = {
        "solver": "mujoco",
        "device": "cuda:0",
        "num_envs": 1,
        "broad_phase": "explicit",
    }
    if config_overrides:
        kwargs.update(config_overrides)

    config = NewtonConfig(**kwargs)
    _backend = NewtonBackend(config)
    _backend_config = kwargs
    return _backend


def _destroy_backend():
    """Destroy the global backend."""
    global _backend, _backend_config
    if _backend is not None:
        _backend.destroy()
        _backend = None
        _backend_config = None


@tool
def newton_sim(
    action: str,
    name: str = "",
    urdf_path: str = "",
    usd_path: str = "",
    position: str = "0,0,0",
    num_envs: int = 1,
    num_steps: int = 1,
    solver: str = "mujoco",
    device: str = "cuda:0",
    robot_name: str = "",
    policy_provider: str = "mock",
    instruction: str = "",
    duration: float = 10.0,
    enable_differentiable: bool = False,
    lr: float = 0.02,
    iterations: int = 100,
    optimize_param: str = "initial_velocity",
    sensor_type: str = "contact",
    env_ids: str = "",
    broad_phase: str = "explicit",
    ground_plane: bool = True,
    scale: float = 1.0,
    data_config: str = "",
    # Cloth params
    density: float = 0.02,
    particle_radius: float = 0.01,
    # Benchmark
    benchmark_envs: int = 4096,
    benchmark_steps: int = 100,
) -> Dict[str, Any]:
    """
    Newton GPU-Accelerated Physics Simulation Tool.

    Provides AI agents with full control over GPU-accelerated physics simulation
    powered by Newton (NVIDIA Warp). Supports 7 solver backends, 4096+ parallel
    environments, differentiable simulation, cloth/cable/MPM, and sensors.

    Args:
        action: The action to perform. One of:
            - "create_world": Initialize simulation world
            - "add_robot": Add robot from URDF/MJCF/USD
            - "add_cloth": Add cloth mesh
            - "add_cable": Add cable/rope
            - "add_particles": Add particles (MPM/granular)
            - "replicate": Clone scene to N parallel environments
            - "step": Advance simulation by num_steps
            - "observe": Get robot observations
            - "reset": Reset to initial state
            - "get_state": Full state snapshot
            - "run_policy": Run policy loop for duration
            - "run_diffsim": Differentiable sim optimization
            - "add_sensor": Add sensor (contact/IMU/camera)
            - "read_sensor": Read sensor data
            - "solve_ik": Inverse kinematics
            - "destroy": Tear down simulation
            - "list_assets": List Newton bundled assets
            - "benchmark": Run performance benchmark
        name: Name for robot/cloth/cable/sensor
        urdf_path: Path to URDF file
        usd_path: Path to USD file
        position: Position as "x,y,z" string
        num_envs: Number of parallel environments for replicate
        num_steps: Number of simulation steps
        solver: Physics solver (mujoco, featherstone, semi_implicit, xpbd, vbd, style3d, implicit_mpm)
        device: Compute device (cuda:0, cpu)
        robot_name: Robot name for observe/policy/ik actions
        policy_provider: Policy provider for run_policy
        instruction: Natural language instruction for policy
        duration: Duration in seconds for run_policy
        enable_differentiable: Enable gradient flow for diffsim
        lr: Learning rate for diffsim optimization
        iterations: Optimization iterations for diffsim
        optimize_param: Parameter to optimize (initial_velocity, initial_position, control)
        sensor_type: Sensor type (contact, imu, tiled_camera)
        env_ids: Comma-separated env IDs for per-env reset
        broad_phase: Collision broad-phase (explicit, sap, nxn)
        ground_plane: Add ground plane
        scale: Scale factor for robot
        data_config: JSON string of extra config
        density: Density for cloth/particles
        particle_radius: Particle radius
        benchmark_envs: Number of envs for benchmark
        benchmark_steps: Number of steps for benchmark

    Returns:
        Dict with status and content
    """
    try:
        pos = tuple(float(x) for x in position.split(",")) if position else (0, 0, 0)
        dc = json.loads(data_config) if data_config else {}

        if action == "create_world":
            _destroy_backend()
            config = {
                "solver": solver,
                "device": device,
                "num_envs": num_envs,
                "broad_phase": broad_phase,
                "enable_differentiable": enable_differentiable,
            }
            backend = _get_backend(config)
            result = backend.create_world(ground_plane=ground_plane)
            text = (
                f"🌍 Newton world created\n"
                f"⚡ Solver: {solver} | Device: {device}\n"
                f"🔧 Broad-phase: {broad_phase} | Ground: {ground_plane}\n"
                f"{'🧮 Differentiable: ON' if enable_differentiable else ''}"
            )
            return {"status": "success", "content": [{"text": text}]}

        elif action == "add_robot":
            backend = _get_backend()
            result = backend.add_robot(
                name=name or "robot",
                urdf_path=urdf_path or None,
                usd_path=usd_path or None,
                data_config=dc,
                position=pos,
                scale=scale,
            )
            if result["success"]:
                ri = result["robot_info"]
                text = (
                    f"🤖 Robot '{ri['name']}' added\n"
                    f"📁 {ri['format'].upper()}: {ri['model_path']}\n"
                    f"🦾 Joints: {ri['num_joints']} | Bodies: {ri['num_bodies']}\n"
                    f"📍 Position: {ri['position']}"
                )
            else:
                text = f"❌ {result['message']}"
            return {"status": "success" if result["success"] else "error", "content": [{"text": text}]}

        elif action == "add_cloth":
            backend = _get_backend()
            result = backend.add_cloth(
                name=name or "cloth",
                usd_path=usd_path or None,
                position=pos,
                density=density,
                particle_radius=particle_radius,
            )
            text = f"{'🧵' if result['success'] else '❌'} {result['message']}"
            return {"status": "success" if result["success"] else "error", "content": [{"text": text}]}

        elif action == "add_cable":
            backend = _get_backend()
            end_pos = dc.get("end", (1, 0, 1))
            result = backend.add_cable(name=name or "cable", start=pos, end=tuple(end_pos))
            text = f"{'🔗' if result['success'] else '❌'} {result['message']}"
            return {"status": "success" if result["success"] else "error", "content": [{"text": text}]}

        elif action == "add_particles":
            backend = _get_backend()
            positions = dc.get("positions", [list(pos)])
            result = backend.add_particles(name=name or "particles", positions=positions, mass=density)
            text = f"{'⚛️' if result['success'] else '❌'} {result['message']}"
            return {"status": "success" if result["success"] else "error", "content": [{"text": text}]}

        elif action == "replicate":
            backend = _get_backend()
            result = backend.replicate(num_envs=num_envs)
            if result["success"]:
                ei = result["env_info"]
                text = (
                    f"🔄 Replicated to {ei['num_envs']} environments\n"
                    f"📊 Total bodies: {ei['bodies_total']} | Joints: {ei['joints_total']}\n"
                    f"🔧 Solver: {ei['solver']} | Device: {ei['device']}"
                )
            else:
                text = f"❌ {result['message']}"
            return {"status": "success" if result["success"] else "error", "content": [{"text": text}]}

        elif action == "step":
            backend = _get_backend()
            t0 = time.time()
            for _ in range(num_steps):
                result = backend.step()
                if not result.get("success"):
                    return {"status": "error", "content": [{"text": f"❌ Step failed: {result.get('error')}"}]}
            elapsed = time.time() - t0
            text = (
                f"⏩ {num_steps} steps completed\n"
                f"⏱️ {elapsed:.3f}s | Sim time: {result['sim_time']:.4f}s\n"
                f"📈 Step count: {result['step_count']}"
            )
            return {"status": "success", "content": [{"text": text}]}

        elif action == "observe":
            backend = _get_backend()
            result = backend.get_observation(robot_name or None)
            if result["success"]:
                obs_text = []
                for rname, obs in result["observations"].items():
                    jp = obs.get("joint_positions")
                    jp_str = (
                        f"[{', '.join(f'{x:.3f}' for x in jp[:6])}{'...' if len(jp) > 6 else ''}]"
                        if jp is not None
                        else "N/A"
                    )
                    obs_text.append(f"  🤖 {rname}: joints={jp_str}")
                text = f"👁️ Observations (t={result['sim_time']:.4f}s):\n" + "\n".join(obs_text)
            else:
                text = "❌ No observations available"
            return {"status": "success", "content": [{"text": text}]}

        elif action == "reset":
            backend = _get_backend()
            eids = [int(x) for x in env_ids.split(",") if x.strip()] if env_ids else None
            result = backend.reset(env_ids=eids)
            text = f"{'🔄' if result['success'] else '❌'} {result['message']}"
            return {"status": "success" if result["success"] else "error", "content": [{"text": text}]}

        elif action == "get_state":
            backend = _get_backend()
            state = backend.get_state()
            text = (
                f"📊 Newton State:\n"
                f"  Solver: {state['config']['solver']} | Device: {state['config']['device']}\n"
                f"  Envs: {state['config']['num_envs']} | Steps: {state['step_count']}\n"
                f"  Sim time: {state['sim_time']:.4f}s\n"
                f"  Robots: {list(state['robots'].keys())}\n"
                f"  Cloths: {list(state['cloths'].keys())}\n"
                f"  Sensors: {state['sensors']}\n"
                f"  Joints/world: {state['joints_per_world']} | Bodies/world: {state['bodies_per_world']}"
            )
            return {"status": "success", "content": [{"text": text}]}

        elif action == "run_policy":
            backend = _get_backend()
            result = backend.run_policy(
                robot_name=robot_name or name,
                policy_provider=policy_provider,
                instruction=instruction,
                duration=duration,
            )
            text = (
                f"{'✅' if result['success'] else '❌'} Policy '{policy_provider}' on '{robot_name or name}'\n"
                f"  Steps: {result['steps_executed']} | Wall: {result['wall_time']:.2f}s\n"
                f"  Realtime factor: {result['realtime_factor']:.1f}×\n"
                f"  Errors: {len(result['errors'])}"
            )
            return {"status": "success" if result["success"] else "error", "content": [{"text": text}]}

        elif action == "run_diffsim":
            backend = _get_backend()
            result = backend.run_diffsim(
                num_steps=num_steps,
                lr=lr,
                iterations=iterations,
                optimize_param=optimize_param,
                verbose=True,
            )
            if result["success"]:
                text = (
                    f"🧮 DiffSim optimization complete\n"
                    f"  Iterations: {result['iterations']}\n"
                    f"  Final loss: {result['final_loss']:.6f}\n"
                    f"  Param: {result['optimize_param']}"
                )
            else:
                text = f"❌ {result['message']}"
            return {"status": "success" if result["success"] else "error", "content": [{"text": text}]}

        elif action == "add_sensor":
            backend = _get_backend()
            result = backend.add_sensor(name=name or "sensor", sensor_type=sensor_type)
            text = f"{'📡' if result['success'] else '❌'} {result['message']}"
            return {"status": "success" if result["success"] else "error", "content": [{"text": text}]}

        elif action == "read_sensor":
            backend = _get_backend()
            result = backend.read_sensor(name=name or "sensor")
            text = f"{'📡' if result['success'] else '❌'} Sensor '{name}': {result.get('data', result.get('message'))}"
            return {"status": "success" if result["success"] else "error", "content": [{"text": text}]}

        elif action == "solve_ik":
            backend = _get_backend()
            result = backend.solve_ik(
                robot_name=robot_name or name,
                target_position=pos,
            )
            if result["success"]:
                text = f"🎯 IK converged in {result['iterations']} iters (error={result['error']:.4f}m)"
            else:
                text = f"❌ {result['message']}"
            return {"status": "success" if result["success"] else "error", "content": [{"text": text}]}

        elif action == "enable_dual_solver":
            backend = _get_backend()
            cloth_solver = data_config if data_config and not data_config.startswith("{") else "vbd"
            result = backend.enable_dual_solver(
                rigid_solver=solver,
                cloth_solver=cloth_solver,
            )
            text = f"{'⚡' if result['success'] else '❌'} {result['message']}"
            return {"status": "success" if result["success"] else "error", "content": [{"text": text}]}

        elif action == "destroy":
            if _backend is not None:
                result = _backend.destroy()
                _destroy_backend()
                text = f"💥 {result['message']}"
            else:
                text = "No active backend to destroy."
            return {"status": "success", "content": [{"text": text}]}

        elif action == "list_assets":
            try:
                import os

                import newton.examples as ne

                asset_dir = ne.get_asset_directory()
                assets = sorted(os.listdir(asset_dir))
                urdf = [a for a in assets if a.endswith(".urdf")]
                usd = [a for a in assets if a.endswith((".usd", ".usda"))]
                xml = [a for a in assets if a.endswith(".xml")]
                text = (
                    f"📦 Newton Bundled Assets ({len(assets)} files):\n"
                    f"  URDF: {', '.join(urdf)}\n"
                    f"  USD:  {', '.join(usd)}\n"
                    f"  MJCF: {', '.join(xml)}\n"
                    f"  Dir:  {asset_dir}"
                )
            except ImportError:
                text = "❌ Newton not installed — cannot list assets"
            return {"status": "success", "content": [{"text": text}]}

        elif action == "benchmark":
            _destroy_backend()
            config = {"solver": solver, "device": device, "num_envs": benchmark_envs, "broad_phase": broad_phase}
            backend = _get_backend(config)
            backend.create_world()

            # Try to add a quadruped
            try:
                import newton.examples as ne

                backend.add_robot("bench_robot", urdf_path=ne.get_asset("quadruped.urdf"))
            except Exception:
                backend._lazy_init()
                import warp as wp

                backend._builder.add_body(xform=wp.transform((0, 0, 0.5), wp.quat_identity()))
                backend._builder.add_shape_sphere(body=0, radius=0.1)
                backend._builder.add_joint_free(child=0)
                backend._model = None

            backend.replicate(num_envs=benchmark_envs)

            # Warmup
            for _ in range(5):
                backend.step()

            # Benchmark
            t0 = time.time()
            for _ in range(benchmark_steps):
                backend.step()
            elapsed = time.time() - t0

            throughput = int(benchmark_steps * benchmark_envs / elapsed)
            ms_per_step = elapsed / benchmark_steps * 1000

            text = (
                f"🏎️ Newton Benchmark Results:\n"
                f"  Solver: {solver} | Device: {device}\n"
                f"  Envs: {benchmark_envs} | Steps: {benchmark_steps}\n"
                f"  ─────────────────────────\n"
                f"  Total time: {elapsed:.3f}s\n"
                f"  Per step: {ms_per_step:.2f}ms\n"
                f"  Throughput: {throughput:,} env-steps/s\n"
                f"  ─────────────────────────"
            )
            _destroy_backend()
            return {"status": "success", "content": [{"text": text}]}

        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Unknown action: '{action}'\n"
                            "Valid: create_world, add_robot, add_cloth, add_cable, add_particles, "
                            "replicate, step, observe, reset, get_state, run_policy, run_diffsim, "
                            "add_sensor, read_sensor, solve_ik, destroy, list_assets, benchmark"
                        )
                    }
                ],
            }

    except Exception as e:
        logger.error(f"newton_sim error: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"❌ Error: {str(e)}"}]}
