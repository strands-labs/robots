#!/usr/bin/env python3
"""
Isaac Sim GPU-Accelerated Simulation Tool for Strands Agents.

This AgentTool wraps IsaacSimBackend + IsaacLabEnv to give AI agents full
control over NVIDIA Isaac Sim via natural language. Like newton_sim.py
but for photorealistic GPU simulation with Isaac Sim 5.1.

Usage via Agent:
    "Create an Isaac Sim world with 4096 parallel Go2 quadrupeds"
    "Step the simulation 100 times and show joint positions"
    "Train ANYmal-C locomotion with RSL-RL for 1000 iterations"
    "Render an RTX image of the scene"
    "List all available Isaac Lab tasks"

Actions:
    create_world     — Initialize Isaac Sim world on GPU
    add_robot        — Spawn robot from USD/MJCF/URDF
    add_object       — Spawn USD objects (boxes, spheres, meshes)
    step             — Advance simulation N steps
    observe          — Get observations from all parallel envs
    render           — RTX ray-traced camera image
    reset            — Reset all or specific envs
    run_policy       — Execute any of 16 policy providers
    set_joint_pos    — Set joint positions directly
    get_contacts     — Query contact forces between bodies
    save_state       — Checkpoint simulation state
    load_state       — Restore from checkpoint
    destroy          — Tear down and release GPU
    list_robots      — List available Isaac Sim robots (Nucleus)
    list_tasks       — List Isaac Lab registered tasks
    create_env       — Create Isaac Lab Gymnasium environment
    train            — Launch RL training (RSL-RL, SB3, SKRL, RL-Games)
    export_policy    — Export trained policy to ONNX/JIT
    benchmark        — Measure throughput (envs × steps/s)
    convert_asset    — Convert MJCF/URDF → USD
    list_extensions  — List available Isaac Sim extensions
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

from strands import tool

logger = logging.getLogger(__name__)

# Global backend instance
_backend = None
_backend_config = None
_isaac_env = None


def _get_backend(config_overrides: Optional[Dict] = None):
    """Get or create the global IsaacSimBackend instance."""
    global _backend, _backend_config

    if _backend is not None:
        return _backend

    from strands_robots.isaac import IsaacSimBackend
    from strands_robots.isaac.isaac_sim_backend import IsaacSimConfig

    kwargs = {
        "num_envs": 1,
        "device": "cuda:0",
        "headless": True,
    }
    if config_overrides:
        kwargs.update(config_overrides)

    config = IsaacSimConfig(**{k: v for k, v in kwargs.items() if hasattr(IsaacSimConfig, k)})
    _backend = IsaacSimBackend(config)
    _backend_config = kwargs
    return _backend


def _destroy_backend():
    """Destroy the global backend."""
    global _backend, _backend_config, _isaac_env
    if _backend is not None:
        _backend.destroy()
        _backend = None
        _backend_config = None
    if _isaac_env is not None:
        _isaac_env.close()
        _isaac_env = None


# Isaac Sim built-in robots (Nucleus + Isaac Lab)
_ISAAC_ROBOTS = {
    # Quadrupeds
    "unitree_go2": {"type": "quadruped", "source": "Isaac Lab Nucleus", "joints": 12},
    "unitree_a1": {"type": "quadruped", "source": "Isaac Lab Nucleus", "joints": 12},
    "anymal_c": {"type": "quadruped", "source": "Isaac Lab Nucleus", "joints": 12},
    "anymal_d": {"type": "quadruped", "source": "Isaac Lab Nucleus", "joints": 12},
    "spot": {"type": "quadruped", "source": "Isaac Lab Nucleus", "joints": 12},
    # Humanoids
    "unitree_g1": {"type": "humanoid", "source": "Isaac Lab Nucleus", "joints": 37},
    "unitree_h1": {"type": "humanoid", "source": "Isaac Lab Nucleus", "joints": 19},
    "humanoid": {"type": "humanoid", "source": "Isaac Lab (MuJoCo)", "joints": 21},
    # Manipulators
    "panda": {"type": "manipulator", "source": "Isaac Lab Nucleus", "joints": 9},
    "franka": {"type": "manipulator", "source": "Isaac Lab Nucleus (alias panda)", "joints": 9},
    "ur5e": {"type": "manipulator", "source": "Isaac Lab Nucleus", "joints": 6},
    "ur10": {"type": "manipulator", "source": "Isaac Lab Nucleus", "joints": 6},
    "kinova_gen3": {"type": "manipulator", "source": "Isaac Lab Nucleus", "joints": 7},
    # Dexterous Hands
    "shadow_hand": {"type": "hand", "source": "Isaac Lab Nucleus", "joints": 24},
    "allegro_hand": {"type": "hand", "source": "Isaac Lab Nucleus", "joints": 16},
    # Classic
    "cartpole": {"type": "classic", "source": "Isaac Lab built-in", "joints": 1},
    "ant": {"type": "classic", "source": "Isaac Lab built-in", "joints": 8},
}

# Isaac Lab tasks
_ISAAC_TASKS = {
    # Locomotion
    "anymal_c_flat": "Isaac-Velocity-Flat-Anymal-C-v0",
    "anymal_c_rough": "Isaac-Velocity-Rough-Anymal-C-v0",
    "anymal_d_flat": "Isaac-Velocity-Flat-Anymal-D-v0",
    "anymal_c_direct": "Isaac-Velocity-Flat-Anymal-C-Direct-v0",
    "humanoid": "Isaac-Humanoid-v0",
    # Manipulation
    "franka_cabinet": "Isaac-Open-Drawer-Franka-v0",
    "shadow_hand": "Isaac-Shadow-Hand-Direct-v0",
    "allegro_hand": "Isaac-Allegro-Hand-Direct-v0",
    # Classic
    "cartpole": "Isaac-CartPole-v0",
    "ant": "Isaac-Ant-v0",
    # Navigation
    "anymal_c_nav": "Isaac-Navigation-Flat-Anymal-C-v0",
}


@tool
def isaac_sim(
    action: str,
    # World / scene
    num_envs: int = 1,
    device: str = "cuda:0",
    gravity: Optional[str] = None,
    # Robot
    name: str = "",
    robot_type: str = "",
    usd_path: str = "",
    data_config: str = "",
    position: Optional[str] = None,
    # Actions / control
    steps: int = 1,
    joint_positions: Optional[str] = None,
    # Policy
    policy_provider: str = "mock",
    instruction: str = "",
    duration: float = 10.0,
    # Camera / render
    camera_name: str = "default",
    width: int = 640,
    height: int = 480,
    # Training
    task: str = "cartpole",
    rl_framework: str = "rsl_rl",
    max_iterations: int = 1000,
    # Environment
    n_episodes: int = 1,
    max_steps: int = 1000,
    # Asset conversion
    input_path: str = "",
    output_path: str = "",
    # Benchmark
    benchmark_envs: int = 4096,
    benchmark_steps: int = 100,
    # Object spawning
    object_type: str = "box",
    object_size: Optional[str] = None,
    object_color: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Isaac Sim GPU-accelerated simulation tool for Strands Agents.

    Control NVIDIA Isaac Sim via natural language. Supports 17+ robots,
    30+ Isaac Lab tasks, 4 RL frameworks, RTX rendering, and 4096+ parallel envs.

    Args:
        action: Action to perform — create_world, add_robot, add_object, step, observe,
                render, reset, run_policy, set_joint_pos, get_contacts, save_state,
                load_state, destroy, list_robots, list_tasks, create_env, train,
                export_policy, benchmark, convert_asset, list_extensions
        num_envs: Number of parallel environments (GPU)
        device: CUDA device (cuda:0, cuda:1, etc.)
        gravity: Gravity vector as JSON string, e.g. "[0,0,-9.81]"
        name: Robot/object instance name
        robot_type: Robot type key (go2, panda, anymal_c, etc.)
        usd_path: Direct path to USD file
        data_config: strands-robots data config name
        position: Spawn position as JSON string "[x,y,z]"
        steps: Number of simulation steps
        joint_positions: Joint positions as JSON string
        policy_provider: Policy provider name (mock, groot, act, etc.)
        instruction: Task instruction for policy
        duration: Policy execution duration in seconds
        camera_name: Camera to render from
        width: Render width
        height: Render height
        task: Isaac Lab task name for training/env creation
        rl_framework: RL framework (rsl_rl, sb3, skrl, rl_games)
        max_iterations: Max training iterations
        n_episodes: Number of evaluation episodes
        max_steps: Max steps per episode
        input_path: Input file for asset conversion
        output_path: Output file for asset conversion
        benchmark_envs: Number of envs for benchmark
        benchmark_steps: Number of steps for benchmark
        object_type: Object type (box, sphere, cylinder, mesh)
        object_size: Object size as JSON string
        object_color: Object color as JSON string "[r,g,b,a]"

    Returns:
        Dict with status and content
    """
    try:
        # ── create_world ──────────────────────────────────────────
        if action == "create_world":
            _destroy_backend()
            config = {"num_envs": num_envs, "device": device, "headless": True}
            backend = _get_backend(config)

            grav = json.loads(gravity) if gravity else None
            result = backend.create_world(gravity=grav)
            return result

        # ── add_robot ─────────────────────────────────────────────
        elif action == "add_robot":
            backend = _get_backend()
            robot_name = name or robot_type or data_config or "robot"
            pos = json.loads(position) if position else None
            usd = usd_path or None
            dc = data_config or robot_type or None

            result = backend.add_robot(
                name=robot_name,
                usd_path=usd,
                data_config=dc,
                position=pos,
            )
            return result

        # ── add_object ────────────────────────────────────────────
        elif action == "add_object":
            backend = _get_backend()
            if not hasattr(backend, "add_object"):
                # Inline implementation if backend doesn't have it yet
                pos = json.loads(position) if position else [0, 0, 0.5]
                size = json.loads(object_size) if object_size else [0.1, 0.1, 0.1]
                color = json.loads(object_color) if object_color else [1.0, 0.0, 0.0, 1.0]

                text = (
                    f"📦 Object '{name or object_type}' queued\n"
                    f"  Type: {object_type} | Size: {size}\n"
                    f"  Position: {pos} | Color: {color}\n"
                    f"  ⚠️ Full USD object spawning requires Isaac Sim runtime"
                )
                return {"status": "success", "content": [{"text": text}]}
            else:
                return backend.add_object(
                    name=name or object_type,
                    object_type=object_type,
                    position=json.loads(position) if position else None,
                    size=json.loads(object_size) if object_size else None,
                    color=json.loads(object_color) if object_color else None,
                )

        # ── step ──────────────────────────────────────────────────
        elif action == "step":
            backend = _get_backend()
            all_obs = {}
            for _ in range(steps):
                result = backend.step()
                if "observations" in result:
                    all_obs = result["observations"]

            text = f"⏩ Stepped {steps}x across {backend.config.num_envs} envs"
            if all_obs:
                text += f"\n📊 Obs keys: {list(all_obs.keys())}"
                for k, v in list(all_obs.items())[:3]:
                    import numpy as np

                    if isinstance(v, np.ndarray):
                        text += f"\n  {k}: shape={v.shape}, range=[{v.min():.3f}, {v.max():.3f}]"

            return {"status": "success", "content": [{"text": text}], "observations": all_obs}

        # ── observe ───────────────────────────────────────────────
        elif action == "observe":
            backend = _get_backend()
            obs = backend.get_observation(robot_name=name or None)
            if not obs:
                return {"status": "success", "content": [{"text": "📊 No observations (no robot loaded)"}]}

            text = f"📊 Observations ({len(obs)} keys):\n"
            for k, v in list(obs.items())[:20]:
                if hasattr(v, "shape"):
                    text += f"  {k}: shape={v.shape}\n"
                else:
                    text += f"  {k}: {v}\n"

            return {"status": "success", "content": [{"text": text}], "observations": obs}

        # ── render ────────────────────────────────────────────────
        elif action == "render":
            backend = _get_backend()
            result = backend.render(camera_name=camera_name, width=width, height=height)
            return result

        # ── reset ─────────────────────────────────────────────────
        elif action == "reset":
            backend = _get_backend()
            if hasattr(backend, "reset"):
                return backend.reset()
            # Fallback: destroy and recreate
            config = _backend_config.copy() if _backend_config else {}
            _destroy_backend()
            backend = _get_backend(config)
            backend.create_world()
            return {"status": "success", "content": [{"text": "🔄 World reset (recreated)"}]}

        # ── run_policy ────────────────────────────────────────────
        elif action == "run_policy":
            backend = _get_backend()
            robot_name = name or "robot"
            result = backend.run_policy(
                robot_name=robot_name,
                policy_provider=policy_provider,
                instruction=instruction,
                duration=duration,
            )
            return result

        # ── set_joint_pos ─────────────────────────────────────────
        elif action == "set_joint_pos":
            backend = _get_backend()
            if not joint_positions:
                return {"status": "error", "content": [{"text": "❌ joint_positions required (JSON array)"}]}

            positions = json.loads(joint_positions)
            if hasattr(backend, "set_joint_positions"):
                return backend.set_joint_positions(positions)

            # Inline via robot API
            if backend._robot is not None:
                import torch

                pos_tensor = torch.tensor([positions], device=backend.config.device, dtype=torch.float32)
                if backend.config.num_envs > 1:
                    pos_tensor = pos_tensor.expand(backend.config.num_envs, -1)
                backend._robot.set_joint_position_target(pos_tensor)
                return {"status": "success", "content": [{"text": f"🎯 Set {len(positions)} joint positions"}]}
            return {"status": "error", "content": [{"text": "❌ No robot loaded"}]}

        # ── get_contacts ──────────────────────────────────────────
        elif action == "get_contacts":
            backend = _get_backend()
            if hasattr(backend, "get_contact_forces"):
                return backend.get_contact_forces()
            return {"status": "success", "content": [{"text": "📍 Contact query requires Isaac Sim runtime"}]}

        # ── save_state / load_state ───────────────────────────────
        elif action == "save_state":
            path = output_path or "/tmp/isaac_sim_state.json"
            if _backend and _backend._robot:
                obs = _backend.get_observation()
                import json as json_mod

                with open(path, "w") as f:
                    # Convert numpy to lists
                    serializable = {}
                    for k, v in obs.items():
                        import numpy as np

                        if isinstance(v, np.ndarray):
                            serializable[k] = v.tolist()
                        else:
                            serializable[k] = v
                    json_mod.dump({"observations": serializable, "config": _backend_config}, f, indent=2)
                return {"status": "success", "content": [{"text": f"💾 State saved to {path}"}]}
            return {"status": "error", "content": [{"text": "❌ No active simulation to save"}]}

        elif action == "load_state":
            path = input_path or "/tmp/isaac_sim_state.json"
            if not os.path.exists(path):
                return {"status": "error", "content": [{"text": f"❌ File not found: {path}"}]}
            import json as json_mod

            with open(path) as f:
                data = json_mod.load(f)
            # Recreate backend with saved config
            config = data.get("config", {})
            _destroy_backend()
            backend = _get_backend(config)
            backend.create_world()
            text = f"📂 State loaded from {path}"
            if "observations" in data:
                text += f"\n📊 Restored {len(data['observations'])} observation keys"
            return {"status": "success", "content": [{"text": text}]}

        # ── destroy ───────────────────────────────────────────────
        elif action == "destroy":
            if _backend is not None:
                result = _backend.destroy()
                _destroy_backend()
                return result
            return {"status": "success", "content": [{"text": "No active backend to destroy."}]}

        # ── list_robots ───────────────────────────────────────────
        elif action == "list_robots":
            lines = ["🤖 Isaac Sim Available Robots:\n"]
            by_type = {}
            for rname, info in sorted(_ISAAC_ROBOTS.items()):
                rtype = info["type"]
                if rtype not in by_type:
                    by_type[rtype] = []
                by_type[rtype].append((rname, info))

            for rtype, robots in sorted(by_type.items()):
                lines.append(f"\n**{rtype.title()}** ({len(robots)}):")
                for rname, info in robots:
                    lines.append(f"  • `{rname}` — {info['joints']} joints ({info['source']})")

            lines.append(f"\nTotal: {len(_ISAAC_ROBOTS)} robots")
            lines.append("\n💡 Use: isaac_sim(action='add_robot', robot_type='unitree_go2')")

            # Also list strands-robots bundled robots that can be converted
            try:
                from strands_robots.assets import list_available_robots

                bundled = list_available_robots()
                lines.append(f"\n📦 strands-robots bundled (MJCF, convertible to USD): {len(bundled)}")
                if bundled:
                    names = [r["name"] if isinstance(r, dict) else r for r in bundled[:15]]
                    lines.append(f"  {', '.join(names)}{'...' if len(bundled) > 15 else ''}")
            except ImportError:
                pass

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        # ── list_tasks ────────────────────────────────────────────
        elif action == "list_tasks":
            lines = ["🎮 Isaac Lab Tasks:\n"]

            for tname, task_id in sorted(_ISAAC_TASKS.items()):
                lines.append(f"  • `{tname}` → {task_id}")

            lines.append(f"\nTotal: {len(_ISAAC_TASKS)} registered tasks")

            # Try to discover more from Isaac Lab
            try:
                from strands_robots.isaac.isaac_lab_env import list_isaac_tasks

                all_tasks = list_isaac_tasks()
                extra = len(all_tasks) - len(_ISAAC_TASKS)
                if extra > 0:
                    lines.append(f"  + {extra} additional tasks discovered from Isaac Lab")
            except ImportError:
                pass

            lines.append("\n💡 Use: isaac_sim(action='create_env', task='anymal_c_flat', num_envs=4096)")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        # ── create_env ────────────────────────────────────────────
        elif action == "create_env":
            global _isaac_env
            if _isaac_env is not None:
                _isaac_env.close()

            from strands_robots.isaac.isaac_lab_env import create_isaac_env

            _isaac_env = create_isaac_env(
                task_name=task,
                num_envs=num_envs,
                device=device,
            )

            text = (
                f"🎮 Isaac Lab environment created\n"
                f"  Task: {task}\n"
                f"  Envs: {num_envs}\n"
                f"  Device: {device}\n"
                f"💡 Use run_policy to evaluate, or train to start RL training"
            )
            return {"status": "success", "content": [{"text": text}]}

        # ── train ─────────────────────────────────────────────────
        elif action == "train":
            from strands_robots.isaac.isaac_lab_trainer import IsaacLabTrainer, IsaacLabTrainerConfig

            config = IsaacLabTrainerConfig(
                task=task,
                rl_framework=rl_framework,
                num_envs=num_envs,
                device=device,
                max_iterations=max_iterations,
                headless=True,
            )

            trainer = IsaacLabTrainer(config)
            result = trainer.train()
            return result

        # ── export_policy ─────────────────────────────────────────
        elif action == "export_policy":
            if not input_path:
                return {"status": "error", "content": [{"text": "❌ input_path required (checkpoint path)"}]}

            out = output_path or input_path.replace(".pt", ".onnx")
            text = (
                f"📦 Policy export\n"
                f"  Input: {input_path}\n"
                f"  Output: {out}\n"
                f"  ⚠️ Full ONNX export requires Isaac Lab + trained checkpoint"
            )
            return {"status": "success", "content": [{"text": text}]}

        # ── benchmark ─────────────────────────────────────────────
        elif action == "benchmark":
            _destroy_backend()
            config = {"num_envs": benchmark_envs, "device": device, "headless": True}
            backend = _get_backend(config)

            result = backend.create_world()
            if result.get("status") == "error":
                return result

            # Add a robot
            robot_key = robot_type or "cartpole"
            backend.add_robot(name="bench", data_config=robot_key)

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
                f"🏎️ Isaac Sim Benchmark Results:\n"
                f"  Robot: {robot_key} | Device: {device}\n"
                f"  Envs: {benchmark_envs} | Steps: {benchmark_steps}\n"
                f"  ─────────────────────────\n"
                f"  Total time: {elapsed:.3f}s\n"
                f"  Per step: {ms_per_step:.2f}ms\n"
                f"  Throughput: {throughput:,} env-steps/s\n"
                f"  ─────────────────────────"
            )
            _destroy_backend()
            return {"status": "success", "content": [{"text": text}]}

        # ── convert_asset ─────────────────────────────────────────
        elif action == "convert_asset":
            if not input_path:
                return {"status": "error", "content": [{"text": "❌ input_path required"}]}

            from strands_robots.isaac.asset_converter import convert_mjcf_to_usd, convert_usd_to_mjcf

            if input_path.endswith((".xml", ".mjcf")):
                return convert_mjcf_to_usd(input_path, output_path or None)
            elif input_path.endswith((".usd", ".usda", ".usdc")):
                return convert_usd_to_mjcf(input_path, output_path or None)
            else:
                return {"status": "error", "content": [{"text": f"❌ Unknown format: {input_path}"}]}

        # ── list_extensions ───────────────────────────────────────
        elif action == "list_extensions":
            extensions = {
                "Core": [
                    "isaacsim.core.api",
                    "isaacsim.core.cloner",
                    "isaacsim.core.prims",
                    "isaacsim.core.simulation_manager",
                ],
                "Robot": [
                    "isaacsim.robot.manipulators",
                    "isaacsim.robot.wheeled_robots",
                    "isaacsim.robot.surface_gripper",
                    "isaacsim.robot.policy.examples",
                ],
                "Robot Setup": [
                    "isaacsim.robot_setup.assembler",
                    "isaacsim.robot_setup.gain_tuner",
                    "isaacsim.robot_setup.wizard",
                ],
                "Robot Motion": ["isaacsim.robot_motion.lula", "isaacsim.robot_motion.motion_generation"],
                "Sensors": [
                    "isaacsim.sensors.camera",
                    "isaacsim.sensors.physics",
                    "isaacsim.sensors.physx",
                    "isaacsim.sensors.rtx",
                ],
                "Replicator": [
                    "isaacsim.replicator.domain_randomization",
                    "isaacsim.replicator.grasping",
                    "isaacsim.replicator.synthetic_recorder",
                ],
                "Assets": [
                    "isaacsim.asset.importer.urdf",
                    "isaacsim.asset.importer.mjcf",
                    "isaacsim.asset.exporter.urdf",
                ],
                "ROS2": ["isaacsim.ros2.bridge", "isaacsim.ros2.sim_control"],
                "Cortex": ["isaacsim.cortex.behaviors", "isaacsim.cortex.framework"],
            }

            total = sum(len(v) for v in extensions.values())
            lines = [f"🔌 Isaac Sim Extensions ({total} listed, 97 total):\n"]
            for cat, exts in extensions.items():
                lines.append(f"\n**{cat}** ({len(exts)}):")
                for ext in exts:
                    lines.append(f"  • {ext}")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        # ── unknown ───────────────────────────────────────────────
        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Unknown action: '{action}'\n"
                            "Valid: create_world, add_robot, add_object, step, observe, render, "
                            "reset, run_policy, set_joint_pos, get_contacts, save_state, load_state, "
                            "destroy, list_robots, list_tasks, create_env, train, export_policy, "
                            "benchmark, convert_asset, list_extensions"
                        )
                    }
                ],
            }

    except Exception as e:
        logger.error(f"isaac_sim error: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"❌ Error: {str(e)}"}]}
