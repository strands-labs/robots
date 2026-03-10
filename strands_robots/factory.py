#!/usr/bin/env python3
"""
Unified Robot Factory — convenience layer over Simulation and HardwareRobot.

Provides:
- Robot("so100") → auto-detects sim/real, returns callable wrapper
- list_robots() → what's available
- create_robot() → alias for Robot()

The wrapper is intentionally thin. For full control, use Simulation or
HardwareRobot directly — they're the real implementations.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Robot registry: name → (sim asset, real hardware config)
# ─────────────────────────────────────────────────────────────────────

_UNIFIED_ROBOTS: Dict[str, Dict[str, Any]] = {
    "so100": {"sim": "so100", "real": "so100_follower", "description": "TrossenRobotics SO-ARM100 (6-DOF)"},
    "so101": {"sim": "so101", "real": "so101_follower", "description": "RobotStudio SO-101 (6-DOF)"},
    "koch": {"sim": "koch", "real": "koch_follower", "description": "Koch v1.1 Low Cost Robot Arm (6-DOF)"},
    "unitree_g1": {"sim": "unitree_g1", "real": "unitree_g1", "description": "Unitree G1 Humanoid (29-DOF)"},
    "panda": {"sim": "panda", "real": None, "description": "Franka Emika Panda (7-DOF)"},
    "aloha": {"sim": "aloha", "real": "bi_so_follower", "description": "ALOHA Bimanual (2× ViperX 300s)"},
    "unitree_h1": {"sim": "unitree_h1", "real": None, "description": "Unitree H1 Humanoid (19-DOF)"},
    "unitree_go2": {"sim": "unitree_go2", "real": None, "description": "Unitree Go2 Quadruped"},
    "spot": {"sim": "spot", "real": None, "description": "Boston Dynamics Spot"},
    "ur5e": {"sim": "ur5e", "real": None, "description": "Universal Robots UR5e (6-DOF)"},
    "fr3": {"sim": "fr3", "real": None, "description": "Franka FR3 (7-DOF)"},
    "xarm7": {"sim": "xarm7", "real": None, "description": "UFactory xArm 7"},
    "vx300s": {"sim": "vx300s", "real": None, "description": "Trossen ViperX 300s"},
    "shadow_hand": {"sim": "shadow_hand", "real": None, "description": "Shadow Dexterous Hand (24-DOF)"},
    "stretch3": {"sim": "stretch3", "real": None, "description": "Hello Robot Stretch 3"},
    "openarm": {"sim": "openarm", "real": None, "description": "Enactic OpenArm (7-DOF, DAMIAO motors, CAN bus)"},
    "lekiwi": {"sim": None, "real": "lekiwi", "description": "LeKiwi mobile robot"},
    "reachy2": {"sim": None, "real": "reachy2", "description": "Pollen Reachy 2"},
    "reachy_mini": {
        "sim": "reachy_mini",
        "real": None,
        "description": "Pollen Reachy Mini (6-DOF Stewart head + antennas)",
    },
    "hope_jr": {"sim": None, "real": "hope_jr", "description": "Hope Junior arm"},
    "kuka_iiwa": {"sim": "kuka_iiwa", "real": None, "description": "KUKA LBR iiwa 14"},
    "kinova_gen3": {"sim": "kinova_gen3", "real": None, "description": "Kinova Gen3 (7-DOF)"},
    "arx_l5": {"sim": "arx_l5", "real": None, "description": "ARX L5 (6-DOF)"},
    "piper": {"sim": "piper", "real": None, "description": "AgileX Piper"},
    "z1": {"sim": "z1", "real": None, "description": "Unitree Z1 + gripper"},
    "trossen_wxai": {"sim": "trossen_wxai", "real": None, "description": "Trossen WidowX AI Bimanual"},
    "leap_hand": {"sim": "leap_hand", "real": None, "description": "LEAP Hand (16-DOF)"},
    "robotiq_2f85": {"sim": "robotiq_2f85", "real": None, "description": "Robotiq 2F-85 Gripper"},
    "fourier_n1": {"sim": "fourier_n1", "real": None, "description": "Fourier N1 Humanoid (26-DOF)"},
    "apollo": {"sim": "apollo", "real": None, "description": "Apptronik Apollo Humanoid"},
    "cassie": {"sim": "cassie", "real": None, "description": "Agility Cassie Bipedal"},
    "unitree_a1": {"sim": "unitree_a1", "real": None, "description": "Unitree A1 Quadruped"},
    "google_robot": {"sim": "google_robot", "real": None, "description": "Google Robot (RT-X)"},
    "open_duck_mini": {
        "sim": "open_duck_mini",
        "real": None,
        "description": "Open Duck Mini V2 (16-DOF expressive biped)",
    },
    "asimov_v0": {"sim": "asimov_v0", "real": None, "description": "Asimov V0 Bipedal Legs (12-DOF)"},
}

_ALIASES: Dict[str, str] = {
    "so100_follower": "so100",
    "so100_dualcam": "so100",
    "so100_4cam": "so100",
    "so101_follower": "so101",
    "so101_dualcam": "so101",
    "so101_tricam": "so101",
    "koch_follower": "koch",
    "koch_v1.1": "koch",
    "g1": "unitree_g1",
    "h1": "unitree_h1",
    "go2": "unitree_go2",
    "a1": "unitree_a1",
    "franka": "panda",
    "franka_panda": "panda",
    "franka_fr3": "fr3",
    "bi_so_follower": "aloha",
    "reachy-mini": "reachy_mini",
    "reachymini": "reachy_mini",
    "gr1": "fourier_n1",
    "fourier_gr1": "fourier_n1",
    "open_duck": "open_duck_mini",
    "mini_bdx": "open_duck_mini",
    "bdx": "open_duck_mini",
    "open_duck_v2": "open_duck_mini",
    "asimov": "asimov_v0",
    "enactic_openarm": "openarm",
    "open_arm": "openarm",
    "openarm_v10": "openarm",
    # GR00T DATA_CONFIG_MAP variant aliases → base robots
    "unitree_g1_locomanip": "unitree_g1",
    "unitree_g1_full_body": "unitree_g1",
    "fourier_gr1_arms_only": "fourier_n1",
    "fourier_gr1_arms_waist": "fourier_n1",
    "fourier_gr1_full_upper_body": "fourier_n1",
    "single_panda_gripper": "panda",
    "bimanual_panda_gripper": "panda",
    "bimanual_panda_hand": "panda",
    "libero_panda": "panda",
    "oxe_droid": "panda",
    "oxe_google": "google_robot",
    "oxe_widowx": "vx300s",
    "agibot_dual_arm": "aloha",
    "agibot_dual_arm_gripper": "aloha",
    "agibot_dual_arm_dexhand": "aloha",
    "agibot_dual_arm_full": "aloha",
    "agibot_genie1": "aloha",
    "galaxea_r1_pro": "aloha",
    "agilex_piper": "piper",
}


def _resolve_name(name: str) -> str:
    name = name.lower().strip().replace("-", "_")
    return _ALIASES.get(name, name)


def _auto_detect_mode(canonical: str, robot_info: Optional[Dict]) -> str:
    """Auto-detect sim vs real mode.

    Priority:
    1. STRANDS_ROBOT_MODE env var (explicit override)
    2. Robot-specific USB detection (Feetech/Dynamixel servo controllers)
    3. Default to sim (safest — never accidentally send commands to hardware)
    """
    env_mode = os.getenv("STRANDS_ROBOT_MODE", "").lower()
    if env_mode in ("sim", "real"):
        return env_mode

    # Only check for real hardware if this robot HAS a real config
    if robot_info and robot_info.get("real"):
        try:
            import serial.tools.list_ports

            ports = list(serial.tools.list_ports.comports())
            # Only match ROBOT-SPECIFIC servo controllers, not generic USB serial
            robot_servo_keywords = ["feetech", "dynamixel", "sts3215", "xl430", "xl330"]
            robot_ports = [
                p
                for p in ports
                if any(
                    kw in (getattr(p, "description", "") + getattr(p, "manufacturer_string", "")).lower()
                    for kw in robot_servo_keywords
                )
                and not any(
                    s in getattr(p, "description", "").lower()
                    for s in ["bluetooth", "internal", "debug", "apple", "modem"]
                )
            ]
            if robot_ports:
                logger.info(f"Auto-detected robot hardware: {[p.device for p in robot_ports]}")
                return "real"
        except (ImportError, Exception):
            pass

    return "sim"


def create_robot(
    name: str,
    mode: str = "auto",
    backend: str = "mujoco",
    cameras: Optional[Dict[str, Dict[str, Any]]] = None,
    position: Optional[List[float]] = None,
    num_envs: int = 1,
    mesh: bool = True,
    peer_id: str = None,
    **kwargs,
):
    """Create a robot — returns Simulation, IsaacSimBackend, or HardwareRobot.

    This is a convenience factory, NOT a wrapper class. You get the real
    backend instance back — with full access to all their methods.

    Args:
        name: Robot name ("so100", "aloha", "unitree_g1", "panda", ...)
        mode: "auto" (detect hardware), "sim", or "real"
        backend: Simulation backend — "mujoco" (CPU, default) or "isaac" (GPU)
        cameras: Camera config for real hardware
        position: Robot position in sim world [x, y, z]
        num_envs: Parallel environments (Isaac backend only, default: 1)
        mesh: Enable Zenoh mesh networking (default: True)
        peer_id: Custom peer ID for Zenoh mesh (auto-generated if None)
        **kwargs: Forwarded to underlying backend

    Returns:
        Simulation (MuJoCo), IsaacSimBackend (Isaac Sim), or HardwareRobot

    Examples:
        # MuJoCo sim (auto-detected — no USB hardware)
        sim = Robot("so100")

        # Isaac Sim (GPU-accelerated, parallel envs)
        sim = Robot("unitree_go2", backend="isaac", num_envs=4096)

        # Newton (GPU-accelerated, differentiable, 4096+ parallel envs)
        sim = Robot("so100", backend="newton", num_envs=4096, solver="mujoco")

        # Real hardware
        hw = Robot("so100", mode="real", cameras={...})
    """
    canonical = _resolve_name(name)
    robot_info = _UNIFIED_ROBOTS.get(canonical)

    if mode == "auto":
        mode = _auto_detect_mode(canonical, robot_info)

    if mode == "sim":
        if backend == "isaac":
            # Isaac Sim GPU backend
            from strands_robots.isaac.isaac_sim_backend import IsaacSimBackend, IsaacSimConfig

            config = IsaacSimConfig(
                num_envs=num_envs,
                device=kwargs.pop("device", "cuda:0"),
            )
            isaac_backend = IsaacSimBackend(config=config)
            isaac_backend.create_world()
            result = isaac_backend.add_robot(
                name=canonical,
                data_config=canonical,
                position=position or [0.0, 0.0, 0.0],
            )
            if result.get("status") == "error":
                raise RuntimeError(f"Failed to create Isaac robot '{canonical}': {result}")
            return isaac_backend

        elif backend == "newton":
            # Newton GPU-accelerated physics backend
            from strands_robots.newton.newton_backend import NewtonBackend, NewtonConfig

            solver = kwargs.pop("solver", "mujoco")
            config = NewtonConfig(
                num_envs=num_envs,
                device=kwargs.pop("device", "cuda:0"),
                solver=solver,
                broad_phase=kwargs.pop("broad_phase", "sap"),
                enable_differentiable=kwargs.pop("enable_differentiable", False),
                enable_cuda_graph=kwargs.pop("enable_cuda_graph", False),
                substeps=kwargs.pop("substeps", 1),
                physics_dt=kwargs.pop("physics_dt", 1.0 / 200.0),
            )
            newton_backend = NewtonBackend(config=config)
            newton_backend.create_world()
            result = newton_backend.add_robot(
                name=canonical,
                data_config={"name": canonical},
                position=tuple(position) if position else (0.0, 0.0, 0.0),
            )
            if result.get("status") == "error":
                raise RuntimeError(f"Failed to create Newton robot '{canonical}': {result.get('message', result)}")
            # Auto-replicate for parallel envs
            if num_envs > 1:
                newton_backend.replicate(num_envs=num_envs)
            return newton_backend

        else:
            # MuJoCo CPU backend (default)
            from strands_robots.simulation import Simulation

            sim_name = canonical
            if robot_info and robot_info.get("sim"):
                sim_name = robot_info["sim"]

            sim = Simulation(tool_name=f"{canonical}_sim", mesh=mesh, peer_id=peer_id, **kwargs)
            sim.create_world()
            result = sim.add_robot(
                name=canonical,
                data_config=sim_name,
                position=position or [0.0, 0.0, 0.0],
            )
            if result.get("status") == "error":
                raise RuntimeError(f"Failed to create sim robot '{canonical}': {result}")
            return sim

    else:
        from strands_robots.robot import Robot as HardwareRobot

        real_type = canonical
        if robot_info and robot_info.get("real"):
            real_type = robot_info["real"]

        return HardwareRobot(
            tool_name=canonical,
            robot=real_type,
            cameras=cameras,
            mesh=mesh,
            peer_id=peer_id,
            **kwargs,
        )


# Backward compat
# Convenience alias — Robot() reads naturally in user code:
#   sim = Robot("so100")
# but create_robot() is the canonical name (it's a factory function, not a class).
Robot = create_robot


def list_robots(mode: str = "all") -> List[Dict[str, Any]]:
    """List available robots.

    Args:
        mode: "all", "sim", "real", or "both" (has both sim and real)
    """
    results = []
    for name, info in sorted(_UNIFIED_ROBOTS.items()):
        has_sim = info.get("sim") is not None
        has_real = info.get("real") is not None

        if mode == "sim" and not has_sim:
            continue
        if mode == "real" and not has_real:
            continue
        if mode == "both" and not (has_sim and has_real):
            continue

        results.append(
            {
                "name": name,
                "description": info.get("description", ""),
                "has_sim": has_sim,
                "has_real": has_real,
            }
        )

    return results


__all__ = ["Robot", "create_robot", "list_robots"]
