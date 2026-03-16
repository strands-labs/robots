"""Unified Robot Factory — convenience layer over Simulation and HardwareRobot.

Provides:
    - ``Robot("so100")`` → auto-detects sim/real, returns the right backend
    - ``list_robots()``  → what's available
"""

import logging
import os
from typing import Any, Dict, List, Optional

from strands_robots.registry import get_hardware_type, has_hardware, resolve_name
from strands_robots.registry import list_robots as _registry_list_robots

logger = logging.getLogger(__name__)


def _auto_detect_mode(canonical: str) -> str:
    """Auto-detect sim vs real mode.

    Priority:
        1. ``STRANDS_ROBOT_MODE`` env var (explicit override)
        2. Robot-specific USB detection (Feetech/Dynamixel servo controllers)
        3. Default to sim (safest — never accidentally send commands to hardware)
    """
    env_mode = os.getenv("STRANDS_ROBOT_MODE", "").lower()
    if env_mode in ("sim", "real"):
        return env_mode

    # Only probe USB if the robot actually has hardware support
    if has_hardware(canonical):
        try:
            import serial.tools.list_ports

            ports = list(serial.tools.list_ports.comports())
            servo_keywords = ["feetech", "dynamixel", "sts3215", "xl430", "xl330"]
            exclude = ["bluetooth", "internal", "debug", "apple", "modem"]
            robot_ports = [
                p
                for p in ports
                if any(kw in (p.description + getattr(p, "manufacturer", "")).lower() for kw in servo_keywords)
                and not any(s in p.description.lower() for s in exclude)
            ]
            if robot_ports:
                logger.info("Auto-detected robot hardware: %s", [p.device for p in robot_ports])
                return "real"
        except (ImportError, Exception):
            pass

    return "sim"


def Robot(
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
    """Create a robot — returns Simulation, IsaacSimBackend, NewtonBackend, or HardwareRobot.

    This is a convenience factory, NOT a wrapper class.  You get the real
    backend instance back — with full access to all its methods.

    Args:
        name: Robot name ("so100", "aloha", "unitree_g1", "panda", ...)
              Accepts any alias defined in ``registry/robots.json``.
        mode: "auto" (detect hardware), "sim", or "real".
        backend: Simulation backend — "mujoco" (CPU), "isaac" (GPU), "newton" (GPU).
        cameras: Camera config for real hardware.
        position: Robot position in sim world [x, y, z].
        num_envs: Parallel environments (isaac/newton only).
        mesh: Enable Zenoh mesh networking.
        peer_id: Custom peer ID for Zenoh mesh.
        **kwargs: Forwarded to the underlying backend.

    Returns:
        Simulation (MuJoCo), IsaacSimBackend, NewtonBackend, or HardwareRobot.

    Examples::

        sim = Robot("so100")                              # MuJoCo (auto)
        sim = Robot("unitree_go2", backend="isaac", num_envs=4096)  # Isaac
        sim = Robot("so100", backend="newton", num_envs=4096)       # Newton
        hw  = Robot("so100", mode="real", cameras={...})  # Real hardware
    """
    canonical = resolve_name(name)

    if mode == "auto":
        mode = _auto_detect_mode(canonical)

    # ── Simulation backends ──
    instance = None
    if mode == "sim":
        if backend == "isaac":
            from strands_robots.isaac.isaac_sim_backend import (
                IsaacSimBackend,
                IsaacSimConfig,
            )

            config = IsaacSimConfig(
                num_envs=num_envs,
                device=kwargs.pop("device", "cuda:0"),
            )
            instance = IsaacSimBackend(config=config)
            instance.create_world()
            result = instance.add_robot(
                name=canonical,
                data_config=canonical,
                position=position or [0.0, 0.0, 0.0],
            )
            if result.get("status") == "error":
                raise RuntimeError(f"Failed to create Isaac robot '{canonical}': {result}")

        elif backend == "newton":
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
            instance = NewtonBackend(config=config)
            instance.create_world()
            result = instance.add_robot(
                name=canonical,
                data_config=canonical,
                position=tuple(position) if position else (0.0, 0.0, 0.0),
            )
            if result.get("status") == "error":
                raise RuntimeError(f"Failed to create Newton robot '{canonical}': {result.get('message', result)}")
            if num_envs > 1:
                instance.replicate(num_envs=num_envs)

        else:
            # MuJoCo CPU backend (default)
            from strands_robots.simulation import Simulation

            sim_name = canonical
            instance = Simulation(
                tool_name=f"{canonical}_sim", mesh=mesh, peer_id=peer_id, **kwargs
            )
            instance._dispatch_action("create_world", {})
            result = instance._dispatch_action("add_robot", {
                "robot_name": canonical,
                "data_config": sim_name,
                "position": position or [0.0, 0.0, 0.0],
            })
            if result.get("status") == "error":
                raise RuntimeError(f"Failed to create sim robot '{canonical}': {result}")

    # ── Real hardware ──
    else:
        from strands_robots.robot import Robot as HardwareRobot

        real_type = get_hardware_type(canonical) or canonical
        instance = HardwareRobot(
            tool_name=canonical,
            robot=real_type,
            cameras=cameras,
            mesh=mesh,
            peer_id=peer_id,
            **kwargs,
        )

    # Store metadata for .run()
    instance._peer_id = peer_id or f"{canonical}-{os.urandom(3).hex()}"
    instance._peer_type = "sim" if mode == "sim" else "robot"
    instance._device_connect_runtime = None

    # Attach .run() method for foreground server mode
    instance.run = lambda: _run_foreground(instance)

    return instance


def _run_foreground(instance):
    """Start Device Connect and block — robot listens for commands.

    Call this to keep the process alive as a server. Ctrl+C to stop.

    Usage:
        r = Robot("so100")
        r.run()  # blocks here, listening for commands
    """
    import signal
    import threading

    peer_id = getattr(instance, "_peer_id", "robot")
    peer_type = getattr(instance, "_peer_type", "robot")

    # Init Device Connect
    try:
        from strands_robots.device_connect import init_device_connect_sync

        instance._device_connect_runtime = init_device_connect_sync(
            instance, peer_id=peer_id, peer_type=peer_type,
        )
    except Exception as e:
        logger.warning("Device Connect init failed: %s", e)
        if hasattr(instance, "_init_mesh_fallback"):
            instance._init_mesh_fallback()

    stop = threading.Event()

    def _shutdown(sig, frame):
        print(f"\n🛑 Shutting down {peer_id}...")
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"🤖 {peer_id} is online. Ctrl+C to stop.")
    stop.wait()

    # Cleanup
    rt = getattr(instance, "_device_connect_runtime", None)
    if rt and hasattr(rt, "_loop"):
        rt._loop.call_soon_threadsafe(rt._loop.stop)
    print(f"👋 {peer_id} stopped.")


def list_robots(mode: str = "all") -> List[Dict[str, Any]]:
    """List available robots.

    Args:
        mode: "all", "sim", "real", or "both" (has both sim and real).

    Returns:
        List of dicts with name, description, has_sim, has_real.
    """
    return _registry_list_robots(mode)


__all__ = ["Robot", "list_robots"]
