"""Unified Robot factory — convenience layer over ``strands_robots.simulation``
and ``strands_robots.hardware_robot``.

Provides:
    - ``Robot("so100")`` → returns a simulation by default (safe)
    - ``Robot("so100", mode="real")`` → explicit real hardware
    - ``Robot("so100", mode="auto")`` → auto-detects sim/real
    - ``list_robots()``  → what's available

Environment Variables:
    STRANDS_ROBOT_MODE: Override mode detection ("sim", "real", "auto").
        Case-insensitive; surrounding whitespace ignored.

Examples::

    # Default: simulation (safe — no physical hardware interaction)
    sim = Robot("so100")

    # Explicit real hardware
    hw = Robot("so100", mode="real", cameras={...})

    # Auto-detect (probes USB for servo controllers)
    robot = Robot("so100", mode="auto")

    # With custom URDF/MJCF path
    sim = Robot("my_arm", urdf_path="/path/to/robot.xml")

Future (not yet implemented)::

    sim = Robot("unitree_go2", backend="isaac", num_envs=4096)
    sim = Robot("so100", backend="newton", num_envs=4096)
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING, Any, Literal, overload

from strands_robots.registry import (
    get_hardware_type,
    get_robot,
    has_hardware,
    has_sim,
    resolve_name,
)

if TYPE_CHECKING:
    from strands_robots.hardware_robot import Robot as HardwareRobot
    from strands_robots.simulation import Simulation

logger = logging.getLogger(__name__)

_VALID_MODES = ("sim", "real", "auto")


def _normalize_mode(mode: Any) -> str:
    """Lowercase + strip a mode value if it's a string. Pass non-str through unchanged
    so the caller can produce a clean ValueError later."""
    if isinstance(mode, str):
        return mode.lower().strip()
    return mode


def _auto_detect_mode(canonical: str) -> str:
    """Auto-detect sim vs real mode.

    Priority:
        1. ``STRANDS_ROBOT_MODE`` env var (explicit override)
        2. Robot-specific USB detection (Feetech/Dynamixel servo controllers)
        3. Default to sim (safest — never accidentally send commands to hardware)
    """
    env_mode = os.getenv("STRANDS_ROBOT_MODE", "").lower().strip()
    if env_mode in ("sim", "real"):
        return env_mode
    if env_mode == "auto":
        # Explicit no-op: user asked for detection, which is what we already do.
        pass
    elif env_mode:
        logger.warning("STRANDS_ROBOT_MODE=%r ignored (expected 'sim', 'real', or 'auto')", env_mode)

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
                if any(
                    kw in ((p.description or "") + (getattr(p, "manufacturer", None) or "")).lower()
                    for kw in servo_keywords
                )
                and not any(s in (p.description or "").lower() for s in exclude)
            ]
            if robot_ports:
                logger.info(
                    "Auto-detected robot hardware: %s",
                    [p.device for p in robot_ports],
                )
                return "real"
        except Exception as e:
            # USB enumeration is best-effort. pyserial usually raises OSError
            # (incl. PermissionError, SerialException) but libusb backends have
            # been observed to raise RuntimeError on hub glitches. Falling back
            # to sim is always safe; we log at debug for diagnosis.
            logger.debug("USB probe failed (%s: %s); falling back to sim", type(e).__name__, e)

    return "sim"


def _validate_known_robot(canonical: str, original: str, urdf_path: str | None) -> None:
    """Reject empty/unknown robot names with a single clean error before we
    descend into the sim or hardware backends. ``urdf_path`` short-circuits the
    check because users supplying an explicit MJCF/URDF don't need a registry
    entry."""
    if urdf_path:
        return
    if not canonical:
        raise ValueError(
            f"Invalid robot name {original!r}. Pass a registered name (see ``list_robots()``) or supply ``urdf_path=``."
        )
    if get_robot(canonical) is None and not (has_sim(canonical) or has_hardware(canonical)):
        raise ValueError(
            f"Unknown robot {original!r} (resolved to {canonical!r}). "
            "Pass a registered name (see ``list_robots()``) or supply ``urdf_path=``."
        )


@overload
def Robot(
    name: str,
    mode: Literal["sim"] = ...,
    backend: str = ...,
    urdf_path: str | None = ...,
    cameras: dict[str, dict[str, Any]] | None = ...,
    position: list[float] | None = ...,
    data_config: str | None = ...,
    **kwargs: Any,
) -> Simulation: ...


@overload
def Robot(
    name: str,
    mode: Literal["real"],
    backend: str = ...,
    urdf_path: str | None = ...,
    cameras: dict[str, dict[str, Any]] | None = ...,
    position: list[float] | None = ...,
    data_config: str | None = ...,
    **kwargs: Any,
) -> HardwareRobot: ...


@overload
def Robot(
    name: str,
    mode: Literal["auto"] | str = ...,
    backend: str = ...,
    urdf_path: str | None = ...,
    cameras: dict[str, dict[str, Any]] | None = ...,
    position: list[float] | None = ...,
    data_config: str | None = ...,
    **kwargs: Any,
) -> Simulation | HardwareRobot: ...


def Robot(  # noqa: N802 — uppercase by design (factory mimicking a class constructor)
    name: str,
    mode: str = "sim",
    backend: str = "mujoco",
    urdf_path: str | None = None,
    cameras: dict[str, dict[str, Any]] | None = None,
    position: list[float] | None = None,
    data_config: str | None = None,
    mesh: bool = True,
    peer_id: str | None = None,
    **kwargs: Any,
) -> Simulation | HardwareRobot:
    """Create a robot — returns a Simulation or HardwareRobot instance.

    This is a convenience factory, NOT a wrapper class.  You get the real
    backend instance back — with full access to all its methods.

    Defaults to simulation mode so that ``Robot("so100")`` never
    accidentally sends commands to physical hardware.  Use
    ``mode="real"`` to explicitly opt into hardware control.

    Args:
        name: Robot name ("so100", "aloha", "unitree_g1", "panda", ...)
              Accepts any alias defined in ``registry/robots.json``.
        mode: "sim" (default — safe), "real" (explicit hardware), or
              "auto" (probes USB for servo controllers, falls back to sim).
              Case-insensitive; surrounding whitespace ignored.
        backend: Simulation backend — currently only "mujoco" (CPU).
                 Future: "isaac" (GPU), "newton" (GPU).
                 Only applies to ``mode="sim"``; ignored for ``mode="real"``.
        urdf_path: Explicit path to URDF/MJCF file. If not provided,
                   resolved via ``strands_robots.simulation.model_registry``
                   (asset manager or ``STRANDS_ASSETS_DIR`` search paths).
        cameras: Camera config for real hardware. Example::

            {"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}}

            Note: In ``mode="sim"``, cameras must be added after creation
            via the simulation tool (``add_camera`` action). They cannot
            be passed to the factory yet.

        position: Robot position in sim world [x, y, z].
        data_config: Data configuration name for observation/action schema.
                     Defaults to the canonical robot name. For multi-camera
                     setups, specify explicitly: ``data_config="so100_dualcam"``.
        **kwargs: Forwarded to the underlying backend constructor.

    Returns:
        ``strands_robots.simulation.Simulation`` (sim) or
        ``strands_robots.hardware_robot.Robot`` (real hardware).

    Raises:
        ValueError: If ``mode`` is not 'sim'/'real'/'auto', if ``cameras=``
                    is passed in sim mode, or if the robot name is empty
                    or not in the registry (and no ``urdf_path=`` given).
        NotImplementedError: If an unimplemented sim backend is requested.
        RuntimeError: If the sim world or robot fails to initialize.

    Examples::

        # Simulation (default — safe)
        sim = Robot("so100")

        # Explicit MJCF model path
        sim = Robot("my_arm", urdf_path="path/to/robot.xml")

        # Real hardware (explicit opt-in)
        hw = Robot("so100", mode="real", cameras={...})

        # Auto-detect (probes USB, falls back to sim)
        robot = Robot("so100", mode="auto")

        # The 5-line promise (defaults to sim — safe, no hardware needed)
        from strands_robots import Robot
        from strands import Agent
        robot = Robot("so100")  # mode="sim" (default)
        agent = Agent(tools=[robot])
        agent("Pick up the red cube")
    """
    canonical = resolve_name(name)
    _validate_known_robot(canonical, name, urdf_path)

    mode = _normalize_mode(mode)

    if mode == "auto":
        mode = _auto_detect_mode(canonical)

    # --- Simulation ---
    if mode == "sim":
        if backend != "mujoco":
            raise NotImplementedError(
                f"Backend {backend!r} is not yet implemented. "
                f"Currently supported: 'mujoco'. "
                f"Isaac and Newton backends are on the roadmap."
            )

        if cameras is not None:
            raise ValueError(
                "cameras= is only supported in mode='real'. "
                "For sim cameras, add them via the simulation tool's "
                "'add_camera' action after creation."
            )

        from strands_robots.simulation import Simulation

        sim = Simulation(
            tool_name=f"{canonical}_sim",
            **kwargs,
        )

        try:
            result = sim._dispatch_action("create_world", {})
            if result.get("status") == "error":
                content = result.get("content", [])
                msg = content[0].get("text", str(result)) if content else str(result)
                raise RuntimeError(f"Failed to create sim world for {canonical!r}: {msg}")

            add_robot_params: dict[str, Any] = {
                "robot_name": canonical,
                "data_config": data_config or canonical,
                "position": position or [0.0, 0.0, 0.0],
            }
            if urdf_path:
                add_robot_params["urdf_path"] = urdf_path

            result = sim._dispatch_action("add_robot", add_robot_params)
            if result.get("status") == "error":
                content = result.get("content", [])
                msg = content[0].get("text", str(result)) if content else str(result)
                raise RuntimeError(f"Failed to create sim robot {canonical!r}: {msg}")
        except BaseException:
            # Cleanup ANY partial-init failure: explicit RuntimeError above OR
            # an unexpected exception from _dispatch_action itself (OOM, OS
            # error during temp-file write, MuJoCo error surfaced as exception).
            # KeyboardInterrupt during creation also lands here so the executor
            # + temp dir + MuJoCo world get released.
            # suppress() ensures destroy() errors don't mask the original exception.
            with contextlib.suppress(Exception):
                sim.destroy()
            raise

        # Attach a Zenoh mesh so the Simulation auto-discovers other peers.
        # Failure to start the mesh must NOT bring down the sim — the user
        # explicitly asked for a Simulation, mesh is an enrichment.
        try:
            from strands_robots.mesh import init_mesh

            sim_mesh = init_mesh(
                sim,
                peer_id=peer_id,
                peer_type="sim",
                mesh=mesh,
            )
            if sim_mesh is not None:
                sim.mesh = sim_mesh
                sim.peer_id = sim_mesh.peer_id
        except Exception as exc:  # noqa: BLE001 — mesh enrichment is best-effort
            logger.warning("Failed to initialise mesh for %r: %s", canonical, exc)

        return sim

    # --- Real hardware (explicit opt-in) ---
    elif mode == "real":
        if backend != "mujoco":
            logger.debug(
                "backend=%r ignored in mode='real' (hardware uses direct servo control)",
                backend,
            )

        from strands_robots.hardware_robot import Robot as HardwareRobotCls

        real_type = get_hardware_type(canonical) or canonical
        hw = HardwareRobotCls(
            tool_name=canonical,
            robot=real_type,
            cameras=cameras,
            **kwargs,
        )

        # Attach a Zenoh mesh so the hardware Robot auto-discovers peers.
        # Best-effort: a mesh failure must not kill a working hardware robot.
        try:
            from strands_robots.mesh import init_mesh

            hw_mesh = init_mesh(
                hw,
                peer_id=peer_id,
                peer_type="robot",
                mesh=mesh,
            )
            if hw_mesh is not None:
                hw.mesh = hw_mesh
                hw.peer_id = hw_mesh.peer_id
        except Exception as exc:  # noqa: BLE001 — mesh enrichment is best-effort
            logger.warning("Failed to initialise mesh for %r: %s", canonical, exc)

        return hw

    else:
        raise ValueError(f"Invalid mode {mode!r}. Choose 'sim', 'real', or 'auto' (case-insensitive).")


__all__ = ["Robot"]
