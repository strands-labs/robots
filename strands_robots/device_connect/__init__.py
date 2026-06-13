"""Device Connect integration for strands-robots.

Provides DeviceDriver adapters that wrap Robot and Simulation instances,
exposing them to Device Connect's device registry, RPC routing, and event system.

Usage:
    from strands_robots.device_connect import init_device_connect

    robot = Robot("so100")
    runtime = await init_device_connect(robot, peer_id="so100-lab-1")

    # Now discoverable via Device Connect tools:
    #   discover_devices(device_type="strands_robot")
    #   invoke_device("so100-lab-1", "execute", {"instruction": "pick up cube"})
"""

import asyncio
import logging
import os
import threading
import uuid

from device_connect_edge import DeviceRuntime

from strands_robots.device_connect.reachy_mini_driver import ReachyMiniDriver
from strands_robots.device_connect.robot_driver import RobotDeviceDriver
from strands_robots.device_connect.sim_driver import SimulationDeviceDriver

logger = logging.getLogger(__name__)

__all__ = [
    "init_device_connect",
    "init_device_connect_sync",
    "resolve_allow_insecure",
    "RobotDeviceDriver",
    "SimulationDeviceDriver",
    "ReachyMiniDriver",
]

_INSECURE_TRUE = ("true", "1", "yes")


def resolve_allow_insecure(
    explicit: bool | None = None,
    env_value: str | None = None,
) -> bool:
    """Resolve the effective ``allow_insecure`` setting (secure by default).

    Precedence: explicit arg > ``DEVICE_CONNECT_ALLOW_INSECURE`` env var >
    secure default (``False``). Insecure transport is never implicit — it
    must be opted into via the argument or the env var.

    Extracted as a pure function so the secure-by-default posture is unit
    testable without standing up a DeviceRuntime.
    """
    if explicit is not None:
        return explicit
    if env_value is not None:
        return env_value.lower() in _INSECURE_TRUE
    return False


async def init_device_connect(
    robot,
    peer_id: str | None = None,
    peer_type: str = "robot",
    messaging_url: str | None = None,
    messaging_backend: str | None = None,
    tenant: str = "default",
    allow_insecure: bool | None = None,
) -> DeviceRuntime:
    """Initialize Device Connect for a Robot or Simulation.

    Drop-in replacement for init_mesh(). Creates a DeviceDriver adapter
    and starts a DeviceRuntime in the background.

    When messaging_backend="zenoh" and messaging_url is None, the runtime
    enters D2D mode — devices discover each other directly via Zenoh
    multicast scouting on the LAN. No broker, no Docker, no env vars.

    Args:
        robot: A Robot or Simulation instance to wrap.
        peer_id: Device ID for registration (auto-generated if None).
        peer_type: "robot" or "sim" — selects the appropriate driver.
        messaging_url: Explicit messaging URL (overrides env vars).
        messaging_backend: Messaging backend — "zenoh" or "nats".
            None = auto-detect from MESSAGING_BACKEND env var (default "zenoh").
        tenant: Device Connect tenant namespace.
        allow_insecure: Allow insecure (unencrypted, unauthenticated)
            transport. None = auto-detect: respects the
            DEVICE_CONNECT_ALLOW_INSECURE env var if set, otherwise defaults
            to False (secure). Insecure transport must be explicitly opted
            into; a prominent warning is logged whenever it is active.

    Returns:
        The running DeviceRuntime instance.
    """
    if peer_type == "sim":
        driver = SimulationDeviceDriver(robot)
    else:
        driver = RobotDeviceDriver(robot)

    device_id = peer_id or f"{getattr(robot, 'tool_name_str', 'robot')}-{uuid.uuid4().hex[:4]}"

    urls = [messaging_url] if messaging_url else None

    # Resolve messaging_backend: explicit arg > env var > default "zenoh"
    if messaging_backend is None:
        messaging_backend = os.environ.get("MESSAGING_BACKEND", "zenoh")

    # Resolve allow_insecure: explicit arg > env var > secure default.
    # Security hardening: insecure (unencrypted, unauthenticated) transport is
    # NO LONGER the default. It must be explicitly opted into — via the
    # ``allow_insecure=True`` argument or ``DEVICE_CONNECT_ALLOW_INSECURE`` env
    # var — and we log a prominent warning whenever it is active so an insecure
    # deployment is never silent.
    allow_insecure = resolve_allow_insecure(allow_insecure, os.environ.get("DEVICE_CONNECT_ALLOW_INSECURE"))

    if allow_insecure:
        logger.warning(
            "Device Connect is running in INSECURE mode (unencrypted, "
            "unauthenticated transport). Robot commands and state are exposed "
            "to the local network. Only use this on a trusted, isolated "
            "network; configure a broker / secure transport for production."
        )

    runtime = DeviceRuntime(
        driver=driver,
        device_id=device_id,
        messaging_urls=urls,
        messaging_backend=messaging_backend,
        tenant=tenant,
        allow_insecure=allow_insecure,
    )

    # Provide robot-specific heartbeat data
    runtime.set_heartbeat_provider(lambda: _build_heartbeat(robot, peer_type))

    # Start runtime in background task; store ref to prevent GC
    runtime._background_task = asyncio.create_task(runtime.run())

    logger.info(
        "Device Connect initialized: %s (%s, backend=%s, d2d=%s)", device_id, peer_type, messaging_backend, urls is None
    )
    return runtime


def init_device_connect_sync(
    robot,
    peer_id: str | None = None,
    peer_type: str = "robot",
    messaging_url: str | None = None,
    messaging_backend: str | None = None,
    tenant: str = "default",
    allow_insecure: bool | None = None,
) -> "DeviceRuntime":
    """Non-blocking sync wrapper around init_device_connect().

    Starts the DeviceRuntime on a dedicated daemon thread so the caller
    returns immediately — matching the Zenoh mesh ``init_mesh()`` pattern.
    The runtime stays alive as long as the process (daemon thread).

    Same parameters as :func:`init_device_connect`.
    """
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    runtime_holder = [None]
    error_holder = [None]

    async def _start():
        try:
            rt = await init_device_connect(
                robot,
                peer_id=peer_id,
                peer_type=peer_type,
                messaging_url=messaging_url,
                messaging_backend=messaging_backend,
                tenant=tenant,
                allow_insecure=allow_insecure,
            )
            runtime_holder[0] = rt
        except Exception as exc:
            error_holder[0] = exc
        finally:
            ready.set()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_start())
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True, name="device-connect-runtime")
    thread.start()
    ready.wait(timeout=30.0)

    if error_holder[0] is not None:
        raise error_holder[0]

    runtime = runtime_holder[0]
    if runtime is not None:
        runtime._loop = loop
        runtime._thread = thread
    return runtime


def _build_heartbeat(robot, peer_type: str) -> dict:
    """Build heartbeat payload with robot-specific metadata."""
    data = {
        "peer_type": peer_type,
        "tool_name": getattr(robot, "tool_name_str", "unknown"),
    }

    if peer_type == "robot":
        task = getattr(robot, "_task_state", None)
        if task:
            data["task_status"] = getattr(task.status, "value", "unknown")
            data["instruction"] = task.instruction or ""
            data["step_count"] = task.step_count
    elif peer_type == "sim":
        world = getattr(robot, "_world", None)
        if world:
            data["sim_time"] = world.sim_time
            data["step_count"] = world.step_count
            data["robots"] = list(world.robots.keys())

    return data
