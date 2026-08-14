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
from typing import Any

from device_connect_edge import DeviceRuntime

from strands_robots.device_connect.reachy_mini_driver import ReachyMiniDriver
from strands_robots.device_connect.robot_driver import RobotDeviceDriver
from strands_robots.device_connect.sim_driver import SimulationDeviceDriver
from strands_robots.utils import is_boolean

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

# ``init_device_connect_sync`` bounds the background bring-up: the awaited half
# constructs a ``DeviceRuntime``, which can block on a broker, so the wrapper
# must not hang its caller forever. Expiry is a failed bring-up, not a slow one
# that later succeeds -- the wrapper has already returned by then.
_INIT_TIMEOUT_S: float = 30.0


def resolve_allow_insecure(
    explicit: bool | None = None,
    env_value: str | None = None,
) -> bool:
    """Resolve the effective ``allow_insecure`` setting (secure by default).

    Precedence: explicit arg > ``DEVICE_CONNECT_ALLOW_INSECURE`` env var >
    secure default (``False``). Insecure transport is never implicit - it
    must be opted into via the argument or the env var.

    Extracted as a pure function so the secure-by-default posture is unit
    testable without standing up a DeviceRuntime.

    The two sources carry the same setting in different shapes, and each is held
    to its own declared type rather than to the other's. An environment variable
    is a string by construction, so *env_value* is **parsed**: only
    ``("true", "1", "yes")`` opt in and every other spelling is secure. The
    argument is declared ``bool | None``, so it is **checked**: a non-boolean is
    refused rather than parsed with that same vocabulary.

    Checking the argument is what keeps the two sources from disagreeing about
    one value. A non-empty string is truthy, so returning the argument as given
    made ``resolve_allow_insecure("false")`` enable insecure transport while
    ``DEVICE_CONNECT_ALLOW_INSECURE=false`` disabled it: every falsy spelling
    inverted, and only on the path documented here as the higher precedence.
    Parsing the argument with the environment vocabulary instead would move
    which spellings invert rather than remove the inversion - ``"on"``,
    ``"enabled"`` and ``"y"`` are absent from that vocabulary, so each would
    silently resolve to secure while reading as an opt-in.

    Args:
        explicit: The caller's setting, or ``None`` to fall through to the
            environment variable. Must be a python or numpy boolean when given.
        env_value: The raw ``DEVICE_CONNECT_ALLOW_INSECURE`` value, or ``None``
            when it is unset.

    Returns:
        Whether insecure transport is enabled, always as a real ``bool`` - so a
        numpy boolean from a caller's own comparison satisfies the annotation
        and the identity assertions the runtime's setting is pinned with.

    Raises:
        ValueError: If *explicit* is neither a boolean nor ``None``, or
            *env_value* is neither a string nor ``None``.
    """
    if explicit is not None:
        if not is_boolean(explicit):
            raise ValueError(
                f"allow_insecure must be a bool or None, got {explicit!r}. A string "
                "spelling is read only from DEVICE_CONNECT_ALLOW_INSECURE, where "
                f"{_INSECURE_TRUE} opt in and anything else is secure; passed as this "
                "argument a non-empty string is truthy, so 'false' would enable insecure "
                "transport rather than refuse it."
            )
        return bool(explicit)
    if env_value is not None:
        if not isinstance(env_value, str):
            raise ValueError(
                f"env_value must be a str or None, got {env_value!r}. It carries the raw "
                "DEVICE_CONNECT_ALLOW_INSECURE value, which is a string by construction; a "
                "caller that has already resolved a boolean should pass it as the explicit "
                "argument instead, where it is checked rather than parsed."
            )
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
    enters D2D mode - devices discover each other directly via Zenoh
    multicast scouting on the LAN. No broker, no Docker, no env vars.

    Args:
        robot: A Robot or Simulation instance to wrap.
        peer_id: Device ID for registration (auto-generated if None).
        peer_type: "robot" or "sim" - selects the appropriate driver.
        messaging_url: Explicit messaging URL (overrides env vars).
        messaging_backend: Messaging backend - "zenoh" or "nats".
            None = auto-detect from MESSAGING_BACKEND env var (default "zenoh").
        tenant: Device Connect tenant namespace.
        allow_insecure: Allow insecure (unencrypted, unauthenticated)
            transport. Must be a boolean or None; a string spelling such as
            ``"false"`` is refused here rather than read, because the string
            vocabulary belongs to DEVICE_CONNECT_ALLOW_INSECURE and a non-empty
            string is truthy as an argument. None = auto-detect: respects the
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
    # NO LONGER the default. It must be explicitly opted into - via the
    # ``allow_insecure=True`` argument or ``DEVICE_CONNECT_ALLOW_INSECURE`` env
    # var - and we log a prominent warning whenever it is active so an insecure
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
    returns immediately - matching the Zenoh mesh ``init_mesh()`` pattern.
    The runtime stays alive as long as the process (daemon thread).

    Same parameters as :func:`init_device_connect`.

    Raises:
        Exception: Whatever :func:`init_device_connect` raised on the
            background thread, re-raised here so a failed bring-up reaches
            the caller rather than being confined to a thread it cannot see.
        TimeoutError: If the bring-up does not finish within the wrapper's
            budget. The runtime is not returned in that case, so the caller
            is never handed ``None`` in place of a ``DeviceRuntime``.
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
    started = ready.wait(timeout=_INIT_TIMEOUT_S)

    # The recorded failure first: ``_start``'s ``finally`` sets the event on both
    # paths, so a bring-up that failed inside the budget arrives with ``started``
    # true and its own exception, which names the cause better than the budget.
    if error_holder[0] is not None:
        raise error_holder[0]
    if not started:
        raise TimeoutError(
            f"init_device_connect_sync: the Device Connect runtime did not come up "
            f"within {_INIT_TIMEOUT_S:g}s. The bring-up is still running on its "
            f"background thread; check that the messaging URL / broker is reachable."
        )

    runtime = runtime_holder[0]
    if runtime is not None:
        runtime._loop = loop
        runtime._thread = thread
    return runtime


def _build_heartbeat(robot: Any, peer_type: str) -> dict[str, Any]:
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
