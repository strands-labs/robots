"""Unified Robot factory - convenience layer over ``strands_robots.simulation``
and ``strands_robots.hardware_robot``.

Provides:
    - ``Robot("so100")`` → returns a simulation by default (safe)
    - ``Robot("so100", mode="real")`` → explicit real hardware
    - ``Robot("so100", mode="auto")`` → auto-detects sim/real
    - ``list_robots()``  → what's available

Environment Variables:
    STRANDS_ROBOT_MODE: Override mode detection ("sim", "real", "auto").
        Case-insensitive; surrounding whitespace ignored.
    STRANDS_MESH: Opt a bare ``Robot()`` into the Zenoh mesh. Mesh is OFF
        unless asked for: only "true"/"1"/"yes" turns it ON, and unset or
        "false" leaves it OFF. An explicit ``mesh=True``/``mesh=False``
        kwarg wins over this default, except that "false" is a hard kill
        switch honoured by ``init_mesh`` even against ``mesh=True``.

Examples::

    # Default: simulation (safe - no physical hardware interaction)
    sim = Robot("so100")

    # Explicit real hardware
    hw = Robot("so100", mode="real", cameras={...})

    # Auto-detect (probes USB for servo controllers)
    robot = Robot("so100", mode="auto")

    # With custom URDF/MJCF path
    sim = Robot("my_arm", urdf_path="/path/to/robot.xml")

GPU backends (resolved through ``create_simulation``; require the backend's
optional dependency or the ``strands-robots-sim`` plugin to be installed)::

    sim = Robot("unitree_go2", backend="isaac", num_envs=4096)
    sim = Robot("so100", backend="newton", num_envs=4096, device="cuda:0")
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from strands_robots.registry import (
    get_hardware_type,
    get_robot,
    has_hardware,
    has_sim,
    is_discoverable,
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
        3. Default to sim (safest - never accidentally send commands to hardware)
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
            servo_keywords = ["feetech", "dynamixel", "sts3215", "xl430", "xl330", "ch340", "ch343"]
            # Servo-bus USB bridge vendor IDs. Feetech/SO-10x controller boards
            # carry WCH CH34x chips that enumerate with the generic description
            # "USB Single Serial" (observed on macOS with SO-101, vid 0x1a86
            # pid 0x55d3), so keyword matching alone misses them entirely and
            # mode="auto" silently falls back to sim with hardware attached.
            servo_vids = {0x1A86, 0x0403}  # WCH CH34x, FTDI
            exclude = ["bluetooth", "internal", "debug", "apple", "modem"]
            robot_ports = [
                p
                for p in ports
                if (
                    any(
                        kw in ((p.description or "") + (getattr(p, "manufacturer", None) or "")).lower()
                        for kw in servo_keywords
                    )
                    or (getattr(p, "vid", None) in servo_vids)
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


def _mesh_env_opt_in() -> bool:
    """Return True when ``STRANDS_MESH`` opts a bare ``Robot()`` into the mesh.

    Mesh is OFF unless it is asked for. ``mesh=None`` (the factory default)
    consults this function, and only ``true``, ``1`` or ``yes``
    (case-insensitive, surrounding whitespace ignored) turns it ON. Every
    other value -- including unset, empty and ``false`` -- leaves mesh OFF,
    so a bare ``Robot("so100")`` never spins up Zenoh, ACL or e-stop
    machinery.

    ``STRANDS_MESH=false`` is additionally a hard kill switch honoured by
    :func:`strands_robots.mesh.init_mesh` even when the caller passed an
    explicit ``mesh=True``; the env var never forces mesh ON there, so an
    explicit ``mesh=False`` is always respected.

    Returns:
        True when the environment opts in, False otherwise.
    """
    return os.getenv("STRANDS_MESH", "").strip().lower() in ("true", "1", "yes")


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
    if (
        get_robot(canonical) is None
        and not (has_sim(canonical) or has_hardware(canonical))
        and not is_discoverable(canonical)
    ):
        # A leader arm is an input device, not a robot: it carries the same
        # servo bus and USB-serial shape as its follower, so "so101_leader"
        # reads as a plausible Robot() name. Answering it with the generic
        # registry listing invites the caller to retry with the follower name
        # on the leader's port - the exact mistake that torque-enables the arm
        # a human is holding. Name the teleoperator entry point instead.
        if canonical.endswith("_leader"):
            raise ValueError(
                f"{original!r} is a teleoperator (leader) device, not a robot. "
                f"Build it with ``Teleoperator({canonical!r}, port=...)`` and attach it to the "
                f"follower it drives: ``Robot('<follower>', mode='real', port=...)"
                f".attach_teleop({canonical!r}, port=...)``. Passing a leader to ``Robot()`` would "
                "drive the arm a human is holding as a position servo."
            )
        raise ValueError(
            f"Unknown robot {original!r} (resolved to {canonical!r}). "
            "Pass a registered name (see ``list_robots()``), one of the "
            "``robot_descriptions`` robots (see ``list_discoverable()``), "
            "or supply ``urdf_path=``."
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
    *,
    orientation: list[float] | None = ...,
    keyframe: str | int | None = ...,
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
    *,
    orientation: list[float] | None = ...,
    keyframe: str | int | None = ...,
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
    *,
    orientation: list[float] | None = ...,
    keyframe: str | int | None = ...,
    **kwargs: Any,
) -> Simulation | HardwareRobot: ...


def Robot(  # noqa: N802 - uppercase by design (factory mimicking a class constructor)
    name: str,
    mode: str = "sim",
    backend: str = "mujoco",
    urdf_path: str | None = None,
    cameras: dict[str, dict[str, Any]] | None = None,
    position: list[float] | None = None,
    data_config: str | None = None,
    mesh: bool | None = None,
    peer_id: str | None = None,
    # Keyword-only: appended after the existing parameters, so no positional
    # call shifts meaning and the overloads above stay positionally faithful.
    *,
    orientation: list[float] | None = None,
    keyframe: str | int | None = None,
    **kwargs: Any,
) -> Simulation | HardwareRobot:
    """Create a robot - returns a Simulation or HardwareRobot instance.

    This is a convenience factory, NOT a wrapper class.  You get the real
    backend instance back - with full access to all its methods.

    Defaults to simulation mode so that ``Robot("so100")`` never
    accidentally sends commands to physical hardware.  Use
    ``mode="real"`` to explicitly opt into hardware control.

    Args:
        name: Robot name ("so100", "aloha", "unitree_g1", "panda", ...)
              Accepts any alias defined in ``registry/robots.json``.
        mode: "sim" (default - safe), "real" (explicit hardware), or
              "auto" (probes USB for servo controllers, falls back to sim).
              Case-insensitive; surrounding whitespace ignored.
        backend: Simulation backend name or alias, resolved through
                 ``strands_robots.simulation.create_simulation`` - the same
                 registry that powers ``create_simulation()`` directly. Built-in:
                 ``"mujoco"`` (CPU, default; aliases ``"mj"``/``"mjc"``/``"mjx"``)
                 and ``"newton"`` (GPU, warp-lang based; alias ``"nt"``). Heavy
                 out-of-tree backends (``"isaac"``) ship in the ``strands-robots-sim``
                 plugin package and resolve once installed. An unavailable backend
                 surfaces the factory's actionable install hint. Backend-specific
                 kwargs (e.g. ``num_envs``, ``device``) are forwarded to the
                 backend constructor. Only applies to ``mode="sim"``; ignored for
                 ``mode="real"``.
        urdf_path: Explicit path to URDF/MJCF file. If not provided,
                   resolved via ``strands_robots.simulation.model_registry``
                   (asset manager or ``STRANDS_ASSETS_DIR`` search paths).
                   Only applies to ``mode="sim"``; in ``mode="real"`` it is
                   ignored and reported at debug level.
        cameras: Camera config for real hardware. Example::

            {"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}}

            Note: In ``mode="sim"``, cameras must be added after creation
            via the simulation tool (``add_camera`` action). They cannot
            be passed to the factory yet.

        position: Robot base position in the sim world, ``[x, y, z]``. Passed
            through to the backend's ``add_robot`` verbatim, so the backend's
            contract governs: omitting it spawns at the origin, and a
            wrong-length, non-numeric or non-finite vector is refused with an
            actionable message (surfaced here as ``RuntimeError``) instead of
            being replaced by the origin. Only applies to ``mode="sim"``; in
            ``mode="real"`` it is ignored and reported at debug level.
        data_config: Data configuration name for observation/action schema.
                     For multi-camera setups, specify explicitly:
                     ``data_config="so100_dualcam"``. Honoured in both modes:
                     ``mode="sim"`` defaults it to the canonical robot name,
                     and ``mode="real"`` forwards it verbatim to
                     ``strands_robots.hardware_robot.Robot``, which carries it
                     into the ``policy_config`` a policy is built with.
        orientation: Robot base orientation in the sim world as a quaternion
            ``[w, x, y, z]``. Forwarded to the backend's ``add_robot`` verbatim
            on the same terms as ``position``: omitting it spawns unrotated, and
            a wrong-length, non-numeric or non-finite quaternion is refused with
            an actionable message (surfaced here as ``RuntimeError``) rather
            than being replaced by the identity rotation. Only applies to
            ``mode="sim"``; in ``mode="real"`` it is ignored and reported at
            debug level.
        keyframe: Spawn the robot in a canonical pose declared by a
            ``<keyframe>`` in its source model (e.g. panda ``"home"``, aloha
            ``"neutral_pose"``) instead of the default all-zero configuration.
            Accepts the keyframe name (``str``) or index (``int``) and is
            forwarded to the backend's ``add_robot`` verbatim, so that method's
            contract governs: the pose is sticky across ``reset()`` and an
            unknown keyframe is a hard error naming the available keyframes
            (surfaced here as ``RuntimeError``) instead of a silent zero-pose
            spawn. Only applies to ``mode="sim"``; in ``mode="real"`` it is
            ignored and reported at debug level.
        mesh: Attach a Zenoh fleet-coordination mesh. ``None`` (default) keeps
              mesh OFF for a quiet bare ``Robot(...)`` but honours the
              ``STRANDS_MESH=true`` opt-in. Pass ``True`` to force it on or
              ``False`` to force it off; an explicit value always wins over the
              env default. ``STRANDS_MESH=false`` is a hard kill switch.
        peer_id: Optional mesh peer identifier. Auto-generated when omitted.
        **kwargs: Forwarded to the underlying backend constructor.

    Returns:
        ``strands_robots.simulation.Simulation`` (sim) or
        ``strands_robots.hardware_robot.Robot`` (real hardware).

    Raises:
        ValueError: If ``mode`` is not 'sim'/'real'/'auto', if ``cameras=``
                    is passed in sim mode, if the robot name is empty
                    or not in the registry (and no ``urdf_path=`` given), or if
                    ``backend`` is not a known backend (the message lists the
                    available backends and any install hint).
        ImportError: If a known backend's optional dependency is missing
                     (e.g. ``newton`` without ``warp``); the message names the
                     pip extra to install.
        RuntimeError: If the sim world or robot fails to initialize.

    Examples::

        # Simulation (default - safe)
        sim = Robot("so100")

        # Explicit MJCF model path
        sim = Robot("my_arm", urdf_path="path/to/robot.xml")

        # Real hardware (explicit opt-in)
        hw = Robot("so100", mode="real", cameras={...})

        # Auto-detect (probes USB, falls back to sim)
        robot = Robot("so100", mode="auto")

        # The 5-line promise (defaults to sim - safe, no hardware needed)
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

    # Resolve the mesh opt-in. Mesh is OFF by default so a bare
    # ``Robot("so100")`` is quiet and never spins up Zenoh/ACL/e-stop
    # machinery. ``mesh=None`` (the default) means "consult the
    # STRANDS_MESH env var": STRANDS_MESH=true opts in without a code
    # change. An explicit ``mesh=True``/``mesh=False`` always wins over
    # the env default. ``STRANDS_MESH=false`` remains a hard kill switch
    # enforced in ``init_mesh`` regardless of this resolution.
    if mesh is None:
        mesh = _mesh_env_opt_in()

    # --- Simulation ---
    if mode == "sim":
        if cameras is not None:
            raise ValueError(
                "cameras= is only supported in mode='real'. "
                "For sim cameras, add them via the simulation tool's "
                "'add_camera' action after creation."
            )

        from strands_robots.simulation import create_simulation

        # Resolve the backend through create_simulation - the single source of
        # truth for backend selection (built-in registry + entry-point plugins +
        # runtime registrations). The MuJoCo path is unchanged (returns the same
        # MuJoCoSimEngine); ``newton`` and plugin backends (``isaac``) construct
        # with the forwarded kwargs (``num_envs``, ``device``, ...), and an
        # unavailable backend surfaces the factory's actionable install hint
        # (e.g. ``pip install strands-robots[sim-newton]``) instead of a blanket
        # NotImplementedError. World/robot population goes through the backend-
        # agnostic SimEngine ABC methods, so it works for every backend.
        # The sim-mode overloads contract a ``Simulation`` return; create_simulation
        # is typed to the SimEngine ABC, so cast to keep that public contract.
        sim = cast("Simulation", create_simulation(backend, tool_name=f"{name}_sim", **kwargs))

        try:
            result = sim.create_world()
            if result.get("status") == "error":
                content = result.get("content", [])
                msg = content[0].get("text", str(result)) if content else str(result)
                raise RuntimeError(f"Failed to create sim world for {canonical!r}: {msg}")

            # Forward ``position`` exactly as the caller wrote it. Reading it
            # by truthiness (``position or [0.0, 0.0, 0.0]``) broke the
            # parameter twice over: a NumPy pose - what any pose arithmetic
            # produces, and a value ``add_robot`` itself accepts - raised a bare
            # ``ValueError: truth value of an array ... is ambiguous`` out of the
            # factory, and an empty vector read as "omitted", so the origin was
            # substituted for a caller mistake that ``add_robot`` refuses with an
            # actionable message. Membership (``is not None``) is the only
            # correct supplied-test for a vector, and here even that is
            # unnecessary: ``position=None`` is what ``add_robot`` already
            # documents as "spawn at the origin", so passing it through keeps ONE
            # source of truth for that default instead of a copy that can drift.
            # ``orientation`` and ``keyframe`` travel with ``position`` for the
            # same reason it is passed through unmodified: all three are
            # ``SimEngine.add_robot`` parameters whose defaults and refusals that
            # method already documents, so forwarding the caller's value verbatim
            # keeps ONE source of truth for each. Dropping them here made the
            # factory silently disagree with the backend it delegates to - a
            # requested rotation or keyframe spawn was absorbed by ``**kwargs``
            # and the robot came up unrotated in the zero configuration, and the
            # refusals ``add_robot`` raises for a malformed quaternion or an
            # unknown keyframe never reached the caller.
            result = sim.add_robot(
                name=name,
                urdf_path=urdf_path,
                data_config=data_config or canonical,
                position=position,
                orientation=orientation,
                keyframe=keyframe,
            )
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
        # Failure to start the mesh must NOT bring down the sim - the user
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
                # The robot was added BEFORE the mesh existed (create_world ->
                # add_robot -> init_mesh), so _attach_robot_to_mesh was a no-op
                # at add_robot time. Attach the already-added robots now so
                # each SimRobot gets its own child peer (which is what
                # publishes per-robot joint state on strands/<peer>/state).
                if hasattr(sim, "_attach_robot_to_mesh"):
                    world = getattr(sim, "_world", None)
                    for _sim_robot in (getattr(world, "robots", None) or {}).values():
                        if getattr(_sim_robot, "mesh", None) is None:
                            sim._attach_robot_to_mesh(_sim_robot)
        except Exception as exc:  # noqa: BLE001 - mesh enrichment is best-effort
            logger.warning("Failed to initialise mesh for %r: %s", canonical, exc)

        _attach_device_connect(sim, canonical, mode, peer_id)
        return sim

    # --- Real hardware (explicit opt-in) ---
    elif mode == "real":
        if backend != "mujoco":
            logger.debug(
                "backend=%r ignored in mode='real' (hardware uses direct servo control)",
                backend,
            )

        # Report the spawn parameters this branch cannot honour. They describe a
        # pose in a simulated world - where to place a base, how to rotate it,
        # which <keyframe> to spawn in, which model file to load - and a physical
        # arm is already wherever it is, so ``hardware_robot.Robot`` accepts none
        # of them. Ignoring them is right; ignoring them silently is not, which is
        # why ``backend`` above says so. Read by ``is not None`` rather than by
        # truthiness: ``keyframe=0`` is a valid keyframe index and an empty
        # ``position``/``orientation`` was still supplied, so a truthiness test
        # would drop exactly the values a caller is most likely to be surprised by.
        ignored_spawn_args = [
            f"{param}={value!r}"
            for param, value in (
                ("urdf_path", urdf_path),
                ("position", position),
                ("orientation", orientation),
                ("keyframe", keyframe),
            )
            if value is not None
        ]
        if ignored_spawn_args:
            logger.debug(
                "%s ignored in mode='real' (spawn pose applies to a simulated world; "
                "a physical robot is already where it is)",
                ", ".join(ignored_spawn_args),
            )

        from strands_robots.hardware_robot import Robot as HardwareRobotCls

        real_type = get_hardware_type(canonical) or canonical
        hw = HardwareRobotCls(
            tool_name=canonical,
            robot=real_type,
            cameras=cameras,
            # Forwarded, not reported: unlike the spawn parameters above, the
            # hardware class declares ``data_config`` and reads it - it is what
            # ends up in the ``policy_config`` a policy is built with, so a
            # multi-camera schema selected here has to arrive. Passed verbatim
            # (``None`` is the hardware class's own default) rather than defaulted
            # to the canonical name the way the sim path does, so a caller who
            # names no config keeps today's behaviour.
            data_config=data_config,
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
        except Exception as exc:  # noqa: BLE001 - mesh enrichment is best-effort
            logger.warning("Failed to initialise mesh for %r: %s", canonical, exc)

        _attach_device_connect(hw, canonical, mode, peer_id)
        return hw

    else:
        raise ValueError(f"Invalid mode {mode!r}. Choose 'sim', 'real', or 'auto' (case-insensitive).")


def _attach_device_connect(instance: Any, canonical: str, mode: str, peer_id: str | None) -> None:
    """Attach a Device Connect ``.run()`` server hook to a robot/sim instance.

    Mirrors the mesh attach above: stores peer metadata and binds ``.run()`` so
    ``Robot("so100").run()`` brings the device online as a Device Connect device
    (the primary networking layer), blocking until Ctrl+C.
    """
    instance._peer_id = peer_id or getattr(instance, "peer_id", None) or f"{canonical}-{os.urandom(3).hex()}"
    instance._peer_type = "sim" if mode == "sim" else "robot"
    instance._device_connect_runtime = None
    instance.run = lambda: _run_device_connect_foreground(instance)


#: Seconds to wait for the Ctrl+C teardown before exiting anyway. Matches
#: ``MuJoCoSimEngine._DEFAULT_POLICY_STOP_TIMEOUT``, which bounds the same
#: decision on the same event: a teardown step that will not finish must not
#: keep the process alive on the way out.
_SHUTDOWN_TIMEOUT_S: float = 5.0


def _release_resources_on_interrupt(instance: Any, peer_id: str) -> str | None:
    """Run the instance's terminal teardown for Ctrl+C, bounded by a budget.

    ``cleanup()`` is where a robot releases what it holds, and on hardware that
    includes the physical devices: it reaches the driver's own ``disconnect()``,
    which is where torque disable and gripper release live. Nothing else in the
    library reaches them - ``cleanup()`` is terminal, so no entry point runs
    after it, and lerobot's ``Robot.disconnect()`` refuses to be called by hand
    once the robot is half-open. So the teardown either happens here or it does
    not happen at all: the caller ends with ``os._exit``, which runs no
    ``atexit`` hook, no ``__del__`` and no ``finally`` block.

    The budget, and why the exit stays abrupt: ``Robot.cleanup()`` drains its
    task executor with ``shutdown(wait=True)``, which a wedged rollout does not
    finish - measured, a submitted item that never returns keeps that call
    running indefinitely, and a ``ThreadPoolExecutor`` worker is not a daemon
    thread, so the interpreter's own exit hook would then join it too. Awaiting
    the teardown on the calling thread, or returning normally and letting the
    interpreter tear down, therefore turns one operator Ctrl+C into a process
    that never exits - and an operator who reaches for ``SIGKILL`` gets no
    teardown at all, which is the outcome this exists to prevent. Running it on
    a daemon thread with a budget keeps the guarantee that Ctrl+C ends the
    process, on the same reasoning
    :meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine.cleanup`
    already applies to a wedged policy worker: bound the wait, report, proceed.

    Args:
        instance: The robot or simulation the runner was bound to. An instance
            exposing no callable ``cleanup`` holds nothing this can release, so
            it is reported as released rather than as a failure.
        peer_id: Peer identifier, used to name the teardown thread.

    Returns:
        ``None`` when the teardown ran to completion, so the caller may report
        the robot stopped. Otherwise a sentence naming what stopped it -- the
        budget expiring, the teardown raising, or a second Ctrl+C arriving
        during the wait -- for a caller that must not claim more than happened.
    """
    cleanup = getattr(instance, "cleanup", None)
    if not callable(cleanup):
        return None

    failure: list[BaseException] = []
    finished = threading.Event()

    def _teardown() -> None:
        try:
            cleanup()
        except Exception as exc:  # noqa: BLE001 - reported to the operator below
            failure.append(exc)
        finally:
            finished.set()

    threading.Thread(target=_teardown, name=f"{peer_id}-shutdown", daemon=True).start()

    try:
        released = finished.wait(timeout=_SHUTDOWN_TIMEOUT_S)
    except KeyboardInterrupt:
        # An impatient operator interrupting again lands here rather than in the
        # loop above, and it must not escape: the caller's ``os._exit`` is what
        # guarantees the process ends, and letting this propagate would instead
        # unwind into interpreter shutdown, where the executor drain this wait
        # is covering gets joined a second time by ``concurrent.futures``' own
        # exit hook. Before this budget existed there was no window to interrupt,
        # so the second Ctrl+C has to keep behaving the way the first one did.
        logger.warning("%s: shutdown interrupted again; exiting immediately.", peer_id)
        return "the shutdown was interrupted again before it finished."

    if not released:
        logger.warning(
            "%s: cleanup() did not finish within %gs; exiting anyway.",
            peer_id,
            _SHUTDOWN_TIMEOUT_S,
        )
        return f"cleanup() did not finish within {_SHUTDOWN_TIMEOUT_S:g}s."
    if failure:
        logger.warning("%s: cleanup() raised during shutdown: %s", peer_id, failure[0])
        return f"cleanup() raised {type(failure[0]).__name__}: {failure[0]}."
    return None


def _run_device_connect_foreground(instance: Any) -> None:
    """Start Device Connect and block - the robot listens for commands.

    Device Connect is the primary networking layer in server mode, so the
    auto-started built-in mesh (if any) is stopped first to avoid running two
    Zenoh presence systems in one process.

    A bring-up that fails keeps the process alive - the operator asked for a
    server and a transient broker outage is not worth losing the process over -
    but the status line reports what actually came up. Claiming the device is
    online is only true of the path where the runtime started; on the other one
    the mesh has already been stopped for a replacement that never arrived, so
    the process serves no transport at all and the operator has to be told
    that rather than the opposite.

    The operator's Ctrl+C is the only teardown this loop ever gets, on either
    branch, so it releases the instance before exiting -- on hardware that is
    what reaches the driver's ``disconnect()``, where torque disable and gripper
    release live. The exit stays abrupt afterwards, and the shutdown line
    reports whether the release actually finished; see
    :func:`_release_resources_on_interrupt`.
    """
    import time

    peer_id = getattr(instance, "_peer_id", None) or "robot"
    peer_type = getattr(instance, "_peer_type", "robot")

    # Device Connect supersedes the built-in mesh in run() mode.
    mesh = getattr(instance, "mesh", None)
    mesh_was_stopped = mesh is not None
    if mesh is not None:
        with contextlib.suppress(Exception):
            mesh.stop()
        instance.mesh = None

    try:
        from strands_robots.device_connect import init_device_connect_sync

        instance._device_connect_runtime = init_device_connect_sync(
            instance,
            peer_id=peer_id,
            peer_type=peer_type,
        )
    except Exception as e:  # noqa: BLE001 - surface but keep the process alive
        # An absent extra is the common cause and the only one with a one-line
        # remedy, so name it here: on its own the ImportError names the
        # distribution's internal module, not the extra that installs it.
        remedy = " Install it with: pip install 'strands-robots[device-connect]'." if isinstance(e, ImportError) else ""
        logger.warning("Device Connect init failed: %s.%s", e, remedy)

    if getattr(instance, "_device_connect_runtime", None) is None:
        lost_transport = (
            "The built-in mesh was stopped for it, so this process now serves no transport."
            if mesh_was_stopped
            else "This process serves no transport."
        )
        print(
            f"{peer_id} is NOT online: the Device Connect runtime did not start "
            f"(see the warning above). {lost_transport} Ctrl+C to stop."
        )
    else:
        print(f"{peer_id} is online. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\nShutting down {peer_id}...", flush=True)
        unreleased = _release_resources_on_interrupt(instance, peer_id)
        if unreleased is None:
            print(f"{peer_id} stopped.", flush=True)
        else:
            print(
                f"{peer_id} is exiting WITHOUT a completed shutdown: {unreleased} "
                f"Devices this process held may not have been released.",
                flush=True,
            )
        os._exit(0)


__all__ = ["Robot"]
