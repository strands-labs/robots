"""Observe actions for a real arm: read the joints and the cameras without a policy.

``Robot("so101", mode="real")`` handed to an agent used to expose four actions -
``execute``, ``start``, ``status``, ``stop`` - so the only way an agent could
*look* at a physical arm was to run a policy on it. Asked to "read the joint
positions, do not move", an agent requested a ten-second mock-policy rollout
on the real arm to do the reading (the operator gate caught it). This module
is the read side the tool lacked: joint positions, torque state, supply
voltage and a camera frame, none of which writes a servo register.

Three facts about real arms shape it:

* **Construction opens nothing.** The hardware class builds the lerobot robot
  and leaves the port closed until a rollout connects, so an observe action
  opens the motor bus itself, lazily, through ``bus.connect()`` - and NOT
  through the robot's ``connect()``, whose ``configure()`` writes operating
  mode and PID registers to every servo. Reading must stay a read.
* **The arm may be uncalibrated.** Without a lerobot calibration file
  ``sync_read("Present_Position")`` raises inside ``_normalize``, and a policy
  cannot run at all. Positions are therefore read in raw ticks always, and
  reported in degrees either from the calibration (when there is one) or as
  an estimate from the servo's own encoder (``(ticks - 2048) * 360 / 4096``
  for a 12-bit servo), labelled as such. An agent that cannot read an
  uncalibrated arm cannot tell the operator to calibrate it - and an arm whose
  calibration flag could not be read is *unread*, not uncalibrated
  (:func:`read_calibration_flag`).
* **Reads share the bus.** Every read takes the device's
  :func:`~strands_robots.bus_access.bus_lock`, the lock the mesh probes and a
  rollout already hold, so an observe action beside a rollout is serialised
  rather than colliding ("Port is in use!").

Nothing here imports lerobot: the bus is duck-typed on the surface lerobot's
``MotorsBus`` exposes (``motors``, ``sync_read``, ``read``, ``is_connected``,
``is_calibrated``, ``connect``, ``port``), which is what lets the tests run on a
fake bus and the same code run on the real one.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from strands_robots.bus_access import bus_lock
from strands_robots.simulation.safe_output import env_flag, resolve_sandbox_root, validate_output_path

logger = logging.getLogger(__name__)

__all__ = [
    "OBSERVE_ACTIONS",
    "TICKS_PER_REV",
    "TICKS_CENTRE",
    "ticks_to_degrees",
    "ensure_bus_open",
    "read_calibration_flag",
    "read_joint_state",
    "format_joint_state",
    "list_cameras",
    "format_cameras",
    "capture_frame",
]

#: The actions this module answers. None of them writes a servo register, so
#: none of them is gated: they are the actions an agent may take to find out
#: what it is about to ask an operator to approve.
OBSERVE_ACTIONS: frozenset[str] = frozenset({"get_state", "get_robot_state", "list_cameras", "render"})

#: Encoder resolution of the 12-bit serial servos (Feetech STS/SCS, Dynamixel
#: X-series) every lerobot arm this class drives is built from. Used only for
#: the *uncalibrated* estimate and named in the text that carries it.
TICKS_PER_REV = 4096
#: The encoder value lerobot's calibration treats as the joint's centre.
TICKS_CENTRE = 2048

#: Registers read per motor beside the position. Each is best-effort: a bus
#: whose control table lacks one reports the key as absent, not the read as failed.
_TORQUE_REGISTER = "Torque_Enable"
_VOLTAGE_REGISTER = "Present_Voltage"
_POSITION_REGISTER = "Present_Position"

#: The render sandbox is the simulation's: one directory for "a frame the agent
#: asked for", whichever engine or camera produced it, and one opt-in variable.
_RENDER_ROOT_ENV = "STRANDS_ROBOTS_RENDER_ROOT"
_RENDER_ALLOW_ABS_ENV = "STRANDS_ROBOTS_RENDER_ALLOW_ABS"


def ticks_to_degrees(ticks: float) -> float:
    """Degrees from the encoder centre, for an arm with no calibration to say better."""
    return (float(ticks) - TICKS_CENTRE) * 360.0 / TICKS_PER_REV


def _bus(robot: Any) -> Any:
    bus = getattr(robot, "bus", None)
    if bus is None or not hasattr(bus, "sync_read"):
        raise RuntimeError(
            f"{robot} exposes no motor bus to read (no `bus.sync_read`); "
            "get_state needs a lerobot arm whose driver owns a MotorsBus."
        )
    return bus


def ensure_bus_open(robot: Any) -> bool:
    """Open the motor bus if it is closed. Returns True when this call opened it.

    Opens the *bus only*: the robot's own ``connect()`` also runs
    ``configure()``, which writes operating-mode and gain registers to every
    servo, and a read must not do that. The port a rollout later needs is the
    same one, and the driver's ``connect()`` cannot start from an open port -
    so the caller records a ``True`` return and hands the bus back before any
    connect (``Robot._hand_back_observe_devices``). Left open it is worse than
    a refused connect: on an arm with no cameras the open bus alone reads as
    ``is_connected``, and a rollout would drive servos ``configure()`` never
    set up.
    """
    bus = _bus(robot)
    if getattr(bus, "is_connected", False):
        return False
    with bus_lock(robot):
        bus.connect()
    logger.info("opened motor bus %s for a read", getattr(bus, "port", "?"))
    return True


def read_calibration_flag(bus: Any) -> tuple[bool | None, str]:
    """The bus's calibration verdict, and the reason when there is none.

    Three answers, not two:

    * ``True``/``False`` - the flag the bus gave.
    * ``None`` - the bus declares ``is_calibrated`` and reading it raised. On a
      lerobot arm that read is ``read_calibration()``, a homing-offset and range
      sweep of every servo, so a contended or half-powered bus fails it; an
      unanswered question is not a "no", and reporting it as one both mislabels
      the degrees and sends the operator to ``lerobot-calibrate`` for a fault
      that is not calibration.
    * A bus with **no notion** of calibration answers ``True``, by lerobot's own
      contract for the property ("should be always True if not applicable") -
      the same reading the connect gate takes, so the read side and the motion
      side cannot disagree about the same arm.

    The write side and the read side differ deliberately on ``None``: the connect
    gate refuses, because nothing may move on an unanswered question, while a
    read that already has the positions in hand reports them and labels the
    calibration unread.
    """
    try:
        return bool(bus.is_calibrated), ""
    except AttributeError as exc:
        if hasattr(type(bus), "is_calibrated"):
            return None, str(exc).strip()  # the property exists; its own read failed
        return True, ""  # no notion of calibration: lerobot calls that calibrated
    except Exception as exc:  # noqa: BLE001 - any failed sweep is "unread", not "uncalibrated"
        return None, str(exc).strip()


def _read_register(bus: Any, register: str, motor: str) -> Any:
    """One register of one motor, or ``None`` for a register this bus cannot answer."""
    try:
        return bus.read(register, motor, normalize=False)
    except TypeError:
        try:
            return bus.read(register, motor)
        except Exception as exc:  # noqa: BLE001 - a register a bus lacks is "absent", not a failed read
            logger.debug("%s read of %s failed: %s", register, motor, exc)
            return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s read of %s failed: %s", register, motor, exc)
        return None


def read_joint_state(robot: Any) -> dict[str, Any]:
    """Read every motor's position, torque state and supply voltage. Writes nothing.

    Returns a JSON-ready dict::

        {"port": ..., "calibrated": bool | None, "opened_bus": bool,
         "joints": {name: {"id", "ticks", "degrees", "degrees_source", "torque_enabled", "voltage_v"}},
         "torque_enabled_any": bool | None}

    ``calibrated`` is ``None`` when the bus could not answer (see
    :func:`read_calibration_flag`), and ``calibration_error`` then carries why;
    ``torque_enabled_any`` is ``None`` when no motor answered the torque
    register. A field nothing measured stays unset rather than defaulting to the
    reassuring answer.

    ``degrees_source`` is ``"calibration"`` when lerobot normalised the reading
    and ``"encoder_estimate"`` when it is :func:`ticks_to_degrees` of the raw
    value because the arm has no calibration. The gripper's calibrated unit is
    lerobot's 0-100 range, reported under ``normalized`` rather than forced
    into degrees.

    Raises:
        Exception: Whatever the bus raises on a failed open or read, unchanged,
            so the caller can name the port in its refusal.
    """
    bus = _bus(robot)
    opened = ensure_bus_open(robot)
    with bus_lock(robot):
        raw = bus.sync_read(_POSITION_REGISTER, normalize=False)
        calibrated, calibration_error = read_calibration_flag(bus)
        normalized: Mapping[str, Any] = {}
        if calibrated is not False:
            try:
                normalized = bus.sync_read(_POSITION_REGISTER)
            except Exception as exc:  # noqa: BLE001 - the raw read already succeeded; degrees fall back to the estimate
                logger.debug("normalized position read failed on a calibrated bus: %s", exc)
                normalized = {}
        torque = {name: _read_register(bus, _TORQUE_REGISTER, name) for name in raw}
        voltage = {name: _read_register(bus, _VOLTAGE_REGISTER, name) for name in raw}

    motors = getattr(bus, "motors", {}) or {}
    joints: dict[str, dict[str, Any]] = {}
    for name, ticks in raw.items():
        motor = motors.get(name) if isinstance(motors, Mapping) else None
        mode = getattr(getattr(motor, "norm_mode", None), "name", None)
        entry: dict[str, Any] = {
            "id": getattr(motor, "id", None),
            "ticks": int(ticks),
            "torque_enabled": bool(torque.get(name)) if torque.get(name) is not None else None,
        }
        if name in normalized:
            if mode == "DEGREES" or mode is None:
                entry["degrees"] = round(float(normalized[name]), 2)
                entry["degrees_source"] = "calibration"
            else:
                entry["normalized"] = round(float(normalized[name]), 2)
                entry["normalized_unit"] = mode.lower()
        else:
            entry["degrees"] = round(ticks_to_degrees(ticks), 2)
            entry["degrees_source"] = "encoder_estimate"
        volts = voltage.get(name)
        if volts is not None:
            entry["voltage_v"] = round(float(volts) / 10.0, 1)
        joints[str(name)] = entry

    # A torque register no motor answered leaves the question open. ``any()``
    # over ``None`` would close it as "off", i.e. "safe to grab" for an arm
    # whose torque was never read.
    answers = [j["torque_enabled"] for j in joints.values() if j.get("torque_enabled") is not None]
    state: dict[str, Any] = {
        "port": getattr(bus, "port", None),
        "calibrated": calibrated,
        "opened_bus": opened,
        "joints": joints,
        "torque_enabled_any": any(answers) if answers else None,
    }
    if calibration_error:
        state["calibration_error"] = calibration_error
    return state


def format_joint_state(tool_name: str, state: Mapping[str, Any]) -> str:
    """The text an agent reads: one header that says what matters, one line per joint."""
    joints: Mapping[str, Mapping[str, Any]] = state.get("joints", {})
    n = len(joints)
    torque_on = [name for name, j in joints.items() if j.get("torque_enabled") is True]
    torque_off = [name for name, j in joints.items() if j.get("torque_enabled") is False]
    torque_unread = [name for name, j in joints.items() if j.get("torque_enabled") is None]
    if not torque_on and not torque_off:
        # "OFF" here would read as "safe to grab" for an arm nobody asked.
        torque_line = "torque state UNREAD on every joint (do not assume the arm is free to move by hand)"
    elif not torque_on and not torque_unread:
        torque_line = "torque OFF on all joints (the arm can be moved by hand)"
    elif len(torque_on) == n:
        torque_line = "torque ON on all joints (holding position)"
    elif not torque_unread:
        torque_line = f"torque ON on {', '.join(torque_on)}; OFF on the rest"
    else:
        parts = [f"ON on {', '.join(torque_on)}"] if torque_on else []
        parts += [f"OFF on {', '.join(torque_off)}"] if torque_off else []
        parts += [f"UNREAD on {', '.join(torque_unread)}"]
        torque_line = "torque " + "; ".join(parts)
    calibrated = state.get("calibrated")
    if calibrated is None:
        calib_line = (
            f"calibration state UNREAD ({state.get('calibration_error') or 'the bus did not answer'}): "
            "each joint below names its own source, `calibration` or `encoder_estimate`"
        )
    elif calibrated:
        calib_line = "calibrated"
    else:
        calib_line = (
            "NOT calibrated: degrees below are an estimate from the encoder centre "
            f"({TICKS_CENTRE} ticks = 0°, {TICKS_PER_REV} ticks/rev); a policy rollout (execute/start) "
            "refuses until `lerobot-calibrate` has run for this arm"
        )
    lines = [f"{tool_name} (real arm on {state.get('port')}): {n} joints, {torque_line}, {calib_line}."]
    for name, j in joints.items():
        pos = f"{j['degrees']:+.1f}°" if "degrees" in j else f"{j.get('normalized')} ({j.get('normalized_unit')})"
        extra = f"  {j['voltage_v']} V" if "voltage_v" in j else ""
        torque = "on" if j.get("torque_enabled") else ("off" if j.get("torque_enabled") is False else "?")
        lines.append(f"  {name} (id {j.get('id')}): {pos}  [{j['ticks']} ticks]  torque {torque}{extra}")
    return "\n".join(lines)


def list_cameras(robot: Any) -> list[dict[str, Any]]:
    """The cameras this arm was configured with and whether each is open. Opens nothing."""
    cameras = getattr(robot, "cameras", None)
    configs = getattr(getattr(robot, "config", None), "cameras", None)
    out: list[dict[str, Any]] = []
    if not isinstance(cameras, Mapping):
        return out
    for name, camera in cameras.items():
        cfg = configs.get(name) if isinstance(configs, Mapping) else None
        entry: dict[str, Any] = {
            "name": str(name),
            "type": type(camera).__name__,
            "connected": bool(getattr(camera, "is_connected", False)),
        }
        for field in ("index_or_path", "width", "height", "fps"):
            value = getattr(cfg, field, None)
            if value is not None:
                entry[field] = str(value) if field == "index_or_path" else value
        out.append(entry)
    return out


def format_cameras(tool_name: str, cameras: Sequence[Mapping[str, Any]]) -> str:
    """The text an agent reads: one line per camera, or how to configure one."""
    if not cameras:
        return (
            f"{tool_name} has no cameras configured. Pass cameras= to Robot(mode='real'), e.g. "
            "cameras={'front': {'type': 'opencv', 'index_or_path': 0, 'width': 1280, 'height': 720, 'fps': 30}}."
        )
    lines = [f'{tool_name}: {len(cameras)} camera(s). render {{"camera_name": <name>}} grabs one frame.']
    for cam in cameras:
        size = f"{cam['width']}x{cam['height']}" if "width" in cam and "height" in cam else "size unset"
        fps = f"@{cam['fps']}fps" if "fps" in cam else ""
        src = f" source {cam['index_or_path']}" if "index_or_path" in cam else ""
        state = "open" if cam.get("connected") else "closed (opens on first render)"
        lines.append(f"  {cam['name']}: {cam['type']}{src} {size}{fps} - {state}")
    return "\n".join(lines)


def _encode_png(frame: Any) -> bytes:
    """PNG bytes from an RGB uint8 array, through whichever encoder is installed."""
    try:
        import cv2  # lerobot's OpenCV camera already depends on it

        ok, buf = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("cv2.imencode returned False")
        return bytes(buf.tobytes())
    except ImportError:
        import io

        from PIL import Image

        out = io.BytesIO()
        Image.fromarray(frame).save(out, format="PNG")
        return out.getvalue()


def capture_frame(
    robot: Any,
    camera_name: str | None,
    output_path: str | None = None,
    *,
    tool_name: str = "robot",
    sandbox_root: Path | None = None,
) -> dict[str, Any]:
    """Grab one frame from a configured camera and save it as PNG in the render sandbox.

    Opens the camera lazily when it is closed (and leaves it open: the next
    frame is then cheap). ``camera_name`` may be omitted when exactly one
    camera is configured. ``output_path`` defaults to
    ``<sandbox>/<tool>-<camera>-<unix ms>.png``; a path outside the sandbox is
    refused the way the simulation's ``render`` refuses it, naming
    ``STRANDS_ROBOTS_RENDER_ALLOW_ABS``.

    Returns ``{"camera", "path", "png", "width", "height", "channels", "opened_camera", "read_ms"}``
    - ``png`` is the encoded frame, for the tool's ``image`` content block.

    Raises:
        ValueError: An unknown camera name, no camera at all, or an unsafe path
            - each message names the remedy.
        Exception: A camera open/read failure, unchanged.
    """
    cameras = getattr(robot, "cameras", None)
    if not isinstance(cameras, Mapping) or not cameras:
        raise ValueError(format_cameras(tool_name, []))
    if camera_name is None or camera_name == "":
        if len(cameras) != 1:
            raise ValueError(
                f"camera_name is required: this arm has {len(cameras)} cameras {sorted(map(str, cameras))}."
            )
        camera_name = next(iter(cameras))
    if camera_name not in cameras:
        raise ValueError(f"Camera {camera_name!r} not found. Available: {sorted(map(str, cameras))}")
    camera = cameras[camera_name]

    opened = False
    if not getattr(camera, "is_connected", False):
        camera.connect()
        opened = True
    t0 = time.monotonic()
    frame = camera.read()
    read_ms = (time.monotonic() - t0) * 1000.0

    root = sandbox_root if sandbox_root is not None else resolve_sandbox_root(_RENDER_ROOT_ENV, "renders")
    if not output_path:
        output_path = f"{tool_name}-{camera_name}-{int(time.time() * 1000)}.png"
    safe = validate_output_path(
        str(output_path),
        sandbox_root=root,
        allow_abs=env_flag(_RENDER_ALLOW_ABS_ENV),
        allow_abs_env=_RENDER_ALLOW_ABS_ENV,
    )
    safe.parent.mkdir(parents=True, exist_ok=True)
    png = _encode_png(frame)
    safe.write_bytes(png)

    shape = tuple(getattr(frame, "shape", ()))
    return {
        "camera": str(camera_name),
        "path": str(safe),
        # The bytes the tool hands the model: a frame the agent cannot see is a
        # frame it will describe from expectation (measured: an agent handed only
        # the path described "the arm resting on the desk" for a shot of the
        # ceiling). Same block the simulation's ``render`` returns.
        "png": png,
        "height": int(shape[0]) if len(shape) > 0 else None,
        "width": int(shape[1]) if len(shape) > 1 else None,
        "channels": int(shape[2]) if len(shape) > 2 else 1,
        "opened_camera": opened,
        "read_ms": round(read_ms, 1),
    }
