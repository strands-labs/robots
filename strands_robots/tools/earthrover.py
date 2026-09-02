"""EarthRover agent verbs - the rover's whole tool surface in one module.

Six ``@tool`` verbs over ``EarthRoverDriver`` (#3081), ported from
``cagataycali/scout-the-rover`` where the transport ran against the real rover.
One module, one accessor table, no per-verb files - the shape the g1
consolidation (#3037/#3070) locked for tool families.

Every verb holds the package's ``@tool`` invariants: the answer is an envelope
and never an exception, the ``driver`` parameter is judged by the shared
:func:`_handle_refusal` before anything else, and the driver's own envelope is
returned verbatim rather than reshaped - the driver already words its refusals
(not connected, HTTP status, dead link) better than a wrapper could.

The ``driver`` handle is a **live Python object** the orchestrator constructed
(typed :class:`~typing.Any` so no import cycle and no type leaks into the tool
schema); an agent cannot synthesize it, and a wrong handle is refused naming
the parameter and the received type. That independence is why the driver is
named above as a literal rather than a ``:class:`` role: a role is a resolvable
target, and this module must import and grade cleanly in a tree where #3081 has
not landed yet.
"""

from __future__ import annotations

import time
from typing import Any

from strands import tool

from strands_robots.utils import boolean_flag_error, positive_finite_number_error

#: The longest twist a single ``rover_move`` call may hold before the forced
#: stop. A tool call is a conversation turn, not a control loop: anything
#: longer belongs to repeated calls, where the agent sees telemetry between
#: legs instead of committing the rover to half a minute blind.
MAX_MOVE_DURATION_S = 30.0

#: verb -> (accessor, expected) for :func:`_handle_refusal`. One row per verb,
#: so adding a verb is adding a row plus its ``@tool`` - never a new module.
_VERBS: dict[str, tuple[str, str]] = {
    "rover_move": ("move", "a twist write (`move`); pass the live EarthRoverDriver handle"),
    "rover_stop": ("stop_task", "a halt (`stop_task`); pass the live EarthRoverDriver handle"),
    "rover_lamp": ("send_action", "a command write (`send_action`); pass the live EarthRoverDriver handle"),
    "rover_state": ("read_state", "a telemetry read (`read_state`); pass the live EarthRoverDriver handle"),
    "rover_camera": ("capture_frame", "a camera read (`capture_frame`); pass the live EarthRoverDriver handle"),
    "rover_speak": ("speak", "a speaker write (`speak`); pass the live EarthRoverDriver handle"),
}


def _refusal(text: str) -> dict[str, Any]:
    """One refusal envelope, so every refusal has the same shape."""
    return {"status": "error", "content": [{"text": text}]}


def _handle_refusal(verb: str, driver: Any) -> dict[str, Any] | None:
    """Judge the live ``driver`` handle, worded from the ``_VERBS`` row.

    Args:
        verb: Which verb is asking, so the message names it.
        driver: The handle to judge - ``None`` and any object without the
            verb's accessor are refused rather than coerced.

    Returns:
        ``None`` when ``driver`` exposes a callable accessor, otherwise the
        refusal envelope every ``@tool`` owes a caller instead of an exception.
    """
    accessor, expected = _VERBS[verb]
    if driver is None:
        return _refusal(
            f"{verb}: `driver` is required. Pass the live EarthRoverDriver handle the "
            "orchestrator constructed - an agent cannot synthesize it, because it holds "
            "the open HTTP session to the earth-rovers-sdk."
        )
    if not callable(getattr(driver, accessor, None)):
        return _refusal(f"{verb}: `driver` of type {type(driver).__name__!r} does not expose {expected}")
    return None


@tool
def rover_move(
    driver: Any,
    linear: float = 0.0,
    angular: float = 0.0,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Drive the rover: one twist, optionally held for a bounded time then stopped.

    With ``duration_s`` unset this sends one twist and returns - the rover
    keeps rolling until the next command, which is the SDK's own contract.
    With ``duration_s`` set the twist is held for that long and a zero twist
    is sent afterwards, and the envelope reports **both** halves: a move whose
    trailing stop failed is not a completed move (the rover is still rolling),
    so that outcome comes back as an error naming the stop.

    Args:
        driver: The live EarthRoverDriver handle the orchestrator constructed.
        linear: Forward speed, ``-1.0`` to ``1.0`` (clamped by the driver);
            negative is reverse.
        angular: Turn rate, ``-1.0`` to ``1.0`` (clamped by the driver);
            positive is left.
        duration_s: How long to hold the twist before the forced stop, in
            seconds - at most :data:`MAX_MOVE_DURATION_S`. ``None`` sends the
            twist and returns immediately.

    Returns:
        The driver's envelope for the twist; for a timed move, a combined
        envelope reporting the move and the trailing stop.
    """
    refusal = _handle_refusal("rover_move", driver)
    if refusal is not None:
        return refusal
    if duration_s is not None:
        if error := positive_finite_number_error(duration_s, "duration_s", "rover_move"):
            return _refusal(error)
        if duration_s > MAX_MOVE_DURATION_S:
            return _refusal(
                f"rover_move: duration_s must be at most {MAX_MOVE_DURATION_S}s, got {duration_s}. "
                "A longer run belongs to repeated calls, where telemetry is read between legs."
            )
    moved = driver.move(linear, angular)
    if duration_s is None or moved["status"] != "success":
        return moved
    time.sleep(duration_s)
    stopped = driver.stop_task()
    outcome = {
        "commanded": moved["content"][0]["json"].get("commanded"),
        "held_s": duration_s,
        "stopped": stopped["status"] == "success",
    }
    if stopped["status"] != "success":
        return {
            "status": "error",
            "content": [
                {
                    "text": "rover_move: the twist was sent but the trailing stop did not reach the SDK - the rover may still be rolling"
                },
                {"json": outcome},
            ],
        }
    return {"status": "success", "content": [{"json": outcome}]}


@tool
def rover_stop(driver: Any) -> dict[str, Any]:
    """Halt the rover - a zero twist through the driver.

    Args:
        driver: The live EarthRoverDriver handle the orchestrator constructed.

    Returns:
        The driver's ``stop_task`` envelope: success only if the zero twist
        actually reached the SDK, never merely a cleared flag.
    """
    refusal = _handle_refusal("rover_stop", driver)
    if refusal is not None:
        return refusal
    return driver.stop_task()


@tool
def rover_lamp(driver: Any, on: bool = True) -> dict[str, Any]:
    """Switch the headlamp - and stop, because the lamp rides the twist frame.

    The SDK carries ``lamp`` inside the one ``/control`` command, so a lamp
    write is a twist write: this verb sends ``{linear: 0, angular: 0, lamp}``,
    exactly what the proven scout transport sent. A rover that must keep
    moving with the lamp on is driven with ``rover_move`` calls after this.

    Args:
        driver: The live EarthRoverDriver handle the orchestrator constructed.
        on: ``True`` for lamp on, ``False`` for off.

    Returns:
        The driver's envelope for the zero-twist-plus-lamp command.
    """
    refusal = _handle_refusal("rover_lamp", driver)
    if refusal is not None:
        return refusal
    if error := boolean_flag_error(on, "on", "rover_lamp"):
        return _refusal(error)
    return driver.send_action({"linear": 0.0, "angular": 0.0, "lamp": on})


@tool
def rover_state(driver: Any) -> dict[str, Any]:
    """Read the rover's telemetry: battery, GPS, orientation, signal, lamp.

    Polls the SDK's ``/data`` fresh and falls back to the driver's cached
    snapshot when the poll fails, so the answer is the last truth the rover
    told rather than an exception.

    Args:
        driver: The live EarthRoverDriver handle the orchestrator constructed.

    Returns:
        A success envelope with a one-line human summary and the full
        telemetry JSON, or a refusal when no telemetry has ever arrived.
    """
    refusal = _handle_refusal("rover_state", driver)
    if refusal is not None:
        return refusal
    data = driver.read_state()
    if not data:
        return _refusal(
            "rover_state: no telemetry yet - the driver has never heard the SDK answer /data. "
            "Is the earth-rovers-sdk running and the rover connected to it?"
        )
    gps_ok = data.get("gps_signal", 0) and data.get("latitude", 1000) != 1000
    gps = f"{data['latitude']:.6f}, {data['longitude']:.6f}" if gps_ok else "no fix"
    summary = (
        f"battery {data.get('battery', '?')}% | signal {data.get('signal_level', '?')}/4 | "
        f"heading {data.get('orientation', '?')} deg | speed {data.get('speed', '?')} | "
        f"lamp {'on' if data.get('lamp') else 'off'} | GPS {gps}"
    )
    return {"status": "success", "content": [{"text": summary}, {"json": data}]}


@tool
def rover_camera(driver: Any, camera: str = "front") -> dict[str, Any]:
    """Grab one camera frame from the rover.

    Args:
        driver: The live EarthRoverDriver handle the orchestrator constructed.
        camera: Which view, ``"front"`` or ``"rear"``.

    Returns:
        The driver's envelope with the frame as an image content block the
        agent can see, or the driver's refusal (unknown view, video session
        down, unreachable SDK) verbatim.
    """
    refusal = _handle_refusal("rover_camera", driver)
    if refusal is not None:
        return refusal
    result = driver.capture_frame(camera)
    if result["status"] != "success":
        return result
    import base64  # noqa: PLC0415 - stdlib, used only on this path

    payload = result["content"][0]["json"]
    return {
        "status": "success",
        "content": [
            {"text": f"[{payload['camera']}]"},
            {"image": {"format": payload["format"], "source": {"bytes": base64.b64decode(payload["b64"])}}},
        ],
    }


@tool
def rover_speak(driver: Any, text: str = "") -> dict[str, Any]:
    """Say ``text`` through the rover's speaker.

    Args:
        driver: The live EarthRoverDriver handle the orchestrator constructed.
        text: What to say; must be non-empty.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("rover_speak", driver)
    if refusal is not None:
        return refusal
    return driver.speak(text)
