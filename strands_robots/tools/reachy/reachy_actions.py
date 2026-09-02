"""reachy_actions - the Reachy Mini's driver-gated execution verbs, one table-driven module.

Twelve ``@tool`` verbs ported from ``cagataycali/tiny-the-reachy`` in the
consolidated shape the g1 family proved (refs #3037, #3070): each makes exactly
one call on a live :class:`~strands_robots.drivers.reachy.ReachyDriver` handle
and returns the driver's envelope verbatim. The driver owns the safety
judgement - the motion envelope, the finite-number gates and the connected
check all live on its write path - so a verb never re-validates what the
driver already refuses.

Every verb holds the package's four ``@tool`` invariants (envelope not
exception; names the verb; names the parameter; names the received type) via
:func:`~strands_robots.tools.reachy._reachy_common.live_handle_refusal`, worded
per verb from the ``_ACTIONS`` table. The ``driver`` parameter is typed
:class:`~typing.Any` so no live-object type leaks into the generated tool
schema.

Four verbs (``reachy_play_sound``, ``reachy_volume``, ``reachy_camera``,
``reachy_look_at``) name driver accessors that do not exist yet - the Mini's
media rail is SDK-client-side and has no daemon REST path in this repository -
so today they refuse by naming the accessor to plumb, exactly as
``g1_move_velocity`` did before ``G1Driver.move_velocity`` landed.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.reachy._reachy_common import live_handle_refusal
from strands_robots.utils import boolean_flag_error


def _refusal(text: str) -> dict[str, Any]:
    """Wrap ``text`` in the ``{"status": "error"}`` envelope every verb owes."""
    return {"status": "error", "content": [{"text": text}]}


# verb -> (accessor, reads, expected) for live_handle_refusal. One row per verb
# because the refusal prose names the remedy, which differs per accessor.
_ACTIONS: dict[str, tuple[str, str, str]] = {
    "reachy_look": (
        "send_action",
        "the verb sends one whole head-pose command on the driver's real-time "
        "link and reads back the envelope the driver produced",
        "a callable ``send_action(action)`` returning the driver's write "
        "envelope - pass the live ReachyDriver handle the orchestrator "
        "constructed",
    ),
    "reachy_antennas": (
        "send_action",
        "the verb sends one antenna command on the driver's real-time link "
        "and reads back the envelope the driver produced",
        "a callable ``send_action(action)`` returning the driver's write "
        "envelope - pass the live ReachyDriver handle the orchestrator "
        "constructed",
    ),
    "reachy_body_turn": (
        "send_action",
        "the verb sends one body-yaw command on the driver's real-time link "
        "and reads back the envelope the driver produced",
        "a callable ``send_action(action)`` returning the driver's write "
        "envelope - pass the live ReachyDriver handle the orchestrator "
        "constructed",
    ),
    "reachy_home": (
        "send_action",
        "the verb sends the neutral whole-pose command on the driver's "
        "real-time link and reads back the envelope the driver produced",
        "a callable ``send_action(action)`` returning the driver's write "
        "envelope - pass the live ReachyDriver handle the orchestrator "
        "constructed",
    ),
    "reachy_stop": (
        "stop_task",
        "the verb asks the daemon to halt any recorded move in progress "
        "through the driver's own stop path and reads back the envelope the "
        "driver produced",
        "a callable ``stop_task()`` returning the driver's envelope - pass "
        "the live ReachyDriver handle the orchestrator constructed",
    ),
    "reachy_wake": (
        "wake_up",
        "the verb plays the daemon's built-in wake-up or go-to-sleep move "
        "through the driver's recorded-move path and reads back the envelope "
        "the driver produced",
        "callables ``wake_up()`` and ``goto_sleep()`` returning the driver's "
        "envelope - pass the live ReachyDriver handle the orchestrator "
        "constructed",
    ),
    "reachy_express": (
        "play_move",
        "the verb plays one recorded emotion or dance through the driver's "
        "recorded-move path and reads back the envelope the driver produced",
        "a callable ``play_move(move_name, library)`` returning the driver's "
        "envelope - pass the live ReachyDriver handle the orchestrator "
        "constructed",
    ),
    "reachy_motors": (
        "set_motors",
        "the verb sets motor torque on the driver's real-time link and reads back the envelope the driver produced",
        "a callable ``set_motors(mode)`` returning the driver's envelope - "
        "pass the live ReachyDriver handle the orchestrator constructed",
    ),
    "reachy_play_sound": (
        "play_sound",
        "the verb plays one sound file through the driver's media path and reads back the envelope the driver produced",
        "a callable ``play_sound(sound_file)`` returning the driver's "
        "envelope - the Mini's media rail is not plumbed into ReachyDriver "
        "yet, so this names the accessor to add",
    ),
    "reachy_volume": (
        "set_volume",
        "the verb sets the speaker volume through the driver's media path "
        "and reads back the envelope the driver produced",
        "a callable ``set_volume(level)`` returning the driver's envelope - "
        "the Mini's media rail is not plumbed into ReachyDriver yet, so this "
        "names the accessor to add",
    ),
    "reachy_camera": (
        "capture_frame",
        "the verb captures one frame from the head camera through the "
        "driver's media path and reads back the envelope the driver produced",
        "a callable ``capture_frame(save_path)`` returning the driver's "
        "envelope - the Mini's media rail is not plumbed into ReachyDriver "
        "yet, so this names the accessor to add",
    ),
    "reachy_look_at": (
        "look_at_image",
        "the verb servos the head toward a camera pixel through the driver's "
        "media path and reads back the envelope the driver produced",
        "a callable ``look_at_image(u, v)`` returning the driver's envelope - "
        "the Mini's media rail is not plumbed into ReachyDriver yet, so this "
        "names the accessor to add",
    ),
}


def _handle_refusal(verb: str, driver: Any, accessor: str | None = None) -> dict[str, Any] | None:
    """The shared live-handle judgement, worded from the ``_ACTIONS`` row.

    Args:
        verb: The ``_ACTIONS`` key.
        driver: The handle to judge.
        accessor: Override for a verb that picks its accessor at call time
            (``reachy_wake`` reads ``wake_up`` or ``goto_sleep``).

    Returns:
        ``None`` or the refusal envelope.
    """
    row_accessor, reads, expected = _ACTIONS[verb]
    return live_handle_refusal(verb, driver, accessor=accessor or row_accessor, reads=reads, expected=expected)


@tool
def reachy_look(
    driver: Any,
    pitch: float = 0.0,
    roll: float = 0.0,
    yaw: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    body_yaw: float | None = None,
    antenna_left: float | None = None,
    antenna_right: float | None = None,
) -> dict[str, Any]:
    """Move the Mini's head to a pose - the primary gesture verb.

    Calls ``ReachyDriver.send_action(...)`` once with a whole head pose: the
    daemon's head command is a pose, not a delta, so an absent axis means zero
    (level), not "leave as it was". The driver refuses non-finite values and any
    axis outside the motion envelope (pitch/roll +/-40 deg, yaw +/-180 deg,
    body +/-160 deg). It also refuses a head-body yaw twist beyond 65 deg, but
    only when ``body_yaw`` is sent in the same call: a ``yaw`` beyond 65 deg on
    its own is reached by the body turning under the head, so it moves the whole
    robot rather than asking for a twist. Pass ``body_yaw`` to say where the body
    should end up instead, and the pair is refused if the two differ by more
    than the limit, because the daemon would otherwise override the body yaw.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.
        pitch: Head pitch, degrees (positive = up).
        roll: Head roll, degrees.
        yaw: Head yaw, degrees (positive = left).
        x: Head translation forward, millimetres (small, ~[-20, 20]).
        y: Head translation left, millimetres.
        z: Head translation up, millimetres.
        body_yaw: Body rotation, degrees; ``None`` lets the daemon turn the body
            as far as the head yaw needs, and no further.
        antenna_left: Left antenna angle, degrees; ``None`` leaves it alone.
        antenna_right: Right antenna angle, degrees; ``None`` leaves it alone.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_look", driver)
    if refusal is not None:
        return refusal
    action: dict[str, Any] = {
        "head_pitch": pitch,
        "head_roll": roll,
        "head_yaw": yaw,
        "head_x": x,
        "head_y": y,
        "head_z": z,
    }
    if body_yaw is not None:
        action["body_yaw"] = body_yaw
    if antenna_left is not None:
        action["antenna_left"] = antenna_left
    if antenna_right is not None:
        action["antenna_right"] = antenna_right
    return driver.send_action(action)


@tool
def reachy_antennas(driver: Any, right: float = 0.0, left: float = 0.0) -> dict[str, Any]:
    """Move just the two antennas (the Mini's 'ears'), in degrees.

    Calls ``ReachyDriver.send_action(...)`` once with only the antenna keys,
    so head and body stay where they are. Both up reads alert, both down
    reads sad, asymmetric reads quizzical.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.
        right: Right antenna angle, degrees (~[-90, 90]).
        left: Left antenna angle, degrees (~[-90, 90]).

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_antennas", driver)
    if refusal is not None:
        return refusal
    return driver.send_action({"antenna_right": right, "antenna_left": left})


@tool
def reachy_body_turn(driver: Any, yaw: float = 0.0) -> dict[str, Any]:
    """Rotate the Mini's body around the vertical axis, in degrees.

    Calls ``ReachyDriver.send_action(...)`` once with only ``body_yaw``; the
    driver refuses values outside +/-160 deg. Use it to turn toward a speaker or
    scan the room while the head stays put - which is also what bounds it: the
    head pose is the daemon's primary task, so the body turns no further than
    65 deg from the head's own yaw target, and a turn past that is refused
    rather than served in part. To turn the whole robot further, send both values
    through ``reachy_look`` so the head comes round with the body.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.
        yaw: Body yaw, degrees, envelope +/-160, and within 65 deg of the head's
            current yaw target.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_body_turn", driver)
    if refusal is not None:
        return refusal
    return driver.send_action({"body_yaw": yaw})


@tool
def reachy_home(driver: Any) -> dict[str, Any]:
    """Return the Mini to the neutral pose: head centred, body forward, ears rest.

    Calls ``ReachyDriver.send_action(...)`` once with every axis at zero.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_home", driver)
    if refusal is not None:
        return refusal
    return driver.send_action(
        {
            "head_pitch": 0.0,
            "head_roll": 0.0,
            "head_yaw": 0.0,
            "head_x": 0.0,
            "head_y": 0.0,
            "head_z": 0.0,
            "body_yaw": 0.0,
            "antenna_left": 0.0,
            "antenna_right": 0.0,
        }
    )


@tool
def reachy_stop(driver: Any) -> dict[str, Any]:
    """Halt any recorded move the daemon is playing.

    Calls ``ReachyDriver.stop_task()`` once - the Mini's closest thing to a
    task is a recorded move, and this is its stop.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_stop", driver)
    if refusal is not None:
        return refusal
    return driver.stop_task()


@tool
def reachy_wake(driver: Any, sleep: bool = False) -> dict[str, Any]:
    """Wake the Mini up (init pose, ears up) or put it to sleep.

    Calls ``ReachyDriver.wake_up()`` or ``ReachyDriver.goto_sleep()`` once -
    the daemon's two built-in recorded moves.

    ``sleep`` selects which of two physical motions is commanded, so it is
    checked against the shared
    :func:`~strands_robots.utils.boolean_flag_error` domain rather than read by
    truthiness - the domain
    :func:`~strands_robots.tools.lerobot_teleoperate.build_lerobot_command`
    already applies to its own flags. Every non-empty string is truthy, so
    ``'false'``, ``'no'`` and ``'0'`` - the spellings a caller reaches for to
    opt *out* - would otherwise command go-to-sleep and report success.

    The check precedes the accessor because the accessor is derived from the
    flag: a misread ``sleep`` also decides which accessor the handle gate
    requires to be callable, so it would refuse a handle that can wake but not
    sleep for a request to wake.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.
        sleep: ``True`` plays go-to-sleep instead of wake-up. Checked, not
            parsed - a truthy spelling of off is refused, never honoured.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    if error := boolean_flag_error(sleep, "sleep", "reachy_wake"):
        return _refusal(error)
    accessor = "goto_sleep" if sleep else "wake_up"
    refusal = _handle_refusal("reachy_wake", driver, accessor=accessor)
    if refusal is not None:
        return refusal
    return getattr(driver, accessor)()


@tool
def reachy_express(driver: Any, emotion: str = "", library: str = "emotions") -> dict[str, Any]:
    """Play a named emotion or dance from the Mini's recorded-move library.

    Calls ``ReachyDriver.play_move(emotion, library)`` once. This is the
    Mini's personality rail - a full head+antenna+body choreography served by
    the daemon; :func:`~strands_robots.tools.reachy.reachy_reads.reachy_list_emotions`
    names the live catalogue.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.
        emotion: The move's name, e.g. ``'happy'``, ``'curious'``, ``'no'``.
        library: ``'emotions'`` (default) or ``'dances'``.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_express", driver)
    if refusal is not None:
        return refusal
    if not emotion:
        return _refusal(
            "reachy_express: `emotion` is required. Pass a recorded move's name "
            "like 'happy' or 'curious' - reachy_list_emotions names the live catalogue."
        )
    return driver.play_move(emotion, library)


@tool
def reachy_motors(driver: Any, mode: str = "") -> dict[str, Any]:
    """Set the Mini's motor torque: ``'enabled'`` holds pose, ``'disabled'`` goes limp.

    Calls ``ReachyDriver.set_motors(mode)`` once. Disabled is the safe-to-
    move-by-hand state; the driver refuses any other word by naming the
    admitted set.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.
        mode: ``'enabled'`` or ``'disabled'``.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_motors", driver)
    if refusal is not None:
        return refusal
    if not mode:
        return _refusal("reachy_motors: `mode` is required. Pass 'enabled' (torque on) or 'disabled' (limp).")
    return driver.set_motors(mode)


@tool
def reachy_play_sound(driver: Any, sound_file: str = "") -> dict[str, Any]:
    """Play a sound file through the Mini's speaker.

    Calls ``ReachyDriver.play_sound(sound_file)`` once. The media rail is not
    plumbed into the driver yet, so today this refuses by naming the accessor
    to add - the same shape ``g1_move_velocity`` had before its driver method
    landed.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.
        sound_file: Path or builtin name of the audio file the daemon can read.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_play_sound", driver)
    if refusal is not None:
        return refusal
    if not sound_file:
        return _refusal("reachy_play_sound: `sound_file` is required. Pass a path or builtin name like 'wake_up.wav'.")
    return driver.play_sound(sound_file)


@tool
def reachy_volume(driver: Any, level: int | None = None) -> dict[str, Any]:
    """Set the Mini's speaker volume, 0-100.

    Calls ``ReachyDriver.set_volume(level)`` once. The media rail is not
    plumbed into the driver yet, so today this refuses by naming the accessor
    to add.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.
        level: Volume, an integer from 0 to 100.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_volume", driver)
    if refusal is not None:
        return refusal
    if level is None:
        return _refusal("reachy_volume: `level` is required. Pass an integer from 0 to 100.")
    if isinstance(level, bool) or not isinstance(level, int):
        return _refusal(f"reachy_volume: `level` must be an integer 0-100, got {type(level).__name__!r}.")
    if not 0 <= level <= 100:
        return _refusal(f"reachy_volume: `level` must be within 0-100, got {level}.")
    return driver.set_volume(level)


@tool
def reachy_camera(driver: Any, save_path: str = "") -> dict[str, Any]:
    """Capture a frame from the Mini's head camera and save it to disk.

    Calls ``ReachyDriver.capture_frame(save_path)`` once. The media rail is
    not plumbed into the driver yet, so today this refuses by naming the
    accessor to add.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.
        save_path: Where to write the JPEG; empty means the driver's default.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_camera", driver)
    if refusal is not None:
        return refusal
    return driver.capture_frame(save_path)


@tool
def reachy_look_at(driver: Any, u: int | None = None, v: int | None = None) -> dict[str, Any]:
    """Servo the Mini's head toward pixel ``(u, v)`` in its camera frame.

    Calls ``ReachyDriver.look_at_image(u, v)`` once. The media rail is not
    plumbed into the driver yet, so today this refuses by naming the accessor
    to add.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.
        u: Pixel column in the camera frame.
        v: Pixel row in the camera frame.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_look_at", driver)
    if refusal is not None:
        return refusal
    for name, value in (("u", u), ("v", v)):
        if value is None:
            return _refusal(f"reachy_look_at: `{name}` is required. Pass the pixel coordinate in the camera frame.")
        if isinstance(value, bool) or not isinstance(value, int):
            return _refusal(f"reachy_look_at: `{name}` must be an integer pixel, got {type(value).__name__!r}.")
    return driver.look_at_image(u, v)
