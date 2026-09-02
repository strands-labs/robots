"""g1_actions - the G1's driver-gated execution verbs, one table-driven module.

Thirteen ``@tool`` verbs ported from ``cagataycali/neon-the-g1`` (refs #358)
that each make exactly one call on a live
:class:`~strands_robots.drivers.g1.G1Driver` handle and return the driver's
envelope verbatim. They are a different rail from
:mod:`~strands_robots.tools.g1.use_unitree`: the dispatcher reaches the raw
SDK clients, while these verbs reach the *driver's* SDK-facing write path so
its motion gates, rc decoding and refusal shapes apply.

Every verb holds the package's four ``@tool`` invariants (envelope not
exception; names the verb; names the parameter; names the received type) via
two shared implementations: :func:`~strands_robots.tools.g1._g1_common.live_handle_refusal`
for the ``driver`` handle (the ``_ACTIONS`` table carries each verb's accessor
and refusal prose) and the shared numeric validators from
:mod:`strands_robots.utils` for the data parameters. The ``driver`` parameter
is typed :class:`~typing.Any` on every verb to keep this module out of the
driver's import cycle (see the package docstring's SDK-load-hygiene note);
importing this module pulls no ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal
from strands_robots.utils import finite_number_error, positive_finite_number_error


def _refusal(text: str) -> dict[str, Any]:
    """Wrap ``text`` in the ``{"status": "error"}`` envelope every verb owes."""
    return {"status": "error", "content": [{"text": text}]}


# verb -> (accessor, reads, expected) for live_handle_refusal. One row per
# verb because the refusal prose names the remedy, which differs per accessor.
_ACTIONS: dict[str, tuple[str, str, str]] = {
    "g1_arm_action": (
        "arm_action",
        "the verb publishes one arm-gesture ExecuteAction call on rt/armsdk "
        "through the driver's own SDK-facing write path and reads back the "
        "envelope the driver produced",
        "a callable ``arm_action(action, action_id)`` returning the driver's "
        "write envelope - pass the live G1Driver handle the orchestrator "
        "constructed",
    ),
    "g1_balance_stand": (
        "balance_stand",
        "the verb requests a G1 BalanceStand transition through the driver's "
        "own SDK-facing write path and reads back the envelope the driver "
        "produced",
        "a callable ``balance_stand(balance_mode)`` returning the driver's "
        "write envelope - pass the live G1Driver handle the orchestrator "
        "constructed",
    ),
    "g1_move_velocity": (
        "move_velocity",
        "the verb requests a G1 velocity command through the driver's own "
        "SDK-facing write path and reads back the envelope the driver produced",
        "a callable ``move_velocity(vx, vy, vyaw, duration)`` returning the "
        "driver's write envelope - pass the live G1Driver handle the "
        "orchestrator constructed",
    ),
    "g1_release_arm": (
        "release_arm",
        "the verb requests a G1 arm-release write through the driver's own "
        "SDK-facing write path and reads back the envelope the driver produced",
        "a callable ``release_arm()`` returning the driver's write envelope - "
        "pass the live G1Driver handle the orchestrator constructed",
    ),
    "g1_safe_lie_to_stand": (
        "safe_lie_to_stand",
        "the verb requests a G1 Damp -> Lie2StandUp compound transition "
        "through the driver's own SDK-facing write path and reads back the "
        "envelope the driver produced",
        "a callable ``safe_lie_to_stand(preamble_s)`` returning the driver's "
        "write envelope - pass the live G1Driver handle the orchestrator "
        "constructed",
    ),
    "g1_safe_squat_to_stand": (
        "safe_squat_to_stand",
        "the verb requests a G1 Damp -> Squat2StandUp compound transition "
        "through the driver's own SDK-facing write path and reads back the "
        "envelope the driver produced",
        "a callable ``safe_squat_to_stand(preamble_s)`` returning the driver's "
        "write envelope - pass the live G1Driver handle the orchestrator "
        "constructed",
    ),
    "g1_safe_stand_to_squat": (
        "safe_stand_to_squat",
        "the verb requests a G1 Damp -> StandUp2Squat compound transition "
        "through the driver's own SDK-facing write path and reads back the "
        "envelope the driver produced",
        "a callable ``safe_stand_to_squat(preamble_s)`` returning the driver's "
        "write envelope - pass the live G1Driver handle the orchestrator "
        "constructed",
    ),
    "g1_set_fsm": (
        "set_fsm",
        "the verb requests a G1 FSM transition through the driver's own "
        "SDK-facing write path and reads back the fsm-before / fsm-after "
        "round-trip the driver produced",
        "a callable ``set_fsm(fsm_id, wait=...)`` returning the driver's "
        "write envelope - pass the live G1Driver handle the orchestrator "
        "constructed",
    ),
    "g1_set_stand_height": (
        "set_stand_height",
        "the verb requests a G1 stand-height transition through the driver's "
        "own SDK-facing write path and reads back the envelope the driver "
        "produced",
        "a callable ``set_stand_height(height)`` returning the driver's write "
        "envelope - pass the live G1Driver handle the orchestrator constructed",
    ),
    "g1_set_swing_height": (
        "set_swing_height",
        "the verb requests a G1 swing-height (walking leg-lift clearance) "
        "transition through the driver's own SDK-facing write path and reads "
        "back the envelope the driver produced",
        "a callable ``set_swing_height(height)`` returning the driver's write "
        "envelope - pass the live G1Driver handle the orchestrator constructed",
    ),
    "g1_shake_hand_loco": (
        "shake_hand_loco",
        "the verb dispatches the G1's built-in LocoClient.ShakeHand task "
        "through the driver's own SDK-facing write path and reads back the "
        "envelope the driver produced",
        "a callable ``shake_hand_loco(stage)`` returning the driver's write "
        "envelope - pass the live G1Driver handle the orchestrator constructed",
    ),
    "g1_stop_move": (
        "stop_move",
        "the verb requests a G1 locomotion halt through the driver's own "
        "SDK-facing write path and reads back the envelope the driver produced",
        "a callable ``stop_move()`` returning the driver's write envelope - "
        "pass the live G1Driver handle the orchestrator constructed",
    ),
    "g1_wave_hand_loco": (
        "wave_hand_loco",
        "the verb dispatches the G1's built-in LocoClient.WaveHand task "
        "through the driver's own SDK-facing write path and reads back the "
        "envelope the driver produced",
        "a callable ``wave_hand_loco(turn_flag)`` returning the driver's "
        "write envelope - pass the live G1Driver handle the orchestrator "
        "constructed",
    ),
}


def _handle_refusal(verb: str, driver: Any) -> dict[str, Any] | None:
    """The shared live-handle judgement, worded from the ``_ACTIONS`` row."""
    accessor, reads, expected = _ACTIONS[verb]
    return live_handle_refusal(verb, driver, accessor=accessor, reads=reads, expected=expected)


def _required_int_error(verb: str, param: str, value: Any, noun: str, required: str, see: str) -> str | None:
    """Refuse a missing / bool / non-int ``param``, or return ``None``.

    ``bool`` is refused before ``int`` because it *is* an ``int`` subclass
    and would coerce silently.
    """
    if value is None:
        return f"{verb}: `{param}` is required. {required} - see {see}."
    if isinstance(value, bool):
        return f"{verb}: `{param}` must be an {noun} (int, not bool), got {type(value).__name__!r}. See {see}."
    if not isinstance(value, int):
        return f"{verb}: `{param}` must be an {noun} (int), got {type(value).__name__!r}. See {see}."
    return None


@tool
def g1_arm_action(driver: Any, action: str = "", action_id: int | None = None) -> dict[str, Any]:
    """Execute one built-in G1 arm gesture (clap, heart, shake hand, ...).

    Calls ``G1Driver.arm_action(action, action_id)`` once - the driver
    resolves the name or numeric id, publishes the SDK's ExecuteAction call
    on ``rt/armsdk`` and produces the envelope this verb returns verbatim.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        action: A gesture name like ``'clap'`` or ``'heart'`` - the full
            name-to-id map is on :func:`~strands_robots.tools.g1.g1_arm_actions.g1_list_arm_actions`.
        action_id: The SDK's numeric id (like ``17`` or ``20``), an
            alternative to ``action``. Pass one of the two.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("g1_arm_action", driver)
    if refusal is not None:
        return refusal
    if not action and action_id is None:
        return _refusal(
            "g1_arm_action: pass one of `action` (a gesture name like 'clap' or "
            "'heart') or `action_id` (the SDK's numeric id like 17 or 20). The "
            "full name-to-id map is on g1_list_arm_actions (refs "
            "strands-labs/robots#2959)."
        )
    return driver.arm_action(action, action_id)


@tool
def g1_balance_stand(driver: Any, balance_mode: int | None = None) -> dict[str, Any]:
    """Put the G1 into BalanceStand in the given balance mode.

    Calls ``G1Driver.balance_stand(balance_mode)`` once and returns the
    driver's envelope verbatim. Neon-bundle-observed admitted set ``{0, 3}``:
    ``0`` = static stand, ``3`` = dynamic balance.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        balance_mode: Integer mode id; ``use_unitree``'s ``describe_operation``
            documents the SDK side.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("g1_balance_stand", driver)
    if refusal is not None:
        return refusal
    error = _required_int_error(
        "g1_balance_stand",
        "balance_mode",
        balance_mode,
        "integer mode id",
        "Pass an integer mode id (neon-bundle-observed admitted set {0, 3}; 0 = static, 3 = dynamic)",
        "use_unitree's describe_operation for the SDK side (refs strands-labs/robots#358)",
    )
    if error is not None:
        return _refusal(error)
    return driver.balance_stand(balance_mode)


@tool
def g1_move_velocity(
    driver: Any,
    vx: float | None = None,
    vy: float | None = None,
    vyaw: float | None = None,
    duration: float | None = None,
) -> dict[str, Any]:
    """Command the G1 to WALK at ``(vx, vy, vyaw)`` for ``duration`` seconds.

    Calls ``G1Driver.move_velocity(vx, vy, vyaw, duration)`` once (the SDK's
    ``SetVelocity``; requires FSM in the walk set) and returns the driver's
    envelope verbatim. The walk window closes after ``duration`` seconds
    unless :func:`g1_stop_move` halts it earlier.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        vx: Forward velocity, m/s (signed finite float).
        vy: Lateral velocity, m/s (positive = strafe left).
        vyaw: Rotation rate, rad/s (positive = counter-clockwise).
        duration: Seconds to hold the triple (positive finite float).

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("g1_move_velocity", driver)
    if refusal is not None:
        return refusal
    for value, name in ((vx, "vx"), (vy, "vy"), (vyaw, "vyaw")):
        error = finite_number_error(value, name, "g1_move_velocity")
        if error is not None:
            return _refusal(error)
    error = positive_finite_number_error(duration, "duration", "g1_move_velocity")
    if error is not None:
        return _refusal(error)
    return driver.move_velocity(vx, vy, vyaw, duration)


@tool
def g1_release_arm(driver: Any) -> dict[str, Any]:
    """Force-release the G1 arm's holding action (ExecuteAction id 99).

    Calls ``G1Driver.release_arm()`` once and returns the driver's envelope
    verbatim.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("g1_release_arm", driver)
    if refusal is not None:
        return refusal
    return driver.release_arm()


def _safe_posture(verb: str, driver: Any, preamble_s: float) -> dict[str, Any]:
    """The shared body of the three Damp-preamble posture transitions."""
    refusal = _handle_refusal(verb, driver)
    if refusal is not None:
        return refusal
    error = positive_finite_number_error(preamble_s, "preamble_s", verb)
    if error is not None:
        return _refusal(error)
    accessor = _ACTIONS[verb][0]
    return getattr(driver, accessor)(preamble_s)


@tool
def g1_safe_lie_to_stand(driver: Any, preamble_s: float = 0.5) -> dict[str, Any]:
    """Stand the G1 up from lying: Damp, wait ``preamble_s``, Lie2StandUp.

    Calls ``G1Driver.safe_lie_to_stand(preamble_s)`` once and returns the
    driver's envelope verbatim.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        preamble_s: Seconds to hold Damp before the transition (positive
            finite float).

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    return _safe_posture("g1_safe_lie_to_stand", driver, preamble_s)


@tool
def g1_safe_squat_to_stand(driver: Any, preamble_s: float = 0.5) -> dict[str, Any]:
    """Stand the G1 up from a squat: Damp, wait ``preamble_s``, Squat2StandUp.

    Calls ``G1Driver.safe_squat_to_stand(preamble_s)`` once and returns the
    driver's envelope verbatim.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        preamble_s: Seconds to hold Damp before the transition (positive
            finite float).

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    return _safe_posture("g1_safe_squat_to_stand", driver, preamble_s)


@tool
def g1_safe_stand_to_squat(driver: Any, preamble_s: float = 0.5) -> dict[str, Any]:
    """Lower the G1 into a squat: Damp, wait ``preamble_s``, StandUp2Squat.

    Calls ``G1Driver.safe_stand_to_squat(preamble_s)`` once and returns the
    driver's envelope verbatim.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        preamble_s: Seconds to hold Damp before the transition (positive
            finite float).

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    return _safe_posture("g1_safe_stand_to_squat", driver, preamble_s)


@tool
def g1_set_fsm(driver: Any, fsm_id: int | None = None, wait: float = 3.0) -> dict[str, Any]:
    """Request a G1 FSM transition and read back the settled state.

    Calls ``G1Driver.set_fsm(fsm_id, wait=wait)`` once and returns the
    driver's fsm-before / fsm-after envelope verbatim.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        fsm_id: Integer id from the SDK's SetFsmId admission set -
            ``use_unitree``'s ``describe_operation`` documents it.
        wait: Seconds between SetFsmId and the fsm-after read (positive
            finite float).

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("g1_set_fsm", driver)
    if refusal is not None:
        return refusal
    if fsm_id is None:
        return _refusal(
            "g1_set_fsm: `fsm_id` is required. Pass an integer id from the "
            "SDK's SetFsmId admission set - see strands_robots.tools.g1 for "
            "the dispatcher that documents it (refs strands-labs/robots#358)."
        )
    if isinstance(fsm_id, bool) or not isinstance(fsm_id, int):
        return _refusal(
            f"g1_set_fsm: `fsm_id` of type {type(fsm_id).__name__!r} is not an "
            "int. Pass an integer id from the SDK's SetFsmId admission set - "
            "see strands_robots.tools.g1 for the dispatcher that documents it "
            "(refs strands-labs/robots#358)."
        )
    error = positive_finite_number_error(wait, "wait", "the seconds to wait between SetFsmId and the fsm-after read")
    if error is not None:
        return _refusal(f"g1_set_fsm: {error}")
    return driver.set_fsm(fsm_id, wait=wait)


def _set_height(verb: str, driver: Any, height: float | None, required: str) -> dict[str, Any]:
    """The shared body of the two height-setting verbs."""
    refusal = _handle_refusal(verb, driver)
    if refusal is not None:
        return refusal
    if height is None:
        return _refusal(f"{verb}: `height` is required. {required} (refs strands-labs/robots#358).")
    error = finite_number_error(height, "height", verb)
    if error is not None:
        return _refusal(error)
    accessor = _ACTIONS[verb][0]
    return getattr(driver, accessor)(height)


@tool
def g1_set_stand_height(driver: Any, height: float | None = None) -> dict[str, Any]:
    """Set the G1's standing height.

    Calls ``G1Driver.set_stand_height(height)`` once and returns the driver's
    envelope verbatim.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        height: Finite float in meters (typical range 0.0..~0.8; 0.0 = LOW /
            crouched, negative = HighStand fallback).

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    return _set_height(
        "g1_set_stand_height",
        driver,
        height,
        "Pass a finite float in meters (typical range 0.0..~0.8; 0.0 = LOW / "
        "crouched, negative = HighStand fallback) - see "
        "G1Driver.set_stand_height for the shape",
    )


@tool
def g1_set_swing_height(driver: Any, height: float | None = None) -> dict[str, Any]:
    """Set the G1's walking leg-lift (swing) clearance.

    Calls ``G1Driver.set_swing_height(height)`` once and returns the driver's
    envelope verbatim.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        height: Finite float in meters (neon-bundle-observed range 0.0..0.2;
            typical safe range 0.05..0.15).

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    return _set_height(
        "g1_set_swing_height",
        driver,
        height,
        "Pass a finite float in meters (neon-bundle-observed range 0.0..0.2; "
        "typical safe range 0.05..0.15) - see use_unitree's "
        "describe_operation for the SDK side",
    )


@tool
def g1_shake_hand_loco(driver: Any, stage: int | None = None) -> dict[str, Any]:
    """Dispatch the G1's built-in LocoClient.ShakeHand task.

    Calls ``G1Driver.shake_hand_loco(stage)`` once and returns the driver's
    envelope verbatim. SDK-observed admitted set ``{-1, 0, 1}``: ``-1`` =
    toggle the SDK's internal counter, ``0`` = reach out, ``1`` = shake.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        stage: Integer stage id from the admitted set.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("g1_shake_hand_loco", driver)
    if refusal is not None:
        return refusal
    error = _required_int_error(
        "g1_shake_hand_loco",
        "stage",
        stage,
        "integer stage id",
        "Pass an integer stage id (SDK-observed admitted set {-1, 0, 1}; -1 = "
        "toggle SDK's internal counter, 0 = reach out, 1 = shake)",
        "use_unitree's describe_operation for the SDK side (refs strands-labs/robots#358)",
    )
    if error is not None:
        return _refusal(error)
    return driver.shake_hand_loco(stage)


@tool
def g1_stop_move(driver: Any) -> dict[str, Any]:
    """Stop all G1 locomotion (zero velocity triple, FSM unchanged).

    Calls ``G1Driver.stop_move()`` once (the SDK's ``StopMove``) and returns
    the driver's envelope verbatim - the halt half of the
    :func:`g1_move_velocity` pair.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("g1_stop_move", driver)
    if refusal is not None:
        return refusal
    return driver.stop_move()


@tool
def g1_wave_hand_loco(driver: Any, turn_flag: bool | None = None) -> dict[str, Any]:
    """Dispatch the G1's built-in LocoClient.WaveHand task.

    Calls ``G1Driver.wave_hand_loco(turn_flag)`` once and returns the
    driver's envelope verbatim.

    Args:
        driver: The live G1Driver handle the orchestrator constructed.
        turn_flag: ``False`` = wave in place, ``True`` = wave and turn around.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("g1_wave_hand_loco", driver)
    if refusal is not None:
        return refusal
    if turn_flag is None:
        return _refusal(
            "g1_wave_hand_loco: `turn_flag` is required. Pass a bool (False = "
            "wave in place, True = wave and turn around) - use_unitree's "
            "describe_operation documents the SDK side (refs "
            "strands-labs/robots#358)."
        )
    if not isinstance(turn_flag, bool):
        return _refusal(
            "g1_wave_hand_loco: `turn_flag` must be a bool (True = wave and "
            "turn around, False = wave in place), got "
            f"{type(turn_flag).__name__!r}. use_unitree's describe_operation "
            "documents the SDK side (refs strands-labs/robots#358)."
        )
    return driver.wave_hand_loco(turn_flag)
