"""reachy_reads - the Reachy Mini's read-only verbs, off the driver's caches and catalogue.

Two ``@tool`` verbs, the read half of the port from ``cagataycali/tiny-the-reachy``
(the write half is :mod:`~strands_robots.tools.reachy.reachy_actions`). Same
contract: one call on a live :class:`~strands_robots.drivers.reachy.ReachyDriver`
handle, envelope back verbatim, live-handle refusals via
:func:`~strands_robots.tools.reachy._reachy_common.live_handle_refusal`.
Both are safe to call at any time - neither moves the robot.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.reachy._reachy_common import live_handle_refusal

# verb -> (accessor, reads, expected), the same table shape reachy_actions uses.
_READS: dict[str, tuple[str, str, str]] = {
    "reachy_get_state": (
        "state_snapshot",
        "the verb reads the joint, pose, IMU and battery caches the driver's link thread fills",
        "a callable ``state_snapshot()`` returning the driver's envelope - "
        "pass the live ReachyDriver handle the orchestrator constructed",
    ),
    "reachy_list_emotions": (
        "list_moves",
        "the verb asks the daemon for a recorded-move library's catalogue through the driver's REST path",
        "a callable ``list_moves(library)`` returning the driver's envelope - "
        "pass the live ReachyDriver handle the orchestrator constructed",
    ),
}


def _handle_refusal(verb: str, driver: Any) -> dict[str, Any] | None:
    """The shared live-handle judgement, worded from the ``_READS`` row."""
    accessor, reads, expected = _READS[verb]
    return live_handle_refusal(verb, driver, accessor=accessor, reads=reads, expected=expected)


@tool
def reachy_get_state(driver: Any) -> dict[str, Any]:
    """Read the Mini's live state: joints, head pose, IMU, battery.

    Calls ``ReachyDriver.state_snapshot()`` once - a cache read, no daemon
    round-trip, so call it freely before and after motion to verify. A
    ``None`` field means that stream has not delivered yet; IMU and battery
    stay ``None`` on a Lite, which has neither.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_get_state", driver)
    if refusal is not None:
        return refusal
    return driver.state_snapshot()


@tool
def reachy_list_emotions(driver: Any, library: str = "emotions") -> dict[str, Any]:
    """List the recorded moves the Mini's daemon can play.

    Calls ``ReachyDriver.list_moves(library)`` once. The names it returns are
    what :func:`~strands_robots.tools.reachy.reachy_actions.reachy_express`
    accepts.

    Args:
        driver: The live ReachyDriver handle the orchestrator constructed.
        library: ``'emotions'`` (default) or ``'dances'``.

    Returns:
        The driver's envelope, success or refusal, unreshaped.
    """
    refusal = _handle_refusal("reachy_list_emotions", driver)
    if refusal is not None:
        return refusal
    return driver.list_moves(library)
