"""How long to wait for a task command's answer - and what a timeout means."""

from __future__ import annotations

import math

# : Ceiling for an ack wait, seconds.
DEFAULT_ACK_CAP_S = 120.0
#: Margin over ``duration`` for a blocking ``execute``.
ROLLOUT_MARGIN_S = 10.0


def task_ack_budget(
    action: str,
    requested_timeout: float | None,
    duration: float | None,
    ack_cap_s: float = DEFAULT_ACK_CAP_S,
) -> tuple[float, str]:
    """Return ``(timeout_s, kind)`` where kind is ``"ack"`` or ``"rollout"``."""
    try:
        asked = float(requested_timeout) if requested_timeout is not None else 0.0
    except (TypeError, ValueError):
        asked = 0.0
    try:
        dur = float(duration) if duration is not None else 0.0
    except (TypeError, ValueError):
        dur = 0.0
    # NaN is rejected by name rather than by ``x != x``: both are correct, but a
    # self-comparison reads as a typo at a glance and a static analyser flags it
    # as one, so the intent is spelled out instead.
    if dur < 0 or math.isnan(dur):
        dur = 0.0
    if asked < 0 or math.isnan(asked):
        asked = 0.0

    if str(action).lower() == "execute":
        return max(asked, dur + ROLLOUT_MARGIN_S), "rollout"
    # "start" and anything unknown: wait for an ack, never for the whole run.
    floor = min(max(dur + ROLLOUT_MARGIN_S, 0.0), max(ack_cap_s, 0.0))
    return max(asked, floor), "ack"


def timeout_verdict(kind: str, timeout_s: float, target: str = "") -> dict[str, object]:
    """What to tell a caller whose task command timed out."""
    who = f" from {target}" if target else ""
    if kind == "rollout":
        return {
            "error": (
                f"no answer{who} within {timeout_s:g}s - the rollout was still running when the wait "
                "ended, so the task may be executing normally"
            ),
            "motion_possible": True,
            "timeout_kind": "rollout",
        }
    return {
        "error": (
            f"no acknowledgement{who} within {timeout_s:g}s - the command was delivered, so the robot "
            "may be loading a policy and about to move. Check its log before retrying."
        ),
        "motion_possible": True,
        "timeout_kind": "ack",
    }
