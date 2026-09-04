from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from strands_robots.utils import boolean_flag_error

__all__ = ["MOTION_ENV", "GATED_ACTIONS", "agent_motion_allowed", "peer_is_physical"]

#: The grant. Set on the dashboard's process (or via the consent screen) to let the agent start
#: physical tasks by itself.
MOTION_ENV = "STRANDS_DASH_AGENT_PHYSICAL_MOTION"

#: Actions that can put a real robot in motion. Everything else -- including every way of STOPPING
#: one -- is deliberately outside this set.
GATED_ACTIONS: frozenset[str] = frozenset({"task"})

_TRUE = ("1", "true", "yes", "on")


def _granted(env: Mapping[str, str] | None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(MOTION_ENV, "")).strip().lower() in _TRUE


def peer_is_physical(peer: Mapping[str, Any] | None) -> tuple[bool, str]:
    """Is this peer metal? Returns (physical, why) -- the server-side twin of lib/runRisk.ts."""
    if not peer:
        return True, "this peer is not on the fleet snapshot, so it cannot be shown to be a sim"
    presence = peer.get("presence") or {}
    hw = presence.get("hw")
    if isinstance(hw, str) and hw.strip():
        return True, f"it reports real hardware ({hw.strip()})"
    robot_type = str(presence.get("robot_type") or "").strip().lower()
    if robot_type in ("sim", "simulation", "mujoco"):
        return False, f"it reports itself as {robot_type}"
    if presence.get("sim") is True or presence.get("mode") == "sim":
        return False, "it reports itself as a simulation"
    if not presence:
        return True, "this peer has announced no presence yet, so it cannot be shown to be a sim"
    return True, "it did not say it was a simulation"


def agent_motion_allowed(
    action: str,
    *,
    peer: Mapping[str, Any] | None = None,
    target: str = "",
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """May the agent perform ``action`` on ``target`` by itself? Returns ``{"allowed": bool,
    "physical": bool, "reason": str}``.
    """
    act = (action or "").strip()
    if act not in GATED_ACTIONS:
        return {"allowed": True, "physical": False, "reason": "", "gated": False}

    physical, why = peer_is_physical(peer)
    if not physical:
        return {"allowed": True, "physical": False, "reason": "", "gated": True}
    if _granted(env):
        return {
            "allowed": True,
            "physical": True,
            "reason": "",
            "gated": True,
            "granted": True,
        }

    shown = target.strip() or "that robot"
    return {
        "allowed": False,
        "physical": True,
        "gated": True,
        "granted": False,
        "reason": (
            f"refused: starting a task on {shown} would MOVE REAL HARDWARE ({why}), and this "
            f"dashboard does not let the agent start physical motion on its own. Nothing was sent. "
            f"The human can press play on {shown}'s card, which confirms the motion and checks that "
            f"the policy fits that robot, or ask for a simulated peer instead. To let the agent do "
            f"it unattended, grant it once: set {MOTION_ENV}=1 for the dashboard. Stopping robots "
            f"is never gated - 'everyone stop' always works."
        ),
    }


# --- the OTHER half of the same asymmetry: the HTTP route
# ------------------------------------------ agent_motion_allowed() guards the in-process
# fleet tool.
TASK_CONFIRM_ENV = "STRANDS_DASH_TASK_REQUIRES_CONFIRM"


def task_confirm_required(env: Mapping[str, str] | None = None) -> bool:
    """Has the operator asked for real-motion task POSTs to carry a confirmation?"""
    env = env if env is not None else os.environ
    return str(env.get(TASK_CONFIRM_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def task_post_allowed(
    *,
    peer: Mapping[str, Any] | None,
    confirmed: bool,
    target: str = "",
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verdict for one task POST. Same shape as ``agent_motion_allowed``.

    Off by default, and never in the way of a simulated peer or of an
    already-confirmed click. ``confirmed`` selects a posture, so on the paths
    that read it, it must be a boolean
    (:func:`~strands_robots.utils.boolean_flag_error`) rather than any truthy
    value: a request whose confirmation is a string is refused, not honoured.
    """
    if not task_confirm_required(env):
        return {"allowed": True, "physical": False, "reason": "", "gated": False}
    if text := boolean_flag_error(confirmed, "confirmed", "task_post_allowed"):
        # Checked on the shared domain rather than read by truthiness, and here
        # rather than at the top: this is the branch that reads it, so a
        # dashboard which never asked for a confirmation is not refused for a
        # field it does not consult. Every non-empty string is truthy, so
        # `"confirmed": "false"` in a task POST body would otherwise select the
        # confirmed posture -- real motion starting with the requirement the
        # operator turned on satisfied by a value that reads as a refusal.
        return {
            "allowed": False,
            "physical": True,
            "gated": True,
            "confirmed": False,
            "reason": (
                f"refused: {text} Nothing was sent. Send a JSON boolean "
                f'(`"confirmed": true`), or press play on that robot\'s card, '
                f"where the browser confirms."
            ),
        }
    if confirmed:
        return {"allowed": True, "physical": True, "reason": "", "gated": True, "confirmed": True}

    physical, why = peer_is_physical(peer)
    if not physical:
        return {"allowed": True, "physical": False, "reason": "", "gated": True}

    shown = target.strip() or "that robot"
    return {
        "allowed": False,
        "physical": True,
        "gated": True,
        "confirmed": False,
        "reason": (
            f"refused: this dashboard is set to require a confirmation before a task starts real "
            f"motion, and this request did not carry one ({shown}: {why}). Nothing was sent. Press play "
            f"on {shown}'s card - the browser confirms there - or, for a script you trust, send "
            f'"confirmed": true in the body. Turn the requirement off by clearing '
            f"{TASK_CONFIRM_ENV}. Stopping is never gated."
        ),
    }
