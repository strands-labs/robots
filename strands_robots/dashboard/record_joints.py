"""Refuse a recording whose arms cannot report where they are."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# : A snapshot older than this says nothing about the arm's present state.
MAX_AGE_S = 30.0


def _joint_count(peer: Any) -> int | None:
    """How many joints this peer's snapshot reports, or None when there is nothing to read."""
    if not isinstance(peer, Mapping):
        return None
    state = peer.get("state")
    if not isinstance(state, Mapping):
        return None
    joints = state.get("joints")
    if joints is None:
        return 0
    if isinstance(joints, Mapping):
        return len(joints)
    if isinstance(joints, (list, tuple)):
        return len(joints)
    return None  # a shape we do not understand is not evidence of absence


def _age_s(peer: Mapping, now: float) -> float | None:
    for key in ("last_seen",):
        try:
            seen = float(peer.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if seen > 0:
            return max(0.0, now - seen)
    return None


def refusal(
    *,
    role: str,
    peer_id: str,
    peer: Any,
    problem: Any = None,
    now: float,
    max_age_s: float = MAX_AGE_S,
) -> str | None:
    """Why this arm cannot be recorded from, or None to proceed."""
    count = _joint_count(peer)
    if count is None or count > 0:
        return None
    age = _age_s(peer, now)  # type: ignore[arg-type]
    if age is None or age > max_age_s:
        # Either we cannot date the reading or it is old. Both mean this is not evidence about now.
        return None
    where = f"{role} '{peer_id}'"
    detail = ""
    if isinstance(problem, Mapping):
        headline = str(problem.get("headline") or "").strip()
        remedy = str(problem.get("remedy") or "").strip()
        detail = " ".join(p for p in (f"{headline}." if headline else "", remedy) if p).strip()
    if not detail:
        detail = (
            "Open this robot's log (devices > logs): the joint probe records the exception it raised there in full."
        )
    return (
        f"{where} is on the mesh but reports NO joint positions (reading is {age:.0f}s old), so a "
        f"recording would capture an arm whose position cannot be read - the episodes would carry no "
        f"{'actions' if role == 'leader' else 'observations'} to learn from. {detail}"
    )
