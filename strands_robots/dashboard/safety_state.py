"""What the dashboard is entitled to say about an e-stop lockout."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Lockout:
    """The fleet-wide lockout as this dashboard understands it."""

    state: str = "unknown"  # locked | clear | unknown
    since: float | None = None
    by: str | None = None
    reason: str = "no e-stop or resume seen since this dashboard started"

    def as_fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {"state": self.state, "reason": self.reason}
        if self.since is not None:
            out["since"] = self.since
        if self.by:
            out["by"] = self.by
        return out


def _source_of(data: dict[str, Any]) -> str | None:
    for key in ("source", "coordinator", "peer_id", "source_peer_id", "by", "sender"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def apply_event(current: Lockout, *, kind: str, data: dict[str, Any], now: float) -> Lockout:
    """Fold one `strands/safety/**` event into the verdict."""
    t_val = data.get("t")
    when = t_val if isinstance(t_val, (int, float)) else now
    who = _source_of(data)
    if kind == "estop":
        return Lockout(
            state="locked",
            since=float(when),
            by=who,
            reason=(f"an e-stop from {who} locked the fleet" if who else "an e-stop locked the fleet"),
        )
    if kind == "resume":
        # NOT clear: every peer re-verifies the override code on its own and may refuse.
        return Lockout(
            state="unknown",
            since=float(when),
            by=who,
            reason=(
                "a resume was broadcast, but each peer verifies the override code itself - "
                "not proof that any of them cleared"
            ),
        )
    return current


def note_command_accepted(current: Lockout, *, now: float) -> Lockout:
    """A peer accepted a command a lockout would have refused: that is proof."""
    if current.state == "clear":
        return current
    return Lockout(
        state="clear",
        since=now,
        by=None,
        reason="a command this peer accepted proves its lockout is not engaged",
    )


#: Actions a locked-out peer still answers, so accepting one proves nothing.
LOCKOUT_EXEMPT_ACTIONS = frozenset({"status", "resume"})


def proves_clear(action: str) -> bool:
    """Would a locked-out peer have refused this action?"""
    return bool(action) and action not in LOCKOUT_EXEMPT_ACTIONS


def peer_lockout(fleet: Lockout, *, first_seen: float | None) -> Lockout:
    """The verdict for ONE peer, given when the dashboard first saw it.

    A peer that appeared after the e-stop is a process that never received it.
    """
    if fleet.state == "locked" and first_seen is not None and fleet.since is not None:
        if first_seen > fleet.since:
            return replace(
                fleet,
                state="unknown",
                reason=(
                    "this peer appeared after the fleet e-stop, so it may never have "
                    "received it - drive it only if you know it is safe"
                ),
            )
    return fleet


def resolve_peer(fleet: Lockout, *, first_seen: float | None = None, proof_at: float | None = None) -> Lockout:
    """The verdict shown on one peer's card."""
    verdict = peer_lockout(fleet, first_seen=first_seen)
    if proof_at is not None and (fleet.since is None or proof_at > fleet.since):
        return note_command_accepted(verdict, now=proof_at)
    return verdict
