"""Is teleop actually working? - the verdict, from the counters."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

__all__ = ["diagnose_receiver", "envelope_refusal", "published_frames", "teleop_health"]

#: The SDK's own refusal, as it appears in the follower's log.
_RANGE_RE = re.compile(
    r"input frame value for '([^']{1,120})' out of range: \|(-?[0-9.eE+]{1,40})\| > ([0-9.eE+]{1,40})"
)
_SLEW_RE = re.compile(r"input frame slew for '([^']{1,120})' out of range: ([0-9.eE+\-]{1,40}) > ([0-9.eE+\-]{1,40})")
_SLEW_NO_TIME_RE = re.compile(
    r"input frame slew for '([^']{1,120})' out of range: moved ([0-9.eE+\-]{1,40}) units with no "
    r"elapsed time.*?\(bound ([0-9.eE+\-]{1,40}) units/s\)"
)


def envelope_refusal(log_tail: Any) -> dict[str, Any] | None:
    """The most recent value/slew refusal in a child's log, parsed. Returns ``{"joint", "value",
    "bound", "kind"}`` or ``None``.
    """
    if not isinstance(log_tail, (list, tuple)):
        return None
    for line in reversed([str(x) for x in log_tail]):
        m = _RANGE_RE.search(line)
        if m:
            try:
                return {
                    "kind": "value",
                    "joint": m.group(1),
                    "value": abs(float(m.group(2))),
                    "bound": float(m.group(3)),
                }
            except ValueError:
                continue
        m = _SLEW_RE.search(line)
        if m:
            try:
                return {"kind": "slew", "joint": m.group(1), "value": float(m.group(2)), "bound": float(m.group(3))}
            except ValueError:
                continue
        m = _SLEW_NO_TIME_RE.search(line)
        if m:
            try:
                # No elapsed time means the implied speed is unbounded, and inf is the honest value: any bound
                # is exceeded.
                return {
                    "kind": "slew",
                    "joint": m.group(1),
                    "value": float("inf"),
                    "bound": float(m.group(3)),
                    "instant": True,
                    "delta": float(m.group(2)),
                }
            except ValueError:
                continue
    return None


def diagnose_receiver(
    stats: Mapping[str, Any],
    log_tail: Any = None,
    source_frames: int | None = None,
) -> dict[str, Any]:
    """One receiver's counters -> a state and a sentence a human can act on."""
    running = bool(stats.get("running"))
    got = int(stats.get("frames_received") or 0)
    rejected = int(stats.get("rejected") or 0)
    slew = int(stats.get("slew_rejected") or 0)
    dropped = int(stats.get("rate_dropped") or 0)
    source = stats.get("source")

    if not running:
        return {"state": "stopped", "headline": "not following anything", "detail": None}

    if got == 0 and (rejected or slew):
        refusal = envelope_refusal(log_tail)
        detail = (
            f"{rejected + slew} frames arrived from {source} and every one was refused, so the follower has not moved."
        )
        out: dict[str, Any] = {
            "state": "refusing",
            "headline": "every frame is being refused",
            "detail": detail,
            "refusal": refusal,
        }
        if refusal and refusal.get("instant"):
            out["detail"] = (
                f"{detail} Two frames carried the SAME timestamp, so the follower could not compute "
                f"a speed for {refusal['joint']} and refused the jump of {refusal.get('delta', 0):g} "
                f"units outright (bound {refusal['bound']:g} units/s). That is a publisher clock "
                f"problem, not an envelope that is too tight: widening the bound cannot help, since "
                f"no interval divides into it."
            )
        elif refusal:
            out["detail"] = (
                f"{detail} The mesh's teleop envelope is {refusal['bound']:g} units, and this leader "
                f"reported {refusal['joint']} at {refusal['value']:g} - the arm reports DEGREES while "
                f"the envelope assumes radians (4·pi)."
            )
        return out

    if got == 0:
        if source_frames:
            # Both ends are healthy on their own terms and the frames still do
            # not meet. Naming that honestly is the whole point: the operator
            # must not go looking at the leader, which is working.
            return {
                "state": "unrouted",
                "headline": f"{source} is publishing, but nothing reaches this follower",
                "detail": (
                    f"the leader has published {source_frames} frames and this follower has "
                    "received none - they are not meeting on the mesh (a subscription bound to a "
                    "session that has since died does this). Restart the follower, then subscribe "
                    "again."
                ),
                "source_frames": source_frames,
            }
        return {
            "state": "silent",
            "headline": f"subscribed to {source}, but no frames are arriving",
            "detail": "the leader is not publishing - start its stream first "
            "(POST /api/robots/{leader}/teleop/publish)"
            + ("" if source_frames is None else " (its publisher reports 0 frames)"),
        }

    hz = float(stats.get("hz_actual") or 0.0)
    note = f"following {source} at {hz:.1f}Hz"
    extra = []
    if rejected or slew:
        extra.append(f"{rejected + slew} frames refused")
    if dropped:
        extra.append(f"{dropped} dropped to the rate cap")
    return {"state": "following", "headline": note, "detail": ", ".join(extra) or None}


def published_frames(status: Any, device_name: str) -> int | None:
    """How many frames a peer's ``device_name`` publisher has sent, if it says."""
    payload = _status_payload(status)
    if payload is None:
        return None
    pubs = payload.get("publishers")
    if not isinstance(pubs, Mapping):
        return None
    stats = pubs.get(device_name)
    if not isinstance(stats, Mapping):
        return None
    try:
        return int(stats.get("frames") or 0)
    except (TypeError, ValueError):
        return None


def _status_payload(status: Any) -> Mapping[str, Any] | None:
    """The counters block inside a tool-envelope status, or the block itself."""
    if not isinstance(status, Mapping):
        return None
    if "receivers" in status or "publishers" in status:
        return status
    for block in status.get("content") or []:
        if isinstance(block, Mapping) and isinstance(block.get("json"), Mapping):
            return cast("Mapping[str, Any]", block["json"])
    return None


def teleop_health(
    status: Any,
    log_tail: Any = None,
    source_frames: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Verdict for a whole peer's teleop status payload."""
    payload = _status_payload(status)
    if payload is None:
        return {"receivers": {}, "publishers": {}, "worst": None}

    receivers = {
        key: diagnose_receiver(stats, log_tail, (source_frames or {}).get(key))
        for key, stats in (payload.get("receivers") or {}).items()
        if isinstance(stats, Mapping)
    }
    publishers = {}
    for name, stats in (payload.get("publishers") or {}).items():
        if not isinstance(stats, Mapping):
            continue
        frames = int(stats.get("frames") or 0)
        hz = float(stats.get("hz_actual") or 0.0)
        target = float(stats.get("hz_target") or 0.0)
        state = "publishing" if stats.get("running") and frames else ("starting" if stats.get("running") else "stopped")
        detail = None
        # A rate far under target is not a fault: on a shared bus the state probe
        # and the camera publisher get their turns too. Say so, so nobody reads
        # it as a bug and "fixes" it by asking for more.
        if state == "publishing" and target and hz < target * 0.6:
            detail = (
                f"{hz:.1f}Hz of the {target:g}Hz requested - the servo bus is shared with "
                "this arm's state and camera reads"
            )
        publishers[name] = {"state": state, "headline": f"{frames} frames at {hz:.1f}Hz", "detail": detail}

    # The one line worth putting on a card: a refusal outranks everything.
    order = {"refusing": 0, "unrouted": 1, "silent": 2, "stopped": 3, "following": 4}
    worst = None
    for key, verdict in sorted(receivers.items(), key=lambda kv: order.get(kv[1]["state"], 9)):
        worst = {"peer_key": key, **verdict}
        break
    return {"receivers": receivers, "publishers": publishers, "worst": worst}
