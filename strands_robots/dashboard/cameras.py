"""Camera facts for the devices roster: requested vs streaming, per managed peer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

#: Probe verdicts, worst-to-best for the UI's sorting purposes.
STATES = ("blocked", "absent", "vanished", "assigned", "in_use", "unreadable", "ready")

_VANISHED_REASON = (
    "a camera answered at this index earlier and none does now - it was unplugged, went to sleep, or its hub dropped it"
)
_VANISHED_REMEDY = (
    "replug it (and prefer a direct port over a hub chain), then rescan. Do NOT assume the index still "
    "means the same camera: macOS renumbers cameras when one is removed, so preview an index before "
    "assigning it to an arm - a shifted index records a dataset from the wrong view while everything "
    "on screen looks healthy"
)


class CameraUnavailable(RuntimeError):
    """A camera fault that knows WHY, and what to do about it."""

    def __init__(self, index: int, state: str, reason: str, remedy: str | None = None) -> None:
        self.index = index
        self.state = state
        self.reason = reason
        self.remedy = remedy
        msg = f"camera index {index}: {reason}"
        if remedy:
            msg += f" - {remedy}"
        super().__init__(msg)


def classify_probe_stderr(text: str) -> tuple[str, str, str | None]:
    """Turn OpenCV's chatter into (state, reason, remedy)."""
    low = (text or "").lower()
    if "not authorized" in low or "not permitted" in low or "access denied" in low:
        return (
            "blocked",
            "macOS has not granted camera access to the process running the dashboard",
            "start the dashboard from Terminal, iTerm or VS Code and allow camera "
            "access when macOS asks (System Settings > Privacy & Security > Camera "
            "lists it afterwards). A dashboard started by a background daemon can "
            "never be granted: macOS refuses to even show the prompt",
        )
    if "out device of bound" in low or "out of bound" in low:
        # AVFoundation's own wording for an index past the last camera it knows: nothing is
        # there, whatever else OpenCV grumbles about the failed open afterwards.
        return ("absent", "no camera answered at this index", None)
    if "device or resource busy" in low or "busy" in low or "in use" in low:
        return (
            "in_use",
            "another program on this machine is holding the camera",
            "quit the app using it (Zoom, Photo Booth, a stray robot process), then rescan",
        )
    if "failed to properly initialize" in low or "can't open camera" in low or "cannot open" in low:
        # An index that the OS roster does not list is simply not there; an index
        # the roster DOES list, that fails this way, is present and unhappy.
        return ("unreadable", "the camera did not start streaming when opened", "unplug and replug it, then rescan")
    return ("absent", "no camera answered at this index", None)


def _geometry(entry: Mapping[str, Any] | None) -> dict[str, Any]:
    if not entry:
        return {}
    out = {k: entry[k] for k in ("width", "height", "fps") if entry.get(k)}
    return out


def roster_name(index: int, roster: Sequence[Mapping[str, Any]]) -> str | None:
    """Best-effort human name for an index, from the platform listing."""
    for item in roster:
        if item.get("listing_index") == index:
            name = item.get("name")
            return str(name) if name else None
    return None


def merge_cameras(
    *,
    probed: Iterable[Mapping[str, Any]],
    claimed: Mapping[int, str],
    roster: Sequence[Mapping[str, Any]] = (),
    remembered: Mapping[int, Mapping[str, Any]] | None = None,
    failures: Mapping[int, str] | None = None,
    streaming: Iterable[int] | None = None,
    max_index: int = 4,
) -> list[dict[str, Any]]:
    """One row per camera this machine could plausibly have, never fewer."""
    remembered = remembered or {}
    failures = failures or {}
    # None means "nobody told us what is actually streaming" - stay with the
    # older, kinder reading rather than accusing every owner of silence.
    streaming_set = None if streaming is None else {int(i) for i in streaming}
    probed_by_index = {int(c["index"]): c for c in probed if c.get("index") is not None}

    universe: set[int] = set(probed_by_index) | set(claimed) | set(failures)
    universe |= {int(r["listing_index"]) for r in roster if isinstance(r.get("listing_index"), int)}
    universe |= set(remembered)
    universe = {i for i in universe if 0 <= i}

    rows: list[dict[str, Any]] = []
    for index in sorted(universe):
        row: dict[str, Any] = {"index": index}
        name = roster_name(index, roster)
        if name:
            # `name_is_guess` is not decoration: without it a UI writes the name
            # next to the index as fact, and on this very machine that mislabelled
            # a Logitech as the wrist camera.
            row["name_hint"] = name
            row["name_is_guess"] = True

        if index in probed_by_index:
            row.update(_geometry(probed_by_index[index]))
            row["state"] = "ready"
            row["reason"] = "opened and delivered a frame just now"
        elif index in claimed:
            owner = claimed[index]
            row["claimed_by"] = owner
            if streaming_set is None or index in streaming_set:
                row["state"] = "in_use"
                row["reason"] = f"streaming for {owner}"
                row["remedy"] = f"despawn {owner} to free it"
            else:
                row["state"] = "assigned"
                row["reason"] = f"assigned to {owner}, but no frames are arriving - that robot could not open it either"
                row["remedy"] = f"check {owner}'s log (devices > logs), then respawn it"
            row.update(_geometry(remembered.get(index)))
            if _geometry(remembered.get(index)):
                row["geometry_from"] = "remembered"
        else:
            state, reason, remedy = classify_probe_stderr(failures.get(index, ""))
            if state == "absent" and index in remembered:
                state, reason, remedy = "vanished", _VANISHED_REASON, _VANISHED_REMEDY
            elif state == "absent" and index not in failures and name:
                # The roster lists it but nothing probed it (a scan capped by
                # max_index). Saying "absent" there would be a claim we did not
                # test.
                state, reason, remedy = "unknown", "not probed in this scan", "rescan to test this index"
            row["state"] = state
            row["reason"] = reason
            if remedy:
                row["remedy"] = remedy
            row.update(_geometry(remembered.get(index)))
            if _geometry(remembered.get(index)):
                row["geometry_from"] = "remembered"

        row["available"] = row["state"] == "ready"
        rows.append(row)
    return rows


def probe_needed(
    *,
    refresh: bool,
    requested_at: float,
    cache_t: float,
    ttl_s: float,
    now: float,
) -> bool:
    """Should THIS request run the camera probe, having waited for the lock?"""
    if cache_t >= requested_at:
        return False
    if refresh:
        return True
    return (now - cache_t) > ttl_s


def blocked_verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """A one-line diagnosis for the whole machine, when there is one."""
    if not rows:
        return None
    if any(r.get("state") == "ready" for r in rows):
        return None
    blocked = [r for r in rows if r.get("state") == "blocked"]
    if not blocked:
        return None
    return {
        "kind": "camera_permission",
        "message": (
            "macOS is blocking camera access for the process running the dashboard, so no camera can be opened here."
        ),
        "remedy": blocked[0].get("remedy"),
        "indices": [r["index"] for r in blocked],
    }
