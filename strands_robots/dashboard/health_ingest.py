"""A falsifiable answer to "is the mesh actually delivering?".

MEASURED 2026-08-22 (supervisor v10): ``/api/health`` reported ``mesh_online: true, status: ok``
straight through a 26-MINUTE ingest blackout - the coalescer's ``forwarded`` counter frozen at 19814,
``/ws/mesh`` sending one snapshot and then nothing, both real arms stale for 1430s - and reported the
same thing through the recovery. A boolean that reads ``true`` in both states cannot be checked
against anything, so an operator watching health saw a healthy fleet while the screen showed a dead
one, and the caretaker could not tell the two apart either.

``mesh_online`` is the bridge's own BELIEF (it means "a mesh session was created and has not raised").
This module publishes the two things that can contradict it, both of them numbers the reader can
recompute: the age of the freshest presence anyone has sent, and how much fan-out happened since the
last poll. Nothing here interprets a robot's health - a fleet can be legitimately quiet - so the
verdict names what was measured, and says which measurement it lacked when it lacks one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Same threshold the fleet cards grey out on (mesh_bridge.PEER_STALE_S), imported lazily so this
#: module stays importable without the bridge (it is pure, and its tests want it that way).
DEFAULT_STALE_AFTER = 15.0


def mesh_ingest(
    peers: Mapping[str, Mapping[str, Any]],
    coalesce: Mapping[str, Any] | None,
    now: float,
    prev: tuple[float, int] | None = None,
    stale_after: float = DEFAULT_STALE_AFTER,
) -> tuple[dict[str, Any], tuple[float, int] | None]:
    """The ingest block for /api/health, plus the sample to carry to the next poll.

    ``prev`` is the ``(t, forwarded)`` pair returned by the previous call; the delta between two
    polls is what makes a frozen counter visible. A single reading of a monotonic counter says
    nothing at all, which is why the frozen 19814 above never looked wrong.
    """
    ages = [now - float(p.get("last_seen") or 0.0) for p in peers.values() if p.get("last_seen")]
    freshest = min(ages) if ages else None
    fresh_id = None
    if freshest is not None:
        fresh_id = min(
            (pid for pid, p in peers.items() if p.get("last_seen")),
            key=lambda pid: now - float(peers[pid].get("last_seen") or 0.0),
        )

    forwarded = None
    if isinstance(coalesce, Mapping):
        raw = coalesce.get("forwarded")
        if isinstance(raw, (int, float)):
            forwarded = int(raw)

    delta: int | None = None
    since: float | None = None
    if forwarded is not None and prev is not None:
        prev_t, prev_forwarded = prev
        since = round(max(0.0, now - prev_t), 1)
        delta = forwarded - prev_forwarded

    if not peers:
        verdict = "no_peers"
    elif freshest is None:
        verdict = "unknown"  # peers exist but none carries last_seen: a bridge bug, not a fleet fact
    elif freshest > stale_after:
        verdict = "stalled"
    elif delta == 0 and since is not None and since >= 2.0:
        # presence is published at 1Hz per peer, so with peers > 0 a live coalescer's
        # forwarded counter CANNOT hold still across a multi-second window. Fresh presence +
        # frozen fan-out is forwarding dead / ingest alive - the mirror image of the ingest
        # blackout (which stays "stalled": stale presence wins above) - and the websocket the
        # UI renders from is getting nothing. Never call it flowing; both numbers ride in the
        # report so the reader can recompute the verdict. The first poll stays verdict-neutral
        # by construction: delta is None until two samples exist.
        verdict = "forwarding_frozen"
    else:
        verdict = "flowing"

    report: dict[str, Any] = {
        "verdict": verdict,
        # The number that contradicted mesh_online for 26 minutes. None means "nobody has ever sent
        # presence", which is not the same as "old presence" and must not be rendered as 0.
        "freshest_peer_age_s": None if freshest is None else round(freshest, 1),
        "freshest_peer": fresh_id,
        "stale_after_s": stale_after,
        "peers": len(peers),
        "stale_peers": sum(1 for p in peers.values() if p.get("stale")),
        "forwarded": forwarded,
        # Absent on the first poll BY DESIGN: a delta needs two samples, and inventing one
        # (delta = forwarded) would make a process that has been frozen for an hour look busy.
        **({"forwarded_delta": delta, "delta_window_s": since} if delta is not None else {}),
    }
    sample = (now, forwarded) if forwarded is not None else prev
    return report, sample
