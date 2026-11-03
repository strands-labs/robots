"""Which reply answers an ``invoke`` (M3): the one from THIS tenant's device, for THIS rpc and request seq."""

from __future__ import annotations

from typing import Any

from . import wire


def the_reply(replies: list[bytes], tenant: str, device_id: str, rpc: str, seq: int) -> dict[str, Any]:
    """The reply that answers THIS request (tenant, device, rpc, seq), never just ``replies[0]``. A ``stop`` or
    ``estop`` that two peers both hold answers more than once; the first matching ack is this caller's own."""
    for raw in replies:
        try:
            r = wire.decode_envelope(raw)
        except ValueError:
            continue
        if (r.get("tenant"), r.get("device_id"), r.get("name"), r.get("seq")) == (tenant, device_id, rpc, seq):
            return r
    raise LookupError(f"{device_id}.{rpc}: {len(replies)} repl(ies), none answers request seq {seq}")
