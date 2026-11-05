"""Which reply answers an ``invoke`` (M3): the one from THIS tenant's device, for THIS rpc and request seq."""

from __future__ import annotations

from typing import Any

from . import wire
from .transports.base import RemoteError

SHARED = ("stop", "estop")
"""Rpcs every holder of a device answers (two peers that both hold it both stop it): their acks may differ."""


def the_reply(replies: list[bytes], tenant: str, device_id: str, rpc: str, seq: int) -> dict[str, Any]:
    """The reply that answers THIS request (tenant, device, rpc, seq), never just ``replies[0]``. A ``stop`` or
    ``estop`` that two peers both hold answers more than once; the first matching ack is this caller's own. Any
    other rpc answered twice, differently, has a second answerer on the mesh that is not the device: the result is
    refused rather than guessed (``RemoteError``)."""
    found: list[dict[str, Any]] = []
    for raw in replies:
        try:
            r = wire.decode_envelope(raw)
        except ValueError:
            continue
        if (r.get("tenant"), r.get("device_id"), r.get("name"), r.get("seq")) == (tenant, device_id, rpc, seq):
            if rpc in SHARED:
                return r
            found.append(r)
    if not found:
        raise LookupError(f"{device_id}.{rpc}: {len(replies)} repl(ies), none answers request seq {seq}")
    if any(_body(r) != _body(found[0]) for r in found[1:]):
        raise RemoteError(
            f"{device_id}.{rpc}: {len(found)} different answers to request seq {seq}; another peer answers for "
            "this device, so none is trusted (secure the mesh: docs/mesh.md, Securing the mesh)"
        )
    return found[0]


def _body(r: dict[str, Any]) -> Any:
    return (r.get("kind"), r.get("result"), r.get("error"))
