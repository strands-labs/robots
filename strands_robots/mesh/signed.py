"""Envelopes on a transport with a mesh identity: sealed on the way out, checked on the way in.

A transport carries an identity (``transport.identity``) when its peers joined over mutual TLS
(``transports.zenoh_tls``); every other transport (``mem://``, IoT, an open zenoh mesh) has none, and envelopes travel
on it as they always did. On a transport with one:

* a request is answered only when it is signed, its ``peer`` is one of the signer's own peer ids, and it is fresh
  (``identity.Freshness``); an e-stop is answered whatever its signature says, a stop must always land;
* a reply, a manifest, a state sample counts only when it is signed by a certificate that serves its device id, and
  a reply only when it is addressed to the caller's own peer id.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any

from . import wire
from .identity import Freshness, Unverified
from .reply import the_reply
from .transports.base import RemoteError, Transport

logger = logging.getLogger(__name__)


def identity_of(transport: Transport) -> Any:
    return getattr(transport, "identity", None)


def peer_id(transports: Sequence[Transport]) -> str:
    """A Mesh's peer id: ``<name>.<random>`` when its transports carry one identity, a random id when none does."""
    names = {i.name for i in map(identity_of, transports) if i is not None}
    if len(names) > 1:
        raise ValueError(f"one Mesh speaks as one mesh identity; its transports carry {sorted(names)}")
    token = uuid.uuid4().hex
    return f"{names.pop()}.{token[:16]}" if names else token


def encode(transport: Transport, env: dict[str, Any]) -> bytes:
    """``env`` as bytes for ``transport``: signed when it carries an identity."""
    ident = identity_of(transport)
    return wire.encode(env if ident is None else ident.seal(env))


def check_mount(transport: Transport, device_id: str) -> None:
    ident = identity_of(transport)
    if ident is not None and not ident.signer.may_serve(device_id):
        raise ValueError(
            f"mesh identity {ident.name!r} may not answer for {device_id!r}: issue it a certificate with "
            f"`python -m strands_robots.mesh cert {ident.name} --serves {device_id}`"
        )


def check_request(transport: Transport, env: dict[str, Any], fresh: Freshness | None, rpc: str) -> str | None:
    """The signer's name of a request on ``transport`` (None on a transport without identities); raises
    ``Unverified`` for a request that does not prove its ``peer``."""
    ident = identity_of(transport)
    if ident is None:
        return None
    try:
        signer = ident.verify(env)
        if not signer.owns(env.get("peer")):
            raise Unverified(f"{signer.name!r} signed a request for peer {env.get('peer')!r}, which is not its own")
        if fresh is not None and rpc != "estop":
            fresh.admit(env["peer"], env["seq"], env["t"])
    except Unverified:
        if rpc == "estop":
            return None
        raise
    return str(signer.name)


def from_server(transport: Transport, env: dict[str, Any], device_id: str) -> bool:
    """Whether ``env`` (a reply, manifest or state) comes from a peer whose certificate serves ``device_id``."""
    ident = identity_of(transport)
    if ident is None:
        return True
    try:
        return bool(ident.verify(env).may_serve(device_id)) and env.get("device_id") == device_id
    except Unverified:
        return False


def replies_for(transport: Transport, replies: list[bytes], device_id: str, peer: str) -> list[bytes]:
    """The replies that may answer ``peer``'s request to ``device_id``: on a transport with identities, only those
    signed by a certificate that serves it and addressed to ``peer``; the rest are another peer's, never a result."""
    if identity_of(transport) is None:
        return replies
    kept = []
    for raw in replies:
        try:
            env = wire.decode_envelope(raw)
        except ValueError:
            continue
        if env.get("peer") == peer and from_server(transport, env, device_id):
            kept.append(raw)
    if replies and not kept:
        raise RemoteError(
            f"{device_id}: {len(replies)} repl(ies), none signed by a certificate that serves {device_id!r}; "
            "the peer answering is not the device (docs/mesh.md, Securing the mesh)"
        )
    if len(kept) < len(replies):
        logger.warning("mesh: dropped %d unsigned or foreign repl(ies) for %s", len(replies) - len(kept), device_id)
    return kept


def answer(
    transport: Transport, replies: list[bytes], request: tuple[str, str, str, int], peer: str, timeout: float
) -> dict[str, Any]:
    """The reply that answers ``request`` (tenant, device id, rpc, seq) of ``peer``, among the replies that may
    (``replies_for``); ``TimeoutError`` when none came within ``timeout``."""
    tenant, device_id, rpc, seq = request
    kept = replies_for(transport, replies, device_id, peer)
    if not kept:
        raise TimeoutError(f"{device_id}.{rpc}: no reply within {timeout} s")
    return the_reply(kept, tenant, device_id, rpc, seq)
