"""What a zenoh link implies for LeRobot's client settings: the LAN keeps LeRobot's defaults, the internet does not."""

from __future__ import annotations

import ipaddress
import os
from typing import Any

WAN_TIMEOUT_S = (
    30.0  # a first call across the internet wakes the model and carries two frames: LeRobot's 5-10 s are too tight
)


def is_local(endpoint: str) -> bool:
    """``<link>/host:port`` names this machine or its LAN.

    Local: loopback, the private and link-local ranges, ``localhost``, mDNS ``*.local``, and a single-label
    hostname (``gpu-box``: no TLD, so it can only resolve through ``/etc/hosts``, mDNS or a local search
    domain and cannot name an internet host as written). A dotted name (``zenoh.example``) or a public
    address is the internet. Decided from the text alone: nothing is resolved, no network is touched.
    """
    host = endpoint.split("/", 1)[-1].rpartition(":")[0].strip("[]")
    if host in ("localhost", "") or host.endswith(".local"):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host  # a hostname: single-label = LAN, dotted = somebody else's domain
    return addr.is_private or addr.is_loopback or addr.is_link_local


ROUTED = ("tls", "quic", "ws", "wss")
"""Links that reach the server through a zenoh router: every client of that router shares the session's keys."""


def routed(endpoint: str) -> bool:
    """``<link>/host:port`` goes through a router (``tls``/``quic``/``ws``), not straight to the server."""
    return endpoint.split("/", 1)[0] in ROUTED


def trusted(asked: bool) -> bool:
    """The operator vouches for the router's clients: ``trusted=True`` or ``STRANDS_ZENOH_TRUSTED=1``."""
    return asked or os.environ.get("STRANDS_ZENOH_TRUSTED", "").lower() in ("1", "true", "yes")


def check_router(who: str, endpoint: str, robot: Any, asked: bool) -> None:
    """A real robot takes actions off its own LAN only once the operator vouches for the link (``PermissionError``).

    Two off-LAN links reach the server without any identity of their own on the wire: a ``tls``/``quic``/``ws``
    link through a router (every client of which can answer for the server) and a plain link to an internet host
    (reached across the network, no server authentication, and weaker than the ``tls`` one). Both are refused for
    a real robot until the operator vouches; a loopback or LAN link, and a simulated robot, are untouched.
    """
    if not (getattr(robot, "is_real", False) and not trusted(asked)):
        return
    if routed(endpoint):
        raise PermissionError(
            f"{who}: a real robot takes actions from {endpoint} only once you vouch for every client of that zenoh "
            "router, since any of them can answer for the server: Policy(..., trusted=True) or "
            "STRANDS_ZENOH_TRUSTED=1 (docs/lerobot.md)"
        )
    if not is_local(endpoint):
        raise PermissionError(
            f"{who}: a real robot takes actions from {endpoint} only once you vouch for it, since it is reached "
            "across the network from a host with no identity on the wire: Policy(..., trusted=True) or "
            "STRANDS_ZENOH_TRUSTED=1 (docs/lerobot.md)"
        )


def wan_defaults(endpoint: str, settings: dict[str, Any], descriptor: dict[str, Any] | None = None) -> dict[str, Any]:
    """LeRobot client settings a link implies, under anything the caller set by name.

    Off the LAN: handshake/action/startup timeouts >= 30 s and JPEG frames. A ``tls``/``quic``/``ws`` link is a
    router, so ``zenoh_mode="client"``. Once the server has spoken, the action timeout also covers its own
    ``action_deadline_s`` (the hello's ``limits``) — a server allowed 5 s to answer is not a client fault at 5 s.
    """
    out: dict[str, Any] = {}
    if routed(endpoint):
        out["zenoh_mode"] = "client"
    if not is_local(endpoint):
        out.update(handshake_timeout_s=WAN_TIMEOUT_S, action_timeout_s=WAN_TIMEOUT_S, startup_timeout_s=WAN_TIMEOUT_S)
        out["encoding"] = "jpeg"
    if descriptor is not None:
        deadline = float((descriptor.get("limits") or {}).get("action_deadline_s") or 0.0)
        if deadline:
            out["action_timeout_s"] = max(
                out.get("action_timeout_s", 5.0), deadline + (0.0 if is_local(endpoint) else 5.0)
            )
    out.update(settings)  # the caller's word wins
    return out
