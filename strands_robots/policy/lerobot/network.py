"""Near or far: whether a session's link leaves the LAN, by the caller's word or by a measured round trip.

The endpoint's text alone cannot tell: a tunnel (``ssh -L``, a TCP proxy) to a GPU box in another region is
``127.0.0.1`` too. This picks timeouts and frame encoding only; who may answer is ``link.check_channel``'s call.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

from .link import is_local

NETWORKS = ("auto", "lan", "wan")

FAR_RTT_S = 0.020
"""A measured round trip above this is off the LAN (wired LAN ~1 ms, busy Wi-Fi < 20 ms, another region >= 20 ms)."""


def round_trip(client: Any, deployment: str, n: int = 3) -> float | None:
    """The median of ``n`` DESCRIBE round trips on an open ``RemoteClient``, in s; None when one fails.

    The query the handshake already made, answered by the server's queryable: no session, a few ms on this machine.
    """
    from lerobot.remote_inference.codec import encode_message
    from lerobot.remote_inference.protocol import deployment_prefix
    from lerobot.transport.protocol import Envelope, MessageType

    key, times = deployment_prefix(deployment) + "/describe", []
    for _ in range(n):
        t0 = time.monotonic()
        try:
            replies = client.transport.query(key, encode_message(Envelope(MessageType.DESCRIBE)), 2.0)
        except Exception:  # noqa: BLE001  (a probe that cannot complete says the link is slow, not a fault)
            return None
        if not replies:
            return None
        times.append(time.monotonic() - t0)
    return statistics.median(times)


def is_far(endpoint: str, asked: str, rtt: Callable[[], float | None]) -> tuple[bool, float | None]:
    """Whether the session's link is off the LAN, and the round trip measured to decide it (None: not measured).

    ``asked``: ``"lan"``/``"wan"`` is the caller's word. ``"auto"``: an endpoint written off the LAN is far; one
    written local is measured (``rtt``), since a tunnel (``ssh -L``, a proxy) to a far machine is 127.0.0.1 too.
    A probe that fails counts as far: wider timeouts and JPEG frames never make a link less safe.
    """
    if asked != "auto":
        return asked == "wan", None
    if not is_local(endpoint):
        return True, None
    measured = rtt()
    return measured is None or measured > FAR_RTT_S, measured
