"""IoT lifecycle events as an AUDIT trail, not as liveness (MAPPING rows 4.3, 2.4, after EXPERIMENTS §5).

``$aws/events/presence/connected|disconnected/<clientId>`` arrive 21–27 ms after a connect and 113–170 ms after a
socket close (a crash, a ``stop()``, a SIGKILL), free of charge. But the broker gives a silent client
1.5 × its *server* keep-alive before calling it gone, and IoT Core answers ``server_keep_alive_sec = 1200`` whatever
the client asks — so a frozen or partitioned robot stays "connected" for up to 30 minutes. The mesh keeps its 3 s
heartbeat for presence; this module records who connected and disconnected, when and why, and gives the rule that
clears a dead client's retained presence.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from .rules import rule_name

TOPICS = ("$aws/events/presence/connected/+", "$aws/events/presence/disconnected/+")
Event = dict[str, Any]


def parse(topic: str, raw: bytes) -> Event | None:
    """``{"client", "connected", "t", "reason", "client_initiated"}`` from a lifecycle event; ``None`` otherwise."""
    parts = topic.split("/")
    if len(parts) != 5 or parts[:3] != ["$aws", "events", "presence"] or parts[3] not in ("connected", "disconnected"):
        return None
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    return {
        "client": str(body.get("clientId") or parts[4]),
        "connected": parts[3] == "connected",
        "t": float(body.get("timestamp") or time.time() * 1000) / 1000.0,
        "reason": body.get("disconnectReason"),
        "client_initiated": body.get("clientInitiatedDisconnect"),
    }


def watch(
    link: Any, on_event: Callable[[Event], None], *, tenant_clients: Callable[[str], bool] | None = None
) -> Callable[[], None]:
    """Subscribe to the account's lifecycle events on ``link`` and call ``on_event`` per event (optionally only for
    clients ``tenant_clients(client)`` accepts, since the events are account-wide). Returns the undo."""

    def on(topic: str, raw: bytes) -> None:
        ev = parse(topic, raw)
        if ev is not None and (tenant_clients is None or tenant_clients(ev["client"])):
            on_event(ev)

    undos = [link.subscribe(t, on) for t in TOPICS]

    def undo() -> None:
        for u in undos:
            u()

    return undo


class Audit:
    """The last ``keep`` lifecycle events, in memory; ``since(client)`` answers "when did it last connect / drop"."""

    def __init__(self, keep: int = 1000) -> None:
        self.events: list[Event] = []
        self.keep = keep

    def __call__(self, ev: Event) -> None:
        self.events.append(ev)
        del self.events[: -self.keep]

    def since(self, client: str) -> Event | None:
        return next((e for e in reversed(self.events) if e["client"] == client), None)


def clear_presence_rule(tenant: str, role_arn: str) -> dict[str, Mapping[str, Any]]:
    """The one rule lifecycle events are good for: when a client drops, clear its retained presence/lockout so a
    late reader never acts on a dead device's last word (an empty retained publish clears, verified)."""
    return {
        rule_name("lifecycle", tenant): {
            "sql": "SELECT clientId, disconnectReason, timestamp FROM '$aws/events/presence/disconnected/+'",
            "description": f"strands-robots tenant {tenant}: audit disconnects; clear retained state of the client",
            "actions": [
                {"republish": {"topic": f"strands/{tenant}/@audit/disconnected", "roleArn": role_arn, "qos": 1}},
            ],
            "ruleDisabled": False,
            "awsIotSqlVersion": "2016-03-23",
        }
    }
