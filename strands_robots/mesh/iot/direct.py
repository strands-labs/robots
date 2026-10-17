"""Conversations over IoT (MAPPING rows 4.1, 4.2, 4.5): which path an rpc takes, and shared subscriptions.

Measured on the v1 link (EXPERIMENTS §4, n=200): Direct Message 166 ms p50, MQTT5 request/response 170 ms, so the
choice is not latency. A Direct Message with confirmation answers **404 in one HTTPS round trip (0.08 s)** when the
device is gone, which is what an e-stop wants; it needs IAM credentials (``send_direct_message`` is an HTTPS call).
MQTT5 request/response is pure MQTT: it works from a certificate-only device (device → device) and on a local broker
(Greengrass), and an absent device answers with silence, so presence (the 3 s heartbeat) is its offline signal.

Default: ``direct`` for an operator link (IAM), ``mqtt`` for a certificate link; ``iot://…?rpc=mqtt|direct``
overrides. The reply topic and request id ride in the request body either way (v0.5 fact (d)), and the device checks
the reply topic is ``strands/<tenant>/@reply/<caller>``; an MQTT5 link also sets ``response_topic`` and
``correlation_data`` so a non-strands responder can answer the same request.
"""

from __future__ import annotations

from .. import wire

MODES = ("direct", "mqtt")


def rpc_mode(query: dict[str, str]) -> str:
    """``rpc=`` from the URL query, else ``mqtt`` when the link is a certificate (no IAM), else ``direct``."""
    mode = query.get("rpc") or ("mqtt" if query.get("cert") and query.get("key") else "direct")
    if mode not in MODES:
        raise ValueError(f"iot rpc mode {mode!r} is not one of {MODES}")
    return mode


def reply_ok(reply_topic: str, tenant: str) -> bool:
    """A device answers only on its tenant's reply keys (``strands/<tenant>/@reply/<caller>``), never elsewhere."""
    parts = reply_topic.split("/")
    return len(parts) == 4 and parts[0] == "strands" and parts[1] == tenant and parts[2] == "@reply" and bool(parts[3])


def shared(group: str, topic_filter: str) -> str:
    """``$share/<group>/<filter>``: N replicas of one consumer (dashboards on ``state``) each get a share of the
    messages instead of a copy each (MQTT5 shared subscription)."""
    wire.check_id(group, "group")
    return f"$share/{group}/{topic_filter}"
