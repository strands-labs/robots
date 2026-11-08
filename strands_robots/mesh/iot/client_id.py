"""The MQTT client id of an ``iot://`` link: a device names its own; an operator gets one its policy admits."""

from __future__ import annotations

import uuid

from .policy import operator_client_prefix


def default_client_id(query: dict[str, str], tenant: str | None = None) -> str:
    """A certificate link is a device (its policy pins the thing name, pass ``client=``). An IAM link is an operator:
    ``strands-op-<tenant>-<random>``, the only ids ``operator_policy`` lets it connect with."""
    if query.get("cert"):
        return f"strands-{uuid.uuid4().hex[:12]}"
    if tenant is None:
        return f"strands-op-{uuid.uuid4().hex[:12]}"
    return f"{operator_client_prefix(tenant)}{uuid.uuid4().hex[:12]}"


def link_client_id(q: dict[str, str], tenant: str | None) -> str:
    """The MQTT client id; an operator ``client=`` outside its tenant's prefix is refused here, not by the broker."""
    client = q.get("client") or default_client_id(q, tenant)
    if q.get("cert") or tenant is None:
        return client
    prefix = operator_client_prefix(tenant)
    if not client.startswith(prefix):
        raise ValueError(f"client={client!r}: an operator (IAM) link of tenant {tenant!r} connects as {prefix}<name>")
    return client
