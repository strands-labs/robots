"""What the registry knows that the mesh cannot (MAPPING row 7.1): devices provisioned for a tenant but silent now.

A dashboard draws them as grey cards. Read-only; every failure is a status word, never an exception, and
``STRANDS_IOT_REGISTRY=off`` turns it off entirely.
"""

from __future__ import annotations

import os
from typing import Any

from .policy import group_name


def list_things(tenant: str, *, region: str, iot: Any = None, limit: int = 250) -> list[dict[str, Any]]:
    """``[{id, device_type, tenant, connected, status}]`` for every thing in the tenant's group. ``connected`` is
    ``True``/``False`` when fleet indexing with connectivity is on, else ``None``; ``status`` is ``"ok"``,
    ``"off"``, ``"unavailable"`` or ``"denied"``."""
    if os.environ.get("STRANDS_IOT_REGISTRY", "").lower() in ("0", "off", "false", "no"):
        return [{"id": "", "device_type": "", "tenant": tenant, "connected": None, "status": "off"}]
    try:
        if iot is None:
            import boto3

            iot = boto3.client("iot", region_name=region)
        names: list[str] = []
        token: str | None = None
        while len(names) < limit:
            kw = {"nextToken": token} if token else {}
            page = iot.list_things_in_thing_group(thingGroupName=group_name(tenant), maxResults=min(limit, 250), **kw)
            names += page.get("things", [])
            token = page.get("nextToken")
            if not token:
                break
        out = []
        for name in names[:limit]:
            attrs = iot.describe_thing(thingName=name).get("attributes", {})
            out.append({"id": name, "device_type": attrs.get("device_type", ""), "tenant": tenant,
                        "connected": _connected(iot, name), "status": "ok"})  # fmt: skip
        return out
    except Exception as exc:  # noqa: BLE001 — the registry is advisory; the word says why it is empty
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
        status = "denied" if "Denied" in code or "Unauthorized" in code else "unavailable"
        return [{"id": "", "device_type": "", "tenant": tenant, "connected": None, "status": status}]


def _connected(iot: Any, name: str) -> bool | None:
    try:
        hits = iot.search_index(queryString=f"thingName:{name}").get("things", [])
        return bool(hits[0].get("connectivity", {}).get("connected")) if hits else None
    except Exception:  # noqa: BLE001 — fleet indexing is optional
        return None
