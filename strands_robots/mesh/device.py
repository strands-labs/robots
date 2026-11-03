"""A device as the mesh sees it, and the selector that picks devices out.

Selector grammar: ``device(*)`` or ``device(<key>:<value>[,<key>:<value>…])``. Keys are the device's labels plus
the reserved ``id``, ``type`` and ``tenant``; every pair must match. A bare id (``"so101-a"``) means
``device(id:so101-a)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_GRAMMAR = "device(<key>:<value>[,<key>:<value>...]) or device(*)"
_WRAP = re.compile(r"^device\((.*)\)$")
_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]+$")


@dataclass(frozen=True)
class Device:
    """A discovered device: who it is, how it is labelled, what it can do (its manifest), when it was last heard."""

    id: str
    device_type: str
    tenant: str
    labels: dict[str, str] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    last_seen: float = 0.0

    def __repr__(self) -> str:
        return f"Device({self.id!r}, type={self.device_type!r}, tenant={self.tenant!r}, labels={self.labels})"

    def facts(self) -> dict[str, str]:
        """Everything a selector can match on."""
        return {**self.labels, "id": self.id, "type": self.device_type, "tenant": self.tenant}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "device_type": self.device_type,
            "tenant": self.tenant,
            "labels": dict(self.labels),
            "manifest": self.manifest,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Device:
        return cls(
            id=d["id"],
            device_type=d.get("device_type", ""),
            tenant=d.get("tenant", ""),
            labels={str(k): str(v) for k, v in (d.get("labels") or {}).items()},
            manifest=d.get("manifest") or {},
            last_seen=float(d.get("last_seen", 0.0)),
        )


@dataclass(frozen=True)
class Selector:
    """Pairs that must all match; empty means ``device(*)``."""

    pairs: tuple[tuple[str, str], ...] = ()

    @classmethod
    def parse(cls, text: str) -> Selector:
        text = (text or "").strip()
        if _TOKEN.match(text) and not text.startswith("device("):
            return cls((("id", text),))
        m = _WRAP.match(text)
        if not m:
            raise ValueError(f"bad selector {text!r}; expected {_GRAMMAR}")
        body = m.group(1).strip()
        if body == "*":
            return cls()
        pairs = []
        for part in body.split(","):
            k, sep, v = part.strip().partition(":")
            if not sep or not _TOKEN.match(k) or not _TOKEN.match(v):
                raise ValueError(f"bad selector {text!r}; expected {_GRAMMAR}")
            pairs.append((k, v))
        return cls(tuple(sorted(pairs)))

    def matches(self, device: Device) -> bool:
        facts = device.facts()
        return all(facts.get(k) == v for k, v in self.pairs)
