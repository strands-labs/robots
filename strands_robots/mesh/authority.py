"""Who may clear an e-stop: the operator who stopped, a named authority, or the holder of an operator key.

Every e-stop over the mesh opens a lockout with an ``epoch`` (a fresh id) and the stopping operator's name. A
device's ``Authority`` decides who may resume it:

* without keys, the operator who stopped or a name in ``names`` (names are claimed, not proven: use keys on any
  network you do not control);
* with ``public_keys``, only a fresh Ed25519-signed assertion for THIS lockout and THIS device:
  ``{"v", "epoch", "targets", "t", "nonce", "by", "sig"}``. A captured assertion cannot clear a later lockout
  (epoch), another device (targets), or be replayed (nonce, ``max_age``).
"""

from __future__ import annotations

import base64
import json
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..robot.safety import RequestRefused

VERSION = 1


def new_epoch() -> str:
    return uuid.uuid4().hex


@dataclass
class Lockout:
    """A device's current e-stop: engaged or not, which stop (epoch) and who stopped it."""

    engaged: bool = False
    epoch: str | None = None
    by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"engaged": self.engaged, "epoch": self.epoch, "by": self.by}


def keygen() -> tuple[Any, str]:
    """A new operator key pair: ``(private_key, public_key_b64)``. Keep the private half on the operator's machine."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    private = Ed25519PrivateKey.generate()
    raw = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return private, base64.b64encode(raw).decode("ascii")


def _signed_bytes(a: dict[str, Any]) -> bytes:
    body = {k: a[k] for k in ("v", "epoch", "targets", "t", "nonce", "by")}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def sign(private_key: Any, epoch: str, targets: list[str], by: str) -> dict[str, Any]:
    """A resume assertion for lockout ``epoch`` on ``targets``, signed by ``private_key``."""
    a = {"v": VERSION, "epoch": epoch, "targets": sorted(targets), "t": time.time(), "nonce": secrets.token_hex(16)}
    a["by"] = by
    a["sig"] = base64.b64encode(private_key.sign(_signed_bytes(a))).decode("ascii")
    return a


@dataclass
class Authority:
    """A device-side resume policy (``Mesh(resume_authority=...)``)."""

    names: tuple[str, ...] = ()
    public_keys: tuple[str, ...] = ()
    max_age: float = 60.0
    trust_names: bool | None = None
    """Accept claimed names without keys. ``None``: only when every transport is loopback or in-process."""
    _seen: set[str] = field(default_factory=set, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(
        self, device_id: str, lockout: Lockout, approved_by: str | None, assertion: Any, *, local: bool = False
    ) -> str:
        """The name to record as the approver, or ``RequestRefused`` with the reason. ``local``: every transport the
        request could have come over is loopback or in-process."""
        if not approved_by or not str(approved_by).strip():
            raise RequestRefused(f"{device_id}: resume needs approved_by=<the operator who cleared the stop>")
        if self.public_keys:
            return self._check_signed(device_id, lockout, assertion)
        if not (self.trust_names if self.trust_names is not None else local):
            raise RequestRefused(
                f"{device_id}: resume over this network needs operator keys "
                "(Mesh(resume_authority=Authority(public_keys=...)), or trust_names=True to accept claimed names)"
            )
        if approved_by == lockout.by or approved_by in self.names:
            return str(approved_by)
        who = [n for n in (lockout.by, *self.names) if n]
        raise RequestRefused(
            f"{device_id}: {approved_by!r} is not a resume authority for this stop (allowed: {who}). Resume as the "
            "operator who stopped it (broadcast/estop by=<operator>), name one in "
            "Mesh(resume_authority=Authority(names=[...])), or use operator keys (Authority(public_keys=...))"
        )

    def _check_signed(self, device_id: str, lockout: Lockout, a: Any) -> str:
        if not isinstance(a, dict) or "sig" not in a:
            raise RequestRefused(f"{device_id}: resume needs an assertion signed by an operator key")
        if a.get("v") != VERSION or a.get("epoch") != lockout.epoch or device_id not in (a.get("targets") or ()):
            raise RequestRefused(f"{device_id}: the resume assertion is not for this device's current stop")
        age = time.time() - float(a.get("t", 0))
        if not -5.0 <= age <= self.max_age:
            raise RequestRefused(f"{device_id}: the resume assertion is stale ({age:.0f} s old)")
        if not self._verify(a):
            raise RequestRefused(f"{device_id}: the resume assertion's signature matches no trusted operator key")
        with self._lock:
            if a["nonce"] in self._seen:
                raise RequestRefused(f"{device_id}: the resume assertion was already used")
            self._seen.add(a["nonce"])
        return str(a.get("by") or "operator-key")

    def _verify(self, a: dict[str, Any]) -> bool:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            sig = base64.b64decode(a["sig"])
            msg = _signed_bytes(a)
        except (KeyError, ValueError, TypeError):
            return False
        for pub in self.public_keys:
            try:
                Ed25519PublicKey.from_public_bytes(base64.b64decode(pub)).verify(sig, msg)
                return True
            except (InvalidSignature, ValueError):
                continue
        return False
