"""A software authenticator: a P-256 key that answers ``navigator.credentials.create`` / ``get`` as a browser would.

It builds attestation ``none`` (what the dashboard asks for) and ES256 assertions, so the server's real
``py_webauthn`` verification runs on every test.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import struct
from typing import Any

import cbor2
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class Authenticator:
    def __init__(self, origin: str, rp_id: str = "localhost") -> None:
        self.origin, self.rp_id = origin, rp_id
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.cred_id = secrets.token_bytes(16)
        self.count = 0

    @property
    def id(self) -> str:
        return b64(self.cred_id)

    def _client(self, kind: str, options: dict[str, Any], origin: str | None) -> bytes:
        doc = {"type": kind, "challenge": options["challenge"], "origin": origin or self.origin, "crossOrigin": False}
        return json.dumps(doc).encode()

    def _auth_data(self, flags: int, attested: bytes = b"") -> bytes:
        return hashlib.sha256(self.rp_id.encode()).digest() + bytes([flags]) + struct.pack(">I", self.count) + attested

    def create(self, options: dict[str, Any], *, origin: str | None = None) -> dict[str, Any]:
        n = self.key.public_key().public_numbers()
        cose = cbor2.dumps({1: 2, 3: -7, -1: 1, -2: n.x.to_bytes(32, "big"), -3: n.y.to_bytes(32, "big")})
        attested = bytes(16) + struct.pack(">H", len(self.cred_id)) + self.cred_id + cose
        att = cbor2.dumps({"fmt": "none", "attStmt": {}, "authData": self._auth_data(0x45, attested)})
        response = {
            "clientDataJSON": b64(self._client("webauthn.create", options, origin)),
            "attestationObject": b64(att),
        }
        return {
            "id": self.id,
            "rawId": self.id,
            "type": "public-key",
            "response": response,
            "clientExtensionResults": {},
        }

    def get(self, options: dict[str, Any], *, origin: str | None = None, count: int | None = None) -> dict[str, Any]:
        self.count = self.count + 1 if count is None else count
        data, client = self._auth_data(0x05), self._client("webauthn.get", options, origin)
        sig = self.key.sign(data + hashlib.sha256(client).digest(), ec.ECDSA(hashes.SHA256()))
        response = {"clientDataJSON": b64(client), "authenticatorData": b64(data), "signature": b64(sig)}
        return {
            "id": self.id,
            "rawId": self.id,
            "type": "public-key",
            "response": response,
            "clientExtensionResults": {},
        }
