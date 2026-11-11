"""Who sent an envelope, proven: the mTLS certificate a peer joined the mesh with also signs what it sends.

zenoh authenticates the link but hands a queryable or subscriber no peer certificate, so the identity travels in
the envelope: ``sig = {"cert": <DER, base64>, "sig": <ECDSA-SHA256, base64>}`` over the envelope's canonical JSON
(sorted keys, without ``sig``). ``Identity.verify`` accepts a certificate issued directly by the mesh CA, inside its
validity, and returns the ``Signer``: its name (the certificate's common name) and the device ids it may answer for.
A peer id on such a mesh is ``<name>.<random>``, so a signer can speak only for its own peers. ``Freshness`` keeps a
captured request from being sent again.
"""

from __future__ import annotations

import base64
import datetime as dt
import fnmatch
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..robot.safety import RequestRefused
from . import wire
from .pki import SERVES, check_name

MAX_SKEW = 30.0
"""How far (s) a signed request's ``t`` may be from the device's clock, and how long its ``(peer, seq)`` is kept."""


class Unverified(RequestRefused):
    """An envelope on a mesh with identities that is unsigned, badly signed, or signed by a stranger."""


@dataclass(frozen=True)
class Signer:
    name: str
    serves: tuple[str, ...] = ()

    def may_serve(self, device_id: str) -> bool:
        return any(fnmatch.fnmatchcase(device_id, p) for p in self.serves)

    def owns(self, peer: Any) -> bool:
        """``peer`` is one of this signer's own peer ids (``<name>.<random>``)."""
        return isinstance(peer, str) and peer.startswith(self.name + ".")


def canonical(env: dict[str, Any]) -> bytes:
    """The bytes a signature covers: the envelope without ``sig``, as sorted, compact JSON."""
    body = {k: v for k, v in env.items() if k != "sig"}
    return json.dumps(body, default=wire.jsonable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _signer(cert: Any) -> Signer:
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    name = check_name(str(cn[0].value) if cn else "")
    try:
        uris = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        found = uris.get_values_for_type(x509.UniformResourceIdentifier)
    except x509.ExtensionNotFound:
        found = []
    return Signer(name, tuple(u[len(SERVES) :] for u in found if u.startswith(SERVES)))


class Identity:
    """This peer's certificate and key, and the mesh CA it trusts."""

    def __init__(self, cert_pem: bytes, key_pem: bytes, ca_pem: bytes) -> None:
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        self._ca = x509.load_pem_x509_certificate(ca_pem)
        cert = x509.load_pem_x509_certificate(cert_pem)
        key = serialization.load_pem_private_key(key_pem, password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey) or key.public_key() != cert.public_key():
            raise ValueError("a mesh identity needs the EC private key of its certificate")
        self._key = key
        self._der = base64.b64encode(cert.public_bytes(serialization.Encoding.DER)).decode("ascii")
        self._cache: dict[str, tuple[Signer, Any, dt.datetime, dt.datetime]] = {}
        self._lock = threading.Lock()
        self.signer = self._check_cert(cert)
        self.name = self.signer.name

    @classmethod
    def load(cls, directory: str | Path) -> Identity:
        """The identity in a directory ``pki.issue`` wrote: ``cert.pem``, ``key.pem``, ``ca.pem``."""
        d = Path(directory).expanduser()
        return cls.from_files(d / "cert.pem", d / "key.pem", d / "ca.pem")

    @classmethod
    def from_files(cls, cert: str | Path, key: str | Path, ca: str | Path) -> Identity:
        return cls(Path(cert).read_bytes(), Path(key).read_bytes(), Path(ca).read_bytes())

    def new_peer_id(self, token: str) -> str:
        return wire.check_id(f"{self.name}.{token}", "peer")

    def seal(self, env: dict[str, Any]) -> dict[str, Any]:
        """``env`` signed by this identity (a new dict)."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        sig = self._key.sign(canonical(env), ec.ECDSA(hashes.SHA256()))
        return {**env, "sig": {"cert": self._der, "sig": base64.b64encode(sig).decode("ascii")}}

    def verify(self, env: dict[str, Any]) -> Signer:
        """The ``Signer`` of ``env``; ``Unverified`` when it is unsigned or the signature does not hold."""
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec

        sig = env.get("sig")
        if not isinstance(sig, dict) or not isinstance(sig.get("cert"), str) or not isinstance(sig.get("sig"), str):
            raise Unverified("the envelope is not signed; every peer on this mesh signs with its mesh certificate")
        signer, public = self._trusted(sig["cert"])
        try:
            public.verify(base64.b64decode(sig["sig"], validate=True), canonical(env), ec.ECDSA(hashes.SHA256()))
        except (InvalidSignature, ValueError) as exc:
            raise Unverified(f"the envelope's signature by {signer.name!r} does not hold") from exc
        return signer

    def _trusted(self, der_b64: str) -> tuple[Signer, Any]:
        digest = hashlib.sha256(der_b64.encode()).hexdigest()
        with self._lock:
            hit = self._cache.get(digest)
        if hit is None:
            from cryptography import x509

            try:
                cert = x509.load_der_x509_certificate(base64.b64decode(der_b64, validate=True))
            except ValueError as exc:
                raise Unverified("the envelope's certificate does not parse") from exc
            hit = (self._check_cert(cert), cert.public_key(), cert.not_valid_before_utc, cert.not_valid_after_utc)
            with self._lock:
                self._cache[digest] = hit
        signer, public, start, end = hit
        if not start <= dt.datetime.now(dt.timezone.utc) <= end:
            raise Unverified(f"the certificate of {signer.name!r} is outside its validity; issue it a new one")
        return signer, public

    def _check_cert(self, cert: Any) -> Signer:
        from cryptography import x509
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import ec

        try:
            cert.verify_directly_issued_by(self._ca)
        except (ValueError, TypeError, InvalidSignature) as exc:
            raise Unverified("the certificate was not issued by this mesh's CA") from exc
        try:
            ca = cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
        except x509.ExtensionNotFound:
            ca = False
        if ca or not isinstance(cert.public_key(), ec.EllipticCurvePublicKey):
            raise Unverified("a mesh peer signs with an EC end-entity certificate, not a CA")
        try:
            return _signer(cert)
        except ValueError as exc:
            raise Unverified(str(exc)) from exc


class Freshness:
    """A device's memory of the signed requests it answered: a request more than ``MAX_SKEW`` s off its clock, or a
    ``(peer, seq)`` it has seen in that window, is refused (a sniffed request sent again is no sign of life)."""

    def __init__(self, max_skew: float = MAX_SKEW) -> None:
        self.max_skew = max_skew
        self._seen: dict[tuple[str, int], float] = {}
        self._lock = threading.Lock()

    def admit(self, peer: str, seq: int, t: float) -> None:
        now = time.time()
        if abs(now - t) > self.max_skew:
            raise Unverified(f"the request is {now - t:+.1f} s off this device's clock; sync the clocks (NTP)")
        with self._lock:
            for k in [k for k, at in self._seen.items() if now - at > 2 * self.max_skew]:
                del self._seen[k]
            if (peer, seq) in self._seen:
                raise Unverified(f"request seq {seq} of {peer} was already answered; a repeat is refused")
            self._seen[(peer, seq)] = now
