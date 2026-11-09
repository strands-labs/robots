"""A mesh certificate authority and the certificates its peers present: the identity a zenoh mesh admits over mTLS
and signs every envelope with.

``make_ca`` writes ``ca.pem`` and ``ca.key`` (keep the key on one admin machine). ``issue`` writes an identity
directory ``{ca.pem, cert.pem, key.pem}`` for one machine: the certificate's common name is the peer's name, and its
``strands-serves:<pattern>`` URIs list the device ids it may answer for (none: an operator that mounts nothing; ``*``:
any id, e.g. a simulated world). Keys are ECDSA P-256, the curve every zenoh TLS build accepts.
``python -m strands_robots.mesh`` runs both from a shell. ``cryptography`` is imported where used.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

HOME = Path("~/.strands/mesh")
DEFAULT_CA = HOME / "ca"
DEFAULT_IDENTITY = HOME / "identity"
"""Where ``issue`` writes by default, and where a ``zenoh://`` mesh looks for its identity when none is named."""

SERVES = "strands-serves:"
"""The URI prefix of each device-id pattern a certificate may answer for."""

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PATTERN = re.compile(r"^[A-Za-z0-9_.:*?-]{1,64}$")


def check_name(name: Any) -> str:
    """A peer name: letters, digits, ``_`` and ``-`` (no dots: a peer id is ``<name>.<random>``)."""
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"mesh identity name {name!r} must be 1-64 letters, digits, _ or - (no dots)")
    return name


def check_serves(patterns: Iterable[str]) -> tuple[str, ...]:
    """Device-id patterns (``fnmatch``: ``*`` and ``?``) a certificate may answer for."""
    out = tuple(str(p).strip() for p in patterns if str(p).strip())
    bad = [p for p in out if not _PATTERN.match(p)]
    if bad:
        raise ValueError(f"serves patterns {bad} must be device ids, with * or ? as wildcards")
    return out


def _write(path: Path, data: bytes, *, private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, 0o600 if private else 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    if private:
        os.chmod(path, 0o600)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _key_pem(key: Any) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )


def make_ca(ca_dir: str | os.PathLike[str] = DEFAULT_CA, *, name: str = "strands-mesh-ca", days: int = 3650) -> Path:
    """A new CA in ``ca_dir`` (``ca.pem``, ``ca.key`` 0600). Refuses to overwrite one: every certificate it issued
    would stop verifying."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    out = Path(ca_dir).expanduser()
    if (out / "ca.key").exists() or (out / "ca.pem").exists():
        raise FileExistsError(f"{out} already holds a mesh CA; remove it first to start a new mesh")
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, check_name(name))])
    usage = x509.KeyUsage(False, False, False, False, False, True, True, False, False)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - dt.timedelta(minutes=5))
        .not_valid_after(_now() + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(usage, critical=True)
        .sign(key, hashes.SHA256())
    )
    _write(out / "ca.key", _key_pem(key), private=True)
    _write(out / "ca.pem", cert.public_bytes(serialization.Encoding.PEM), private=False)
    return out


def issue(
    ca_dir: str | os.PathLike[str],
    name: str,
    out_dir: str | os.PathLike[str] = DEFAULT_IDENTITY,
    *,
    serves: Iterable[str] = (),
    days: int = 825,
) -> Path:
    """Issue ``name`` a certificate from the CA in ``ca_dir`` and write its identity directory ``out_dir``
    (``ca.pem``, ``cert.pem``, ``key.pem`` 0600). The key is made here and never leaves ``out_dir``."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    src = Path(ca_dir).expanduser()
    if not (src / "ca.key").exists():
        raise FileNotFoundError(f"no mesh CA in {src}: make one with `python -m strands_robots.mesh ca --ca {src}`")
    ca_cert = x509.load_pem_x509_certificate((src / "ca.pem").read_bytes())
    ca_key = serialization.load_pem_private_key((src / "ca.key").read_bytes(), password=None)
    if not isinstance(ca_key, ec.EllipticCurvePrivateKey):
        raise ValueError(f"{src / 'ca.key'} is not an EC key this module made")
    patterns = check_serves(serves)
    key = ec.generate_private_key(ec.SECP256R1())
    san: list[x509.GeneralName] = [x509.UniformResourceIdentifier(SERVES + p) for p in patterns]
    san = san or [x509.DNSName(check_name(name))]
    eku = x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH])
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, check_name(name))]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - dt.timedelta(minutes=5))
        .not_valid_after(min(_now() + dt.timedelta(days=days), ca_cert.not_valid_after_utc))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(eku, critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    out = Path(out_dir).expanduser()
    _write(out / "key.pem", _key_pem(key), private=True)
    _write(out / "cert.pem", cert.public_bytes(serialization.Encoding.PEM), private=False)
    _write(out / "ca.pem", (src / "ca.pem").read_bytes(), private=False)
    return out
