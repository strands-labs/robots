"""Test-only PKI helpers: build a CA + leaf certs in-process for mTLS tests.

Used by ``tests/mesh/test_zenoh_transport_security.py`` to spin up a Zenoh fleet
with real mTLS + ACL gating in a single Python process. The certs are
written to ``tmp_path`` so each test gets its own ephemeral CA.

This is NOT production code -- ``cryptography`` is in the test extras
only. Production cert provisioning happens through AWS IoT
``strands_robots.mesh.iot.provision`` or an operator's existing PKI.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

# Anchored allowlist for test-cert common names. Same shape rule used by
# the mesh's source-id / peer-id validators (alphanumerics, dot, underscore,
# hyphen, must start with alphanumeric). Rejects path traversal (``..``,
# ``/``) and shell-meta characters that could land in interpolated paths,
# DNSName extensions, or audit log lines downstream.
_CN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_cn(common_name: str) -> None:
    """Reject CNs that aren't a strict subset of [A-Za-z0-9._-].

    Raises:
        ValueError: if *common_name* is empty, leading non-alphanumeric,
            or contains any character outside the allowlist.
    """
    if not _CN_RE.fullmatch(common_name):
        raise ValueError(f"invalid CN for test cert: {common_name!r}")


@dataclass(frozen=True)
class EphemeralCA:
    """A test CA with a helper to mint leaf certs.

    Renamed from ``TestCA`` to avoid pytest's ``Test*`` collection rule
    (a frozen dataclass with a generated ``__init__`` would otherwise
    trigger ``PytestCollectionWarning`` on every run).
    """

    cert: x509.Certificate
    key: rsa.RSAPrivateKey
    cert_path: Path

    def issue(self, common_name: str, out_dir: Path) -> tuple[Path, Path]:
        """Mint a leaf cert + key for *common_name* under *out_dir*.

        Returns ``(cert_path, key_path)``. Both files are written with
        the encoded PEM bytes; permissions on the key file are tightened
        to ``0o600``. Validates *common_name* against the test-CN
        allowlist before interpolating it into the filesystem path or
        ``x509.DNSName(...)``.
        """
        _validate_cn(common_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        leaf_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "strands-robots-test"),
            ]
        )
        leaf = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self.cert.subject)
            .public_key(leaf_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(_dt.datetime.now(_dt.UTC) - _dt.timedelta(minutes=1))
            .not_valid_after(_dt.datetime.now(_dt.UTC) + _dt.timedelta(hours=1))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName(common_name), x509.DNSName("localhost")]),
                critical=False,
            )
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .sign(self.key, hashes.SHA256())
        )
        cert_path = out_dir / f"{common_name}.crt"
        key_path = out_dir / f"{common_name}.key"
        cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            leaf_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        key_path.chmod(0o600)
        return cert_path, key_path


def make_test_ca(out_dir: Path) -> EphemeralCA:
    """Build a self-signed CA in *out_dir*. Returns the generated CA."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Hard-coded CA CN: validated as a free assertion that the format is
    # what we think it is.
    _validate_cn("strands-robots-test-ca")
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "strands-robots-test-ca"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "strands-robots-test"),
        ]
    )
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_dt.datetime.now(_dt.UTC) - _dt.timedelta(minutes=1))
        .not_valid_after(_dt.datetime.now(_dt.UTC) + _dt.timedelta(days=1))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    cert_path = out_dir / "ca.crt"
    key_path = out_dir / "ca.key"
    cert_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        ca_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    return EphemeralCA(cert=ca_cert, key=ca_key, cert_path=cert_path)
