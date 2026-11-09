"""LeRobot's role-based zenoh setup for one deployment: a router that lets each certificate do only its role's part.

The same configs LeRobot ships (``examples/remote_inference/zenoh/{router,server,robot}.json5``), for any deployment:
mutual TLS on every link, discovery off, both ends router clients, and a deny-by-default router ACL keyed on the
certificate's common name. Only ``policy-server`` may answer ``describe``/``open``/``control``, publish
``.../act`` and read ``.../obs``; only ``robot-client`` may publish ``.../obs`` and read ``.../act``. Any other
certificate from the same CA, or a second robot certificate, cannot answer for the server.
"""

from __future__ import annotations

import datetime
import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SERVER, ROBOT, ROUTER = "policy-server", "robot-client", "router"
PREFIX = "lerobot/inference/v1/deployments/{}"
"""LeRobot's deployment address (``remote_inference/protocol.py:deployment_prefix``); the ACL covers only it."""

_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_S = "instances/*/sessions/*/"
_CONTROL = ("describe", "instances/*/open", _S + "control")
_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("server-control", ("declare_queryable", "reply"), _CONTROL),
    ("server-output", ("put",), (_S + "act", _S + "language/result")),
    ("server-input", ("declare_subscriber",), (_S + "obs", _S + "language/request")),
    ("server-alive", ("liveliness_token",), ("instances/*/alive",)),
    ("server-presence", ("declare_liveliness_subscriber", "liveliness_query"), (_S + "alive",)),
    ("robot-control", ("query",), _CONTROL),
    ("robot-output", ("put",), (_S + "obs", _S + "language/request")),
    ("robot-input", ("declare_subscriber",), (_S + "act", _S + "language/result")),
    ("robot-alive", ("liveliness_token",), (_S + "alive",)),
    ("robot-presence", ("declare_liveliness_subscriber", "liveliness_query"), ("instances/*/alive",)),
)
_EGRESS = (
    "declare_queryable", "query", "reply", "declare_subscriber", "put", "delete",
    "liveliness_token", "declare_liveliness_subscriber", "liveliness_query",
)  # fmt: skip


@dataclass(frozen=True)
class Roles:
    """The files ``make`` wrote: run ``zenohd -c router``, serve with ``server``, connect the robot with ``robot``."""

    router: Path
    server: Path
    robot: Path
    endpoint: str


def prefix(deployment: str) -> str:
    """The deployment's key prefix, or ``ValueError`` for a name LeRobot would refuse."""
    if not _SEGMENT.match(deployment):
        raise ValueError(f"deployment {deployment!r} is not a LeRobot name ([A-Za-z0-9][A-Za-z0-9_.-]{{0,127}})")
    return PREFIX.format(deployment)


def _tls(certs: Path, cn: str, side: str) -> dict[str, Any]:
    return {
        "protocols": ["tls"],
        "tls": {
            "root_ca_certificate": str(certs / "ca.pem"),
            f"{side}_certificate": str(certs / f"{cn}.pem"),
            f"{side}_private_key": str(certs / f"{cn}-key.pem"),
            "enable_mtls": True,
            "close_link_on_expiration": True,
        },
    }


_KEY_USAGE_OFF = ("digital_signature", "content_commitment", "key_encipherment", "data_encipherment", "key_agreement",
                  "encipher_only", "decipher_only")  # fmt: skip
_NO_SCOUTING = {"multicast": {"enabled": False}, "gossip": {"enabled": False}}


def router_config(deployment: str, listen: str, certs: Path) -> dict[str, Any]:
    """LeRobot's ``router.json5`` for ``deployment``: mutual TLS on ``listen``, the role ACL on its keys."""
    d = prefix(deployment)
    rules = [
        {
            "id": i,
            "permission": "allow",
            "flows": ["ingress"],
            "messages": list(m),
            "key_exprs": [f"{d}/{e}" for e in keys],
        }
        for i, m, keys in _RULES
    ]
    rules.append(
        {
            "id": "authenticated-egress",
            "permission": "allow",
            "flows": ["egress"],
            "messages": list(_EGRESS),
            "key_exprs": [f"{d}/**"],
        }
    )
    policy = {
        who: [i for i, _, _ in _RULES if i.startswith(who)] + ["authenticated-egress"] for who in ("server", "robot")
    }
    return {
        "mode": "router",
        "listen": {"endpoints": [listen]},
        "scouting": _NO_SCOUTING,
        "transport": {"link": _tls(certs, ROUTER, "listen")},
        "access_control": {
            "enabled": True,
            "default_permission": "deny",
            "rules": rules,
            "subjects": [
                {"id": "server", "cert_common_names": [SERVER]},
                {"id": "robot", "cert_common_names": [ROBOT]},
            ],
            "policies": [{"subjects": [who], "rules": r} for who, r in policy.items()],
        },
    }


def role_config(role: str, connect: str, certs: Path) -> dict[str, Any]:
    """LeRobot's ``server.json5``/``robot.json5``: a router client presenting the ``role`` certificate."""
    return {
        "mode": "client",
        "connect": {"endpoints": [connect]},
        "scouting": _NO_SCOUTING,
        "transport": {"link": _tls(certs, role, "connect")},
    }


def _key(path: Path) -> Any:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    if path.exists():
        return serialization.load_pem_private_key(path.read_bytes(), None)
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    path.touch(0o600)
    path.write_bytes(pem)
    return key


def _cert(key: Any, cn: str, issuer: Any, signer: Any, ca: bool, sans: list[Any], days: int) -> Any:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.UTC)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
    b = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer or subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=0 if ca else None), critical=True)
    )
    if ca:
        off = dict.fromkeys(_KEY_USAGE_OFF, False)
        b = b.add_extension(x509.KeyUsage(key_cert_sign=True, crl_sign=True, **off), critical=True)
    else:
        usage = [ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]
        b = b.add_extension(x509.ExtendedKeyUsage(usage), critical=False)
        b = b.add_extension(x509.SubjectAlternativeName(sans), critical=False)
    return b.sign(signer or key, hashes.SHA256())


def _names(hosts: tuple[str, ...]) -> list[Any]:
    from cryptography import x509

    out: list[Any] = []
    for h in hosts:
        try:
            out.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            out.append(x509.DNSName(h))
    return out


def issue(certs: Path, cn: str, name: str | None = None, hosts: tuple[str, ...] = ("localhost",)) -> Path:
    """A leaf certificate with common name ``cn`` from the CA in ``certs`` (made if absent), kept if it exists.

    Written as ``<name>.pem`` / ``<name>-key.pem`` (``name`` defaults to ``cn``): e.g. a second robot's
    ``issue(certs, "robot-client", "robot-2")``. The CA key stays in ``certs/ca-key.pem``; keep it off the robots.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    certs.mkdir(parents=True, exist_ok=True)
    ca_key = _key(certs / "ca-key.pem")
    if not (certs / "ca.pem").exists():
        ca = _cert(ca_key, "strands lerobot CA", None, None, True, [], 3650)
        (certs / "ca.pem").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    ca = x509.load_pem_x509_certificate((certs / "ca.pem").read_bytes())
    out = certs / f"{name or cn}.pem"
    if not out.exists():
        key = _key(certs / f"{name or cn}-key.pem")
        leaf = _cert(key, cn, ca.subject, ca_key, False, _names(hosts), 825)
        out.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    return out


def make(deployment: str, out: str | Path, hosts: tuple[str, ...] = ("localhost",), port: int = 7447) -> Roles:
    """Certificates (CA, router, ``policy-server``, ``robot-client``) and the three configs for ``deployment``.

    ``hosts``: the names/addresses robots and server dial the router by (its certificate's SANs); the first one is
    written into the client configs. Existing certificates are kept, so re-running only rewrites the configs.
    """
    prefix(deployment)
    root = Path(out).expanduser().resolve()
    certs = root / "certs"
    issue(certs, ROUTER, hosts=hosts)
    for role in (SERVER, ROBOT):
        issue(certs, role)
    host = hosts[0]
    endpoint = f"tls/{f'[{host}]' if ':' in host else host}:{port}"
    files = {
        "router": router_config(deployment, f"tls/[::]:{port}", certs),
        "server": role_config(SERVER, endpoint, certs),
        "robot": role_config(ROBOT, endpoint, certs),
    }
    for name, cfg in files.items():
        (root / f"{name}.json5").write_text(json.dumps(cfg, indent=2) + "\n")
    return Roles(root / "router.json5", root / "server.json5", root / "robot.json5", endpoint)


def common_name(config: str) -> str | None:
    """The common name of the certificate a zenoh config presents when it connects, or None if it presents none."""
    import zenoh
    from cryptography import x509
    from cryptography.x509.oid import NameOID

    try:
        path = json.loads(zenoh.Config.from_file(config).get_json("transport/link/tls/connect_certificate"))
    except Exception as e:  # zenoh raises its own error type for a missing or malformed file
        raise ValueError(f"zenoh config {config}: {e}") from e
    if not path:
        return None
    try:
        cert = x509.load_pem_x509_certificate(Path(path).expanduser().read_bytes())
    except (OSError, ValueError) as e:
        raise ValueError(f"zenoh config {config}: certificate {path}: {e}") from e
    names = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    return str(names[0].value) if names else None
