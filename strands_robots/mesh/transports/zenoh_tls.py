"""The mutual-TLS zenoh config a mesh identity makes, and the identity a zenoh config carries.

``tls_config(dir)`` is a session that speaks only ``tls/`` (a ``tcp/`` locator offered by a scouting peer is never
dialled, and no tcp listener opens), presents the identity's certificate on every link it opens or accepts, and
admits only peers whose certificate the mesh CA issued (``enable_mtls``). Names are not checked against endpoints
(``verify_name_on_connect`` off): endpoints are addresses, and who a peer is travels signed in every envelope
(``mesh.identity``). Scouting stays on the strands group, so certificate holders on a LAN still find each other with
no endpoints at all.

``identity_dir(url)`` picks the identity a ``zenoh://`` URL uses: ``?identity=<dir>``, else ``STRANDS_MESH_IDENTITY``,
else ``~/.strands/mesh/identity`` when it holds one; ``none`` for any of them means an open mesh.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..pki import DEFAULT_IDENTITY

logger = logging.getLogger(__name__)

LINK_PROTOCOLS = ("tls", "quic")
"""Link protocols that carry TLS: a config that allows any other admits peers without a certificate."""


def identity_dir(url: str) -> str | None:
    """The identity directory ``url`` uses, or None for an open mesh."""
    q = parse_qs(urlparse(url).query)
    named = (q.get("identity") or [os.environ.get("STRANDS_MESH_IDENTITY", "")])[0]
    if named.lower() == "none":
        return None
    if named:
        return os.path.expanduser(named)
    if "config" in q or os.environ.get("STRANDS_ZENOH_CONFIG"):
        return None  # a config file says for itself whether it carries an identity
    default = DEFAULT_IDENTITY.expanduser()
    return str(default) if (default / "cert.pem").is_file() else None


def tls_section(directory: str | Path) -> dict[str, Any]:
    """``transport.link`` for an identity directory (``cert.pem``, ``key.pem``, ``ca.pem``)."""
    d = Path(directory).expanduser().resolve()
    missing = [n for n in ("cert.pem", "key.pem", "ca.pem") if not (d / n).is_file()]
    if missing:
        hint = f"`python -m strands_robots.mesh cert NAME --out {d}`"
        raise FileNotFoundError(f"mesh identity {d} lacks {missing}: issue one with {hint}")
    cert, key = str(d / "cert.pem"), str(d / "key.pem")
    tls = {
        "root_ca_certificate": str(d / "ca.pem"),
        "listen_certificate": cert,
        "listen_private_key": key,
        "connect_certificate": cert,
        "connect_private_key": key,
        "enable_mtls": True,
        "verify_name_on_connect": False,
    }
    return {"protocols": ["tls"], "tls": tls}


def check_endpoints(endpoints: list[str]) -> list[str]:
    bad = [e for e in endpoints if e.split("/", 1)[0] not in LINK_PROTOCOLS]
    if bad:
        raise ValueError(f"a mesh with an identity speaks only tls/ (or quic/) endpoints, not {bad}")
    return endpoints


def tls_config(
    directory: str | Path,
    *,
    mode: str = "peer",
    listen: list[str] | None = None,
    connect: list[str] | None = None,
    scout: str | None = None,
) -> dict[str, Any]:
    """A whole zenoh config (a dict, JSON-able) for ``directory``'s identity: what ``python -m strands_robots.mesh
    config`` writes for a router or another zenoh client. ``listen`` defaults to ``tls/[::]:0`` for a peer or router
    (zenoh's default tcp listener would not open on a tls-only session); explicit endpoints switch scouting off."""
    if mode not in ("peer", "client", "router"):
        raise ValueError(f"zenoh mode must be peer, client or router, not {mode!r}")
    cfg: dict[str, Any] = {"mode": mode, "transport": {"link": tls_section(directory)}}
    cfg["listen"] = {"endpoints": check_endpoints(listen or ([] if mode == "client" else ["tls/[::]:0"]))}
    if connect:
        cfg["connect"] = {"endpoints": check_endpoints(connect)}
    if listen or connect or mode == "client":
        cfg["scouting"] = {"multicast": {"enabled": False}}
    elif scout:
        cfg["scouting"] = {"multicast": {"enabled": True, "address": scout}}
    return cfg


def authenticated(link: dict[str, Any]) -> bool:
    """Whether a zenoh ``transport.link`` admits only certificate holders: mutual TLS on, a root CA, and only TLS
    link protocols allowed (without ``protocols`` a peer can still join over plain ``tcp/``)."""
    tls = link.get("tls") or {}
    protocols = link.get("protocols")
    return (
        tls.get("enable_mtls") is True
        and bool(tls.get("root_ca_certificate"))
        and isinstance(protocols, list)
        and bool(protocols)
        and all(p in LINK_PROTOCOLS for p in protocols)
    )


def identity_of(link: dict[str, Any]) -> Any:
    """The ``mesh.identity.Identity`` an authenticated ``transport.link`` presents, else None (with a log line saying
    why a config that looks secured still leaves the mesh open)."""
    from ..identity import Identity, Unverified

    if not authenticated(link):
        return None
    tls = link["tls"]
    cert = tls.get("connect_certificate") or tls.get("listen_certificate")
    key = tls.get("connect_private_key") or tls.get("listen_private_key")
    try:
        if not (cert and key):
            raise ValueError("it names no certificate and key files")
        return Identity.from_files(cert, key, tls["root_ca_certificate"])
    except (OSError, ValueError, Unverified) as exc:
        logger.warning("mesh: the zenoh config's TLS identity cannot sign envelopes (%s); the mesh stays open", exc)
        return None


def link_of(cfg: Any) -> dict[str, Any]:
    """``transport.link`` of a ``zenoh.Config`` as a dict."""
    link = json.loads(cfg.get_json("transport/link"))
    return link if isinstance(link, dict) else {}
