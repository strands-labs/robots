"""What a zenoh link means for a LeRobot session: who can be on the other end, and the client settings it implies.

A real robot takes actions only over an authenticated channel: a ``tls``/``quic`` link opened with a zenoh config
that does mutual TLS. The same machine is the one exception, pinned to its loopback address. Anything else needs
the operator's word (``trusted=True``): a plain or one-way TLS link cannot tell a policy server from a shared zenoh
router, every client of which can read the frames and answer for the server.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
from pathlib import Path
from typing import Any

WAN_TIMEOUT_S = (
    30.0  # a first call across the internet wakes the model and carries two frames: LeRobot's 5-10 s are too tight
)

Address = ipaddress.IPv4Address | ipaddress.IPv6Address

ROUTED = ("tls", "quic", "ws", "wss")
"""Links that usually reach the server through a zenoh router."""

AUTHENTICATED = ("tls", "quic")
"""Links on which a zenoh config's mutual TLS applies."""


def split(endpoint: str) -> tuple[str, str, str]:
    """``<link>/host:port`` -> (link, host, port); an IPv6 host loses its brackets, a bare ``host:port`` is tcp."""
    link, _, rest = endpoint.partition("/") if "/" in endpoint else ("tcp", "", endpoint)
    host, _, port = rest.rpartition(":")
    return link, host.removeprefix("[").removesuffix("]"), port


def address(host: str) -> Address | None:
    """The address a host names as written, as ``inet_aton`` reads it; None for a hostname. Nothing is resolved.

    Every literal form counts: IPv6 with a zone (``fe80::1%en0``), IPv4-mapped and 6to4 IPv6 (judged as the IPv4
    address they carry), and the numeric IPv4 forms ``inet_aton`` reads (``134744072``, ``0x08080808``,
    ``010.0.0.1``, ``127.1``). A trailing dot is dropped. Resolvers differ on some forms (octal); that is harmless
    because a loopback verdict is pinned to the literal address it judged (``loopback``) and nothing else is trusted.
    """
    text = host.split("%", 1)[0].rstrip(".")
    try:
        addr: Address = ipaddress.ip_address(text)
    except ValueError:
        try:
            addr = ipaddress.IPv4Address(socket.inet_aton(text))
        except OSError:
            return None
    if isinstance(addr, ipaddress.IPv6Address):
        return addr.ipv4_mapped or addr.sixtofour or addr
    return addr


def is_local(endpoint: str) -> bool:
    """``<link>/host:port`` names this machine or its LAN, as written. Picks timeouts and frame encoding only.

    Local: loopback, private and link-local addresses, ``localhost``, mDNS ``*.local``, and a single-label hostname.
    A dotted name or a public address is the internet. A guess from the text that never decides trust:
    ``check_channel`` does.
    """
    host = split(endpoint)[1].rstrip(".").lower()
    addr = address(host)
    if addr is None:
        return host in ("localhost", "") or host.endswith(".local") or "." not in host
    return addr.is_private or addr.is_loopback or addr.is_link_local


def loopback(endpoint: str) -> str | None:
    """A plain ``tcp``/``udp`` link to this machine, pinned to its address (``tcp/127.0.0.1:7447``); else None.

    ``localhost`` and every numeric loopback form become the literal address, so zenoh never resolves a name this
    check did not judge.
    """
    link, host, port = split(endpoint)
    if link not in ("tcp", "udp"):
        return None
    addr = ipaddress.IPv4Address("127.0.0.1") if host.rstrip(".").lower() == "localhost" else address(host)
    if addr is None or not addr.is_loopback:
        return None
    return f"{link}/{addr}:{port}" if addr.version == 4 else f"{link}/[{addr}]:{port}"


def routed(endpoint: str) -> bool:
    """``<link>/host:port`` goes through a router (``tls``/``quic``/``ws``), not straight to the server."""
    return split(endpoint)[0] in ROUTED


def trusted(asked: bool) -> bool:
    """The operator vouches for whoever answers: ``trusted=True`` or ``STRANDS_ZENOH_TRUSTED=1``."""
    return asked or os.environ.get("STRANDS_ZENOH_TRUSTED", "").lower() in ("1", "true", "yes")


def zenoh_config(asked: str | None) -> str | None:
    """The zenoh config a session opens with: ``zenoh_config=`` or ``STRANDS_LEROBOT_ZENOH_CONFIG`` (``~`` expanded)."""
    path = asked or os.environ.get("STRANDS_LEROBOT_ZENOH_CONFIG") or None
    return None if path is None else str(Path(path).expanduser())


def mutual_tls(path: str) -> bool:
    """The zenoh config at ``path`` makes each end of a link prove a certificate from its CA.

    Read with zenoh's own parser, so this judges what zenoh will load: ``transport.link.tls.enable_mtls`` true with
    a CA, a certificate and a private key to present. A file zenoh cannot read raises ``ValueError``.
    """
    import zenoh

    try:
        tls = json.loads(zenoh.Config.from_file(path).get_json("transport/link/tls")) or {}
    except Exception as e:  # zenoh raises its own error type for a missing or malformed file
        raise ValueError(f"zenoh config {path}: {e}") from e
    keys = ("root_ca_certificate", "connect_certificate", "connect_private_key")
    return tls.get("enable_mtls") is True and all(tls.get(k) for k in keys)


def robot_role(config: str, mode: str | None = None) -> str | None:
    """Why the zenoh config at ``config`` is not LeRobot's robot role on a router, or None when it is.

    The role: mutual TLS presenting a ``robot-client`` certificate (the router's ACL gives the robot's part to that
    name only, ``roles.py``), joined as a router client (``zenoh_mode`` client; LeRobot turns discovery off).
    """
    from .roles import ROBOT, common_name

    if not mutual_tls(config):
        return f"{config} does not set transport.link.tls.enable_mtls with a CA, a certificate and a key"
    cn = common_name(config)
    if cn != ROBOT:
        return f"{config} presents the certificate {cn!r}, and the router lets only {ROBOT!r} take the robot's part"
    if mode not in (None, "client"):
        return f"zenoh_mode={mode!r}: a robot joins the router as a client, where its ACL applies"
    return None


def check_channel(who: str, endpoint: str, robot: Any, asked: bool, config: str | None, mode: str | None) -> str:
    """The endpoint a session may open for ``robot``, or ``PermissionError``: a real robot needs an authenticated link.

    ``config``: the session's zenoh config (``zenoh_config``); ``mode``: the ``zenoh_mode`` the caller set, if any.

    Allowed for a real robot: a ``tls``/``quic`` link with a zenoh config in LeRobot's robot role (``robot_role``);
    a plain link to this machine (returned pinned to its loopback address); anything the operator vouches for. A
    simulated robot is not gated.
    """
    if not getattr(robot, "is_real", False) or trusted(asked):
        return endpoint
    config, link = zenoh_config(config), split(endpoint)[0]
    why = robot_role(config, mode) if link in AUTHENTICATED and config is not None else None
    if link in AUTHENTICATED and config is not None and why is None:
        return endpoint
    pinned = loopback(endpoint)
    if pinned is not None:
        return pinned
    why = why or (
        "the link carries no identity, so whoever shares it (a zenoh router's clients, anyone answering at "
        "that address) can read the cameras and answer for the server"
    )
    raise PermissionError(
        f"{who}: a real robot takes actions from {endpoint} only over an authenticated channel; {why}. Make the "
        "role configs (python -m strands_robots.policy.lerobot zenoh-config --deployment <d> --out <dir>), run "
        "zenohd -c <dir>/router.json5 and connect with zenoh://tls/<router>:7447/<d> and "
        "zenoh_config=<dir>/robot.json5, or vouch for it: Policy(..., trusted=True) or STRANDS_ZENOH_TRUSTED=1 "
        "(docs/lerobot.md)"
    )


def server_zenoh(endpoint: str, config: str | None) -> Any:
    """LeRobot's ``ZenohConfig`` for a server at ``endpoint`` opened with the zenoh config ``config``.

    A config in LeRobot's ``policy-server`` role (``roles.py``) joins the router at ``endpoint`` as a client, so the
    router's ACL applies to it; any other config, or none, listens at ``endpoint``.
    """
    from lerobot.transport import ZenohConfig

    from .roles import SERVER, common_name

    path = None if config is None else str(Path(config).expanduser())
    if path is not None and common_name(path) == SERVER:
        return ZenohConfig(mode="client", connect_endpoints=[endpoint], config_file=path)
    return ZenohConfig(listen_endpoints=[endpoint], config_file=path)


def wan_defaults(endpoint: str, settings: dict[str, Any], descriptor: dict[str, Any] | None = None) -> dict[str, Any]:
    """LeRobot client settings a link implies, under anything the caller set by name.

    Off the LAN: handshake/action/startup timeouts >= 30 s and JPEG frames. A ``tls``/``quic``/``ws`` link is a
    router, so ``zenoh_mode="client"``. Once the server has spoken, the action timeout also covers its own
    ``action_deadline_s`` (the hello's ``limits``) — a server allowed 5 s to answer is not a client fault at 5 s.
    """
    out: dict[str, Any] = {}
    if routed(endpoint):
        out["zenoh_mode"] = "client"
    if not is_local(endpoint):
        out.update(handshake_timeout_s=WAN_TIMEOUT_S, action_timeout_s=WAN_TIMEOUT_S, startup_timeout_s=WAN_TIMEOUT_S)
        out["encoding"] = "jpeg"
    if descriptor is not None:
        deadline = float((descriptor.get("limits") or {}).get("action_deadline_s") or 0.0)
        if deadline:
            out["action_timeout_s"] = max(
                out.get("action_timeout_s", 5.0), deadline + (0.0 if is_local(endpoint) else 5.0)
            )
    out.update(settings)  # the caller's word wins
    return out
