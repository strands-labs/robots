"""Is this viewer on the same network as the dashboard - and should it stop leaving it?"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable, Sequence

#: A /64 is the unit a home network is delegated, and the unit SLAAC hosts share.
V6_NETWORK_BITS = 64


def _global_v6(addr: str) -> ipaddress.IPv6Address | None:
    """Parse a routable IPv6 address, or None for anything that proves nothing."""
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])  # strip any zone id
    except ValueError:
        return None
    if not isinstance(ip, ipaddress.IPv6Address):
        return None
    if ip.is_link_local or ip.is_loopback or ip.is_private or ip.is_multicast:
        return None
    return ip


def same_network(client_ip: str | None, own_addrs: Iterable[str]) -> bool | None:
    """True/False when it can be decided from local facts, None when it cannot. Loopback is trivially
    local (the dashboard's own machine).
    """
    if not client_ip:
        return None
    bare = client_ip.split("%")[0]
    try:
        parsed = ipaddress.ip_address(bare)
    except ValueError:
        return None
    if parsed.is_loopback:
        return True
    client = _global_v6(bare)
    if client is None:
        return None  # IPv4 (NAT hides the answer) or a non-routable address
    nets = {
        ipaddress.ip_network(f"{a}/{V6_NETWORK_BITS}", strict=False)
        for a in (_global_v6(x) for x in own_addrs)
        if a is not None
    }
    if not nets:
        return None  # we have no global address of our own to compare against
    return any(client in n for n in nets)


def lan_urls(private_addrs: Sequence[str], port: int) -> list[str]:
    """Direct URLs for a viewer that turns out to be local, best candidate first."""
    out: list[str] = []
    for a in private_addrs:
        try:
            ip = ipaddress.ip_address(a.split("%")[0])
        except ValueError:
            continue
        if isinstance(ip, ipaddress.IPv4Address) and ip.is_private and not ip.is_loopback:
            url = f"http://{ip}:{port}"
            if url not in out:
                out.append(url)
    return out


def hint(client_ip: str | None, own_addrs: Sequence[str], port: int) -> dict:
    """The payload the UI renders. `same_network=None` must render NOTHING."""
    local = same_network(client_ip, own_addrs)
    urls = lan_urls(own_addrs, port) if local else []
    if local and not urls:
        # Local, but we cannot name a usable address - say so instead of implying one.
        return {
            "same_network": True,
            "client_ip": client_ip,
            "lan_urls": [],
            "why": "you appear to be on this network, but the dashboard could not find a "
            "private address of its own to point you at",
        }
    if local:
        return {
            "same_network": True,
            "client_ip": client_ip,
            "lan_urls": urls,
            "why": "you are on the same network as this dashboard - the local address skips "
            "the trip out to Cloudflare and back, which is where camera streams stall",
        }
    if local is False:
        return {
            "same_network": False,
            "client_ip": client_ip,
            "lan_urls": [],
            "why": "you are reaching this dashboard from another network",
        }
    return {
        "same_network": None,
        "client_ip": client_ip,
        "lan_urls": [],
        "why": "cannot tell from an IPv4 address behind NAT whether you are local",
    }
