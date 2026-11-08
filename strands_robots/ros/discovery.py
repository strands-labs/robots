"""``host=`` -> how this process finds a robot's ROS graph, set once before ``rclpy.init``.

The ROS context is process-global: once rclpy is initialised its domain and middleware are fixed, so a
second, different request is refused instead of silently publishing to the wrong graph.

- ``rmw_zenoh_cpp``: connect the local zenoh session to the robot's router,
  ``tcp/<host>:7447``. Works across subnets and through a tunnel, since it is one TCP connection.
- DDS (cyclone / fastdds, Unitree, most robots): unicast peer via ``ROS_STATIC_PEERS`` (Jazzy+) and
  ``ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET``; on Humble DDS needs the peer in the vendor XML, so we say so.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .names import domain_id as _check_domain

_APPLIED: dict[str, dict[str, str]] = {}  # "env": what the first apply() wrote into this process


@dataclass(frozen=True)
class Discovery:
    host: str | None = None
    rmw: str | None = None  # None: whatever RMW_IMPLEMENTATION already says
    domain_id: int | None = None
    zenoh_port: int = 7447
    iface: str | None = None  # DDS: pin discovery to one NIC (a Unitree's eth0 is the only route to its MCU)

    def env(self) -> dict[str, str]:
        """The environment this discovery needs, without touching ``os.environ``."""
        out: dict[str, str] = {}
        if self.domain_id is not None:
            out["ROS_DOMAIN_ID"] = str(_check_domain(self.domain_id))
        rmw = self.rmw or os.environ.get("RMW_IMPLEMENTATION")
        if self.rmw:
            out["RMW_IMPLEMENTATION"] = self.rmw
        if self.iface:
            if not self.iface.replace("_", "").replace(".", "").isalnum():
                raise ValueError(f"iface must be a NIC name like eth0, got {self.iface!r}")
            out["CYCLONEDDS_URI"] = (
                f'<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="{self.iface}"/>'
                "</Interfaces></General></Domain></CycloneDDS>"
            )
        if self.host in (None, "", "localhost", "127.0.0.1"):
            return out
        if not isinstance(self.host, str) or any(c.isspace() or c in "\"';" for c in self.host):
            raise ValueError(f"host must be a hostname or IP, got {self.host!r}")
        if rmw == "rmw_zenoh_cpp":
            ep = json.dumps([f"tcp/{self.host}:{int(self.zenoh_port)}"])
            out["ZENOH_CONFIG_OVERRIDE"] = f"connect/endpoints={ep}"
            out["ZENOH_ROUTER_CHECK_ATTEMPTS"] = "-1"  # wait for the robot's router instead of failing at init
        else:
            out["ROS_STATIC_PEERS"] = self.host
            out["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "SUBNET"
        return out


def apply(d: Discovery, *, initialised: bool) -> dict[str, str]:
    """Write ``d.env()`` into the process before the first ``rclpy.init``; refuse a different second one."""
    want = d.env()
    if initialised or "env" in _APPLIED:
        have = _APPLIED.get("env", {})
        clash = {k: (os.environ.get(k), v) for k, v in want.items() if os.environ.get(k) != v}
        if clash:
            raise RuntimeError(
                f"this process's ROS context is already up with other settings {sorted(clash)}; a ROS context's "
                "domain and middleware are fixed at rclpy.init(). Use one process per robot graph, "
                "or the same host/domain."
            )
        return have
    os.environ.update(want)
    _APPLIED["env"] = dict(want)
    return want


def reset_for_tests() -> None:
    _APPLIED.clear()
