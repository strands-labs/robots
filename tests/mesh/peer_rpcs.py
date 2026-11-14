"""A device process for the device-rpc tests: a simulated SO-101 with two device rpcs, or (``real``) a memory-driver
body mounted with no watchdog, so it is observe-only. argv: url tenant device_id [real]. Prints ``ready``, then
``gripper <width>`` each time the gripper rpc runs."""

import sys
import time

from strands_robots import Robot
from strands_robots.mesh import Mesh, device_rpc

url, tenant, device_id = sys.argv[1:4]
real = sys.argv[4:5] == ["real"]
arm = Robot("so101", driver="memory", twin=False) if real else Robot("so101")


def gripper(width: float, by: str | None = None) -> dict:
    """Open the gripper to ``width`` (0..1)."""
    print(f"gripper {width} by={by}", flush=True)
    return {"width": width}


def battery() -> dict:
    """The battery level."""
    return {"percent": 87}


mesh = Mesh(url, tenant=tenant)
rpcs = {"gripper": gripper, "battery": device_rpc(battery, write=False)}
mesh.mount(arm, device_id=device_id, watchdog_ms=None if real else 300.0, rpcs=rpcs)
print("ready", flush=True)
while True:
    time.sleep(1)
