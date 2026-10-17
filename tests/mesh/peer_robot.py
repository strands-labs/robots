"""A device process for the multi-process mesh tests: mounts a simulated SO-101 and waits to be killed.

argv: url tenant device_id [state_hz|-] [watchdog_ms]. Prints ``ready``, then ``hold <monotonic>`` on each hold.
"""

import sys
import time

from strands_robots import Robot
from strands_robots.mesh import Mesh

url, tenant, device_id = sys.argv[1:4]
state_hz = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
watchdog_ms = float(sys.argv[5]) if len(sys.argv) > 5 else 500.0
arm = Robot("so101")
real_hold = arm.body.hold


def hold():
    print(f"hold {time.monotonic():.6f}", flush=True)
    real_hold()


arm.body.hold = hold
mesh = Mesh(url, tenant=tenant)
mesh.mount(arm, device_id=device_id, labels={"role": "arm"}, state_hz=state_hz, watchdog_ms=watchdog_ms)
print("ready", flush=True)
while True:
    time.sleep(1)
