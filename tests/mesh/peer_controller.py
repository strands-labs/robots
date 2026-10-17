"""A controller process: drives one device at 30 Hz until killed. argv: url tenant device_id."""

import sys
import time

from strands_robots.mesh import Mesh

url, tenant, device_id = sys.argv[1:4]
mesh = Mesh(url, tenant=tenant)
mesh.discover(device_id, min_count=1, timeout=5.0)
i = 0
while True:
    mesh.invoke(device_id, "act", action={"shoulder_pan": 0.2 + 0.1 * ((i // 30) % 2)})
    if i == 10:
        print("moving", flush=True)
    i += 1
    time.sleep(1 / 30)
