"""A device process for the multi-process mesh tests: mounts one fake arm and waits to be killed."""

import sys
import time

from strands_robots.mesh import Mesh


class Arm:
    def describe(self):
        return {"name": "arm", "device_type": "arm", "dof": 6}


url, tenant, device_id = sys.argv[1:4]
mesh = Mesh(url, tenant=tenant)
mesh.mount(Arm(), device_id=device_id, labels={"role": "arm"})
print("ready", flush=True)
while True:
    time.sleep(1)
