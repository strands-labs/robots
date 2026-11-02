"""A second peer for the Connect gates: serves a simulated arm (with a wrist camera) over the mesh until killed.

argv: url tenant device_id. Trusts operator names (``Authority(trust_names=True)``), so the page that stopped it may
resume it. Prints ``ready`` once mounted.
"""

import sys
import time

from strands_robots import Robot, Simulation
from strands_robots.mesh import Mesh
from strands_robots.mesh.authority import Authority

url, tenant, device_id = sys.argv[1:4]
world = Simulation()
arm = Robot("so101", sim=world, id=device_id)
world.camera("wrist_view", attach=arm, position=(0.45, -0.35, 0.35), target=(0.0, 0.0, 0.12), size=(320, 240))
mesh = Mesh(url, tenant=tenant, resume_authority=Authority(trust_names=True))
mesh.mount(arm, labels={"site": "bench-b"})
print("ready", flush=True)
while True:
    time.sleep(1)
