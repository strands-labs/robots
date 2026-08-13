#!/usr/bin/env python3
"""Join the robot mesh and discover peers on the local network.

Goal: Show that Robot() with mesh=True gives automatic peer discovery via
Zenoh. Every robot that joins the mesh is visible to every other - no DHCP,
no config server, no manual IP lists.

Dependencies: pip install "strands-robots[sim-mujoco,mesh]"
Expected output: Prints local robot info and discovered peer list.
Runtime: ~3 seconds (the script exits; it releases the mesh session).

Note: Set STRANDS_MESH_LOCAL_DEV=1 to skip TLS for local development.
      Set STRANDS_MESH=0 to disable mesh entirely (the example still runs
      but reports empty peer lists).
"""

import os

os.environ.setdefault("STRANDS_MESH_LOCAL_DEV", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

from strands_robots import Robot
from strands_robots.mesh import get_local_robots, get_peers

# Robot with mesh=True (default) auto-joins the mesh on creation. The factory
# also builds the world and adds the "so100" robot, so no create_world/add_robot
# is needed. Use mesh=False in CI or when Zenoh is unavailable.
use_mesh = os.environ.get("STRANDS_MESH", "true").lower() != "0"
sim = Robot("so100", mesh=use_mesh, peer_id="example-arm-01")

# Query the mesh - see who is online.
local = get_local_robots()
peers = get_peers()

print(f"Local robots in this process: {list(local.keys())}")
print(f"Discovered mesh peers: {len(peers)}")

# Each peer entry is a dict with peer_id, peer_type, capabilities, etc.
for peer in peers:
    print(f"  {peer.get('peer_id', '?')}: type={peer.get('peer_type', '?')}")

# Cleanup. Release the Zenoh session through the attribute the Robot factory
# assigns (`sim.mesh`) - the same one strands_robots.robot's own teardown reads.
# The session runs on non-daemon threads, so a cleanup that silently reads a
# name the SDK never sets leaves this script running forever.
mesh = getattr(sim, "mesh", None)
if mesh:
    mesh.stop()
