"""Robot mesh networking — peer-to-peer presence, state, RPC, and teleoperation.

This package provides the Zenoh-based mesh layer for strands-robots. Each robot
(hardware or simulated) owns a :class:`Mesh` component that broadcasts its
presence, publishes sensor streams, and responds to RPC commands from peers.

Typical usage::

    from strands_robots.mesh import init_mesh

    mesh = init_mesh(robot, peer_id="arm-001")
    if mesh is not None:
        print(mesh.alive)  # True
        print(mesh.peers)  # discovered peers
        mesh.stop()

Submodules
----------
- ``session`` — Shared Zenoh session singleton and peer registry
- ``audit`` — Append-only safety event audit log
- ``core`` — The Mesh class (lifecycle, presence, state, RPC, subscribe)
- ``sensors`` — Extended sensor topic loops (pose, health, IMU, odom, lidar, hand, map)
- ``input`` — InputPublisher / InputReceiver for teleoperation over mesh
"""

from strands_robots.mesh.audit import log_safety_event
from strands_robots.mesh.core import (
    _LOCAL_ROBOTS,
    _LOCAL_ROBOTS_LOCK,
    Mesh,
    get_local_robots,
    init_mesh,
)
from strands_robots.mesh.input import InputPublisher, InputReceiver
from strands_robots.mesh.session import (
    clear_peers,
    current_session,
    get_peers,
    get_session,
    prune_peers,
    put,
    release_session,
    session_alive,
    update_peer,
)

__all__ = [
    # Core types
    "Mesh",
    "InputPublisher",
    "InputReceiver",
    # Factory & registry
    "init_mesh",
    "get_local_robots",
    # Session helpers (re-exported from .session for convenience)
    "put",
    "get_session",
    "release_session",
    "current_session",
    "session_alive",
    # Peer registry
    "get_peers",
    "update_peer",
    "clear_peers",
    "prune_peers",
    # Safety
    "log_safety_event",
    # Private (exposed for test patching only)
    "_LOCAL_ROBOTS",
    "_LOCAL_ROBOTS_LOCK",
]
