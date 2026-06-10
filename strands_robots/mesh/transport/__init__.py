"""Mesh transport layer — pluggable backends behind a single Protocol.

Exports the :class:`MeshTransport` Protocol, both concrete backends
(:class:`ZenohTransport`, :class:`IotMqttTransport`), and the process-wide
:func:`get_transport` / :func:`release_transport` factory selected by
``STRANDS_MESH_BACKEND``.

See :mod:`strands_robots.mesh.transport.base` for the protocol details and
:mod:`strands_robots.mesh.transport.factory` for the selection rules.
"""

from strands_robots.mesh.transport.base import (
    MeshTransport,
    Sample,
    SubHandle,
)
from strands_robots.mesh.transport.bridge_transport import BridgeTransport
from strands_robots.mesh.transport.factory import (
    current_backend,
    current_transport,
    get_transport,
    release_transport,
)
from strands_robots.mesh.transport.iot_transport import IotMqttTransport
from strands_robots.mesh.transport.zenoh_transport import ZenohTransport

__all__ = [
    "MeshTransport",
    "Sample",
    "SubHandle",
    "ZenohTransport",
    "IotMqttTransport",
    "BridgeTransport",
    "get_transport",
    "release_transport",
    "current_transport",
    "current_backend",
]
