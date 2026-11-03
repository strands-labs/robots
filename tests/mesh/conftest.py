import itertools
import os
import socket
import uuid

import pytest

_SEQ = itertools.count()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def tenant() -> str:
    return f"t{os.getpid()}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(params=["mem", "zenoh"])
def pair(request):
    """Two connected transports (a listener and a peer), closed after the test."""
    from strands_robots.mesh.transports import open_transport

    if request.param == "mem":
        name = f"bus{next(_SEQ)}-{uuid.uuid4().hex[:6]}"
        a, b = open_transport(f"mem://{name}"), open_transport(f"mem://{name}")
    else:
        pytest.importorskip("zenoh")
        port = free_port()
        a = open_transport(f"zenoh://?listen=tcp/127.0.0.1:{port}")
        b = open_transport(f"zenoh://?connect=tcp/127.0.0.1:{port}")
    yield a, b
    b.close()
    a.close()


@pytest.fixture(params=["mem", "zenoh", "iot-fake"])
def mesh_pair(request, tenant):
    """Two Meshes (a listener and a peer) in one tenant, closed after the test."""
    from strands_robots.mesh import Mesh

    if request.param == "mem":
        name = f"mesh{next(_SEQ)}-{uuid.uuid4().hex[:6]}"
        ua = ub = f"mem://{name}"
    elif request.param == "iot-fake":
        name = f"broker{next(_SEQ)}-{uuid.uuid4().hex[:6]}"
        ua = ub = f"iot://fake?broker={name}"
    else:
        pytest.importorskip("zenoh")
        port = free_port()
        ua, ub = f"zenoh://?listen=tcp/127.0.0.1:{port}", f"zenoh://?connect=tcp/127.0.0.1:{port}"
    a, b = Mesh(ua, tenant=tenant), Mesh(ub, tenant=tenant)
    yield a, b
    b.close()
    a.close()
