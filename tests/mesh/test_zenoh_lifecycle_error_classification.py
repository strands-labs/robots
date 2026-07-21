"""Zenoh lifecycle error classification for the mesh session and node.

Best-effort Zenoh lifecycle operations (``open`` / ``declare_subscriber`` /
``undeclare`` / ``close``) must catch the errors real ``eclipse-zenoh`` raises
so a transport-side failure cannot escape partial-failure cleanup or
``Mesh.stop()``. The native ``zenoh.ZError`` subclasses ``Exception`` directly
- NOT ``RuntimeError`` - so it has to be named explicitly in the ``except``
tuple. :func:`strands_robots.mesh.session.zenoh_error_types` centralises that
tuple; these tests pin the contract and the fail-soft teardown behaviour.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh.core import Mesh
from strands_robots.mesh.session import zenoh_error_types


def test_error_tuple_always_includes_transport_builtins():
    types = zenoh_error_types()
    assert isinstance(types, tuple)
    for builtin in (RuntimeError, OSError, ConnectionError):
        assert builtin in types
    # Programmer errors must NOT be swallowed by best-effort cleanup.
    assert TypeError not in types
    assert AttributeError not in types


def test_error_tuple_names_zenoh_zerror_explicitly():
    zenoh = pytest.importorskip("zenoh")
    zerr = getattr(zenoh, "ZError", None)
    if not isinstance(zerr, type):
        pytest.skip("zenoh.ZError is not a real class in this environment")
    # The whole reason ZError is named explicitly: it is an Exception subclass,
    # NOT a RuntimeError, so (RuntimeError, OSError) alone would let it escape.
    assert not issubclass(zerr, RuntimeError)
    assert zerr in zenoh_error_types()


def _bare_mesh_for_stop(subs=None, safety_publishers=None):
    """A ``Mesh`` carrying only the attributes ``stop()`` touches.

    Built with ``Mesh.__new__`` (the shared mesh-unit-test pattern) so no zenoh
    transport or live robot is required.
    """
    mesh = Mesh.__new__(Mesh)
    mesh.peer_id = "test__arm"
    mesh._running = True
    mesh._lifecycle_lock = threading.RLock()
    mesh._stop_event = MagicMock()
    mesh._subs_lock = threading.RLock()
    mesh._subs = list(subs or [])
    mesh._user_subs = set()
    mesh._inbox_lock = threading.RLock()
    mesh.inbox = []
    mesh._safety_publishers_lock = threading.RLock()
    mesh._safety_publishers = dict(safety_publishers or {})
    mesh._rpc_lock = threading.RLock()
    mesh._pending = {}
    mesh._responses = {}
    mesh._has_session_ref = False
    return mesh


def _zerror_raiser():
    """A publisher/subscriber whose ``undeclare`` raises a real ``zenoh.ZError``."""
    zenoh = pytest.importorskip("zenoh")
    zerr = getattr(zenoh, "ZError", None)
    if not isinstance(zerr, type):
        pytest.skip("zenoh.ZError is not a real class in this environment")
    obj = MagicMock()
    obj.undeclare.side_effect = zerr("simulated broker drop during teardown")
    return obj


def test_stop_is_fail_soft_when_safety_publisher_undeclare_raises_zerror():
    pub = _zerror_raiser()
    mesh = _bare_mesh_for_stop(safety_publishers={"estop": pub})

    mesh.stop()  # pre-fix: ZError escapes the (RuntimeError, OSError) catch

    pub.undeclare.assert_called_once()
    assert mesh.alive is False
    assert mesh._safety_publishers == {}


def test_stop_is_fail_soft_when_subscriber_undeclare_raises_zerror():
    sub = _zerror_raiser()
    mesh = _bare_mesh_for_stop(subs=[sub])

    mesh.stop()

    sub.undeclare.assert_called_once()
    assert mesh.alive is False
    assert mesh._subs == []


def test_stop_does_not_swallow_programmer_error_from_undeclare():
    # A TypeError is a bug, not a transport failure: it must surface rather than
    # be silently swallowed by best-effort teardown.
    sub = MagicMock()
    sub.undeclare.side_effect = TypeError("undeclare() takes no arguments")
    mesh = _bare_mesh_for_stop(subs=[sub])

    with pytest.raises(TypeError):
        mesh.stop()
