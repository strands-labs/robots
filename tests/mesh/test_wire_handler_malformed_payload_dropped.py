"""Regression tests: the response + safety-estop wire handlers DROP a
malformed or non-dict payload at runtime without crashing or acting on it.

Wire authentication (mTLS + ACL) only proves a sample came from a fleet
member; it does not prove the *body* is well-formed. A CA-signed but buggy
(or hostile) peer can publish non-JSON bytes or a valid-JSON-but-non-dict
body (a bare list / number / string) on ``strands/response`` or
``strands/safety/estop``. Both handlers guard this with

    try:
        data = json.loads(sample.payload.to_bytes().decode())
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return

so a malformed body is dropped silently rather than raising out of the Zenoh
callback (which runs on a transport thread) or being acted upon.

``tests/mesh/test_wire_handler_narrow_except.py`` pins the *source shape* of
that guard (the narrow exception tuple); these tests pin its *runtime
behavior*: the drop branches were previously unexercised, so removing either
guard leaves the tests green there while reintroducing a crash-on-hostile-input
here. Pre-fix verification -- delete the ``except``/``not isinstance`` return
in ``_on_response`` or ``_on_safety_estop``:

* non-JSON body -> ``json.JSONDecodeError`` escapes the handler,
* JSON list body -> ``AttributeError`` from ``[...].get(...)`` escapes,

and every test below fails.
"""

from __future__ import annotations

import json
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from strands_robots.mesh import Mesh


class _FakeRobot:
    """Minimal duck-typed robot; dispatch is never reached in these tests."""

    def __init__(self) -> None:
        self.tool_name_str = "fakebot"


def _sample(raw: bytes) -> Any:
    """A fake zenoh sample whose payload decodes to ``raw`` bytes."""
    sample = MagicMock()
    sample.payload.to_bytes.return_value = raw
    return sample


# Two flavours of malformed body, one per guard branch:
#   * non-JSON text   -> the ``except (... json.JSONDecodeError)`` return
#   * valid JSON list -> the ``if not isinstance(data, dict)`` return
_MALFORMED = {
    "non_json": b"{not valid json",
    "json_non_dict": json.dumps([1, 2, 3]).encode(),
}


@pytest.fixture
def mesh() -> Mesh:
    """A Mesh instance (not started -- handlers are driven directly)."""
    return Mesh(_FakeRobot(), peer_id="peer-a", peer_type="robot")


@pytest.mark.parametrize("flavour", sorted(_MALFORMED))
def test_on_response_drops_malformed_body(mesh: Mesh, flavour: str) -> None:
    """A malformed response body is dropped: no crash, waiter untouched, nothing recorded."""
    turn = "turn-1"
    event = threading.Event()
    with mesh._rpc_lock:
        mesh._pending[turn] = event
        mesh._responses[turn] = []
        mesh._expected_responders[turn] = "peer-b"

    # Must not raise out of the transport callback.
    mesh._on_response(_sample(_MALFORMED[flavour]))

    with mesh._rpc_lock:
        assert mesh._responses[turn] == []
    assert not event.is_set()


@pytest.mark.parametrize("flavour", sorted(_MALFORMED))
def test_on_safety_estop_drops_malformed_body(mesh: Mesh, flavour: str) -> None:
    """A malformed estop body is dropped: no crash and the local lockout stays clear."""
    assert not mesh._estop_lockout.is_set()

    # Must not raise, and must not engage the emergency-stop lockout.
    mesh._on_safety_estop(_sample(_MALFORMED[flavour]))

    assert not mesh._estop_lockout.is_set()
