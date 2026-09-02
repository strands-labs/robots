"""A :class:`PolicyServer` told to stop must stop serving the clients it has.

Closing the listening socket is half a teardown. A client that is already
connected is served on a thread of its own, so a teardown that shuts only the
listener down returns while that client goes on streaming observations in and
receiving action chunks back: the wrapped policy is still being invoked, and on a
robot it is still driving the arm, after the caller was told the server stopped.

That is what ``websockets.sync.server.Server.shutdown()`` did through 16.x, and
measured on 16.1.1 the two cells below fail on these same unchanged sources -
``stop()`` returned in 0.18ms and the same open connection was answered with
actions 19 more times over the following second. From 17.0 ``shutdown()`` closes
the connections it accepted with code 1001 and returns only once every connection
handler has terminated, so the property holds; the package declares that release
as its floor, and
``tests/test_websockets_floor_ships_the_imported_api.py`` owns the floor. These
cells are the other half of that record: they grade the property a client
observes, so a downgrade of the floor - or a teardown that stops calling
``shutdown()`` - fails here rather than on a bench.

Nothing else covers it, because the lifecycle tests grade the server's own
*state*: ``test_stop_is_idempotent`` and ``test_context_manager_starts_and_stops``
in ``test_policy_server_lifecycle.py`` assert ``_server is None``, which is
equally true of a server that is still serving. That is the same gap
``test_policy_server_shutdown_does_not_kill_the_serving_thread.py`` records for
the other half of this teardown (a dead accept loop also leaves ``_server``
cleared), so these cells assert on what a *client* observes instead.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from strands_robots.inference import PolicyServer
from strands_robots.inference.client import RemotePolicy
from strands_robots.policies.mock import MockPolicy

#: How long the blocking policy below stays inside one inference call. Long
#: enough that a teardown returning early is unambiguous, short enough to pay for
#: in a unit suite.
_INFERENCE_S = 0.5


def _wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    """Poll ``predicate`` until true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _connected_client(port: int) -> RemotePolicy:
    """A connected client that has exchanged one action request already."""
    client = RemotePolicy(host="127.0.0.1", port=port)
    client.set_robot_state_keys(["joint_0"])
    assert client.get_actions_sync({"joint_0": 0.0}, ""), "the server did not serve the client before teardown"
    return client


def _served_after_teardown(client: RemotePolicy) -> bool:
    """Whether ``client``'s existing connection is still answered with actions."""
    try:
        return bool(client.get_actions_sync({"joint_0": 0.0}, ""))
    except Exception:  # noqa: BLE001 - any refusal is the pass condition
        return False


class _SlowPolicy(MockPolicy):
    """A policy whose inference call takes :data:`_INFERENCE_S` to return.

    Models the handler the teardown has to account for: one inside a long
    inference call - a big VLA step, a stalled remote GPU - which cannot notice a
    closed connection until that call returns, and which has an action chunk to
    send when it does.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.left = threading.Event()

    def get_actions_sync(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Signal that inference started, take a while, then signal that it ended."""
        self.entered.set()
        try:
            time.sleep(_INFERENCE_S)
            return super().get_actions_sync(observation_dict, instruction, **kwargs)
        finally:
            self.left.set()


def test_a_connection_open_at_stop_is_no_longer_served() -> None:
    """``stop()`` ends the connections it accepted, not just the listener."""
    server = PolicyServer(policy_provider="mock", port=0).start()
    client = _connected_client(server.port)
    try:
        server.stop()

        assert not _served_after_teardown(client), (
            "a client connected when stop() was called was still served afterwards: "
            "the policy is still being invoked on a server reported stopped"
        )
    finally:
        client.close()
        server.stop()


def test_a_connection_open_when_serve_returns_is_no_longer_served() -> None:
    """The foreground ``serve()`` door carries the same obligation as ``stop()``."""
    server = PolicyServer(policy_provider="mock", port=0)
    serving = threading.Thread(target=server.serve, daemon=True)
    serving.start()
    try:
        assert _wait_until(lambda: server._server is not None), "serve() never bound"
        client = _connected_client(server.port)
        # serve() owns the socket in its own `with` block, so this is how it is
        # asked to return - the same call stop() makes.
        assert server._server is not None
        server._server.shutdown()
        serving.join(timeout=5.0)
        assert not serving.is_alive(), "serve() did not return"

        assert not _served_after_teardown(client), (
            "a client connected when serve() returned was still served afterwards"
        )
    finally:
        client.close()


def test_stop_does_not_return_while_an_inference_can_still_emit_a_chunk() -> None:
    """A stop landing mid-inference waits for that call, rather than racing it.

    The teardown closes the connection immediately, but the handler is inside the
    wrapped policy and holds an action chunk it has not produced yet. ``stop()``
    returning first would hand the caller a stopped server with one more chunk
    still to come, which on a robot is one more motion after the operator asked
    for none.
    """
    policy = _SlowPolicy()
    server = PolicyServer(policy=policy, port=0).start()
    client = RemotePolicy(host="127.0.0.1", port=server.port)
    client.set_robot_state_keys(["joint_0"])
    asking = threading.Thread(target=_served_after_teardown, args=(client,), daemon=True)
    asking.start()
    try:
        assert policy.entered.wait(timeout=5.0), "the handler never reached inference"

        started = time.monotonic()
        server.stop()
        elapsed = time.monotonic() - started

        assert policy.left.is_set(), (
            "stop() returned while the wrapped policy was still inside an inference call, "
            "so a chunk it had not produced yet could still have been sent"
        )
        # stop() is called within a millisecond or two of `entered` being set, so
        # nearly the whole inference call is still ahead of it. Half of it is the
        # margin: enough that a teardown which waited is unmistakable, loose
        # enough not to grade the runner's scheduler.
        assert elapsed >= _INFERENCE_S / 2, (
            f"stop() returned in {elapsed:.3f}s, less than half the {_INFERENCE_S}s inference it "
            "landed in the middle of, so it cannot have waited for the handler"
        )
        assert not _served_after_teardown(client), "the connection was still served after stop() returned"
    finally:
        asking.join(timeout=10.0)
        client.close()
