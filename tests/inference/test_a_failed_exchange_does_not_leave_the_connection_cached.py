"""A connection is only reusable if the last exchange on it finished.

``RemotePolicy._request`` sends a message and then reads one reply. When that
read does not arrive - the ``request_timeout`` deadline expires on a slow remote
VLA, the ordinary reason that budget exists - the reply is still produced and
still queued on the socket. Nothing closed the connection and nothing cleared
``self._ws``, so ``_ensure_connected`` short-circuited on it and the next
request read the *previous* request's answer, and the one after that read the
one before it, for the life of the connection.

Neither layer could see it. ``_request`` inspected the reply only for
``MSG_ERROR``, so a stale action chunk passed the check it does make - a chunk
computed for an observation the robot has already moved past carries exactly the
``MSG_ACTIONS`` type the caller expects. And a ``MSG_OK`` (what ``reset``,
``set_robot_state_keys`` and ``set_control_frequency`` are answered with) read
as an action chunk becomes ``[]`` through ``reply.get("actions", [])``, which is
indistinguishable from a policy that chose to emit nothing.

The connect sequence had the same shape one step earlier. ``_connect`` assigns
``self._ws`` before it reads the handshake, so the two ``ConnectionError``\\s it
raises - a first frame that is not ``MSG_READY``, and a protocol version this
client does not speak - left a live socket cached behind a refusal. The
mismatch was reported once and the next request then served on the very
connection the client had just declared unusable.

One rule covers both: a connection whose exchange did not complete is in an
unknown position, so it is discarded rather than reused. The next
``_ensure_connected`` opens a fresh one and replays the pending config, which is
the path that already existed for a connection that was never opened.

Companion to ``test_connect_replay_serialises_with_other_wire_users``. That
file holds the *concurrent* hazard - two threads overlapping one read. This one
is sequential: a single-threaded caller desynchronises its own stream, so the
lock cannot help and did not.
"""

import ast
import inspect
import json
import textwrap
import threading
import time
from typing import Any

import pytest

from strands_robots.inference import PolicyServer, RemotePolicy, protocol
from strands_robots.policies import MockPolicy

pytest.importorskip("websockets")

#: Seconds the server holds a parked reply before giving up on it. This is what
#: makes a missed read *deterministic* rather than a race: the reply is provably
#: still withheld when the client's deadline expires, so every budget below that
#: has to expire must stay well under this.
PARK_HOLD = 10.0

#: Deadline for a call that must *complete*. A successful call returns as soon as
#: its reply lands, so this budget is never waited out on a passing run and a
#: generous value costs one nothing - it buys immunity from the scheduling stalls
#: a loaded runner adds to a loopback round trip, which are what a tight budget
#: here measures instead of the behaviour under test. Generous but bounded: a
#: genuinely hung call must still report ``TimeoutError`` from this client rather
#: than be killed by the suite's ``--timeout=120``, which reports nothing useful.
ROUND_TRIP_TIMEOUT = 5.0

#: Deadline for the one read per scenario that must *miss* its reply. Unlike
#: ``ROUND_TRIP_TIMEOUT`` this one is waited out in full on every run, so it is
#: the budget that costs wall clock and the only reason to keep a value small.
PARKED_READ_TIMEOUT = 0.3

#: The concurrent-discard scenario cannot separate the two budgets above: thread
#: A's completing call and thread B's missed read share one client, and
#: ``_request`` takes its deadline from instance state with no per-call override,
#: so one value serves both. Bounded on both sides - above a round trip on a
#: loaded runner (A must finish), below ``THREAD_JOIN_TIMEOUT`` (B must be seen
#: to time out).
SHARED_DISCARD_TIMEOUT = 3.0

#: How long to wait for a thread in that scenario to finish. Exceeds the deadline
#: the thread is waiting out, and stays under ``PARK_HOLD`` so a join cannot
#: outlive the park it depends on.
THREAD_JOIN_TIMEOUT = 9.0


class TaggedPolicy(MockPolicy):  # type: ignore[misc]
    """Echoes each observation's own marker back in the action chunk.

    Tagging the chunk is what makes staleness observable at all: a desynchronised
    stream returns a *well-formed* chunk of the right type, so only its content
    says which observation it was computed for.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.park = threading.Event()
        self.parked = threading.Event()
        self.release = threading.Event()
        self.seen: list[str] = []

    async def get_actions(  # type: ignore[override]
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        marker = observation_dict.get("marker", "?")
        self.seen.append(marker)
        if self.park.is_set():
            # One-shot: park this reply past the client's deadline, then serve
            # every later request promptly.
            self.park.clear()
            self.parked.set()
            self.release.wait(timeout=PARK_HOLD)
        return [{"tag": marker}]


@pytest.fixture
def tagged() -> Any:
    """A live server whose policy tags chunks, plus a client pointed at it."""
    policy = TaggedPolicy()
    server = PolicyServer(policy=policy, host="127.0.0.1", port=0).start()
    client = RemotePolicy(host="127.0.0.1", port=server.port, request_timeout=ROUND_TRIP_TIMEOUT)
    try:
        yield policy, server, client
    finally:
        policy.release.set()
        client.close()
        server.stop()


def _strand(client: RemotePolicy, policy: TaggedPolicy) -> Any:
    """Leave one reply undelivered: park it, let the read time out, then free it.

    Returns the connection the client was using when the read timed out, so a
    caller can ask what became of it.

    Only the parked read is put on the short budget. The warm call above it has
    to *complete* - it is what caches the connection this helper returns - so it
    keeps the fixture's round-trip budget, and the narrowing is unwound in a
    ``finally`` because callers go on using this client after the strand: a
    failed assertion inside the window would otherwise leave the short deadline
    on it and relocate this flake into whichever test ran next.
    """
    client.get_actions_sync({"marker": "warm"}, "")
    stranded = client._ws
    policy.park.set()
    client.request_timeout = PARKED_READ_TIMEOUT
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            client.get_actions_sync({"marker": "OBS-A"}, "")
        # The read expired on its own deadline, not because the server stopped
        # parking: past PARK_HOLD the reply is delivered and there is no missed
        # read left for these scenarios to be about.
        assert time.monotonic() - started < PARK_HOLD, "the parked reply was released before the read gave up"
    finally:
        client.request_timeout = ROUND_TRIP_TIMEOUT
    policy.release.set()
    # Give the server time to finish producing the reply nobody read.
    time.sleep(0.5)
    return stranded


def _serve_first_frame(first_frame: dict[str, Any]) -> Any:
    """Serve a WebSocket whose first frame is ``first_frame``, then only ``MSG_OK``.

    Stands in for a peer this client refuses to talk to - a newer server, or one
    that is not a ``PolicyServer`` at all. The shipped server always opens with
    ``MSG_READY``, so its own handshake can never be rejected and it cannot
    reach the branch under test.
    """
    from websockets.sync.server import serve

    def handler(websocket: Any) -> None:
        websocket.send(json.dumps(first_frame))
        for _raw in websocket:
            websocket.send(json.dumps({"type": protocol.MSG_OK}))

    server = serve(handler, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class TestThePremisesThisRestsOn:
    """Facts about the protocol and the lock that the rule depends on."""

    def test_only_get_actions_is_answered_with_an_action_chunk(self) -> None:
        """Three of the four request types are answered ``MSG_OK``.

        So a stream that slips by one hands an action-chunk reader a reply that
        carries no ``actions`` key at all.
        """
        source = inspect.getsource(PolicyServer._dispatch)
        assert source.count("protocol.MSG_OK") >= 3
        assert "protocol.MSG_ACTIONS" in source

    def test_a_reply_with_no_actions_key_reads_as_an_empty_chunk(self) -> None:
        """``reply.get("actions", [])`` cannot distinguish absent from empty."""
        ok_reply: dict[str, Any] = {"type": protocol.MSG_OK}
        assert ok_reply.get("actions", []) == []

    def test_the_server_serves_every_request_in_order(self, tagged: Any) -> None:
        """Holds either way, and rules out a server that dropped or reordered.

        The slip is entirely the client's bookkeeping: the server received and
        answered all three observations, in order, both before and after the
        fix. Without this the stale chunk could be read as a server defect.
        """
        policy, _server, client = tagged
        _strand(client, policy)
        try:
            client.get_actions_sync({"marker": "OBS-B"}, "")
        except Exception:  # noqa: BLE001 - pre-fix this call reads a stale reply
            pass
        assert policy.seen == ["warm", "OBS-A", "OBS-B"]

    def test_every_budget_that_must_expire_does_so_while_the_reply_is_still_parked(self) -> None:
        """The ordering that makes a missed read deterministic rather than a race.

        These scenarios need one read to *not* arrive, and what withholds it is
        the server parking the reply for ``PARK_HOLD``. So each budget that has
        to expire must expire inside that window: past it the reply is delivered,
        the read succeeds, and the test goes green while asserting nothing about
        the discard it names. That failure is silent - a delivered reply is
        indistinguishable from a passing test - which is why it is pinned here
        rather than left to the arithmetic between two unrelated literals.

        The motive that produced the original single budget (keep the suite
        quick) applies just as well to shortening ``PARK_HOLD``, so the guard is
        on the relationship, not on any one value.

        This pins the *meaning* of the parked scenarios. It cannot pin their
        freedom from timing flakes: that a completing call fits inside
        ``ROUND_TRIP_TIMEOUT`` depends on the runner and is not statically
        assertable, which is the separate reason that budget is generous.
        """
        assert PARKED_READ_TIMEOUT < PARK_HOLD, "the strand's read would be answered instead of missed"
        assert SHARED_DISCARD_TIMEOUT < PARK_HOLD, "thread B would be answered instead of timing out"

        # A join has to outlast the deadline whose expiry it is waiting to
        # observe, or B is still blocked when its result is read; and it has to
        # stay inside the park, or it outlives what holds the reply back.
        assert SHARED_DISCARD_TIMEOUT < THREAD_JOIN_TIMEOUT < PARK_HOLD

        # The two budgets are separate so that the one paid on every run stays
        # small while the one that is never waited out can be generous.
        assert PARKED_READ_TIMEOUT < ROUND_TRIP_TIMEOUT

    def test_the_wire_lock_is_not_reentrant(self) -> None:
        """Which is why the discard is inlined instead of routed through ``close``.

        ``close`` takes ``self._lock``; every caller of the discard already
        holds it.
        """
        client = RemotePolicy(host="127.0.0.1", port=1)
        assert isinstance(client._lock, type(threading.Lock()))
        assert client._lock.acquire(blocking=False) is True
        try:
            assert client._lock.acquire(blocking=False) is False
        finally:
            client._lock.release()


class TestAnUndeliveredReplyIsNotHandedToTheNextRequest:
    """The sequential desync: face one of the same rule."""

    def test_a_timed_out_request_discards_the_connection(self, tagged: Any) -> None:
        policy, _server, client = tagged
        _strand(client, policy)
        assert client._ws is None

    def test_the_abandoned_socket_is_closed_not_merely_forgotten(self, tagged: Any) -> None:
        """Clearing ``self._ws`` alone would leak the connection.

        A socket that is forgotten but left open holds a server-side handler
        thread and a file descriptor for the life of the process. A rollout that
        overruns its request budget once per episode would leak one per episode.
        """
        policy, _server, client = tagged
        client.get_actions_sync({"marker": "warm"}, "")
        assert client._ws is not None and client._ws.close_code is None
        stranded = _strand(client, policy)
        assert client._ws is None
        assert stranded.close_code is not None

    def test_the_next_request_gets_its_own_chunk_not_the_previous_one(self, tagged: Any) -> None:
        """The headline: an arm must not execute a chunk for a stale observation."""
        policy, _server, client = tagged
        _strand(client, policy)
        chunk = client.get_actions_sync({"marker": "OBS-B"}, "")
        assert chunk == [{"tag": "OBS-B"}]

    def test_the_slip_does_not_persist_across_later_requests(self, tagged: Any) -> None:
        """Left uncorrected the offset survived for the life of the connection."""
        policy, _server, client = tagged
        _strand(client, policy)
        assert client.get_actions_sync({"marker": "OBS-B"}, "") == [{"tag": "OBS-B"}]
        assert client.get_actions_sync({"marker": "OBS-C"}, "") == [{"tag": "OBS-C"}]

    def test_a_reset_s_acknowledgement_is_not_read_as_an_action_chunk(self, tagged: Any) -> None:
        """The silent shape: ``MSG_OK`` became ``[]``, a legitimate empty chunk."""
        policy, _server, client = tagged
        _strand(client, policy)
        client.reset(seed=7)
        assert client.get_actions_sync({"marker": "OBS-D"}, "") == [{"tag": "OBS-D"}]


class TestARejectedHandshakeIsNotCachedAsUsable:
    """The connect sequence: face two of the same rule."""

    @pytest.mark.parametrize(
        ("label", "first_frame", "expected"),
        [
            pytest.param(
                "newer-protocol",
                {"type": protocol.MSG_READY, "protocol_version": protocol.PROTOCOL_VERSION + 1, "metadata": {}},
                "protocol version mismatch",
                id="a-server-speaking-a-newer-protocol",
            ),
            pytest.param(
                "foreign-frame",
                {"type": "hello", "protocol_version": protocol.PROTOCOL_VERSION},
                "expected a 'ready' handshake",
                id="a-peer-that-is-not-a-policy-server",
            ),
        ],
    )
    def test_the_refused_connection_is_not_left_cached(
        self, label: str, first_frame: dict[str, Any], expected: str
    ) -> None:
        del label
        server = _serve_first_frame(first_frame)
        port = server.socket.getsockname()[1]
        client = RemotePolicy(host="127.0.0.1", port=port, connect_timeout=2.0, request_timeout=2.0)
        try:
            with pytest.raises(ConnectionError, match=expected):
                client.get_actions_sync({"a": 1.0}, "")
            assert client._ws is None
        finally:
            client.close()
            server.shutdown()

    @pytest.mark.parametrize(
        ("first_frame", "expected"),
        [
            pytest.param(
                {"type": protocol.MSG_READY, "protocol_version": protocol.PROTOCOL_VERSION + 1, "metadata": {}},
                "protocol version mismatch",
                id="a-server-speaking-a-newer-protocol",
            ),
            pytest.param(
                {"type": "hello", "protocol_version": protocol.PROTOCOL_VERSION},
                "expected a 'ready' handshake",
                id="a-peer-that-is-not-a-policy-server",
            ),
        ],
    )
    def test_the_refusal_is_repeated_rather_than_bypassed(self, first_frame: dict[str, Any], expected: str) -> None:
        """A rejected peer stays rejected.

        Cached, the second call served on the connection the client had just
        refused - and answered ``[]``, which reads as a policy with nothing to
        say rather than as a peer it cannot talk to.
        """
        server = _serve_first_frame(first_frame)
        port = server.socket.getsockname()[1]
        client = RemotePolicy(host="127.0.0.1", port=port, connect_timeout=2.0, request_timeout=2.0)
        try:
            for _attempt in (1, 2):
                with pytest.raises(ConnectionError, match=expected):
                    client.get_actions_sync({"a": 1.0}, "")
        finally:
            client.close()
            server.shutdown()


class TestWhatAFailedExchangeDoesNotMean:
    """Boundaries: the rule must not throw away a connection that is in step."""

    def test_a_server_side_error_reply_keeps_the_connection(self, tagged: Any) -> None:
        """``MSG_ERROR`` *is* this request's reply, so the stream is in step.

        The server marshals any dispatch failure back and carries on serving;
        discarding here would drop a healthy connection on every recoverable
        error.
        """
        _policy, _server, client = tagged
        client.get_actions_sync({"marker": "warm"}, "")
        with client._lock:
            with pytest.raises(RuntimeError, match="remote policy server error"):
                client._request({"type": "no-such-message-type"})
        assert client._ws is not None

    def test_the_connection_still_serves_after_an_error_reply(self, tagged: Any) -> None:
        _policy, _server, client = tagged
        client.get_actions_sync({"marker": "warm"}, "")
        with client._lock:
            with pytest.raises(RuntimeError):
                client._request({"type": "no-such-message-type"})
        assert client.get_actions_sync({"marker": "after"}, "") == [{"tag": "after"}]

    def test_a_completed_request_leaves_the_connection_cached(self, tagged: Any) -> None:
        """The ordinary path must not pay a reconnect per request."""
        _policy, _server, client = tagged
        client.get_actions_sync({"marker": "warm"}, "")
        before = client._ws
        client.get_actions_sync({"marker": "again"}, "")
        assert client._ws is before is not None

    def test_a_successful_handshake_is_cached(self, tagged: Any) -> None:
        _policy, _server, client = tagged
        client.get_actions_sync({"marker": "warm"}, "")
        assert client._ws is not None

    def test_discarding_is_not_giving_up_the_next_call_reconnects(self, tagged: Any) -> None:
        """Holds either way, and it is what stops the fix being a regression.

        A rule that dropped the connection and left it dropped would trade a
        silent wrong answer for a client that never speaks again.
        """
        policy, _server, client = tagged
        _strand(client, policy)
        client.get_actions_sync({"marker": "OBS-B"}, "")
        assert client._ws is not None

    def test_close_is_still_idempotent(self, tagged: Any) -> None:
        _policy, _server, client = tagged
        client.get_actions_sync({"marker": "warm"}, "")
        client.close()
        client.close()
        assert client._ws is None


class TestTheDiscardCoversMoreThanAnExceptionWould:
    """Structural: how the bookkeeping is written is part of the contract."""

    def test_the_request_discard_is_a_finally_not_an_except(self) -> None:
        """A ``finally`` covers a ``BaseException`` too.

        A cancellation landing between the send and the receive leaves the same
        undelivered reply behind as a timeout does, and an ``except Exception``
        would step over it. Written as ``except BaseException`` this would also
        join the tree's ``py/catch-base-exception`` census for no gain.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(RemotePolicy._request)))
        tries = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        assert len(tries) == 1
        assert tries[0].handlers == []
        assert tries[0].finalbody

    def test_the_connect_discard_is_a_finally_not_an_except(self) -> None:
        tree = ast.parse(textwrap.dedent(inspect.getsource(RemotePolicy._connect)))
        guards = [node for node in ast.walk(tree) if isinstance(node, ast.Try) and not node.handlers and node.finalbody]
        assert len(guards) == 1

    def test_the_discard_does_not_take_the_lock_its_callers_hold(self) -> None:
        """Taking it would deadlock: ``_lock`` is a plain ``Lock``.

        Read from the tree rather than the text: the docstring explains the
        deadlock and so names ``_lock`` itself, which a source-string scan
        cannot tell from a use.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(RemotePolicy._discard_connection)))
        body = [node for node in tree.body[0].body if not isinstance(node, ast.Expr)]  # type: ignore[attr-defined]
        touched = {
            node.attr
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
        }
        assert touched == {"_ws"}

    def test_both_wire_paths_route_through_the_one_discard(self) -> None:
        """One helper, so the two paths cannot drift apart."""
        for method in (RemotePolicy._request, RemotePolicy._connect):
            tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_discard_connection"
            ]
            assert len(calls) == 1, method.__name__


class TestAConcurrentCallerSurvivesADiscardByAnotherThread:
    """The under-the-lock re-check in _get_actions_blocking reconnects when
    another thread discarded the connection while this thread was waiting.

    The race: thread A passes _ensure_connected (sees _ws non-None), then blocks
    on self._lock. Thread B (holding the lock) times out in _request and calls
    _discard_connection, nulling self._ws. Thread A acquires the lock and must
    not crash on a None _ws - it must reconnect and produce a fresh result.
    """

    def test_a_concurrent_caller_reconnects_after_a_discard(self, tagged: Any) -> None:
        """Thread A gets a fresh chunk after thread B discards the connection."""
        policy, _server, client = tagged
        results: dict[str, Any] = {}
        errors: dict[str, Any] = {}

        def thread_b() -> None:
            """Time out, causing the discard."""
            try:
                result = client.get_actions_sync({"marker": "B-timeout"}, "")
                results["B"] = result
            except Exception as exc:
                errors["B"] = exc

        def thread_a() -> None:
            """Enters while B's exchange is in flight, so it waits on the lock B holds."""
            # Deterministic, not a timed guess: the server sets ``parked`` only
            # after it has taken B's request, so B is provably inside
            # ``_request`` holding the lock with ``self._ws`` still live. A sleep
            # past the deadline lands after the discard, where this caller takes
            # the ordinary cold-start path and exercises nothing.
            assert policy.parked.wait(timeout=10.0), "thread B never reached the server"
            try:
                result = client.get_actions_sync({"marker": "A-after-discard"}, "")
                results["A"] = result
            except Exception as exc:
                errors["A"] = exc

        # Warm the connection so _ensure_connected's fast path passes for both.
        client.get_actions_sync({"marker": "warm"}, "")

        # One deadline has to serve both threads here - B must miss its reply and
        # A must complete on the same client, concurrently - so it is set once,
        # after the warm call, and not restored: nothing runs on this client
        # afterwards, and the fixture is per-test.
        client.request_timeout = SHARED_DISCARD_TIMEOUT

        # Park one reply so thread B times out.
        policy.park.set()

        t_b = threading.Thread(target=thread_b)
        t_a = threading.Thread(target=thread_a)
        t_b.start()
        t_a.start()
        t_b.join(timeout=THREAD_JOIN_TIMEOUT)
        # Release the parked reply so the server can serve thread A's request.
        policy.release.set()
        t_a.join(timeout=THREAD_JOIN_TIMEOUT)

        # Thread B timed out (expected).
        assert "B" in errors, f"thread B should have timed out, got result: {results.get('B')}"
        assert isinstance(errors["B"], TimeoutError)

        # Thread A must NOT crash - it should reconnect and get its own chunk.
        assert "A" not in errors, f"thread A crashed: {errors.get('A')}"
        assert results.get("A") == [{"tag": "A-after-discard"}]

    def test_a_request_on_a_discarded_connection_names_the_condition(self, tagged: Any) -> None:
        """The refusal a bare ``assert`` could not make.

        Driven directly, because the caller above now re-checks: this guard is
        reachable only from a future caller that does not, and an
        ``AssertionError`` carrying no message names neither the connection nor
        the sibling that took it away.
        """
        _policy, _server, client = tagged
        client.get_actions_sync({"marker": "warm"}, "")
        client._discard_connection()
        with pytest.raises(ConnectionError, match="discarded after a failed exchange"):
            client._request({"type": protocol.MSG_GET_ACTIONS, "observation": {}, "instruction": "", "kwargs": {}})


class TestEveryWireCallerHoldingTheLockRechecksUnderIt:
    """Derived, so a wire caller added later is held to the rule on arrival.

    ``reset``, ``set_robot_state_keys`` and ``set_control_frequency`` already
    re-checked ``self._ws`` inside ``with self._lock``; they *defer* when it is
    gone, because the connect replay applies the config they carry.
    ``_get_actions_blocking`` was the only lock-holding wire caller that did not
    re-check, and it is the one that cannot defer - it owes the caller a chunk.
    """

    @staticmethod
    def _lock_holding_wire_callers() -> dict[str, str]:
        """Map each method that sends on the wire under the lock to its locked body."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(RemotePolicy)))
        found: dict[str, str] = {}
        for node in tree.body[0].body:  # type: ignore[attr-defined]
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            locked = [
                block
                for block in ast.walk(node)
                if isinstance(block, ast.With)
                and any("self._lock" in ast.unparse(item.context_expr) for item in block.items)
            ]
            if not locked:
                continue
            body = "\n".join(ast.unparse(block) for block in locked)
            if "_request(" in body or "_connect(" in body:
                found[node.name] = body
        return found

    def test_the_survey_reaches_the_callers_it_is_about(self) -> None:
        """Non-vacuity: an empty survey would pass the rule below while saying nothing."""
        callers = self._lock_holding_wire_callers()
        assert {"reset", "set_robot_state_keys", "set_control_frequency", "_get_actions_blocking"} <= set(callers)

    def test_each_one_rechecks_the_connection_under_the_lock(self) -> None:
        """The unlocked check is a fast path, so the locked one is the decision."""
        missing = [name for name, body in self._lock_holding_wire_callers().items() if "self._ws is" not in body]
        assert missing == [], f"sends on the wire under the lock without re-checking it: {missing}"

    def test_the_hot_path_opens_the_connection_rather_than_deferring(self) -> None:
        """Its siblings return; this one owes a chunk, so it connects."""
        assert "self._connect()" in self._lock_holding_wire_callers()["_get_actions_blocking"]

    def test_it_calls_connect_and_not_the_wrapper_that_takes_the_lock(self) -> None:
        """``_ensure_connected`` takes ``self._lock``, which is not reentrant.

        Calling it with the lock already held deadlocks, so the caller uses the
        inner ``_connect``. Pinned because the wrapper is the more natural name
        to reach for.
        """
        assert type(RemotePolicy(host="127.0.0.1", port=1)._lock) is type(threading.Lock())
        assert "_ensure_connected()" not in self._lock_holding_wire_callers()["_get_actions_blocking"]


class TestWhatTheRecheckDoesNotChange:
    """An ordinary concurrent pair still gets its own answers.

    The re-check is on the failure path; nothing about it serialises differently
    or costs a reconnect while the connection is fine.
    """

    def test_two_healthy_callers_each_get_their_own_chunk(self, tagged: Any) -> None:
        """No timeout, no discard: one connection serves both."""
        _policy, _server, client = tagged
        client.get_actions_sync({"marker": "warm"}, "")
        established = client._ws
        results: dict[str, Any] = {}

        def call(marker: str) -> None:
            results[marker] = client.get_actions_sync({"marker": marker}, "")

        threads = [threading.Thread(target=call, args=(marker,)) for marker in ("P", "Q")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20.0)
        assert results == {"P": [{"tag": "P"}], "Q": [{"tag": "Q"}]}
        assert client._ws is established, "a healthy exchange must not cost a reconnect"
