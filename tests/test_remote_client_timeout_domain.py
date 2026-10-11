"""One timeout domain for the remote-inference client, and why ``inf`` is in it.

:class:`~strands_robots.inference.RemotePolicy` (WebSocket) takes
``connect_timeout`` and ``request_timeout``, and used to store what it was
handed and pass it to the transport unexamined. The constructor already refused
its *other* numeric parameters - ``port`` via ``tcp_port_error``,
``actions_per_step`` via ``chunk_count_error`` - so the two timeouts were the
knobs left behind on it.

What made that worse than a late crash is where the failure surfaced. The
client wraps the first transport failure in a ``ConnectionError`` that names the
server and tells the operator to start one:

    RemotePolicy could not reach a PolicyServer at ws://127.0.0.1:8765.
    Start one first, e.g.: python -m strands_robots.inference.server ...

Measured against a **live, reachable** server (``websockets`` 17.0.1), the
values that produce exactly that message are ``0``, ``0.0``, ``-1`` and ``True``
- the transport times out at once, ``TimeoutError`` is inside the clause that
composes the message, and the operator is pointed at the one thing that was not
wrong. ``nan``, ``inf`` and a numeric string instead escaped that clause as a
``ValueError`` / ``OverflowError`` / ``TypeError`` raised from library
internals, naming no parameter, and - because the client connects lazily on
first use - landing mid-rollout rather than at construction.

``inf`` is the interesting one and is pinned separately below. It is the single
value a caller would pass deliberately, meaning "no deadline, wait as long as it
takes", and the transport does *not* honour it: ``websockets`` raises
``OverflowError`` computing the deadline.
So an unbounded wait is not expressible through these knobs at all, and the
finiteness clause of :func:`~strands_robots.utils.positive_finite_number_error`
is load-bearing here rather than inherited.

These are constructor contracts, so almost nothing here opens a socket. The one
exception is deliberate: a real loopback server proves the refusal is about the
value and not about reachability, which is the whole claim.

Regression for #1984.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from strands_robots.inference import PolicyServer, RemotePolicy
from strands_robots.policies import MockPolicy
from strands_robots.utils import positive_finite_number_error

#: Values that name no wait budget. Each is refused by both knobs.
#:
#: ``True`` / ``False`` matter because ``bool`` is an ``int`` subclass, so a bare
#: ``> 0`` test admits ``True`` as a silent one-second budget - measured as a
#: 1.0002 s ``recv`` timeout, not as an error. ``'10'`` matters because the
#: client applies no ``float()``, so a numeric string reached the transport and
#: blew up inside it (``unsupported operand type(s) for +: 'float' and 'str'``)
#: rather than at the boundary. ``nan`` and ``inf`` are the two the transport
#: rejects itself, from inside its own deadline arithmetic.
UNUSABLE_TIMEOUTS: list[Any] = [
    0,
    0.0,
    -1,
    -0.5,
    math.nan,
    math.inf,
    -math.inf,
    True,
    False,
    "10",
    None,
    [10],
]

#: Accepted. ``np.float32`` is here because a timeout read out of a config array
#: is a real spelling and the shared domain documents it as usable - the client
#: does not coerce, so this pins that no coercion is needed.
USABLE_TIMEOUTS: list[Any] = [0.001, 1, 10.0, 60.0, np.float32(0.5)]

#: The two knobs.
TIMEOUT_PARAMS = ["connect_timeout", "request_timeout"]


def _ws_client(**kwargs: Any) -> RemotePolicy:
    """Construct the WebSocket client, splatted so off-type values reach the guard.

    Both parameters are annotated ``float``; several cases here pass something
    else on purpose, to prove the runtime refuses it rather than the type
    checker.
    """
    return RemotePolicy(**kwargs)


CLIENTS = [("RemotePolicy", _ws_client)]


class TestATimeoutThatNamesNoBudgetIsRefused:
    """Both knobs refuse the same values, naming the class, param and domain."""

    @pytest.mark.parametrize("param", TIMEOUT_PARAMS)
    def test_it_is_refused_at_construction(self, param: str) -> None:
        """The refusal is a ``ValueError`` that identifies what the caller got wrong.

        Nothing in it suggests starting or reaching a server: a ``0`` connect
        timeout used to produce "could not reach a PolicyServer ... Start one
        first" against a server that was running and reachable, because
        ``TimeoutError`` is inside the clause that composes that message.
        """
        for name, build in CLIENTS:
            for value in UNUSABLE_TIMEOUTS:
                with pytest.raises(ValueError) as exc:
                    build(**{param: value})
                text = str(exc.value)
                assert name in text, f"the refusal must name the class, got {text!r}"
                assert param in text, f"the refusal must name the parameter, got {text!r}"
                assert "must be a positive finite number" in text, f"the refusal must state the domain, got {text!r}"
                assert "Start one first" not in text and "could not reach" not in text, f"{value!r}: {text!r}"


class TestARunningServerIsNoLongerBlamedForTheCallersTimeout:
    """The failure this issue is about: an unusable timeout read as an absent server."""

    def test_it_is_a_value_error_rather_than_a_connection_error(self) -> None:
        """The exception type carries the same distinction as the message.

        ``ConnectionError`` is what a caller retries or escalates to whoever owns
        the server; ``ValueError`` is what a caller fixes in their own call. The
        old behaviour raised the former for a mistake that is the latter.
        """
        with pytest.raises(ValueError):
            _ws_client(connect_timeout=0)

    def test_no_websocket_is_dialled_for_a_refused_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guard precedes the transport, so the server is never asked.

        Pinned by making a dial fail the test outright: the old code reached
        ``connect`` (lazily, on first use) and let the transport's verdict stand
        in for validation.
        """
        import websockets.sync.client as ws_client

        def explode(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("connect() must not be reached for a refused timeout")

        monkeypatch.setattr(ws_client, "connect", explode)
        with pytest.raises(ValueError, match="connect_timeout"):
            _ws_client(connect_timeout=0)

    def test_a_positive_timeout_reaches_the_same_live_server(self) -> None:
        """The refused value is the only thing wrong: this server is reachable.

        Without this, every assertion above is consistent with "the client can no
        longer connect at all". A real loopback ``PolicyServer`` on an ephemeral
        port, dialled with a small positive budget, separates the two.
        """
        server = PolicyServer(policy=MockPolicy(), port=0).start()
        try:
            endpoint = f"ws://127.0.0.1:{server.port}"
            client = RemotePolicy(endpoint=endpoint, connect_timeout=5.0, request_timeout=5.0)
            try:
                # Touching a mirrored-metadata property forces the lazy connect
                # and the handshake, both of which use ``connect_timeout``.
                assert client.provider_name == "remote"
                assert client.execution_horizon >= 1
            finally:
                client.close()

            # Same endpoint, same live server, unusable budget: refused before
            # the dial rather than reported as an unreachable server.
            with pytest.raises(ValueError, match="connect_timeout"):
                RemotePolicy(endpoint=endpoint, connect_timeout=0)
        finally:
            server.stop()


class TestInfinityIsRefusedRatherThanReadAsNoDeadline:
    """``inf`` is the one value a caller means, and the transport refuses it."""

    @pytest.mark.parametrize("param", TIMEOUT_PARAMS)
    @pytest.mark.parametrize(("name", "build"), CLIENTS)
    def test_it_is_refused_at_construction(self, name: str, build: Any, param: str) -> None:
        """Refused, deliberately - not admitted as an unbounded wait.

        A later change that reads ``inf`` as "wait forever" fails here first, and
        should read the premise test below before deciding this test is wrong.
        """
        with pytest.raises(ValueError, match="must be a positive finite number"):
            build(**{param: math.inf})

    def test_the_websocket_transport_does_not_honour_it(self) -> None:
        """Premise: ``websockets`` raises rather than waiting indefinitely.

        This is the justification for the finiteness clause on the WebSocket
        side. If a future ``websockets`` starts honouring ``inf``, this test fails
        and the refusal above becomes a choice worth re-making rather than a
        consequence of the transport.
        """
        import time

        with pytest.raises(OverflowError):
            # The deadline arithmetic ``connect``/``recv`` perform on the value.
            time.gmtime(time.monotonic() + math.inf)


class TestTheClientDefersToTheSharedDomain:
    """The verdict is the shared one, so the two cannot drift apart."""

    @pytest.mark.parametrize("param", TIMEOUT_PARAMS)
    def test_the_client_agrees_with_the_shared_verdict(self, param: str) -> None:
        """Refuse exactly when :func:`positive_finite_number_error` refuses.

        Pinned over the accepted values too, so a client that grows a private
        extra restriction - a minimum budget, say - fails here rather than
        drifting silently from the shared rule.
        """
        for value in [*UNUSABLE_TIMEOUTS, *USABLE_TIMEOUTS]:
            shared_refuses = positive_finite_number_error(value, param, "Ctx") is not None
            for _name, build in CLIENTS:
                if shared_refuses:
                    with pytest.raises(ValueError, match="must be a positive finite number"):
                        build(**{param: value})
                else:
                    build(**{param: value})  # constructs; no transport is touched


class TestAnAcceptedTimeoutIsStoredUnchanged:
    """A usable value is kept as given - no coercion stands in for the guard."""

    @pytest.mark.parametrize("param", TIMEOUT_PARAMS)
    def test_it_survives_construction(self, param: str) -> None:
        """The attribute the transport reads carries the caller's value."""
        for _name, build in CLIENTS:
            for value in USABLE_TIMEOUTS:
                client = build(**{param: value})
                assert getattr(client, param) == value, f"{value!r}"

    @pytest.mark.parametrize(("name", "build"), CLIENTS)
    def test_the_defaults_are_inside_the_domain(self, name: str, build: Any) -> None:
        """The shipped defaults pass their own guard.

        A default outside the domain would make every unconfigured construction
        raise, so this fails loudly rather than in every caller.
        """
        client = build()
        assert client.connect_timeout > 0
        assert client.request_timeout > 0


class TestTheGuardSitsWhereTheOtherTransportKnobsAre:
    """Ordering, so the most specific verdict wins when a caller gets two wrong."""

    def test_an_unusable_port_is_reported_before_an_unusable_timeout(self) -> None:
        """``port`` first: "this address cannot be dialled" is the narrower fact."""
        with pytest.raises(ValueError, match="invalid port"):
            _ws_client(port=-1, connect_timeout=0)

    def test_an_explicit_endpoint_does_not_exempt_the_timeout(self) -> None:
        """``endpoint`` supersedes ``host``/``port``, never the wait budget.

        The ``port`` guard is deliberately skipped when an endpoint is given, so
        this pins that the timeout guard was not folded into that condition.
        """
        with pytest.raises(ValueError, match="connect_timeout"):
            _ws_client(endpoint="ws://gpu-box:8765", connect_timeout=0)
        with pytest.raises(ValueError, match="request_timeout"):
            _ws_client(endpoint="ws://gpu-box:8765", request_timeout=math.nan)
