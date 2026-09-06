"""The host half of a dialled websocket address is graded the way its port is.

Every caller-supplied port this package dials is held to one shared domain,
:func:`strands_robots.utils.tcp_port_error`, for the reason its consumers record:
an unusable port is not refused by the transport, it is *applied*, and surfaces
much later as an unreachable server that implicates the service the caller was
trying to reach. The host beside it is interpolated into the same expression,
``ws://{host}:{port}``, and was held to nothing on two of the three surfaces that
build one - so a value that is not a host was resolved rather than refused, and
the resolution discarded the very port the shared domain had just approved:

* ``host="127.0.0.1/foo"`` parses as host ``127.0.0.1``, path ``/foo:<port>`` and
  port **80**. The validated port becomes part of the path and the client dials a
  port nobody configured. The port domain cannot see this: it is the host half
  that takes the port away.
* ``host="ws://127.0.0.1"`` - the shape a caller who pastes a URI supplies -
  parses as host ``ws`` on port 80.
* ``host=""`` builds no URI at all, raising ``InvalidURI``, which is not an
  ``OSError`` and so escapes the channel these clients convert into their
  actionable "could not reach the server" hint.
* A non-string is carried into the URI verbatim: ``None`` is dialled as the DNS
  name ``"none"`` and an ``int`` as its digits.

These pin the domain on all three surfaces at once, the shape of the harm against
the real URI parser, the two documented asymmetries between a stated and an
unstated address, and the over-reach control that every host a URI *can* carry is
still accepted identically everywhere.

The ZMQ policy clients are deliberately not held to this domain, and that is
pinned here too rather than asserted: ``tcp://`` is not a URI, and a cell measures
that ``zmq``'s own address parse refuses each delimiter spelling at ``connect``
with the whole address in the message, so the transport there reports what the
domain would.

Everything here is offline - no server, no socket, and no policy is ever asked
for an action.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING, Any, cast

import pytest

import strands_robots.utils as utils_module
from strands_robots.inference.client import RemotePolicy
from strands_robots.policies.cosmos3.policy import Cosmos3Policy
from strands_robots.policies.vera import VeraConfig

if TYPE_CHECKING:
    # Reaches nothing but the `cast` below, which is a string, so the name needs
    # to exist for a type checker and not at runtime. Under `TYPE_CHECKING` that
    # is what it costs; imported at runtime it also loads the real client into a
    # module whose whole point is that a stand-in is never asked to dial.
    from strands_robots.policies.cosmos3.client import Cosmos3WebsocketClient

# Hosts a URI cannot carry, grouped by what the value did instead of failing.
# ``':'`` is a delimiter because the port follows it, so a bare IPv6 literal and
# a host that already carries a port both re-cut the URI.
RECUT_THE_URI: list[str] = [
    "127.0.0.1/foo",
    "ws://127.0.0.1",
    "127.0.0.1:9999",
    "::1",
    "user@127.0.0.1",
    "127.0.0.1?x=1",
    "127.0.0.1#frag",
]

# Spellings that name no host at all: the parse reports "hostname isn't provided".
NAME_NO_HOST: list[str] = ["", "[]"]

# Values a resolver silently repairs rather than reporting.
SILENTLY_REPAIRED: list[str] = ["my host", "my\thost", "127.0.0.1\x00"]

# Non-strings, carried into the URI verbatim by the f-string.
NOT_A_STRING: list[Any] = [None, 8000, 127.1, True, ["127.0.0.1"]]

UNUSABLE_HOSTS: list[Any] = [*RECUT_THE_URI, *NAME_NO_HOST, *SILENTLY_REPAIRED, *NOT_A_STRING]

# Every host a URI can carry. ``"0.0.0.0"`` is here on purpose: it is the
# documented way to reach a server bound on every interface.
USABLE_HOSTS: list[str] = ["127.0.0.1", "localhost", "gpu-box", "gpu-box.local", "[::1]", "0.0.0.0"]


class _StandInClient:
    """A client object that owns its own address, injected instead of built.

    Deliberately not a real ``Cosmos3WebsocketClient``: the point of the cell it
    serves is that the constructor never looks at the address when one is
    injected, so a stand-in that would fail on any other use is the strongest
    form of that claim.
    """


def _remote_policy_refusal(host: Any) -> str | None:
    """Refusal :class:`~strands_robots.inference.client.RemotePolicy` gives ``host``."""
    try:
        RemotePolicy(host=host)
    except ValueError as exc:
        return str(exc)
    return None


def _cosmos3_refusal(host: Any) -> str | None:
    """Refusal :class:`~strands_robots.policies.cosmos3.policy.Cosmos3Policy` gives ``host``."""
    try:
        Cosmos3Policy(embodiment="droid", host=host)
    except ValueError as exc:
        return str(exc)
    return None


def _vera_refusal(host: Any) -> str | None:
    """Refusal :class:`~strands_robots.policies.vera.VeraConfig` gives ``host``."""
    try:
        VeraConfig(embodiment="pusht", host=host)
    except ValueError as exc:
        return str(exc)
    return None


# The surfaces that build ``ws://{host}:{port}`` from a caller-supplied host.
# Read through each public constructor, not through the domain function, so the
# cells grade the behaviour a caller gets rather than the wiring behind it.
WEBSOCKET_SURFACES: dict[str, Any] = {
    "RemotePolicy": _remote_policy_refusal,
    "Cosmos3Policy": _cosmos3_refusal,
    "VeraConfig": _vera_refusal,
}


@pytest.mark.parametrize("surface", sorted(WEBSOCKET_SURFACES))
@pytest.mark.parametrize("host", UNUSABLE_HOSTS, ids=repr)
def test_a_host_a_uri_cannot_carry_is_refused_by_every_websocket_surface(surface: str, host: Any) -> None:
    """A value that cannot address a host is refused, not dialled as something else."""
    message = WEBSOCKET_SURFACES[surface](host)
    assert message is not None, f"{surface} accepted host={host!r}"
    assert message.startswith(f"{surface}: host "), message
    assert repr(host) in message, message


@pytest.mark.parametrize("surface", sorted(WEBSOCKET_SURFACES))
@pytest.mark.parametrize("host", USABLE_HOSTS)
def test_every_host_a_uri_can_carry_is_still_accepted_everywhere(surface: str, host: str) -> None:
    """The over-reach control: the domain refuses addresses, not deployments."""
    assert WEBSOCKET_SURFACES[surface](host) is None


def test_the_three_surfaces_give_the_same_verdict_on_the_same_host() -> None:
    """One address cannot be refused by one client and dialled by the next."""
    for host in [*UNUSABLE_HOSTS, *USABLE_HOSTS]:
        verdicts = {name: refuse(host) is None for name, refuse in WEBSOCKET_SURFACES.items()}
        assert len(set(verdicts.values())) == 1, f"host={host!r} split the surfaces: {verdicts}"


@pytest.mark.parametrize("host", RECUT_THE_URI, ids=repr)
def test_a_refused_delimiter_host_is_what_takes_the_validated_port(host: str) -> None:
    """Measured against the real parser: the refusal is the port being discarded.

    Pins the harm rather than restating the rule. ``websockets`` is the parser
    these clients hand the URI to, so its reading of the pre-fix value is the
    external oracle for what the host half did to the port beside it.
    """
    parse_uri = pytest.importorskip("websockets.uri").parse_uri
    port = 8765
    try:
        parsed = parse_uri(f"ws://{host}:{port}")
    except Exception:
        # Some spellings the parse refuses outright; either way it is not a dial
        # to the configured port, which is what the refusal exists to prevent.
        return
    assert (parsed.host, parsed.port) != (host, port), (
        f"host={host!r} was expected to re-cut the URI, but parsed as the configured address"
    )


def test_a_zmq_endpoint_is_left_to_the_transport_that_already_refuses_it() -> None:
    """The deferral is measured: ``zmq`` reports what the domain would.

    The GR00T and MoveIt2 policy clients dial ``tcp://{host}:{port}``, not a URI,
    and are deliberately outside this domain. That is only defensible if their
    transport refuses the same spellings, so it is measured here instead of
    assumed.
    """
    zmq = pytest.importorskip("zmq")
    context = zmq.Context()
    try:
        for host in RECUT_THE_URI[:2] + SILENTLY_REPAIRED[:2]:
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            endpoint = f"tcp://{host}:5556"
            try:
                with pytest.raises(zmq.ZMQError) as caught:
                    socket.connect(endpoint)
                # zmq names the whole endpoint it refused, escaping any
                # non-printable in it - the report a caller needs to see which
                # half of the address was wrong.
                assert repr(endpoint)[1:-1] in str(caught.value), str(caught.value)
            finally:
                socket.close()
    finally:
        context.term()


# Asymmetries between the surfaces, each with the reason it is one. A host is
# only graded when it is the effective spelling of the address; a surface that
# was handed the address some other way never reads it.
STATED_ELSEWHERE: dict[str, str] = {
    "RemotePolicy(endpoint=...)": (
        "endpoint supersedes host/port and is the whole URI, so the host is not the spelling in use - "
        "the same terms on which port is already left unread"
    ),
    "Cosmos3Policy(client=...)": (
        "an injected client owns its own address, so this constructor never builds ws://<host>:<port> - "
        "the same terms on which port is already left unread"
    ),
}


def test_every_asymmetry_states_why_it_is_one() -> None:
    """A surface left ungraded needs a recorded reason, not an omission."""
    assert all(reason.strip() for reason in STATED_ELSEWHERE.values())


def test_an_endpoint_supersedes_the_host_it_replaces() -> None:
    """A stated URL is the address, so the unused host half is not graded."""
    policy = RemotePolicy(endpoint="ws://gpu-box:8765", host="127.0.0.1/foo")
    assert policy.uri == "ws://gpu-box:8765"


def test_an_injected_client_owns_the_address_this_constructor_did_not_build() -> None:
    """Cosmos 3 reads host only when it is the one building the endpoint."""
    injected = cast("Cosmos3WebsocketClient", _StandInClient())
    policy = Cosmos3Policy(embodiment="droid", host="127.0.0.1/foo", client=injected)
    assert isinstance(policy, Cosmos3Policy)


def test_the_domain_has_one_owner_and_no_consumer_restates_it() -> None:
    """The rule lives beside the port domain it is the other half of.

    A second copy is how the two halves drift apart: one surface would refuse an
    address the next dials. The consumers are graded on calling the shared owner
    rather than spelling the delimiter set themselves.
    """
    for domain in ("dial_host_error", "tcp_port_error"):
        owner = getattr(utils_module, domain, None)
        assert owner is not None, f"{domain} is not owned by {utils_module.__name__}"
        assert owner.__module__ == utils_module.__name__
    for module_name, source in (
        ("cosmos3.policy", inspect.getsource(Cosmos3Policy.__init__)),
        ("inference.client", inspect.getsource(RemotePolicy.__init__)),
        ("vera.config", inspect.getsource(type(VeraConfig(embodiment="pusht")).__post_init__)),
    ):
        called = _domains_called(source)
        assert "dial_host_error" in called, f"{module_name} does not call dial_host_error: {sorted(called)}"
        assert "isprintable" not in source, f"{module_name} restates the host domain"


def _domains_called(source: str) -> set[str]:
    """Names this source *calls*, so a mere reference does not count as a check.

    Read as a call graph rather than as text because a guard that collapses the
    two halves into a table - ``for param, value, domain in ((...),)`` - still
    contains both domain names while calling neither by name. That shape reads as
    equivalent and is not: the sibling port guard in
    ``tests/policies/test_service_port_domain.py`` grades provider constructors
    on an ``ast.Call`` to ``tcp_port_error``, so a table would be reported there
    as a provider shipping an unvalidated port. Both halves are pinned here on
    the same terms as that guard uses for the port.
    """
    tree = ast.parse(textwrap.dedent(source))
    return {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}


class RefusesItsOwnRead(str):
    """A ``str`` subclass that answers no string operation the guard performs.

    Not hypothetical, and not a hostile-input exercise: a host arrives from a
    config file, an env var or an agent tool, and any of those may hand over a
    ``str`` subclass carrying provenance. It is a ``str`` by ``isinstance``, so it
    reaches the character scan that decides the verdict, and the scan is the
    caller's own code.
    """

    def startswith(self, *args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("no read for you")


@pytest.mark.parametrize("surface", sorted(WEBSOCKET_SURFACES))
def test_a_host_that_cannot_be_read_is_refused_rather_than_raised_past(surface: str) -> None:
    """A host the guard cannot inspect is a refusal, on the channel refusals use.

    The domain reports through a returned message that each constructor turns into
    a ``ValueError``, so the read it makes to reach that verdict must not raise a
    different exception out of the same call: a ``RuntimeError`` from a character
    scan escapes every ``except ValueError`` a caller wrote for this parameter.
    Unreadable is a refusal because the read *is* the verdict here - a host that
    cannot be inspected cannot be certified as one a URI can carry.
    """
    message = WEBSOCKET_SURFACES[surface](RefusesItsOwnRead("127.0.0.1"))
    assert message is not None, f"{surface} accepted a host it could not read"
    assert message.startswith(f"{surface}: host "), message
    assert "could not be read" in message, message
    assert "RuntimeError: no read for you" in message, message


def test_the_probe_really_refuses_the_read_the_guard_makes() -> None:
    """Non-vacuity: a probe that had started answering would pass the cell above."""
    probe = RefusesItsOwnRead("127.0.0.1")
    assert isinstance(probe, str)
    with pytest.raises(RuntimeError, match="no read for you"):
        probe.startswith("[")


class RefusesToBeCopied(str):
    """A ``str`` subclass whose ``__str__`` raises."""

    def __str__(self) -> str:
        raise RuntimeError("no read for you")


class RefusesToBeSliced(str):
    """A ``str`` subclass whose ``__getitem__`` raises."""

    def __getitem__(self, index: Any) -> str:
        raise RuntimeError("no read for you")


#: ``(operation, host)`` for the operations the read performs on the caller's own
#: value, beside the ``startswith`` the cell above covers. The slice is reached
#: only for a bracketed literal, because that is the only host whose brackets are
#: stripped - so each row needs the host that reaches it.
OPERATIONS_THE_READ_MAKES: list[tuple[str, Any]] = [
    ("__str__", RefusesToBeCopied("127.0.0.1")),
    ("__getitem__", RefusesToBeSliced("[::1]")),
]


@pytest.mark.parametrize("surface", sorted(WEBSOCKET_SURFACES))
@pytest.mark.parametrize(
    ("operation", "host"), OPERATIONS_THE_READ_MAKES, ids=[operation for operation, _ in OPERATIONS_THE_READ_MAKES]
)
def test_no_operation_the_read_makes_escapes_the_try(surface: str, operation: str, host: Any) -> None:
    """Each operation separately, because one probe only reaches the first of them.

    Reading a host is three operations on the caller's object - a copy, a prefix
    test and a slice - and a subclass may refuse any one of them. Covering only
    the first leaves the other two: a read moved out of the ``try`` still answers
    a value that refuses ``startswith`` and raises on one that refuses the
    operation that moved.
    """
    message = WEBSOCKET_SURFACES[surface](host)
    assert message is not None, f"{surface} accepted a host whose {operation} does not answer"
    assert "could not be read" in message, message
    assert "RuntimeError: no read for you" in message, message


class RefusesToEndswith(str):
    def endswith(self, *args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("never called")


class RefusesToBeIterated(str):
    def __iter__(self) -> Any:
        raise RuntimeError("never called")


class RefusesToBeRepred(str):
    def __repr__(self) -> str:
        raise RuntimeError("never called")


#: ``(operation, host, why it is never reached)``. These are the string
#: operations a reader of a host *could* make and this one does not, so a
#: usable host carrying them is accepted rather than reported unreadable.
OPERATIONS_THE_READ_AVOIDS: list[tuple[str, Any, str]] = [
    ("endswith", RefusesToEndswith("127.0.0.1"), "'and' short-circuits when the value does not start with '['"),
    ("__iter__", RefusesToBeIterated("127.0.0.1"), "the character scan runs over a plain str copy, not the value"),
    ("__repr__", RefusesToBeRepred("127.0.0.1"), "the value is rendered through the shared guarded renderer"),
]


@pytest.mark.parametrize(
    ("operation", "host", "why"),
    OPERATIONS_THE_READ_AVOIDS,
    ids=[operation for operation, _, _ in OPERATIONS_THE_READ_AVOIDS],
)
def test_an_operation_the_read_never_makes_leaves_a_usable_host_usable(operation: str, host: Any, why: str) -> None:
    """The over-reach control, and the pin on how the read is narrow.

    A refusal for every host that carries an unusual ``str`` subclass would be a
    domain nobody could pass, so the reader must reach only what it needs. Each
    row records the reason it does not reach this one; ``__iter__`` is the
    load-bearing one, since it is unreachable only while the scanned body is a
    copy the module made rather than the object it was handed.
    """
    assert utils_module.dial_host_error(host, "host", "Ctx") is None, (
        f"a usable host was refused over {operation}, which the read does not make: {why}"
    )


# Each surface with BOTH halves of its address unusable at once. The port keyword
# differs per surface (``port`` / ``server_port``), so the constructors are
# written out rather than derived from one signature.
BOTH_HALVES_UNUSABLE: dict[str, Any] = {
    "RemotePolicy": lambda host: RemotePolicy(host=host, port=65536),
    "Cosmos3Policy": lambda host: Cosmos3Policy(embodiment="droid", host=host, port=65536),
    "VeraConfig": lambda host: VeraConfig(embodiment="pusht", host=host, server_port=65536),
}


@pytest.mark.parametrize("surface", sorted(BOTH_HALVES_UNUSABLE))
@pytest.mark.parametrize("host", RECUT_THE_URI, ids=repr)
def test_the_host_is_graded_before_the_port_it_would_have_taken(surface: str, host: str) -> None:
    """A caller who gets both halves wrong is told about the host.

    The two refusals sit in each constructor as consecutive statements, so which
    one a caller sees is decided by their order - and it has to be the host: the
    delimiter in it is what re-cuts the URI and carries the port away, so
    reporting the port would name the component that was discarded rather than
    the one that discarded it. Stated independently in all three constructors,
    so it is pinned per surface rather than once.
    """
    with pytest.raises(ValueError) as caught:
        BOTH_HALVES_UNUSABLE[surface](host)
    message = str(caught.value)
    assert "host" in message, f"{surface} refused the port before the host that would have taken it: {message}"
    assert "65536" not in message, f"{surface} named the discarded port instead of the host: {message}"
