"""The Microduck transport knobs are checked, never handed on unexamined.

``MicroduckDriver.__init__`` takes three numeric knobs. Each reaches a consumer
that cannot report what it was given:

* ``timeout`` goes to ``socket.settimeout`` and to the reply wait. A ``nan``, an
  ``inf``, a negative or a numeric string raised out of :meth:`connect_eagerly`
  from inside the socket call - naming neither the driver nor the parameter, out
  of a method declared ``-> str | None``. ``True`` acted as a silent one second
  and ``None`` left both the socket and the reply wait unbounded.
* ``subscribe_hz`` is interpolated straight into the ``robot.subscribe`` params,
  so ``nan``/``inf`` put the bare ``NaN``/``Infinity`` tokens on the wire. Those
  are not JSON (RFC 8259), so a strict daemon parser refuses the frame while
  ``connect_eagerly`` reports the connection as established.

* ``api_version`` is interpolated into the ``hello`` params the same way, as
  robotd's ``HelloParams.api_version: u32``. It used to be covered by the
  handshake refusing any reply that did not echo it; that gate also refused
  every robotd built after the pin, so it is gone and the knob has a domain.

This is the same convention the driver's actuation flags already follow through
``boolean_flag_error`` (``tests/drivers/microduck/test_microduck_flag_domain.py``
holds that half), and the same one the sibling drivers' constructors follow -
``G1Driver`` holds ``battery_floor_pct`` to ``finite_number_error`` and
``ReachyDriver`` refuses an unusable ``api_port`` before it has a daemon to talk
to. These knobs were the ones left outside it.
"""

from __future__ import annotations

import ast
import inspect
import json
import socket
import threading

import numpy as np
import pytest

from strands_robots.drivers import microduck as microduck_module
from strands_robots.drivers.microduck import MICRODUCK_API_VERSION, MicroduckDriver
from tests.mocks.microduck_robotd import MockRobotd

#: A path no daemon answers: these cells grade the constructor, not the socket.
ABSENT_SOCKET = "/tmp/microduck-transport-domain-no-such.sock"

#: Spellings a caller reaches for that the transport cannot use as a timeout.
#: ``0.0`` puts the socket in non-blocking mode rather than timing out, and
#: ``None`` removes the bound the parameter exists to impose.
UNUSABLE_TIMEOUTS: list[object] = [
    float("nan"),
    float("inf"),
    -1.0,
    0.0,
    True,
    False,
    None,
    "5",
    [5],
]

#: Spellings robotd is not sent an integer decimation by. ``numpy`` integers are
#: refused because ``json.dumps`` cannot serialise them at all.
UNUSABLE_HZ: list[object] = [
    float("nan"),
    float("inf"),
    True,
    False,
    0,
    -5,
    2.5,
    "30",
    np.int64(3),
]

#: The knobs that must reach a shared domain; none is exempt.
GUARDED_KNOBS = ("timeout", "subscribe_hz", "api_version")
EXEMPT_KNOBS: dict[str, str] = {}

#: The shared numeric domains in ``strands_robots.utils`` this driver may use.
SHARED_NUMERIC_DOMAINS = (
    "finite_number_error",
    "positive_finite_number_error",
    "positive_count_error",
    "positive_whole_number_error",
    "non_negative_whole_number_error",
    "tcp_port_error",
)


def _numeric_init_params() -> list[str]:
    """Every numeric-annotated constructor parameter, read off the signature.

    Returns:
        The parameter names annotated as an ``int``/``float`` (optionally
        ``| None``), so a knob added later is graded rather than inheriting an
        exemption by being absent from a literal list.
    """
    numeric = {"int", "float", "int | None", "float | None"}
    found = []
    for name, parameter in inspect.signature(MicroduckDriver.__init__).parameters.items():
        annotation = parameter.annotation
        text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
        if text in numeric:
            found.append(name)
    return found


def _init_body_source() -> str:
    """The constructor's own source, for reading which domains it consults."""
    return inspect.getsource(MicroduckDriver.__init__)


class TestTheConsumersCannotReportWhatTheyWereHanded:
    """Premise: the harm is a property of the consumers, not of this driver."""

    def test_a_non_finite_timeout_raises_inside_the_socket_call(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            with pytest.raises(ValueError, match="NaN"):
                sock.settimeout(float("nan"))
        finally:
            sock.close()

    def test_a_true_timeout_is_a_silent_one_second_and_none_is_unbounded(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(True)
            assert sock.gettimeout() == 1.0, "True must not be readable as a usable timeout"
            sock.settimeout(None)
            assert sock.gettimeout() is None, "None removes the bound rather than setting one"
        finally:
            sock.close()

    def test_the_reply_wait_returns_at_once_for_a_non_finite_bound(self) -> None:
        # A nan bound makes the wait fall straight through, so the request is
        # reported as timed out having waited for nothing.
        assert threading.Event().wait(float("nan")) is False

    def test_a_non_finite_hz_is_not_json_and_a_numpy_int_is_not_serialisable(self) -> None:
        def refuse_bare_constant(token: str) -> float:
            raise ValueError(f"bare {token} is not JSON")

        for value in (float("nan"), float("inf")):
            frame = json.dumps({"hz": value})
            with pytest.raises(ValueError, match="not JSON"):
                json.loads(frame, parse_constant=refuse_bare_constant)
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps({"hz": np.int64(3)})


class TestTimeoutIsAPositiveFiniteNumber:
    """Every unusable timeout is refused at construction, naming itself."""

    @pytest.mark.parametrize("value", UNUSABLE_TIMEOUTS, ids=[repr(v)[:12] for v in UNUSABLE_TIMEOUTS])
    def test_an_unusable_timeout_is_refused_naming_the_driver_and_the_knob(self, value: object) -> None:
        with pytest.raises(ValueError) as caught:
            MicroduckDriver(port=ABSENT_SOCKET, timeout=value)  # type: ignore[arg-type]
        reason = str(caught.value)
        assert "MicroduckDriver" in reason, f"the refusal must name the driver, got {reason!r}"
        assert "timeout" in reason, f"the refusal must name the knob, got {reason!r}"

    @pytest.mark.parametrize("value", UNUSABLE_TIMEOUTS, ids=[repr(v)[:12] for v in UNUSABLE_TIMEOUTS])
    def test_connect_eagerly_is_never_reached_with_an_unusable_timeout(self, value: object) -> None:
        # The whole point of refusing in the constructor: ``connect_eagerly`` is
        # declared ``-> str | None`` and used to raise from inside the socket.
        with pytest.raises(ValueError):
            MicroduckDriver(port=ABSENT_SOCKET, timeout=value).connect_eagerly()  # type: ignore[arg-type]


class TestSubscribeHzIsAPositiveIntegerWhenGiven:
    """Every unusable decimation is refused before it can reach the wire."""

    @pytest.mark.parametrize("value", UNUSABLE_HZ, ids=[repr(v)[:12] for v in UNUSABLE_HZ])
    def test_an_unusable_hz_is_refused_naming_the_driver_and_the_knob(self, value: object) -> None:
        with pytest.raises(ValueError) as caught:
            MicroduckDriver(port=ABSENT_SOCKET, subscribe_hz=value)  # type: ignore[arg-type]
        reason = str(caught.value)
        assert "MicroduckDriver" in reason, f"the refusal must name the driver, got {reason!r}"
        assert "subscribe_hz" in reason, f"the refusal must name the knob, got {reason!r}"


class TestTheUsableSpellingsAreUnchanged:
    """Over-reach controls: nothing a caller legitimately passes is refused."""

    @pytest.mark.parametrize("value", [5.0, 2, 0.25, 30.0], ids=["5.0", "2", "0.25", "30.0"])
    def test_a_usable_timeout_still_builds_and_reports_rather_than_raising(self, value: float) -> None:
        driver = MicroduckDriver(port=ABSENT_SOCKET, timeout=value)
        reason = driver.connect_eagerly()
        assert isinstance(reason, str) and "did not answer" in reason, (
            f"an absent daemon must be reported, not raised; got {reason!r}"
        )

    def test_the_default_subscribe_sends_no_hz_key(self) -> None:
        with MockRobotd(api_version=MICRODUCK_API_VERSION) as server:
            driver = MicroduckDriver(port=server.path, timeout=2.0)
            try:
                assert driver.connect_eagerly() is None
                frame = _subscribe_frame(server)
                assert frame["params"] == {}, f"the default must send no decimation, got {frame['params']!r}"
            finally:
                driver.cleanup()

    def test_a_positive_hz_reaches_the_wire_as_an_integer(self) -> None:
        with MockRobotd(api_version=MICRODUCK_API_VERSION) as server:
            driver = MicroduckDriver(port=server.path, timeout=2.0, subscribe_hz=30)
            try:
                assert driver.connect_eagerly() is None
                frame = _subscribe_frame(server)
                assert frame["params"] == {"hz": 30}, f"expected an integer decimation, got {frame['params']!r}"
                assert isinstance(frame["params"]["hz"], int), "the wire value must stay an integer"
            finally:
                driver.cleanup()


class TestApiVersionIsAPositiveInteger:
    """Sent as ``HelloParams.api_version: u32``; the handshake no longer gates it."""

    @pytest.mark.parametrize("value", UNUSABLE_HZ, ids=[repr(v)[:12] for v in UNUSABLE_HZ])
    def test_an_unusable_api_version_is_refused_naming_the_driver_and_the_knob(self, value: object) -> None:
        with pytest.raises(ValueError) as caught:
            MicroduckDriver(port=ABSENT_SOCKET, api_version=value)  # type: ignore[arg-type]
        reason = str(caught.value)
        assert "MicroduckDriver" in reason, f"the refusal must name the driver, got {reason!r}"
        assert "api_version" in reason, f"the refusal must name the knob, got {reason!r}"


class TestEveryNumericKnobIsAccountedFor:
    """Derived: a knob added later is graded, not exempted by absence."""

    def test_the_numeric_knobs_are_the_ones_this_file_grades(self) -> None:
        found = set(_numeric_init_params())
        assert found, "the derivation must find the constructor's numeric knobs"
        assert found == set(GUARDED_KNOBS) | set(EXEMPT_KNOBS), (
            f"a numeric constructor knob is neither guarded nor exempted here: {found!r}"
        )

    def test_each_guarded_knob_reaches_a_shared_numeric_domain(self) -> None:
        source = _init_body_source()
        consulted = {name for name in SHARED_NUMERIC_DOMAINS if f"{name}(" in source}
        assert consulted, f"the constructor consults no shared numeric domain: {sorted(SHARED_NUMERIC_DOMAINS)}"
        for knob in GUARDED_KNOBS:
            guarded = any(f"{name}({knob}," in source for name in SHARED_NUMERIC_DOMAINS)
            assert guarded, f"{knob} is not handed to a shared numeric domain in __init__"

    def test_every_exemption_states_a_reason(self) -> None:
        for knob, reason in EXEMPT_KNOBS.items():
            assert reason.strip(), f"{knob} is exempted without a reason"

    def test_the_refusal_precedes_every_recorded_attribute(self) -> None:
        # A refused construction must leave no half-built driver behind, so the
        # guards sit above the first ``self._x = ...`` in the body.
        tree = ast.parse(inspect.getsource(microduck_module))
        init = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
            and any("subscribe_hz" == argument.arg for argument in node.args.kwonlyargs)
        )
        raises = [node.lineno for node in ast.walk(init) if isinstance(node, ast.Raise)]
        stores = [
            node.lineno
            for node in ast.walk(init)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
        ]
        assert raises and stores, "expected both a refusal and recorded state in the constructor"
        assert max(raises) < min(stores), (
            f"the refusals must precede the first recorded attribute: raises={raises} stores={stores}"
        )


def _subscribe_frame(server: MockRobotd) -> dict[str, object]:
    """The ``robot.subscribe`` request the driver put on the wire.

    Args:
        server: The mock daemon whose received lines are read.

    Returns:
        The decoded JSON-RPC request.
    """
    for raw in server.received:
        frame = json.loads(raw)
        if frame.get("method") == "robot.subscribe":
            return dict(frame)
    raise AssertionError(f"no robot.subscribe frame was sent; got {[bytes(r) for r in server.received]!r}")
