"""A connect that gives up part way releases the subscribers it already made.

:meth:`~strands_robots.drivers.g1.G1Driver.connect_eagerly` builds a
:class:`~strands_robots.tools.g1._dds_engine.DDSSubscriberSet`, subscribes four
topics through it, then starts a publisher, and records the set on
``self._subs`` only once all of that has succeeded. So a ladder that gave up
part way left the topics it had already subscribed running with nothing holding
a reference the driver could reach: :meth:`G1Driver.cleanup` closes
``self._subs``, and on those exits ``self._subs`` is still ``None``.

The publisher-failure exit already rolled back with ``subs.close()`` and said
why in a comment. Its two siblings inside the subscribe loop - a topic whose IDL
class will not resolve, and a subscriber the SDK refuses to construct - returned
the reason without it. One function, three failure exits, one of them releasing
what it built.

What that costs is written down in :meth:`DDSSubscriberSet.close`: every
subscriber is built with a non-zero queue length, and above zero
``unitree_sdk2py`` starts a ``ch_reader`` daemon thread whose target is a bound
method of the channel's reader. The thread keeps the channel reachable, so the
CycloneDDS finaliser never runs - the reader stays matched and the decoder
callbacks keep filling ``_lowstate`` and ``_battery`` for a driver whose
``_connected`` is ``False``. And because :meth:`connect_eagerly` documents a
retry as the supported response to a failure, the next attempt subscribes the
same topics again: the duplicate subscription that the idempotence guard on a
*connected* driver exists to prevent, reached instead through a failed one.

The SDK is absent on a headless runner, so these cells drive the recording
stand-in the sibling release suite already uses, and count ``Close()`` on the
endpoints ``subscribe`` really built.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections import Counter
from typing import Any

import pytest

import strands_robots.drivers.g1 as g1_module
import strands_robots.tools.g1._dds_engine as engine_module
from strands_robots.drivers.g1 import G1Driver
from strands_robots.tools.g1._dds_engine import DDSSubscriberSet
from tests.drivers.test_g1_dds_engine_release import _install_channel, _RecordingEndpoint


class _RefusingEndpoint(_RecordingEndpoint):
    """An endpoint whose ``Init`` fails for one topic, the way the SDK's can."""

    refuse_topic: str = ""

    def Init(self, handler: Any = None, queue_len: int | None = None) -> None:  # noqa: N802 - SDK spelling
        if self.topic == type(self).refuse_topic:
            raise RuntimeError("dds reader could not be created")
        super().Init(handler, queue_len)


def _topics() -> tuple[str, ...]:
    """The topics the driver's own subscription plan names, in order."""
    return tuple(topic for topic, _cls, _decoder in G1Driver()._subscription_plan())


def _bring_up(
    monkeypatch: pytest.MonkeyPatch,
    *,
    refuse_topic: str | None = None,
    resolve_fails_on_call: int | None = None,
    dds_error: str | None = None,
    attempts: int = 1,
) -> tuple[G1Driver, list[_RecordingEndpoint], str | None]:
    """Run ``connect_eagerly`` against recording subscribers and report what it built.

    Args:
        monkeypatch: Pytest's patcher, used for every seam so a machine that
            really has the SDK gets it back.
        refuse_topic: A topic whose subscriber construction fails, driving the
            ``subscribe`` exit.
        resolve_fails_on_call: Make the Nth ``_resolve_message_class`` call
            return a reason, driving the IDL-resolution exit.
        dds_error: Make ``ensure_dds`` fail, driving the ``start`` exit before
            anything is subscribed.
        attempts: Call ``connect_eagerly`` this many times; two models the retry
            the method's docstring invites.

    Returns:
        The driver, every endpoint ``subscribe`` built across all attempts, and
        the last returned reason.
    """
    endpoint_class: type[_RecordingEndpoint] = _RecordingEndpoint
    if refuse_topic is not None:
        endpoint_class = type("_Refusing", (_RefusingEndpoint,), {"refuse_topic": refuse_topic})
    built = _install_channel(monkeypatch, "ChannelSubscriber", endpoint_class)
    monkeypatch.setattr(engine_module, "ensure_dds", lambda _interface: dds_error)

    # Stand in for the IDL lookup unconditionally.  The real one imports the
    # SDK's generated message modules, which a headless runner does not have,
    # so resolving for real would fail the very first topic and every cell
    # below would grade a ladder that never got started.
    calls = {"n": 0}

    def _resolve(class_path: tuple[str, str]) -> Any:
        calls["n"] += 1
        if resolve_fails_on_call is not None and calls["n"] == resolve_fails_on_call:
            return "transient: IDL module not loaded yet"
        return type(class_path[1], (), {})

    monkeypatch.setattr(g1_module, "_resolve_message_class", _resolve)

    driver = G1Driver()
    reason: str | None = None
    for _attempt in range(attempts):
        reason = driver.connect_eagerly()
    return driver, built, reason


def _open(endpoints: list[_RecordingEndpoint]) -> list[_RecordingEndpoint]:
    """The endpoints nothing has closed."""
    return [endpoint for endpoint in endpoints if endpoint.closes == 0]


def _connect_source() -> str:
    """``connect_eagerly``'s source, dedented so :mod:`ast` will parse it."""
    return textwrap.dedent(inspect.getsource(G1Driver.connect_eagerly))


def _returns_after_the_set_exists() -> list[ast.Return]:
    """Every ``return`` of a reason that happens once ``subs`` has been built.

    Read from the source rather than listed, so an exit added later is held to
    the same rule instead of inheriting an exemption by being absent from a
    tuple here.
    """
    function = ast.parse(_connect_source()).body[0]
    assert isinstance(function, ast.FunctionDef)
    constructed = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "subs" for target in node.targets)
    ]
    assert constructed, "connect_eagerly no longer assigns a local named 'subs'"
    first = min(constructed)
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and node.lineno > first
        and not (isinstance(node.value, ast.Constant) and node.value.value is None)
    ]


class TestAFailedBringUpReleasesWhatItBuilt:
    """The regression: every failure exit past the constructor closes the set."""

    def test_a_refused_subscriber_does_not_leave_the_earlier_topics_running(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        topics = _topics()
        driver, built, reason = _bring_up(monkeypatch, refuse_topic=topics[2])
        assert reason is not None and topics[2] in reason
        assert len(built) == 3, "the ladder should have reached the third topic"
        assert _open(built) == [], "subscribers from the topics that did succeed were left open"

    def test_an_unresolvable_idl_class_does_not_leave_the_earlier_topics_running(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        driver, built, reason = _bring_up(monkeypatch, resolve_fails_on_call=3)
        assert reason == "transient: IDL module not loaded yet"
        assert len(built) == 2, "the ladder should have subscribed the first two topics"
        assert _open(built) == [], "subscribers from the topics that did succeed were left open"

    def test_the_retry_the_docstring_invites_does_not_duplicate_a_subscription(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transient failure then a second attempt: one reader per topic, not two."""
        driver, built, reason = _bring_up(monkeypatch, resolve_fails_on_call=3, attempts=2)
        assert reason is None, "the second attempt should have connected"
        assert driver._connected is True
        duplicated = {topic: count for topic, count in Counter(e.topic for e in _open(built)).items() if count > 1}
        assert duplicated == {}, f"the retry left duplicate subscriptions: {duplicated}"

    def test_cleanup_after_the_retry_leaves_no_orphan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing survives teardown, because everything is on the recorded set."""
        driver, built, _reason = _bring_up(monkeypatch, resolve_fails_on_call=3, attempts=2)
        driver.cleanup()
        assert _open(built) == [], "a subscriber outlived cleanup()"


class TestTheFailureIsStillReportedTheSameWay:
    """Releasing the set must not cost the caller the reason it gave up."""

    def test_the_refusal_reason_is_returned_and_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        topics = _topics()
        driver, _built, reason = _bring_up(monkeypatch, refuse_topic=topics[2])
        assert reason is not None
        assert driver._connect_error == reason
        assert driver._connected is False

    def test_the_unresolvable_class_reason_is_returned_and_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _built, reason = _bring_up(monkeypatch, resolve_fails_on_call=3)
        assert reason == "transient: IDL module not loaded yet"
        assert driver._connect_error == reason
        assert driver._connected is False


class TestWhatIsUnchanged:
    """A successful bring-up, and the exit that had nothing to release."""

    def test_a_successful_connect_keeps_its_subscribers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The set is recorded and left running - closing it here would be the opposite bug."""
        driver, built, reason = _bring_up(monkeypatch)
        assert reason is None
        assert len(built) == len(_topics())
        assert _open(built) == built, "a successful connect closed its own subscribers"
        assert driver._subs is not None

    def test_a_successful_connect_is_still_torn_down_by_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, built, _reason = _bring_up(monkeypatch)
        driver.cleanup()
        assert _open(built) == []

    def test_a_dds_init_failure_still_reports_without_subscribing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, built, reason = _bring_up(monkeypatch, dds_error="dds init failed: no such interface")
        assert reason == "dds init failed: no such interface"
        assert built == [], "nothing should be subscribed once start() has failed"
        assert driver._connect_error == reason


class TestPremises:
    """Why an unrecorded set is unreachable, and why the first exit may share the path."""

    def test_cleanup_can_only_close_the_recorded_set(self) -> None:
        """``cleanup`` reads ``self._subs`` and nothing else, so a local is invisible to it."""
        source = textwrap.dedent(inspect.getsource(G1Driver.cleanup))
        closes = [
            ast.unparse(node.func)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "close"
        ]
        assert "self._subs.close" in closes
        assert all(target.startswith("self._") for target in closes), closes

    def test_closing_a_set_that_never_subscribed_is_a_no_op(self) -> None:
        """Which is what lets the ``start``-failure exit share the release path."""
        DDSSubscriberSet("eth0").close()

    def test_the_set_is_recorded_only_after_the_publisher_is_up(self) -> None:
        """The assignment ordering is the reason a partial ladder is unreachable."""
        source = _connect_source()
        assert source.index("self._pubs = pubs") < source.index("self._subs = subs")


class TestOneOwnerReleasesThePartialSet:
    """Structural: no exit past the constructor may forget the rollback."""

    def test_every_failure_exit_routes_through_the_release_helper(self) -> None:
        returns = _returns_after_the_set_exists()
        assert len(returns) >= 4, f"expected the four failure exits, found {len(returns)}"
        routed = [
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "_abort_connect"
            for node in returns
        ]
        assert all(routed), (
            "a failure exit past the DDSSubscriberSet constructor returns without releasing it: "
            f"{[ast.unparse(node) for node, ok in zip(returns, routed, strict=True) if not ok]}"
        )

    def test_the_connect_path_does_not_close_the_set_itself(self) -> None:
        """One owner: ``subs.close()`` belongs to the helper, not to the ladder."""
        assert "subs.close()" not in _connect_source()
        assert "subs.close()" in textwrap.dedent(inspect.getsource(G1Driver._abort_connect))
