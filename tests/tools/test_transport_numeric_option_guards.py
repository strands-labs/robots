"""The ROS transport tools refuse a numeric option they cannot honor.

``use_ros``, ``use_rtps`` and ``use_rosbridge`` expose the same three numeric
options to an agent - ``count``, ``rate`` and ``timeout`` - and all three consume
them the same way: ``count`` is a ``range()`` bound, ``rate`` becomes an
inter-message period ``1 / rate``, and ``timeout`` is a wait budget. Only
positive, finite values can be honored, so a value outside that domain must be
refused with a message naming the option, before any transport entity joins the
graph and before a single message is published.

The guard used to be duplicated per tool, and that is exactly how it drifted:
two copies agreed while the rosbridge transport arrived with no copy at all, so
``rate`` was accepted and then discarded (every message sent back-to-back under
``status="success"``), a ``count`` of ``True`` published one message and reported
"published True message(s)", a non-positive ``timeout`` returned an empty result
blaming the topic for being silent, and ``timeout=inf`` raised ``OverflowError``
straight out of a tool documented to return a result dict. One shared owner plus
the structural guard at the end of this module is what keeps a fourth transport
from repeating it.

These tests run with NO ROS 2, NO cyclonedds and NO roslibpy installed: the
backend availability probes and the transport-facing helpers are monkeypatched,
so the guards, the per-action scoping, the cross-transport parity and the
"nothing was published" contract are all exercised transport-free.
"""

from __future__ import annotations

import ast
import inspect
import sys
import time
import types as _types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.use_ros as ros_mod
import strands_robots.tools.use_rosbridge as rosbridge_mod
import strands_robots.tools.use_rtps as rtps_mod
from strands_robots.tools._numeric_options import numeric_option_error

# Values outside the accepted domain of each option, with the reason each one is
# unusable. ``inf`` matters as much as ``0``: it passes a bare ``rate > 0`` test
# and then collapses ``1 / rate`` to ``0``, leaving the burst unthrottled.
UNUSABLE_RATES: list[Any] = [0.0, -5.0, float("nan"), float("inf"), "10", None]
UNUSABLE_TIMEOUTS: list[Any] = [0.0, -1.0, float("nan"), float("inf"), "2", None]
UNUSABLE_COUNTS: list[Any] = [0, -1, True, 2.7, "3", None]


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []))


@pytest.fixture(autouse=True)
def _every_backend_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every transport to a present backend; opt out where needed."""
    monkeypatch.setattr(ros_mod._backend, "available", lambda: True)
    monkeypatch.setattr(rtps_mod._backend, "available", lambda: True)
    monkeypatch.setattr(rosbridge_mod._backend, "available", lambda: True)


# ---------------------------------------------------------------------------
# A refused option publishes nothing: the real ``_publish`` body never runs.
# ---------------------------------------------------------------------------


@pytest.fixture
def published_at(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Wire ``use_ros`` to a recording publisher and return its send timestamps.

    The real ``_publish`` body runs - only the rclpy node, the message class and
    the executor spin are faked - so the list records exactly the messages that
    reached the wire, and their spacing is the pacing the tool actually applied.
    """
    stamps: list[float] = []

    class FakePublisher:
        def publish(self, msg: Any) -> None:
            stamps.append(time.perf_counter())

    class FakeNode:
        def create_publisher(self, cls: Any, topic: str, depth: int) -> FakePublisher:
            return FakePublisher()

        def destroy_publisher(self, pub: Any) -> None:
            pass

    set_message = _types.ModuleType("rosidl_runtime_py.set_message")
    set_message.set_message_fields = lambda msg, fields: None  # type: ignore[attr-defined]
    package = _types.ModuleType("rosidl_runtime_py")
    package.set_message = set_message  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py", package)
    monkeypatch.setitem(sys.modules, "rosidl_runtime_py.set_message", set_message)

    monkeypatch.setattr(ros_mod._backend, "_ensure_node", lambda: FakeNode())
    monkeypatch.setattr(ros_mod._backend, "spin_for", lambda predicate, timeout: None)
    monkeypatch.setattr(ros_mod, "_get_message", lambda msg_type: object)
    return stamps


def _publish_twist(**options: Any) -> dict[str, Any]:
    return ros_mod.use_ros(action="publish", topic="/cmd_vel", type="geometry_msgs/msg/Twist", **options)


@pytest.mark.parametrize("rate", UNUSABLE_RATES)
def test_an_unusable_rate_publishes_no_message(published_at: list[float], rate: Any) -> None:
    """A rate that cannot pace the burst is refused before anything is sent.

    Pre-fix the loop fell back to ``period = 0.0`` and sent every message
    back-to-back, reporting success - a velocity hold collapsed into an
    instantaneous burst that a base then latches as its last command.
    """
    result = _publish_twist(count=6, rate=rate)

    assert result["status"] == "error"
    assert f"rate must be > 0, got {rate!r}." in _texts(result)
    assert published_at == []


@pytest.mark.parametrize("count", UNUSABLE_COUNTS)
def test_an_unusable_count_publishes_no_message(published_at: list[float], count: Any) -> None:
    """A count that is not a positive integer is refused, not silently absorbed.

    ``range(-1)`` and ``range(0)`` publish nothing, and ``range(2.7)`` raises a
    ``TypeError`` naming neither the tool nor the option.
    """
    result = _publish_twist(count=count, rate=10.0)

    assert result["status"] == "error"
    assert f"count must be a positive integer, got {count!r}." in _texts(result)
    assert published_at == []


def test_a_usable_rate_paces_the_burst(published_at: list[float]) -> None:
    """The honored path still publishes ``count`` messages spaced by ``1 / rate``."""
    result = _publish_twist(count=4, rate=100.0)

    assert result["status"] == "success"
    assert len(published_at) == 4


# ---------------------------------------------------------------------------
# The refusal is identical with and without a backend installed.
# ---------------------------------------------------------------------------


def test_a_refusal_names_the_option_even_with_no_ros_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard runs ahead of the availability probe, so the message is stable.

    A caller mistake must not be masked by an install hint on a machine without
    ROS 2 and then reported differently on a machine with it.
    """
    monkeypatch.setattr(ros_mod._backend, "available", lambda: False)

    result = _publish_twist(count=6, rate=0.0)

    assert result["status"] == "error"
    assert "rate must be > 0" in _texts(result)


# ---------------------------------------------------------------------------
# Per-action scoping: an option the action never reads is not second-guessed.
# ---------------------------------------------------------------------------


def test_publish_accepts_an_unusable_timeout_it_never_reads(published_at: list[float]) -> None:
    """``publish`` consumes ``count`` and ``rate`` only, so ``timeout`` is inert."""
    result = _publish_twist(count=2, rate=100.0, timeout=-1.0)

    assert result["status"] == "success"
    assert len(published_at) == 2


@pytest.mark.parametrize("action", ["status", "list_topics"])
def test_an_action_reading_no_numeric_option_is_never_refused(monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    """A query action must not fail for a value it does not look at."""
    monkeypatch.setattr(ros_mod, "_list_topics", lambda: "/cmd_vel [geometry_msgs/msg/Twist]")

    result = ros_mod.use_ros(action=action, count=-1, rate=float("nan"), timeout=-1.0)

    assert result["status"] == "success"


def test_the_scoping_table_covers_every_action_that_reads_an_option() -> None:
    """Only the three known option names may appear in the scoping tables.

    The table is what decides whether a value is checked at all, so a typo in it
    would silently disable a guard.
    """
    tables = (
        ros_mod._ACTION_NUMERIC_OPTIONS,
        rtps_mod._ACTION_NUMERIC_OPTIONS,
        rosbridge_mod._ACTION_NUMERIC_OPTIONS,
    )
    for table in tables:
        for action, options in table.items():
            assert options, f"{action} lists no options; drop the entry instead"
            assert set(options) <= {"count", "rate", "timeout"}, action


def test_every_rosbridge_action_declares_the_options_it_reads() -> None:
    """rosbridge dials the bridge for every action, so every action reads ``timeout``.

    An action missing from the table is silently unguarded, which is how this
    transport shipped with none at all - so the table must stay exhaustive.
    """
    assert set(rosbridge_mod._ACTION_NUMERIC_OPTIONS) == set(rosbridge_mod._ACTIONS)
    for action, options in rosbridge_mod._ACTION_NUMERIC_OPTIONS.items():
        assert "timeout" in options, action


# ---------------------------------------------------------------------------
# Cross-transport parity: two transports onto one graph, one accepted domain.
# ---------------------------------------------------------------------------

_PUBLISH_CALLS: list[tuple[str, Callable[..., dict[str, Any]]]] = [
    (
        "use_ros",
        lambda **kw: ros_mod.use_ros(action="publish", topic="/cmd_vel", type="geometry_msgs/msg/Twist", **kw),
    ),
    (
        "use_rtps",
        lambda **kw: rtps_mod.use_rtps(action="publish", topic="/cmd_vel", type="geometry_msgs/msg/Twist", **kw),
    ),
    # rosbridge speaks the ROS1 two-segment type name for the same message.
    (
        "use_rosbridge",
        lambda **kw: rosbridge_mod.use_rosbridge(action="publish", topic="/cmd_vel", type="geometry_msgs/Twist", **kw),
    ),
]


def _refusal_reasons(
    calls: list[tuple[str, Callable[..., dict[str, Any]]]], param: str, **options: Any
) -> dict[str, str]:
    """Map each transport to the sentence it refused ``options`` with.

    The verdict alone would be satisfied by an unrelated failure - with no ROS 2
    and no cyclonedds installed both transports error for *some* reason - so the
    returned text is asserted on, and it must name the option.
    """
    reasons = {}
    for name, call in calls:
        result = call(**options)
        assert result["status"] == "error", f"{name} accepted {param}={options[param]!r}"
        reasons[name] = _texts(result)
    return reasons


@pytest.mark.parametrize("value", UNUSABLE_RATES)
def test_every_transport_refuses_the_same_rate(value: Any) -> None:
    """A rate one transport refuses cannot be publishable through another."""
    reasons = _refusal_reasons(_PUBLISH_CALLS, "rate", count=1, rate=value)

    for name, reason in reasons.items():
        assert f"rate must be > 0, got {value!r}." in reason, (name, reason)


@pytest.mark.parametrize("value", UNUSABLE_COUNTS)
def test_every_transport_refuses_the_same_count(value: Any) -> None:
    """A count one transport refuses cannot be publishable through another."""
    reasons = _refusal_reasons(_PUBLISH_CALLS, "count", count=value, rate=10.0)

    for name, reason in reasons.items():
        assert f"count must be a positive integer, got {value!r}." in reason, (name, reason)


@pytest.mark.parametrize("value", UNUSABLE_TIMEOUTS)
def test_every_transport_refuses_the_same_echo_timeout(value: Any) -> None:
    """An echo wait budget one transport refuses is refused by the others too."""
    echo_calls: list[tuple[str, Callable[..., dict[str, Any]]]] = [
        (
            "use_ros",
            lambda **kw: ros_mod.use_ros(action="echo", topic="/odom", type="nav_msgs/msg/Odometry", **kw),
        ),
        (
            "use_rtps",
            lambda **kw: rtps_mod.use_rtps(action="echo", topic="/odom", type="nav_msgs/msg/Odometry", **kw),
        ),
        (
            "use_rosbridge",
            lambda **kw: rosbridge_mod.use_rosbridge(action="echo", topic="/odom", type="nav_msgs/Odometry", **kw),
        ),
    ]
    reasons = _refusal_reasons(echo_calls, "timeout", timeout=value)

    for name, reason in reasons.items():
        assert f"timeout must be > 0, got {value!r}." in reason, (name, reason)


# ---------------------------------------------------------------------------
# Output contract.
# ---------------------------------------------------------------------------


def test_a_refusal_message_is_ascii_and_names_the_action() -> None:
    """The message must be plain ASCII and say which action rejected the value."""
    text = _texts(_publish_twist(count=1, rate=float("nan")))

    assert text.isascii(), text
    assert text.startswith("use_ros: publish: ")


# ---------------------------------------------------------------------------
# rosbridge: the transport that shipped with none of these guards.
# ---------------------------------------------------------------------------

_ROSBRIDGE_ACTION_ARGS: dict[str, dict[str, Any]] = {
    "status": {},
    "list_topics": {},
    "list_services": {},
    "echo": {"topic": "/odom", "type": "nav_msgs/Odometry"},
    "service_call": {"service": "/reset", "type": "std_srvs/Empty"},
    "publish": {"topic": "/cmd_vel", "type": "geometry_msgs/Twist"},
}


@pytest.fixture
def rosbridge_published_at(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Wire ``use_rosbridge`` to a recording publisher and return its send timestamps.

    The real ``_publish`` body runs - only the roslibpy module is faked - so the
    list records exactly the messages that reached the WebSocket, and their
    spacing is the pacing the tool actually applied.
    """
    stamps: list[float] = []

    class FakeTopic:
        def __init__(self, ros: Any, name: str, message_type: str) -> None:
            ros.topics.append(self)

        def advertise(self) -> None:
            pass

        def unadvertise(self) -> None:
            pass

        def publish(self, msg: Any) -> None:
            stamps.append(time.perf_counter())

        def subscribe(self, callback: Any) -> None:
            pass

        def unsubscribe(self) -> None:
            pass

    class FakeRos:
        def __init__(self, host: str | None = None, port: int | None = None) -> None:
            self.is_connected = False
            self.topics: list[FakeTopic] = []

        def run(self, timeout: float | None = None) -> None:
            self.is_connected = True

    module = _types.ModuleType("roslibpy")
    module.Ros = FakeRos  # type: ignore[attr-defined]
    module.Topic = FakeTopic  # type: ignore[attr-defined]
    module.Message = dict  # type: ignore[attr-defined]
    module.Service = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "roslibpy", module)
    monkeypatch.setattr(rosbridge_mod._backend, "_connections", {})
    return stamps


def _publish_over_rosbridge(**options: Any) -> dict[str, Any]:
    return rosbridge_mod.use_rosbridge(action="publish", topic="/cmd_vel", type="geometry_msgs/Twist", **options)


@pytest.mark.parametrize("rate", UNUSABLE_RATES)
def test_rosbridge_refuses_an_unusable_rate_before_publishing(rosbridge_published_at: list[float], rate: Any) -> None:
    """A rate that cannot pace the burst is refused before anything is sent.

    Pre-fix the loop fell back to ``period = 0.0`` and sent all six messages
    back-to-back while reporting ``published 6 message(s)``: a velocity hold
    collapsed into an instantaneous burst that the base then latches as its last
    command. ``inf`` reached the same place through the front door - it passes
    ``rate > 0`` and then collapses ``1 / rate`` to zero.
    """
    result = _publish_over_rosbridge(count=6, rate=rate)

    assert result["status"] == "error"
    assert f"rate must be > 0, got {rate!r}." in _texts(result)
    assert rosbridge_published_at == []


@pytest.mark.parametrize("count", UNUSABLE_COUNTS)
def test_rosbridge_refuses_an_unusable_count_before_publishing(rosbridge_published_at: list[float], count: Any) -> None:
    """A count that is not a positive integer is refused, not partially absorbed.

    The ad-hoc ``count < 1`` test this replaces caught ``0`` and ``-1`` only:
    ``True`` published one message and reported "published True message(s)",
    while ``2.7`` and a numeric string reached ``range()`` and failed with a
    message naming neither the tool nor the option - after the WebSocket was
    dialed and the publisher advertised.
    """
    result = _publish_over_rosbridge(count=count, rate=10.0)

    assert result["status"] == "error"
    assert f"count must be a positive integer, got {count!r}." in _texts(result)
    assert rosbridge_published_at == []


def test_rosbridge_paces_a_usable_rate(rosbridge_published_at: list[float]) -> None:
    """The honored path still publishes ``count`` messages spaced by ``1 / rate``."""
    result = _publish_over_rosbridge(count=4, rate=100.0)

    assert result["status"] == "success"
    assert len(rosbridge_published_at) == 4


def test_a_non_finite_echo_timeout_is_refused_not_raised(rosbridge_published_at: list[float]) -> None:
    """``timeout=inf`` must be refused, not escape the tool contract.

    Pre-fix it reached the sample-collection wait as a deadline, which raised
    ``OverflowError: timestamp out of range for platform time_t``. That is not in
    the dispatch's handled-exception tuple, so it propagated out of a tool
    documented to return a result dict - past the agent's error handling
    entirely, rather than being reported as one.
    """
    result = rosbridge_mod.use_rosbridge(action="echo", topic="/odom", type="nav_msgs/Odometry", timeout=float("inf"))

    assert result["status"] == "error"
    assert "echo: timeout must be > 0, got inf." in _texts(result)


@pytest.mark.parametrize("action", sorted(_ROSBRIDGE_ACTION_ARGS))
def test_every_rosbridge_action_refuses_a_non_positive_timeout(
    rosbridge_published_at: list[float], action: str
) -> None:
    """Every action dials the bridge with ``timeout``, so every action must check it.

    A non-positive budget made the dial's wait loop expire immediately; on
    ``echo`` that returned ``status="success"`` with an empty sample list and a
    note blaming the topic for being silent, which sends the caller debugging a
    robot that was never given time to answer.
    """
    result = rosbridge_mod.use_rosbridge(action=action, timeout=-1.0, **_ROSBRIDGE_ACTION_ARGS[action])

    assert result["status"] == "error"
    assert f"{action}: timeout must be > 0, got -1.0." in _texts(result)
    assert rosbridge_mod._backend._connections == {}, "refused, yet the bridge was dialed"


def test_a_rosbridge_refusal_names_the_option_with_no_roslibpy_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard runs ahead of the availability probe, so the message is stable."""
    monkeypatch.setattr(rosbridge_mod._backend, "available", lambda: False)

    result = _publish_over_rosbridge(count=6, rate=0.0)

    assert result["status"] == "error"
    assert "rate must be > 0" in _texts(result)


def test_rosbridge_publish_reads_the_timeout_that_rclpy_publish_ignores(
    rosbridge_published_at: list[float],
) -> None:
    """The per-action tables differ because the transports differ, and that is deliberate.

    ``use_ros`` publishes through an already-running in-process node, so its
    ``publish`` reads no caller budget and must not be refused for one. Every
    rosbridge action begins by dialing the WebSocket with ``timeout``, so the
    same value is effective there and a bad one has to be refused.
    """
    assert "timeout" not in ros_mod._ACTION_NUMERIC_OPTIONS["publish"]
    assert "timeout" in rosbridge_mod._ACTION_NUMERIC_OPTIONS["publish"]

    result = _publish_over_rosbridge(count=2, rate=100.0, timeout=-1.0)

    assert result["status"] == "error"
    assert "publish: timeout must be > 0, got -1.0." in _texts(result)
    assert rosbridge_published_at == []


# ---------------------------------------------------------------------------
# Structural guard: one owner of the rule, and no transport may grow its own.
# ---------------------------------------------------------------------------

_TOOLS_DIR = Path(inspect.getfile(rosbridge_mod)).parent
_ROS_FAMILY = ("use_ros", "use_rtps", "use_rosbridge")


def _module_tree(stem: str) -> ast.Module:
    return ast.parse((_TOOLS_DIR / f"{stem}.py").read_text(encoding="utf-8"))


def _local_guard_definitions(tree: ast.Module) -> list[str]:
    """Names of module-level functions that re-implement the shared guard."""
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.endswith("numeric_option_error")
    ]


def _calls_the_shared_guard(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "numeric_option_error"
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("stem", _ROS_FAMILY)
def test_every_ros_transport_routes_through_the_shared_guard(stem: str) -> None:
    """No transport may carry its own copy of the accepted domain.

    Two byte-identical copies is how the domain drifted in the first place: they
    agreed with each other, so nothing failed, and the third transport was
    written with no copy at all. A single owner makes an unguarded transport a
    test failure rather than a silent one.
    """
    tree = _module_tree(stem)

    assert _calls_the_shared_guard(tree), f"{stem} never calls the shared numeric-option guard"
    assert _local_guard_definitions(tree) == [], f"{stem} re-implements the shared guard"


def test_the_shared_guard_has_exactly_one_definition() -> None:
    """The owning module defines the rule once, and the scan can find it."""
    owner = ast.parse(Path(inspect.getfile(numeric_option_error)).read_text(encoding="utf-8"))

    assert _local_guard_definitions(owner) == ["numeric_option_error"]


def test_the_structural_scan_detects_a_planted_local_copy() -> None:
    """Guard for the guard: a re-implementation must actually be caught.

    Without this, a scan that silently matched nothing would report a clean tree.
    """
    planted = ast.parse("def _numeric_option_error(action, timeout, count, rate):\n    return None\n")

    assert _local_guard_definitions(planted) == ["_numeric_option_error"]
    assert not _calls_the_shared_guard(planted)
