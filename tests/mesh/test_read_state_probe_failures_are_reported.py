"""A ``_read_state`` probe that fails is reported, exactly once per category.

:meth:`strands_robots.mesh.core.Mesh._read_state` probes a robot driver
defensively so that a flaky read cannot kill the state thread: every probe is
wrapped in ``except Exception``. That breadth is deliberate and stays. What did
not stay is the handler body -- each one was a bare ``pass``, so a probe that
raised left no trace at any log level.

The consequence is not a missing line in a log nobody reads. ``_read_state``
returns ``None`` when nothing but ``peer_id`` and ``t`` survived, and the state
loop publishes nothing for a ``None``, so a peer whose joint probe raises stops
publishing state entirely while its presence broadcast still advertises it as
connected. It is indistinguishable, on the wire and in the log, from a healthy
peer that simply has no joints to report.

The loop's own report cannot cover this. ``_state_loop`` wraps the call in
``except Exception`` and logs at debug, but every statement in ``_read_state``
that can raise is already inside one of the probe ``try`` blocks, so nothing
reaches the loop's handler except a failure of ``publish`` itself. These tests
pin the contract that replaced the silence:

* the first failure of each category is logged at WARNING, naming the category,
  the peer and the exception's ``repr``;
* later failures of the same category drop to debug, because the loop retries at
  ``STATE_HZ`` and a persistent fault would otherwise emit ten warnings a second;
* a healthy peer's snapshot is byte-for-byte what it always was.

The log was the whole of that first fix, and reporting a fault only where the
peer's own process can see it left the harm above standing: an observer still
had to read that peer's log to explain an absent section, so the fleet dashboard
grew a regex over mesh's log lines and used it as an API. So the snapshot now
carries the same verdict the log does, in a ``degraded`` block keyed by category
-- ``reason`` (the exception's type name, which the reporter's own docstring
names as the discriminator that selects the operator's next move), ``detail``,
``failures`` and ``for_seconds``. Two consequences are pinned below: while a
probe fails the snapshot says so, and because it says so the snapshot is no
longer empty, so the hardware-only peer above keeps publishing instead of going
silent. See :mod:`tests.mesh.test_state_degraded_probes_are_published` for the
block's own contract.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest

from strands_robots.mesh.core import Mesh

CORE_LOGGER = "strands_robots.mesh.core"

#: Every category :meth:`Mesh._warn_read_state_once` is called with, taken from
#: the source rather than restated, so a fifth probe cannot be added silently.
EXPECTED_CATEGORIES = {"hw_joints", "task_state", "sim_world", "sim_joints"}


class _RaisingInner:
    """A connected hardware driver whose ``get_observation`` always raises."""

    is_connected = True

    def __init__(self, exc: BaseException) -> None:
        self.config = SimpleNamespace(cameras={})
        self._exc = exc
        self.calls = 0

    def get_observation(self) -> dict[str, Any]:
        self.calls += 1
        raise self._exc


class _Host:
    """A robot-shaped host: only the attributes ``_read_state`` reads.

    ``_task_state`` is declared rather than only assigned by the tests that need
    it, because ``_read_state`` reaches it through ``getattr(r, "_task_state",
    None)`` -- an absent attribute is a supported shape, not an error.
    """

    _task_state: Any = None

    def __init__(self, inner: Any = None) -> None:
        self.robot = inner


class _RaisingTaskState:
    """A task-state object whose ``status`` raises when read."""

    @property
    def status(self) -> Any:
        raise RuntimeError("task state unavailable")


def _mesh(host: Any, peer_id: str = "arm") -> Mesh:
    """A ``Mesh`` wired to ``host`` without opening a session.

    ``__new__`` because ``Mesh.__init__`` starts subscriber threads and wants a
    live Zenoh session; ``_read_state`` needs only ``robot`` and ``peer_id``.
    That also exercises the ``getattr`` default in ``_warn_read_state_once``:
    bookkeeping must not raise inside the handler that exists to report a failure.
    """
    m = Mesh.__new__(Mesh)
    m.peer_id = peer_id
    m.robot = host
    return m


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _debugs(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]


class TestAFailedProbeIsReported:
    """A probe that raises names itself, once, at WARNING."""

    @pytest.mark.parametrize(
        "exc",
        [
            RuntimeError("this arm has no calibration registered"),
            ConnectionError("Port is in use!"),
            OSError("[Errno 6] Device not configured"),
        ],
        ids=["uncalibrated", "port-contention", "unplugged"],
    )
    def test_a_raising_joint_probe_is_warned_about(self, caplog: pytest.LogCaptureFixture, exc: BaseException) -> None:
        """The joint probe is the one whose silence hid two live arms for hours.

        The exception type is part of the message because it selects the
        operator's next move: a contended port is a different job from an
        uncalibrated arm, and both used to arrive as nothing at all.
        """
        m = _mesh(_Host(_RaisingInner(exc)))
        with caplog.at_level(logging.DEBUG, logger=CORE_LOGGER):
            out = m._read_state()

        assert out is not None, "a probe failure is a diagnosis to publish, not a reason to go quiet"
        assert "joints" not in out, "the failed section is still omitted"
        assert out["degraded"]["hw_joints"]["reason"] == type(exc).__name__, (
            "the wire must carry the same discriminator the log does"
        )
        warned = _warnings(caplog)
        assert len(warned) == 1, f"expected exactly one warning, got {warned}"
        assert "hw_joints" in warned[0]
        assert "arm" in warned[0], "the peer id must be in the line: a fleet has many arms"
        assert type(exc).__name__ in warned[0], f"the exception type must survive into the report, got {warned[0]!r}"

    def test_a_raising_task_probe_is_warned_about_and_keeps_the_joints(self, caplog: pytest.LogCaptureFixture) -> None:
        """One failed probe must not cost the sections that DID work."""
        host = _Host(_RaisingInner(RuntimeError("unused")))
        host.robot = SimpleNamespace(
            is_connected=True,
            config=SimpleNamespace(cameras={}),
            get_observation=lambda: {"j1": 0.5},
        )
        host._task_state = _RaisingTaskState()
        m = _mesh(host)
        with caplog.at_level(logging.DEBUG, logger=CORE_LOGGER):
            out = m._read_state()

        assert out is not None
        assert out["joints"] == {"j1": 0.5}, "the readable section still publishes"
        assert "task" not in out, "the failed section is still omitted"
        assert set(out["degraded"]) == {"task_state"}, "only the probe that failed is named"
        warned = _warnings(caplog)
        assert len(warned) == 1, f"expected exactly one warning, got {warned}"
        assert "task_state" in warned[0]

    def test_a_raising_world_probe_is_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        """A sim peer's world back-reference can raise on attribute access."""

        class _RaisingWorldHost:
            robot = None

            @property
            def _world(self) -> Any:
                raise RuntimeError("world unavailable")

        m = _mesh(_RaisingWorldHost(), peer_id="sim")
        with caplog.at_level(logging.DEBUG, logger=CORE_LOGGER):
            out = m._read_state()

        assert out is not None, "a probe failure is a diagnosis to publish, not a reason to go quiet"
        assert out["degraded"]["sim_world"]["reason"] == "RuntimeError"
        warned = _warnings(caplog)
        assert len(warned) == 1, f"expected exactly one warning, got {warned}"
        assert "sim_world" in warned[0]


class TestTheReportIsOncePerCategory:
    """``STATE_HZ`` is 10, so a persistent fault must not warn ten times a second."""

    def test_repeat_failures_of_one_category_drop_to_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        inner = _RaisingInner(RuntimeError("joint bus read failed"))
        m = _mesh(_Host(inner))
        with caplog.at_level(logging.DEBUG, logger=CORE_LOGGER):
            for _ in range(5):
                m._read_state()

        assert inner.calls == 5, "premise: the probe really ran five times"
        assert len(_warnings(caplog)) == 1, "the fault is announced once, not per tick"
        assert len(_debugs(caplog)) == 4, "the later ticks are still recorded, at debug"
        assert all("still failing" in d for d in _debugs(caplog))

    def test_two_categories_are_reported_independently(self, caplog: pytest.LogCaptureFixture) -> None:
        """Suppressing one category must not suppress a different fault."""
        host = _Host(_RaisingInner(RuntimeError("joint bus read failed")))
        host._task_state = _RaisingTaskState()
        m = _mesh(host)
        with caplog.at_level(logging.DEBUG, logger=CORE_LOGGER):
            m._read_state()
            m._read_state()

        warned = _warnings(caplog)
        assert len(warned) == 2, f"one warning per category, got {warned}"
        assert {"hw_joints", "task_state"} == {c for c in ("hw_joints", "task_state") if any(c in w for w in warned)}


class TestAHealthyProbeStaysSilent:
    """The report exists for failures; a working arm must not grow a log line."""

    def test_a_readable_arm_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        host = _Host(
            SimpleNamespace(
                is_connected=True,
                config=SimpleNamespace(cameras={}),
                get_observation=lambda: {"j1": 0.5, "j2": -0.25},
            )
        )
        m = _mesh(host)
        with caplog.at_level(logging.DEBUG, logger=CORE_LOGGER):
            out = m._read_state()

        assert out is not None and out["joints"] == {"j1": 0.5, "j2": -0.25}
        assert caplog.records == [], f"a healthy probe is silent, got {caplog.records}"

    def test_a_host_with_no_hardware_and_no_world_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        """A gateway peer legitimately has neither, and must not complain.

        Nothing raises on this path, so the categories are never reached -- the
        report is keyed to a thrown exception, not to an absent section.
        """
        m = _mesh(_Host(None), peer_id="gateway")
        with caplog.at_level(logging.DEBUG, logger=CORE_LOGGER):
            out = m._read_state()

        assert out is None
        assert caplog.records == [], f"an armless peer is silent, got {caplog.records}"


class TestNoProbeSwallowsItsFailure:
    """The root cause, pinned structurally so a fifth probe cannot re-open it."""

    def test_read_state_has_no_handler_that_only_passes(self) -> None:
        """A bare ``pass`` handler is the shape that made all of this silent."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(Mesh._read_state)))
        swallowed = [
            ast.unparse(handler.type) if handler.type else "bare except"
            for node in ast.walk(tree)
            if isinstance(node, ast.Try)
            for handler in node.handlers
            if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass)
        ]
        assert swallowed == [], (
            f"_read_state swallows {len(swallowed)} probe failure(s) with a bare pass "
            f"({swallowed}); the state loop's own debug handler cannot see them, because "
            "every statement that can raise is already inside a probe try block"
        )

    def test_every_probe_reports_through_the_one_helper(self) -> None:
        """Each ``try`` in ``_read_state`` routes its handler to the reporter.

        Derived from the source rather than counted, so a probe added later is
        held to the same rule instead of being able to end in ``pass`` again.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(Mesh._read_state)))
        handlers = [h for n in ast.walk(tree) if isinstance(n, ast.Try) for h in n.handlers]
        assert handlers, "premise: _read_state still probes defensively"
        for handler in handlers:
            body = ast.unparse(ast.Module(body=handler.body, type_ignores=[]))
            assert "_warn_read_state_once" in body, f"a probe handler does not report its failure: {body!r}"

    def test_the_reported_categories_are_the_documented_ones(self) -> None:
        """The categories named in the helper's docstring are the ones used."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(Mesh._read_state)))
        used = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func).endswith("_warn_read_state_once")
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        assert used == EXPECTED_CATEGORIES, f"categories drifted: {used}"
        doc = inspect.getdoc(Mesh._warn_read_state_once) or ""
        for category in EXPECTED_CATEGORIES:
            assert category in doc, f"{category} is used but not documented"
