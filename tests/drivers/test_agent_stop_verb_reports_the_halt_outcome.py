"""The agent-facing ``stop`` verb reports the halt outcome it establishes.

:data:`~strands_robots.drivers.base.DRIVER_SURFACE` carries two ways to halt a
robot, and they are not the same contract.  ``stop`` is the protocol's shutdown
hook: it is annotated ``-> None`` on every shipped driver, so it *cannot* carry a
verdict, and both daemon drivers use that freedom to swallow a failure - the
Microduck returns early for a client that is gone and logs an ``OSError`` from
robotd, the Mini logs a daemon that declined.  ``stop_task`` returns an envelope
and *decides*: it refuses "not connected", "robotd refused the stop: ..." and
"daemon refused the stop: ..." on exactly those states.

The verb an **agent** reaches is ``stream({"action": "stop"})``, and on both
drivers it built its envelope beside ``await self.stop()``.  An envelope written
next to a hook that carries no verdict can only restate the intent, so the agent
read ``status="success"`` and text asserting the daemon had been asked - on a
driver whose client was ``None``, where nothing had been written to the socket
the text names.  One driver, two stop surfaces, two contracts, and the agent got
the one that cannot say it failed.

:pr:`2828` taught :class:`~strands_robots.drivers.g1.G1Driver` to return
``stop_task()``'s envelope from that branch for this reason.  These two are the
same shape, and the fleet relation below is what makes the next driver graded on
arrival: **a driver whose ``stop_task`` can refuse must report that verdict from
the agent verb.**  ``DynamixelDriver`` and ``FeetechDriver`` are exempt and the
exemption is *derived* rather than listed - their ``stop_task`` has no refusal
path at all, because their serial bus is not wired, so there is no verdict to
report.

Why nothing caught it: the Mini's own suite already argues the principle.
``test_a_daemon_that_refuses_the_stop_does_not_report_it_as_halted`` carries the
comment "Reporting a halt that did not happen is an affirmative lie on a safety
path" - and grades ``stop``'s effect on the ``motion_stopped`` *flag*.
``test_stop_task_reports_a_daemon_refusal`` grades ``stop_task``'s *envelope*.
The one surface where that lie is actually told to an agent was the one not
graded, and the Microduck's ``stream`` was uncovered outright.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from typing import Any

import pytest

import strands_robots.drivers as drivers_pkg
import strands_robots.drivers.microduck as microduck_mod
from strands_robots.drivers.base import DRIVER_SURFACE
from strands_robots.drivers.microduck import MicroduckDriver
from strands_robots.drivers.registry import get_native_driver_class

# Reused wholesale from the Mini's own suite: the daemon double, the recording
# link and the connect helper.  ``_install``/``_connected``/``_run_tool`` are
# plain functions rather than fixtures, so importing them binds no fixture name.
from tests.drivers.test_reachy_driver import (
    _connected,
    _run_tool,
    _text,
)

_SOCKET = "/run/robotd-under-test.sock"

# The stale text each verb used to assert, kept verbatim so a revert is caught by
# name rather than by a status code alone.
_STALE_MICRODUCK = "asked robotd at"
_STALE_REACHY = "asked the daemon at"


# --------------------------------------------------------------------------- #
# Doubles.                                                                    #
# --------------------------------------------------------------------------- #


class _GoneClient:
    """A robotd client whose connection has dropped."""

    def __init__(self) -> None:
        self.alive = False
        self.calls: list[str] = []

    def call(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        raise AssertionError("a client that is not alive must not be written to")

    def close(self) -> None:
        return None


class _RefusingClient:
    """A live robotd client whose socket errors on the stop request."""

    def __init__(self) -> None:
        self.alive = True
        self.calls: list[str] = []

    def call(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.calls.append(method)
        raise OSError("connection reset by peer")

    def close(self) -> None:
        return None


class _RecordingClient:
    """A healthy robotd client that records the methods it was asked to send."""

    def __init__(self) -> None:
        self.alive = True
        self.calls: list[str] = []

    def call(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        self.calls.append(method)
        return {"result": {}}

    def close(self) -> None:
        return None


def _duck(client: Any) -> MicroduckDriver:
    """Build a Microduck driver holding ``client`` as its robotd connection."""
    driver = MicroduckDriver(port=_SOCKET)
    driver._client = client
    return driver


def _stop_envelope(driver: Any) -> dict[str, Any]:
    """Drive one ``stream({"action": "stop"})`` call and return its envelope."""

    async def _drive() -> dict[str, Any]:
        results = [
            event
            async for event in driver.stream(
                {"toolUseId": "call-1", "name": "t", "input": {"action": "stop"}},
                {},
            )
        ]
        assert len(results) == 1, f"the verb must yield exactly one result, got {len(results)}"
        return results[0]

    return asyncio.run(_drive())


def _envelope_text(envelope: dict[str, Any]) -> str:
    """Join every text block, for substring assertions."""
    return " ".join(block.get("text", "") for block in envelope.get("content", []) if "text" in block)


# --------------------------------------------------------------------------- #
# Derived fleet relation.                                                     #
# --------------------------------------------------------------------------- #


def _driver_classes() -> dict[str, type]:
    """Every distinct native driver class the registry can build."""
    return {
        cls.__name__: cls
        for cls in (get_native_driver_class(robot) for robot in drivers_pkg.list_native_drivers())
        if cls is not None
    }


def _method_ast(cls: type, name: str) -> ast.AST:
    """Parse one method of ``cls`` into an AST node."""
    return ast.parse(textwrap.dedent(inspect.getsource(getattr(cls, name)))).body[0]


def _stop_task_can_refuse(node: ast.AST) -> bool:
    """Does this ``stop_task`` body have a path that reports a non-success?

    A refusal is spelled either through the module's ``_refuse`` helper or as a
    literal ``"error"`` status, so both are read.
    """
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "_refuse":
            return True
        if isinstance(inner, ast.Constant) and inner.value == "error":
            return True
    return False


def _stream_stop_branch(node: ast.AST) -> str:
    """The source of the final ``else`` of a ``stream`` action dispatch."""
    branch: list[ast.stmt] | None = None
    for inner in ast.walk(node):
        if isinstance(inner, ast.If) and inner.orelse and not any(isinstance(x, ast.If) for x in inner.orelse):
            branch = inner.orelse
    if branch is None:
        return ""
    return ast.unparse(ast.Module(body=branch, type_ignores=[]))


def _reports_the_verdict(branch: str) -> bool:
    """Does the stop branch return the sibling's envelope rather than build one?"""
    return "stop_task()" in branch


# --------------------------------------------------------------------------- #
# Premises.                                                                   #
# --------------------------------------------------------------------------- #


class TestThePremise:
    """Why an envelope built beside ``stop`` can only restate the intent."""

    def test_the_shutdown_hook_is_part_of_the_protocol(self) -> None:
        # The branch no longer calling ``stop`` is a change of reporting, not a
        # deletion: ``stop`` remains a member every driver must implement.
        assert "stop" in DRIVER_SURFACE
        assert "stop_task" in DRIVER_SURFACE

    @pytest.mark.parametrize("name", sorted(_driver_classes()))
    def test_the_shutdown_hook_carries_no_verdict(self, name: str) -> None:
        node = _method_ast(_driver_classes()[name], "stop")
        assert isinstance(node, ast.AsyncFunctionDef)
        assert node.returns is not None
        assert ast.unparse(node.returns) == "None", f"{name}.stop must be the verdict-free hook"

    def test_the_sibling_refuses_the_states_the_hook_swallows(self) -> None:
        # The verdict exists; it was simply not the one the agent was handed.
        assert _duck(None).stop_task()["status"] == "error"
        assert _duck(_RefusingClient()).stop_task()["status"] == "error"

    def test_the_client_doubles_expose_what_the_driver_reads(self) -> None:
        # A double more permissive than the real client could not see this.
        real = microduck_mod._RobotdClient(_SOCKET)
        for double in (_GoneClient(), _RefusingClient(), _RecordingClient()):
            for member in ("alive", "call", "close"):
                assert hasattr(real, member) and hasattr(double, member)
        assert inspect.signature(_RecordingClient.call) == inspect.signature(microduck_mod._RobotdClient.call)


# --------------------------------------------------------------------------- #
# Regression - the two drivers that had a verdict and discarded it.           #
# --------------------------------------------------------------------------- #


class TestTheMicroduckVerbReportsTheHaltOutcome:
    """A stop robotd never received is reported, not claimed."""

    @pytest.mark.parametrize(
        ("label", "make_client", "expected"),
        [
            ("no-client", lambda: None, "not connected"),
            ("client-gone", _GoneClient, "not connected"),
            ("robotd-refused", _RefusingClient, "robotd refused the stop"),
        ],
    )
    def test_the_verb_reports_an_error(self, label: str, make_client: Any, expected: str) -> None:
        envelope = _stop_envelope(_duck(make_client()))
        assert envelope["status"] == "error", label

    @pytest.mark.parametrize(
        ("label", "make_client", "expected"),
        [
            ("no-client", lambda: None, "not connected"),
            ("client-gone", _GoneClient, "not connected"),
            ("robotd-refused", _RefusingClient, "robotd refused the stop"),
        ],
    )
    def test_the_text_names_the_reason(self, label: str, make_client: Any, expected: str) -> None:
        assert expected in _envelope_text(_stop_envelope(_duck(make_client())))

    @pytest.mark.parametrize("make_client", [lambda: None, _GoneClient, _RefusingClient])
    def test_the_text_no_longer_claims_the_socket_was_asked(self, make_client: Any) -> None:
        # The stale text named ``_socket_path`` - a socket nothing was written to.
        text = _envelope_text(_stop_envelope(_duck(make_client())))
        assert _STALE_MICRODUCK not in text
        assert _SOCKET not in text


class TestTheMiniVerbReportsTheHaltOutcome:
    """A recorded move the daemon declined to stop is reported, not claimed."""

    def test_the_verb_reports_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch, stop_result={"error": "busy"})
        assert _run_tool(driver, "stop")["status"] == "error"

    def test_the_text_names_the_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch, stop_result={"error": "busy"})
        text = _text(_run_tool(driver, "stop"))
        assert "daemon refused the stop" in text
        assert "busy" in text

    def test_the_text_no_longer_claims_the_daemon_was_asked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch, stop_result={"error": "busy"})
        assert _STALE_REACHY not in _text(_run_tool(driver, "stop"))


# --------------------------------------------------------------------------- #
# Controls - a stop that landed still reports success, and still lands.       #
# --------------------------------------------------------------------------- #


class TestAStopThatLandedStillReportsSuccess:
    """The healthy path is unchanged, wire call included."""

    def test_the_microduck_verb_reports_success(self) -> None:
        assert _stop_envelope(_duck(_RecordingClient()))["status"] == "success"

    def test_the_microduck_stop_reached_the_wire(self) -> None:
        client = _RecordingClient()
        driver = _duck(client)
        _stop_envelope(driver)
        assert client.calls == [microduck_mod._M_STOP]
        assert driver._stopped is True

    def test_the_mini_verb_reports_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch)
        assert _run_tool(driver, "stop")["status"] == "success"

    def test_the_mini_stop_reached_the_daemon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, daemon, _ = _connected(monkeypatch)
        _run_tool(driver, "stop")
        assert any(call[2] == "/api/move/stop" for call in daemon.calls)
        assert driver._stopped is True

    def test_the_mini_stop_leaves_sensors_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A halted Mini is still observable - the property its own suite pins.
        driver, _, link = _connected(monkeypatch)
        _run_tool(driver, "stop")
        assert not link.stopped


class TestWhatHoldsEitherWay:
    """Recorded because they pass before and after: that is what they assert.

    The envelope's transport shape is unchanged - only its verdict moved - and
    the Mini's ``motion_stopped`` flag was already honest about a refused stop.
    The defect was that the agent-facing envelope disagreed with that flag; these
    pin the two halves the fix must not disturb while bringing them into
    agreement.
    """

    def test_the_tool_use_id_is_still_echoed(self) -> None:
        assert _stop_envelope(_duck(None))["toolUseId"] == "call-1"

    def test_the_flag_already_said_the_motion_did_not_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _, _ = _connected(monkeypatch, stop_result={"error": "busy"})
        _run_tool(driver, "stop")
        assert asyncio.run(driver.get_status())["content"][0]["json"]["motion_stopped"] is False


class TestTheOtherActionsAreUnchanged:
    """Only the stop branch moved; the read verbs are untouched."""

    def test_sensors_still_reports_the_four_caches(self) -> None:
        driver = _duck(_RecordingClient())

        async def _drive() -> dict[str, Any]:
            return [
                event
                async for event in driver.stream({"toolUseId": "c", "name": "t", "input": {"action": "sensors"}}, {})
            ][0]

        envelope = asyncio.run(_drive())
        assert envelope["status"] == "success"
        assert sorted(envelope["content"][0]["json"]) == ["battery", "imu", "joints", "pose"]

    def test_sensors_writes_nothing_to_the_wire(self) -> None:
        client = _RecordingClient()
        driver = _duck(client)

        async def _drive() -> None:
            async for _ in driver.stream({"toolUseId": "c", "name": "t", "input": {"action": "sensors"}}, {}):
                pass

        asyncio.run(_drive())
        assert client.calls == []


# --------------------------------------------------------------------------- #
# The fleet relation.                                                         #
# --------------------------------------------------------------------------- #


class TestTheVerdictIsSingleSourced:
    """A driver that can refuse a stop reports that verdict from the agent verb."""

    def test_the_inventory_is_not_empty(self) -> None:
        classes = _driver_classes()
        assert len(classes) >= 5, f"the driver census went blind: {sorted(classes)}"
        assert {"MicroduckDriver", "ReachyDriver", "G1Driver"} <= set(classes)

    def test_every_driver_that_can_refuse_a_stop_reports_it(self) -> None:
        offenders = []
        for name, cls in sorted(_driver_classes().items()):
            if not _stop_task_can_refuse(_method_ast(cls, "stop_task")):
                continue
            if not _reports_the_verdict(_stream_stop_branch(_method_ast(cls, "stream"))):
                offenders.append(name)
        assert offenders == [], (
            "these drivers decide a stop verdict in stop_task and then discard it in the agent verb, "
            f"so an agent reads success for a halt that did not happen: {offenders}"
        )

    def test_the_exemption_is_derived_from_the_absence_of_a_verdict(self) -> None:
        # Not a hand-listed allowlist: these two are exempt because their
        # stop_task has no refusal path, their serial bus being unwired.
        exempt = sorted(
            name for name, cls in _driver_classes().items() if not _stop_task_can_refuse(_method_ast(cls, "stop_task"))
        )
        assert exempt == ["DynamixelDriver", "FeetechDriver"], exempt

    @pytest.mark.parametrize("name", ["MicroduckDriver", "ReachyDriver", "G1Driver"])
    def test_the_branch_does_not_restate_a_verdict(self, name: str) -> None:
        branch = _stream_stop_branch(_method_ast(_driver_classes()[name], "stream"))
        assert "success" not in branch, f"{name} re-derives a verdict the sibling already decided"


class TestTheRuleIsNotVacuous:
    """The relation is graded on constructed sources, not only on a clean tree."""

    _COMPLIANT = """
        async def stream(self, tool_use, invocation_state, **kwargs):
            if action == "sensors":
                envelope = {"status": "ok"}
            elif action == "status":
                envelope = {"status": "ok"}
            else:
                envelope = self.stop_task()
            yield envelope
    """

    _VIOLATING = """
        async def stream(self, tool_use, invocation_state, **kwargs):
            if action == "sensors":
                envelope = {"status": "ok"}
            elif action == "status":
                envelope = {"status": "ok"}
            else:
                await self.stop()
                envelope = {"status": "success", "content": [{"text": "asked it to stop"}]}
            yield envelope
    """

    def test_a_compliant_branch_is_accepted(self) -> None:
        node = ast.parse(textwrap.dedent(self._COMPLIANT)).body[0]
        assert _reports_the_verdict(_stream_stop_branch(node)) is True

    def test_a_branch_built_beside_the_hook_is_refused(self) -> None:
        node = ast.parse(textwrap.dedent(self._VIOLATING)).body[0]
        assert _reports_the_verdict(_stream_stop_branch(node)) is False

    def test_the_predicate_reaches_both_outcomes(self) -> None:
        outcomes = {
            _reports_the_verdict(_stream_stop_branch(ast.parse(textwrap.dedent(src)).body[0]))
            for src in (self._COMPLIANT, self._VIOLATING)
        }
        assert outcomes == {True, False}

    def test_a_stop_task_without_a_refusal_is_recognised(self) -> None:
        no_verdict = 'def stop_task(self):\n    return {"status": "success", "content": []}\n'
        with_verdict = 'def stop_task(self):\n    if not self._alive:\n        return _refuse("nope")\n    return {}\n'
        assert _stop_task_can_refuse(ast.parse(no_verdict).body[0]) is False
        assert _stop_task_can_refuse(ast.parse(with_verdict).body[0]) is True
