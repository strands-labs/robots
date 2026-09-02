"""The verdict-free ``stop`` hook logs a halt it could not complete.

:data:`~strands_robots.drivers.base.DRIVER_SURFACE` carries two ways to halt a
robot and only one of them can answer. ``stop_task`` returns an envelope and
*decides*; ``stop`` is annotated ``-> None`` on every shipped driver, so it
carries no verdict at all. ``tests/drivers/test_agent_stop_verb_reports_the_halt_outcome``
grades the first consequence of that asymmetry - the agent-facing verb must hand
back ``stop_task``'s envelope rather than build one beside the hook - and its own
premise names the second: ``stop`` is verdict-free *because* it logs, "the
Microduck ... logs an ``OSError`` from robotd, the Mini logs a daemon that
declined".

That premise was never graded, and three drivers did not hold it. Each one's
``stop`` delegated to a verb that returns an envelope and dropped it on the
floor:

======================  ====================  ==================================
driver                  discarded envelope    what a refusal leaves running
======================  ====================  ==================================
``BoosterDriver``       ``stop_task()``       the T1 walking, or the host still
                                              holding the upper body
``EarthRoverDriver``    ``stop_task()``       the rover driving at the last
                                              commanded velocity
``CrazyflieDriver``     ``land()``            the aircraft flying
======================  ====================  ==================================

For the two ground robots the refusal is an ordinary hardware failure -
``BoosterDriver.move`` reports a ``RuntimeError`` from ``MoveCommand`` as
``"the T1 refused the twist"``, and ``EarthRoverDriver.send_action`` reports a
``/control`` POST that did not land. Because ``stop`` returns ``None`` and wrote
nothing, a fleet teardown that called it had **no** surface anywhere - envelope,
flag or log - saying the robot had not stopped. ``CrazyflieDriver.land`` re-checks
the link that ``stop`` just checked, so its refusal needs a disconnect racing the
two, which makes it a latent hole rather than a reproduced field failure; it is
fixed here because the rule below is derived from the code and does not have a
category for "discards a verdict, but only rarely".

The relation is what makes the next driver graded on arrival, and it is derived
rather than listed: *a ``stop`` that calls one of its own envelope-returning
verbs must read what that verb answered.* Nine of the twelve shipped drivers
already satisfied it - seven log the failure directly at the wire and two read a
delegated envelope first - so this pins the shape they already have.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import sys
import textwrap
import types
from typing import Any

import pytest

import strands_robots.drivers as drivers_pkg
from strands_robots.drivers.base import halt_failure_detail
from strands_robots.drivers.registry import get_native_driver_class

# Reused wholesale from each driver's own suite: the SDK doubles and the
# connected-driver helpers. Every name imported here is a plain class or
# function, never a fixture, so importing them binds no fixture name.
from tests.drivers.test_booster_driver import _FakeSdk
from tests.drivers.test_booster_driver import _live_driver as _live_booster
from tests.drivers.test_earthrover_driver import _FakeResponse, _FakeSession
from tests.drivers.test_earthrover_driver import _live_driver as _live_rover

#: The annotated return type that makes a driver method a status envelope.
_ENVELOPE_RETURN = "dict[str, Any]"


# --------------------------------------------------------------------------- #
# The derived relation.                                                       #
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


def _envelope_verbs(cls: type) -> set[str]:
    """Which of ``cls``'s own methods answer with a status envelope.

    Read from the annotation rather than from a name list, so a halt verb spelled
    something other than ``stop_task`` - the aircraft's ``land`` - is in scope
    without being named here.
    """
    verbs = set()
    for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction):
        try:
            node = _method_ast(cls, name)
        except (OSError, TypeError, SyntaxError):  # pragma: no cover - C or builtin
            continue
        returns = getattr(node, "returns", None)
        if returns is not None and ast.unparse(returns) == _ENVELOPE_RETURN:
            verbs.add(name)
    return verbs


def _discarded_envelopes(node: ast.AST, verbs: set[str]) -> list[str]:
    """Envelope verbs this body calls as a bare statement, throwing the answer away."""
    discarded = []
    for stmt in ast.walk(node):
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        func = stmt.value.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
            and func.attr in verbs
        ):
            discarded.append(func.attr)
    return sorted(discarded)


class TestThePremise:
    """The hook cannot answer, so the log is the only place a failure survives."""

    def test_the_census_reaches_the_whole_shipped_fleet(self) -> None:
        classes = _driver_classes()
        assert len(classes) >= 10, f"the driver census went blind: {sorted(classes)}"
        assert {"BoosterDriver", "CrazyflieDriver", "EarthRoverDriver"} <= set(classes)

    @pytest.mark.parametrize("name", sorted(_driver_classes()))
    def test_the_hook_carries_no_verdict(self, name: str) -> None:
        node = _method_ast(_driver_classes()[name], "stop")
        assert isinstance(node, ast.AsyncFunctionDef)
        assert node.returns is not None, f"{name}.stop must annotate its return"
        assert ast.unparse(node.returns) == "None", f"{name}.stop must be the verdict-free hook"

    def test_the_halt_verbs_that_can_refuse_really_do(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The verdict exists on both ground robots; the hook simply dropped it.
        sdk = _FakeSdk()
        monkeypatch.setitem(sys.modules, "booster_robotics_sdk_python", sdk)
        booster = _live_booster(sdk)
        sdk.client.refuse.add("MoveCommand")
        assert booster.stop_task()["status"] == "error"

        session = _install_requests(monkeypatch)
        rover = _live_rover(session)
        session.post_response = OSError("no route to host")
        assert rover.stop_task()["status"] == "error"


class TestEveryStopHookReadsTheHaltItDelegated:
    """The relation, over the derived fleet."""

    def test_no_stop_hook_discards_an_envelope(self) -> None:
        offenders = {}
        for name, cls in sorted(_driver_classes().items()):
            discarded = _discarded_envelopes(_method_ast(cls, "stop"), _envelope_verbs(cls))
            if discarded:
                offenders[name] = discarded
        assert offenders == {}, (
            "these stop hooks call a verb that decides a halt verdict and then discard it. "
            "stop() returns None, so nothing anywhere records that the robot did not stop: "
            f"{offenders}"
        )

    @pytest.mark.parametrize("name", ["BoosterDriver", "CrazyflieDriver", "EarthRoverDriver"])
    def test_the_halt_verb_is_in_scope_by_its_annotation(self, name: str) -> None:
        # The relation finds ``land`` for the same reason it finds ``stop_task``:
        # both answer with an envelope. Neither is named by the predicate.
        verbs = _envelope_verbs(_driver_classes()[name])
        assert "stop_task" in verbs
        assert "land" in verbs if name == "CrazyflieDriver" else True


class TestTheRelationIsNotVacuous:
    """Graded on constructed sources, so a clean tree is not the only evidence."""

    _COMPLIANT = """
        async def stop(self) -> None:
            if (detail := halt_failure_detail(self.stop_task())) is not None:
                logger.error("%s.stop(): it did not stop: %s", self._tool_name, detail)
    """
    _VIOLATING = """
        async def stop(self) -> None:
            self.stop_task()
    """

    def test_a_hook_that_reads_the_envelope_is_accepted(self) -> None:
        node = ast.parse(textwrap.dedent(self._COMPLIANT)).body[0]
        assert _discarded_envelopes(node, {"stop_task"}) == []

    def test_a_hook_that_drops_the_envelope_is_refused(self) -> None:
        node = ast.parse(textwrap.dedent(self._VIOLATING)).body[0]
        assert _discarded_envelopes(node, {"stop_task"}) == ["stop_task"]

    def test_a_call_to_something_that_answers_nothing_is_not_a_violation(self) -> None:
        # ``_halt_repeater`` returns None: there is no verdict to read, so a bare
        # call to it is not the shape this relation is about.
        node = ast.parse("async def stop(self) -> None:\n    self._halt_repeater()\n").body[0]
        assert _discarded_envelopes(node, {"stop_task", "land"}) == []


# --------------------------------------------------------------------------- #
# Regression - a refused halt is logged, naming what may still be moving.      #
# --------------------------------------------------------------------------- #


def _install_requests(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    """Install a fake ``requests`` whose ``Session()`` is one shared recorder."""
    session = _FakeSession()
    module = types.ModuleType("requests")
    module.Session = lambda: session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", module)
    return session


def _errors(caplog: pytest.LogCaptureFixture) -> str:
    """Every ERROR-or-worse message captured, joined for substring assertions."""
    return " ".join(record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR)


def _refusing_booster(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A T1 whose halting twist the locomotion controller refuses."""
    sdk = _FakeSdk()
    monkeypatch.setitem(sys.modules, "booster_robotics_sdk_python", sdk)
    driver = _live_booster(sdk)
    sdk.client.refuse.add("MoveCommand")
    return driver


def _refusing_rover(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A rover whose zero twist cannot leave the host."""
    session = _install_requests(monkeypatch)
    driver = _live_rover(session)
    session.post_response = OSError("no route to host")
    return driver


def _refusing_rover_http(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A second door to the same failure: the SDK answered, and refused."""
    session = _install_requests(monkeypatch)
    driver = _live_rover(session)
    session.post_response = _FakeResponse(503, None, "controller busy")
    return driver


def _refusing_aircraft(monkeypatch: pytest.MonkeyPatch) -> Any:
    """An aircraft whose descent is refused.

    Stubbed rather than driven: ``land`` re-checks the link ``stop`` just
    checked, so reaching its refusal for real needs a disconnect racing the two.
    What is under test is that the hook reads the answer it is given.
    """
    driver = _flying(monkeypatch)
    monkeypatch.setattr(
        driver, "land", lambda **_: {"status": "error", "content": [{"text": "land: the link went away"}]}
    )
    return driver


def _flying(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A Crazyflie driver that reports an open link."""
    from strands_robots.drivers import crazyflie as module

    driver = module.CrazyflieDriver()
    monkeypatch.setattr(driver, "_connected", True)
    monkeypatch.setattr(driver, "_cf", object())
    assert driver.is_connected
    return driver


def _healthy_booster(monkeypatch: pytest.MonkeyPatch) -> Any:
    sdk = _FakeSdk()
    monkeypatch.setitem(sys.modules, "booster_robotics_sdk_python", sdk)
    return _live_booster(sdk)


def _healthy_rover(monkeypatch: pytest.MonkeyPatch) -> Any:
    return _live_rover(_install_requests(monkeypatch))


def _healthy_aircraft(monkeypatch: pytest.MonkeyPatch) -> Any:
    driver = _flying(monkeypatch)
    monkeypatch.setattr(driver, "land", lambda **_: {"status": "success", "content": []})
    return driver


class TestARefusedHaltIsLoggedNamingWhatMayStillMove:
    """The regression: on ``main`` every one of these logged nothing at all."""

    @pytest.mark.parametrize(
        ("label", "build", "expected"),
        [
            ("the T1 refused the halting twist", _refusing_booster, ("may still be walking",)),
            # The T1's halt has two halves and the envelope reports each, so an
            # operator learns it was the locomotion half that did not land.
            ("the failing half is named", _refusing_booster, ("locomotion_halted=False",)),
            (
                "the rover's zero twist could not send",
                _refusing_rover,
                ("may still be driving at the commanded velocity", "no route to host"),
            ),
            ("the rover's SDK answered and refused", _refusing_rover_http, ("may still be driving", "HTTP 503")),
            ("the descent was refused", _refusing_aircraft, ("may still be flying", "the link went away")),
        ],
    )
    def test_the_hook_logs_an_error(
        self,
        label: str,
        build: Any,
        expected: tuple[str, ...],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        driver = build(monkeypatch)
        with caplog.at_level(logging.DEBUG):
            assert asyncio.run(driver.stop()) is None, "the hook still carries no verdict"
        errors = _errors(caplog)
        for fragment in expected:
            assert fragment in errors, f"{label}: {fragment!r} not in {errors!r}"

    @pytest.mark.parametrize(
        ("label", "build"),
        [
            ("the T1 halted", _healthy_booster),
            ("the rover halted", _healthy_rover),
            ("the aircraft is descending", _healthy_aircraft),
        ],
    )
    def test_a_halt_that_landed_logs_no_error(
        self, label: str, build: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        driver = build(monkeypatch)
        with caplog.at_level(logging.DEBUG):
            asyncio.run(driver.stop())
        assert _errors(caplog) == "", label

    def test_the_healthy_halt_still_reaches_the_wire(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Reading a verdict must not have replaced sending the command.
        sdk = _FakeSdk()
        monkeypatch.setitem(sys.modules, "booster_robotics_sdk_python", sdk)
        asyncio.run(_live_booster(sdk).stop())
        assert ("MoveCommand", (0.0, 0.0, 0.0)) in sdk.client.calls

        session = _install_requests(monkeypatch)
        asyncio.run(_live_rover(session).stop())
        assert session.posts, "the zero twist must still reach /control"

    def test_a_disconnected_aircraft_still_only_halts_the_repeater(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The branch with nothing to descend is unchanged, and silent: there is
        # no halt to report on an aircraft that is not flying.
        from strands_robots.drivers import crazyflie as module

        driver = module.CrazyflieDriver()
        monkeypatch.setattr(driver, "land", lambda **_: pytest.fail("a disconnected aircraft must not be landed"))
        with caplog.at_level(logging.DEBUG):
            asyncio.run(driver.stop())
        assert _errors(caplog) == ""


# --------------------------------------------------------------------------- #
# The shared detail reader.                                                   #
# --------------------------------------------------------------------------- #


class TestTheDetailReader:
    """One reader, because the shipped halt verbs answer in two shapes."""

    @pytest.mark.parametrize(
        ("label", "envelope", "expected"),
        [
            ("success is not a failure", {"status": "success", "content": [{"text": "halted"}]}, None),
            (
                "a refusal text is quoted",
                {"status": "error", "content": [{"text": "move: the T1 refused the twist: code = 100"}]},
                "move: the T1 refused the twist: code = 100",
            ),
            (
                "a per-half outcome names the half",
                {"status": "error", "content": [{"json": {"locomotion_halted": False, "upper_body_released": True}}]},
                "locomotion_halted=False, upper_body_released=True",
            ),
            ("an unparsable failure is still a failure", {"status": "error", "content": []}, "no detail reported"),
            ("a missing status is not success", {"content": [{"text": "who knows"}]}, "who knows"),
        ],
    )
    def test_it_renders_the_shape_it_was_given(
        self, label: str, envelope: dict[str, Any], expected: str | None
    ) -> None:
        assert halt_failure_detail(envelope) == expected, label

    def test_it_never_reads_a_failure_as_a_landed_halt(self) -> None:
        # ``None`` means "the halt landed", so no non-success envelope may yield
        # it - including the malformed ones a caller cannot anticipate.
        malformed: list[Any] = [[], [{}], [{"json": {}}], ["not a block"], None]
        for content in malformed:
            assert halt_failure_detail({"status": "error", "content": content}) is not None
