"""``g1_get_state`` refuses a handle whose status envelope is not a G1's.

``g1_get_state`` takes a live driver handle typed :class:`~typing.Any` -- the
annotation carries no type into the generated tool schema, so a caller reaches
the verb with whatever it has. The verb's whole answer (``admits_arm`` /
``admits_loco``) is decided from the ``fsm_id`` the handle's status envelope
reports, and ``get_status`` is a member of the shared
:data:`~strands_robots.drivers.base.DRIVER_SURFACE` contract that *every*
native driver implements. So a duck-type check on ``get_status`` cannot tell a
G1 from any other driver in a mixed fleet.

The reads that used to be unguarded made that a silent wrong answer rather than
a refusal. A sibling driver's envelope carries the shared ``tool_name`` /
``connected`` / ``battery_pct`` triple and no FSM field, so ``inner.get("fsm_id")``
answered ``None`` -- and the verb documents ``None`` as *"a read that never
arrived"*, reporting both admit booleans ``False``. That is a decided,
confident-looking answer to "would the arm gate admit today", computed from a
robot with no gate, and it is byte-identical to the answer a genuinely
disconnected G1 earns. Nothing downstream can separate them.

These tests fix both halves of the handle contract:

* the four unusable shapes (``None``, an object with no ``get_status``, one
  whose ``get_status`` returns something other than the envelope, and one whose
  envelope is another driver's) are each refused with an error envelope naming
  the parameter, the type and the remedy, rather than raising past the ``@tool``
  boundary or answering;
* the disconnected-G1 answer the verb documents is unchanged -- a G1 envelope
  reporting ``fsm_id=None`` still earns ``success`` and two ``False`` booleans.

The discriminator is read off the shipped drivers rather than restated here, so
a driver that later starts reporting an FSM field moves this file with it.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from typing import Any

import pytest

from strands_robots.drivers import DRIVER_SURFACE, get_native_driver_class, list_native_drivers
from strands_robots.tools.g1 import g1_state
from strands_robots.tools.g1._g1_common import HANDSHAKE_FSMS
from strands_robots.tools.g1.g1_state import g1_get_state


def _call(handle: Any) -> dict[str, Any]:
    """Await the verb against ``handle`` and return its envelope."""
    return asyncio.run(g1_get_state(driver=handle))


def _text(envelope: dict[str, Any]) -> str:
    """Join every text block of ``envelope`` so an assertion reads the message."""
    return " ".join(block.get("text", "") for block in envelope.get("content", []))


def _envelope(inner: dict[str, Any]) -> dict[str, Any]:
    """Wrap ``inner`` in the driver status envelope shape."""
    return {"status": "success", "content": [{"json": inner}]}


class _Handle:
    """A driver double whose ``get_status`` returns a caller-chosen value."""

    def __init__(self, answer: Any) -> None:
        self._answer = answer

    async def get_status(self) -> Any:
        return self._answer


#: The FSM fields a G1 status envelope carries, stated here rather than read
#: off the module under test.
#:
#: The premise cells below assert facts about the *shipped drivers* - that the
#: shared triple cannot discriminate and that these five can. Reading the
#: module's own constant for that would make those cells assertions about the
#: fix instead of about the drivers, and unrunnable on a tree without it. One
#: cell in the structural class asserts the two agree, which is where a drift
#: between this list and the verb's belongs.
_CONTRACT_KEYS = (
    "fsm_id",
    "mode_machine",
    "fsm_mode_name",
    "fsm_refusal",
    "motion_switcher_open_error",
)

#: A G1 status envelope's inner dict, disconnected: every field ``None``.
_G1_DISCONNECTED = {
    "tool_name": "g1",
    "connected": False,
    "connect_error": None,
    "port": None,
    "network_interface": "eth0",
    "fsm_id": None,
    "mode_machine": None,
    "battery_pct": None,
    "fsm_mode_name": None,
    "fsm_refusal": None,
    "motion_switcher_open_error": None,
}

#: A sibling driver's inner dict: the shared triple, and no FSM field.
_SIBLING_INNER = {
    "tool_name": "microduck",
    "connected": False,
    "connect_error": None,
    "battery_pct": 91.0,
    "motion_stopped": False,
    "socket": "/run/robotd.sock",
}


def _driver_class(robot: str) -> Any:
    """Return the native driver class registered for ``robot``.

    The registry lookup is typed ``type | None``, so calling its result
    directly is a type error at every site. One accessor carries the assert.
    """
    cls = get_native_driver_class(robot)
    assert cls is not None, f"no native driver is registered for {robot!r}"
    return cls


def _constructible_native_drivers() -> dict[str, Any]:
    """Return one arg-free instance per shipped native driver class.

    Two of the five driver classes take a required positional argument, so this
    reaches the ones a test can build with no hardware and no fixture. The
    callers assert a floor on the population rather than a count, so a driver
    that gains or loses an arg-free constructor does not silently empty the
    sweep.
    """
    instances: dict[str, Any] = {}
    for robot in sorted(list_native_drivers()):
        cls = get_native_driver_class(robot)
        if cls is None or cls.__name__ in instances:
            continue
        try:
            instances[cls.__name__] = cls()
        except TypeError:
            continue
    return instances


class TestPremisesTheDiscriminatorRestsOn:
    """The facts that make a G1 envelope distinguishable, read off the drivers."""

    def test_get_status_is_a_shared_driver_surface_member(self) -> None:
        """So a duck-type check on it cannot identify one driver among many."""
        assert "get_status" in DRIVER_SURFACE

    def test_every_shipped_driver_reports_the_shared_triple(self) -> None:
        """The triple is the contract, so it cannot discriminate."""
        instances = _constructible_native_drivers()
        assert len(instances) >= 3, f"expected at least three arg-free drivers, got {sorted(instances)}"
        for name, driver in sorted(instances.items()):
            inner = asyncio.run(driver.get_status())["content"][0]["json"]
            for shared in ("tool_name", "connected", "battery_pct"):
                assert shared in inner, f"{name} does not report {shared!r}"

    def test_no_non_g1_driver_reports_any_contract_key(self) -> None:
        """Which is what makes the FSM fields a sound discriminator."""
        instances = _constructible_native_drivers()
        siblings = {name: d for name, d in instances.items() if name != "G1Driver"}
        assert siblings, f"expected a non-G1 driver to compare against, got {sorted(instances)}"
        for name, driver in sorted(siblings.items()):
            inner = asyncio.run(driver.get_status())["content"][0]["json"]
            declared = [key for key in _CONTRACT_KEYS if key in inner]
            assert declared == [], f"{name} reports {declared}, so it is no longer distinguishable"

    def test_the_g1_driver_reports_every_contract_key(self) -> None:
        """The accepted path must not be refused by its own discriminator."""
        driver = _driver_class("unitree_g1")()
        inner = asyncio.run(driver.get_status())["content"][0]["json"]
        absent = [key for key in _CONTRACT_KEYS if key not in inner]
        assert absent == [], f"G1Driver.get_status does not report {absent}"


class TestASiblingDriverIsRefusedRatherThanAnswered:
    """The silent half: a real driver of another class must not earn an answer."""

    def test_a_real_sibling_driver_is_refused_by_name(self) -> None:
        """Driven through the shipped classes, not a double."""
        instances = _constructible_native_drivers()
        siblings = sorted(name for name in instances if name != "G1Driver")
        assert siblings, f"expected a non-G1 driver, got {sorted(instances)}"
        for name in siblings:
            envelope = _call(instances[name])
            assert envelope["status"] == "error", f"{name} earned {envelope['status']!r}"
            assert name in _text(envelope), f"the refusal does not name {name}"

    def test_the_refusal_names_the_parameter_and_the_absent_fields(self) -> None:
        envelope = _call(_Handle(_envelope(dict(_SIBLING_INNER))))
        message = _text(envelope)
        assert envelope["status"] == "error"
        assert "`driver`" in message
        assert "fsm_id" in message

    def test_no_gate_answer_is_reported_for_a_sibling_envelope(self) -> None:
        """The verb must not carry ``admits_arm`` on a refusal."""
        envelope = _call(_Handle(_envelope(dict(_SIBLING_INNER))))
        assert "admits_arm" not in envelope
        assert "admits_loco" not in envelope

    def test_a_g1_envelope_missing_one_contract_key_is_still_refused(self) -> None:
        """A partial envelope is not a G1 envelope."""
        inner = dict(_G1_DISCONNECTED)
        del inner["fsm_id"]
        envelope = _call(_Handle(_envelope(inner)))
        assert envelope["status"] == "error"
        assert "fsm_id" in _text(envelope)


class TestAnUnusableHandleIsRefusedNotRaised:
    """The loud half: every shape used to raise past the tool boundary."""

    def test_a_missing_handle_is_refused(self) -> None:
        envelope = _call(None)
        assert envelope["status"] == "error"
        assert "`driver` is required" in _text(envelope)

    def test_a_handle_without_get_status_names_its_type(self) -> None:
        envelope = _call("unitree_g1")
        message = _text(envelope)
        assert envelope["status"] == "error"
        assert "'str'" in message
        assert "get_status" in message

    @pytest.mark.parametrize(
        "answer",
        [None, "a string", [], {"status": "success"}, {"status": "success", "content": []}],
        ids=["none", "string", "list", "no-content", "empty-content"],
    )
    def test_a_get_status_that_returns_no_envelope_is_refused(self, answer: Any) -> None:
        envelope = _call(_Handle(answer))
        assert envelope["status"] == "error"
        assert "envelope" in _text(envelope)

    def test_a_refused_handle_never_reaches_the_driver_call(self) -> None:
        """A shape refusal precedes the dial; a contents refusal follows it.

        The zero below is only a measurement beside the one under it. ``None``
        and a string carry no ``get_status`` to dial, so a counter watching
        them cannot leave zero however the verb is ordered. A sibling handle is
        the input that can move it -- the verb cannot know an envelope is
        another driver's until it has read it -- so the pair grades the refusal
        *order*, and pins the refused path to a single dial. Nothing else in
        this file watches the error path for a repeated call.
        """
        calls = 0

        class _CountingSibling:
            """A sibling-driver handle that records every dial."""

            async def get_status(self) -> dict[str, Any]:
                nonlocal calls
                calls += 1
                return _envelope(dict(_SIBLING_INNER))

        assert _call(None)["status"] == "error"
        assert _call("a name")["status"] == "error"
        assert calls == 0, "a handle with no get_status must be refused before the call"

        assert _call(_CountingSibling())["status"] == "error"
        assert calls == 1, "a sibling handle is refused on its envelope, so it is dialled exactly once"

    def test_a_content_block_without_a_json_body_is_refused(self) -> None:
        envelope = _call(_Handle({"status": "success", "content": [{"text": "not json"}]}))
        assert envelope["status"] == "error"
        assert "envelope" in _text(envelope)


class TestTheDocumentedG1AnswersAreUnchanged:
    """Over-reach controls: the accepted path must behave exactly as before."""

    def test_a_disconnected_g1_still_reports_the_documented_none_answer(self) -> None:
        """The crux: ``fsm_id=None`` on a *G1* envelope is a read that never arrived."""
        envelope = _call(_Handle(_envelope(dict(_G1_DISCONNECTED))))
        assert envelope["status"] == "success"
        assert envelope["fsm_id"] is None
        assert envelope["admits_arm"] is False
        assert envelope["admits_loco"] is False

    def test_a_wired_g1_still_round_trips_every_field(self) -> None:
        inner = dict(_G1_DISCONNECTED)
        admitting = sorted(HANDSHAKE_FSMS)[0]
        inner.update(connected=True, fsm_id=admitting, mode_machine=9, battery_pct=92.0, fsm_mode_name="ai")
        envelope = _call(_Handle(_envelope(inner)))
        assert envelope["status"] == "success"
        assert envelope["fsm_id"] == admitting
        assert envelope["mode_machine"] == 9
        assert envelope["battery_pct"] == 92.0
        assert envelope["fsm_mode_name"] == "ai"
        assert envelope["admits_arm"] is True

    def test_the_verb_still_awaits_the_driver_call_exactly_once(self) -> None:
        calls = 0

        class _Counting:
            async def get_status(self) -> dict[str, Any]:
                nonlocal calls
                calls += 1
                return _envelope(dict(_G1_DISCONNECTED))

        assert _call(_Counting())["status"] == "success"
        assert calls == 1


class TestTheContractKeysAreTheOnesTheVerbDecidesFrom:
    """Structural: the discriminator is derived from the verb's own reads."""

    def test_every_contract_key_is_a_field_the_verb_reads(self) -> None:
        source = inspect.getsource(g1_state)
        read = {
            node.args[0].value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "inner"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        assert read, "found no inner.get(...) reads; the scan has gone blind"
        missing = [key for key in g1_state._G1_STATUS_CONTRACT_KEYS if key not in read | {"fsm_id"}]
        assert missing == [], f"the discriminator names fields the verb does not read: {missing}"

    def test_fsm_id_is_among_the_contract_keys(self) -> None:
        """It is the field both gate answers are computed from."""
        assert "fsm_id" in g1_state._G1_STATUS_CONTRACT_KEYS

    def test_the_module_constant_and_this_files_list_agree(self) -> None:
        """The one place the two accounts of the discriminator are compared."""
        assert tuple(g1_state._G1_STATUS_CONTRACT_KEYS) == _CONTRACT_KEYS

    def test_the_check_iterates_the_named_constant_not_a_literal(self) -> None:
        """A restated literal keeps the values and loses the single owner.

        Comparing the two lists above cannot see that: an inline tuple with
        today's five strings satisfies it while leaving two places to edit. The
        constant is what a reader widening the discriminator finds, so the check
        has to read it by name.
        """
        source = inspect.getsource(g1_state)
        iterated = {
            ast.unparse(generator.iter)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ListComp | ast.SetComp | ast.GeneratorExp)
            for generator in node.generators
            if "not in inner" in ast.unparse(node)
        }
        assert iterated, "found no comprehension testing membership of `inner`; the scan has gone blind"
        assert iterated == {"_G1_STATUS_CONTRACT_KEYS"}, (
            f"the absent-field check iterates {sorted(iterated)} rather than the named constant"
        )
