"""A robot this package drives natively is named as such, not answered with lerobot's list.

Two registries answer "what builds this robot": lerobot's ``RobotConfig``
ChoiceRegistry, and this package's native-driver registry. A robot can be in the
second and not the first - that gap is what a native driver exists to close, and
:mod:`strands_robots.drivers.reachy` records it in as many words: "the Reachy
Mini has no lerobot robot type, so before this driver ``mode="real"`` raised
``ValueError: Unsupported robot type: 'reachy_mini'``".

The Reachy Mini also declares ``hardware.driver="strands"`` on its registry
entry, so :func:`~strands_robots.drivers.resolve_driver` sends it to its driver
and it never meets that refusal. Four robots the Dynamixel driver serves declare
nothing, so the default routes them to lerobot - which has no robot type for any
of them. They reached the generic listing of lerobot's sixteen robot types, and
that listing never mentioned that this package ships the driver that builds
them: an answer to the wrong question, and a dead end for a caller who has no
reason to guess at ``driver="strands"``.

The site already had this shape for the other wrong entry point. A leader arm is
a lerobot *teleoperator*, and
:func:`strands_robots.teleoperator._other_lerobot_kind_refusal` names it as one
rather than listing follower types - its docstring is where "answering it with
the names of the kind it is not answers the wrong question" is written down.
:func:`strands_robots.drivers.registry._native_driver_refusal` is that function's
sibling for a natively driven robot, consulted at the same site and returning
``None`` the same way, so a name with no native driver keeps the listing that is
the right answer for it. It is consulted first, because the two populations
overlap: the G1 is a lerobot teleoperator type as well as a natively driven
robot, and for a name in that overlap the driver is what a ``Robot()`` caller
asked for.

What is deliberately unchanged: which driver *wins*. Resolution precedence is
untouched, no registry entry gains a declaration, and a robot lerobot can build
still goes to lerobot. Whether ``koch`` and ``aloha`` - which have both a
working lerobot type and a native driver - should prefer the native one is a
preference, and ``unitree_g1`` shows the registry is where such a preference is
declared. This changes only what a caller is told when the driver they were
routed to cannot build the robot at all.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest

import strands_robots.drivers.registry as drivers_registry_mod
from strands_robots import Robot
from strands_robots.drivers import get_native_driver_class, resolve_driver
from strands_robots.drivers.registry import _native_driver_refusal
from strands_robots.registry import get_robot, list_robots

#: The robots that reach lerobot with a native driver already registered for
#: them. Literal rather than derived: a rule narrowed by mistake would
#: *deselect* a derived case and still report success, where a literal keeps
#: running and fails. :class:`TestTheDerivedPopulationIsExactlyThese` grades the
#: rule itself, so another robot arriving in this position is caught there -
#: which is how the Franka arms and the UR arms arrived here, each having moved
#: out of :data:`NO_DRIVER_OF_EITHER_KIND` when its own driver landed.
NATIVELY_DRIVEN_WITHOUT_A_LEROBOT_TYPE = (
    "vx300s",
    "wx250s",
    "trossen_wxai",
    "dynamixel_2r",
    "open_duck_mini",
    "panda",
    "fr3",
    "fr3_v2",
    "ur5e",
    "ur10e",
)

#: Robots that reach the same site with no native driver, so the listing of
#: lerobot's robot types is the right answer and must survive. Two hands and an
#: arm: every one is a real registry entry with a simulation asset and no
#: real-mode support of any kind. ``panda`` was a member until
#: :class:`~strands_robots.drivers.franka.driver.FrankaDriver` gave the Franka
#: family a real-mode path, and ``ur5e`` until :class:`~strands_robots.drivers.ur.URDriver`
#: did the same for the UR arms, which is exactly the transition these two tuples
#: exist to keep honest. ``xarm7`` takes the slot ``ur5e`` vacated.
NO_DRIVER_OF_EITHER_KIND = ("shadow_hand", "allegro_hand", "xarm7")

#: The generic listing's own words, which must be absent from a refusal that has
#: a better answer and present from one that does not.
_LEROBOT_LISTING = "Known lerobot robot types"


def _lerobot_robot_types() -> set[str]:
    """Every robot type lerobot's ChoiceRegistry knows, after its own discovery."""
    pytest.importorskip("lerobot")
    from strands_robots.hardware_robot import _ensure_lerobot_robots_registered

    _ensure_lerobot_robots_registered()
    from lerobot.robots.config import RobotConfig

    return set(RobotConfig.get_known_choices())


def _type_handed_to_lerobot(name: str) -> str:
    """The device type string ``hardware_robot`` receives for ``name``.

    A robot's declared ``hardware.lerobot_type`` when it has one, and its
    canonical name otherwise - which is why a robot with no lerobot type is
    looked up in the native registry under the name the caller passed.
    """
    hardware = (get_robot(name) or {}).get("hardware") or {}
    return str(hardware.get("lerobot_type") or name)


def _refusal_for(name: str) -> str:
    """Build ``name`` in real mode with no ``driver=`` and return the refusal."""
    with pytest.raises(ValueError) as excinfo:
        Robot(name, mode="real")
    return str(excinfo.value)


class TestThePremise:
    """The facts the regression rests on, so a passing cell cannot be vacuous."""

    @pytest.mark.parametrize("name", NATIVELY_DRIVEN_WITHOUT_A_LEROBOT_TYPE)
    def test_lerobot_has_no_robot_type_for_it(self, name: str) -> None:
        """lerobot structurally cannot build these, so there is no choice to make."""
        assert _type_handed_to_lerobot(name) not in _lerobot_robot_types()

    @pytest.mark.parametrize("name", NATIVELY_DRIVEN_WITHOUT_A_LEROBOT_TYPE)
    def test_a_native_driver_is_registered_for_it(self, name: str) -> None:
        """The better answer exists in this package, which is the whole point."""
        assert get_native_driver_class(name) is not None

    @pytest.mark.parametrize("name", NATIVELY_DRIVEN_WITHOUT_A_LEROBOT_TYPE)
    def test_the_default_still_routes_it_to_lerobot(self, name: str) -> None:
        """Why they meet this refusal at all: they declare no driver preference."""
        assert resolve_driver(name, None) == "lerobot"

    @pytest.mark.parametrize("name", NO_DRIVER_OF_EITHER_KIND)
    def test_the_control_robots_have_no_native_driver(self, name: str) -> None:
        """Otherwise the over-reach controls would be graded on the wrong robots."""
        assert get_native_driver_class(name) is None


class TestANativelyDrivenRobotIsNamed:
    """The regression: the refusal names the driver that builds it."""

    @pytest.mark.parametrize("name", NATIVELY_DRIVEN_WITHOUT_A_LEROBOT_TYPE)
    def test_the_refusal_names_the_driver_class(self, name: str) -> None:
        driver_cls = get_native_driver_class(name)
        assert driver_cls is not None, "premise: this robot has a native driver"
        assert driver_cls.__name__ in _refusal_for(name)

    @pytest.mark.parametrize("name", NATIVELY_DRIVEN_WITHOUT_A_LEROBOT_TYPE)
    def test_the_refusal_spells_out_the_call_that_builds_it(self, name: str) -> None:
        """A caller cannot guess ``driver='strands'``, so the retry is written out.

        The whole call, not the keyword alone: the message names that keyword a
        second time as the registry declaration a maintainer could add, and a
        caller cannot copy that one.
        """
        retry = f"Robot({name!r}, mode='real', driver='strands'"
        assert retry in _refusal_for(name)

    @pytest.mark.parametrize("name", NATIVELY_DRIVEN_WITHOUT_A_LEROBOT_TYPE)
    def test_the_refusal_names_the_robot_that_was_asked_for(self, name: str) -> None:
        assert repr(name) in _refusal_for(name)

    @pytest.mark.parametrize("name", NATIVELY_DRIVEN_WITHOUT_A_LEROBOT_TYPE)
    def test_the_refusal_does_not_answer_with_lerobots_vocabulary(self, name: str) -> None:
        """Sixteen robot types, none of them this one, is the wrong question answered."""
        assert _LEROBOT_LISTING not in _refusal_for(name)


class TestTheDerivedPopulationIsExactlyThese:
    """The rule, graded rather than the names: a robot arriving here is caught."""

    @staticmethod
    def _routed_to_lerobot_with_a_native_driver() -> set[str]:
        known = _lerobot_robot_types()
        return {
            entry["name"]
            for entry in list_robots("all")
            if resolve_driver(entry["name"], None) == "lerobot"
            and _type_handed_to_lerobot(entry["name"]) not in known
            and get_native_driver_class(entry["name"]) is not None
        }

    def test_every_such_robot_is_covered_by_the_regression(self) -> None:
        assert self._routed_to_lerobot_with_a_native_driver() == set(NATIVELY_DRIVEN_WITHOUT_A_LEROBOT_TYPE)

    def test_the_population_is_not_empty(self) -> None:
        """Non-vacuity: an empty set would satisfy every containment above."""
        assert self._routed_to_lerobot_with_a_native_driver()


class TestTheGenericListingSurvivesWhereItIsTheRightAnswer:
    """Over-reach: a robot with no native driver must still be told lerobot's names."""

    @pytest.mark.parametrize("name", NO_DRIVER_OF_EITHER_KIND)
    def test_lerobots_vocabulary_is_still_reported(self, name: str) -> None:
        assert _LEROBOT_LISTING in _refusal_for(name)

    @pytest.mark.parametrize("name", NO_DRIVER_OF_EITHER_KIND)
    def test_no_native_driver_is_claimed_for_it(self, name: str) -> None:
        """Naming a driver that does not exist would send a caller to a dead end."""
        assert "native driver" not in _refusal_for(name)


class TestTheTeleoperatorRefusalIsUnchanged:
    """The sibling check at the same site, and the overlap that decides the order."""

    def test_a_leader_arm_is_still_named_as_a_teleoperator(self) -> None:
        pytest.importorskip("lerobot.teleoperators.so_leader")
        refusal = _refusal_for("so101_leader")
        assert "teleoperator" in refusal
        assert _LEROBOT_LISTING not in refusal

    @staticmethod
    def _in_both_registries() -> set[str]:
        """Names lerobot lists as a teleoperator that also have a native driver."""
        pytest.importorskip("lerobot")
        from strands_robots.teleoperator import _ensure_lerobot_teleoperators_registered

        _ensure_lerobot_teleoperators_registered()
        from lerobot.teleoperators.config import TeleoperatorConfig

        return {name for name in TeleoperatorConfig.get_known_choices() if get_native_driver_class(name) is not None}

    def test_the_two_checks_overlap_so_their_order_is_a_decision(self) -> None:
        """The premise for checking the native driver first.

        Were the populations disjoint the order would be free. They are not: the
        G1 is a lerobot teleoperator type - the humanoid used as a motion source
        - and a robot this package drives natively. A second name arriving in
        this overlap needs the same question asked of it, which is why this reads
        the registries rather than asserting a count.
        """
        assert self._in_both_registries() == {"unitree_g1"}

    def test_the_overlapping_name_prefers_its_native_driver(self) -> None:
        """For a name in the overlap, the native answer is the right one."""
        from strands_robots.teleoperator import _other_lerobot_kind_refusal

        for name in self._in_both_registries():
            native = _native_driver_refusal(name)
            assert native is not None, f"{name} has a native driver, so it has a native answer"
            assert "driver='strands'" in native
            teleop = _other_lerobot_kind_refusal(name, wanted="robot")
            assert teleop is not None, f"{name} is in the teleoperator registry too"
            assert "Teleoperator(" in teleop, "the answer the order must not let win for a robot"


class TestTheHelperReportsRatherThanRaises:
    """A unit view, so the rule is graded on an install with no lerobot at all."""

    def test_a_robot_with_no_native_driver_gets_no_reason(self) -> None:
        """``None`` is how the caller keeps its own listing."""
        assert _native_driver_refusal("shadow_hand") is None

    def test_a_name_no_registry_knows_gets_no_reason(self) -> None:
        assert _native_driver_refusal("no-such-robot-anywhere") is None

    def test_a_robot_with_a_native_driver_gets_a_reason(self) -> None:
        reason = _native_driver_refusal("vx300s")
        assert reason is not None
        assert "DynamixelDriver" in reason

    def test_an_alias_finds_the_same_driver_as_its_canonical_name(self) -> None:
        """The lookup goes through ``resolve_name``, so an alias is not a miss."""
        canonical = _native_driver_refusal("unitree_g1")
        alias = _native_driver_refusal("g1")
        assert canonical is not None and alias is not None
        assert "G1Driver" in canonical and "G1Driver" in alias

    def test_the_reason_quotes_the_name_the_caller_passed(self) -> None:
        """An alias is echoed back as itself: the retry is the caller's own call."""
        alias = _native_driver_refusal("g1")
        assert alias is not None
        assert "Robot('g1'" in alias, alias

    def test_a_driver_registered_later_is_named_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Read from the registry, not from a list of today's four robots."""

        class _PlantedDriver:
            pass

        assert _native_driver_refusal("kuka_iiwa") is None, "premise: kuka_iiwa has no driver today"
        monkeypatch.setitem(drivers_registry_mod._NATIVE_DRIVERS, "kuka_iiwa", _PlantedDriver)
        reason = _native_driver_refusal("kuka_iiwa")
        assert reason is not None
        assert "_PlantedDriver" in reason


class TestTheOrderIsStated:
    """Structural: the better answer is consulted before the generic listing."""

    @staticmethod
    def _except_handler() -> ast.ExceptHandler:
        from strands_robots import hardware_robot

        source = inspect.getsource(hardware_robot)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ExceptHandler) and _LEROBOT_LISTING in (ast.get_source_segment(source, node) or ""):
                return node
        raise AssertionError(f"no except handler raises the {_LEROBOT_LISTING!r} listing")

    @staticmethod
    def _sole_line(handler: ast.ExceptHandler, node_type: type, marker: str, what: str) -> int:
        """The line of the one statement of ``node_type`` whose source holds ``marker``."""
        found = [
            # ``ast.stmt`` first: that is the node kind carrying ``lineno``, which
            # a dynamic ``node_type`` cannot tell a type checker on its own.
            node.lineno
            for node in ast.walk(handler)
            if isinstance(node, ast.stmt) and isinstance(node, node_type) and marker in ast.unparse(node)
        ]
        assert len(found) == 1, f"expected one {what}, found {len(found)}"
        return found[0]

    def test_the_native_check_precedes_the_generic_listing(self) -> None:
        handler = self._except_handler()
        guard = self._sole_line(handler, ast.If, "_native_driver_refusal", "native-driver guard")
        listing = self._sole_line(handler, ast.Raise, _LEROBOT_LISTING, "generic listing")
        assert guard < listing, "the listing would win and the better answer never run"

    def test_the_native_check_precedes_the_teleoperator_check(self) -> None:
        """Pinned here because no name in the overlap reaches this site today.

        ``unitree_g1`` is in both registries, so the order decides which answer
        it would get - and lerobot's robot registry also knows it, so it resolves
        and never arrives. Nothing observable turns on the order until that
        changes, which is exactly when a silent reordering would start handing a
        natively driven robot an instruction to build a teleoperator.
        """
        handler = self._except_handler()
        native = self._sole_line(handler, ast.If, "_native_driver_refusal", "native-driver guard")
        teleop = self._sole_line(handler, ast.If, "_other_lerobot_kind_refusal", "teleoperator guard")
        assert native < teleop, "a robot in both registries would be named as a teleoperator"


class TestNothingElseChanged:
    """Over-reach: which driver wins, and every other refusal, are untouched."""

    @pytest.mark.parametrize("name", NATIVELY_DRIVEN_WITHOUT_A_LEROBOT_TYPE)
    def test_an_explicit_strands_choice_still_builds_the_driver(self, name: str) -> None:
        robot: Any = Robot(name, mode="real", driver="strands", port="/dev/ttyUSB0")
        assert type(robot) is get_native_driver_class(name)

    @pytest.mark.parametrize("name", ["koch", "aloha"])
    def test_a_robot_lerobot_can_resolve_is_not_diverted(self, name: str) -> None:
        """These have both drivers, so the preference is the registry's to declare."""
        assert _type_handed_to_lerobot(name) in _lerobot_robot_types()
        refusal = _refusal_for(name)
        assert "driver='strands'" not in refusal
        assert _LEROBOT_LISTING not in refusal, "it resolved, so it fails on config fields"
