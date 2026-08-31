"""Every documented attribute read on a ``Robot()`` result must exist on what it returns.

``Robot()`` is polymorphic. ``mode="sim"`` builds a
:class:`~strands_robots.simulation.Simulation`; ``mode="real"`` builds a
:class:`strands_robots.hardware_robot.Robot` wrapper *unless* the call resolves
to a native driver, in which case the factory returns that driver itself, whose
surface is :data:`~strands_robots.drivers.DRIVER_SURFACE`. So
``robot.attach_teleop(...)`` is a correct line for a lerobot-backed robot and an
:class:`AttributeError` for a natively-driven one, and the two are spelled
identically at the call site.

Which driver a documented call resolves to is asked of
:func:`~strands_robots.drivers.resolve_driver` rather than re-derived here, so
this grader tracks the factory's own precedence: an explicit ``driver="strands"``
in the call, then the robot's registry ``hardware.driver``, then the default.
Reading only the registry would grade a documented ``driver="strands"`` call
against the lerobot wrapper - and that keyword is the *documented* way to reach
the driver of a robot whose registry entry declares nothing, which is every
robot a driver package registers late.

Neither existing docs grader can see that class of error:

* ``tests/test_docs_real_mode_invocations.py`` grades the robot *name* and the
  *keywords* inside the ``Robot(...)`` call.
* ``tests/test_docs_python_examples_are_callable.py`` grades keyword sets
  against signatures, and its ``_accepted_keywords`` returns ``None`` ("any
  keyword binds") for a callee carrying ``**kwargs`` - which ``Robot`` does.

Attribute access on the factory's *return value* is outside both, so a
documented read of a name the returned object does not carry renders verbatim in
the docs and raises on the first line a reader copies. This module closes that
gap: it resolves each documented read to the type the factory would return for
that ``(name, mode)`` pair and requires the name to exist there.

The surface a read is graded against is the union of three sources, because an
attribute can arrive from any of them:

* the class and its MRO, via :func:`dir`;
* ``self.X = ...`` assigned anywhere in the MRO - instance state a class sets up
  in ``__init__`` or a lazy initialiser, which :func:`dir` on the class cannot
  see;
* ``instance.X = ...`` bound by the factory in :mod:`strands_robots.robot`.
  ``run``, ``mesh`` and ``peer_id`` are bound there and appear on no class, so
  omitting this third source reports three offenders that are not defects.

Because the documented corpus is expected to be clean, the rule is also graded
over constructed exemplars rather than relying on the corpus to exercise its
own failing branch.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path

import pytest

import strands_robots
import strands_robots.hardware_robot as hardware_robot
import strands_robots.robot as robot_factory
from strands_robots.drivers import get_native_driver_class, resolve_driver
from strands_robots.registry import get_robot, resolve_name

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)

#: Modes whose returned type is a simulation rather than a hardware surface.
_SIM_MODES = frozenset({None, "sim"})


def _docs_sources() -> list[Path]:
    """Every documentation file whose ``python`` fences are graded."""
    files = sorted(_REPO_ROOT.glob("docs/**/*.md")) + [_REPO_ROOT / "README.md"]
    return [path for path in files if path.exists()]


def _runnable(block: str) -> str:
    """Return the parseable source of a fence, dropping doctest output lines."""
    if ">>> " in block:
        return "\n".join(line[4:] for line in block.splitlines() if line.startswith((">>> ", "... ")))
    return block


def _instance_surface(cls: type) -> set[str]:
    """Names reachable on an instance of ``cls``.

    ``dir`` misses instance state, so every ``self.X = ...`` target anywhere in
    the MRO is credited as well.
    """
    names = set(dir(cls))
    for klass in cls.__mro__:
        try:
            source = textwrap.dedent(inspect.getsource(klass))
        except (OSError, TypeError):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Store)
            ):
                names.add(node.attr)
    return names


def _factory_bound_names() -> set[str]:
    """Attributes the ``Robot()`` factory binds onto the instance it returns.

    ``run`` is bound here rather than defined on any class, so a surface derived
    from the classes alone would report every documented ``.run()`` as missing.
    """
    tree = ast.parse(inspect.getsource(robot_factory))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            names.add(node.args[1].value)
    return names


def _declares_a_native_driver(name: str) -> bool:
    """Whether the registry entry for ``name`` asks for its native driver."""
    record = get_robot(resolve_name(name)) or {}
    return ((record.get("hardware") or {}).get("driver")) == "strands"


def _native_driver_class(name: str) -> type:
    """The driver class the registry names for ``name``.

    ``get_native_driver_class`` answers ``None`` for a robot with no native
    driver, which :func:`_builds_a_native_driver` has already ruled out.
    """
    driver = get_native_driver_class(resolve_name(name))
    assert driver is not None, f"{name!r} declares hardware.driver='strands' but registers no driver class"
    return driver


def _builds_a_native_driver(name: str, driver: str | None) -> bool:
    """Whether ``Robot(name, mode="real", driver=driver)`` returns a native driver.

    Asks :func:`~strands_robots.drivers.resolve_driver` rather than reading the
    registry directly, so the caller's explicit ``driver=`` is honoured with the
    precedence the factory itself applies.

    Args:
        name: The robot name as documented.
        driver: The ``driver=`` keyword in the documented call, or ``None``.

    Returns:
        ``True`` when the resolved driver is the native one *and* a class is
        registered for the robot. A resolved ``"strands"`` with no class is a
        registry defect the seam's own tests grade, and reporting it from here
        as a missing attribute would name the wrong thing.
    """
    if resolve_driver(resolve_name(name), driver) != "strands":
        return False
    return get_native_driver_class(resolve_name(name)) is not None


def _hardware_surfaces(name: str, mode: str | None, driver: str | None = None) -> dict[str, set[str]]:
    """Surfaces ``Robot(name, mode=mode, driver=driver)`` can return, for a non-sim mode."""
    factory = _factory_bound_names()
    if mode == "real" and _builds_a_native_driver(name, driver):
        native = _native_driver_class(name)
        return {native.__name__: _instance_surface(native) | factory}
    wrapper = {"HardwareRobot": _instance_surface(hardware_robot.Robot) | factory}
    if mode == "real":
        return wrapper
    # "auto" resolves at runtime from the environment, so either is acceptable.
    return wrapper | {"Simulation": _simulation_surface()}


def _simulation_surface() -> set[str]:
    """The surface of the simulation the factory builds, or an empty set."""
    from strands_robots.simulation import Simulation

    return _instance_surface(Simulation) | _factory_bound_names()


#: One documented read: ``(origin, robot_name, mode, driver, attribute)``. ``mode``
#: and ``driver`` are the keywords the documented ``Robot(...)`` call passed, or
#: ``None`` where it passed neither - both are needed to resolve the read, because
#: the factory's precedence reads an explicit ``driver`` before the registry. Named
#: once because the shape drifted when ``driver`` was added: six annotations still
#: described four fields while the scan itself produced five.
_Read = tuple[str, str, str | None, str | None, str]


def _documented_reads(source: str, origin: str) -> list[_Read]:
    """Return a :data:`_Read` for each attribute read on a ``Robot()`` in ``source``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    def _call_target(call: ast.Call) -> tuple[str | None, str | None, str | None]:
        keywords = {kw.arg: getattr(kw.value, "value", None) for kw in call.keywords}
        name = call.args[0].value if call.args and isinstance(call.args[0], ast.Constant) else keywords.get("name")
        return (name if isinstance(name, str) else None), keywords.get("mode"), keywords.get("driver")

    bound: dict[str, tuple[str, str | None, str | None]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "Robot"
        ):
            name, mode, driver = _call_target(node.value)
            if name is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound[target.id] = (name, mode, driver)

    reads: list[_Read] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        if isinstance(base, ast.Name) and base.id in bound:
            name, mode, driver = bound[base.id]
        elif isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and base.func.id == "Robot":
            resolved, mode, driver = _call_target(base)
            if resolved is None:
                continue
            name = resolved
        else:
            continue
        reads.append((origin, name, mode, driver, node.attr))
    return reads


def _all_documented_reads() -> list[_Read]:
    """Every distinct attribute read on a documented ``Robot()`` result."""
    reads: list[_Read] = []
    for path in _docs_sources():
        origin = str(path.relative_to(_REPO_ROOT))
        for match in _PYTHON_FENCE.finditer(path.read_text(encoding="utf-8")):
            reads.extend(_documented_reads(_runnable(match.group(1)), origin))
    seen: set[_Read] = set()
    distinct: list[_Read] = []
    for read in reads:
        if read not in seen:
            seen.add(read)
            distinct.append(read)
    return distinct


def _unresolved(reads: list[_Read]) -> list[str]:
    """Return a report line for each read the returned type cannot answer."""
    offenders: list[str] = []
    for origin, name, mode, driver, attribute in reads:
        try:
            resolve_name(name)
        except ValueError:
            # An unregistered name is graded by test_docs_real_mode_invocations.
            continue
        surfaces = _simulation_surfaces() if mode in _SIM_MODES else _hardware_surfaces(name, mode, driver)
        if not any(attribute in surface for surface in surfaces.values()):
            shown = f"Robot({name!r}, mode={mode!r}" + (f", driver={driver!r})" if driver else ")")
            offenders.append(f"{origin}: {shown}.{attribute} is not on {sorted(surfaces)}")
    return offenders


def _simulation_surfaces() -> dict[str, set[str]]:
    """The sim-mode surface, keyed for reporting."""
    return {"Simulation": _simulation_surface()}


#: Pages that must contribute a read in each partition, set below today's
#: counts (5 hardware-mode, 20 sim-mode) so an ordinary docs edit does not trip
#: it while a narrowed file glob does.
_MINIMUM_HARDWARE_SOURCES = 4
_MINIMUM_SIMULATION_SOURCES = 10


def _assert_the_scan_reaches_the_docs_tree(reads: list[_Read], minimum: int) -> None:
    """Refuse a corpus small enough that the guard could be passing vacuously."""
    assert reads, "no documented attribute read was found - the scan stopped seeing the docs"
    sources = {read[0] for read in reads}
    under_docs = {source for source in sources if source.startswith("docs/")}
    assert len(sources) >= minimum, (
        f"only {sorted(sources)} contributed a read, fewer than the {minimum} expected; "
        "a narrowed file glob would leave the rest of the documentation ungraded"
    )
    assert under_docs, f"no page under docs/ contributed a read, only {sorted(sources)}"


class TestEveryDocumentedReadResolves:
    """A documented read must name something the returned object carries."""

    def test_no_hardware_mode_read_is_unresolvable(self) -> None:
        reads = [read for read in _all_documented_reads() if read[2] not in _SIM_MODES]
        _assert_the_scan_reaches_the_docs_tree(reads, _MINIMUM_HARDWARE_SOURCES)
        offenders = _unresolved(reads)
        assert offenders == [], "documented reads that raise AttributeError as written:\n  " + "\n  ".join(offenders)

    def test_no_simulation_read_is_unresolvable(self) -> None:
        pytest.importorskip("mujoco", reason="the simulation surface needs the [sim-mujoco] extra")
        reads = [read for read in _all_documented_reads() if read[2] in _SIM_MODES]
        _assert_the_scan_reaches_the_docs_tree(reads, _MINIMUM_SIMULATION_SOURCES)
        offenders = _unresolved(reads)
        assert offenders == [], "documented reads that raise AttributeError as written:\n  " + "\n  ".join(offenders)


class TestThePolymorphismThisGrades:
    """The premise: which surface a read is graded against depends on the robot."""

    def test_a_robot_declaring_a_native_driver_returns_the_driver_surface(self) -> None:
        natively_driven = [name for name in ("unitree_g1",) if _declares_a_native_driver(name)]
        assert natively_driven, "no registry entry declares hardware.driver='strands' - the premise is gone"
        surfaces = _hardware_surfaces(natively_driven[0], "real")
        assert "HardwareRobot" not in surfaces, "a natively-driven robot must not be graded as the lerobot wrapper"

    def test_an_explicit_driver_keyword_moves_the_surface(self) -> None:
        """The same robot and mode, graded differently because of ``driver=``.

        ``ur5e`` declares no ``hardware.driver``, so the registry alone would
        send both spellings to the lerobot wrapper - and the documented way to
        reach its native driver is the keyword.
        """
        assert not _declares_a_native_driver("ur5e"), "premise: ur5e declares no driver in the registry"
        assert "HardwareRobot" in _hardware_surfaces("ur5e", "real")
        assert "HardwareRobot" not in _hardware_surfaces("ur5e", "real", "strands")

    def test_an_explicit_lerobot_keyword_keeps_the_wrapper(self) -> None:
        """A caller overriding a registry declaration is honoured too."""
        assert _declares_a_native_driver("unitree_g1"), "premise: the G1 declares its native driver"
        assert "HardwareRobot" in _hardware_surfaces("unitree_g1", "real", "lerobot")

    def test_the_two_hardware_surfaces_disagree_about_a_real_method(self) -> None:
        """``attach_teleop`` exists on one surface and not the other, which is the whole risk."""
        wrapper = _instance_surface(hardware_robot.Robot)
        driver = _instance_surface(_native_driver_class("unitree_g1"))
        assert "attach_teleop" in wrapper
        assert "attach_teleop" not in driver

    def test_a_factory_bound_attribute_is_credited(self) -> None:
        """``run`` is bound onto the instance and is on no class in the MRO."""
        bound = _factory_bound_names()
        assert "run" in bound
        assert not hasattr(hardware_robot.Robot, "run")
        assert "run" in _hardware_surfaces("so100", "real")["HardwareRobot"]


class TestTheRuleIsGradedOnConstructedExemplars:
    """The documented corpus is expected to be clean, so the rule is graded directly."""

    _ACCEPTED = 'r = Robot("so100", mode="real")\nr.attach_teleop("so101_leader")\n'
    _MISSPELLED = 'r = Robot("so100", mode="real")\nr.attach_teleoperator("so101_leader")\n'
    _WRONG_SURFACE = 'r = Robot("unitree_g1", mode="real")\nr.attach_teleop("so101_leader")\n'
    _FACTORY_BOUND = 'r = Robot("so100", mode="real")\nr.run()\n'
    # control_frequency is assigned as self.X and is on no class, so this
    # exemplar is answered only by the MRO-assignment source.
    _INSTANCE_STATE = 'r = Robot("so100", mode="real")\nr.control_frequency\n'
    # The driver= keyword decides the surface for a robot whose registry entry
    # declares nothing. Both spellings name a real method of the object the
    # factory returns; each raises AttributeError on the other's object.
    _KEYWORD_DRIVER = 'r = Robot("ur5e", mode="real", driver="strands", port="10.0.0.2")\nr.connect_eagerly()\n'
    _KEYWORD_DRIVER_WRONG = (
        'r = Robot("ur5e", mode="real", driver="strands", port="10.0.0.2")\nr.attach_teleop("so101_leader")\n'
    )
    _KEYWORD_LEROBOT = 'r = Robot("so100", mode="real", driver="lerobot")\nr.attach_teleop("so101_leader")\n'

    @pytest.mark.parametrize(
        ("exemplar", "expected"),
        [
            (_ACCEPTED, False),
            (_FACTORY_BOUND, False),
            (_INSTANCE_STATE, False),
            (_KEYWORD_DRIVER, False),
            (_KEYWORD_LEROBOT, False),
            (_MISSPELLED, True),
            (_WRONG_SURFACE, True),
            (_KEYWORD_DRIVER_WRONG, True),
        ],
        ids=[
            "accepted",
            "factory-bound",
            "instance-state",
            "keyword-driver",
            "keyword-lerobot",
            "misspelled",
            "wrong-surface",
            "keyword-driver-wrong-surface",
        ],
    )
    def test_the_rule_separates_these(self, exemplar: str, expected: bool) -> None:
        offenders = _unresolved(_documented_reads(exemplar, "exemplar.md"))
        assert bool(offenders) is expected, f"exemplar graded {bool(offenders)}, expected {expected}: {offenders}"

    def test_the_exemplars_reach_both_verdicts(self) -> None:
        outcomes = {
            bool(_unresolved(_documented_reads(exemplar, "exemplar.md")))
            for exemplar in (
                self._ACCEPTED,
                self._FACTORY_BOUND,
                self._INSTANCE_STATE,
                self._KEYWORD_DRIVER,
                self._KEYWORD_LEROBOT,
                self._MISSPELLED,
                self._WRONG_SURFACE,
                self._KEYWORD_DRIVER_WRONG,
            )
        }
        assert outcomes == {True, False}, f"the exemplars only ever produce {outcomes}"
