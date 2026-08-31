"""Every g1 verb taking a live driver handle refuses a wrong one by name.

The sensor verbs in ``strands_robots.tools.g1`` read a cache the driver's own
DDS subscriber writes.  The handle is a live Python object, so it is annotated
:class:`~typing.Any` - the driver module reaches into this package for
``ensure_dds``, so a real annotation would close an import cycle, and ``Any``
carries no type into the generated tool schema either.  A model reading that
schema therefore has nothing telling it the argument cannot be synthesized, and
will reach the verb with a robot *name* or with nothing at all.

``strands_robots.tools.run_policy`` is the one other ``@tool`` in the tree whose
parameter is a live handle typed :class:`~typing.Any`, and it refuses both
shapes with a message naming the parameter, the type it received and the remedy
- its own comment says an agent cannot synthesize the argument and that this is
the point.  The sensor verbs called the accessor as their first statement, so
the same two shapes surfaced as ``AttributeError`` naming a private attribute.
``AGENTS.md`` puts it plainly: an ``AgentTool`` handler returns an error dict and
never raises past the structured response.

## What each class grades

``TestEveryLiveHandleVerbRefusesAWrongHandle`` is **derived**: it discovers the
population by signature and by definition site rather than naming it, so a verb
added to this package with a live-handle parameter is held to the rule the hour
it lands instead of inheriting an exemption by being absent from a list.  The
discovery is deliberately *not* keyed on the module name - ``g1_state.py``
defines ``g1_get_state`` and ``g1_task_status.py`` defines
``g1_get_task_status``, and a scan reading ``getattr(module, module_name)``
grades neither while appearing to grade the package.  Both are async-or-not and
both are called through ``_call``, because a coroutine returned unawaited is a
verb inside the population that is still ungraded.

``TestTheDiscoveryIsCrossChecked`` grades the discovery itself, by deriving the
population a second time from the package *source* and comparing.  Every rule
here iterates one scan, so a scan that narrows does not fail - it grades less
and still reports green.  The non-vacuity check below cannot catch that, because
it fires only on an *empty* population, and a partial narrowing is both likelier
and quieter than a total one.

``TestTheRefusalNamesWhatACallerNeeds`` grades the message, because a refusal
that does not name the parameter leaves a caller no better off than the
traceback did.

``TestAHealthyHandleIsUntouched`` is the over-reach control for the *shared*
guard: it must not disturb either answer a working handle produces - the empty
cache reports ``present=False`` and a written cache reports ``present=True``.
Its population is the AST-derived set of verbs that actually call
``snapshot_handle_refusal``, because the rest of the family answers a different
accessor and a different shape and carries its own healthy-handle control in its
own suite.  The refusal rules above stay universal.

``TestPremises`` records the two facts the fix rests on: the precedent really
does refuse both shapes, and the shared module really is free of the vendor SDK
at import, so calling it from a verb cannot break the import-hygiene pin each
verb suite already carries.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.g1 as g1_package
from strands_robots.tools.g1._g1_common import snapshot_handle_refusal


class _AccessorIsData:
    """A handle carrying ``_snapshot`` as a value rather than a method.

    A dataclass or a namespace built from a cache dump lands here, and the name
    being *present* is what makes it distinct from every other wrong handle: a
    guard testing only for presence would accept it and then fail on the call.
    """

    _snapshot = None


# A handle a caller might plausibly reach a verb with, none of which can answer
# the accessor.  ``None`` is the "the model left it out" shape; the rest are the
# "the model synthesized something" shape.
WRONG_HANDLES: tuple[tuple[str, Any], ...] = (
    ("omitted", None),
    ("a robot name", "unitree_g1"),
    ("an empty mapping", {}),
    ("an integer", 3),
    ("a list", []),
    ("an object whose accessor is data, not a method", _AccessorIsData()),
)


class _CacheOnlyDriver:
    """A handle that answers the accessor, standing in for a wired driver."""

    def __init__(self, cache: dict[str, Any] | None) -> None:
        self._cache = cache

    def _snapshot(self, attr: str) -> dict[str, Any] | None:
        return self._cache


def _package_modules() -> list[Any]:
    """Import and return every public module in the g1 tools package."""
    package_dir = Path(g1_package.__file__).parent
    modules = []
    for info in sorted(pkgutil.iter_modules([str(package_dir)]), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        modules.append(importlib.import_module(f"{g1_package.__name__}.{info.name}"))
    return modules


def _tools_defined_in(module: Any) -> dict[str, Any]:
    """Return every ``@tool`` *defined* in ``module``, keyed by its function name.

    Keyed by the function name and found by walking the module, because the
    function name and the module name are not the same thing: ``g1_state.py``
    defines ``g1_get_state`` and ``g1_task_status.py`` defines
    ``g1_get_task_status``.  A scan reading ``getattr(module, module_name)``
    sees neither, and a verb it cannot see inherits an exemption from every
    rule in this file.

    ``__wrapped__.__module__`` filters out a tool this module merely imported,
    so a re-export is not graded a second time under another module's name.
    """
    found: dict[str, Any] = {}
    for attribute in sorted(dir(module)):
        candidate = getattr(module, attribute, None)
        wrapped = getattr(candidate, "__wrapped__", None)
        if wrapped is None or not callable(wrapped):
            continue
        if getattr(wrapped, "__module__", None) != module.__name__:
            continue
        found[wrapped.__name__] = candidate
    return found


def _live_handle_verbs() -> dict[str, Any]:
    """Return every ``@tool`` in the g1 package whose first parameter is a handle.

    The population is derived from the signature - a first parameter annotated
    ``Any`` - and from the *definition site* rather than from a name, so a verb
    added later is graded without this file being edited whatever its module is
    called.  Annotations are strings here because the modules carry
    ``from __future__ import annotations``.
    """
    found: dict[str, Any] = {}
    for module in _package_modules():
        for name, tool in _tools_defined_in(module).items():
            parameters = list(inspect.signature(tool.__wrapped__).parameters.values())
            if not parameters:
                continue
            if parameters[0].annotation in ("Any", Any):
                found[name] = tool
    return found


# The spellings of ``Any`` a first parameter can carry.  :func:`_live_handle_verbs`
# compares against the bare string or the object, so a verb annotated
# ``typing.Any`` sits outside its population; the source scan accepts every
# spelling on purpose, so that gap is a failure here rather than an exemption
# there.
_ANY_SPELLINGS = frozenset({"Any", "typing.Any", "t.Any"})


def _is_tool_decorated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether a definition carries the ``@tool`` decorator, bare or called."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if getattr(target, "id", None) == "tool" or getattr(target, "attr", None) == "tool":
            return True
    return False


def _live_handle_verbs_from_source() -> set[str]:
    """The live-handle population read from the package *source*, not from imports.

    This shares no mechanism with :func:`_live_handle_verbs`: it reads text and
    never imports, so it cannot inherit a narrowing from the ``pkgutil``
    enumeration, from ``__wrapped__`` being absent on a verb, or from
    ``__wrapped__.__module__`` disagreeing with the module a verb is defined in.
    That independence is the point.  #2958 measured what a narrowing costs:
    keyed on the module name, the discovery found 7 of 9 verbs, and the two it
    dropped were ``g1_get_state`` and ``g1_get_task_status`` - the second of
    which had no handle guard at all and raised past the structured response on
    all six wrong handles, while every rule in this file read green.

    It is deliberately *wider* than the runtime scan in two ways, so that either
    kind of narrowing fails rather than passing quietly:

    * Private modules are read too.  A ``@tool`` in ``_g1_common.py`` is
      invisible to the runtime scan, which skips a leading underscore.  A verb
      belongs in a public module and a shared helper is not a verb, so the two
      routes disagreeing here is the right outcome and it names the file.
    * Every spelling of ``Any`` is accepted, not only the bare one.  A verb
      annotated ``typing.Any`` would sit outside the runtime population; here it
      is inside it, so the disagreement is loud instead of silent.

    Positional-only parameters are included because ``inspect.signature`` counts
    them, so both routes read the same first parameter.
    """
    found: set[str] = set()
    package_dir = Path(g1_package.__file__).parent
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_tool_decorated(node):
                continue
            positional = node.args.posonlyargs + node.args.args
            if not positional or positional[0].annotation is None:
                continue
            if ast.unparse(positional[0].annotation).strip("\"'") in _ANY_SPELLINGS:
                found.add(node.name)
    return found


def _snapshot_family_verbs() -> dict[str, Any]:
    """The live-handle verbs reading their handle through the shared snapshot guard.

    Derived by AST from the call rather than from a list, so a verb that stops
    calling :func:`snapshot_handle_refusal` leaves the family the moment it
    does and the over-reach control stops asserting a ``present`` answer it no
    longer owes.

    The rest of the population answers a different accessor and a different
    shape - ``g1_get_state`` reads an async ``get_status`` and reports gate
    membership rather than ``present`` - and each carries its own healthy-handle
    control in its own suite.  The *refusal* rules above are the universal part
    and run against everything; this narrower set is only for the control that
    grades what a working handle gets back.
    """
    live = _live_handle_verbs()
    family: dict[str, Any] = {}
    for module in _package_modules():
        source = Path(module.__file__).read_text(encoding="utf-8")
        calls_guard = any(
            isinstance(node, ast.Call) and getattr(node.func, "id", None) == "snapshot_handle_refusal"
            for node in ast.walk(ast.parse(source))
        )
        if not calls_guard:
            continue
        for name in _tools_defined_in(module):
            if name in live:
                family[name] = live[name]
    return family


def _call(tool: Any, *args: Any) -> Any:
    """Call a verb's undecorated function, awaiting it when it is a coroutine.

    ``g1_get_state`` is ``async`` and its siblings are not.  A sweep calling
    every verb synchronously takes a coroutine object back from the async one,
    so its guard never runs: the rules below would grade the coroutine rather
    than the refusal, and the verb would be inside the population while still
    being effectively ungraded.
    """
    undecorated = tool.__wrapped__
    if inspect.iscoroutinefunction(undecorated):
        return asyncio.run(undecorated(*args))
    return undecorated(*args)


def _refusal_text(envelope: Any) -> str:
    """Return the text of an error envelope, or ``""`` when it is not one."""
    if not isinstance(envelope, dict) or envelope.get("status") != "error":
        return ""
    blocks = envelope.get("content") or []
    return " ".join(str(block.get("text", "")) for block in blocks if isinstance(block, dict))


class TestEveryLiveHandleVerbRefusesAWrongHandle:
    """The derived rule: a handle that cannot answer is refused, not dereferenced."""

    def test_the_population_is_not_empty(self) -> None:
        """Non-vacuity: a scan that finds nothing would pass every rule below."""
        verbs = _live_handle_verbs()
        assert verbs, (
            "found no g1 verb whose first parameter is a live handle typed Any; "
            "the discovery has gone blind and every rule in this class is vacuous"
        )

    def test_the_scan_reaches_the_three_sensor_verbs_on_main(self) -> None:
        """The population contains the verbs this rule was written for."""
        verbs = set(_live_handle_verbs())
        expected = {"g1_imu", "g1_lidar_state", "g1_lidar_summary"}
        assert expected <= verbs, f"expected {sorted(expected)} among {sorted(verbs)}"

    def test_a_verb_named_unlike_its_module_is_still_graded(self) -> None:
        """The blind spot this scan used to have, pinned by the two verbs in it.

        ``getattr(module, info.name)`` reaches a verb only when its function
        name equals its module name, so ``g1_state.py``'s ``g1_get_state`` and
        ``g1_task_status.py``'s ``g1_get_task_status`` both sat outside the
        population while the docstring above claimed a verb "is held to the rule
        the hour it lands".  ``g1_get_task_status`` arrived with no handle guard
        at all - all six wrong handles raised ``AttributeError`` past the
        structured response - and every rule in this class passed, because the
        scan could not see it.
        """
        verbs = _live_handle_verbs()
        for verb, module_stem in (
            ("g1_get_state", "g1_state"),
            ("g1_get_task_status", "g1_task_status"),
        ):
            assert verb in verbs, f"{verb} (defined in {module_stem}.py) is not in the population: {sorted(verbs)}"

    def test_the_scan_is_not_keyed_on_the_module_name(self) -> None:
        """Non-vacuity for the rule above: some verb must disagree with its module.

        If every graded verb happened to be named after its module, this file
        could go back to ``getattr(module, info.name)`` and no test here would
        notice.  The assertion is that the population actually exercises the
        name-independent path.
        """
        disagreeing = [
            name
            for name, tool in _live_handle_verbs().items()
            if tool.__wrapped__.__module__.rsplit(".", 1)[-1] != name
        ]
        assert disagreeing, (
            "no graded verb's function name differs from its module name, so "
            "this scan cannot demonstrate it is not name-keyed"
        )

    @pytest.mark.parametrize("label,handle", WRONG_HANDLES, ids=[h[0].replace(" ", "-") for h in WRONG_HANDLES])
    def test_a_wrong_handle_is_an_error_envelope_not_an_exception(self, label: str, handle: Any) -> None:
        """No verb dereferences a handle it has not judged."""
        for name, tool in sorted(_live_handle_verbs().items()):
            result = _call(tool, handle)
            assert isinstance(result, dict), f"{name} returned {type(result).__name__} for {label}"
            assert result.get("status") == "error", f"{name} did not refuse {label}: {result!r}"


class TestTheDiscoveryIsCrossChecked:
    """The population every rule above grades is derived twice and compared.

    ``test_the_population_is_not_empty`` fires only when the scan finds
    *nothing*.  #2958's narrowing left five verbs in the population, so it
    passed, and so did every rule that iterates it.  Comparing against a second
    route derived by a different mechanism is what turns "the discovery
    narrowed" into a failure, and a count derived that way is preferred to a
    literal, which would be churn on every verb the package gains.
    """

    def test_the_source_scan_is_not_empty(self) -> None:
        """Non-vacuity: an empty second route agrees with any runtime population."""
        assert _live_handle_verbs_from_source(), (
            "the source scan found no live-handle verb, so it agrees with any "
            "runtime population and the cross-check below is vacuous"
        )

    def test_the_runtime_scan_and_the_source_scan_find_the_same_verbs(self) -> None:
        """A verb in the source and outside the population is one nothing grades."""
        runtime = set(_live_handle_verbs())
        source = _live_handle_verbs_from_source()
        assert source == runtime, (
            "the two discovery routes disagree, so the rules in this file do not "
            "grade the package they claim to. Defined in the source and ungraded: "
            f"{sorted(source - runtime)}. Graded but absent from the source scan: "
            f"{sorted(runtime - source)}"
        )


class TestTheRefusalNamesWhatACallerNeeds:
    """A refusal a caller cannot act on is no better than the traceback."""

    def test_the_refusal_names_the_verb_and_the_parameter(self) -> None:
        for name, tool in sorted(_live_handle_verbs().items()):
            text = _refusal_text(_call(tool, None))
            assert name in text, f"{name}'s refusal does not name the verb: {text!r}"
            assert "`driver`" in text, f"{name}'s refusal does not name the parameter: {text!r}"

    def test_the_refusal_for_a_wrong_type_names_the_type_it_received(self) -> None:
        for name, tool in sorted(_live_handle_verbs().items()):
            text = _refusal_text(_call(tool, "unitree_g1"))
            assert "'str'" in text, f"{name}'s refusal does not name the type received: {text!r}"

    def test_the_omitted_handle_refusal_says_an_agent_cannot_supply_it(self) -> None:
        """The schema cannot carry that constraint, so the refusal has to."""
        text = _refusal_text(snapshot_handle_refusal("g1_probe", None))
        assert "cannot synthesize" in text, text

    def test_no_refusal_leans_on_the_private_attribute_alone(self) -> None:
        """A caller told only ``'NoneType' has no attribute '_snapshot'`` is stuck."""
        text = _refusal_text(snapshot_handle_refusal("g1_probe", None))
        assert "object has no attribute" not in text, text


class TestAHealthyHandleIsUntouched:
    """Over-reach control: the shared guard must not disturb a working handle."""

    def test_the_family_is_not_empty(self) -> None:
        """Non-vacuity: an AST scan that matched nothing would pass both rules."""
        assert _snapshot_family_verbs(), (
            "no verb was found calling snapshot_handle_refusal; the AST scan has "
            "gone blind and both controls below are vacuous"
        )

    def test_an_empty_cache_still_reports_absent(self) -> None:
        for name, tool in sorted(_snapshot_family_verbs().items()):
            result = _call(tool, _CacheOnlyDriver(None))
            assert result["status"] == "success", f"{name}: {result!r}"
            assert result["present"] is False, f"{name}: {result!r}"

    def test_a_written_cache_still_reports_present(self) -> None:
        for name, tool in sorted(_snapshot_family_verbs().items()):
            result = _call(tool, _CacheOnlyDriver({"t": 1.0}))
            assert result["status"] == "success", f"{name}: {result!r}"
            assert result["present"] is True, f"{name}: {result!r}"

    def test_the_guard_answers_none_for_a_handle_that_can_read(self) -> None:
        assert snapshot_handle_refusal("g1_probe", _CacheOnlyDriver(None)) is None


class TestPremises:
    """The two facts the fix rests on, recorded so a change to either is loud."""

    def test_the_precedent_refuses_both_shapes(self) -> None:
        """``run_policy`` is the shape this guard mirrors."""
        from strands_robots.tools.run_policy import run_policy

        # ``DecoratedFunctionTool`` does not declare ``__wrapped__`` in its type,
        # so reading it is a type error even though every ``@tool`` carries it.
        undecorated = run_policy.__wrapped__  # type: ignore[attr-defined]
        omitted = _refusal_text(undecorated(None, n_episodes=1))
        assert "`simulation` is required" in omitted, omitted
        wrong = _refusal_text(undecorated("sim", n_episodes=1))
        assert "does not expose" in wrong, wrong

    def test_the_shared_module_pulls_no_vendor_sdk_at_import(self) -> None:
        """Calling the guard cannot break the import-hygiene pin each verb carries."""
        before = {name for name in sys.modules if "unitree" in name or "cyclonedds" in name}
        importlib.import_module("strands_robots.tools.g1._g1_common")
        after = {name for name in sys.modules if "unitree" in name or "cyclonedds" in name}
        assert after - before == set(), sorted(after - before)

    @pytest.mark.parametrize("guard", ["live_handle_refusal", "snapshot_handle_refusal"])
    def test_the_guard_has_one_owner(self, guard: str) -> None:
        """Six callers share these; a verb restating the rule would drift from them.

        ``live_handle_refusal`` builds the envelope and keeps the four invariants
        this file grades; ``snapshot_handle_refusal`` binds it to the ``_snapshot``
        accessor the five sensor verbs read.  A verb that reimplemented either
        would pass the rules above on the day it landed and drift afterwards,
        which is the failure a single definition site prevents.
        """
        package_dir = Path(g1_package.__file__).parent
        definitions = [
            path.name
            for path in sorted(package_dir.glob("*.py"))
            if any(
                isinstance(node, ast.FunctionDef) and node.name == guard
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            )
        ]
        assert definitions == ["_g1_common.py"], definitions
