"""Notebook 7's authored task must stay compilable, and its refusals must stay refusals.

``examples/notebooks/07_evaluate_a_policy.ipynb`` teaches layer one of evaluation:
score a policy on a registered task, read the per-attempt rows, then author a new
task as a spec dict. Three properties of it are load-bearing, and none is visible
in a run of the test suite that ignores the notebook:

* **The authored ``SPEC`` compiles against the live predicate registry.** Every
  ``predicate`` name and every keyword in that dict is resolved against
  :data:`~strands_robots.simulation.predicates.PREDICATE_REGISTRY` at
  ``from_dict`` time. Renaming a predicate, or changing one's signature, breaks
  the notebook for the next reader who runs it and for nobody else - the package's
  own tests keep passing, because the spec lives in a ``.ipynb`` that no import
  reaches.

* **The spec is pure data.** The notebook's claim that a spec is safe to load
  from a file an agent produced rests on there being nothing to execute in it.
  This module asserts that by parsing ``SPEC`` with :func:`ast.literal_eval`
  rather than ``exec`` - a spec that grew a function call, an f-string or a name
  reference would fail here, which is the same property the loader depends on.

* **The two broken specs are still refused.** The notebook prints them as
  evidence that the registry is closed. If an unknown predicate name or an
  unexpected keyword ever became acceptable, the notebook would be teaching
  something false, and the cell's own ``else`` branch would raise for a reader
  rather than for CI.

The notebook is the artifact under test, so this module reads its cells rather
than a copy of the spec - a copy would keep passing after the notebook regressed.
Cells are selected by content, not by index, so inserting a cell above them does
not silently point these assertions at the wrong code.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.predicates import PREDICATE_REGISTRY

_NOTEBOOK = Path(__file__).resolve().parent.parent / "examples" / "notebooks" / "07_evaluate_a_policy.ipynb"


def _code_cell_containing(needle: str) -> str:
    """Source of the one code cell containing ``needle``.

    Selected by content so a cell inserted above it does not shift an index.
    """
    nb = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    matches = [
        "".join(cell.get("source", []))
        for cell in nb.get("cells", [])
        if cell.get("cell_type") == "code" and needle in "".join(cell.get("source", []))
    ]
    assert len(matches) == 1, (
        f"expected exactly one code cell containing {needle!r} in {_NOTEBOOK.name}, found {len(matches)}. "
        "The cell moved or was duplicated; update this test to match."
    )
    return matches[0]


@pytest.fixture(scope="module")
def authored_spec() -> dict[str, Any]:
    """The notebook's ``SPEC`` dict, parsed as a literal rather than executed.

    ``literal_eval`` is the assertion, not a convenience: it accepts only
    containers, strings and numbers, so a spec carrying anything executable
    fails here. That is the property the loader's safety claim rests on.
    """
    source = _code_cell_containing("SPEC = {")
    module = ast.parse(source, filename=str(_NOTEBOOK))
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SPEC" for t in node.targets)
    ]
    assert len(assignments) == 1, f"expected one SPEC assignment in {_NOTEBOOK.name}, found {len(assignments)}"
    try:
        spec = ast.literal_eval(assignments[0].value)
    except ValueError as exc:  # pragma: no cover - fires only on a regressed notebook
        raise AssertionError(
            f"{_NOTEBOOK.name}'s SPEC is no longer a pure literal ({exc}). The notebook tells the reader "
            "a spec is safe to load from an agent-produced file because there is nothing to execute in "
            "it; a call, name reference or f-string in the spec makes that claim false."
        ) from exc
    assert isinstance(spec, dict), f"SPEC parsed to {type(spec).__name__}, expected dict"
    return spec


def _predicate_entries(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Every ``{predicate: ..., **kwargs}`` entry the spec names, from all three clauses."""
    entries: list[dict[str, Any]] = []
    for clause_key in ("success", "failure"):
        clause = spec.get(clause_key) or {}
        for group in ("all", "any"):
            entries.extend(clause.get(group, []))
    entries.extend(spec.get("dense_reward", []))
    return entries


def test_the_authored_spec_compiles_against_the_live_predicate_registry(authored_spec: dict[str, Any]) -> None:
    """The notebook's task must compile, or the notebook is broken for its reader.

    This is the assertion that catches a renamed predicate or a changed
    predicate signature, neither of which any other test in the suite would
    notice, because nothing imports a notebook.
    """
    benchmark = DeclarativeBenchmark.from_dict(authored_spec)
    assert benchmark.name == authored_spec["name"]
    assert benchmark.default_robot == authored_spec["default_robot"]


def test_every_condition_the_notebook_names_is_in_the_closed_registry(authored_spec: dict[str, Any]) -> None:
    """Named individually, so a failure reports which predicate went missing.

    ``from_dict`` above already refuses an unknown name, but it stops at the
    first one and reports it inside a longer message. Checking each name gives
    the whole list in one run, which is what a maintainer renaming a predicate
    needs.
    """
    missing = sorted(
        {
            entry["predicate"]
            for entry in _predicate_entries(authored_spec)
            if entry["predicate"] not in PREDICATE_REGISTRY
        }
    )
    assert not missing, (
        f"{_NOTEBOOK.name} names predicates that are no longer registered: {missing}. "
        f"Registered names: {sorted(PREDICATE_REGISTRY)}"
    )


def test_the_spec_exercises_all_three_clause_kinds(authored_spec: dict[str, Any]) -> None:
    """Without this the compile check above would prove less than it appears to.

    A spec with only a success condition still compiles, so it would pass the
    first test while leaving the failure clause and the reward terms - the two
    parts a reader is most likely to copy - ungraded.
    """
    assert authored_spec.get("success", {}).get("all"), "spec declares no success condition"
    assert authored_spec.get("failure", {}).get("any"), "spec declares no failure condition"
    assert authored_spec.get("dense_reward"), "spec declares no reward terms"


def test_the_broken_specs_are_refused_before_a_scene_is_built(authored_spec: dict[str, Any]) -> None:
    """Run the notebook's own refusal cell; its ``else`` branch is the assertion.

    The cell asserts for itself that each broken spec raises. Executing it here
    means the notebook's evidence that the registry is closed is checked in CI
    rather than discovered by a reader.
    """
    source = _code_cell_containing("refused at compile time")
    namespace: dict[str, Any] = {"SPEC": authored_spec, "DeclarativeBenchmark": DeclarativeBenchmark}
    exec(compile(source, str(_NOTEBOOK), "exec"), namespace)  # noqa: S102


@pytest.mark.parametrize(
    ("label", "mutation"),
    [
        ("unknown predicate name", {"success": {"all": [{"predicate": "base_beyond_z", "x": 4.0}]}}),
        ("unexpected keyword", {"success": {"all": [{"predicate": "base_beyond_x", "distance": 4.0}]}}),
    ],
)
def test_the_refusals_the_notebook_demonstrates_are_the_real_behaviour(
    authored_spec: dict[str, Any], label: str, mutation: dict[str, Any]
) -> None:
    """The same two refusals, asserted directly rather than through the cell.

    The cell above proves the notebook's printed output is honest. This proves
    the underlying behaviour, so a regression is attributed to the loader rather
    than to the notebook's phrasing.
    """
    with pytest.raises(ValueError):
        DeclarativeBenchmark.from_dict({**authored_spec, **mutation, "name": "should_not_register"})


def test_the_scored_task_is_one_the_builtin_registration_provides() -> None:
    """The notebook indexes ``registry[TASK]``, which raises ``KeyError`` if the name moved.

    Read from the notebook rather than hard-coded here, so renaming the built-in
    fails on the notebook's own choice of task instead of on this test's copy of it.
    """
    from strands_robots.simulation import register_builtin_benchmarks
    from strands_robots.simulation.benchmark import list_benchmarks

    source = _code_cell_containing('TASK = "')
    module = ast.parse(source, filename=str(_NOTEBOOK))
    task_names = [
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TASK" for t in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert len(task_names) == 1, f"expected one TASK assignment in {_NOTEBOOK.name}, found {len(task_names)}"

    register_builtin_benchmarks()
    registered = list_benchmarks()
    assert task_names[0] in registered, (
        f"{_NOTEBOOK.name} scores {task_names[0]!r}, which register_builtin_benchmarks no longer provides. "
        f"Registered: {sorted(registered)}"
    )


def test_the_install_line_upgrades_a_stale_environment() -> None:
    """Same rule the rest of the series is held to, asserted on this notebook's own line.

    ``tests/test_notebook_min_version_docs.py`` sweeps every notebook for this
    already. It is repeated here so a failure names notebook 7 and its markdown
    rather than arriving as one entry in a swept list.
    """
    nb = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    install_lines = [
        line
        for cell in nb["cells"]
        for line in "".join(cell["source"]).splitlines()
        if "pip install" in line and "strands-robots[" in line
    ]
    assert install_lines, f"{_NOTEBOOK.name} states no install line"
    for line in install_lines:
        upgrades = " -U " in line or " --upgrade " in line
        pinned = ">=" in line.split("strands-robots[", 1)[1]
        assert upgrades or pinned, (
            f"{_NOTEBOOK.name} install line leaves a stale release in place: {line.strip()}. Add -U or a >= floor."
        )
