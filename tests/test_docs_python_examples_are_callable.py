"""Every documented Python example must be callable as written.

A prose reference can be read loosely, but a fenced ``python`` block is code a
reader copies. When such a block passes a keyword the callable does not accept,
the example raises :class:`TypeError` on the first line the reader runs, and
nothing in the repository notices: the docs build renders the block verbatim and
the test suite never executes it.

This module grades every ``python`` fence in ``docs/**/*.md`` and ``README.md``
against the real signatures of the public library surface. For each call with at
least one keyword argument whose callee name resolves to a known public
callable, the keyword set must be accepted by at least one same-named candidate.
Names outside the library surface (third-party calls, illustrative helpers
defined in the block itself) are not graded - a missing candidate degrades to
"not graded", never to a failure.

The distinction this pins in practice is a sink one: ``run_policy`` expands
``policy_config`` into the policy constructor and forwards ``policy_kwargs`` to
every ``get_actions()`` call, and it accepts no ``**kwargs``, so a per-call goal
written as ``run_policy(target_pose=...)`` is not a slower path - it is a
``TypeError``.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
from pathlib import Path
from typing import Any

import strands_robots

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent

_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)

# Modules whose public callables are graded. ``strands_robots.tools`` is walked
# so every ``@tool`` entry point is included under its own name.
_CANDIDATE_MODULES: tuple[str, ...] = (
    "strands_robots",
    "strands_robots.registry",
    "strands_robots.policies",
)

# Classes whose public methods are graded, keyed by method name. A documented
# ``sim.run_policy(...)`` cannot be attributed to one backend from the source
# text alone, so every backend contributes a candidate and acceptance by any one
# of them is enough.
_CANDIDATE_CLASSES: tuple[tuple[str, str], ...] = (
    ("strands_robots.simulation.base", "SimEngine"),
    ("strands_robots.simulation.mujoco.simulation", "MuJoCoSimEngine"),
    ("strands_robots.simulation.newton.simulation", "NewtonSimEngine"),
    ("strands_robots.simulation.isaac.simulation", "IsaacSimulation"),
    ("strands_robots.hardware_robot", "Robot"),
    ("strands_robots.policies.base", "Policy"),
    ("strands_robots.dataset_recorder", "DatasetRecorder"),
    ("strands_robots.mesh.core", "Mesh"),
)

# A fence that grades nothing is indistinguishable from a clean sweep, so the
# corpus size is asserted. The floor is well below the current count; it only
# has to fail if the extractor stops reaching the documentation.
_MINIMUM_GRADED_CALLS = 150


def _unwrap(obj: object) -> object:
    """Return the plain function behind a decorated tool, else *obj* itself.

    ``@tool`` replaces a module-level function with a wrapper object whose
    ``__wrapped__`` attribute holds the real signature.
    """
    return getattr(obj, "__wrapped__", obj)


def _candidates() -> dict[str, list[Any]]:
    """Map a callee name to every public callable in the library with that name."""
    found: dict[str, list[Any]] = {}

    def record(name: str, obj: object) -> None:
        target = _unwrap(obj)
        if callable(target):
            found.setdefault(name, []).append(target)

    module_names = list(_CANDIDATE_MODULES)
    tools_pkg = importlib.import_module("strands_robots.tools")
    module_names += [f"strands_robots.tools.{info.name}" for info in pkgutil.iter_modules(tools_pkg.__path__)]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        for name in dir(module):
            if name.startswith("_"):
                continue
            try:
                record(name, getattr(module, name))
            except AttributeError:
                # A lazily exported symbol whose optional dependency is absent.
                continue

    for module_name, class_name in _CANDIDATE_CLASSES:
        klass = getattr(importlib.import_module(module_name), class_name)
        record(class_name, klass)
        for method_name, method in inspect.getmembers(klass, predicate=inspect.isfunction):
            if not method_name.startswith("_"):
                record(method_name, method)

    return found


def _accepted_keywords(candidate: Any) -> set[str] | None:
    """Return the keyword names *candidate* accepts, or ``None`` for any keyword."""
    try:
        parameters = inspect.signature(candidate).parameters
    except (TypeError, ValueError):
        return set()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return set(parameters)


def _callee_name(call: ast.Call) -> str | None:
    """Return the trailing name of a call target (``a.b.run_policy`` -> ``run_policy``)."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _documentation_files() -> list[Path]:
    """Return every markdown file whose ``python`` fences are graded."""
    files = sorted(_REPO_ROOT.glob("docs/**/*.md"))
    readme = _REPO_ROOT / "README.md"
    if readme.exists():
        files.append(readme)
    return files


def _grade_documentation() -> tuple[list[str], int]:
    """Return ``(offenders, graded_call_count)`` for the whole documentation set."""
    candidates = _candidates()
    offenders: list[str] = []
    graded = 0

    for path in _documentation_files():
        text = path.read_text(encoding="utf-8")
        for fence in _PYTHON_FENCE.finditer(text):
            block = fence.group(1)
            fence_line = text[: fence.start()].count("\n") + 1
            try:
                tree = ast.parse(block)
            except SyntaxError:
                # An intentionally elided snippet ("..." inside a signature).
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not node.keywords:
                    continue
                if any(keyword.arg is None for keyword in node.keywords):
                    # ``**mapping`` - the keyword names are not in the source.
                    continue
                name = _callee_name(node)
                if name is None:
                    continue
                same_named = candidates.get(name)
                if not same_named:
                    continue
                graded += 1
                written = {keyword.arg for keyword in node.keywords if keyword.arg}
                accepted_by_any: set[str] = set()
                binds = False
                for candidate in same_named:
                    accepted = _accepted_keywords(candidate)
                    if accepted is None or written <= accepted:
                        binds = True
                        break
                    accepted_by_any |= accepted
                if binds:
                    continue
                rejected = sorted(written - accepted_by_any)
                offenders.append(
                    f"{path.relative_to(_REPO_ROOT)}:{fence_line + node.lineno} "
                    f"{name}(...) passes {rejected}, which no public {name} accepts"
                )
    return offenders, graded


def test_every_documented_keyword_is_one_the_callable_accepts() -> None:
    """No documented example may pass a keyword its callee would reject."""
    offenders, graded = _grade_documentation()
    assert graded >= _MINIMUM_GRADED_CALLS, (
        f"only {graded} documented calls were graded (expected at least "
        f"{_MINIMUM_GRADED_CALLS}); the extractor is no longer reaching the "
        "documentation, so a clean result would be meaningless"
    )
    assert not offenders, "documented examples that raise TypeError as written:\n  " + "\n  ".join(offenders)


def test_run_policy_has_no_goal_parameter_so_a_goal_needs_policy_kwargs() -> None:
    """The sink the documentation must name for a per-call goal is ``policy_kwargs``.

    ``run_policy`` deliberately takes no ``**kwargs``: an unknown keyword is a
    caller mistake, not a payload to forward. So the goal vocabulary a planner
    policy reads from ``get_actions(**kwargs)`` can only arrive via
    ``policy_kwargs``, and the documentation has exactly one correct spelling.
    """
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    parameters = inspect.signature(MuJoCoSimEngine.run_policy).parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    assert "policy_kwargs" in parameters
    for goal_keyword in ("target_pose", "target_joints", "target_velocity", "gait_frequency", "world_update"):
        assert goal_keyword not in parameters, (
            f"run_policy grew a {goal_keyword!r} parameter; the documentation says a "
            "per-call goal travels in policy_kwargs and must be updated with it"
        )


def test_start_task_forwards_a_goal_payload() -> None:
    """``start_task`` accepts ``**policy_kwargs`` so a goal can flow through.

    The premise this replaces asserted the opposite - that the parameter list
    was fixed and a goal-bearing call raised ``TypeError``. That was true, and
    it was the defect: the mesh dispatch collects checkpoint/goal keywords
    (``model_path``, ``target_pose``, ...) from the wire command, and the
    hardware entry points had no way to receive them, so checkpoint providers
    were unrunnable on hardware over the mesh while sim peers accepted the
    same command. The named parameters stay pinned so a rename/reorder cannot
    silently rebind a positional call.
    """
    from strands_robots.hardware_robot import Robot

    parameters = inspect.signature(Robot.start_task).parameters
    var_kw = [p for p in parameters.values() if p.kind is inspect.Parameter.VAR_KEYWORD]
    assert [p.name for p in var_kw] == ["policy_kwargs"]
    assert set(parameters) == {
        "self",
        "instruction",
        "policy_port",
        "policy_host",
        "policy_provider",
        "duration",
        "policy_kwargs",
    }


def test_the_grader_reports_a_keyword_the_callable_would_reject() -> None:
    """A planted bad keyword is reported, so a clean sweep means something.

    Without this the repository-wide assertion could pass because the grader
    accepts everything rather than because the documentation is correct.
    """
    candidates = _candidates()
    accepted = _accepted_keywords(candidates["run_policy"][0])
    assert accepted is not None and "target_pose" not in accepted

    block = "sim.run_policy(robot_name='panda', target_pose=[0.0] * 7)\n"
    tree = ast.parse(block)
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call))
    written = {keyword.arg for keyword in call.keywords}
    binds = any(
        (_accepted_keywords(candidate) is None or written <= (_accepted_keywords(candidate) or set()))
        for candidate in candidates["run_policy"]
    )
    assert not binds, "a goal keyword passed straight to run_policy must not be graded as acceptable"
