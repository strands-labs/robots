"""Every allowlist in this package anchors at the absolute end of the string.

Python's ``$`` matches at the end of the string *or immediately before a single
trailing newline*, so ``re.match(r"^[a-z]+$", "abc\\n")`` succeeds. An allowlist
spelled that way therefore admits a value whose characters it says it forbids.

Two modules already knew, and each wrote the reason down:

* ``strands_robots.mesh.core`` anchors its ``peer_id`` and wire-ZID allowlists
  with ``\\Z``, pinned by
  ``tests/mesh/test_mesh_identifier_trailing_newline_rejected.py``;
* ``strands_robots.mesh.iot.provision`` reaches the same end by calling
  ``fullmatch``, pinned by ``tests/mesh/test_iot_provision.py`` whose
  ``TestValidateThingNameFullmatch`` docstring states the mechanism verbatim --
  "``re.match(r'^[a-zA-Z0-9_-]{1,128}$', s)`` accepts ``'robot\\n'`` because in
  non-MULTILINE mode ``$`` matches just before a trailing newline".

That is a property of ``$``, not of the mesh, and nothing held the rest of the
package to it. These tests do, in two halves: a derived source-level cell over
every regex the package compiles, so an allowlist added later is graded on
arrival rather than when someone thinks to audit it, and behavioural cells that
drive the surfaces where the admitted value reached a wire name, a file name or
a managed-job name.
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest

import strands_robots

_PKG_ROOT = Path(strands_robots.__file__).resolve().parent

#: ``re`` functions whose first positional argument is a pattern.
_RE_FUNCTIONS = frozenset({"compile", "match", "search", "fullmatch", "sub", "subn", "split", "findall", "finditer"})

#: Names that hold a pattern; the assignment is graded even when the call that
#: consults it lives in another module (a class attribute, or a pattern passed
#: to a shared validator).
_PATTERN_NAME_SUFFIXES = ("_RE", "_REGEX", "_PATTERN", "_PAT")


def _has_end_of_line_anchor(pattern: str) -> bool:
    """Whether *pattern* contains a ``$`` acting as an anchor.

    A ``$`` that is escaped (``\\$``) or inside a character class matches a
    literal dollar sign and is not an anchor, so neither is reported.
    """
    index = 0
    in_class = False
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            index += 2
            continue
        if in_class:
            if char == "]":
                in_class = False
        elif char == "[":
            in_class = True
        elif char == "$":
            return True
        index += 1
    return False


def _regex_literals() -> list[tuple[Path, int, str]]:
    """Every regex pattern the package spells as a literal.

    Returns:
        ``(file, line, pattern)`` for each string literal that is either the
        first positional argument of an ``re`` function or the value assigned
        to a name suffixed like a pattern constant.
    """
    found: list[tuple[Path, int, str]] = []
    for path in sorted(_PKG_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _RE_FUNCTIONS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "re"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.append((path, node.args[0].lineno, node.args[0].value))
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(node, ast.Assign):
                targets, value = list(node.targets), node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            if value is None or not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id.endswith(_PATTERN_NAME_SUFFIXES):
                    found.append((path, value.lineno, value.value))
    return found


def test_no_regex_in_the_package_anchors_with_the_end_of_line_marker() -> None:
    """No pattern this package compiles uses ``$``; the anchor is ``\\Z``.

    ``$`` and ``\\Z`` differ on exactly one input -- a value carrying a single
    trailing newline -- and no pattern here is compiled with ``re.MULTILINE``,
    so ``$``'s extra reach is never the intent. Grading the spelling rather
    than each validator's behaviour is what makes this hold for an allowlist
    added after this test was written.
    """
    literals = _regex_literals()
    assert literals, "the sweep resolved no regex literals; its population is wrong, not the package"
    offenders = [
        f"{path.relative_to(_PKG_ROOT.parent)}:{line}: {pattern!r}"
        for path, line, pattern in literals
        if _has_end_of_line_anchor(pattern)
    ]
    assert not offenders, (
        f"{len(offenders)} regex literal(s) anchor with '$', which also matches immediately "
        "before a trailing newline; anchor with '\\\\Z' (or consult the pattern with "
        "fullmatch):\n  " + "\n  ".join(offenders)
    )


@pytest.fixture(autouse=True)
def _isolated_memory_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the ``harness_memory`` surface out of the developer's real store.

    The task name under test becomes a file name, so the cell has to be allowed
    to write one; ``STRANDS_MEMORY_DIR`` relocates the whole store.
    """
    monkeypatch.setenv("STRANDS_MEMORY_DIR", str(tmp_path / "memory"))


def _refuses_value_error(function: Callable[[str], object]) -> Callable[[str], bool]:
    """Adapt a function that raises ``ValueError`` on refusal to a verdict."""

    def refuses(value: str) -> bool:
        try:
            function(value)
        except ValueError:
            return True
        return False

    return refuses


def _refuses_task_name(value: str) -> bool:
    """Whether the ``harness_memory`` tool refuses *value* as a task name.

    The task name becomes the store's file name, so the tool's own allowlist is
    the only thing standing between an agent-supplied string and a path.
    """
    from strands_robots.tools.harness_memory import harness_memory

    result = harness_memory(
        action="save_trace",
        task=value,
        trace=[{"action": "get_state"}],
        summary={"task": "pick the cube", "success": True},
    )
    text = "\n".join(item.get("text", "") for item in result.get("content", []))
    return result["status"] == "error" and "task name" in text


def _refuses_base_job_name(value: str) -> bool:
    """Whether ``SagemakerTrainer.validate`` refuses *value* as a job-name prefix."""
    from strands_robots.training import TrainSpec
    from strands_robots.training.sagemaker import SagemakerTrainer

    trainer = SagemakerTrainer(
        image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/trainer:1",
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        base_job_name=value,
    )
    spec = TrainSpec(dataset_root="s3://bucket/data", output_dir="s3://bucket/out")
    return any("base_job_name" in problem for problem in trainer.validate(spec))


@dataclasses.dataclass(frozen=True)
class _Surface:
    """One allowlist reached through a public entry point.

    Attributes:
        name: Test id.
        clean: A value the allowlist accepts, so a refusal cannot be vacuous.
        refuses: Verdict for one candidate value.
        sink: What the accepted value goes on to become, quoted in the failure.
    """

    name: str
    clean: str
    refuses: Callable[[str], bool]
    sink: str


def _surfaces() -> list[_Surface]:
    """The public entry points graded below.

    ``rtps.mangling`` is imported here rather than at module scope so a failure
    to import is attributed to the surface that needs it.
    """
    from strands_robots.mesh.iot import provision
    from strands_robots.rtps import mangling

    return [
        _Surface(
            "rtps.dds_topic_name",
            "/turtle1/cmd_vel",
            _refuses_value_error(mangling.dds_topic_name),
            "a DDS topic name published on the graph",
        ),
        _Surface(
            "rtps.dds_type_name",
            "std_msgs/msg/String",
            _refuses_value_error(mangling.dds_type_name),
            "a DDS type name whose trailing '_' would land after the line break",
        ),
        _Surface(
            "rtps.ros_topic_name",
            "rt/turtle1/cmd_vel",
            _refuses_value_error(mangling.ros_topic_name),
            "a ROS 2 topic name handed back as recovered from the graph",
        ),
        _Surface(
            "harness_memory.task",
            "pick_cube",
            _refuses_task_name,
            "the trace store's file name",
        ),
        _Surface(
            "sagemaker.base_job_name",
            "strands-train",
            _refuses_base_job_name,
            "the TrainingJobName sent to SageMaker",
        ),
        _Surface(
            "mesh.iot.thing_name",
            "robot-01",
            _refuses_value_error(provision._validate_thing_name),
            "an AWS Thing name and the cert paths derived from it",
        ),
    ]


@pytest.mark.parametrize("surface", _surfaces(), ids=lambda s: s.name)
def test_an_otherwise_valid_value_with_a_trailing_newline_is_refused(surface: _Surface) -> None:
    """A trailing newline does not smuggle a value past an allowlist.

    ``mesh.iot.thing_name`` is the control: it already refused, because it
    consults its pattern with ``fullmatch``. The others differed from it only
    in the spelling of the anchor.
    """
    assert not surface.refuses(surface.clean), (
        f"{surface.name}: the clean value {surface.clean!r} must be accepted, "
        "otherwise the refusal below proves nothing"
    )
    assert surface.refuses(surface.clean + "\n"), (
        f"{surface.name}: {surface.clean + chr(10)!r} was accepted and became {surface.sink}"
    )


def test_the_base_job_name_length_bound_holds_for_a_trailing_newline() -> None:
    """The documented 1-32 bound counts a newline as a character.

    ``_BASE_JOB_NAME_RE`` spends its length bound in a lookahead
    (``(?=.{1,32}\\Z)``), so the anchor decides the bound as well as the
    character set: with ``$`` a 33-character value was refused unless its 33rd
    character was a newline, which the refusal's own text ("1-32 ...
    characters") does not allow for.
    """
    assert not _refuses_base_job_name("a" * 32), "32 characters is inside the documented bound"
    assert _refuses_base_job_name("a" * 33), "33 characters is outside it"
    assert _refuses_base_job_name("a" * 32 + "\n"), "a 33rd character that is a newline is still a 33rd character"


def test_a_dollar_that_is_not_an_anchor_is_not_reported() -> None:
    """The sweep reads regex syntax, not the character.

    A ``$`` inside a character class or escaped is a literal dollar sign, so
    reporting it would make the rule above unsatisfiable for a pattern that
    legitimately matches one.
    """
    assert _has_end_of_line_anchor(r"^[a-z]+$")
    assert _has_end_of_line_anchor(r"^(?=.{1,32}$)[a-z]+\Z")
    assert not _has_end_of_line_anchor(r"^[a-z$]+\Z")
    assert not _has_end_of_line_anchor(r"^\$[0-9]+\Z")
    assert not _has_end_of_line_anchor(r"^[a-z]+\Z")


def test_every_graded_surface_names_a_distinct_module() -> None:
    """The table is a census of modules, not several views of one validator."""
    names: list[str] = [surface.name for surface in _surfaces()]
    assert len(names) == len(set(names))
    modules: set[str] = {name.split(".")[0] for name in names}
    assert len(modules) >= 4, f"only {sorted(modules)} are represented; widen the table"


def test_the_sweep_reads_the_whole_package() -> None:
    """The derived population covers every module, not a listed subset."""
    literals = _regex_literals()
    files = {path for path, _, _ in literals}
    assert len(files) > 20, f"only {len(files)} files contributed a pattern; the sweep is too narrow"
