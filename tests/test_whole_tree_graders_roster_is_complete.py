"""Pin: the preflight roster is derived from the tree, so a new grader joins it.

Why this exists
---------------
``scripts/check_whole_tree_graders.py`` runs the tests whose input is the
*rest* of the repository rather than the file under change - the class of
check a diff-scoped ``pytest`` selector (a ``-k`` keyword, a
``tests/drivers/`` path) does not collect. Issue #2940 documents the failure
mode: two consecutive verb-port PRs (#2934, #2938) cited a green
``pytest tests/drivers/ -k g1`` in their descriptions and both landed on CI
with ``call-test-lint`` red on the exact class of grader a narrow selector
skipped.

That remedy was first written as a hand-maintained roster, and issue #3105
records the second-order failure: a grader added later is absent from a hand
list *by default*, and its absence is silent in the reassuring direction. A
branch cited a green preflight over a roster of seven while the required check
went red on ``tests/test_mesh_pacing_ticker.py``, which walks the installed
package and was never named. The preflight could not have said otherwise - it
never collected the file that failed.

So the roster is now derived: a grader whose population is the tree is one that
walks the repository root or a top-level Python area. This module grades that
derivation, in both directions, because a derivation that silently selects
nothing reads exactly like a tree with nothing to select:

- the roster covers the graders the two issues name, including #3105's live
  instance;
- the derivation is not vacuous, and each spelling of "the package root" a
  grader in this tree actually uses resolves (a resolver blind to one spelling
  is the defect #3105 describes, one layer down);
- a walk of a fixture directory or a ``tmp_path`` is *not* selected, which is
  the property that makes deriving viable at all - an earlier AST-shape scan
  was rejected because it could not tell those apart;
- every entry in ``UNDERIVABLE_GRADERS`` is genuinely invisible to the
  derivation, so that list cannot quietly grow back into the hand roster this
  replaced.

A roster only helps a caller who knows to run it. Issue #2940's failure mode
was a *green* narrow run, so the remedy is only reachable if the pre-push
instructions a contributor reads name the command - which is why
``test_agents_md_names_the_preflight_command`` reads the command's name out of
``pyproject.toml`` and requires ``AGENTS.md`` to name it. Renaming the hatch
script then fails here rather than leaving the document pointing at a command
that no longer exists.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_ROOT.parent

_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_whole_tree_graders.py"
_spec = importlib.util.spec_from_file_location("_cwtg", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None, (
    f"pin cannot locate the preflight script at {_SCRIPT_PATH}. Either the "
    "script moved and this pin needs the matching path, or the working tree "
    "is missing scripts/check_whole_tree_graders.py entirely."
)
_cwtg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cwtg)


# The graders the two issues name by filename. #2940 named five; #3105 named
# the sixth, which a hand roster had omitted and which is the reason the roster
# is derived rather than written down.
_NAMED_BY_ISSUES: frozenset[str] = frozenset(
    {
        "tests/test_docstring_xref_roles_resolve.py",
        "tests/test_no_host_paths.py",
        "tests/test_dependency_audit.py",
        "tests/tools/test_agent_tool_parameter_descriptions.py",
        "tests/test_parameter_deletes_precede_the_body_they_narrow.py",
        "tests/test_mesh_pacing_ticker.py",
    }
)


@pytest.fixture(scope="module")
def roster() -> tuple[str, ...]:
    """Every grader a preflight run of this tree collects."""
    return _cwtg.roster(_REPO_ROOT)


@pytest.fixture(scope="module")
def derived() -> tuple[str, ...]:
    """The half of the roster read off the tree."""
    return _cwtg.derive_graders(_REPO_ROOT)


def _selects(source: str, module_path: Path) -> bool:
    """Whether the derivation would call a module with ``source`` a whole-tree grader.

    :param source: Planted module source.
    :param module_path: Where the module would live, so ``__file__`` resolves.
    """
    walked = _cwtg.walked_paths(source, module_path, _REPO_ROOT)
    return bool(walked & _cwtg.walk_targets(_REPO_ROOT))


def test_the_roster_covers_every_grader_the_issues_name(roster: tuple[str, ...]) -> None:
    """The graders #2940 and #3105 name are all collected by a preflight run.

    #3105's instance is the one that matters here: it was absent from the hand
    roster, so a preflight run over that roster passed while the required
    check failed on it.
    """
    missing = _NAMED_BY_ISSUES - set(roster)
    assert not missing, (
        "a preflight run does not collect graders that issues #2940 and #3105 "
        "name by filename. Each walks the tree, so a diff-scoped pytest "
        "selector does not collect it either, and the failure mode both issues "
        "document returns unreported:\n" + "\n".join(f"  - {path}" for path in sorted(missing))
    )


def test_the_derivation_is_not_vacuous(derived: tuple[str, ...]) -> None:
    """A derivation that selected nothing would report a clean sweep.

    The floor is deliberately far below the measured count (60 at the time of
    writing): the point is to fail when the resolver breaks and selects
    almost nothing, not to pin a number that every added grader edits.
    """
    assert len(derived) >= 20, (
        f"the derivation selected only {len(derived)} whole-tree graders. It "
        "resolves each walk's receiver to a concrete path, so a change that "
        "stops resolving the spellings this tree uses makes the preflight "
        "silently collect almost nothing."
    )


@pytest.mark.parametrize(
    ("label", "root_expression"),
    [
        ("__file__ ancestry", "ROOT = pathlib.Path(__file__).resolve().parents[1]"),
        ("module __file__", "ROOT = pathlib.Path(strands_robots.__file__).resolve().parent"),
        ("inspect.getfile", "ROOT = pathlib.Path(inspect.getfile(strands_robots)).parent"),
        ("helper returning the root", "def _root():\n    return pathlib.Path(inspect.getfile(strands_robots)).parent"),
    ],
)
def test_every_spelling_of_the_root_a_grader_uses_resolves(label: str, root_expression: str) -> None:
    """Each way this tree names the repository or package root is selected.

    Graders reach the same directory three ways, and a resolver that knew only
    one would report a clean sweep over a tree using another - which is the
    shape of defect #3105 describes, one layer down. ``label`` names the
    spelling under test so a failure says which one stopped resolving.
    """
    walker = "_root()" if root_expression.startswith("def ") else "ROOT"
    source = f"import inspect\nimport pathlib\n\nimport strands_robots\n\n{root_expression}\n\nfor p in {walker}.rglob('*.py'):\n    pass\n"
    assert _selects(source, _TESTS_ROOT / "test_planted.py"), (
        f"the derivation cannot resolve a root spelled as {label}, so a grader "
        "written that way is invisible to the preflight rather than merely "
        "unrostered."
    )


@pytest.mark.parametrize(
    ("label", "walk"),
    [
        ("a fixture directory beside the test", "(pathlib.Path(__file__).parent / 'fixtures').rglob('*.py')"),
        ("a tmp_path handed in by pytest", "tmp_path.rglob('*.py')"),
        (
            "a subpackage of the installed package",
            "(pathlib.Path(strands_robots.__file__).parent / 'policies').rglob('*.py')",
        ),
    ],
)
def test_a_subject_scoped_walk_is_not_a_whole_tree_grader(label: str, walk: str) -> None:
    """Walking a fixture directory, a tmp_path or a subpackage is not selected.

    This is the property that makes deriving viable. An AST-shape scan was
    tried and rejected because a fixture glob is shape-identical to a
    repository walk; it is not value-identical, and resolving the path is what
    separates them. The subpackage case is excluded on its own merit: a
    path-scoped run over the mirroring test directory does collect it, so it
    is not in the class the preflight exists to rescue.
    """
    source = f"import pathlib\n\nimport strands_robots\n\ndef test_it(tmp_path):\n    for p in {walk}:\n        pass\n"
    assert not _selects(source, _TESTS_ROOT / "subject" / "test_planted.py"), (
        f"the derivation selected a module that only walks {label}. Every "
        "subject test that globs its own fixtures would join the preflight, "
        "which is the reason an earlier shape-based scan was rejected."
    )


def test_each_explicitly_named_grader_is_invisible_to_the_derivation(derived: tuple[str, ...]) -> None:
    """``UNDERIVABLE_GRADERS`` holds only graders the derivation cannot see.

    An entry the derivation already selects is a hand-maintained roster
    growing back one line at a time, which is the thing #3105 asks to end.
    """
    redundant = [path for path, _reason in _cwtg.UNDERIVABLE_GRADERS if path in set(derived)]
    assert not redundant, (
        "these graders are named explicitly in UNDERIVABLE_GRADERS but the "
        "derivation already selects them; drop the entries so the list stays "
        "a record of genuine blind spots:\n" + "\n".join(f"  - {path}" for path in redundant)
    )


def test_each_explicitly_named_grader_states_why_it_is_invisible() -> None:
    """Every explicit entry carries a reason.

    A path with no reason cannot be re-examined later: a reader cannot tell
    whether the derivation still misses it or whether the entry outlived the
    blind spot it was added for.
    """
    unexplained = [path for path, reason in _cwtg.UNDERIVABLE_GRADERS if not reason.strip()]
    assert not unexplained, (
        "these UNDERIVABLE_GRADERS entries carry no reason, so a later reader "
        "cannot tell whether the derivation still misses them:\n" + "\n".join(f"  - {path}" for path in unexplained)
    )


def test_every_roster_entry_points_at_a_real_grader_file(roster: tuple[str, ...]) -> None:
    """Every entry in the roster is a file that exists.

    Only ``UNDERIVABLE_GRADERS`` can go stale this way - the derived half is
    read off the tree - and the preflight script already refuses at runtime.
    This pin runs under the required check, catching a stale entry at review
    time rather than at the preflight run that would exit 1 on the same tree.
    """
    missing = [entry for entry in roster if not (_REPO_ROOT / entry).is_file()]
    assert not missing, (
        "the preflight roster names files that do not exist on disk; either the "
        "grader was moved or renamed and scripts/check_whole_tree_graders.py's "
        "UNDERIVABLE_GRADERS needs the matching update, or the entry was added "
        "in error:\n" + "\n".join(f"  - {entry!r}" for entry in missing)
    )


def test_the_roster_runs_each_grader_once(roster: tuple[str, ...]) -> None:
    """No grader is collected twice.

    The derived half and the explicit half are separate sources for one list,
    so an entry could appear in both; a duplicate would run the grader twice
    against the same tree for no added signal.
    """
    duplicates = sorted({entry for entry in roster if list(roster).count(entry) > 1})
    assert not duplicates, (
        "the preflight roster carries duplicate entries, so these graders run "
        "twice against the same tree:\n" + "\n".join(f"  - {entry!r}" for entry in duplicates)
    )


def test_script_has_no_arguments_beyond_the_program_name() -> None:
    """The preflight script's ``main`` refuses extra arguments.

    The input set is a property of the tree, not something a caller composes.
    A future edit that accepted an argument would break that guarantee
    silently for any caller who happened not to pass one; this pin makes such
    an edit a required-check failure.
    """
    # A ``main`` that returns 2 on any extra arg is the guarantee; check
    # the behaviour directly instead of an AST scan for signatures.
    assert _cwtg.main(["prog", "--anything"]) == 2, (
        "check_whole_tree_graders.py accepted an argument other than its "
        "program name. The input set is derived from the tree; a "
        "caller-composed one defeats the point of the preflight script."
    )


def _preflight_hatch_script_name() -> str:
    """The hatch script name that runs the preflight, read from ``pyproject.toml``.

    Derived rather than repeated so this pin grades the document against the
    command that exists, not against a string a previous edit happened to
    write down.
    """
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    envs = pyproject.get("tool", {}).get("hatch", {}).get("envs", {})
    script_path = _SCRIPT_PATH.relative_to(_REPO_ROOT).as_posix()
    names = sorted(
        {
            name
            for env in envs.values()
            for name, command in (env.get("scripts") or {}).items()
            if script_path in (command if isinstance(command, str) else " ".join(command))
        }
    )
    assert len(names) == 1, (
        f"expected exactly one hatch script to invoke {script_path!r}, found {names}. "
        "This pin derives the documented command from pyproject.toml; with zero or "
        "several it cannot say which name AGENTS.md should carry."
    )
    return names[0]


def test_agents_md_names_the_preflight_command() -> None:
    """The pre-push instructions name the command that runs these graders.

    The graders in the roster are the ones a diff-scoped selector structurally
    cannot collect, so an author who narrows the run to the area they changed
    gets a green result that reads exactly like a green full run - the failure
    mode issue #2940 documents, observed three times. A preflight command that
    exists but is unnamed in the document a contributor reads before pushing
    does not close it.
    """
    command = f"hatch run {_preflight_hatch_script_name()}"
    agents_md = _REPO_ROOT / "AGENTS.md"
    assert agents_md.is_file(), f"pin cannot locate {agents_md}"
    assert command in agents_md.read_text(encoding="utf-8"), (
        f"AGENTS.md does not name {command!r}. The roster in "
        f"{_SCRIPT_PATH.relative_to(_REPO_ROOT).as_posix()} only helps a caller who "
        "knows to run it, and the graders it collects are exactly the ones a "
        "path-or-keyword selector over the changed area does not. Name it in the "
        "pre-push instructions, or rename the hatch script and update both."
    )
