"""Pin: the whole-tree graders named in strands-labs/robots#2940 are in the roster.

Why this exists
---------------
``scripts/check_whole_tree_graders.py`` names the tests whose input is the
*rest* of the repository rather than the file under change - the class of
check a diff-scoped ``pytest`` selector (a ``-k`` keyword, a
``tests/drivers/`` path) does not collect. Issue #2940 documents the failure
mode: two consecutive verb-port PRs (#2934, #2938) cited a green
``pytest tests/drivers/ -k g1`` in their descriptions and both landed on CI
with ``call-test-lint`` red on the exact class of grader a narrow selector
skipped.

The remedy in ``scripts/check_whole_tree_graders.py`` is a fixed roster. This
pin refuses if:

- an entry the roster names does not exist on disk (a rename that dropped the
  script out of sync with the tree), or
- one of the five graders named in issue #2940 is absent from the roster (the
  five the issue documents by name, which are the minimum guarantee this
  script offers a preflight caller).

The pin does *not* try to auto-discover further whole-tree graders. That was
tried and rejected: any test module that imports ``pathlib`` and calls
``.glob(...)`` on a subject subtree looks structurally identical to one that
walks the whole repository, so an AST-shape scan flags several hundred tests
whose input is a single fixture directory. Enumerating that boundary is a
maintainer-scoped decision - which tests are the ones a *narrow* selector
cannot see - and belongs in the roster's own edits, not in an auto-scan that
would demand an "add to exemptions" step every time a subject test uses
``rglob``.

If a new whole-tree grader is added later, the roster in the script grows by
one line, this test's ``_ROSTERED_BY_ISSUE_2940`` set stays unchanged (its
role is to pin the issue's roster, not to enumerate every future one), and
``scripts/check_whole_tree_graders.py``'s module docstring's list of graders
grows alongside.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


# The five graders issue #2940 names by filename. Reproduced verbatim from the
# issue body so the diagnostic can point back at the source of the guarantee.
_ROSTERED_BY_ISSUE_2940: frozenset[str] = frozenset(
    {
        "tests/test_docstring_xref_roles_resolve.py",
        "tests/test_no_host_paths.py",
        "tests/test_dependency_audit.py",
        "tests/tools/test_agent_tool_parameter_descriptions.py",
        "tests/test_parameter_deletes_precede_the_body_they_narrow.py",
    }
)


def test_every_grader_named_in_issue_2940_is_in_the_roster() -> None:
    """The five graders issue #2940 names are all rostered in the script.

    The preflight script exists to run *these* five graders under a single
    command; if one is missing from the roster, a preflight run silently
    skips it and the issue's failure mode returns unreported.
    """
    rostered = set(_cwtg.WHOLE_TREE_GRADERS)
    missing = _ROSTERED_BY_ISSUE_2940 - rostered
    assert not missing, (
        "the preflight script's WHOLE_TREE_GRADERS roster is missing "
        "graders that issue #2940 names by filename; add each of them to "
        "scripts/check_whole_tree_graders.py so a `hatch run "
        "whole-tree-check` collects the class of test the issue documents:\n"
        + "\n".join(f"  - {p!r}," for p in sorted(missing))
    )


def test_every_roster_entry_points_at_a_real_grader_file() -> None:
    """Every entry in the script's roster is a file that exists.

    The preflight script already refuses at runtime when a rostered path is
    missing on disk, but the required call-test-lint check runs this test at
    review time - catching a stale roster before a preflight run that would
    exit 1 on the same tree.
    """
    missing = [entry for entry in _cwtg.WHOLE_TREE_GRADERS if not (_REPO_ROOT / entry).is_file()]
    assert not missing, (
        "the preflight script's WHOLE_TREE_GRADERS roster names files that "
        "do not exist on disk; either the grader was moved/renamed and the "
        "roster needs the matching update, or the entry was added in "
        "error:\n" + "\n".join(f"  - {p!r}" for p in missing)
    )


def test_roster_has_no_duplicate_entries() -> None:
    """A duplicate entry would silently run one grader twice.

    Enforced separately so the diagnostic names the duplicate rather than
    trying to explain a set-equality failure.
    """
    roster = list(_cwtg.WHOLE_TREE_GRADERS)
    seen: set[str] = set()
    duplicates: list[str] = []
    for entry in roster:
        if entry in seen:
            duplicates.append(entry)
        else:
            seen.add(entry)
    assert not duplicates, (
        "the preflight script's WHOLE_TREE_GRADERS roster carries duplicate "
        "entries; each grader should appear exactly once so a preflight run "
        "does not repeat the work of one against the same tree:\n" + "\n".join(f"  - {p!r}" for p in duplicates)
    )


def test_script_has_no_arguments_beyond_the_program_name() -> None:
    """The preflight script's ``main`` refuses extra arguments.

    The whole point of the roster is that the input set is fixed rather than
    composed by a caller. A future edit that accepted an argument would break
    that guarantee silently for any caller who happened not to pass one; this
    pin makes such an edit a required-check failure.
    """
    # A ``main`` that returns 2 on any extra arg is the guarantee; check
    # the behaviour directly instead of an AST scan for signatures.
    assert _cwtg.main(["prog", "--anything"]) == 2, (
        "check_whole_tree_graders.py accepted an argument other than its "
        "program name. The roster is meant to be fixed; a caller-composed "
        "input set defeats the point of the preflight script."
    )
