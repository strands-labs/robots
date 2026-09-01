#!/usr/bin/env python3
"""Run the whole-tree graders a diff-scoped ``pytest`` selector cannot see.

Why this exists
---------------
Several tests in this repository derive their expectation from the whole tree
rather than from any single file under change:

- ``tests/test_docstring_xref_roles_resolve.py`` -- every fully-qualified
  ``strands_robots.*`` cross-reference role must resolve against the real API.
- ``tests/test_no_host_paths.py`` -- no source file may hard-code a per-machine
  host path (``/Users/<name>``, ``/home/<user>/...`` and siblings).
- ``tests/test_dependency_audit.py`` -- every declared dependency is either
  imported or explicitly documented.
- ``tests/tools/test_agent_tool_parameter_descriptions.py`` -- every
  ``@tool``-decorated verb pins a parameter description for each argument.
- ``tests/test_parameter_deletes_precede_the_body_they_narrow.py`` -- a
  ``del <param>`` inside a function must precede the block it narrows, not
  trail the return.
- ``tests/test_test_module_names_do_not_spell_a_tracker_coordinate.py`` -- no
  test module may be named after the tracker item that birthed it.
- ``tests/tools/test_agent_tool_parameters_reach_the_body.py`` -- no ``@tool``
  may advertise a parameter its body never reads.

These have one property in common: their input is the *rest* of the repository,
not the file under change. A path- or ``-k``-scoped pytest run collects none of
them (``tests/test_docstring_xref_roles_resolve.py`` is not under
``tests/drivers/`` and does not match ``-k g1``, and so on), so a narrow local
selector that reads green does not certify anything about the class of check
that would gate a real merge.

Issue strands-labs/robots#2940 documents two consecutive verb ports
(#2934, #2938) that shipped a dead ``:mod:`` role behind exactly this shape of
green narrow run: the qualified role named a sibling module that lived in a
still-open PR, so it was correct in the author's mental model of the port
series and dead on arrival in the branch's tree. Both were caught by a
reviewer, not by CI on the branch (``call-test-lint`` reads the head alone, so
being behind ``main`` does not update the graders' input).

What this does
--------------
Collects and runs exactly the whole-tree graders, by their fully-qualified
node ids, so a preflight step cannot silently skip one by keyword or path
selection. The roster is a single ``WHOLE_TREE_GRADERS`` tuple below; every
entry is a test file this script guarantees to collect. The tuple is *pinned*
by ``tests/test_whole_tree_graders_roster_is_complete.py`` so that a new
whole-tree grader added to ``tests/`` without an entry here fails a required
check -- which is what keeps the two artifacts (the grader itself and the
preflight command) from drifting the way the two PRs above drifted from
``main``.

Exit codes
----------
- ``0`` -- every grader collected and passed.
- ``1`` -- at least one grader failed, could not be collected, or the
  ``pytest`` invocation itself errored.
- ``2`` -- the invocation was misconfigured (an argument this script does not
  accept, a working directory without a ``tests/`` subtree).

The script forwards its own stdout/stderr from ``pytest`` unchanged, so the
diagnosis a reviewer would read on a red required check is the diagnosis a
preflight run prints locally, byte-for-byte.

Usage
-----
::

    # Direct
    python scripts/check_whole_tree_graders.py

    # Via hatch (installs the test extras first)
    hatch run whole-tree-check

Neither form takes arguments. This is deliberate: the whole point of the
script is that its input set is fixed, not composed by a caller.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# The roster of whole-tree graders. Every entry is a path relative to the
# repository root that ``pytest`` collects as a test module. Order is only
# cosmetic (pytest resolves each independently), and duplicates are refused by
# the roster-completeness pin, not by this script.
#
# Add a new grader here when you add one to ``tests/``. The roster-completeness
# pin will fail a required check otherwise.
WHOLE_TREE_GRADERS: tuple[str, ...] = (
    "tests/test_docstring_xref_roles_resolve.py",
    "tests/test_no_host_paths.py",
    "tests/test_dependency_audit.py",
    "tests/tools/test_agent_tool_parameter_descriptions.py",
    "tests/test_parameter_deletes_precede_the_body_they_narrow.py",
    "tests/test_test_module_names_do_not_spell_a_tracker_coordinate.py",
    "tests/tools/test_agent_tool_parameters_reach_the_body.py",
)


def _repo_root() -> Path:
    """Return the repository root, resolved from this file's location.

    The script lives at ``<root>/scripts/check_whole_tree_graders.py``. Reading
    the parent of ``__file__`` twice reaches the root regardless of the
    caller's working directory, so a preflight invocation from a subshell in
    ``strands_robots/`` behaves the same as one from the root.
    """
    return Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    """Run every entry in :data:`WHOLE_TREE_GRADERS` as a pytest invocation.

    :param argv: The command-line arguments this script received. Only the
        program name is honored; any additional argument prints usage on
        ``stderr`` and returns ``2``. This is intentional -- see the module
        docstring's *Usage* section.
    :returns: The exit code described in the module docstring's *Exit codes*
        section.
    """
    args = argv if argv is not None else sys.argv
    if len(args) > 1:
        sys.stderr.write(
            "check_whole_tree_graders.py takes no arguments; "
            "the grader roster is fixed by design (see module docstring).\n"
        )
        return 2

    root = _repo_root()
    if not (root / "tests").is_dir():
        sys.stderr.write(
            f"check_whole_tree_graders.py: no 'tests/' directory under {root}. "
            "Run this from a clone of strands-labs/robots.\n"
        )
        return 2

    # Verify each grader path exists before invoking pytest. A missing path
    # would make pytest exit 4 ("usage error"), which reads to a caller like
    # the script itself is broken; the earlier check names the missing file
    # instead.
    missing = [p for p in WHOLE_TREE_GRADERS if not (root / p).is_file()]
    if missing:
        sys.stderr.write("check_whole_tree_graders.py: the following graders in the roster do not exist on disk:\n")
        for path in missing:
            sys.stderr.write(f"  - {path}\n")
        sys.stderr.write(
            "Either the grader was moved or renamed and the roster in this "
            "script needs the matching update, or the working tree is stale. "
            "The roster-completeness pin at "
            "tests/test_whole_tree_graders_roster_is_complete.py grades the "
            "opposite direction (a grader present in tests/ that is absent "
            "from the roster).\n"
        )
        return 1

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        # Disable coverage: the pyproject default runs a full-tree ``--cov``
        # sweep that pulls in every ``strands_robots`` import, which is
        # neither useful for a preflight check nor faithful to what the
        # required ``call-test-lint`` job does (it runs a separate coverage
        # gate).
        "--no-cov",
        # ``-p no:cacheprovider`` keeps a preflight run from writing a
        # ``.pytest_cache`` that a subsequent full run would then respect.
        "-p",
        "no:cacheprovider",
        *WHOLE_TREE_GRADERS,
    ]
    result = subprocess.run(cmd, cwd=root, check=False)
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
