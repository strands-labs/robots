#!/usr/bin/env python3
"""Report the files a pull request edits that its base also changed since it branched.

Why this exists
---------------
``main`` went red at ``0e636f8`` from two pull requests that were each
individually green and textually non-conflicting. #1766 and #1763 both edit
``_recompile_preserving_state`` in
``strands_robots/simulation/mujoco/scene_ops.py``, for unrelated reasons. #1766
landed first. Every signal the merge gate offers then read green on #1763 --
``reviewDecision: APPROVED``, ``statusCheckRollup: SUCCESS``,
``mergeable: MERGEABLE``, ``mergeStateStatus: CLEAN`` -- and the squash still
broke the suite, because #1763 carried a *premise* test asserting the exact
defect #1766 had just fixed::

    FAILED test_a_tendon_driven_actuator_is_outside_the_joint_matched_id_scope
            AssertionError: assert 2 not in [2]

None of those four signals could have caught it. They are all computed against
the base the branch was tested on: #1763's checks ran against ``32dc3f5b``,
which predates #1766, so the first evaluation of the two changes *together* was
``main`` itself. ``mergeStateStatus: CLEAN`` in particular is a statement about
**text** -- git had no conflicting hunks to report, and it is not git's job to
know that one branch's assertion describes the other branch's bug.

What this computes
------------------
The overlap between two path sets, both taken from the branch's merge base ``M``
with its base branch:

- ``M..head`` -- what the pull request edits.
- ``M..base`` -- what landed on the base branch after the pull request branched.

A non-empty intersection does not prove the combination is broken. It proves
something weaker and still worth blocking on: **the combination has never been
compiled**, so every green check on the pull request is evidence about a tree
that is not the tree being merged. For the pair above the intersection is
exactly one entry, ``strands_robots/simulation/mujoco/scene_ops.py``, and one
``pytest`` invocation over the two touched test files would have caught it.

Why the remedy is cheap, and self-clearing
------------------------------------------
Merging the base branch into the pull request advances the merge base to the
base tip. The ``M..base`` set becomes empty, so the intersection does too, and
the checks then re-run against a base that *contains* the newly-landed commits.
The check needs no override to clear: doing the thing it asks for makes it pass.

This is the targeted form of branch protection's "Require branches to be up to
date before merging". That setting demands an update plus a full re-run before
*every* merge, which serialises merges and costs a ~14.7k-test suite each time.
This demands one only when the branch and its base actually edited the same
file, which is the only case where the base moving can invalidate a result.

Prose is reported but does not block
------------------------------------
An overlap confined to ``.md`` / ``.rst`` / ``.txt`` cannot change what the test
suite or the built package does; if two branches edit the same prose region, git
reports a conflict and the merge gate already stops it. Those paths are listed
in the report -- suppressing them entirely would hide a signal a reader may
want -- but they do not set the exit status, so a docs PR that happens to share
a file with a landed docs PR is not asked to re-run a full test suite for a
result that cannot change.

Usage
-----
``--base-ref``  the branch being merged into (default ``main``). Resolved as
                ``origin/<ref>`` when that exists, else as ``<ref>``.
``--head``      the commit under test (default ``HEAD``). In CI this must be the
                pull request's *head* commit, never the
                ``refs/pull/<n>/merge`` commit ``actions/checkout`` produces by
                default -- that commit already contains the base tip, which
                drives the merge base to the base tip and the overlap to the
                empty set, so the check would pass unconditionally. CI *names*
                that commit rather than checking it out, and runs this script
                from the base branch instead: a branch that forked before a gate
                landed does not carry that gate's script (issue #1791). Sound
                because every input below is read from the object database and
                never from the working tree.
``--repo``      repository root (default: the current working directory).

Exit status is ``1`` when a behaviour-bearing path overlaps, else ``0``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

#: Suffixes whose overlap cannot change the outcome of the test suite or the
#: contents of the built package, and so is reported without blocking.
PROSE_SUFFIXES = frozenset({".md", ".rst", ".txt"})


class GitError(RuntimeError):
    """A git invocation this script depends on did not succeed.

    Raised rather than returning a sentinel: every caller here needs the real
    commit or path set to say anything true, and a check that silently reports
    "no overlap" because it could not reach the base branch is worse than one
    that fails loudly. A missing base ref in CI is a workflow bug, not a
    property of the pull request.
    """


def _git(*args: str, repo: Path | None = None) -> str:
    """Run one git command and return its stdout, raising ``GitError`` on failure."""
    command = ["git"]
    if repo is not None:
        command += ["-C", str(repo)]
    command += list(args)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise GitError(f"{' '.join(command)} exited {completed.returncode}: {completed.stderr.strip()}")
    return completed.stdout


def resolve_base_ref(base_ref: str, repo: Path | None = None) -> str:
    """Return the revision to treat as the base branch tip.

    Prefers the remote-tracking ref, because a CI checkout of a pull request
    head usually has no local branch for the base: ``actions/checkout`` fetches
    the base as ``refs/remotes/origin/<ref>`` and never creates ``<ref>``. Falls
    back to the bare name so the script is runnable in a normal local clone.
    """
    for candidate in (f"origin/{base_ref}", base_ref):
        try:
            _git("rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}", repo=repo)
        except GitError:
            continue
        return candidate
    raise GitError(f"cannot resolve base ref {base_ref!r} as either 'origin/{base_ref}' or '{base_ref}'")


def merge_base(base: str, head: str, repo: Path | None = None) -> str:
    """Return the commit where ``head`` diverged from ``base``."""
    revision = _git("merge-base", base, head, repo=repo).strip()
    if not revision:
        raise GitError(f"no merge base between {base!r} and {head!r} - is the history shallow?")
    return revision


def changed_paths(start: str, end: str, repo: Path | None = None) -> frozenset[str]:
    """Return the paths that differ between two commits.

    ``--no-renames`` is deliberate. With rename detection a file the base
    renamed appears only under its new name, so a pull request still editing the
    old name would not intersect it. Reporting a rename as its delete plus its
    add puts both names in the set, which is the conservative direction for a
    check whose failure mode is a missed overlap.
    """
    output = _git("diff", "--name-only", "--no-renames", f"{start}..{end}", repo=repo)
    return frozenset(line for line in output.splitlines() if line)


def is_prose(path: str) -> bool:
    """Whether a path is documentation, and so reported without blocking."""
    return Path(path).suffix.lower() in PROSE_SUFFIXES


def partition_overlap(paths: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split overlapping paths into ``(behaviour_bearing, prose)``, each sorted.

    Sorting is what makes the report and the annotations reproducible across
    runs: ``git diff`` order follows the tree, and a set has no order at all.
    """
    ordered = sorted(set(paths))
    return (
        tuple(path for path in ordered if not is_prose(path)),
        tuple(path for path in ordered if is_prose(path)),
    )


def overlapping_paths(pr_paths: Iterable[str], base_paths: Iterable[str]) -> tuple[str, ...]:
    """Return the sorted paths edited both by the pull request and by its base."""
    return tuple(sorted(frozenset(pr_paths) & frozenset(base_paths)))


def render_report(
    *,
    base_ref: str,
    merge_base_sha: str,
    blocking: Sequence[str],
    prose: Sequence[str],
    base_change_count: int,
) -> str:
    """Render the Markdown report written to stdout and the CI job summary.

    Every multi-line paragraph below is built as a named local with explicit
    ``+`` rather than from adjacent literals inside the ``lines`` list. Implicit
    concatenation there is indistinguishable from a forgotten comma: a paragraph
    split across two elements silently becomes two report lines, and two
    paragraphs missing their separator silently become one. The join that
    produces the report cannot tell the difference, and neither can a reader.
    """
    lines = ["## Merge-base overlap check", ""]

    if not blocking and not prose:
        no_overlap = (
            f"No overlap. This branch edits nothing that `{base_ref}` has changed since "
            + f"the two diverged at `{merge_base_sha[:8]}` "
            + f"({base_change_count} path(s) changed on `{base_ref}` in that span)."
        )
        lines.append(no_overlap)
        lines.append("")
        lines.append("The checks on this branch were computed against a base that cannot have invalidated them.")
        return "\n".join(lines) + "\n"

    if blocking:
        heading = (
            f"This branch and `{base_ref}` have both changed **{len(blocking)}** "
            + f"behaviour-bearing path(s) since they diverged at `{merge_base_sha[:8]}`:"
        )
        why = (
            "Every check on this branch ran against a base that predates those commits, "
            + "so the combination has not been compiled. A green result here is evidence "
            + "about a different tree than the one that would be merged."
        )
        remedy = (
            "**To clear this:** merge "
            + f"`{base_ref}` into this branch and push. That advances the merge base, "
            + "re-runs the checks against a base containing the landed commits, and makes "
            + "this check pass. Run the tests covering the paths above first - that is "
            + "the cheap part, and it is what the check exists to prompt."
        )
        lines.append(heading)
        lines.append("")
        lines += [f"- `{path}`" for path in blocking]
        lines.append("")
        lines.append(why)
        lines.append("")
        lines.append(remedy)

    if prose:
        if blocking:
            lines.append("")
        prose_heading = (
            f"Also overlapping, not blocking ({len(prose)} documentation path(s)) - "
            + "prose cannot change what the suite or the package does, and a genuine "
            + "collision inside one would surface as a merge conflict:"
        )
        lines.append(prose_heading)
        lines.append("")
        lines += [f"- `{path}`" for path in prose]

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Compute the overlap and return the process exit status."""
    parser = argparse.ArgumentParser(
        prog="check_merge_base_overlap.py",
        description="Report files a pull request edits that its base also changed since it branched.",
    )
    parser.add_argument("--base-ref", default="main", help="branch being merged into (default: main)")
    parser.add_argument("--head", default="HEAD", help="commit under test (default: HEAD)")
    parser.add_argument("--repo", default=None, help="repository root (default: current directory)")
    args = parser.parse_args(argv)

    repo = Path(args.repo) if args.repo is not None else None

    try:
        base = resolve_base_ref(args.base_ref, repo=repo)
        fork_point = merge_base(base, args.head, repo=repo)
        pr_paths = changed_paths(fork_point, args.head, repo=repo)
        base_paths = changed_paths(fork_point, base, repo=repo)
    except GitError as error:
        # Loud and non-zero: a check that cannot compute its answer must not
        # report the reassuring one.
        print(f"::error::merge-base overlap check could not run: {error}", file=sys.stderr)
        return 1

    blocking, prose = partition_overlap(overlapping_paths(pr_paths, base_paths))
    report = render_report(
        base_ref=args.base_ref,
        merge_base_sha=fork_point,
        blocking=blocking,
        prose=prose,
        base_change_count=len(base_paths),
    )

    print(report, end="")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report)

    for path in blocking:
        annotation = (
            f"::error file={path}::{path} was also changed on {args.base_ref} after this "
            + "branch diverged; the checks on this branch never compiled the two together."
        )
        print(annotation)

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
