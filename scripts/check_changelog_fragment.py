#!/usr/bin/env python3
"""Refuse a pull request that writes a new entry straight into ``CHANGELOG.md``.

Why this exists
---------------
``changelog.d/README.md`` states the rule and its cost: every pull request that
appended straight to ``## [Unreleased]`` inserted at the *same anchor*, so any
two open at once conflicted the moment either merged -- on ordering alone, never
on meaning -- and because stale approvals are dismissed on push, clearing that
conflict cost a full re-approval round per affected branch. ``AGENTS.md`` repeats
it. Sixty-seven fragments and the ten most recent merged pull requests follow it.

Nothing enforced it. Measured on ``ebe2297b``, with three ``### `` entries
inserted directly beneath ``## [Unreleased]``::

    $ python scripts/assemble_changelog.py --check
    changelog fragments OK (66 pending)                      # exit 0
    $ pytest tests/test_changelog_format.py tests/test_changelog_fragments.py
    30 passed

Both suites are about the *shape* of the log and the *contents* of the fragment
directory. Neither can see a fragment that was never written, because a missing
file leaves nothing to inspect. So the one rule the convention rests on was the
one rule with no gate behind it, and a pull request could reach ``APPROVED`` /
``SUCCESS`` / ``CLEAN`` having ignored it -- which is how this check came to be
written.

Why it is a base diff, not a test
---------------------------------
The obvious pin -- assert ``[Unreleased]`` holds no entries, since the assembler
fills it at release time -- cannot be written here: ``[Unreleased]`` on ``main``
already carries **168** entries from before the convention existed. A static
assertion would have to fail on ``main`` today or grandfather a threshold that
silently ratchets. What is actually wrong is not the section's contents but the
*act* of adding to it, and an act is only visible as a difference between two
commits. So this compares the entry headings under ``[Unreleased]`` at the merge
base against the same set at the branch head: the legacy 168 appear on both
sides and cancel, and only what this branch adds is left.

Why the release path is not caught by it
---------------------------------------
``[Unreleased]`` does legitimately gain entries -- ``assemble_changelog.py
--apply`` folds the accumulated fragments into it when a tag is cut. Two
properties of that tool make the exemption exact rather than a blanket "skip
release pull requests":

- it renders a fragment verbatim (``render`` joins ``fragment.body.strip("\\n")``),
  so a folded entry's heading is byte-identical to the one in its fragment; and
- it deletes each fragment it consumed (``fragment.path.unlink()``).

So every entry an assemble run adds is accounted for by a ``changelog.d/*.md``
file *deleted in the same diff* whose heading it is. That is checked per entry,
not per pull request: a release that also hand-writes an extra entry is still
refused, and only the extra one is named.

Collapsing ``[Unreleased]`` into a dated section at release does not trip this
either -- the entries move *out* of ``[Unreleased]``, and a heading added under
``## [1.2.3] - 2026-01-01`` is not an addition to ``[Unreleased]``.

What is deliberately not checked
--------------------------------
Editing an entry already in the log: fixing a typo, rewording a summary,
reordering the section. Those change no heading set, or move headings that were
already present, and none of them create the anchor-conflict this exists to
prevent. Only a *new* entry heading counts, so release bookkeeping and prose
repair stay unimpeded.

The remedy, like the merge-base overlap check's, is self-clearing: move the
entry into ``changelog.d/<number>-<slug>.md`` and the addition disappears from
the diff.

Usage
-----
``--base-ref``  the branch being merged into (default ``main``). Resolved as
                ``origin/<ref>`` when that exists, else as ``<ref>``.
``--head``      the commit under test (default ``HEAD``). CI *names* the pull
                request's head commit here rather than checking it out, and runs
                this script from the base branch instead: a branch that forked
                before this gate landed does not carry the script, so running it
                out of the head tree died with exit 2 before the check began
                (issue #1791). Nothing is lost by that -- every input below is
                read from the object database and never from the working tree, so
                which tree is checked out cannot change the answer. Unlike the
                merge-base overlap check, this one is also not defeated by the
                ``refs/pull/<n>/merge`` commit ``actions/checkout`` produces by
                default: the question here is which entries the head carries
                that the base does not, and a branch's appended entry is on the
                head and absent from the base whichever commit is used. Pinned
                by ``test_a_merge_commit_head_still_sees_the_branchs_append``.
``--repo``      repository root (default: the current working directory).

Exit status is ``1`` when an unaccounted entry was added, else ``0``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

#: The log the convention protects, and the anchor within it.
CHANGELOG_PATH = "CHANGELOG.md"
UNRELEASED_HEADING = "## [Unreleased]"

#: Where a behavioural change records itself instead.
FRAGMENT_DIR = "changelog.d"

#: Files in the fragment directory that are documentation, not entries. Kept in
#: step with ``scripts/assemble_changelog.py``'s ``RESERVED_NAMES``; a deleted
#: README is not a consumed fragment and must not excuse an added entry.
RESERVED_NAMES = frozenset({"README.md"})


class GitError(RuntimeError):
    """A git invocation this script depends on did not succeed.

    Raised rather than returning a sentinel. Every caller needs the real commit
    or file contents to say anything true, and a check that reports "nothing was
    added" because it could not reach the base branch is worse than one that
    fails loudly. An unresolvable base ref in CI is a workflow bug, not a
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


def file_at(revision: str, path: str, repo: Path | None = None) -> str:
    """Return a file's contents at a revision, or ``""`` when it is absent there.

    Absence is not an error: a branch may introduce ``CHANGELOG.md`` itself, and
    an empty base-side log correctly yields an empty base-side entry set.
    """
    try:
        return _git("show", f"{revision}:{path}", repo=repo)
    except GitError:
        return ""


def deleted_fragments(start: str, end: str, repo: Path | None = None) -> tuple[str, ...]:
    """Return the fragment paths this branch deleted, sorted.

    ``--no-renames`` so a fragment recorded as a rename is reported as its delete
    plus its add. A renamed fragment has not been consumed by the assembler and
    must not excuse an added entry, and the conservative direction for this check
    is to see the delete and then find no matching heading.
    """
    output = _git(
        "diff",
        "--name-only",
        "--no-renames",
        "--diff-filter=D",
        f"{start}..{end}",
        "--",
        FRAGMENT_DIR,
        repo=repo,
    )
    return tuple(sorted(line for line in output.splitlines() if line and Path(line).name not in RESERVED_NAMES))


def unreleased_entries(changelog_text: str) -> tuple[str, ...]:
    """Return the ``### `` entry headings under ``[Unreleased]``, in file order.

    Reading stops at the next level-2 heading, so entries belonging to a released
    version are not counted. A log with no ``[Unreleased]`` heading yields an
    empty tuple: there is no section to append to, so nothing can be appended.
    """
    entries: list[str] = []
    inside = False
    for line in changelog_text.splitlines():
        if line.strip() == UNRELEASED_HEADING:
            inside = True
            continue
        if inside and line.startswith("## "):
            break
        if inside and line.startswith("### "):
            entries.append(line.rstrip())
    return tuple(entries)


def fragment_entry(fragment_text: str) -> str | None:
    """Return a fragment's entry heading, or ``None`` if it has none.

    The heading is its first non-blank line, which is the contract
    ``scripts/assemble_changelog.py`` validates fragments against. Returned
    ``rstrip``ed to match ``unreleased_entries``, so trailing whitespace cannot
    make an entry look unaccounted for.
    """
    for line in fragment_text.splitlines():
        if not line.strip():
            continue
        return line.rstrip() if line.startswith("### ") else None
    return None


def added_entries(base_entries: Iterable[str], head_entries: Iterable[str]) -> tuple[str, ...]:
    """Return the entry headings present at the head and not at the base.

    A multiset difference, not a set difference: ``[Unreleased]`` on ``main``
    contains two entries whose headings are both a bare ``### Fixed:``, so a set
    difference would let a branch add a third copy of an existing heading
    unnoticed. Sorted for a reproducible report.
    """
    surplus = Counter(head_entries) - Counter(base_entries)
    return tuple(sorted(surplus.elements()))


def unaccounted_entries(added: Iterable[str], consumed: Iterable[str]) -> tuple[str, ...]:
    """Return the added entries no consumed fragment accounts for.

    Also a multiset difference: folding two fragments in must delete two
    fragments, and one deleted fragment cannot license two identical entries.
    """
    surplus = Counter(added) - Counter(consumed)
    return tuple(sorted(surplus.elements()))


def render_report(
    *,
    base_ref: str,
    merge_base_sha: str,
    added: Sequence[str],
    accounted: Sequence[str],
    unaccounted: Sequence[str],
) -> str:
    """Render the job-summary report for this run."""
    lines = [
        "## Changelog fragment check",
        "",
        f"Base `{base_ref}`, merge base `{merge_base_sha[:12]}`.",
        "",
    ]

    if not added:
        lines += [
            f"No entry was added to `{CHANGELOG_PATH}`'s `{UNRELEASED_HEADING}` section.",
            "",
        ]
        return "\n".join(lines) + "\n"

    if unaccounted:
        lines += [
            f"This branch adds {len(unaccounted)} entr"
            + ("y" if len(unaccounted) == 1 else "ies")
            + f" to `{UNRELEASED_HEADING}` in `{CHANGELOG_PATH}` that no fragment accounts for:",
            "",
        ]
        lines += [f"- `{entry}`" for entry in unaccounted]
        lines += [
            "",
            f"Record each one as `{FRAGMENT_DIR}/<number>-<slug>.md` instead, where `<number>` is this "
            "pull request's number, and drop it from the log. The fragment holds exactly the text that "
            f"would have gone into `{CHANGELOG_PATH}` -- see `{FRAGMENT_DIR}/README.md`.",
            "",
            "Every branch appends at the same anchor, so two doing it at once conflict on ordering alone, "
            + "and clearing that conflict costs a re-approval round because a push dismisses a stale approval. "
            + "A fragment is its own file, so there is nothing to reconcile.",
            "",
            f"`{CHANGELOG_PATH}` is assembled from the accumulated fragments when a tag is cut: "
            "`python scripts/assemble_changelog.py --apply`.",
            "",
        ]

    if accounted:
        lines += [
            f"Accounted for by a fragment this branch consumed ({len(accounted)}):",
            "",
        ]
        lines += [f"- `{entry}`" for entry in accounted]
        lines += [""]

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the two entry sets and return the process exit status."""
    parser = argparse.ArgumentParser(
        prog="check_changelog_fragment.py",
        description="Refuse a pull request that writes a new CHANGELOG.md entry instead of a changelog.d fragment.",
    )
    parser.add_argument("--base-ref", default="main", help="branch being merged into (default: main)")
    parser.add_argument("--head", default="HEAD", help="commit under test (default: HEAD)")
    parser.add_argument("--repo", default=None, help="repository root (default: current directory)")
    args = parser.parse_args(argv)

    repo = Path(args.repo) if args.repo is not None else None

    try:
        base = resolve_base_ref(args.base_ref, repo=repo)
        fork_point = merge_base(base, args.head, repo=repo)
        base_entries = unreleased_entries(file_at(fork_point, CHANGELOG_PATH, repo=repo))
        head_entries = unreleased_entries(file_at(args.head, CHANGELOG_PATH, repo=repo))
        consumed = [
            entry
            for path in deleted_fragments(fork_point, args.head, repo=repo)
            if (entry := fragment_entry(file_at(fork_point, path, repo=repo))) is not None
        ]
    except GitError as error:
        # Loud and non-zero: a check that cannot compute its answer must not
        # report the reassuring one.
        print(f"::error::changelog fragment check could not run: {error}", file=sys.stderr)
        return 1

    added = added_entries(base_entries, head_entries)
    unaccounted = unaccounted_entries(added, consumed)
    accounted = unaccounted_entries(added, unaccounted)

    report = render_report(
        base_ref=args.base_ref,
        merge_base_sha=fork_point,
        added=added,
        accounted=accounted,
        unaccounted=unaccounted,
    )

    print(report, end="")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report)

    for entry in unaccounted:
        print(
            f"::error file={CHANGELOG_PATH}::{entry} was added to {UNRELEASED_HEADING} directly; "
            f"record it as {FRAGMENT_DIR}/<number>-<slug>.md instead (see {FRAGMENT_DIR}/README.md)."
        )

    return 1 if unaccounted else 0


if __name__ == "__main__":
    sys.exit(main())
