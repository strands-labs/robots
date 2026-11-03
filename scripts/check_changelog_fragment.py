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

The second question this answers
--------------------------------
A branch that correctly records its change as a fragment adds no heading to the
log, so the comparison above is silent -- correctly. Nothing in it looks at the
fragment's own contents, and nothing else fast did either: fragment validity was
enforced only by ``tests/test_changelog_fragments.py``, inside the one required
check. Measured on #2144, whose only defect was a fragment first line reading
``Fixed: ...`` instead of ``### Fixed: ...``::

    Refuse a changelog entry written outside a fragment    SUCCESS      10s
    call-test-lint / Test and Lint                         FAILURE    20.8 min

The check named for the convention passed it, and the verdict arrived from a
general suite instead -- 125x later, worded as a test failure on a branch whose
diff was 28 lines of ``strands_robots/utils.py`` and two test files. A misnamed
file is the same class reached by a different path: ``collect_fragments`` raises,
so ``validate_fragments`` returns that one message and the same suite carries it.

So this also validates the fragments the branch *adds or modifies*, using the
assembler's own ``FRAGMENT_NAME`` and ``validate_fragment`` rather than a second
copy of either rule -- a copy could refuse a fragment the assembler accepts,
which is the contradiction #2139 had to remove for ``save_episode``.

Two boundaries are deliberate rather than incidental:

- *Only the branch's own fragments.* ``validate_fragments()`` walks the whole
  directory, 315 pending fragments of it, so wiring that in would accuse a branch
  of a malformed fragment already on the base -- the positional-baseline mistake
  #1879 paid a review round for.
- *Bodies read from the object database.* ``git show <head>:<path>``, never the
  working tree, so the verdict stays independent of the checked-out tree. The
  validator itself does come from the checked-out tree, which in CI is the base
  branch, and that is deliberate twice over: the base is the only tree guaranteed
  to carry it, and a branch cannot relax the gate it is judged by in the same diff
  that trips it.

Both remedies stay self-clearing, as the append one is: add the ``### ``, or
rename the file. See issue #2163.

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

Exit status is ``1`` when an unaccounted entry was added or a fragment this
branch adds is invalid, else ``0``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType

#: The log the convention protects, and the anchor within it.
CHANGELOG_PATH = "CHANGELOG.md"
UNRELEASED_HEADING = "## [Unreleased]"

#: Where a behavioural change records itself instead.
FRAGMENT_DIR = "changelog.d"

#: Files in the fragment directory that are documentation, not entries. Kept in
#: step with ``scripts/assemble_changelog.py``'s ``RESERVED_NAMES``; a deleted
#: README is not a consumed fragment and must not excuse an added entry.
RESERVED_NAMES = frozenset({"README.md"})


def _load_assembler() -> ModuleType:
    """Load ``assemble_changelog`` from beside this script.

    By path rather than by name: ``scripts/`` is not a package and is not on
    ``sys.path`` when this module is loaded by its own tests, which use
    ``spec_from_file_location``. Loading by path also keeps the import free of a
    ``sys.path`` mutation at module scope.

    Registered in ``sys.modules`` *before* execution, and not merely returned.
    ``@dataclass`` resolves its own module to decide whether an annotation is a
    ``KW_ONLY`` sentinel -- ``sys.modules.get(cls.__module__).__dict__`` -- so a
    module executed outside the table dies on its first dataclass with
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``, naming
    neither this file nor the reason. An already-loaded copy is reused, so the
    assembler is executed once however many callers ask for it.

    Which tree it comes from is load-bearing. In CI that is the base branch, and
    both reasons matter: the base is the only tree guaranteed to carry the
    validator (issue #1791), and a branch therefore cannot relax the naming or
    heading rule it is judged by in the same diff that trips it.
    """
    loaded = sys.modules.get("assemble_changelog")
    if loaded is not None:
        return loaded

    path = Path(__file__).resolve().parent / "assemble_changelog.py"
    spec = importlib.util.spec_from_file_location("assemble_changelog", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load the fragment validator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["assemble_changelog"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Do not leave a half-executed module behind for the next importer to
        # find and trust. Re-raised lexically, so nothing is swallowed.
        del sys.modules["assemble_changelog"]
        raise
    return module


#: The single source of the fragment naming and heading contracts. Reused rather
#: than restated, so this check and the assembler can never disagree about
#: whether a fragment is valid.
_ASSEMBLER = _load_assembler()


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


def changed_fragments(start: str, end: str, repo: Path | None = None) -> tuple[str, ...]:
    """Return the fragment paths this branch adds or modifies, sorted.

    ``--diff-filter=AM``, because those are the fragments the branch is answerable
    for. A fragment already on the base is not this branch's to fix, and naming it
    would accuse a branch of a defect it did not introduce.

    ``--no-renames`` so a renamed fragment arrives as an add under its new name,
    which is the name that has to satisfy the naming rule.

    The reserved and dotfile skips mirror ``collect_fragments``, so a file the
    assembler never reads as a fragment is not judged as one here either.
    """
    output = _git(
        "diff",
        "--name-only",
        "--no-renames",
        "--diff-filter=AM",
        f"{start}..{end}",
        "--",
        FRAGMENT_DIR,
        repo=repo,
    )
    return tuple(
        sorted(
            line
            for line in output.splitlines()
            if line and Path(line).name not in RESERVED_NAMES and not Path(line).name.startswith(".")
        )
    )


def fragment_problems(
    paths: Iterable[str], head: str, repo: Path | None = None
) -> tuple[tuple[str, str], ...]:
    """Return ``(path, problem)`` for every contract violation in those fragments.

    The verdicts are the assembler's own and are passed through verbatim, so the
    wording a contributor reads here is the wording
    ``python scripts/assemble_changelog.py --check`` prints locally.

    A name that fails ``FRAGMENT_NAME`` short-circuits: ``collect_fragments``
    would refuse the whole directory on it, so there is no ``Fragment`` to build
    and reporting the heading of a file the assembler will not read would be
    noise.

    Bodies come from ``git show``, never from the working tree, which is what
    keeps the verdict independent of the checked-out tree.
    """
    problems: list[tuple[str, str]] = []
    for path in paths:
        name = Path(path).name
        match = _ASSEMBLER.FRAGMENT_NAME.match(name)
        if match is None:
            problems.append(
                (
                    path,
                    f"{name}: not a valid fragment name - expected '<number>-<slug>.md' "
                    "(lowercase slug, words joined by '-')",
                )
            )
            continue
        fragment = _ASSEMBLER.Fragment(
            path=Path(path),
            number=int(match.group("number")),
            body=file_at(head, path, repo=repo),
        )
        problems.extend((path, problem) for problem in _ASSEMBLER.validate_fragment(fragment))
    return tuple(problems)


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
    problems: Sequence[tuple[str, str]] = (),
) -> str:
    """Render the job-summary report for this run."""
    lines = [
        "## Changelog fragment check",
        "",
        f"Base `{base_ref}`, merge base `{merge_base_sha[:12]}`.",
        "",
    ]

    if not added:
        # No early return: a branch can add no entry to the log and still add a
        # fragment the assembler would refuse, which is the #2144 shape.
        lines += [
            f"No entry was added to `{CHANGELOG_PATH}`'s `{UNRELEASED_HEADING}` section.",
            "",
        ]

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

    if problems:
        lines += [
            f"This branch adds or changes {len(problems)} changelog fragment"
            + ("" if len(problems) == 1 else "s")
            + " the assembler would refuse:",
            "",
        ]
        # The assembler prefixes every verdict with the fragment's own name, and
        # the line already names the path, so drop that prefix rather than print
        # the name twice. Only the exact prefix is dropped: an unrecognised
        # wording is passed through whole rather than sliced blind, and the
        # annotation below is always the assembler's message byte for byte.
        for path, problem in problems:
            prefix = f"{Path(path).name}: "
            detail = problem[len(prefix) :] if problem.startswith(prefix) else problem
            lines.append(f"- `{path}`: {detail}")
        lines += [
            "",
            "Each verdict is `python scripts/assemble_changelog.py --check`'s own, so that command "
            + "reproduces this locally. Only the fragments this branch adds or modifies are read, "
            + f"so nothing already in `{FRAGMENT_DIR}` is attributed to it.",
            "",
            f"An invalid fragment is a hard error rather than a skip - see `{FRAGMENT_DIR}/README.md` - "
            + "so the entry would otherwise be dropped from the assembled log entirely.",
            "",
        ]

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
        problems = fragment_problems(
            changed_fragments(fork_point, args.head, repo=repo), args.head, repo=repo
        )
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
        problems=problems,
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

    for path, problem in problems:
        print(f"::error file={path},line=1::{problem}")

    return 1 if unaccounted or problems else 0


if __name__ == "__main__":
    sys.exit(main())
