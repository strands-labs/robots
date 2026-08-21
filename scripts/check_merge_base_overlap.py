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

What one branch cannot see
--------------------------
``M..base`` is empty **by construction** whenever ``M`` is the base the branch
was evaluated against, which is every run: the check clears itself not after
wall-clock time but when the pull request is *re-evaluated*. Two consequences,
both invisible to every per-branch signal:

- For two pull requests that are both still **open**, ``M..base`` contains
  neither of them, so the intersection is empty and the check reports clean on
  both. The first evaluation of the two changes together is ``main``. That is the
  #1763/#1766 topology arriving from the open set rather than from the merged
  base, and it is a property of the *set*, which is the same reason the
  duplicate-claim check had to exist (#2017) -- every other check here reads one
  pull request at a time. This one keys on the file rather than the issue, so two
  pull requests claiming different issues are silent to that check by design.
- Once the sibling merges, nothing invalidates the second pull request's green.
  Stale *approvals* are dismissed on push; a stale *pass* has no equivalent, and
  a pull request idle in review never re-runs. So the exposure is not the interval
  between two merges: it is the interval until the second one's next push, which
  is unbounded for anything sitting in a review queue.

``--all-open`` is the caller for both. It reads the open set from the API and
computes the same intersection twice per pull request -- once against each
sibling's ``M..head``, once against what has landed on the base since its own
``M`` -- reusing the path-set helpers the single-branch mode uses, so the two
modes cannot disagree about what counts as an overlap or as prose. That parity
includes renames: git reports one as its delete plus its add under
``--no-renames``, and the API side takes ``previous_filename`` alongside
``filename`` to reach the same set, because a branch that renames a file and a
branch that edits its old name compose without a conflict marker.

The two sides come from different endpoints, and so have different ceilings. The
head side -- the input to the pairwise mode -- is read from the paginated
pull-request files endpoint, which carries ten times what the compare endpoint's
``files`` does. The base side has no paginated equivalent and keeps the compare
cap, reported as unevaluated for that mode alone.

A file carrying a ``strict=True`` xfail is the highest-value overlap candidate
there is: its whole purpose is to fail when a sibling change lands, so it breaks
a composition that git merges without a single conflict marker. #2233 pinned a
defect that way and #2235 fixed it; composed, the tree was red with no conflict
to resolve, which is why ``mergeStateStatus: CLEAN`` is not merely unhelpful
here but actively reassuring.

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

What the path relation cannot see
---------------------------------
Every relation here intersects changed **paths**, and a test that resolves its
population from a filesystem walk is coupled to files it never names. Its
intersection with the sibling it grades is empty, so the sweep reports clean
while the composition is in fact untested. #2557 added
``tests/test_log_strings_are_ascii.py``, whose population is a walk of the
package, and merged in a batch with #2559 and #2560; both siblings added
tool-result prose, which is exactly the surface that grader scores, and the
pairwise path intersection with each was ``[]``. The batch was safe, but only
because it was checked by hand.

Widening the path set to the *walked root* is not the remedy. Measured on this
repository's open set of 9 pull requests, that relation selects 11 of the 36
pairs -- 9 of them not already reported by the path intersection, and none of
them a defect. The reason is structural: of the 125 walk-population tests in
the tree, the ones that reach furthest are rooted at ``strands_robots`` entire,
so they intersect nearly every open branch, while a narrowly-rooted grader
(``strands_robots/mesh/``) selects only the pair the path intersection already
reports. A relation that fires on a third of all pairs and names no defect is
the ``awaiting-first-review`` failure mode: a finding attached to that much of
the queue reads as boilerplate, and the one batch where it mattered is not
distinguishable from the rest.

What does separate them is composing the two branches and running the grader --
it needs no model of what the grader reads, because the grader reads it. Run by
hand over the open set for #2562's whole-tree grader, that cost about 10 s per
composition and correctly left alone three siblings whose new ``except`` tuples
a naive path-or-keyword heuristic would have flagged. It cannot live in this
mode: the sweep reads the open set from the API and no checkout at all, which
is what lets a caller reporting repository health run it without a clone (pinned
by ``tests/test_merge_base_overlap.py``, whose every sweep test runs from a
directory that is not a repository). So the honest change here is scope, not a
new heuristic: this mode reports what it measured -- shared paths -- and names
the composition class it cannot describe. See #2561.

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
``--repo``      repository root (default: the current working directory). One
                branch only: the sweep reads no checkout, so ``--all-open``
                refuses it rather than ignoring it. The sibling gate scripts
                spell ``owner/name`` ``--repo``, so a caller who reaches for it
                here names a path, the sweep reads ``$GITHUB_REPOSITORY``
                instead, and the report describes a repository nobody asked
                about (issue #2569).

``--all-open``  Sweep the open set instead of one branch (see above). Mutually
                exclusive with ``--head`` and ``--repo``, which name a local
                commit and a local checkout.
``--github-repo``
                ``owner/name`` for ``--all-open`` (default:
                ``$GITHUB_REPOSITORY``). Deliberately not ``--repo``: in this
                script ``--repo`` is already a local checkout path, and one flag
                that means a filesystem path in one mode and a slug in the other
                is the kind of ambiguity a caller discovers by getting a wrong
                answer.
``--token``     API token for ``--all-open`` (default: ``$GITHUB_TOKEN``). Needs
                ``pull-requests: read``.

Exit status is ``1`` when a behaviour-bearing path overlaps, else ``0``. That is a
blocking result for one branch, and a report for ``--all-open``: the sweep's
remedy is a decision about merge order plus possibly one composition run, which
no push by either author turns green, so it is deliberately absent from the
required set. A gate a branch cannot clear by doing anything is a report,
whatever it is wired to.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
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


API_ROOT = "https://api.github.com"

#: Pagination ceiling for the open-pull-request listing, mirroring the sibling
#: sweep in ``scripts/check_last_push_approval.py``. A repository holding more
#: open pull requests than this at once has a different problem, and an unbounded
#: loop over a paginated endpoint is how a transient API shape change becomes a
#: hang rather than an error.
_MAX_PAGES = 20

#: GitHub's compare endpoint returns at most this many entries in ``files``. A
#: truncated list is indistinguishable from a complete one in the payload, so a
#: path set that reaches the cap is reported as unevaluated rather than as not
#: overlapping. This check's failure mode is a *missed* overlap, and quietly
#: intersecting a truncated set is exactly how one goes missing. Only the
#: base-side set is read from this endpoint; the head side has a paginated one.
_COMPARE_FILE_CAP = 300

#: The pull-request files endpoint is paginated and stops at this many entries,
#: ten times the compare cap. Reaching it carries the same ambiguity a capped
#: compare does and is reported the same way -- but it is ten times further
#: away, and the head-side set is the input to the pairwise mode, which is the
#: mode that finds defects. The largest pull request in this repository's history
#: is 153 files (#1667, closed unmerged); the largest open one today is 10.
_PULL_FILE_CAP = 3000

#: Page size for that endpoint. ``_PULL_FILE_CAP`` must stay a whole multiple of
#: it, because the loop bound below is the quotient.
_PULL_FILE_PAGE = 100


class ApiError(RuntimeError):
    """A GitHub API call the sweep depends on did not succeed.

    Separate from ``GitError`` because the two modes have disjoint inputs: the
    single-branch check reads the object database and never the network, and the
    sweep reads the network and never a checkout. One shared exception would let
    a report offer a remedy that cannot apply to the mode that produced it.
    """


@dataclasses.dataclass(frozen=True)
class OpenPullRequest:
    """One open pull request's path sets, both taken from its own merge base.

    ``edits`` comes from the paginated pull-request files endpoint and
    ``landed_since`` from the compare endpoint, so the two sides have different
    ceilings: the base side has no paginated equivalent and keeps
    ``_COMPARE_FILE_CAP``.

    ``landed_since`` is ``None`` when that side could not be read. It is an input
    to the stale-base mode only, so an unreadable one excludes the pull request
    from that mode while leaving it in the pairwise comparison -- which matters:
    the base-side set is the one that grows without bound and so the one that
    hits the file cap, and dropping the whole pull request for it would discard a
    pairwise finding this check exists to make.
    """

    number: int
    head_sha: str
    merge_base: str
    behind_by: int
    edits: frozenset[str]
    landed_since: frozenset[str] | None


def _get(url: str, token: str) -> object:
    """Fetch and decode one API response."""
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "check-merge-base-overlap",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed API host
        return json.load(response)


def resolve_open_pull_requests(repo: str, token: str) -> list[tuple[int, str]]:
    """Return ``(number, head_sha)`` for every open non-draft pull request, sorted.

    Drafts are excluded for the reason the sibling sweep gives: a draft cannot
    merge whatever else is true of it, so a finding on one does not mean what a
    finding here means. Sorting keeps a diff of two reports about changed
    verdicts rather than reordered rows.
    """
    found: list[tuple[int, str]] = []
    for page in range(1, _MAX_PAGES + 1):
        payload = _get(f"{API_ROOT}/repos/{repo}/pulls?state=open&per_page=100&page={page}", token)
        rows = payload if isinstance(payload, list) else []
        for row in rows:
            if not isinstance(row, dict) or row.get("draft"):
                continue
            number = row.get("number")
            head = ((row.get("head") or {}).get("sha")) or ""
            if isinstance(number, int) and head:
                found.append((number, str(head)))
        if len(rows) < 100:
            break
    return sorted(found)


def paths_from_entries(entries: Iterable[object]) -> frozenset[str]:
    """Return every path a file-list entry names, a rename's old name included.

    Both file-carrying endpoints report a rename as a single entry whose
    ``filename`` is the new path and whose ``previous_filename`` is the old one.
    Reading only the first is a false negative in the direction that matters: a
    branch renaming ``foo.py`` and a branch editing ``foo.py`` then share no
    path, while git -- which does detect the rename -- applies the second
    branch's edit to the new name with no conflict to report. That is exactly
    the composition this file exists to find, arriving invisibly.

    This is the API counterpart of ``changed_paths``'s ``--no-renames``, and it
    is what keeps the two modes agreeing about what an overlap is. Taking both
    names can only widen the reported set, which is the safe direction for a
    check whose failure mode is a missed overlap.
    """
    return frozenset(
        str(value)
        for entry in entries
        if isinstance(entry, dict)
        for key in ("filename", "previous_filename")
        if (value := entry.get(key))
    )


def _compare_payload(repo: str, base: str, head: str, token: str) -> tuple[list[object], str, int]:
    """Fetch a three-dot ``base...head`` and return ``(file entries, merge_base_sha, behind_by)``.

    The three-dot form is what makes this the same question the single-branch
    mode asks with git: the ``files`` it reports are the diff from
    ``merge_base(base, head)`` to ``head``, i.e. ``M..head``. Swapping the
    operands therefore yields ``M..base`` from the same endpoint, which is how
    the sweep obtains the base side without resolving a merge base itself or
    fetching a single commit.

    ``_COMPARE_FILE_CAP`` is deliberately not enforced here. Whether a truncated
    ``files`` list is a problem depends on whether the caller reads it, and the
    two callers differ: one wants the paths, the other wants only the two fields
    the paginated endpoint does not carry.
    """
    url = f"{API_ROOT}/repos/{repo}/compare/{base}...{head}"
    payload = _get(url, token)
    if not isinstance(payload, dict):
        raise ApiError(f"{url}: expected an object, got {type(payload).__name__}")
    entries = payload.get("files")
    entries = entries if isinstance(entries, list) else []
    commit = payload.get("merge_base_commit")
    merge_base_sha = str((commit or {}).get("sha") or "") if isinstance(commit, dict) else ""
    behind_by = payload.get("behind_by")
    return entries, merge_base_sha, behind_by if isinstance(behind_by, int) else 0


def compare_paths(repo: str, base: str, head: str, token: str) -> tuple[frozenset[str], str, int]:
    """Return ``(paths, merge_base_sha, behind_by)`` for a three-dot ``base...head``.

    Enforces the compare endpoint's file cap: a path set that reached it is
    incomplete, and intersecting it would report "no overlap" while meaning "did
    not look". Used for the base side, which has no paginated equivalent.
    """
    entries, merge_base_sha, behind_by = _compare_payload(repo, base, head, token)
    if len(entries) >= _COMPARE_FILE_CAP:
        raise ApiError(
            f"{API_ROOT}/repos/{repo}/compare/{base}...{head}: the file list reached the "
            + f"{_COMPARE_FILE_CAP}-entry cap, so the path set is incomplete and an overlap "
            + "computed from it could be a false negative"
        )
    return paths_from_entries(entries), merge_base_sha, behind_by


def compare_fork_point(repo: str, base: str, head: str, token: str) -> tuple[str, int]:
    """Return ``(merge_base_sha, behind_by)`` for ``base...head``, ignoring ``files``.

    The head side takes its paths from ``pull_request_paths``, so this call is
    needed only for the two fields that endpoint does not carry -- and it must
    not fail on a capped ``files`` list it never reads. It once did, via
    ``compare_paths``, which removed a large pull request from the *pairwise*
    comparison as well: the one mode where a 300-file branch is the most likely
    thing on the queue to collide with something.
    """
    _, merge_base_sha, behind_by = _compare_payload(repo, base, head, token)
    return merge_base_sha, behind_by


def pull_request_paths(repo: str, number: int, token: str) -> frozenset[str]:
    """Return the paths pull request ``number`` edits, from the paginated endpoint.

    Below the compare cap this is the same set ``base...head`` reports -- measured
    on this repository at 7, 10 and 153 files (#1035, #1722, #1667), identical
    both ways, and byte-identical across the whole open queue -- so no verdict
    changes. What changes is the ceiling: paginating raises the head side's from
    ``_COMPARE_FILE_CAP`` to ``_PULL_FILE_CAP``.

    Reaching that ceiling is reported like a capped compare, for the same reason:
    the endpoint stops there without saying so, and a silently short path set is
    how a missed overlap is manufactured.
    """
    collected: list[object] = []
    for page in range(1, _PULL_FILE_CAP // _PULL_FILE_PAGE + 1):
        url = f"{API_ROOT}/repos/{repo}/pulls/{number}/files?per_page={_PULL_FILE_PAGE}&page={page}"
        payload = _get(url, token)
        rows = payload if isinstance(payload, list) else []
        collected.extend(rows)
        if len(rows) < _PULL_FILE_PAGE:
            return paths_from_entries(collected)
    raise ApiError(
        f"{API_ROOT}/repos/{repo}/pulls/{number}/files: the file list reached the "
        + f"{_PULL_FILE_CAP}-entry ceiling, so the path set is incomplete and an overlap "
        + "computed from it could be a false negative"
    )


def collect_open_pull_requests(
    repo: str, base_ref: str, token: str
) -> tuple[list[OpenPullRequest], list[tuple[int, str]]]:
    """Read both path sets for every open pull request.

    Returns the rows it could evaluate and ``(number, reason)`` for every side it
    could not. A failure on one pull request is named and skipped rather than
    raised: the sweep exists to surface findings across the standing population,
    and one unreachable pull request must not take the rest of the report with
    it. Naming the skips is the same requirement -- an unevaluated pull request
    that says nothing is the failure mode this whole file is written against.
    """
    rows: list[OpenPullRequest] = []
    unevaluated: list[tuple[int, str]] = []
    lookup_failures = (ApiError, urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError)
    for number, head_sha in resolve_open_pull_requests(repo, token):
        try:
            fork_point, behind_by = compare_fork_point(repo, base_ref, head_sha, token)
            edits = pull_request_paths(repo, number, token)
        except lookup_failures as error:
            unevaluated.append((number, f"not evaluated in either mode - own path set unreadable: {error}"))
            continue
        landed_since: frozenset[str] | None
        try:
            landed_since, _, _ = compare_paths(repo, head_sha, base_ref, token)
        except lookup_failures as error:
            landed_since = None
            unevaluated.append((number, f"stale-base mode only - base-side path set unreadable: {error}"))
        rows.append(
            OpenPullRequest(
                number=number,
                head_sha=head_sha,
                merge_base=fork_point,
                behind_by=behind_by,
                edits=edits,
                landed_since=landed_since,
            )
        )
    return rows, unevaluated


def pair_overlaps(
    pull_requests: Iterable[OpenPullRequest],
) -> list[tuple[int, int, tuple[str, ...], tuple[str, ...]]]:
    """Return ``(left, right, blocking, prose)`` for every pair sharing a path."""
    ordered = sorted(pull_requests, key=lambda row: row.number)
    found: list[tuple[int, int, tuple[str, ...], tuple[str, ...]]] = []
    for left, right in itertools.combinations(ordered, 2):
        blocking, prose = partition_overlap(overlapping_paths(left.edits, right.edits))
        if blocking or prose:
            found.append((left.number, right.number, blocking, prose))
    return found


def stale_base_overlaps(
    pull_requests: Iterable[OpenPullRequest],
) -> list[tuple[int, int, tuple[str, ...], tuple[str, ...]]]:
    """Return ``(number, behind_by, blocking, prose)`` for every stale-base overlap.

    This is the single-branch computation applied to the population: the same
    intersection, with the base branch substituted for a sibling's head. A pull
    request level with its base has nothing to compare, so it is skipped rather
    than reported as clean.
    """
    found: list[tuple[int, int, tuple[str, ...], tuple[str, ...]]] = []
    for row in sorted(pull_requests, key=lambda row: row.number):
        if row.landed_since is None or row.behind_by == 0:
            continue
        blocking, prose = partition_overlap(overlapping_paths(row.edits, row.landed_since))
        if blocking or prose:
            found.append((row.number, row.behind_by, blocking, prose))
    return found


def render_sweep(
    *,
    repo: str,
    base_ref: str,
    pull_requests: Sequence[OpenPullRequest],
    pairs: Sequence[tuple[int, int, tuple[str, ...], tuple[str, ...]]],
    stale: Sequence[tuple[int, int, tuple[str, ...], tuple[str, ...]]],
    unevaluated: Sequence[tuple[int, str]],
) -> str:
    """Render the sweep report.

    Every multi-line paragraph is a named local joined with explicit ``+``, for
    the reason ``render_report`` gives: implicit concatenation inside a list of
    report lines is indistinguishable from a forgotten comma.
    """
    pair_count = len(pull_requests) * (len(pull_requests) - 1) // 2
    lines = ["## Merge-base overlap check - open set", ""]
    header = (
        f"`{repo}`, base `{base_ref}`: {len(pull_requests)} open non-draft pull request(s), "
        + f"{pair_count} pair(s) compared."
    )
    lines.append(header)
    lines.append("")

    blocking_pairs = [row for row in pairs if row[2]]
    prose_pairs = [row for row in pairs if not row[2]]
    blocking_stale = [row for row in stale if row[2]]

    if not blocking_pairs and not blocking_stale:
        clean = (
            "No pair in the open set shares a changed path, and no pull request shares one "
            + "with what has landed on its base. Nothing here needs a merge-order decision."
        )
        lines.append(clean)
        lines.append("")
    if blocking_pairs:
        pair_heading = f"### Pairs editing the same behaviour-bearing path ({len(blocking_pairs)})"
        pair_why = (
            "Neither pull request's checks have compiled the other's changes, and neither can: "
            + "each ran against a base that contains neither. Whichever merges second inherits a "
            + "green result about a tree that no longer exists."
        )
        lines.append(pair_heading)
        lines.append("")
        lines.append(pair_why)
        lines.append("")
        lines.append("| pull requests | shared path(s) |")
        lines.append("|---|---|")
        for left, right, paths, _ in blocking_pairs:
            rendered = ", ".join(f"`{path}`" for path in paths)
            lines.append(f"| #{left} + #{right} | {rendered} |")
        lines.append("")
    if blocking_stale:
        stale_heading = f"### Pull requests whose base moved under a path they edit ({len(blocking_stale)})"
        stale_why = (
            "The single-branch check would report each of these today, and reported none of them "
            + "when it ran: it reads the base as of that run, and the commits below landed after. "
            + "Nothing re-runs a pull request idle in review, so the green stands until its next push."
        )
        lines.append(stale_heading)
        lines.append("")
        lines.append(stale_why)
        lines.append("")
        lines.append(f"| pull request | behind `{base_ref}` by | path(s) also changed on `{base_ref}` |")
        lines.append("|---|---|---|")
        for number, behind_by, paths, _ in blocking_stale:
            rendered = ", ".join(f"`{path}`" for path in paths)
            lines.append(f"| #{number} | {behind_by} | {rendered} |")
        lines.append("")
    if prose_pairs:
        prose_heading = (
            f"Also sharing a path, not reported ({len(prose_pairs)} prose-only pair(s)) - prose "
            + "cannot change what the suite or the package does, and a genuine collision inside one "
            + "surfaces as a merge conflict:"
        )
        lines.append(prose_heading)
        lines.append("")
        for left, right, _, paths in prose_pairs:
            rendered = ", ".join(f"`{path}`" for path in paths)
            lines.append(f"- #{left} + #{right}: {rendered}")
        lines.append("")
    if unevaluated:
        unevaluated_heading = (
            f"Unevaluated ({len(unevaluated)}) - named rather than counted as clean, because a "
            + "pull request this check could not read is not a pull request it cleared:"
        )
        lines.append(unevaluated_heading)
        lines.append("")
        for number, reason in unevaluated:
            lines.append(f"- #{number}: {reason}")
        lines.append("")

    remedy = (
        "**To clear a row above:** decide the merge order, run the tests covering the shared "
        + "paths against the composition once, and merge. This is a report, not a required check: "
        + "no push by either author makes a *pair* green."
    )
    lines.append(remedy)
    lines.append("")
    # Unconditional, and last: it qualifies a clean report at least as much as a
    # populated one. A reader who sees no rows is entitled to know which
    # compositions the relations above were never able to describe.
    limits = (
        "**What no relation above covers:** each one intersects changed *paths*. A test that "
        + "resolves its population from a filesystem walk grades files it never names, so it "
        + "shares no path with the siblings it grades and no row above can describe that "
        + "composition. Widening the path set to the walked root was measured and rejected as "
        + "unselective; only composing the branches and running the grader settles those, and "
        + "that needs a checkout this mode does not have. See #2561."
    )
    lines.append(limits)
    return "\n".join(lines) + "\n"


def _emit(report: str) -> None:
    """Print the report and append it to the CI job summary when there is one."""
    print(report, end="")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report)


def _run_sweep(repo: str, base_ref: str, token: str) -> int:
    """Sweep the open set. Exit 1 when a behaviour-bearing composition is untested."""
    try:
        pull_requests, unevaluated = collect_open_pull_requests(repo, base_ref, token)
    except (ApiError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as error:
        # Listing the pull requests is the one lookup with no partial result to
        # fall back on: without it there is no population to sweep.
        print(f"::error::merge-base overlap sweep could not list open pull requests: {error}", file=sys.stderr)
        return 1

    pairs = pair_overlaps(pull_requests)
    stale = stale_base_overlaps(pull_requests)
    _emit(
        render_sweep(
            repo=repo,
            base_ref=base_ref,
            pull_requests=pull_requests,
            pairs=pairs,
            stale=stale,
            unevaluated=unevaluated,
        )
    )
    return 1 if any(row[2] for row in pairs) or any(row[2] for row in stale) else 0


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
    parser.add_argument(
        "--head",
        default=None,
        help="commit under test (default: HEAD); one branch only, not --all-open",
    )
    parser.add_argument("--repo", default=None, help="repository root (default: current directory)")
    parser.add_argument(
        "--all-open",
        action="store_true",
        help="sweep the open set from the API instead of checking one branch",
    )
    parser.add_argument(
        "--github-repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/name for --all-open (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="API token for --all-open (default: $GITHUB_TOKEN)",
    )
    args = parser.parse_args(argv)

    # Mutually exclusive rather than ignored: --head names one commit and --repo
    # one checkout, and the sweep reads neither, so honouring either would answer
    # a question the caller did not ask while looking like it had. --repo is the
    # one a caller is most likely to pass by mistake, because the sibling gate
    # scripts spell owner/name --repo: accepted as a path, it leaves the sweep
    # reading $GITHUB_REPOSITORY and reporting on a repository nobody asked about
    # (issue #2569). Every value-bearing flag the sweep does read is passed to
    # _run_sweep below, which is what pins this partition rather than a list.
    if args.all_open and args.head is not None:
        parser.error("--all-open sweeps the open set and reads no local commit; --head names one branch")
    if args.all_open and args.repo is not None:
        parser.error(
            "--all-open sweeps the open set from the API and reads no local checkout; --repo names one. "
            "To name the repository the sweep reads, pass --github-repo owner/name."
        )
    if args.all_open:
        if not args.github_repo:
            parser.error("--all-open needs --github-repo owner/name (or $GITHUB_REPOSITORY)")
        if not args.token:
            parser.error("--all-open needs --token (or $GITHUB_TOKEN)")
        return _run_sweep(args.github_repo, args.base_ref, args.token)

    head = args.head if args.head is not None else "HEAD"
    repo = Path(args.repo) if args.repo is not None else None

    try:
        base = resolve_base_ref(args.base_ref, repo=repo)
        fork_point = merge_base(base, head, repo=repo)
        pr_paths = changed_paths(fork_point, head, repo=repo)
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
