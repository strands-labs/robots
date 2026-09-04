#!/usr/bin/env python3
"""Report a pull request whose only approvals come from the account that pushed its head.

Why this exists
---------------
The active ``default`` BRANCH ruleset carries::

    requiredApprovingReviewCount: 1
    requireLastPushApproval:      true    <-- the relevant one
    dismissStaleReviewsOnPush:    true

``require_last_push_approval`` means the most recent push must be approved by
**someone other than whoever pushed it**. So when the only account that has
approved a pull request is also the account that pushed its head commit, that
approval stops counting, ``reviewDecision`` reads ``REVIEW_REQUIRED``, and
``mergeStateStatus`` sits at ``BLOCKED`` for as long as nobody else reviews.
Further approvals from that same account cannot clear it.

That state is indistinguishable, in every field a status sweep reads, from a
pull request that simply has not been reviewed yet -- both read
``REVIEW_REQUIRED`` / ``BLOCKED``. The two need opposite actions: one needs the
existing reviewer to get to it, the other needs a *different* reviewer and will
never merge without one. Conflating them is not hypothetical; it stood in eight
consecutive scheduled scan summaries as "reviewer bandwidth is the sole
constraint" while #1722 and #1035 sat permanently unmergeable. See issue #1905
and the "PR Workflow" section of AGENTS.md.

Why the pusher is read from a workflow run
------------------------------------------
The pusher appears in none of the fields a reviewer would naturally check, and
commit metadata answers the question wrongly in both directions. Measured on
four pull requests in this repository:

===========  ==================  ================  =====================
pull request ``triggering_actor``  approved by       ``commit.author.login``
===========  ==================  ================  =====================
#1894        yinsong1986         cagataycali       yinsong1986
#1920        cagataycali         yinsong1986       ``None``
#1722        cagataycali         cagataycali       cagataycali
#1035        cagataycali         cagataycali       cagataycali
===========  ==================  ================  =====================

#1920 and #1722 share a pusher and differ only in whether the approver is a
different account; #1920 merged and #1722 has been blocked since 2026-08-01, so
that is the variable. And #1920's head was committed under the
``strands-robots`` git identity, which has no linked GitHub account: its
``commit.author.login`` is ``None``, so commit metadata does not merely mislead
there, it declines to answer while ``triggering_actor`` answers correctly.
#1722's metadata names the pusher and #1920's does not, for the same verdict --
which is why this reads the workflow run::

    GET /repos/{owner}/{repo}/actions/runs?head_sha=<head>  ->  triggering_actor

What this reports, and what it deliberately does not
----------------------------------------------------
Three outcomes, distinguished because they ask for three different things:

``satisfied``
    At least one current approval comes from an account other than the pusher.
    Nothing to say.

``awaiting-first-review``
    No current approval at all. This is the ordinary case and is *not* a
    finding: the pull request is waiting on a reviewer, which is true and
    already visible. Reported as passing so that the check's red state means
    one specific thing.

``pusher-only-approval``
    Every current approval belongs to the pusher. The finding. The remedy is a
    second approver, or the branch author re-pushing the head so the existing
    approval counts again -- which costs a re-approval round, because
    ``dismiss_stale_reviews_on_push`` is also set.

A "current" approval follows the same rule ``reviewDecision`` uses: per author,
their most recent review that expresses a position. ``COMMENTED`` expresses
none and is skipped, so a reviewer who approves and later comments is still an
approver; ``DISMISSED`` and ``CHANGES_REQUESTED`` supersede an earlier approval.

This check does not gate a merge and is not in the required set -- whether a
finding blocks is a branch-protection decision, not a property of this file. It
also cannot be self-cleared by the pull request author alone, which is exactly
why it reports rather than blocks: a gate whose remedy is "find another human"
would otherwise sit red on a branch that has done nothing wrong.

Why there is a sweep mode
-------------------------
The workflow that runs this is driven by ``pull_request`` and
``pull_request_review`` events, so it can only ever evaluate a pull request
that has had one **since the workflow landed**. The population the check was
written for is precisely the population that has not.

Measured on #1035: its head was pushed 2026-08-01T08:00:37Z and the approval
submitted 51 minutes later, both before the workflow existed on 2026-08-04, so
no qualifying event has fired on it since and ``Detect an approval the last
pusher cannot supply`` is absent from the 11 check runs on that head. Nothing
about the verdict was ever wrong -- invoked directly against the same pull
request the classifier answers immediately::

    python3 scripts/check_last_push_approval.py --pr 1035
    -> Outcome: pusher-only-approval, pushed by cagataycali, exit 1

So the gap was never in the reasoning; it was that on a standing pull request
nothing asks for it. Issue #1905 attributes the silence to this workflow's
base-branch guard instead. That is not the cause and the distinction matters,
because it points at a different fix: the guard checks out the **base**, ``main``
carries the script, so the guard passes and would run it. What is missing is a
caller.

``--all-open`` is that caller. It classifies every open pull request on demand,
which is what a scheduled status scan needs in order to tell "waiting on a
reviewer" from "cannot merge on any further review by this account" -- the two
states this file exists to separate, and the ones that stood conflated in eight
consecutive scan summaries. Draft pull requests are excluded: a draft cannot
merge whatever its approvals say, so a finding on one does not mean what a
finding here means.

A per-pull-request lookup failure inside a sweep is reported and skipped rather
than allowed to abort the run, because one rate-limited pull request must not
suppress a finding on the others. Skipped numbers are named in the report for
the same reason the outcomes are: an unevaluated pull request that says nothing
is the failure mode this whole file is written against.

Usage
-----
``--repo``   ``owner/name`` (default: ``$GITHUB_REPOSITORY``).
``--pr``     pull request number (default: ``$PR_NUMBER``).
``--all-open``
             Sweep every open non-draft pull request instead of one. Mutually
             exclusive with ``--pr``.
``--head``   head commit SHA (default: ``$HEAD_SHA``; falls back to the pull
             request's own ``head.sha``).
``--token``  API token (default: ``$GITHUB_TOKEN``). Needs ``actions: read``
             for the run lookup and ``pull-requests: read`` for the reviews.

Exit status is ``1`` for ``pusher-only-approval``, else ``0``. A lookup that
cannot determine the pusher exits ``0`` and says so: an unknown pusher is not
evidence of a deadlock, and this check refusing to guess is the whole point of
reading ``triggering_actor`` rather than the commit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

API_ROOT = "https://api.github.com"

# Review states that express a position on the change. A COMMENTED review does
# not, which is why it cannot retract an earlier approval.
POSITION_STATES = frozenset({"APPROVED", "CHANGES_REQUESTED", "DISMISSED"})

# Workflow-run events whose ``triggering_actor`` is the account that put the
# commit there. Every other event names whoever caused *that* event instead, and
# filtering on this set is load-bearing rather than defensive: this check's own
# ``pull_request_review`` trigger produces a run on the same head sha attributed
# to the **reviewer**, newer than the push's own runs. Reading the newest run
# unfiltered therefore named the approver as the pusher and reported
# ``pusher-only-approval`` for every approved pull request -- observed on #1921,
# this check's own, where GitHub read `APPROVED`/`UNSTABLE` and the check
# disagreed. Pinned by
# tests/test_last_push_approval.py::test_a_review_triggered_run_does_not_name_the_pusher
PUSH_ATTRIBUTING_EVENTS = frozenset({"push", "pull_request"})

# Pagination ceiling for the open-pull-request listing. A repository with more
# than this many open pull requests at once has a different problem, and an
# unbounded loop over a paginated endpoint is how a transient API shape change
# becomes a hang rather than an error.
_MAX_PAGES = 20

SATISFIED = "satisfied"
AWAITING_FIRST_REVIEW = "awaiting-first-review"
PUSHER_ONLY_APPROVAL = "pusher-only-approval"
UNKNOWN_PUSHER = "unknown-pusher"


# The remedy text, shared by the single-pull-request report and the sweep so
# the two cannot drift into describing different remedies for one rule.
WHAT_CLEARS_THIS: tuple[str, ...] = (
    "",
    "### What clears this",
    "",
    "1. A second reviewer approves. Preserves the guarantee the rule exists",
    "   to provide, and is the cheapest of the three.",
    "2. The branch author pushes the head commit, which makes the existing",
    "   approval count again. `dismiss_stale_reviews_on_push` is also set, so",
    "   this costs a re-approval round -- but one that then counts.",
    "3. Admin bypass. Not recommended: the rule exists to stop one account",
    "   both writing and approving a change, so bypassing it to merge code",
    "   that account pushed defeats the control rather than working around a",
    "   technicality.",
    "",
    "Avoiding it next time: whatever puts the head commit there becomes the last",
    "push, and consumes the approval of the account it is attributed to. That is",
    '`git push`, and it is equally the **"Update branch" button** on the pull',
    "request page: one click, no local checkout, no token the operator handles,",
    "and the same new last pusher. The button is not an exception because it is",
    "one click. A base refresh on a contributor's branch is theirs to make, so",
    "prefer leaving the change as review feedback for the author, or land it as a",
    "separate pull request against the base branch.",
)


@dataclass(frozen=True)
class Verdict:
    """The outcome, its one-line reason, and the accounts it was computed from."""

    outcome: str
    pusher: str | None
    approvers: tuple[str, ...] = ()

    @property
    def is_finding(self) -> bool:
        return self.outcome == PUSHER_ONLY_APPROVAL

    @property
    def summary(self) -> str:
        if self.outcome == SATISFIED:
            others = [a for a in self.approvers if a != self.pusher]
            return (
                f"Approved by {_join(others)}, who did not push the head "
                f"(pushed by {self.pusher}). require_last_push_approval is satisfied."
            )
        if self.outcome == AWAITING_FIRST_REVIEW:
            return (
                f"No current approval yet; head pushed by {self.pusher}. "
                "Waiting on a first review, which is the ordinary state."
            )
        if self.outcome == UNKNOWN_PUSHER:
            return (
                "Could not determine who pushed the head commit: no workflow run "
                "reports a triggering_actor for it. Not treated as a finding, "
                "because commit metadata is not a sound substitute."
            )
        return (
            f"The only current approval is from {_join(self.approvers)}, which is "
            f"also the account that pushed the head commit. Under "
            f"require_last_push_approval that approval does not count, so this "
            f"pull request needs a second approver and cannot be merged on the "
            f"strength of further reviews from {self.pusher}."
        )


@dataclass
class Review:
    """One review, reduced to the fields the decision depends on."""

    author: str
    state: str
    submitted_at: str = ""
    _order: int = field(default=0, compare=False)


def _join(names: Sequence[str]) -> str:
    names = list(names)
    if not names:
        return "nobody"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def current_approvers(reviews: Iterable[Review]) -> tuple[str, ...]:
    """Return the accounts whose latest position on the change is an approval.

    Mirrors how ``reviewDecision`` is computed: a reviewer's standing is their
    most recent review that expresses a position, so ``COMMENTED`` is skipped
    entirely rather than treated as a retraction. Ordering is by the review
    sequence as returned by the API, which is chronological, with
    ``submitted_at`` only as a tiebreak -- two reviews can share a timestamp to
    the second, and the later one in the list is the later one.
    """
    latest: dict[str, Review] = {}
    for order, review in enumerate(reviews):
        state = review.state.upper()
        if state not in POSITION_STATES:
            continue
        keyed = Review(review.author, state, review.submitted_at, order)
        held = latest.get(review.author)
        if held is None or (keyed.submitted_at, keyed._order) >= (held.submitted_at, held._order):
            latest[review.author] = keyed
    return tuple(sorted(a for a, r in latest.items() if r.state == "APPROVED"))


def classify(pusher: str | None, reviews: Iterable[Review]) -> Verdict:
    """Decide which of the three states a pull request is in.

    ``pusher`` of ``None`` means the lookup could not attribute the push. That
    is reported as its own outcome rather than folded into a pass, so a silent
    API change cannot quietly turn this check into a no-op that always agrees.
    """
    approvers = current_approvers(reviews)
    if pusher is None:
        return Verdict(UNKNOWN_PUSHER, None, approvers)
    if not approvers:
        return Verdict(AWAITING_FIRST_REVIEW, pusher, approvers)
    if any(a != pusher for a in approvers):
        return Verdict(SATISFIED, pusher, approvers)
    return Verdict(PUSHER_ONLY_APPROVAL, pusher, approvers)


def _get(url: str, token: str) -> object:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "strands-robots-check-last-push-approval",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed API host
        return json.load(response)


def resolve_pusher(repo: str, head_sha: str, token: str) -> str | None:
    """Return the account GitHub attributes the head push to, or ``None``.

    Reads ``triggering_actor`` from the workflow runs for the commit, counting
    only runs whose event is one a push produces (``PUSH_ATTRIBUTING_EVENTS``).
    A run started by anything else -- a review, a comment, a manual dispatch --
    is attributed to whoever did *that*, not to the pusher. When several
    qualifying runs exist they normally agree; when they do not, the most
    recently created wins, because a re-push under a different account produces
    newer runs than the ones the previous push left behind.

    Returns ``None`` when no qualifying run exists, which the caller reports as
    ``unknown-pusher`` rather than falling back to the commit metadata.
    """
    payload = _get(f"{API_ROOT}/repos/{repo}/actions/runs?head_sha={head_sha}&per_page=100", token)
    runs = payload.get("workflow_runs", []) if isinstance(payload, dict) else []
    dated: list[tuple[str, str]] = []
    for run in runs:
        if run.get("event") not in PUSH_ATTRIBUTING_EVENTS:
            continue
        actor = (run.get("triggering_actor") or {}).get("login")
        if actor:
            dated.append((run.get("created_at") or "", actor))
    if not dated:
        return None
    return max(dated, key=lambda pair: pair[0])[1]


def resolve_reviews(repo: str, pr: int, token: str) -> list[Review]:
    """Return every review on the pull request, in the order the API lists them."""
    payload = _get(f"{API_ROOT}/repos/{repo}/pulls/{pr}/reviews?per_page=100", token)
    rows = payload if isinstance(payload, list) else []
    return [
        Review(
            author=(row.get("user") or {}).get("login") or "",
            state=row.get("state") or "",
            submitted_at=row.get("submitted_at") or "",
        )
        for row in rows
    ]


def resolve_head_sha(repo: str, pr: int, token: str) -> str:
    payload = _get(f"{API_ROOT}/repos/{repo}/pulls/{pr}", token)
    head = (payload.get("head") or {}) if isinstance(payload, dict) else {}
    return head.get("sha") or ""


@dataclass(frozen=True)
class SweepRow:
    """One open pull request and the verdict computed for it."""

    pr: int
    head_sha: str
    verdict: Verdict


def resolve_open_pull_requests(repo: str, token: str) -> list[tuple[int, str]]:
    """Return ``(number, head_sha)`` for every open non-draft pull request.

    Sorted by number so the sweep report is stable between runs and a diff of
    two reports shows changed verdicts rather than reordered rows.
    """
    found: list[tuple[int, str]] = []
    for page in range(1, _MAX_PAGES + 1):
        payload = _get(f"{API_ROOT}/repos/{repo}/pulls?state=open&per_page=100&page={page}", token)
        rows = payload if isinstance(payload, list) else []
        for row in rows:
            if row.get("draft"):
                continue
            number = row.get("number")
            head = ((row.get("head") or {}).get("sha")) or ""
            if isinstance(number, int) and head:
                found.append((number, head))
        if len(rows) < 100:
            break
    return sorted(found)


def sweep(repo: str, token: str) -> tuple[list[SweepRow], list[int]]:
    """Classify every open non-draft pull request.

    Returns the rows it could evaluate and the numbers it could not. A failure
    on one pull request is skipped rather than raised: the sweep exists to
    surface findings across the standing population, and one unreachable pull
    request must not take the rest of the report with it.
    """
    rows: list[SweepRow] = []
    skipped: list[int] = []
    for pr, head_sha in resolve_open_pull_requests(repo, token):
        try:
            pusher = resolve_pusher(repo, head_sha, token)
            reviews = resolve_reviews(repo, pr, token)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            print(f"check_last_push_approval: {repo}#{pr} lookup failed, not evaluated: {exc}", file=sys.stderr)
            skipped.append(pr)
            continue
        rows.append(SweepRow(pr, head_sha, classify(pusher, reviews)))
    return rows, skipped


def render_sweep(rows: Sequence[SweepRow], skipped: Sequence[int], repo: str) -> str:
    """Render one table for the whole sweep, findings named up front."""
    findings = [row for row in rows if row.verdict.is_finding]
    lines = [
        "## Last-push approval sweep",
        "",
        f"Evaluated {len(rows)} open non-draft pull request(s) in {repo}.",
        "",
    ]
    if findings:
        named = ", ".join(f"#{row.pr}" for row in findings)
        lines += [
            f"**{len(findings)} needs an approver who did not push the head:** {named}.",
            "",
            "Each is approved, and blocked anyway. No further review by the pushing",
            "account can clear it, so this does not resolve with reviewer time.",
            "",
        ]
    else:
        lines += ["No pull request is held by an approval its pusher supplied.", ""]
    lines += [
        "| pull request | outcome | pushed by | current approvals |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| #{row.pr} | {row.verdict.outcome} | {row.verdict.pusher or '(undetermined)'} "
            f"| {_join(row.verdict.approvers)} |"
        )
    if skipped:
        lines += [
            "",
            "Not evaluated (lookup failed): " + ", ".join(f"#{pr}" for pr in skipped) + ".",
            "A pull request this run could not read is named rather than omitted, so a",
            "silent gap in coverage cannot read as a clean sweep.",
        ]
    if findings:
        lines += list(WHAT_CLEARS_THIS)
    return "\n".join(lines)


def render(verdict: Verdict, repo: str, pr: int, head_sha: str) -> str:
    lines = [
        "## Last-push approval",
        "",
        f"Outcome: **{verdict.outcome}**",
        "",
        verdict.summary,
        "",
        "| field | value |",
        "|---|---|",
        f"| pull request | {repo}#{pr} |",
        f"| head | `{head_sha}` |",
        f"| pushed by (`triggering_actor`) | {verdict.pusher or '(undetermined)'} |",
        f"| current approvals | {_join(verdict.approvers)} |",
    ]
    if verdict.is_finding:
        lines += [
            *WHAT_CLEARS_THIS,
        ]
    return "\n".join(lines)


def _emit(report: str) -> None:
    """Print the report, and append it to the step summary when there is one."""
    print(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")


def _run_sweep(repo: str, token: str) -> int:
    """Sweep the open pull requests, reporting every finding. Exit 1 if any."""
    try:
        rows, skipped = sweep(repo, token)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        # Listing the pull requests is the one lookup with no partial result to
        # report, so it follows the same rule as a single-pull-request failure.
        print(f"check_last_push_approval: could not list open pull requests: {exc}", file=sys.stderr)
        return 0

    _emit(render_sweep(rows, skipped, repo))
    findings = [row for row in rows if row.verdict.is_finding]
    for row in findings:
        print(f"::warning title=Needs an approver who did not push the head::{repo}#{row.pr}: {row.verdict.summary}")
    return 1 if findings else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", ""))
    parser.add_argument("--head", default=os.environ.get("HEAD_SHA", ""))
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument(
        "--all-open",
        action="store_true",
        help="Sweep every open non-draft pull request instead of one.",
    )
    args = parser.parse_args(argv)

    if not args.token:
        parser.error("--token is required (or set GITHUB_TOKEN)")
    if not args.repo:
        parser.error("--repo is required (or set GITHUB_REPOSITORY)")
    # Refused rather than resolved in either direction: silently ignoring --pr
    # would report on pull requests the caller did not ask about, and silently
    # ignoring --all-open would report on one when a sweep was wanted. Both
    # read as a successful run of the other thing.
    if args.all_open and args.pr:
        parser.error("--all-open and --pr are mutually exclusive")
    if not args.all_open and not args.pr:
        parser.error("--pr is required (or set PR_NUMBER), or pass --all-open")

    if args.all_open:
        return _run_sweep(args.repo, args.token)

    pr = int(args.pr)
    try:
        head_sha = args.head or resolve_head_sha(args.repo, pr, args.token)
        pusher = resolve_pusher(args.repo, head_sha, args.token) if head_sha else None
        reviews = resolve_reviews(args.repo, pr, args.token)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        # A lookup failure is not a finding. Say so on stderr and pass, rather
        # than accusing a branch of a deadlock this run could not observe.
        print(f"check_last_push_approval: lookup failed, reporting nothing: {exc}", file=sys.stderr)
        return 0

    verdict = classify(pusher, reviews)
    report = render(verdict, args.repo, pr, head_sha)
    _emit(report)

    if verdict.is_finding:
        print(f"::warning title=Needs an approver who did not push the head::{verdict.summary}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
