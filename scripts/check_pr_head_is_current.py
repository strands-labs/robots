#!/usr/bin/env python3
"""Report a pull request whose recorded head commit is not its branch's tip.

Why this exists
---------------
A pull request has two answers to "what is the head commit", and they can
disagree for hours. ``pullRequest { headRefOid }`` is the value the pull request
*records*; the tip of the branch in the head repository is the value that
exists. GitHub normally reconciles the two within a second of a push, and when
it does not, nothing on the pull request says so.

Measured on #2508, which sat approved, green and unmergeable for over five
hours::

    headRefOid reported by the API      21ea097e   pushed 07:56:47   8 check suites, green
    tip of yinsong1986:docs/...         271ec912   pushed 13:58:16   0 check suites

Every merge attempt against it refused, by both APIs, with one message::

    mergePullRequest         ->  Head branch is out of date. Review and try the merge again.
    PUT /pulls/2508/merge    ->  409  Head branch is out of date. ...

That message names a gate this repository does not have, which is what makes it
so expensive to act on. The ``default`` ruleset sets
``strict_required_status_checks_policy: false`` and ``main`` carries no classic
protection (``GET /branches/main/protection`` -> ``404 Branch not protected``),
so being behind the base is not a merge requirement here at all. The branch was
not in conflict either: ``git merge-tree --write-tree --messages`` against
``main`` returned a tree and zero conflict messages. Nothing about the *branch*
was stale. The pull request's record was.

Why every other check agrees the pull request is ready
------------------------------------------------------
Each gate AGENTS.md tells you to poll resolves the head through the pull
request's own view of it, so they all answer correctly about a commit that is no
longer the tip, and they all agree with each other:

==========================================  ============================  ==============================
signal                                      value during the lag          true of ``21ea097e``?
==========================================  ============================  ==============================
``reviewDecision``                          ``APPROVED``                  yes
``call-test-lint / Test and Lint``          ``SUCCESS``                   yes
``reviewThreads``                           none unresolved               yes
``check_last_push_approval.py --all-open``  ``satisfied``                 yes
``check_merge_base_overlap.py --all-open``  no finding                    yes
``mergeable`` / ``mergeStateStatus``        ``UNKNOWN``                   the only anomaly
==========================================  ============================  ==============================

So the presentation is a pull request that has cleared every gate and is waiting
for somebody to press merge. That is the same presentation #1905 records for
``pusher-only-approval`` and #1917 for a workflow-touching ``BLOCKED`` read, and
this is a third cause of it. Unlike those two it does not resolve with time:
nothing about waiting reconciles the record.

``UNKNOWN`` is the one field that moves, and it is the weakest signal available.
AGENTS.md already documents ``mergeable`` as lazily computed -- it "reads
``unknown`` first and the settled value second" -- so a single ``UNKNOWN`` is
both expected and benign, and is normally cleared by reading again. Here it
never settles, and "read it once more" is indistinguishable from the documented
benign case for as long as one is willing to keep reading. Five consecutive
reads on #2508 returned ``UNKNOWN``, across both the GraphQL and REST shapes.

What this reads instead
-----------------------
The head repository's own ref, which is the value under suspicion read from the
side that is not suspected::

    query($owner:String!,$name:String!,$ref:String!){
      repository(owner:$owner,name:$name){ ref(qualifiedName:$ref){ target { oid } } }
    }

resolved from ``pullRequest { headRepository { nameWithOwner } headRefName }``.
Reading the head through the pull request cannot work here by construction: that
is the field being checked.

The ``ref`` read is sound and needs no clone. Swept across all 10 open
non-draft pull requests in this repository it agreed with ``git ls-remote`` on
every one, and it named #2508 as the only stale record before the
reconciliation and none afterwards.

Outcomes
--------
``current``
    The record matches the tip. Nothing to say, and the ordinary state.

``stale-head-record``
    The finding. The pull request's gates, its checks and its approval all
    describe a commit that is not the branch tip, and a merge will refuse.

``unresolvable-head``
    The head repository or its branch could not be read -- a deleted fork, a
    deleted branch. Reported as its own outcome rather than folded into a pass,
    so a permissions change or an API shape change cannot quietly turn this
    check into one that always agrees. It is not a finding: a pull request whose
    fork is gone has a different problem with a different remedy.

The remedy is a reopen, not a push
----------------------------------
A close/reopen reconciles the record without touching the branch. On #2508 it
moved ``headRefOid`` to ``271ec912`` and queued the nine check suites that
commit had never had. This is the same remedy as the no-check-suite case in
AGENTS.md step 8 (#1987) and for a closely related reason: ``pull_request``
takes the default types, so ``reopened`` recomputes with no commit, therefore no
push, therefore neither ``dismiss_stale_reviews_on_push`` nor a new last pusher.

Do not push the branch, even though "out of date" reads like a request to
refresh it. On a contributor's branch a push consumes the approval of whoever
owns the pushing token under ``require_last_push_approval``, converting a pull
request one maintainer could merge into one that needs a second approver -- the
state #1035 has been in since 2026-08-01. On #2508 the branch also needed
nothing: the author had already merged the base cleanly.

Read the close/reopen history first, per AGENTS.md step 8, counting ``nodes``
rather than ``totalCount``. #2508 had zero, so a single flip was safe. An
alternating run means something is undoing you and another flip only lengthens
it.

Why there is a sweep mode
-------------------------
The same reason ``check_last_push_approval.py`` has one. A stale record produces
no event -- that *is* the defect, that the push was not registered against the
pull request -- so a workflow driven by ``pull_request`` cannot fire on the
population this is written for. ``--all-open`` is the caller a scheduled health
scan needs. Draft pull requests are excluded: a draft cannot merge whatever its
record says, so a finding on one does not mean what a finding here means.

This reports and does not gate. Its remedy is a reopen by someone with write
access, which is not something a branch clears by pushing, and a check a branch
cannot turn green by doing anything belongs in a report.

Usage
-----
::

    python3 scripts/check_pr_head_is_current.py --repo strands-labs/robots --pr 2508
    python3 scripts/check_pr_head_is_current.py --repo strands-labs/robots --all-open

Exit 1 when at least one evaluated pull request has a stale head record.

Pinned by tests/test_pr_head_is_current.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

# Pagination ceiling for the open-pull-request listing, matching
# check_last_push_approval.py. A repository with more open pull requests than
# this at once has a different problem, and an unbounded loop over a paginated
# endpoint is how a transient API shape change becomes a hang rather than an
# error.
_MAX_PAGES = 20

CURRENT = "current"
STALE_HEAD_RECORD = "stale-head-record"
UNRESOLVABLE_HEAD = "unresolvable-head"

_API = "https://api.github.com/graphql"

# The remedy text, shared by the single-pull-request report and the sweep so the
# two cannot drift into describing different remedies for one rule.
WHAT_CLEARS_THIS: tuple[str, ...] = (
    "",
    "### What clears this",
    "",
    "Close and reopen the pull request. That reconciles the record without",
    "touching the branch: `reopened` recomputes the same head, so there is no",
    "commit, no push, and therefore neither `dismiss_stale_reviews_on_push`",
    "nor a new last pusher.",
    "",
    "Read the close/reopen history first and count `nodes`, not `totalCount`.",
    "An alternating run means something is undoing you and a further flip only",
    "lengthens it.",
    "",
    "Do **not** push the branch to refresh it, however much `Head branch is out",
    "of date` reads like a request to. There is no staleness gate on this",
    "repository to satisfy -- the `default` ruleset sets",
    "`strict_required_status_checks_policy: false` and `main` has no classic",
    "protection -- and on a contributor's branch a push consumes the approval of",
    "whoever owns the pushing token under `require_last_push_approval`, which",
    "turns a pull request one maintainer could merge into one needing a second",
    "approver.",
    "",
    "Expect `reviewDecision` to move to `REVIEW_REQUIRED` once the record",
    "catches up, and read that as bookkeeping rather than lost review: the",
    "approval was already attributed to a commit that was not the tip. Check",
    "whether the new head changed the pull request's own diff before asking for",
    "a re-review -- a clean base merge leaves it byte-identical.",
)


# Named rather than inlined: as a list element it would be two adjacent string
# parts, which a reader cannot tell from a dropped comma.
_UNRESOLVABLE_PREAMBLE = (
    "Unresolvable (named rather than counted as clean, because a head this"
    + " check could not read is not a head it cleared):"
)


@dataclass(frozen=True)
class Verdict:
    """The outcome and the two commits it was computed from."""

    outcome: str
    recorded: str | None
    tip: str | None

    @property
    def is_finding(self) -> bool:
        return self.outcome == STALE_HEAD_RECORD

    @property
    def summary(self) -> str:
        if self.outcome == CURRENT:
            return f"The recorded head {_short(self.recorded)} is the branch tip. Nothing to report."
        if self.outcome == UNRESOLVABLE_HEAD:
            return (
                "Could not read the head repository's branch, so the recorded head "
                f"{_short(self.recorded)} could not be compared against anything. Not treated "
                "as a finding: a deleted fork or branch is a different problem."
            )
        return (
            f"The pull request records its head as {_short(self.recorded)}, but the branch tip "
            f"is {_short(self.tip)}. Its checks, its approval and every merge gate describe "
            f'{_short(self.recorded)}, and a merge will refuse with "Head branch is out of '
            'date" naming a staleness gate this repository does not have. Reopen the pull '
            "request to reconcile the record; do not push the branch."
        )


@dataclass(frozen=True)
class Row:
    """One evaluated pull request, for the sweep report."""

    pr: int
    verdict: Verdict
    head_repo: str
    head_ref: str
    mergeable: str
    merge_state_status: str
    review_decision: str


def _short(oid: str | None) -> str:
    """Render a commit for a human without pretending an absent one is a commit."""
    if not oid:
        return "(unknown)"
    return f"`{oid[:8]}`"


def classify(recorded: str | None, tip: str | None) -> Verdict:
    """Decide whether a pull request's recorded head is its branch's tip.

    ``tip`` of ``None`` means the head repository's ref could not be read, and
    ``recorded`` of ``None`` means the pull request did not report a head at
    all. Either way there is nothing to compare, which is reported as its own
    outcome rather than as a pass -- an unreadable head is not a head that
    matches, and folding the two together is how this check would become a
    no-op that always agrees.
    """
    if not recorded or not tip:
        return Verdict(UNRESOLVABLE_HEAD, recorded, tip)
    if recorded == tip:
        return Verdict(CURRENT, recorded, tip)
    return Verdict(STALE_HEAD_RECORD, recorded, tip)


def _graphql(query: str, variables: dict[str, object], token: str) -> dict:
    request = urllib.request.Request(
        _API,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "User-Agent": "strands-robots-check-pr-head-is-current",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed API host
        payload = json.load(response)
    if payload.get("errors"):
        raise ValueError(f"GraphQL errors: {payload['errors']}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("GraphQL response carried no data object")
    return data


_PR_FIELDS = """
  number isDraft headRefName headRefOid
  headRepository { nameWithOwner }
  mergeable mergeStateStatus reviewDecision
"""

_FIELDS_PLACEHOLDER = "__PR_FIELDS__"


def _with_pr_fields(query: str) -> str:
    """Substitute the shared field list into a query document.

    Neither an f-string nor ``%`` formatting: a GraphQL document is almost all
    braces, so both spellings would require escaping every one of them and the
    query would stop being copy-pasteable into the API explorer. The two shapes
    share one field list so a field added for the sweep cannot go missing from
    the single-pull-request path.
    """
    return query.replace(_FIELDS_PLACEHOLDER, _PR_FIELDS)


_ONE_PR = _with_pr_fields(
    """
query($owner:String!,$name:String!,$number:Int!){
  repository(owner:$owner,name:$name){
    pullRequest(number:$number){ __PR_FIELDS__ }
  }
}
"""
)

_OPEN_PRS = _with_pr_fields(
    """
query($owner:String!,$name:String!,$cursor:String){
  repository(owner:$owner,name:$name){
    pullRequests(states: OPEN, first: 50, after: $cursor){
      pageInfo { hasNextPage endCursor }
      nodes { __PR_FIELDS__ }
    }
  }
}
"""
)

_REF_TIP = """
query($owner:String!,$name:String!,$ref:String!){
  repository(owner:$owner,name:$name){
    ref(qualifiedName:$ref){ target { oid } }
  }
}
"""


def _split_repo(repo: str) -> tuple[str, str]:
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"--repo must be owner/name, got {repo!r}")
    owner, name = repo.split("/")
    return owner, name


def branch_tip(head_repo: str, head_ref: str, token: str) -> str | None:
    """Return the tip of ``head_ref`` in ``head_repo``, or ``None`` if unreadable.

    Deliberately reads the head *repository*, not the pull request. The pull
    request's own answer is the value being checked, so consulting it here would
    compare a field against itself and agree every time.
    """
    try:
        owner, name = _split_repo(head_repo)
    except ValueError:
        return None
    try:
        data = _graphql(
            _REF_TIP,
            {"owner": owner, "name": name, "ref": f"refs/heads/{head_ref}"},
            token,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return None
    repository = data.get("repository") or {}
    ref = repository.get("ref") or {}
    target = ref.get("target") or {}
    oid = target.get("oid")
    return oid if isinstance(oid, str) else None


def evaluate(node: dict, token: str) -> Row:
    """Evaluate one pull request node from either query shape."""
    head_repository = node.get("headRepository") or {}
    head_repo = head_repository.get("nameWithOwner") or ""
    head_ref = node.get("headRefName") or ""
    recorded = node.get("headRefOid")
    tip = branch_tip(head_repo, head_ref, token) if head_repo and head_ref else None
    return Row(
        pr=int(node["number"]),
        verdict=classify(recorded, tip),
        head_repo=head_repo or "(unknown)",
        head_ref=head_ref or "(unknown)",
        mergeable=str(node.get("mergeable")),
        merge_state_status=str(node.get("mergeStateStatus")),
        review_decision=str(node.get("reviewDecision")),
    )


def fetch_one(repo: str, number: int, token: str) -> dict:
    owner, name = _split_repo(repo)
    data = _graphql(_ONE_PR, {"owner": owner, "name": name, "number": number}, token)
    repository = data.get("repository") or {}
    node = repository.get("pullRequest")
    if not isinstance(node, dict):
        raise ValueError(f"{repo}#{number} did not resolve to a pull request")
    return node


def fetch_open(repo: str, token: str) -> list[dict]:
    owner, name = _split_repo(repo)
    nodes: list[dict] = []
    cursor: str | None = None
    for _ in range(_MAX_PAGES):
        data = _graphql(_OPEN_PRS, {"owner": owner, "name": name, "cursor": cursor}, token)
        connection = (data.get("repository") or {}).get("pullRequests") or {}
        nodes.extend(connection.get("nodes") or [])
        page = connection.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            break
        cursor = page.get("endCursor")
    return nodes


def render_sweep(repo: str, rows: Sequence[Row], skipped: Sequence[int]) -> str:
    findings = [r for r in rows if r.verdict.is_finding]
    unresolvable = [r for r in rows if r.verdict.outcome == UNRESOLVABLE_HEAD]
    lines = [
        "## Pull-request head record sweep",
        "",
        f"Evaluated {len(rows)} open non-draft pull request(s) in {repo}.",
        "",
    ]
    if findings:
        named = ", ".join(f"#{r.pr}" for r in findings)
        lines += [
            f"**{len(findings)} record(s) a commit that is not the branch tip:** {named}.",
            "",
            "Every merge gate on these describes the recorded commit, so all of them",
            "read ready while the merge refuses. This does not resolve with time.",
            "",
        ]
    else:
        lines += ["Every recorded head is its branch's tip.", ""]

    lines += [
        "| pull request | recorded head | branch tip | outcome | mergeable | mergeStateStatus | review |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| #{row.pr} | {_short(row.verdict.recorded)} | {_short(row.verdict.tip)} | "
            f"{row.verdict.outcome} | {row.mergeable} | {row.merge_state_status} | "
            f"{row.review_decision} |"
        )
    if unresolvable:
        lines += ["", _UNRESOLVABLE_PREAMBLE, ""]
        lines += [f"- #{r.pr}: `{r.head_repo}` `{r.head_ref}` could not be read" for r in unresolvable]
    if skipped:
        lines += ["", "Skipped: " + ", ".join(f"#{n}" for n in skipped) + "."]
    if findings:
        lines += list(WHAT_CLEARS_THIS)
    return "\n".join(lines)


def render_one(repo: str, row: Row) -> str:
    lines = [
        "## Pull-request head record",
        "",
        f"Outcome: **{row.verdict.outcome}**",
        "",
        row.verdict.summary,
        "",
        "| field | value |",
        "|---|---|",
        f"| pull request | {repo}#{row.pr} |",
        f"| head repository | `{row.head_repo}` |",
        f"| head branch | `{row.head_ref}` |",
        f"| recorded head | {_short(row.verdict.recorded)} |",
        f"| branch tip | {_short(row.verdict.tip)} |",
        f"| mergeable | {row.mergeable} |",
        f"| mergeStateStatus | {row.merge_state_status} |",
        f"| reviewDecision | {row.review_decision} |",
    ]
    if row.verdict.is_finding:
        lines += list(WHAT_CLEARS_THIS)
    return "\n".join(lines)


def _publish(report: str) -> None:
    print(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")


def _annotate(repo: str, row: Row) -> None:
    print(f"::warning title=Recorded head is not the branch tip::{repo}#{row.pr}: {row.verdict.summary}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", ""))
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument(
        "--all-open",
        action="store_true",
        help="Evaluate every open non-draft pull request instead of one.",
    )
    args = parser.parse_args(argv)

    if not args.repo:
        print("check_pr_head_is_current: --repo is required", file=sys.stderr)
        return 0
    if not args.token:
        print("check_pr_head_is_current: no token; set --token or GITHUB_TOKEN", file=sys.stderr)
        return 0

    if args.all_open:
        try:
            nodes = fetch_open(args.repo, args.token)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            # Listing is the one lookup with no partial result to report, so it
            # fails open rather than failing the caller's run.
            print(f"check_pr_head_is_current: could not list open pull requests: {exc}", file=sys.stderr)
            return 0
        rows: list[Row] = []
        skipped: list[int] = []
        for node in nodes:
            if node.get("isDraft"):
                continue
            try:
                rows.append(evaluate(node, args.token))
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as exc:
                # One unreadable pull request must not suppress a finding on the
                # others, so it is named and skipped rather than fatal.
                number = node.get("number")
                print(f"check_pr_head_is_current: skipped #{number}: {exc}", file=sys.stderr)
                if isinstance(number, int):
                    skipped.append(number)
        rows.sort(key=lambda r: r.pr)
        _publish(render_sweep(args.repo, rows, skipped))
        findings = [r for r in rows if r.verdict.is_finding]
        for row in findings:
            _annotate(args.repo, row)
        return 1 if findings else 0

    if not args.pr:
        print("check_pr_head_is_current: --pr or --all-open is required", file=sys.stderr)
        return 0
    try:
        row = evaluate(fetch_one(args.repo, int(args.pr), args.token), args.token)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        print(f"check_pr_head_is_current: could not evaluate {args.repo}#{args.pr}: {exc}", file=sys.stderr)
        return 0
    _publish(render_one(args.repo, row))
    if row.verdict.is_finding:
        _annotate(args.repo, row)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
