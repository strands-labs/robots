#!/usr/bin/env python3
"""Name the branch-ruleset rule a blocked pull request has left unsatisfied.

Why this exists
---------------
``mergeStateStatus: BLOCKED`` is one word for at least six unrelated
situations, and it names none of them. The rule that is actually unsatisfied
decides who owes the next action, so collapsing them loses the only fact a
triage pass needs. Three cases measured in this repository on 2026-08-21, all
reading ``blocked``, all needing different people:

===========  ==========================================  ====================
pull request what was actually unsatisfied               who could clear it
===========  ==========================================  ====================
#2566        one unresolved review thread, whose fix     **the author**
             had already landed and which the reviewer
             had approved past
#2574        nothing at all -- the ``blocked`` was a      anyone, by retrying
             stale computation, and the merge succeeded
             on the first attempt with no state change
#2497        no approving review yet                     any reviewer
===========  ==========================================  ====================

#2566 and #2574 both sat idle after approval (31 and 45 minutes) because a
scheduled author-side pass read ``APPROVED`` plus green checks plus ``BLOCKED``
and concluded the remaining obligation was somebody else's. It was not. That
conclusion is the same one issue #1905 documents under the heading "it presents
as reviewer bandwidth", reached through a different door: #1905's door is the
``require_last_push_approval`` topology, and these two are not that, so nothing
built for #1905 detects them.

The missing read is cheap. The ruleset is published, every input it refers to
is queryable, and no new policy or gate is needed to say which of its rules is
unmet.

Why the ruleset is read rather than inferred
--------------------------------------------
The rules in force are a property of the branch, not of this file, so hardcoding
them would drift the moment one is changed in the repository settings. Read from::

    GET /repos/{owner}/{repo}/rules/branches/{base_ref}

which on this repository's ``main`` returns, among others::

    pull_request.required_approving_review_count:    1
    pull_request.required_review_thread_resolution:  true
    pull_request.require_last_push_approval:         true
    pull_request.require_extra_approval_for_unattributed_changes: true
    required_status_checks.required_status_checks:   ["call-test-lint / Test and Lint"]

Reading it also bounds the report honestly: a rule the branch does not carry is
never named as a blocker, and a rule this file cannot evaluate is listed as
such rather than silently passed. ``require_code_owner_review`` is the standing
example of the first -- it is set on this branch and vacuous, because the
repository has no CODEOWNERS file.

What this reports, and what it deliberately does not
----------------------------------------------------
Each blocker carries the rule that produced it and the party who can clear it.
The distinction that matters is not blocked-versus-clean, it is *whose move it
is*, so the outcomes group that way rather than by severity:

``merge-conflict``, ``draft``, ``required-check-failing``, ``unresolved-threads``
    The **author** owes the next action. These are the misfiled class: each one
    is indistinguishable from waiting on a reviewer in every field a status
    sweep reads, and each is clearable without anyone else.

``missing-approval``
    A **reviewer** owes the next action. The ordinary, honest state, and
    reported as passing for the same reason the sibling check reports
    ``awaiting-first-review`` as passing: if the common case is a finding, the
    finding means nothing.

``pusher-only-approval``
    A reviewer **other than the pusher** owes it. Not re-derived here; see the
    sibling note below.

``required-check-pending``
    Nobody. The answer is not in yet.

``required-check-absent``
    A **maintainer**, by authorising or re-running the workflow. A fork pull
    request whose runs are held at ``action_required`` reports ``completed``
    with a null-ish rollup, which reads identically to "never ran".

``no-unsatisfied-rule``
    Every rule the branch carries is satisfied and it still reads blocked. This
    is the #2574 case and the one most worth saying out loud, because the
    remedy is to attempt the merge: ``mergePullRequest`` refuses with ``Pull
    Request is not mergeable``, which names nothing, while ``PUT
    /repos/{owner}/{repo}/pulls/{n}/merge`` either succeeds or names the
    unsatisfied requirement. A merge attempt is cheap and self-verifying.

    One caveat is deliberately printed with it rather than left to be
    rediscovered: an Actions installation token reads ``BLOCKED`` on any pull
    request that touches ``.github/workflows/**`` regardless of the rules, so
    on such a pull request this outcome may be an artifact of the token that
    asked. Re-read with a personal access token before concluding the state is
    stale. See the "PR Workflow" section of AGENTS.md.

This does not merge anything, does not gate anything, and is not in the
required set. It answers one question and exits.

Why it composes the sibling rather than re-deriving it
-----------------------------------------------------
``require_last_push_approval`` already has an owner:
``scripts/check_last_push_approval.py``, which carries the measured evidence
for it and the semantics of what a "current" approval is -- per author, their
most recent review that expresses a position, so ``COMMENTED`` is not a
retraction. Those semantics are subtle enough that a second copy would drift,
and this file needs the same primitive to count approvals at all. So it imports
``current_approvers`` and the two resolvers from that module instead of
restating them, which is the repository's own rule for a guard that acquires a
second caller (AGENTS.md, Key Conventions 11). ``scripts`` is not a package, so
the import goes by path.

Usage
-----
::

    python3 scripts/check_merge_blockers.py --repo strands-labs/robots --pr 2574
    python3 scripts/check_merge_blockers.py --repo strands-labs/robots --all-open

Exit status follows the sibling's contract, so the two can be run side by side
and read the same way: ``1`` is a finding, ``0`` is clean or undeterminable,
and ``2`` means the check could not compute an answer -- red here never means
"this branch needs another human".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

API_ROOT = "https://api.github.com"

_MAX_PAGES = 20
_TIMEOUT = 30


# --------------------------------------------------------------------------
# The sibling check owns "who currently approves" and "who pushed the head".
# Imported by path because ``scripts`` is not a package; see the module
# docstring for why this is a shared primitive rather than a copy.
# --------------------------------------------------------------------------
_SIBLING = Path(__file__).resolve().parent / "check_last_push_approval.py"


def _load_sibling() -> Any:
    spec = importlib.util.spec_from_file_location("check_last_push_approval", _SIBLING)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {_SIBLING}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_approval = _load_sibling()
current_approvers = _approval.current_approvers
resolve_pusher = _approval.resolve_pusher
resolve_reviews = _approval.resolve_reviews


# Outcome names. Ordered here the way they bind in practice, which is also the
# order they are reported in: a conflict makes the approval question moot, and
# an unresolved thread makes it moot for a different reason.
MERGE_CONFLICT = "merge-conflict"
DRAFT = "draft"
REQUIRED_CHECK_FAILING = "required-check-failing"
REQUIRED_CHECK_PENDING = "required-check-pending"
REQUIRED_CHECK_ABSENT = "required-check-absent"
UNRESOLVED_THREADS = "unresolved-threads"
MISSING_APPROVAL = "missing-approval"
PUSHER_ONLY_APPROVAL = "pusher-only-approval"
NO_UNSATISFIED_RULE = "no-unsatisfied-rule"

# Who owes the next action. The whole point of the report.
AUTHOR = "the author"
REVIEWER = "any reviewer"
OTHER_REVIEWER = "a reviewer other than the pusher"
MAINTAINER = "a maintainer"
NOBODY = "nobody"
ANYONE = "anyone, by attempting the merge"

_OWED_BY: dict[str, str] = {
    MERGE_CONFLICT: AUTHOR,
    DRAFT: AUTHOR,
    REQUIRED_CHECK_FAILING: AUTHOR,
    REQUIRED_CHECK_PENDING: NOBODY,
    REQUIRED_CHECK_ABSENT: MAINTAINER,
    UNRESOLVED_THREADS: AUTHOR,
    MISSING_APPROVAL: REVIEWER,
    PUSHER_ONLY_APPROVAL: OTHER_REVIEWER,
    NO_UNSATISFIED_RULE: ANYONE,
}

# A gating blocker makes every rule after it unanswerable rather than merely
# also-unsatisfied: a conflicted or draft branch cannot have its approval
# question settled, because the diff a reviewer would approve is not the diff
# that would merge. Distinguished so the report names one next action instead
# of a set the reader has to order, which is the mistake #1905 records as
# "necessary but no longer sufficient".
_GATING: frozenset[str] = frozenset({MERGE_CONFLICT, DRAFT})


# The outcomes a scheduled author-side pass can act on without anyone else.
# These are the ones that get misread as reviewer bandwidth, so these are the
# ones worth an exit status. MISSING_APPROVAL is excluded deliberately: it is
# the ordinary state, and a finding that fires on the ordinary state is noise.
_FINDINGS: frozenset[str] = frozenset(
    {
        MERGE_CONFLICT,
        DRAFT,
        REQUIRED_CHECK_FAILING,
        UNRESOLVED_THREADS,
        NO_UNSATISFIED_RULE,
    }
)

_STALE_STATE_REMEDY: tuple[str, ...] = (
    "",
    "### What clears this",
    "",
    "Attempt the merge. `mergeStateStatus` is a cached computation and is not",
    "authoritative: #2574 read `blocked` from both GraphQL and REST with zero",
    "review threads and every check green, and `PUT /pulls/{n}/merge` then",
    "succeeded on the first attempt with no state having changed in between.",
    "",
    "Prefer REST for the attempt. GraphQL's `mergePullRequest` refuses with",
    "`Pull Request is not mergeable`, which names nothing; the REST refusal",
    "names the requirement that is unmet.",
    "",
    "Before concluding the state is stale, check the token: an Actions",
    "installation token reads `blocked` on any pull request touching",
    "`.github/workflows/**` whatever the rules say. Re-read with a personal",
    "access token.",
)


@dataclass(frozen=True)
class Blocker:
    """One unsatisfied rule, the party who can clear it, and the detail."""

    outcome: str
    rule: str
    detail: str

    @property
    def owed_by(self) -> str:
        return _OWED_BY.get(self.outcome, NOBODY)

    @property
    def is_finding(self) -> bool:
        return self.outcome in _FINDINGS

    @property
    def is_gating(self) -> bool:
        """Whether the rules after this one cannot be assessed until it clears."""
        return self.outcome in _GATING


@dataclass(frozen=True)
class Ruleset:
    """The subset of the branch ruleset this check can evaluate."""

    required_approving_review_count: int = 0
    required_review_thread_resolution: bool = False
    require_last_push_approval: bool = False
    required_contexts: tuple[str, ...] = ()
    # Named so the report can say a rule is carried but not evaluated here,
    # rather than implying the branch does not carry it.
    unevaluated: tuple[str, ...] = ()


@dataclass(frozen=True)
class PullRequestState:
    """Everything about one pull request the rules are evaluated against."""

    number: int
    head_sha: str
    base_ref: str
    draft: bool
    mergeable: bool | None
    merge_state: str
    unresolved_threads: int
    check_conclusions: dict[str, str | None] = field(default_factory=dict)
    approvers: tuple[str, ...] = ()
    pusher: str | None = None


# --------------------------------------------------------------------------
# Rule evaluation. Pure, so the fixtures in the tests are the real
# observations rather than a mock of the transport.
# --------------------------------------------------------------------------


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def evaluate(state: PullRequestState, rules: Ruleset) -> tuple[Blocker, ...]:
    """Return every rule the pull request leaves unsatisfied, in binding order.

    A rule the branch does not carry is never evaluated, so the report cannot
    name a blocker that is not in force. When nothing is unsatisfied the result
    is a single ``no-unsatisfied-rule`` blocker rather than an empty tuple: the
    caller asked why a pull request is blocked, and "no reason found" is an
    answer with a remedy, not an absence of one.
    """
    found: list[Blocker] = []

    if state.draft:
        found.append(
            Blocker(
                DRAFT,
                "pull_request",
                "The pull request is a draft, so no rule can be satisfied yet.",
            )
        )

    # Upstream of every rule below: a branch that does not merge cleanly cannot
    # be merged by satisfying anything else, and the required check that is
    # green is green on a head that predates the conflict.
    if state.mergeable is False:
        found.append(
            Blocker(
                MERGE_CONFLICT,
                "(not a ruleset rule)",
                f"The branch conflicts with {state.base_ref}; merge state is "
                f"{state.merge_state or 'dirty'}. No approval can merge it while "
                f"this stands, and the required check's green describes a head "
                f"that predates the conflict.",
            )
        )

    for context in rules.required_contexts:
        if context not in state.check_conclusions:
            found.append(
                Blocker(
                    REQUIRED_CHECK_ABSENT,
                    "required_status_checks",
                    f"Required check {context!r} has not reported on "
                    f"{state.head_sha[:8] or '(unknown head)'}. A fork run held at "
                    f"action_required reads the same as one that never started.",
                )
            )
            continue
        conclusion = state.check_conclusions[context]
        if conclusion is None:
            found.append(
                Blocker(
                    REQUIRED_CHECK_PENDING,
                    "required_status_checks",
                    f"Required check {context!r} is still running.",
                )
            )
        elif conclusion.lower() not in ("success", "neutral", "skipped"):
            found.append(
                Blocker(
                    REQUIRED_CHECK_FAILING,
                    "required_status_checks",
                    f"Required check {context!r} concluded {conclusion}.",
                )
            )

    if rules.required_review_thread_resolution and state.unresolved_threads:
        found.append(
            Blocker(
                UNRESOLVED_THREADS,
                "required_review_thread_resolution",
                f"{_plural(state.unresolved_threads, 'review thread')} unresolved. "
                f"This blocks an approved pull request with every check green, and "
                f"it is clearable by the author alone -- including when the fix has "
                f"already landed and the reviewer approved past the thread.",
            )
        )

    if rules.required_approving_review_count:
        # The pusher's own approval is discounted only when the branch actually
        # carries require_last_push_approval. Filtering unconditionally would
        # invent a blocker on a branch that permits a self-approved head.
        eligible = (
            [a for a in state.approvers if a != state.pusher]
            if rules.require_last_push_approval
            else list(state.approvers)
        )
        if not state.approvers:
            found.append(
                Blocker(
                    MISSING_APPROVAL,
                    "required_approving_review_count",
                    f"{len(state.approvers)} of {rules.required_approving_review_count} "
                    f"required approvals. Waiting on a first review, which is the "
                    f"ordinary state.",
                )
            )
        elif rules.require_last_push_approval and not eligible:
            found.append(
                Blocker(
                    PUSHER_ONLY_APPROVAL,
                    "require_last_push_approval",
                    f"Every current approval is from {state.pusher}, the account that "
                    f"pushed the head, so none of them counts. See "
                    f"scripts/check_last_push_approval.py and issue #1905.",
                )
            )
        elif len(eligible) < rules.required_approving_review_count:
            found.append(
                Blocker(
                    MISSING_APPROVAL,
                    "required_approving_review_count",
                    f"{len(eligible)} of {rules.required_approving_review_count} "
                    f"required approvals from an account that did not push the head.",
                )
            )

    if not found:
        return (
            Blocker(
                NO_UNSATISFIED_RULE,
                "(none)",
                "Every rule this check can evaluate is satisfied. If the pull "
                "request still reads blocked, the state is stale or the token "
                "asking cannot see past a workflow change.",
            ),
        )
    return tuple(found)


def parse_ruleset(payload: object) -> Ruleset:
    """Reduce the branch-rules payload to the rules this check evaluates.

    Unknown rule types are ignored rather than refused: a ruleset gaining a
    rule must not turn this check red, and a rule it cannot evaluate is named
    in ``unevaluated`` so the report does not imply the branch is without it.
    """
    rules = payload if isinstance(payload, list) else []
    count = 0
    threads = False
    last_push = False
    contexts: list[str] = []
    unevaluated: list[str] = []

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        kind = rule.get("type")
        params = rule.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        if kind == "pull_request":
            count = int(params.get("required_approving_review_count") or 0)
            threads = bool(params.get("required_review_thread_resolution"))
            last_push = bool(params.get("require_last_push_approval"))
            # Carried by the branch and not answerable here. Named for the
            # reason the docstring gives: silence would read as absence.
            if params.get("require_code_owner_review"):
                unevaluated.append("require_code_owner_review")
            if params.get("require_extra_approval_for_unattributed_changes"):
                unevaluated.append("require_extra_approval_for_unattributed_changes")
        elif kind == "required_status_checks":
            for entry in params.get("required_status_checks") or []:
                if isinstance(entry, dict) and entry.get("context"):
                    contexts.append(str(entry["context"]))

    return Ruleset(
        required_approving_review_count=count,
        required_review_thread_resolution=threads,
        require_last_push_approval=last_push,
        required_contexts=tuple(contexts),
        unevaluated=tuple(unevaluated),
    )


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "strands-robots-check-merge-blockers",
    }


def _get(url: str, token: str) -> object:
    request = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310 - fixed API host
        return json.load(response)


_THREAD_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100) { nodes { isResolved isOutdated } }
    }
  }
}
"""


def resolve_unresolved_threads(repo: str, pr: int, token: str) -> int:
    """Count review threads that are neither resolved nor outdated.

    GraphQL only: REST exposes review comments but not whether the thread they
    belong to has been resolved, and resolution is the whole question. An
    outdated thread does not block a merge, so it is not counted -- the rule is
    about threads the ruleset considers live.
    """
    owner, _, name = repo.partition("/")
    body = json.dumps({"query": _THREAD_QUERY, "variables": {"owner": owner, "name": name, "number": pr}}).encode()
    headers = _headers(token)
    headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{API_ROOT}/graphql", data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310 - fixed API host
        payload = json.load(response)
    if payload.get("errors"):
        raise ValueError(f"reviewThreads query failed: {payload['errors']}")
    threads = (
        (((payload.get("data") or {}).get("repository") or {}).get("pullRequest") or {}).get("reviewThreads") or {}
    ).get("nodes") or []
    return sum(1 for t in threads if not t.get("isResolved") and not t.get("isOutdated"))


def resolve_check_conclusions(repo: str, head_sha: str, token: str) -> dict[str, str | None]:
    """Map check name to conclusion on the head, plus legacy commit statuses.

    A check that is queued or in progress has a ``None`` conclusion, which is
    kept as ``None`` rather than coerced: "still running" and "failed" ask for
    different things. Both surfaces are read because a required context can be
    supplied by either, and the ruleset names a context without saying which.
    """
    conclusions: dict[str, str | None] = {}

    payload = _get(f"{API_ROOT}/repos/{repo}/commits/{head_sha}/check-runs?per_page=100", token)
    runs = payload.get("check_runs", []) if isinstance(payload, dict) else []
    for run in runs:
        name = run.get("name")
        if not name:
            continue
        conclusion = run.get("conclusion")
        # A context appearing twice keeps its worst answer: a re-run that
        # succeeded does not retire a sibling that did not.
        if name in conclusions and conclusions[name] not in (None, "success"):
            continue
        conclusions[str(name)] = conclusion

    status = _get(f"{API_ROOT}/repos/{repo}/commits/{head_sha}/status", token)
    if isinstance(status, dict):
        for entry in status.get("statuses") or []:
            context = entry.get("context")
            if context and context not in conclusions:
                state = entry.get("state")
                conclusions[str(context)] = None if state == "pending" else state

    return conclusions


def resolve_ruleset(repo: str, base_ref: str, token: str) -> Ruleset:
    quoted = urllib.parse.quote(base_ref, safe="")
    return parse_ruleset(_get(f"{API_ROOT}/repos/{repo}/rules/branches/{quoted}", token))


def resolve_state(repo: str, pr: int, token: str) -> PullRequestState:
    """Read one pull request and everything the rules are evaluated against."""
    payload = _get(f"{API_ROOT}/repos/{repo}/pulls/{pr}", token)
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected payload for {repo}#{pr}")
    head_sha = ((payload.get("head") or {}).get("sha")) or ""
    base_ref = ((payload.get("base") or {}).get("ref")) or ""
    reviews = resolve_reviews(repo, pr, token)
    return PullRequestState(
        number=pr,
        head_sha=head_sha,
        base_ref=base_ref,
        draft=bool(payload.get("draft")),
        mergeable=payload.get("mergeable"),
        merge_state=str(payload.get("mergeable_state") or ""),
        unresolved_threads=resolve_unresolved_threads(repo, pr, token),
        check_conclusions=resolve_check_conclusions(repo, head_sha, token) if head_sha else {},
        approvers=current_approvers(reviews),
        pusher=resolve_pusher(repo, head_sha, token) if head_sha else None,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def primary(blockers: Sequence[Blocker]) -> Blocker:
    """Return the blocker that owns the next action.

    A gating blocker wins outright, because nothing after it is answerable. Then
    a finding, because that is the class a scheduled pass can act on alone. Then
    the first blocker somebody actually owes, so a rule that is merely waiting
    on a clock cannot mask one that is waiting on a person: a pull request whose
    required check is still running and which has no approval is answerable by a
    reviewer now, and reporting it as owed by nobody would park it. Only if no
    blocker is owed by anyone does the earliest-binding one stand.

    Both renderers and both warning paths route through this, so precedence
    cannot be applied to one report and not the other.
    """
    return next(
        (b for b in blockers if b.is_gating),
        next(
            (b for b in blockers if b.is_finding),
            next((b for b in blockers if b.owed_by != NOBODY), blockers[0]),
        ),
    )


def _next_action(blockers: Sequence[Blocker]) -> list[str]:
    """Render the one next action, honouring precedence between blockers.

    A gating blocker is reported alone. Listing it beside the approval rule it
    sits upstream of is what produced the misreading #1905 records: the reader
    sees two owners, picks the reviewer, and supplies an approval that cannot
    merge anything. Non-gating blockers genuinely are parallel -- a pending
    check and a missing approval wait on different people at once -- so those
    are listed together.
    """
    gating = primary(blockers) if any(b.is_gating for b in blockers) else None
    if gating is not None:
        trailing = [b for b in blockers if b is not gating]
        lines = [f"Next action is owed by {gating.owed_by}, on `{gating.outcome}`."]
        if trailing:
            counted = "rule" if len(trailing) == 1 else f"{len(trailing)} rules"
            lines.append(
                f"The {counted} below it cannot be assessed until that clears: an"
                " approval is necessary but not sufficient while it stands."
            )
        return lines
    owed = sorted({b.owed_by for b in blockers if b.owed_by != NOBODY})
    if owed:
        return [f"Next action is owed by {_join(owed)}."]
    return ["No party owes an action; the answer is not in yet."]


def render(state: PullRequestState, rules: Ruleset, blockers: Sequence[Blocker], repo: str) -> str:
    """Render one pull request's blockers, the party owing each named first."""
    lines = [
        "## Merge blockers",
        "",
        f"Outcome: **{', '.join(b.outcome for b in blockers)}**",
        "",
        *_next_action(blockers),
        "",
    ]
    lines += [
        "| field | value |",
        "|---|---|",
        f"| pull request | {repo}#{state.number} |",
        f"| head | `{state.head_sha[:8] or '(unknown)'}` |",
        f"| base | {state.base_ref or '(unknown)'} |",
        f"| merge state (cached, not authoritative) | {state.merge_state or '(unknown)'} |",
        f"| unresolved review threads | {state.unresolved_threads} |",
        f"| current approvals | {_join(state.approvers)} |",
        f"| head pushed by | {state.pusher or '(undetermined)'} |",
        "",
        "| unsatisfied rule | owed by | detail |",
        "|---|---|---|",
    ]
    for blocker in blockers:
        lines.append(f"| `{blocker.rule}` | {blocker.owed_by} | {blocker.detail} |")

    if rules.unevaluated:
        lines += [
            "",
            "Carried by the branch and not evaluated here: "
            + ", ".join(f"`{name}`" for name in rules.unevaluated)
            + ". Named so their absence from the table above does not read as the"
            " branch not carrying them.",
        ]
    if any(b.outcome == NO_UNSATISFIED_RULE for b in blockers):
        lines += list(_STALE_STATE_REMEDY)
    return "\n".join(lines)


def _join(names: Sequence[str]) -> str:
    names = list(names)
    if not names:
        return "nobody"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


@dataclass(frozen=True)
class SweepRow:
    """One open pull request and the blockers computed for it."""

    pr: int
    blockers: tuple[Blocker, ...]

    @property
    def is_finding(self) -> bool:
        return any(b.is_finding for b in self.blockers)

    @property
    def primary(self) -> Blocker:
        return primary(self.blockers)


def sweep(repo: str, token: str) -> tuple[list[SweepRow], list[int], Ruleset]:
    """Evaluate every open non-draft pull request against its base ruleset.

    One unreadable pull request is skipped and named, never allowed to take the
    rest of the report with it. Rulesets are cached per base branch, because a
    sweep of thirty pull requests onto one branch asks the same question thirty
    times otherwise.
    """
    rows: list[SweepRow] = []
    skipped: list[int] = []
    cache: dict[str, Ruleset] = {}
    last = Ruleset()

    for pr in resolve_open_pull_requests(repo, token):
        try:
            state = resolve_state(repo, pr, token)
            if state.base_ref not in cache:
                cache[state.base_ref] = resolve_ruleset(repo, state.base_ref, token)
            rules = cache[state.base_ref]
            last = rules
            rows.append(SweepRow(pr, evaluate(state, rules)))
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as exc:
            print(
                f"check_merge_blockers: {repo}#{pr} lookup failed, not evaluated: {exc}",
                file=sys.stderr,
            )
            skipped.append(pr)
    return rows, skipped, last


def resolve_open_pull_requests(repo: str, token: str) -> list[int]:
    """Return every open non-draft pull request number, ascending.

    Sorted so two reports differ by changed verdicts rather than by ordering.
    """
    found: list[int] = []
    for _ in range(_MAX_PAGES):
        page = len(found) // 100 + 1
        payload = _get(f"{API_ROOT}/repos/{repo}/pulls?state=open&per_page=100&page={page}", token)
        rows = payload if isinstance(payload, list) else []
        for row in rows:
            if row.get("draft"):
                continue
            number = row.get("number")
            if isinstance(number, int):
                found.append(number)
        if len(rows) < 100:
            break
    return sorted(found)


def render_sweep(rows: Sequence[SweepRow], skipped: Sequence[int], repo: str) -> str:
    """Render the sweep, separating what the author owes from what a reviewer does."""
    findings = [row for row in rows if row.is_finding]
    lines = [
        "## Merge blocker sweep",
        "",
        f"Evaluated {len(rows)} open non-draft pull request(s) in {repo}.",
        "",
    ]
    if findings:
        named = ", ".join(f"#{row.pr}" for row in findings)
        lines += [
            f"**{len(findings)} blocked on something no reviewer can clear:** {named}.",
            "",
            "Each of these reads the same as waiting on a reviewer and is not.",
            "",
        ]
    else:
        lines += [
            "No pull request is blocked on an author-clearable rule. Anything",
            "blocked here is waiting on review, which is the ordinary state.",
            "",
        ]
    lines += [
        "| pull request | unsatisfied rule(s) | owed by |",
        "|---|---|---|",
    ]
    for row in rows:
        outcomes = ", ".join(b.outcome for b in row.blockers)
        lines.append(f"| #{row.pr} | {outcomes} | {row.primary.owed_by} |")
    if skipped:
        lines += [
            "",
            "Not evaluated (lookup failed): " + ", ".join(f"#{pr}" for pr in skipped) + ".",
            "Named rather than omitted, so a gap in coverage cannot read as a clean sweep.",
        ]
    return "\n".join(lines)


def _emit(report: str) -> None:
    """Print the report, and append it to the step summary when there is one."""
    print(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")


def _describe(blockers: Iterable[Blocker]) -> str:
    """One line naming each blocker, its rule, and who owes the next action."""
    return "; ".join(f"{b.outcome} ({b.rule}), owed by {b.owed_by}" for b in blockers)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", ""))
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
    # Refused rather than resolved, for the reason the sibling gives: silently
    # ignoring either flag reads as a successful run of the other thing.
    if args.all_open and args.pr:
        parser.error("--all-open and --pr are mutually exclusive")
    if not args.all_open and not args.pr:
        parser.error("--pr is required (or set PR_NUMBER), or pass --all-open")

    if args.all_open:
        try:
            rows, skipped, rules = sweep(args.repo, args.token)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            print(f"check_merge_blockers: could not list open pull requests: {exc}", file=sys.stderr)
            return 0
        _emit(render_sweep(rows, skipped, args.repo))
        findings = [row for row in rows if row.is_finding]
        for row in findings:
            print(
                f"::warning title=Blocked on something no reviewer can clear::"
                f"{args.repo}#{row.pr}: {_describe([row.primary])}"
            )
        return 1 if findings else 0

    pr = int(args.pr)
    try:
        state = resolve_state(args.repo, pr, args.token)
        rules = resolve_ruleset(args.repo, state.base_ref, args.token)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as exc:
        # A lookup failure is not a finding: say so on stderr and pass, rather
        # than accusing a branch of a blocker this run could not observe.
        print(f"check_merge_blockers: lookup failed, reporting nothing: {exc}", file=sys.stderr)
        return 0

    blockers = evaluate(state, rules)
    _emit(render(state, rules, blockers, args.repo))
    if any(b.is_finding for b in blockers):
        print(f"::warning title=Blocked on something no reviewer can clear::{_describe([primary(blockers)])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
