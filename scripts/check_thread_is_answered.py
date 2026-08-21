#!/usr/bin/env python3
"""Report which of a pull request's review threads are still owed an author reply.

Why this exists
---------------
A review thread is append-only, and an agent that rebuilds its context each run
cannot tell "this question is unanswered" from "this question is answered and the
answer is sitting in the payload I just read". The reviewer's question is still
there verbatim either way. Nothing in the serialised thread says *answered*.

So the same thread gets answered repeatedly. Measured on two threads in this
repository:

============  ==========================  ===============  =========================
pull request  thread                       author replies   what the replies after
                                                            the first one added
============  ==========================  ===============  =========================
#2511         ``PRRT_kwDORUMiZs6arq8z``    3                nothing -- the same fix,
                                                            the same commit
                                                            (``c2969d4``), the same
                                                            6 failures reproduced
                                                            three more times, over
                                                            27 minutes
#2577         ``PRRT_kwDORUMiZs6bQMZz``    2                nothing -- ``d04a8969``
                                                            was already named by the
                                                            first reply, 10 minutes
                                                            earlier
============  ==========================  ===============  =========================

The cost is not the wasted run. It is that a duplicate reply is permanent: every
later reviewer and every later run has to read it, and reading it is itself what
makes the next run likely to add one more. #2520 records the loop.

Why the rule that already exists did not stop it
------------------------------------------------
AGENTS.md is not silent here. It already says a thread that ``isResolved`` or
``isOutdated`` is terminal and must not be replied to, and that a thread whose
last non-bot comment is yours has already been answered. #2577's thread was
**both** ``isResolved: true`` and ``isOutdated: true`` when it received its third
comment at 19:28:20Z, and its second and third comments were both the author's.
Every field needed to refuse that reply was in the payload, and had been fetched.

That is the argument for a command rather than another paragraph. The rule asks a
reader to re-derive one boolean from a payload that also contains a reviewer's
question addressed to them; the command asks for a pull request number and prints
a verdict. The same reasoning put ``check_duplicate_claim.py`` in ``scripts/``
after prose alone failed to prevent three duplicate claims.

Why a reply cannot be keyed to the commit that answers it
---------------------------------------------------------
The obvious implementation is to read the commit a reply was written against and
compare it to the branch head. It cannot work: ``originalCommit`` is a property of
the **thread**, not of the comment. On #2511 all four comments -- one reviewer
comment and three author replies spanning 04:19 to 04:46 -- report the same
``originalCommit`` ``ee3e4526``, while the branch head by the last reply was
``0a69634d``:

===================  =========================  ==========================
comment              author                     ``originalCommit``
===================  =========================  ==========================
1 (04:19 reviewer)   ``yinsong1986``            ``ee3e4526``
2 (04:19 reply)      ``cagataycali``            ``ee3e4526``
3 (04:31 reply)      ``cagataycali``            ``ee3e4526``
4 (04:46 reply)      ``cagataycali``            ``ee3e4526``
===================  =========================  ==========================

So a reply carries no record of the commit it was verified against, and the only
commit a thread can be keyed to is the pull request's own ``headRefOid``. This
reads that, reports it beside every thread, and says whether the head has moved
off the commit the thread was written against -- which is what a second run needs
in order to tell "already fixed at ``d04a8969``" from "not yet fixed" (#2520).

What decides the outcome, and what is only reported
---------------------------------------------------
One rule decides it: **whoever spoke last owes the next move, unless a reviewer
resolved the thread.** Resolution is terminal because it is a reviewer action.
Everything else is the authorship test.

Two fields that look like they belong in that decision are reported beside it
instead, and both exclusions are deliberate.

``isOutdated`` is not consulted. It describes the diff, not the conversation: it
says the lines the thread was written against have moved. As a stand-in for "this
has been answered" it is wrong in both directions, and each direction is measured
here.

It can be ``false`` on a thread that *is* answered. #2480's thread still reads
``isOutdated: false`` after ``e83cf51`` fixed the finding it raised, because the
fix added a method rather than rewriting the commented line. So outdated is not
evidence of an answer.

And a thread keeps accepting comments after it flips to ``true``. #2577's thread
is ``isOutdated: true`` -- ``d04a8969`` moved those lines -- and it took two
further comments at 19:17:50Z and 19:28:20Z regardless. Both happened to be the
author's, but nothing about the flag prevents a reviewer's from arriving the same
way, and a rule that reads outdated as terminal would file that demand as settled.
Authorship subsumes what the proxy was reaching for: if the last word is yours
there is nothing to say, however the diff moved -- and if it is not, there is.
Pinned by ``test_a_reviewer_demand_on_an_outdated_thread_is_owed``.

Whether the head has moved is not consulted either. An author reply that explains
rather than changes -- "this is intentional, here is why" -- is a complete answer
that leaves the head exactly where the thread was written, and folding the head
comparison into the verdict would call it unanswered and invite the duplicate
reply this exists to prevent. Pinned by
``test_an_answer_with_no_commit_after_it_is_still_answered``.

What the steady state looks like, and why that is the point
-----------------------------------------------------------
Swept over the 150 most recently updated closed pull requests here, every one of
the 26 review threads found is ``isResolved: true`` and therefore ``settled``, and
all 150 report ``nothing-owed``. The same sweep over the 11 currently open pull
requests also reports ``nothing-owed`` throughout.

That is the expected result, not a broken check: threads in this repository do get
resolved, so on a finished pull request ``settled`` is the only outcome left. The
two discriminating outcomes exist only in the window between a reviewer's comment
and its resolution -- which is exactly the window a scheduled run reads in, and
exactly when the wrong answer costs a permanent duplicate reply. Their coverage
therefore comes from ``tests/test_thread_is_answered.py``, which replays the
measured payloads of the two incidents above with the flags as they stood in that
window, rather than from a historical sweep that cannot contain them.

The incidents are attributable on timestamped fields alone, which is why
authorship carries the rule. Thread resolution has no timestamp in the API, so
"was it resolved yet" is not answerable after the fact -- but at 19:28:20Z on
#2577 the last non-bot comment was the author's own reply from 19:17:50Z, and at
04:31 and 04:46 on #2511 the last non-bot comment was the author's reply from
04:19. The authorship test alone returns ``answered`` for all three, with no
appeal to a field whose history cannot be read back.

The last *non-bot* comment decides, because ``github-advanced-security`` posts
directly into review threads -- measured on #2480 and on all four threads of
#1722 -- and a bot commenting after the author's reply would otherwise flip an
answered thread back to owed. AGENTS.md already words its rule as "last non-bot
comment", so this implements the documented rule rather than inventing one. A
thread carrying only bot comments has no author answer and is owed.

Outcomes
--------
Per thread:

``settled``
    ``isResolved``. Terminal; a reviewer has closed the business, and reopening it
    to restate a landed fix reads as noise.
``answered``
    Open, and the last non-bot comment is the pull request author's. The author
    has had the last word; the next move is the reviewer's.
``awaiting-the-author``
    Open, and the last non-bot comment belongs to someone other than the author
    (or there is no non-bot comment). This is the only outcome that is work.

Per pull request: ``nothing-owed``, or ``author-owes-a-reply`` when at least one
thread is ``awaiting-the-author``.

What this deliberately does not do
----------------------------------
It does not read comment bodies. A reply naming a commit and a reply naming
nothing are the same to this check, because judging whether an answer is adequate
is the reviewer's job and a check that attempted it would be wrong in the
direction of arguing with a reviewer.

It does not gate, and is wired to no workflow. Its answer is about what an author
should do next, which is not something a branch can turn green, and this
repository already keeps that kind of check out of CI
(``check_pr_head_is_current.py``, ``check_duplicate_claim.py``,
``check_merge_blockers.py``).

Usage
-----
::

    python3 scripts/check_thread_is_answered.py --repo strands-labs/robots --pr 2577
    python3 scripts/check_thread_is_answered.py --repo strands-labs/robots --all-open

Exit 1 when at least one evaluated pull request has a thread awaiting its author.

Pinned by tests/test_thread_is_answered.py.
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
# check_pr_head_is_current.py and check_last_push_approval.py.
_MAX_PAGES = 20

# Read ceilings, chosen against the GraphQL node budget rather than by taste:
# the sweep multiplies all three, and 25 x 100 x 20 stays an order of magnitude
# under the 500,000-node limit that 50 x 100 x 100 would sit exactly on.
#
# Threads are read from the *start* and the shortfall is named in the report,
# because the unread tail of a long thread list is its newest threads and those
# are the likeliest to be owed -- silently dropping them would report
# nothing-owed for the one case that is work. Comments are read from the *end*,
# because only the last non-bot comment decides and a thread's early history
# cannot change it.
_PRS_PER_PAGE = 25
_THREADS_PER_PR = 100
_COMMENTS_PER_THREAD = 20

SETTLED = "settled"
ANSWERED = "answered"
AWAITING_THE_AUTHOR = "awaiting-the-author"

NOTHING_OWED = "nothing-owed"
AUTHOR_OWES_A_REPLY = "author-owes-a-reply"

_API = "https://api.github.com/graphql"

# The remedy text, shared by the single-pull-request report and the sweep so the
# two cannot drift into describing different remedies for one outcome.
WHAT_TO_DO: tuple[str, ...] = (
    "",
    "### What clears this",
    "",
    "Answer each `awaiting-the-author` thread **once**, then resolve it. One",
    "concern, one commit, one reply -- and the reply is owed only because the",
    "last word is not yours yet.",
    "",
    "Threads reported `answered` or `settled` are not work. Do not reply to them",
    "to restate a fix that has landed: the reviewer's question stays in the",
    "thread verbatim after it is answered, so its presence is not evidence that",
    "anything is outstanding. If code is owed, push it -- the push is the",
    "message.",
    "",
    "`head moved` beside an `answered` thread says a commit followed the thread,",
    "not that the commit fixed it. It is there so a later run can tell",
    '"already fixed at <oid>" from "not yet fixed" without re-deriving the fix.',
)


def _short(oid: str | None) -> str:
    """Render a commit for a human without pretending an absent one is a commit."""
    if not oid:
        return "(unknown)"
    return f"`{oid[:8]}`"


@dataclass(frozen=True)
class Thread:
    """One review thread, and the two facts its outcome was computed from."""

    thread_id: str
    path: str
    outcome: str
    last_author: str | None
    comments: int
    original_commit: str | None
    head_moved: bool
    outdated: bool

    @property
    def is_owed(self) -> bool:
        return self.outcome == AWAITING_THE_AUTHOR


@dataclass(frozen=True)
class Row:
    """One evaluated pull request."""

    pr: int
    author: str
    head: str | None
    threads: tuple[Thread, ...]
    unread_threads: int

    @property
    def owed(self) -> tuple[Thread, ...]:
        return tuple(t for t in self.threads if t.is_owed)

    @property
    def outcome(self) -> str:
        return AUTHOR_OWES_A_REPLY if self.owed else NOTHING_OWED

    @property
    def is_finding(self) -> bool:
        return bool(self.owed)

    @property
    def summary(self) -> str:
        if self.owed:
            named = ", ".join(f"`{t.path}`" for t in self.owed)
            return (
                f"{len(self.owed)} of {len(self.threads)} thread(s) are awaiting @{self.author}: "
                f"{named}. Answer each once, then resolve it."
            )
        if not self.threads:
            return "No review threads. Nothing is owed here."
        return (
            f"All {len(self.threads)} thread(s) are settled or already answered by @{self.author}. "
            "Nothing is owed here; the next move belongs to a reviewer."
        )


def last_non_bot_author(comments: Sequence[dict]) -> str | None:
    """Return the login of the last comment not written by a bot.

    ``None`` when every comment is a bot's, which is not the same as an empty
    thread and is treated the same way by the caller: neither carries an answer
    from the author.
    """
    for comment in reversed(list(comments)):
        author = comment.get("author") or {}
        if author.get("__typename") == "Bot":
            continue
        login = author.get("login")
        if isinstance(login, str) and login:
            return login
    return None


def classify(node: dict, pr_author: str, head: str | None) -> Thread:
    """Decide whether one review thread is owed a reply from ``pr_author``.

    ``head`` is the pull request's recorded head commit. It is used only to
    compute ``head_moved`` for the report -- never to choose the outcome. See the
    module docstring: an answer that explains rather than changes leaves the head
    where the thread was written, and is still an answer.
    """
    comments = ((node.get("comments") or {}).get("nodes")) or []
    first = comments[0] if comments else {}
    original = ((first.get("originalCommit") or {}).get("oid")) or None
    last_author = last_non_bot_author(comments)
    head_moved = bool(head and original and head != original)

    if node.get("isResolved"):
        outcome = SETTLED
    elif last_author is not None and last_author == pr_author:
        outcome = ANSWERED
    else:
        outcome = AWAITING_THE_AUTHOR

    return Thread(
        thread_id=str(node.get("id") or "(unknown)"),
        path=str(node.get("path") or "(unknown)"),
        outcome=outcome,
        last_author=last_author,
        comments=int((node.get("comments") or {}).get("totalCount") or len(comments)),
        original_commit=original,
        head_moved=head_moved,
        outdated=bool(node.get("isOutdated")),
    )


def evaluate(node: dict) -> Row:
    """Evaluate one pull request node from either query shape."""
    author = ((node.get("author") or {}).get("login")) or "(unknown)"
    head = node.get("headRefOid")
    connection = node.get("reviewThreads") or {}
    nodes = connection.get("nodes") or []
    total = connection.get("totalCount")
    unread = max(0, int(total) - len(nodes)) if isinstance(total, int) else 0
    return Row(
        pr=int(node["number"]),
        author=author,
        head=head if isinstance(head, str) else None,
        threads=tuple(classify(t, author, head if isinstance(head, str) else None) for t in nodes),
        unread_threads=unread,
    )


def _graphql(query: str, variables: dict[str, object], token: str) -> dict:
    request = urllib.request.Request(
        _API,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "User-Agent": "strands-robots-check-thread-is-answered",
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
  number isDraft headRefOid
  author { login }
  reviewThreads(first: __THREADS__) {
    totalCount
    nodes {
      id isResolved isOutdated path
      comments(last: __COMMENTS__) {
        totalCount
        nodes { author { login __typename } originalCommit { oid } }
      }
    }
  }
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
    fields = _PR_FIELDS.replace("__THREADS__", str(_THREADS_PER_PR)).replace("__COMMENTS__", str(_COMMENTS_PER_THREAD))
    return query.replace(_FIELDS_PLACEHOLDER, fields)


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
    pullRequests(states: OPEN, first: __PAGE__, after: $cursor){
      pageInfo { hasNextPage endCursor }
      nodes { __PR_FIELDS__ }
    }
  }
}
""".replace("__PAGE__", str(_PRS_PER_PAGE))
)


def _split_repo(repo: str) -> tuple[str, str]:
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"--repo must be owner/name, got {repo!r}")
    owner, name = repo.split("/")
    return owner, name


def fetch_one(repo: str, number: int, token: str) -> dict:
    owner, name = _split_repo(repo)
    data = _graphql(_ONE_PR, {"owner": owner, "name": name, "number": number}, token)
    node = (data.get("repository") or {}).get("pullRequest")
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


def _thread_rows(row: Row) -> list[str]:
    lines = [
        "| thread | outcome | last non-bot comment | comments | thread commit | head moved | outdated |",
        "|---|---|---|---|---|---|---|",
    ]
    for thread in row.threads:
        who = f"@{thread.last_author}" if thread.last_author else "(only bots)"
        lines.append(
            f"| `{thread.path}` | {thread.outcome} | {who} | {thread.comments} | "
            f"{_short(thread.original_commit)} | {'yes' if thread.head_moved else 'no'} | "
            f"{'yes' if thread.outdated else 'no'} |"
        )
    return lines


def _unread_note(row: Row) -> list[str]:
    if not row.unread_threads:
        return []
    return [
        "",
        f"{row.unread_threads} thread(s) beyond the first {_THREADS_PER_PR} were not read, so this "
        "report is silent about them rather than clearing them.",
    ]


def render_one(repo: str, row: Row) -> str:
    lines = [
        "## Review threads owed a reply",
        "",
        f"Outcome: **{row.outcome}**",
        "",
        row.summary,
        "",
        "| field | value |",
        "|---|---|",
        f"| pull request | {repo}#{row.pr} |",
        f"| author | @{row.author} |",
        f"| recorded head | {_short(row.head)} |",
        f"| threads | {len(row.threads)} |",
        f"| awaiting the author | {len(row.owed)} |",
    ]
    if row.threads:
        lines += [""] + _thread_rows(row)
    lines += _unread_note(row)
    if row.is_finding:
        lines += list(WHAT_TO_DO)
    return "\n".join(lines)


def render_sweep(repo: str, rows: Sequence[Row], skipped: Sequence[int]) -> str:
    findings = [r for r in rows if r.is_finding]
    lines = [
        "## Review threads owed a reply -- sweep",
        "",
        f"Evaluated {len(rows)} open non-draft pull request(s) in {repo}.",
        "",
    ]
    if findings:
        named = ", ".join(f"#{r.pr}" for r in findings)
        lines += [f"**{len(findings)} pull request(s) have a thread awaiting their author:** {named}.", ""]
    else:
        lines += ["Every review thread is settled or already answered by its author.", ""]

    lines += [
        "| pull request | author | outcome | threads | awaiting | answered | settled |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        answered = sum(1 for t in row.threads if t.outcome == ANSWERED)
        settled = sum(1 for t in row.threads if t.outcome == SETTLED)
        lines.append(
            f"| #{row.pr} | @{row.author} | {row.outcome} | {len(row.threads)} | "
            f"{len(row.owed)} | {answered} | {settled} |"
        )
    for row in findings:
        lines += ["", f"### #{row.pr}", ""] + _thread_rows(row)
    for row in rows:
        note = _unread_note(row)
        if note:
            lines += [note[0], f"#{row.pr}: {note[1]}"]
    if skipped:
        lines += ["", "Skipped: " + ", ".join(f"#{n}" for n in skipped) + "."]
    if findings:
        lines += list(WHAT_TO_DO)
    return "\n".join(lines)


def _publish(report: str) -> None:
    print(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")


def _annotate(repo: str, row: Row) -> None:
    print(f"::warning title=A review thread is awaiting its author::{repo}#{row.pr}: {row.summary}")


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

    # Refused rather than resolved, for the reason the siblings give: silently
    # ignoring either flag reads as a successful run of the other thing.
    if args.all_open and args.pr:
        parser.error("--all-open and --pr are mutually exclusive")

    if not args.repo:
        print("check_thread_is_answered: --repo is required", file=sys.stderr)
        return 0
    if not args.token:
        print("check_thread_is_answered: no token; set --token or GITHUB_TOKEN", file=sys.stderr)
        return 0

    if args.all_open:
        try:
            nodes = fetch_open(args.repo, args.token)
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
            # Listing is the one lookup with no partial result to report, so it
            # fails open rather than failing the caller's run.
            print(f"check_thread_is_answered: could not list open pull requests: {exc}", file=sys.stderr)
            return 0
        rows: list[Row] = []
        skipped: list[int] = []
        for node in nodes:
            if node.get("isDraft"):
                continue
            try:
                rows.append(evaluate(node))
            except (ValueError, KeyError, TypeError) as exc:
                # One unreadable pull request must not suppress a finding on the
                # others, so it is named and skipped rather than fatal.
                number = node.get("number")
                print(f"check_thread_is_answered: skipped #{number}: {exc}", file=sys.stderr)
                if isinstance(number, int):
                    skipped.append(number)
        rows.sort(key=lambda r: r.pr)
        _publish(render_sweep(args.repo, rows, skipped))
        findings = [r for r in rows if r.is_finding]
        for row in findings:
            _annotate(args.repo, row)
        return 1 if findings else 0

    if not args.pr:
        print("check_thread_is_answered: --pr or --all-open is required", file=sys.stderr)
        return 0
    try:
        row = evaluate(fetch_one(args.repo, int(args.pr), args.token))
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, TypeError) as exc:
        # A lookup failure is not a finding: say so on stderr and pass, rather
        # than accusing an author of a reply this run could not observe.
        print(f"check_thread_is_answered: could not evaluate {args.repo}#{args.pr}: {exc}", file=sys.stderr)
        return 0
    _publish(render_one(args.repo, row))
    if row.is_finding:
        _annotate(args.repo, row)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
