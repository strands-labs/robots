#!/usr/bin/env python3
"""Report whether an open pull request already closes a given issue.

Why this exists
---------------
``scripts/check_closing_reference.py`` reads one pull request in isolation: it
compares the issues a *title* claims against that same pull request's
``closingIssuesReferences``. Both of its inputs are per-pull-request, so a branch
that claims an issue **another open pull request already claims** passes every
check in this repository, and measured over the last 100 pull requests (#1867
through #2016) the notice has arrived after review every time.

Three pairs claimed one issue each -- six pull requests, three of them wasted::

    issue   pair            opened apart   abandoned   had reached
    #1942   #1944, #1946    8m 32s         #1944       APPROVED
    #1994   #1995, #1996    31m 58s        #1995       APPROVED
    #2007   #2015, #2016    4m 9s          #2016       APPROVED

The costly property is the last column. What a duplicate wastes is not the
authoring, it is a **review approval spent on a change that could never ship**,
and review is the scarcest resource here -- #1905 measures the same scarcity from
the other direction. Two of the three pairs could not both land regardless: the
closing comments on #1944 and #1995 each record a ``git merge-tree`` content
conflict, so the duplication was a merge failure waiting on whichever pull request
lost the race.

Every pair also opened inside one ~35-minute window, and that is what decides
where this belongs. The collision is an **intake** failure -- the second pull
request is opened before anything has observed the first -- not a drift that
accumulates over days. A check that reads two existing pull requests recovers the
review cost but not the authoring cost; a check run *before* the second one is
opened prevents both. So the primary entry point is ``--issue``, asked at intake,
and AGENTS.md step 1 is where it is asked.

Why the existing gate cannot see it
-----------------------------------
That check reads the title and the link set GitHub publishes, and that scope is
right for what it does. A duplicate claim is a property of the *set* of open pull
requests, so no single-pull-request assertion can reach it. The link set is still
the right thing to read -- it just has to be read across the open pull requests
rather than for one.

Why there is no rule about which of the two is at fault
------------------------------------------------------
Issue #2017 proposed failing "the newer of the two". Measured against the three
pairs, that names the wrong pull request twice::

    issue   older   newer   merged
    #1942   #1944   #1946   #1946   <- the newer one
    #1994   #1995   #1996   #1996   <- the newer one
    #2007   #2015   #2016   #2015

So in two of three the newer pull request is the one that survived, and an age
rule would have accused the eventual survivor. It is also not this check's
question: both numbers go in the report and which claim to drop stays with the
people who know what the two branches do.

That a repository-read token can see another pull request's link set is not
assumed. Issue #1961 records ``PullRequest.projectItems`` returning a false ``0``
under ``GITHUB_TOKEN``, so the lens was checked before relying on it: an Actions
token and a personal token return identical link sets for all the open pull
requests, including the one linking #1034. A false empty would turn this into a
silent no-op that always agrees, which is why an incomplete read is reported as
``unknown-claims`` rather than as a pass.

The open list is read through ``repository.pullRequests(states: OPEN)`` and not
through ``search``. Search is eventually consistent, so a pull request opened
seconds ago may not be indexed -- and the missing row would be a false clean in
exactly the ~35-minute window where every observed collision happened.

Three questions, two keys
-------------------------
``--issue N``
    Intake, keyed on the claim. *Before* authoring: does an open pull request
    already close #N? Nothing is excluded from the comparison, because no pull
    request for it exists yet.

``--pr N``
    Review, keyed on the claim. Do this pull request's own claims collide with
    another open one's? #N is excluded from its own comparison -- a pull request
    always shares its own claim.

``--all-open``
    Review, keyed on what a branch creates and what it says. Do two open pull
    requests create the same file, name one changelog entry, or describe one
    change over one test? Needs no issue number, which is the point: see below.

The key a claim-free pair collides on
-------------------------------------
Both claim-keyed questions read ``closingIssuesReferences``, and **249 of the
last 300 pull requests (#2345 through #2708) link no issue at all**. For that 83%
of the traffic there is no key to collide on, so both modes report a unique claim
while looking straight at a duplicate pair. Issue #2709 is the third recorded
instance.

That residual was listed here as out of scope on the strength of one
measurement: 18 of the last 30 merges would fail a rule *requiring* a claim. The
measurement stands and the conclusion drawn from it was wider than it supports.
It rules out demanding a claim. It says nothing about colliding on a different
key -- and a changed-path set exists for every branch whether it claims anything
or not.

The part of that set which answers this question is the paths a branch **adds**.
Two branches editing one file is a composition to verify, which is the sibling
sweep's question in ``scripts/check_merge_base_overlap.py``; its remedy is a
merge order plus possibly one test run. Two branches *creating* one file is not a
composition at all. It is two answers to one question, and one of them is going
to be closed.

Measured over 353 pull requests (#2345 through #2767), on the 2002 pairs that
were open at the same instant::

    relation                        pairs   duplicates among them
    both edit a path                  127   a composition question, not this one
    both add a path                     2   2
    both create the same thing          3   3   <- what this sweep pairs on

All three are duplicates, and none was reachable from a claim::

    what both create                                        pair           closed
    tests/test_recorder_counters_track_on_disk_frames.py     #2388, #2389   #2389
    tests/training/test_checkpoint_cadence_domain.py         #2707, #2708   #2707
    changelog.d/*-g1-send-action-wired.md                    #2766, #2767   open

When two authors choose two names for one change
-----------------------------------------------
The created-path key pairs on a name: a path, or the slug of a changelog
fragment. That is exactly what two authors describing one change need not share.
#2820 and #2822 fixed one defect thirteen minutes apart and wrote
``feetech-broadcast-is-not-a-reply-address`` and
``feetech-motor-id-excludes-the-broadcast``; two names, no collision, and
``--all-open`` reported ``unique-additions`` while both were open (#2823).

So there is a second, weaker key: two branches whose fragments share at least
:data:`FRAGMENT_TOKEN_FLOOR` **words** and which both edit one pre-existing test.
The conjunction is the relation -- neither half is usable alone.

Measured over the 2199 pairs open at the same instant in #2345 through #2825,
against the **eleven** pairs whose closed half names, in its own closing comment,
the pull request that superseded it::

    relation                            pairs  precision  recall  sweeps firing
    both edit a source file                80       8.8%   63.6%          29.9%
    fragments share one word              151       6.6%   90.9%             -
    fragments share two words              33      30.3%   90.9%          37.2%
    both edit one test                     26      11.5%   81.8%             -
    two words AND one test                 14      64.3%   81.8%           5.6%
    created path (the first key)            7      71.4%   45.5%           1.3%
    both keys together                     15      66.7%   90.9%           5.9%

"Sweeps firing" replays ``--all-open`` at each of the 374 moments a pull request
was opened with another already open, because that -- a median of six open pull
requests -- is what the sweep reads, not a corpus.

Two results decided the shape. Pairing on a shared *edited source file*, the
obvious widening and the one #2823 proposed, is 8.8% precise and fires on nearly
a third of sweeps: `strands_robots/simulation/mujoco/simulation.py` alone accounts
for 17 of its 80 pairs. And one shared *word* is the repository's own house style
rather than a subject -- ``names`` and ``the`` appear in 39 and 20 of the window's
401 fragments. Only the conjunction is precise, and it needs no stop list, no
path exclusions and no tuning against the corpus it was measured on.

That eleven is larger than the five this file counted before, because a closing
comment reaches pairs neither key found: #2370/#2373, #2383/#2384 and #2429/#2431
are three duplicate pairs in the earlier window that the created-path table above
does not list. The denominator was undercounted, so the first key's recall was
too -- it is 45.5% of the declared set, not most of it.

The two keys are complementary rather than nested, which is why this is a second
key and not a replacement for the first. The same window holds two *issue-keyed*
pairs -- #2570/#2571 on #2569, and #2480/#2508 on #2466 -- and neither of them
shares an added path, while no claim-free pair claims an issue. Five duplicate
pairs, two reachable from the claim and three from what they create, none from
both.

The 3-of-2002 rate is the whole claim. The relation the sibling file rejected --
widening a path intersection to a test's walked root -- selected 11 of 36 pairs
and named no defect, and a finding attached to a third of the queue reads as
boilerplate.

Why a raw path was not the key
------------------------------
The third row was not reachable when this sweep shipped, and the reason is worth
stating because it was written down as a certainty. A test asserted that a
changelog fragment can never be the shared path, "because its name embeds the
number", and the fixtures carried one to demonstrate it. That is true of the raw
path, and it is exactly why the number has to be dropped: a fragment is named
``<number>-<slug>.md``, so two branches describing one change write one slug under
two numbers and collide on nothing.

The fear behind that exclusion was that keying on a fragment "would fire on every
pair in the queue". Measured, it does not. Of the 353 pull requests, 350 add a
fragment, and between them they use **350 distinct slugs**; exactly two slugs are
used twice, and both times the two users are a duplicate pair. The number is the
noisy half, not the slug: it is meant to be the pull request's own number and in
40 of those 350 it is not, because it is chosen before the pull request exists and
races with whatever merges first.

So the key is the identity a created path declares -- see :func:`addition_key`.
For everything except a fragment that is the path itself, which is why widening it
selected the same two pairs plus one and lost none.

Why this question cannot be asked at intake
-------------------------------------------
``--issue`` is asked before authoring, so it prevents the work. ``--all-open``
cannot be, and not for want of trying: a path set is a property of a *pushed
branch*, and at intake there is no branch to read one from. So this mode caps the
review cost of a collision rather than preventing the authoring -- the same thing
``--pr`` does for a claim, and the reason both live at review while ``--issue``
lives at step 1.

It still arrives early. Both measured pairs opened inside the ~35-minute window
every other observed collision shares -- 14m 41s and 29m 26s apart -- so a sweep
run when the second one opens sees it while the first is still in review.

What this reports, and what it deliberately does not
----------------------------------------------------
``no-claim``
    The pull request links no issue, so it can collide with nothing. Reachable
    only from ``--pr``; an issue is always its own claim.

``unique-claim``
    Nothing else claims it. The convention working.

``duplicate-claim``
    The finding.

``unknown-claims``
    A link set, or the open-pull-request list, could not be read completely. Not a
    finding -- an unreadable field is not evidence of a duplicate.

``--all-open`` reports the same three shapes over the other key:

``unique-additions``
    No two open pull requests create the same file, name one changelog entry, or
    describe one change over one test. The convention working.

``duplicate-addition``
    The finding.

``unknown-additions``
    A file list, or the open-pull-request list, could not be read completely.
    Not a finding, for the same reason.

Out of scope, deliberately:

- **A pull request that claims nothing, in the claim-keyed modes.** Two competing
  branches that both omit the keyword collide invisibly there, and still do:
  requiring a claim is what 18 of the last 30 merges would fail, so neither
  claim-keyed mode demands one. What changed is that the pair is no longer
  unreachable -- ``--all-open`` collides it on what the two branches create
  instead, and the measurement above is the argument that this is narrow enough to
  be worth reporting. See #2709 and #1961.
- **Whether the issue exists, or is already closed.** A stale number is a
  different defect, and refusing it would report a finding against correct work
  whose issue someone else closed first.
- **Draft pull requests are included.** A draft's link is a real claim -- GitHub
  will close the issue when it merges -- so excluding drafts would hide a
  collision for as long as one side stays a draft.

Not wired to CI here, and that is a scope statement rather than an oversight. A
``pull_request`` job running ``--pr`` on ``opened``/``edited`` is the natural
follow-up and would cap the review cost of a collision this query did not
prevent; it needs a credential that can write ``.github/workflows/``, which is a
separate change. The intake half stands on its own: it is the half that stops the
work from being done twice, and it is complete as shipped.

Usage
-----
``--repo``    ``owner/name``. Required in intake mode; in the two review modes it
              defaults to ``$GITHUB_REPOSITORY``. See
              :func:`inferred_repository_refusal` for why intake differs.
``--issue``   an issue number, asked at intake. Exactly one of the three subject
              flags is required.
``--pr``      a pull request number (default: ``$PR_NUMBER``).
``--all-open``
              sweep the open set for two pull requests answering one question,
              on both keys. Takes no
              number, and keeps the inferred repository default for the reason
              ``--pr`` does: its caller is a workflow running where the pull
              requests live. Unlike an issue number, a sweep of the wrong
              repository is visible in its own report, which lists the pull
              requests and paths it read.
``--token``   API token (default: ``$GITHUB_TOKEN``). Needs ``pull-requests: read``.

Exit status is ``1`` for ``duplicate-claim`` or ``duplicate-addition``, else ``0``. A
usage error, including an intake question whose repository was left to be inferred,
exits ``2``.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

API_ROOT = "https://api.github.com"

#: How many linked issues to ask for per pull request. A set longer than this is
#: treated as unreadable rather than as a short list, because a link set cut off
#: here could hide the very issue that collides and report a clean answer.
LINK_PAGE_SIZE = 100

#: How many open pull requests to read per page.
OPEN_PAGE_SIZE = 100

#: A bound on pagination, so a pathological repository cannot spin this job.
#: Reaching it is reported as ``unknown-claims`` -- an unread page is not
#: evidence of no collision.
MAX_OPEN_PAGES = 20

#: How many changed files to ask for per page. A pull request with more is
#: completed by :func:`complete_file_nodes` rather than read short, for the
#: reason :data:`LINK_PAGE_SIZE` gives: a file list cut off here could omit the
#: very path that collides and report clean.
FILE_PAGE_SIZE = 100

#: A bound on file pagination, in the shape :data:`MAX_OPEN_PAGES` uses. A pull
#: request changing more files than this is still refused: a file list cut off
#: here could omit the very path that collides.
MAX_FILE_PAGES = 30

NO_CLAIM = "no-claim"
UNIQUE_CLAIM = "unique-claim"
DUPLICATE_CLAIM = "duplicate-claim"
UNKNOWN_CLAIMS = "unknown-claims"

#: Outcomes of the added-path key. Named apart from the claim-keyed four rather
#: than shared with them, because a report reader has to be able to tell which
#: relation produced a finding: the two have different remedies.
UNIQUE_ADDITIONS = "unique-additions"
DUPLICATE_ADDITION = "duplicate-addition"
UNKNOWN_ADDITIONS = "unknown-additions"

#: The directory whose created files name a change rather than being one.
FRAGMENT_DIR = "changelog.d/"

#: The tree the second key's shared-edit half reads. A pre-existing test is what
#: two authors fixing one defect both correct, and unlike a source file it is
#: rarely edited by two branches for unrelated reasons: pairing on a shared
#: edited test alone selects 26 of the 2199 co-open pairs where a shared edited
#: source file selects 80.
TESTS_DIR = "tests/"


def _load_assembler() -> ModuleType:
    """Load ``assemble_changelog`` from beside this script, for the naming rule.

    By path, registered in :data:`sys.modules` before execution and reusing an
    already-loaded copy: the shape and every reason for it are
    ``scripts/check_changelog_fragment.py``'s :func:`_load_assembler`, which
    reuses this same module for this same rule.
    """
    loaded = sys.modules.get("assemble_changelog")
    if loaded is not None:
        return loaded
    path = Path(__file__).resolve().parent / "assemble_changelog.py"
    spec = importlib.util.spec_from_file_location("assemble_changelog", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the fragment naming rule from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["assemble_changelog"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules["assemble_changelog"]
        raise
    return module


#: The single source of what a fragment name is. Reused rather than restated so
#: this sweep and the assembler cannot disagree about whether a created file
#: under :data:`FRAGMENT_DIR` is a fragment at all -- a name the assembler would
#: reject keys on itself here, which is the conservative direction.
_ASSEMBLER = _load_assembler()


#: The one ``changeType`` this key reads. GitHub's enum also carries
#: ``MODIFIED``, ``REMOVED``, ``RENAMED``, ``COPIED`` and ``CHANGED``, and every
#: one of those describes a file that already exists on the base -- which is the
#: sibling sweep's composition question rather than this one. Reading them here
#: is what turns a 2-of-1802 relation into a 117-of-1802 one.
ADDED_CHANGE_TYPE = "ADDED"

#: The change type the second key reads. A ``MODIFIED`` path is a file that
#: exists on the base, so on its own it is the sibling sweep's composition
#: question -- measured over the 2199 co-open pairs in #2345..#2825, pairing on a
#: shared modified path under ``strands_robots/`` selects 80 of them at 8.8%
#: precision and fires on 29.9% of replayed sweeps. It carries the second key
#: only in conjunction with :data:`FRAGMENT_TOKEN_FLOOR`, which is what makes the
#: pair precise: 14 selected, 64.3% precise.
EDITED_CHANGE_TYPE = "MODIFIED"

#: How many words two changelog slugs must share before the pair is reported.
#: :func:`addition_key` already treats a slug as a name for a piece of work, and
#: two authors describing one change write two names for it rather than one -- so
#: the exact-slug key misses them. Words rather than the whole slug is the
#: weakening that reaches them; two rather than one is what keeps it usable.
#:
#: Measured over the same 2199 pairs, against the 11 pairs whose closed half says
#: in its own closing comment which pull request superseded it:
#:
#: ===============================  ========  =========  ======  =============
#: relation                         selected  precision  recall  sweeps firing
#: ===============================  ========  =========  ======  =============
#: one shared word                       151       6.6%   90.9%          n/a
#: two shared words                       33      30.3%   90.9%         37.2%
#: three shared words                     13      53.8%   63.6%          8.0%
#: two shared words + shared test         14      64.3%   81.8%          5.6%
#: ===============================  ========  =========  ======  =============
#:
#: One word is the repository's own house style rather than a subject -- ``names``
#: and ``the`` appear in 39 and 20 of the 401 fragments in the window, so a
#: one-word rule fires on more than a third of sweeps. Three words loses a third
#: of the duplicates. Two words *and* a shared pre-existing test is the pair that
#: holds: neither half is usable alone and together they are 64.3% precise.
FRAGMENT_TOKEN_FLOOR = 2

#: ``closingIssuesReferences`` is GraphQL-only -- no REST field carries it.
_SELF_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $links: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number
      closingIssuesReferences(first: $links) {
        totalCount
        nodes { number }
      }
    }
  }
}
"""

_OPEN_QUERY = """
query($owner: String!, $name: String!, $links: Int!, $open: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: $open, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        closingIssuesReferences(first: $links) {
          totalCount
          nodes { number }
        }
      }
    }
  }
}
"""


#: Read through ``repository.pullRequests(states: OPEN)`` for the reason the
#: claim query is: ``search`` is eventually consistent, and a pull request opened
#: seconds ago is exactly the row this sweep exists to find. Drafts are included,
#: matching this file's claim-keyed policy rather than the sibling sweep's -- a
#: draft's new file is authored work whatever its merge state, so excluding one
#: would hide a collision for as long as either side stayed a draft.
#:
#: ``changeType`` is GraphQL-only: the REST file list spells the same thing
#: ``status``, in lower case.
_ADDITIONS_QUERY = """
query($owner: String!, $name: String!, $files: Int!, $open: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: OPEN, first: $open, after: $after) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number
        files(first: $files) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes { path changeType }
        }
      }
    }
  }
}
"""


class ClaimSetUnreadable(RuntimeError):
    """A link set, or the list of open pull requests, could not be read.

    Named apart from the builtin to keep the distinction explicit at the call
    site: this is "GitHub did not answer", not "the answer was empty". The two
    must not collapse, because an empty answer is a pass and an unanswered one is
    neither a pass nor a finding.
    """


@dataclass(frozen=True)
class Verdict:
    """The outcome and the claim sets it was computed from."""

    outcome: str
    #: The issues the pull request under test links.
    claimed: tuple[int, ...] = ()
    #: ``(issue, the other open pull requests linking it)``, sorted by issue.
    collisions: tuple[tuple[int, tuple[int, ...]], ...] = ()
    #: How many other open pull requests were compared against.
    scanned: int = 0
    detail: str = ""

    @property
    def is_finding(self) -> bool:
        return self.outcome == DUPLICATE_CLAIM

    @property
    def rivals(self) -> tuple[int, ...]:
        """Every other open pull request implicated, sorted and deduplicated."""
        return tuple(sorted({pr for _, prs in self.collisions for pr in prs}))

    @property
    def summary(self) -> str:
        if self.outcome == NO_CLAIM:
            return "This pull request links no issue, so it cannot duplicate another's claim."
        if self.outcome == UNIQUE_CLAIM:
            # Worded without a subject so it reads correctly for both entry points:
            # an issue checked at intake has no pull request to call "this" yet.
            return (
                f"{_issues(self.claimed)} "
                f"{'are' if len(self.claimed) > 1 else 'is'} claimed by no open pull request "
                f"({self.scanned} compared)."
            )
        if self.outcome == UNKNOWN_CLAIMS:
            return (
                f"Could not read every claim: {self.detail} Not treated as a finding, because "
                "an unreadable link set is not evidence that an issue is claimed twice."
            )
        # Semicolons rather than :func:`_join`, which would put an "and" between two
        # clauses that already each contain one ("... by #5 and #6 and #30 is ...").
        parts = "; ".join(f"{_issues((issue,))} is also claimed by {_pulls(prs)}" for issue, prs in self.collisions)
        return (
            f"{parts}. Two pull requests closing one issue cannot both "
            "land as written, and whichever loses the race spends a review approval on a "
            "change that will be closed."
        )


@dataclass(frozen=True)
class PullFiles:
    """The paths one pull request creates and the paths it edits.

    Both sets come from one file list, so they are read together rather than by
    two walks of the same response: the sweep's two keys ask different questions
    of the same data.
    """

    #: Paths whose ``changeType`` is :data:`ADDED_CHANGE_TYPE`, sorted.
    created: tuple[str, ...] = ()
    #: Paths whose ``changeType`` is :data:`EDITED_CHANGE_TYPE`, sorted.
    edited: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdditionVerdict:
    """The outcome of the added-path sweep and the pairs it was computed from."""

    outcome: str
    #: ``(left, right, the paths both branches create)``. Ascending in all three
    #: axes, so a diff of two reports shows changed verdicts rather than
    #: reordered rows.
    collisions: tuple[tuple[int, int, tuple[str, ...]], ...] = ()
    #: ``(left, right, the words both slugs use, the tests both branches edit)``
    #: for the second key. Kept apart from :attr:`collisions` because the two
    #: support different conclusions: a shared created path means one of the two
    #: branches is redundant, and a shared description over a shared test means
    #: they may be answering one question -- which is a pair to read, not a fact.
    echoes: tuple[tuple[int, int, tuple[str, ...], tuple[str, ...]], ...] = ()
    #: How many open pull requests were read.
    scanned: int = 0
    detail: str = ""

    @property
    def is_finding(self) -> bool:
        return self.outcome == DUPLICATE_ADDITION

    @property
    def compared(self) -> int:
        """How many pairs the sweep computed, which is what it looked at."""
        return self.scanned * (self.scanned - 1) // 2

    @property
    def implicated(self) -> tuple[int, ...]:
        """Every pull request either key reports, sorted and deduplicated."""
        pairs = [(left, right) for left, right, _ in self.collisions]
        pairs += [(left, right) for left, right, _, _ in self.echoes]
        return tuple(sorted({number for pair in pairs for number in pair}))

    @property
    def summary(self) -> str:
        if self.outcome == UNKNOWN_ADDITIONS:
            return (
                f"Could not read every file list: {self.detail} Not treated as a finding, because "
                "an unreadable file list is not evidence that two branches create one thing."
            )
        if self.outcome == UNIQUE_ADDITIONS:
            return (
                f"No two of the {self.scanned} open pull requests create the same file, name the "
                f"same changelog entry, or describe one change over one test "
                f"({self.compared} pair(s) compared)."
            )
        parts = [f"{_pulls((left, right))} both create {_paths(paths)}" for left, right, paths in self.collisions]
        parts += [
            f"{_pulls((left, right))} describe their change with {_paths(words)} and both edit {_paths(tests)}"
            for left, right, words, tests in self.echoes
        ]
        return (
            f"{'; '.join(parts)}. Two branches creating one file, two fragments declaring one "
            "changelog entry, or two descriptions of one change over one test, are two answers to "
            "one question rather than a composition to verify, and whichever is closed spends a "
            "review approval on a change that will not ship."
        )


def _join(parts: Sequence[str]) -> str:
    """Render a clause list as ``a`` / ``a and b`` / ``a, b and c``."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _issues(numbers: Sequence[int]) -> str:
    """Render an issue list as ``#1`` / ``#1 and #2`` / ``#1, #2 and #3``."""
    return _join([f"#{n}" for n in numbers]) or "no issue"


def _pulls(numbers: Sequence[int]) -> str:
    """Render a pull-request list the same way, for the report's other half."""
    return _join([f"#{n}" for n in numbers]) or "no pull request"


def _paths(paths: Sequence[str]) -> str:
    """Render a path list the same way, backquoted so a report reads them as code."""
    return _join([f"`{path}`" for path in paths]) or "no file"


def find_collisions(
    claimed: Sequence[int], others: Mapping[int, Sequence[int]]
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return each claimed issue that another open pull request also claims.

    Deterministic in both axes -- issues ascending, and the pull requests
    claiming each ascending -- so the report reads the same on a re-run.
    """
    collisions = []
    for issue in sorted(set(claimed)):
        rivals = tuple(sorted(pr for pr, links in others.items() if issue in set(links)))
        if rivals:
            collisions.append((issue, rivals))
    return tuple(collisions)


def addition_key(path: str) -> str:
    """Return the identity a created path declares, which is what collides.

    Almost every created path *is* its own identity: two branches writing
    ``tests/foo.py`` have written one file, and the path is what says so.

    A changelog fragment is the exception, and it is the only one. Its name is
    ``<number>-<slug>.md`` by convention, so two branches describing the same
    change write the same slug under two numbers and their paths differ -- while
    the entry they are each declaring is the same entry. Measured over
    #2345..#2767: 350 of the 353 pull requests add a fragment, between them using
    **350 distinct slugs**, and the only two slugs used twice are #2388/#2389 and
    #2766/#2767 -- both duplicate pairs. So a slug is in practice a name for a
    piece of work, and two of them colliding is the thing this sweep looks for.

    The number is dropped rather than compared because it carries no signal: it is
    meant to be the pull request's own number and in 40 of those 350 it is not,
    since a number is chosen before the pull request is opened and races with
    whatever merges first.

    Returned as a glob (``changelog.d/*-<slug>.md``) so the report names what the
    two branches share without naming a file that exists on neither. A name the
    assembler would not accept as a fragment -- a reserved ``README.md``, a
    nested path, a slug that is not lowercase-and-hyphens -- keys on itself, which
    is the conservative direction: it can only fail to report a pair, never invent
    one.
    """
    if not path.startswith(FRAGMENT_DIR):
        return path
    name = path[len(FRAGMENT_DIR) :]
    # Checked before the pattern rather than left to it. The pattern happens to
    # reject today's only reserved name, but only because ``README.md`` has no
    # leading digits; a reserved name that did would match it, and the assembler's
    # list is the authority on which names are not fragments. A nested path needs
    # no such check -- ``/`` is outside every character class in the pattern.
    if name in _ASSEMBLER.RESERVED_NAMES:
        return path
    match = _ASSEMBLER.FRAGMENT_NAME.match(name)
    if match is None:
        return path
    return f"{FRAGMENT_DIR}*-{name[match.end('number') + 1 :]}"


def find_addition_collisions(
    additions: Mapping[int, Sequence[str]],
) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    """Return every pair of open pull requests that creates the same thing.

    Paired on :func:`addition_key` rather than on the raw path, so a pair whose
    only shared creation is a changelog entry under two fragment numbers is
    reported. For every other path the key is the path, so this is a strict
    widening: measured over the 2002 co-open pairs in #2345..#2767 it selects the
    same 2 pairs the raw path selects, plus #2766/#2767, and loses none.

    Every pair is compared rather than only adjacent ones: two runs a few minutes
    apart usually get consecutive numbers, but nothing guarantees it, and a
    relation that holds for a pair is not a property of their distance.

    Deterministic in all three axes -- the pairs by lower then higher number, and
    each shared list sorted -- for the reason :func:`find_collisions` gives.
    """
    ordered = sorted(additions)
    found: list[tuple[int, int, tuple[str, ...]]] = []
    for left, right in itertools.combinations(ordered, 2):
        shared = tuple(
            sorted({addition_key(path) for path in additions[left]} & {addition_key(path) for path in additions[right]})
        )
        if shared:
            found.append((left, right, shared))
    return tuple(found)


def fragment_tokens(paths: Sequence[str]) -> frozenset[str]:
    """Return the words the changelog fragments among ``paths`` use.

    :func:`addition_key` already reads a fragment's slug as a name for a piece of
    work. This reads the same slug one notch weaker, as the *words* of that name,
    which is what two authors describing one change have in common when they did
    not happen to choose the same name for it: #2820 wrote
    ``feetech-broadcast-is-not-a-reply-address`` and #2822 wrote
    ``feetech-motor-id-excludes-the-broadcast`` for one defect, so the exact-slug
    key sees nothing and the words ``feetech`` and ``broadcast`` are shared.

    A path that is not a fragment contributes nothing, and a fragment whose name
    the assembler would reject contributes nothing either -- the same
    conservative direction :func:`addition_key` takes, for the same reason: this
    can only fail to report a pair, never invent one.

    No stop list. Measured over #2345..#2825 a frequency-derived one is a
    refinement rather than a requirement (43.5% precision against 30.3% at the
    same recall), and it cannot be computed where this runs: the sweep sees the
    open set, which is a median of 6 pull requests, not a corpus. What replaces
    it is the conjunction in :func:`find_echo_collisions`, which is both
    parameter-free and more precise than any stop list measured here.
    """
    words: set[str] = set()
    for path in paths:
        if not path.startswith(FRAGMENT_DIR):
            continue
        name = path[len(FRAGMENT_DIR) :]
        if name in _ASSEMBLER.RESERVED_NAMES:
            continue
        match = _ASSEMBLER.FRAGMENT_NAME.match(name)
        if match is None:
            continue
        words.update(name[match.end("number") + 1 : -len(".md")].split("-"))
    return frozenset(words)


def find_echo_collisions(
    files: Mapping[int, PullFiles],
) -> tuple[tuple[int, int, tuple[str, ...], tuple[str, ...]], ...]:
    """Return every pair of open pull requests that describes one change twice.

    Two conditions, and the conjunction is the whole relation:

    * their changelog fragments share at least :data:`FRAGMENT_TOKEN_FLOOR`
      words, so both branches are describing the same subject; and
    * both edit at least one pre-existing file under :data:`TESTS_DIR`, so both
      are correcting the same case rather than writing about the same area.

    Either half alone is unusable. Over the 2199 co-open pairs in #2345..#2825,
    two shared words selects 33 pairs and fires on 37.2% of replayed sweeps --
    the repository names its changes in a house style, so ``names`` and ``the``
    collide constantly. A shared edited test selects 26. Together they select 14,
    of which 9 are among the 11 pairs whose closed half names its supersedor:
    64.3% precision at 81.8% recall, firing on 5.6% of sweeps.

    This is a strictly weaker signal than :func:`find_addition_collisions`, and
    that is why the two are reported apart. A shared created path is a fact about
    the branches -- neither file exists on the base, so one of the two is
    redundant. A shared description over a shared test is a question: of the five
    pairs it selects that are not declared duplicates, two are members of the
    #2785/#2787/#2790/#2792 supersede cluster and one (#2713/#2714) is two
    branches on one subject that both shipped. So the remedy is to read the other
    pull request, not to close one.

    Deterministic in all four axes for the reason :func:`find_collisions` gives.
    """
    ordered = sorted(files)
    found: list[tuple[int, int, tuple[str, ...], tuple[str, ...]]] = []
    for left, right in itertools.combinations(ordered, 2):
        words = fragment_tokens(files[left].created) & fragment_tokens(files[right].created)
        if len(words) < FRAGMENT_TOKEN_FLOOR:
            continue
        tests = {path for path in files[left].edited if path.startswith(TESTS_DIR)} & {
            path for path in files[right].edited if path.startswith(TESTS_DIR)
        }
        if not tests:
            continue
        found.append((left, right, tuple(sorted(words)), tuple(sorted(tests))))
    return tuple(found)


def classify_additions(files: Mapping[int, PullFiles] | None, detail: str = "") -> AdditionVerdict:
    """Decide which of the three states the open set is in, on both keys.

    ``None`` means the set could not be read, which is its own outcome for the
    reason :func:`classify` gives: a silent API or permission change must not be
    able to turn this sweep into a no-op that always agrees.

    Both keys are computed from one argument rather than from an optional second
    mapping. An ``edits`` that defaulted to empty would let the second key report
    nothing while looking like it had run, which is the failure mode this module
    refuses everywhere else: an unread set and an empty one must not collapse.
    """
    if files is None:
        return AdditionVerdict(UNKNOWN_ADDITIONS, (), (), 0, detail)
    collisions = find_addition_collisions({number: pull.created for number, pull in files.items()})
    echoes = find_echo_collisions(files)
    return AdditionVerdict(
        DUPLICATE_ADDITION if collisions or echoes else UNIQUE_ADDITIONS,
        collisions,
        echoes,
        len(files),
    )


def classify(
    claimed: Sequence[int] | None,
    others: Mapping[int, Sequence[int]] | None,
    detail: str = "",
) -> Verdict:
    """Decide which of the four states this pull request is in.

    Either argument being ``None`` means that half could not be read, which is
    its own outcome and folded into neither a pass nor a finding: a silent API or
    permission change must not be able to turn this check into a no-op that
    always agrees, nor into one that accuses every branch.
    """
    if claimed is None or others is None:
        return Verdict(UNKNOWN_CLAIMS, tuple(sorted(set(claimed or ()))), (), 0, detail)
    claimed_sorted = tuple(sorted(set(claimed)))
    if not claimed_sorted:
        return Verdict(NO_CLAIM, (), (), len(others))
    collisions = find_collisions(claimed_sorted, others)
    if not collisions:
        return Verdict(UNIQUE_CLAIM, claimed_sorted, (), len(others))
    return Verdict(DUPLICATE_CLAIM, claimed_sorted, collisions, len(others))


def _post(query: str, variables: dict[str, object], token: str) -> object:
    request = urllib.request.Request(
        f"{API_ROOT}/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "strands-robots-check-duplicate-claim",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed API host
        return json.load(response)


def _repository(payload: object) -> dict[str, object]:
    """Return the ``repository`` object, or refuse the answer."""
    if not isinstance(payload, dict):
        raise ClaimSetUnreadable("the API response was not a JSON object.")
    if payload.get("errors"):
        raise ClaimSetUnreadable(f"the API returned errors: {json.dumps(payload['errors'])[:400]}")
    repository = (payload.get("data") or {}).get("repository")
    if not isinstance(repository, dict):
        raise ClaimSetUnreadable("the response carried no repository.")
    return repository


def link_numbers(pull: Mapping[str, object]) -> tuple[int, ...]:
    """Return the issues one pull-request node links, refusing a truncated set.

    A set cut off at :data:`LINK_PAGE_SIZE` could omit the issue that collides,
    so it is refused rather than read short -- the one thing this check must never
    do is report clean because it did not look far enough.
    """
    references = pull.get("closingIssuesReferences") or {}
    if not isinstance(references, dict):
        raise ClaimSetUnreadable(f"#{pull.get('number')} carried no link set.")
    nodes = references.get("nodes") or []
    total = references.get("totalCount")
    if isinstance(total, int) and total > len(nodes):
        raise ClaimSetUnreadable(f"#{pull.get('number')}'s link set is truncated ({total} links, {len(nodes)} read).")
    return tuple(sorted({int(node["number"]) for node in nodes if isinstance(node, dict) and "number" in node}))


def file_sets(pull: Mapping[str, object]) -> PullFiles:
    """Return the paths one pull-request node creates and edits, refusing a truncated list.

    The two change types the sweep's two keys read, split out of one file list.
    Every other ``changeType`` GitHub's enum carries -- ``REMOVED``, ``RENAMED``,
    ``COPIED``, ``CHANGED`` -- is dropped: a removal is not work either key asks
    about, and a rename is the sibling sweep's composition question.

    A list cut off at :data:`FILE_PAGE_SIZE` is refused rather than read short,
    exactly as :func:`link_numbers` refuses a truncated link set -- the one thing
    this sweep must never do is report clean because it did not look far enough.
    Refused once here rather than once per key, so neither key can be computed
    from a prefix while the other is refused.
    """
    files = pull.get("files") or {}
    if not isinstance(files, dict):
        raise ClaimSetUnreadable(f"#{pull.get('number')} carried no file list.")
    nodes = files.get("nodes") or []
    total = files.get("totalCount")
    if isinstance(total, int) and total > len(nodes):
        raise ClaimSetUnreadable(f"#{pull.get('number')}'s file list is truncated ({total} files, {len(nodes)} read).")

    def paths(change_type: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    str(node["path"])
                    for node in nodes
                    if isinstance(node, dict) and node.get("changeType") == change_type and node.get("path")
                }
            )
        )

    return PullFiles(created=paths(ADDED_CHANGE_TYPE), edited=paths(EDITED_CHANGE_TYPE))


def resolve_claim(repo: str, pr: int, token: str) -> tuple[int, ...]:
    """Return the issues the pull request under test links.

    Read by number rather than from the open list, so the answer does not depend
    on the pull request having been indexed or on it fitting the first page.
    """
    owner, _, name = repo.partition("/")
    if not owner or not name:
        raise ClaimSetUnreadable(f"repository {repo!r} is not in owner/name form.")
    repository = _repository(
        _post(_SELF_QUERY, {"owner": owner, "name": name, "number": pr, "links": LINK_PAGE_SIZE}, token)
    )
    pull = repository.get("pullRequest")
    if not isinstance(pull, dict):
        raise ClaimSetUnreadable(f"no pull request {repo}#{pr} in the response.")
    return link_numbers(pull)


def resolve_open_claims(repo: str, token: str, pr: int | None = None) -> dict[int, tuple[int, ...]]:
    """Return ``{open pull request number: the issues it links}``, excluding ``pr``.

    ``pr`` of ``None`` excludes nothing, which is the intake question: *is anything
    already claiming this issue?* There is no pull request to leave out yet.

    Paginates rather than reading one page: a repository with more open pull
    requests than :data:`OPEN_PAGE_SIZE` would otherwise get a clean answer
    computed from a prefix of the set.
    """
    owner, _, name = repo.partition("/")
    if not owner or not name:
        raise ClaimSetUnreadable(f"repository {repo!r} is not in owner/name form.")
    claims: dict[int, tuple[int, ...]] = {}
    cursor: str | None = None
    for _ in range(MAX_OPEN_PAGES):
        repository = _repository(
            _post(
                _OPEN_QUERY,
                {
                    "owner": owner,
                    "name": name,
                    "links": LINK_PAGE_SIZE,
                    "open": OPEN_PAGE_SIZE,
                    "after": cursor,
                },
                token,
            )
        )
        page = repository.get("pullRequests")
        if not isinstance(page, dict):
            raise ClaimSetUnreadable("the response carried no open pull requests.")
        for node in page.get("nodes") or []:
            if not isinstance(node, dict) or "number" not in node:
                continue
            number = int(node["number"])
            if pr is not None and number == pr:
                continue
            claims[number] = link_numbers(node)
        info = page.get("pageInfo") or {}
        if not info.get("hasNextPage"):
            return claims
        cursor = info.get("endCursor")
        if not isinstance(cursor, str):
            raise ClaimSetUnreadable("the open pull request list is paged but carried no cursor.")
    raise ClaimSetUnreadable(f"more than {MAX_OPEN_PAGES * OPEN_PAGE_SIZE} open pull requests; the list was truncated.")


_PULL_FILES_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $files: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      files(first: $files, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes { path changeType }
      }
    }
  }
}
"""


def complete_file_nodes(owner: str, name: str, pull: dict[str, object], token: str) -> None:
    """Read the rest of one pull request's changed-file list into ``pull``, in place.

    The first page arrives with the open-pull-request page it was requested
    beside; a pull request changing more files than :data:`FILE_PAGE_SIZE`
    carries a cursor, and the pages after it are read here so
    :func:`file_sets` is handed the whole list rather than a prefix of it.

    Refusing a long list instead is safe for that pull request and unsafe for
    the sweep. :func:`resolve_open_file_sets` resolves every open pull request
    before any pair is compared, so one refusal answers ``unknown-additions``
    for the whole board -- reporting nothing about any other pair *because it
    did not look far enough*, which is the one thing :func:`file_sets` says
    this sweep must never do.

    Bounded by :data:`MAX_FILE_PAGES`, so the ceiling moves rather than
    disappearing: a list longer than that is still refused.
    """
    files = pull.get("files")
    if not isinstance(files, dict):
        return
    nodes = files.get("nodes")
    if not isinstance(nodes, list):
        return
    info: object = files.get("pageInfo") or {}
    for _ in range(MAX_FILE_PAGES - 1):
        if not isinstance(info, dict) or not info.get("hasNextPage"):
            return
        cursor = info.get("endCursor")
        if not isinstance(cursor, str):
            raise ClaimSetUnreadable(f"#{pull.get('number')}'s file list is paged but carried no cursor.")
        repository = _repository(
            _post(
                _PULL_FILES_QUERY,
                {
                    "owner": owner,
                    "name": name,
                    "number": int(pull["number"]),  # type: ignore[arg-type]
                    "files": FILE_PAGE_SIZE,
                    "after": cursor,
                },
                token,
            )
        )
        node = repository.get("pullRequest")
        more = node.get("files") if isinstance(node, dict) else None
        if not isinstance(more, dict):
            raise ClaimSetUnreadable(f"#{pull.get('number')}'s file list stopped short of its own total.")
        nodes.extend(entry for entry in (more.get("nodes") or []) if isinstance(entry, dict))
        info = more.get("pageInfo") or {}
    raise ClaimSetUnreadable(
        f"#{pull.get('number')} changes more than {MAX_FILE_PAGES * FILE_PAGE_SIZE} files; the list was truncated."
    )


def resolve_open_file_sets(repo: str, token: str) -> dict[int, PullFiles]:
    """Return ``{open pull request number: the paths it creates and edits}``.

    Nothing is excluded: unlike the claim-keyed review question there is no pull
    request "under test" here, so there is no self-comparison to leave out. The
    subject is the set.

    Paginated for the reason :func:`resolve_open_claims` is -- a repository with
    more open pull requests than :data:`OPEN_PAGE_SIZE` would otherwise get a
    clean answer computed from a prefix of the set. Each node's own file list is
    completed by :func:`complete_file_nodes` before it is read, so one pull
    request changing more files than :data:`FILE_PAGE_SIZE` cannot answer
    ``unknown-additions`` for every pair on the board.
    """
    owner, _, name = repo.partition("/")
    if not owner or not name:
        raise ClaimSetUnreadable(f"repository {repo!r} is not in owner/name form.")
    files: dict[int, PullFiles] = {}
    cursor: str | None = None
    for _ in range(MAX_OPEN_PAGES):
        repository = _repository(
            _post(
                _ADDITIONS_QUERY,
                {
                    "owner": owner,
                    "name": name,
                    "files": FILE_PAGE_SIZE,
                    "open": OPEN_PAGE_SIZE,
                    "after": cursor,
                },
                token,
            )
        )
        page = repository.get("pullRequests")
        if not isinstance(page, dict):
            raise ClaimSetUnreadable("the response carried no open pull requests.")
        for node in page.get("nodes") or []:
            if not isinstance(node, dict) or "number" not in node:
                continue
            complete_file_nodes(owner, name, node, token)
            files[int(node["number"])] = file_sets(node)
        info = page.get("pageInfo") or {}
        if not info.get("hasNextPage"):
            return files
        cursor = info.get("endCursor")
        if not isinstance(cursor, str):
            raise ClaimSetUnreadable("the open pull request list is paged but carried no cursor.")
    raise ClaimSetUnreadable(f"more than {MAX_OPEN_PAGES * OPEN_PAGE_SIZE} open pull requests; the list was truncated.")


def render_additions(verdict: AdditionVerdict, repo: str) -> str:
    """Render the added-path sweep's report.

    Each multi-line paragraph is a named local joined with explicit ``+``, the
    convention the sibling sweep's renderer states: implicit concatenation inside
    a list of report lines is indistinguishable from a forgotten comma.
    """
    lines = [
        "## Duplicate work - two branches answering one question",
        "",
        f"Outcome: **{verdict.outcome}**",
        "",
        verdict.summary,
        "",
        "| field | value |",
        "|---|---|",
        f"| repository | {repo} |",
        f"| open pull requests read | {verdict.scanned} |",
        f"| pairs compared | {verdict.compared} |",
    ]
    if not verdict.is_finding:
        return "\n".join(lines)
    lines += [f"| pull requests implicated | {_pulls(verdict.implicated)} |"]
    if verdict.collisions:
        lines += ["", "| pull requests | what both create |", "|---|---|"]
        for left, right, paths in verdict.collisions:
            lines.append(f"| #{left} + #{right} | {_paths(paths)} |")
        created = (
            "Neither branch's file exists on the base, so this is not a merge order to decide: "
            + "the two are answering the same question, and the second one to be read is work that "
            + "was already done. Read the other pull request before continuing with either."
        )
        lines += ["", "### What this means - a created path", "", created]
    if verdict.echoes:
        lines += ["", "| pull requests | words both descriptions use | tests both edit |", "|---|---|---|"]
        for left, right, words, tests in verdict.echoes:
            lines.append(f"| #{left} + #{right} | {_paths(words)} | {_paths(tests)} |")
        described = (
            "These two name their change with the same words and correct the same pre-existing "
            + "test, which is what two authors fixing one defect do when they did not happen to "
            + "choose the same file name for the fix. This is the weaker of the two keys and it "
            + "reports a pair to read rather than a fact: measured over #2345..#2825 it is 64.3% "
            + "precise, so roughly one pair in three is two branches that share a subject without "
            + "sharing a change. Read both descriptions before continuing with either."
        )
        lines += ["", "### What this means - one change described twice", "", described]
    clears = (
        "Close whichever of the two is redundant, or -- if both are wanted -- change one so it "
        + "no longer creates the same path, names the same changelog entry, or describes the same "
        + "change over the same test, which is the case where the shared name was the accident "
        + "rather than the work. "
        + "This is a report rather than a branch-clearable gate: "
        + "the remedy is a decision between two authors, and no push by one of them settles it."
    )
    blind = (
        "Neither pull request need have claimed an issue for this to be reported, which is the "
        + "point of these keys: measured over #2345..#2825, of the eleven pairs whose closed half "
        + "names the pull request that superseded it, the created-path key reaches five and the "
        + "described-change key reaches nine, for ten of eleven between them."
    )
    lines += ["", "### What clears this", "", clears, "", blind]
    return "\n".join(lines)


def render(verdict: Verdict, repo: str, pr: int | None = None, issue: int | None = None) -> str:
    """Render the report for this run.

    Exactly one of ``pr`` and ``issue`` names the subject: a pull request whose
    claims were compared, or -- at intake -- an issue checked before a pull
    request for it exists.
    """
    subject = f"| pull request | {repo}#{pr} |" if pr is not None else f"| issue | {repo}#{issue} |"
    lines = [
        "## Duplicate closing claim",
        "",
        f"Outcome: **{verdict.outcome}**",
        "",
        verdict.summary,
        "",
        "| field | value |",
        "|---|---|",
        subject,
        f"| issues it closes | {_issues(verdict.claimed)} |",
    ]
    # Only stated when a comparison happened. A pull request claiming nothing
    # short-circuits before the open list is read, and printing "0 compared"
    # there would read as "the check could not see the other pull requests".
    if verdict.outcome != NO_CLAIM:
        lines.append(f"| other open pull requests compared | {verdict.scanned} |")
    if verdict.is_finding and issue is not None:
        lines += [
            f"| already claimed by | {_pulls(verdict.rivals)} |",
            "",
            "### What this means",
            "",
            f"{_pulls(verdict.rivals)} is already open against {_issues((issue,))}. Read it before",
            "starting: if it does the work, there is nothing to author, and if it does not, say so",
            "there rather than opening a second pull request against the same issue. If a competing",
            "implementation is wanted on purpose, exactly one of the two should claim the close and",
            "the other should cross-reference instead (`per #N`, `towards #N`).",
        ]
    elif verdict.is_finding:
        lines += [
            f"| also claimed by | {_pulls(verdict.rivals)} |",
            "",
            "### What clears this",
            "",
            "1. Decide which pull request closes the issue. The other drops the keyword from",
            "   its description -- `per #N`, `follow-up to #N`, `towards #N` all still",
            "   cross-reference without linking. GitHub drops the link when the description is",
            "   saved and this check re-runs on `edited`, so either side can clear its own run",
            "   without a push.",
            "2. If one of the two is redundant, close it.",
            "",
            "Both pull requests are named above rather than one being blamed: measured over the",
            "last 100 pull requests, the newer of a duplicate pair is the one that merged in two",
            "of the three cases, so which to keep is not something this check can decide.",
        ]
    return "\n".join(lines)


def inferred_repository_refusal(inferred: str) -> str:
    """Return why an inferred repository cannot answer an intake question.

    ``$GITHUB_REPOSITORY`` names the repository a command is *running in*. For the
    ``--pr`` mode that is the right answer by construction -- a workflow reviewing a
    pull request runs in the repository the pull request lives in, which is why the
    default is kept there. Intake runs *before* any pull request exists, so it is a
    local invocation by whoever is about to do the work, and nothing ties their
    working directory to the repository the issue belongs to.

    The failure that follows is silent rather than loud, which is why this is a
    refusal rather than a warning. The check reads a different repository's open
    pull requests, finds none of them claiming that number, and reports
    ``unique-claim`` with exit ``0``. Nothing in that report distinguishes it from
    the answer to the question that was meant, and it only misleads in the
    reassuring direction: a spurious collision would be investigated and found
    nonexistent, whereas a missed one is invisible.

    Nor can the substitution be detected after the fact. An issue number alone does
    not name a repository, so there is no second source to compare the resolved one
    against, and issue numbers are dense enough that an unrelated repository very
    often has one at the same number -- which is also why confirming the issue
    *exists* would not be a reliable substitute for naming the repository, on top of
    reversing this script's deliberate decision not to read the issue at all.
    """
    return (
        "intake mode must name the repository: pass --repo owner/name. The environment "
        f"infers {inferred!r}, which is where this command is running rather than "
        "necessarily the repository the issue belongs to, and an intake check that reads "
        "the wrong repository reports no duplicate."
    )


def _emit(report: str) -> None:
    """Print the report and append it to the CI job summary when there is one."""
    print(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")


def _run_addition_sweep(repo: str, token: str) -> int:
    """Sweep the open set on both keys: one created path, or one change described twice."""
    files: dict[int, PullFiles] | None
    detail = ""
    try:
        files = resolve_open_file_sets(repo, token)
    except (ClaimSetUnreadable, urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as exc:
        files, detail = None, str(exc)
        print(f"check_duplicate_claim: {detail}", file=sys.stderr)

    verdict = classify_additions(files, detail)
    _emit(render_additions(verdict, repo))
    if verdict.is_finding:
        print(f"::error title=Two open pull requests answer one question::{verdict.summary}")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ``None`` rather than the environment value, so "was it supplied?" stays
    # answerable after parsing -- the intake refusal below turns on that, not on
    # what the repository resolves to.
    parser.add_argument("--repo", default=None)
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", ""))
    parser.add_argument("--issue", default="")
    parser.add_argument("--all-open", action="store_true")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    args = parser.parse_args(argv)

    inferred = os.environ.get("GITHUB_REPOSITORY", "")
    repo = inferred if args.repo is None else args.repo

    if not repo:
        parser.error("--repo is required (or set GITHUB_REPOSITORY)")
    if [bool(args.pr), bool(args.issue), args.all_open].count(True) != 1:
        parser.error(
            "pass exactly one of --pr (review a pull request's claims), --issue (intake) "
            "and --all-open (sweep the open set for two branches answering one question)"
        )
    if args.repo is None and args.issue:
        parser.error(inferred_repository_refusal(repo))
    if not args.token:
        parser.error("--token is required (or set GITHUB_TOKEN)")

    if args.all_open:
        return _run_addition_sweep(repo, args.token)

    pr = int(args.pr) if args.pr else None
    issue = int(args.issue) if args.issue else None
    claimed: tuple[int, ...] | None
    others: dict[int, tuple[int, ...]] | None
    detail = ""
    try:
        # In intake mode the "claim" is the issue being considered, and nothing is
        # excluded from the comparison because no pull request for it exists yet.
        claimed = (issue,) if issue is not None else resolve_claim(repo, int(args.pr), args.token)
        others = resolve_open_claims(repo, args.token, pr) if claimed else {}
    except (ClaimSetUnreadable, urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as exc:
        claimed, others, detail = None, None, str(exc)
        print(f"check_duplicate_claim: {detail}", file=sys.stderr)

    verdict = classify(claimed, others, detail)
    _emit(render(verdict, repo, pr, issue))

    if verdict.is_finding:
        print(f"::error title=Two open pull requests claim one issue::{verdict.summary}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
