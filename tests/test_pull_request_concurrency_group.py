"""Contract pins for the concurrency group of every workflow a pull_request can start.

A concurrency group is per workflow: ``pr-and-push.yml`` keys its own on
``github.workflow``, so cancelling a superseded run of the required check says
nothing about the ten other workflows a pull request starts. Each of those has to
declare its own group or it declares none, and five of them declared none -
``codeql.yml``, ``dependency-review.yml``, ``breaking-change-check.yml``,
``agent-api-check.yml`` and ``llm-input-safety.yml`` - while six already had one.

What a missing group costs is not a wrong answer but a run of the *right* answer
to a question nobody asked. A push to a pull request replaces the head sha these
jobs read, so a run still going over the previous sha can only recompute a verdict
about a commit no reviewer will open - and it recomputes it on a runner the
current head is queued behind.

Measured 2026-08-19 over the last 100 ``pull_request`` runs of each of the five,
from job-level ``started_at`` / ``completed_at`` rather than the run object's
``updated_at``: the latter is bumped after the jobs finish, which reported 478
runner-minutes where the jobs account for 67. A run counts as superseded when the
next run on the same branch was created before it finished, which is exactly the
condition ``cancel-in-progress`` acts on::

    workflow                    sampled window   superseded   runner-min after
    CodeQL                      08-18 .. 08-19       13            48.2
    Detect AgentTool API ...    08-16 .. 08-19       21            16.4
    Detect Breaking Changes     08-17 .. 08-19        4             1.2
    dependency-review           08-18 .. 08-19        6             0.8
    LLM Input Safety            08-17 .. 08-19        1             0.1
                                                     45            66.7

Three of the five account for ~2 minutes between them, so for those this is not a
minutes argument and is not offered as one. How much a workflow wastes depends on
how often a second push lands while its run is still going - a property of the
review loop and of the job's own duration, not of the workflow - so grading it
per workflow would make the fleet's behaviour depend on which workflow happened to
be sampled in a busy week. The rule is the one the other six already follow: a
pull_request run whose sha has been replaced should not still be running.

The queue is what turns those minutes into latency rather than cost. At 19:15Z
that day 35 runs were queued against 1 in progress, and the in-progress run of the
required check had waited 53 minutes between ``created_at`` 17:28Z and
``run_started_at`` 18:21Z. An obsolete run does not merely spend a minute; it
spends it ahead of the head everybody is waiting on.

``closing-reference.yml`` is exempt and must stay exempt, which is why the
exemption below records a pin rather than a preference. It is the one workflow
subscribed to an activity type that cannot change the head sha (``edited``), so it
is the only one that can have two runs on a single head - and #2216 measured what
cancelling one of those does: the head carries ``SUCCESS`` and ``CANCELLED``
together for the same check, and the roll-up read ``SUCCESS``, then ``FAILURE``,
then ``SUCCESS`` across three reads of one unchanged sha. A cancelled context on a
head that satisfies the check is unclearable without a push, which defeats the
self-clearing property that gate exists to have.
``tests/test_pull_request_trigger_types.py`` states the fleet-wide rule and
``tests/test_closing_reference_gate.py`` pins the absence of the block in that
file; this module is the third face of it - the sweep that would otherwise hand
that workflow a group for uniformity's sake.

The parsing helpers are imported from ``tests/test_push_concurrency_group.py``
rather than re-derived. That module grades the same ``concurrency:`` blocks from
the push side, and two readers of one block can disagree - which would let both
pins pass while describing different files. The reasons *not* to share are about
optional dependencies (``pyyaml``) and about YAML parsing, and neither applies to
a reader that is already text-only and already in the test tree.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from tests.test_push_concurrency_group import (
    _concurrency,
    _pull_request_context,
    _render,
    _workflow_name,
    _workflow_paths,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"

#: Workflows a pull request starts that must **not** declare a group, mapped to
#: the reason. This is the opposite shape to the sibling module's exemption table:
#: there an entry excuses a group that collapses two commits, here it excuses the
#: absence of a group entirely. An entry has to name where the remaining question
#: is settled, so a deferral cannot hide here as a decision.
_MUST_NOT_CANCEL_ITS_OWN_RUN = {
    "closing-reference.yml": (
        "Subscribed to `edited`, which cannot change the head sha, so it is the only "
        "workflow here that can have two runs on one head. #2216 measured the result of "
        "cancelling one of them: the head carried the same check as SUCCESS and CANCELLED "
        "together and its roll-up read SUCCESS, then FAILURE, then SUCCESS across three "
        "reads of one unchanged sha. A cancelled context on a head that satisfies the check "
        "needs a push to clear, which is the self-clearing property the gate exists to have. "
        "Pinned from the other side by tests/test_closing_reference_gate.py::"
        "test_the_gate_does_not_cancel_its_own_run and stated fleet-wide by "
        "tests/test_pull_request_trigger_types.py::test_an_exempt_workflow_cannot_cancel_its_own_run."
    ),
}

#: Matches a ``pull_request:`` trigger key. Read as a key rather than as a
#: substring: ``paths`` filters, job conditions and prose mention the event freely,
#: and ``pull_request_review`` and ``pull_request_target`` are different triggers
#: that must not be read as this one.
_PULL_REQUEST_TRIGGER_RE = re.compile(r"^\s+pull_request:\s*$", re.MULTILINE)


def _subscribes_to_pull_request(text: str) -> bool:
    return _PULL_REQUEST_TRIGGER_RE.search(text) is not None


def _pull_request_workflows() -> dict[str, str]:
    return {
        path.name: text
        for path in _workflow_paths()
        if _subscribes_to_pull_request(text := path.read_text(encoding="utf-8"))
    }


def _faults(workflows: Mapping[str, str]) -> dict[str, str]:
    """Report, per workflow, why its group does not supersede its own previous run.

    A pure function of the workflow texts so the vacuity tests below can hand it a
    synthesized fleet. Four distinct faults, because a group can fail to supersede
    in four ways and only one of them is the absence of a block:

    - no top-level ``concurrency`` at all, so nothing is ever cancelled;
    - a block that does not cancel, which makes the group key decide nothing;
    - a group two different pull requests share, so one branch's push kills
      another's run;
    - a group two pushes to *one* pull request do not share, which is the subtle
      one - the block is present, the syntax is right, and the superseded run
      still runs to completion.
    """
    faults: dict[str, str] = {}
    for name, text in sorted(workflows.items()):
        if name in _MUST_NOT_CANCEL_ITS_OWN_RUN:
            continue
        concurrency = _concurrency(text)
        if concurrency is None:
            faults[name] = (
                "declares no top-level concurrency, so a push to a pull request leaves the "
                "previous head's run going on a runner the new head is queued behind"
            )
            continue
        group, cancel = concurrency
        if cancel != "true":
            faults[name] = f"cancel-in-progress is {cancel!r}, so the group key decides nothing"
            continue
        workflow_name = _workflow_name(text)
        first = _render(group, _pull_request_context(workflow_name, "4242", "aaaaaaaaaaaa"))
        second = _render(group, _pull_request_context(workflow_name, "4242", "bbbbbbbbbbbb"))
        other = _render(group, _pull_request_context(workflow_name, "4243", "aaaaaaaaaaaa"))
        if first != second:
            faults[name] = (
                f"two pushes to one pull request render different groups ({first!r} vs {second!r}), "
                "so the run over the replaced sha is not cancelled"
            )
        elif first == other:
            faults[name] = (
                f"two different pull requests render the same group ({first!r}), so a push on one "
                "branch cancels a run about another"
            )
    return faults


def test_the_scanner_finds_the_pull_request_workflows() -> None:
    """Guard every pin below against a scanner that quietly matches nothing.

    Asserted as a floor plus two named members rather than as an equality: adding
    a pull_request workflow is routine and must inherit the rule instead of failing
    this, but a regex that stopped matching would otherwise satisfy the sweep with
    an empty fleet.
    """
    workflows = _pull_request_workflows()
    assert len(workflows) >= 8, sorted(workflows)
    assert "pr-and-push.yml" in workflows, sorted(workflows)
    assert "codeql.yml" in workflows, sorted(workflows)
    assert "pypi-publish-on-release.yml" not in workflows, "the trigger regex is matching prose"


def test_every_pull_request_workflow_supersedes_its_own_previous_run() -> None:
    """The rule the next pull_request workflow inherits instead of rediscovering."""
    faults = _faults(_pull_request_workflows())
    assert not faults, (
        "these workflows do not cancel their own superseded run for a pull request: "
        + "; ".join(f"{name}: {why}" for name, why in faults.items())
        + ". Give them `group: ${{ github.workflow }}-${{ github.event.pull_request.number }}` "
        "with `cancel-in-progress: true`, or record in _MUST_NOT_CANCEL_ITS_OWN_RUN why a "
        "cancelled run of this one would be worse than a superseded one."
    )


def test_a_workflow_with_no_concurrency_block_is_reported() -> None:
    """Vacuity: the sweep above passes today, so its grader must be shown to bite."""
    faults = _faults({"invented.yml": "name: Invented\non:\n  pull_request:\n\njobs: {}\n"})
    assert "invented.yml" in faults
    assert "no top-level concurrency" in faults["invented.yml"]


def test_a_workflow_that_declares_a_group_but_does_not_cancel_is_reported() -> None:
    """A present block with cancelling off is the same behaviour as no block."""
    text = (
        "name: Invented\non:\n  pull_request:\n\nconcurrency:\n"
        "  group: ${{ github.workflow }}-${{ github.event.pull_request.number }}\n"
        "  cancel-in-progress: false\n\njobs: {}\n"
    )
    faults = _faults({"invented.yml": text})
    assert "invented.yml" in faults
    assert "cancel-in-progress is 'false'" in faults["invented.yml"]


def test_a_group_two_pull_requests_share_is_reported() -> None:
    """The over-broad key: one branch's push cancels a run about another branch."""
    text = (
        "name: Invented\non:\n  pull_request:\n\nconcurrency:\n"
        "  group: ${{ github.workflow }}\n  cancel-in-progress: true\n\njobs: {}\n"
    )
    faults = _faults({"invented.yml": text})
    assert "invented.yml" in faults
    assert "same group" in faults["invented.yml"]


def test_a_group_split_by_head_sha_is_reported() -> None:
    """The plausible near-miss: a block that is present and supersedes nothing.

    Keying on the head sha reads like a refinement and is the whole defect - every
    push gets its own bucket, so the run over the replaced sha is never cancelled
    and the block buys nothing but the appearance of one.
    """
    text = (
        "name: Invented\non:\n  pull_request:\n\nconcurrency:\n"
        "  group: ${{ github.workflow }}-${{ github.event.pull_request.head.sha }}\n"
        "  cancel-in-progress: true\n\njobs: {}\n"
    )
    faults = _faults({"invented.yml": text})
    assert "invented.yml" in faults
    assert "different groups" in faults["invented.yml"]


def test_the_exempt_workflow_is_not_graded_by_the_sweep() -> None:
    """Vacuity in the other direction: an exemption that grades nothing is noise.

    The exempt file declares no block, which is the first fault ``_faults``
    reports, so a table that had stopped being consulted would show up here rather
    than as a silently passing sweep.
    """
    for name in _MUST_NOT_CANCEL_ITS_OWN_RUN:
        text = (_WORKFLOW_DIR / name).read_text(encoding="utf-8")
        assert name not in _faults({name: text})
        assert _faults({"copy-of-" + name: text}), (
            f"{name} would not be reported even without its exemption, so the entry is "
            "documenting a fault that no longer exists; drop it"
        )


def test_one_workflow_cannot_cancel_another_ones_run() -> None:
    """The premise every exemption elsewhere in the tree leans on.

    ``tests/test_pull_request_trigger_types.py`` admits ``closing-reference.yml``'s
    sha-invariant trigger on the grounds that an edit starts no run in the required
    check's group. That is only true while every group is keyed on the workflow, so
    it is a property of the whole fleet rather than of either file, and it is read
    back here for one pull request across every group at once.

    Whether every workflow *has* a group is the sweep's question, so the floor here
    is two rather than the fleet's size: this one is about the groups that exist not
    colliding, and it should not fail a second time for the reason already reported.
    """
    workflows = _pull_request_workflows()
    groups: dict[str, str] = {}
    for name, text in sorted(workflows.items()):
        concurrency = _concurrency(text)
        if concurrency is None:
            continue
        group, _ = concurrency
        groups[name] = _render(group, _pull_request_context(_workflow_name(text), "4242", "aaaaaaaaaaaa"))

    assert len(groups) >= 2, groups
    collisions = {value: [name for name in groups if groups[name] == value] for value in set(groups.values())}
    shared = {value: names for value, names in collisions.items() if len(names) > 1}
    assert not shared, (
        f"these workflows share a concurrency group on one pull request: {shared}. One of them "
        "then cancels the other's run, and the cancelled context lands on a live head"
    )


def test_every_exemption_still_applies_to_a_workflow_a_pull_request_can_start() -> None:
    """An exemption may not outlive the situation that justified it.

    Each entry claims three things - the workflow exists, a pull request starts it,
    and it declares no group - and cites where the question is settled. If any of
    them stops holding, the next reader inherits a reason for a rule that is not
    there.
    """
    assert _MUST_NOT_CANCEL_ITS_OWN_RUN, "the exemption table is empty; this pin has nothing to check"
    for name, reason in _MUST_NOT_CANCEL_ITS_OWN_RUN.items():
        path = _WORKFLOW_DIR / name
        assert path.exists(), f"{name} is exempt but does not exist"
        text = path.read_text(encoding="utf-8")
        assert _subscribes_to_pull_request(text), f"{name} is exempt but no pull request can start it; drop the entry"
        assert _concurrency(text) is None, (
            f"{name} is exempt from declaring a concurrency group and declares one anyway; either "
            "the exemption is stale or the block is the defect #2216 measured"
        )
        assert "#" in reason, (
            f"the exemption for {name} cites no issue. An exemption without somewhere the "
            "remaining question is being settled is a deferral, and reads as a decision"
        )


def test_the_exempt_workflow_still_needs_the_exemption() -> None:
    """The exemption's own premise: the trigger that makes cancelling harmful.

    Without a sha-invariant activity type, ``closing-reference.yml`` could not have
    two runs on one head, the #2216 measurement would not be reachable, and the
    entry would be excusing a workflow that should simply follow the rule. Read
    back here so the table cannot outlive its cause even if the trigger list is
    edited for an unrelated reason.
    """
    for name in _MUST_NOT_CANCEL_ITS_OWN_RUN:
        text = (_WORKFLOW_DIR / name).read_text(encoding="utf-8")
        assert re.search(r"^\s+types:.*\bedited\b", text, re.MULTILINE), (
            f"{name} no longer subscribes to a type that cannot change the head sha, so it can no "
            "longer have two runs on one head; drop the exemption and let the sweep cover it"
        )
