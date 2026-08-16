"""Contract pins for the concurrency group of every workflow a push can start.

``pr-and-push.yml`` cancels an in-flight run when a new one starts in the same
concurrency group. Keyed on ``github.event.pull_request.number || github.ref``
that was two different behaviours wearing one expression, because only the first
operand is ever set:

- On a ``pull_request`` event the number is present, so the group is per pull
  request. Cancelling is then exactly right - a new push supersedes the previous
  verdict about the same branch - and ``tests/test_pull_request_trigger_types.py``
  is about making sure nothing *but* a new push can trigger it.
- On a push to ``main`` or ``dev`` the number is empty and the fallback answered
  ``refs/heads/main`` for **every** commit. One group for the whole branch, so
  each merge cancelled the previous merge's run and the commit it was testing kept
  a permanently unfinished ``call-test-lint``.

A cancelled check run is not ``SUCCESS``, and a single non-``SUCCESS`` context
drags ``statusCheckRollup.state`` to ``FAILURE``, so the second case left the
commit reading red with no failing check anywhere in it. Measured over the last 25
commits on ``main`` (#2304): 24 had a settled rollup, 11 of those read
``FAILURE``, and **9 of the 11 had no failing check at all** - their only
non-``SUCCESS`` context was ``CANCELLED``. The false reds arrive in bursts because
that is what the mechanism requires: three of them landed inside 55 seconds and
two more inside 7.

#1800 measured the same mechanism on #1788/#1794/#1796 and concluded it was
tolerable to read around, which is why ``AGENTS.md`` carried a reading rule for it
rather than a fix. Two things #2304 adds turn it from a wrong colour into a wrong
answer:

- **It corrupts the only method available for dating a breakage.** #2303 diagnosed
  a red ``main`` by reading rollups along the branch. That conclusion holds, but
  the same burst that broke ``main`` also cancelled the runs of the two commits
  immediately before it, which now read red identically to the commit that
  actually broke. The burst destroyed the evidence for which commit in it was at
  fault in the same act as creating the fault.
- **Nothing ever re-establishes it.** A branch's stale red clears on its next
  push. A commit on ``main`` is immutable and already merged, so it gets no next
  push and the wrong answer is permanent.

The remedy keys the push side on the commit rather than the ref. Each pushed
commit then gets its own group and runs to completion, and the pull-request
operand is untouched. The cost is that a burst of N merges runs N suites instead
of 1, which is the point: on a branch nobody rewrites, each commit is a distinct
artifact, and *was this commit green* is a question only its own completed run can
answer.

The two spellings #2304 offers are **not** interchangeable, which is why this
module renders the group rather than asserting a string. Keeping ``github.ref``
and setting ``cancel-in-progress: ${{ github.event_name == 'pull_request' }}``
still leaves one group for the branch, and GitHub holds at most one *pending* run
per group - so in a burst of three merges the third cancels the second while it is
still queued, reproducing the CANCELLED context the change is removing.
``test_two_commits_pushed_to_main_do_not_share_a_group`` is written against the
rendered group for that reason: it fails on the ref-keyed spelling however
``cancel-in-progress`` is written.

These are text assertions rather than parsed YAML for the reason the sibling
CI-config pins state (``tests/test_codeql_query_filters.py``,
``tests/test_pull_request_trigger_types.py``): ``pyyaml`` is an optional
dependency here, and a pin that skips when a dep is missing is not a pin.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"
_REQUIRED_CHECK_WORKFLOW = _WORKFLOW_DIR / "pr-and-push.yml"

#: Workflows a push may start whose group is allowed to collapse two commits into
#: one bucket, mapped to the reason. An entry is not a waiver of the rule but a
#: statement that the job being cancelled is not a per-commit verdict, and it has
#: to say which job and where the remaining question is being settled - so a
#: deferral cannot hide here as a decision.
_CANCELS_A_SUPERSEDED_DEPLOY = {
    "docs.yml": (
        "Its `deploy` job publishes to GitHub Pages. A superseded run there is not a "
        "superseded verdict but a superseded deployment - the newest commit's site is "
        "the one that should be live - so cancelling an older in-flight publish is "
        "wanted, and a per-commit group would let two publishes of different trees race "
        "for one resource. Its `build` job is a per-commit verdict and does share the "
        "defect, so the fix is splitting the two rather than changing this key: "
        "decision tracked as #2305, with the three commits it accounts for."
    ),
}

#: Matches a top-level ``concurrency:`` key (column 0), which is the only scope
#: that can cancel a whole workflow run. A job-level block is indented and is a
#: different statement.
_TOP_LEVEL_CONCURRENCY = "concurrency:"

#: Matches one ``${{ ... }}`` expression. GitHub allows no nested braces inside
#: one, so a non-greedy body is exact rather than approximate.
_EXPRESSION_RE = re.compile(r"\$\{\{(?P<body>[^}]*)\}\}")


def _workflow_paths() -> list[Path]:
    return sorted(_WORKFLOW_DIR.glob("*.yml"))


def _workflow_name(text: str) -> str:
    """Return a workflow's ``name:``, which is the ``github.workflow`` its group keys on."""
    match = re.search(r"^name:\s*(?P<name>.+?)\s*$", text, re.MULTILINE)
    return match.group("name") if match else ""


def _subscribes_to_push(text: str) -> bool:
    """Whether the workflow has a ``push:`` trigger.

    Read as a trigger key rather than as a substring: ``paths`` filters and job
    steps mention ``push`` freely, and ``pypi-publish-on-release.yml`` mentions it
    in prose.
    """
    return re.search(r"^\s+push:\s*$", text, re.MULTILINE) is not None


def _concurrency(text: str) -> tuple[str, str] | None:
    """Return ``(group, cancel_in_progress)`` from the top-level block, or ``None``.

    ``None`` means the workflow declares no workflow-level concurrency, so it
    cancels nothing and the question this module asks does not arise for it -
    ``codeql.yml`` is in that position and is deliberately not exempted, because
    there is nothing to exempt.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.rstrip() != _TOP_LEVEL_CONCURRENCY:
            continue
        group = ""
        cancel = ""
        for follower in lines[index + 1 :]:
            if not follower.startswith((" ", "\t")):
                break
            stripped = follower.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith("group:"):
                group = stripped[len("group:") :].strip()
            elif stripped.startswith("cancel-in-progress:"):
                cancel = stripped[len("cancel-in-progress:") :].strip()
        return group, cancel
    return None


def _render(expression: str, context: Mapping[str, str]) -> str:
    """Evaluate the ``${{ a || b }}`` idiom the group keys use, under one context.

    Only the operators the groups in this repository actually use are modelled -
    ``||`` over context lookups and single-quoted literals. An operand outside the
    model raises rather than rendering empty: a group written with an expression
    this cannot evaluate would otherwise collapse to a constant and satisfy every
    assertion below for the wrong reason.
    """

    def _one(match: re.Match[str]) -> str:
        for operand in (part.strip() for part in match.group("body").split("||")):
            if operand.startswith("'") and operand.endswith("'"):
                return operand[1:-1]
            if operand not in context:
                raise AssertionError(
                    f"the concurrency group reads {operand!r}, which this module does not "
                    "model. Add it to the simulated contexts below so the pin keeps "
                    "meaning what it says, rather than passing on an unevaluated key."
                )
            value = context[operand]
            if value:
                return value
        return ""

    return _EXPRESSION_RE.sub(_one, expression)


def _push_context(workflow_name: str, sha: str, ref: str = "refs/heads/main") -> dict[str, str]:
    """The context a push to a protected branch supplies.

    ``github.event.pull_request.*`` is empty rather than absent, which is the whole
    mechanism: the fallback operand is the one that gets used.
    """
    return {
        "github.workflow": workflow_name,
        "github.event_name": "push",
        "github.ref": ref,
        "github.sha": sha,
        "github.event.pull_request.number": "",
        "github.event.pull_request.head.sha": "",
    }


def _pull_request_context(workflow_name: str, number: str, head_sha: str) -> dict[str, str]:
    """The context a ``pull_request`` event supplies, where cancelling is wanted."""
    return {
        "github.workflow": workflow_name,
        "github.event_name": "pull_request",
        "github.ref": f"refs/pull/{number}/merge",
        "github.sha": f"merge-of-{head_sha}",
        "github.event.pull_request.number": number,
        "github.event.pull_request.head.sha": head_sha,
    }


def test_the_scanner_finds_the_push_workflows() -> None:
    """Guard every pin below against a scanner that quietly matches nothing.

    Each assertion here is a statement about the workflows the scanner found; if
    it found none they would all hold for the wrong reason. Asserted as a floor
    rather than an equality, because adding a push workflow is routine and must
    inherit the rule instead of failing this.
    """
    paths = _workflow_paths()
    assert paths, f"no workflows under {_WORKFLOW_DIR}"
    pushed = sorted(p.name for p in paths if _subscribes_to_push(p.read_text(encoding="utf-8")))
    assert _REQUIRED_CHECK_WORKFLOW.name in pushed, pushed
    assert len(pushed) >= 2, pushed
    assert _concurrency(_REQUIRED_CHECK_WORKFLOW.read_text(encoding="utf-8")) is not None


def test_two_commits_pushed_to_main_do_not_share_a_group() -> None:
    """The fix: a pushed commit's run is not cancellable by the next merge.

    Written against the rendered group, not the expression's text, because the two
    spellings #2304 proposes are not equivalent - keeping ``github.ref`` and
    gating ``cancel-in-progress`` on the event name leaves one group per branch,
    where a third merge cancels the second's *pending* run and produces the same
    CANCELLED context. Only a group that distinguishes the commits settles it.
    """
    text = _REQUIRED_CHECK_WORKFLOW.read_text(encoding="utf-8")
    concurrency = _concurrency(text)
    assert concurrency is not None
    group, _ = concurrency
    name = _workflow_name(text)

    first = _render(group, _push_context(name, "aaaaaaaaaaaa"))
    second = _render(group, _push_context(name, "bbbbbbbbbbbb"))

    assert first != second, (
        f"two commits pushed to main both render the group {first!r}, so the second merge "
        "cancels the first's run of the required check and the first commit keeps a "
        "CANCELLED context forever - which reads as FAILURE and, unlike a branch, gets no "
        "next push to clear it. Key the push side on the commit (github.sha)."
    )


def test_two_commits_pushed_to_dev_do_not_share_a_group() -> None:
    """``dev`` is a push branch too, so the property cannot be main-specific.

    The defect was never about ``main``'s name - it was about a group keyed on
    something every commit on a branch shares - and a fix that special-cased the
    default branch would leave ``dev`` behind while reading as done.
    """
    text = _REQUIRED_CHECK_WORKFLOW.read_text(encoding="utf-8")
    assert "branches: [ main, dev ]" in text, "the push branch list moved; re-derive this pin"
    concurrency = _concurrency(text)
    assert concurrency is not None
    group, _ = concurrency
    name = _workflow_name(text)

    first = _render(group, _push_context(name, "cccccccccccc", ref="refs/heads/dev"))
    second = _render(group, _push_context(name, "dddddddddddd", ref="refs/heads/dev"))

    assert first != second, f"two commits pushed to dev both render the group {first!r}"


def test_two_pushes_to_one_pull_request_still_share_a_group() -> None:
    """Preservation control: cancelling per pull request is wanted and unchanged.

    Passes before and after the change, which is what makes it a control: the
    pull-request operand is what supersedes a stale verdict, and #1914's whole
    finding is that the *cost* of cancelling there is a run that had nothing new
    to say. A group that also split by commit would keep both runs alive and
    leave the superseded one to finish, which is the behaviour that file exists to
    prevent.
    """
    text = _REQUIRED_CHECK_WORKFLOW.read_text(encoding="utf-8")
    concurrency = _concurrency(text)
    assert concurrency is not None
    group, _ = concurrency
    name = _workflow_name(text)

    first = _render(group, _pull_request_context(name, "2306", "eeeeeeeeeeee"))
    second = _render(group, _pull_request_context(name, "2306", "ffffffffffff"))

    assert first == second, (
        f"two pushes to one pull request render different groups ({first!r} vs {second!r}), so "
        "the superseded run is no longer cancelled and a 27-minute suite runs to completion "
        "over a commit nobody will read a verdict about"
    )


def test_two_pull_requests_do_not_share_a_group() -> None:
    """Preservation control: one pull request's push cannot cancel another's run."""
    text = _REQUIRED_CHECK_WORKFLOW.read_text(encoding="utf-8")
    concurrency = _concurrency(text)
    assert concurrency is not None
    group, _ = concurrency
    name = _workflow_name(text)

    assert _render(group, _pull_request_context(name, "2306", "eeeeeeeeeeee")) != _render(
        group, _pull_request_context(name, "2307", "eeeeeeeeeeee")
    )


def test_the_required_check_still_cancels_in_progress() -> None:
    """Premise: the group key is load-bearing only while cancelling happens.

    Pins today's behaviour rather than a wish. Were ``cancel-in-progress`` turned
    off, the group key would stop deciding anything and every assertion above
    would hold vacuously - so this failing is a prompt to re-read the module, not
    to delete it. It is also the premise
    ``tests/test_pull_request_trigger_types.py`` states for the pull-request side.
    """
    concurrency = _concurrency(_REQUIRED_CHECK_WORKFLOW.read_text(encoding="utf-8"))
    assert concurrency is not None
    _, cancel = concurrency
    assert cancel == "true", (
        f"cancel-in-progress is {cancel!r}; if cancelling has been turned off, the reasoning "
        "in this module and in tests/test_pull_request_trigger_types.py both need re-deriving"
    )


def test_every_push_workflow_keys_its_group_on_the_commit() -> None:
    """The rule the next push workflow inherits instead of rediscovering.

    Swept over the whole directory rather than asserted of one file, because the
    defect is a property of the idiom (``pull_request.number || <something a whole
    branch shares>``) and #2304 found it by noticing the idiom appears more than
    once. A workflow with no top-level ``concurrency`` cancels nothing and is
    skipped rather than exempted.
    """
    offenders: dict[str, str] = {}
    scanned: list[str] = []
    for path in _workflow_paths():
        text = path.read_text(encoding="utf-8")
        if not _subscribes_to_push(text):
            continue
        concurrency = _concurrency(text)
        if concurrency is None:
            continue
        group, cancel = concurrency
        if cancel != "true":
            continue
        scanned.append(path.name)
        if path.name in _CANCELS_A_SUPERSEDED_DEPLOY:
            continue
        name = _workflow_name(text)
        first = _render(group, _push_context(name, "111111111111"))
        second = _render(group, _push_context(name, "222222222222"))
        if first == second:
            offenders[path.name] = first

    assert _REQUIRED_CHECK_WORKFLOW.name in scanned, (
        f"the sweep did not reach the required check ({scanned}); it cancels in progress on a "
        "push, so it must be one of the workflows this rule is checked against"
    )
    assert not offenders, (
        "these workflows cancel in progress and give two different pushed commits the same "
        f"concurrency group: {offenders}. The second push then cancels the first commit's run, "
        "and the cancelled context is permanent, reads as FAILURE, and gets no next push to "
        "clear it. Key the push side on github.sha, or record why the cancelled job is not a "
        "per-commit verdict in _CANCELS_A_SUPERSEDED_DEPLOY."
    )


def test_every_exemption_still_applies_to_a_workflow_a_push_can_start() -> None:
    """An exemption may not outlive the situation that justified it.

    Each entry claims a workflow is started by a push and cancels in progress. If
    either stops being true the entry is documenting a defect that no longer
    exists, and the next reader inherits a reason for a rule that is not there.
    """
    assert _CANCELS_A_SUPERSEDED_DEPLOY, "the exemption table is empty; this pin has nothing to check"
    for name, reason in _CANCELS_A_SUPERSEDED_DEPLOY.items():
        path = _WORKFLOW_DIR / name
        assert path.exists(), f"{name} is exempt but does not exist"
        text = path.read_text(encoding="utf-8")
        assert _subscribes_to_push(text), f"{name} is exempt but no push can start it; drop the entry"
        concurrency = _concurrency(text)
        assert concurrency is not None, f"{name} is exempt but declares no concurrency; drop the entry"
        group, cancel = concurrency
        assert cancel == "true", f"{name} is exempt but no longer cancels in progress; drop the entry"
        assert "#" in reason, (
            f"the exemption for {name} cites no issue. An exemption without somewhere the "
            "remaining question is being settled is a deferral, and reads as a decision"
        )
        first = _render(group, _push_context(_workflow_name(text), "333333333333"))
        second = _render(group, _push_context(_workflow_name(text), "444444444444"))
        assert first == second, (
            f"{name} now gives two pushed commits distinct groups ({first!r} vs {second!r}), so it "
            "no longer needs the exemption; drop the entry and let the sweep cover it"
        )
