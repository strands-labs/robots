"""Pins the closing-reference check against the titles this repository actually shipped.

The measurement that motivates the check is a corpus, not an anecdote: over the
last 100 pull requests, 29 titles carry a closing keyword before an issue number,
27 of those also linked the issue, and exactly 2 did not. So the interesting
properties are the two the corpus below fixes in place -- that the scanner fires
on the two known incidents, and that it stays silent on the 71 ordinary titles,
whose ``fix(scope):`` prefixes and bare ``#1722`` references are precisely the
shapes a careless keyword regexp mistakes for a claim.

The other load-bearing pin is ``test_a_body_that_mentions_the_keyword_cannot_clear_a_claim``.
#1894's body *does* contain the words ``closes #1891``, inside a code span, and
GitHub linked nothing -- so the obvious implementation of this check passes the
incident it was written for. Reading the published link set instead is the design
decision that test exists to protect.

See scripts/check_closing_reference.py, issue #1961, and the "PR Workflow"
section of AGENTS.md.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "check_closing_reference.py"
_WORKFLOW = _ROOT / ".github" / "workflows" / "closing-reference.yml"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_closing_reference", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The script is annotated and mypy-clean on its own
# (mypy scripts/check_closing_reference.py); it is reached through importlib here
# because scripts/ is not an importable package, so its members are module
# attributes at runtime rather than names mypy can resolve to types.
mod = _load()


# --------------------------------------------------------------------------
# The measured corpus. Real titles, with the link set GitHub reported for each
# (verified identical under GITHUB_TOKEN and a personal token, because #1961
# records PullRequest.projectItems returning a false 0 under the former).
#
# The two findings are the whole point; the clean rows are what proves the
# scanner is not simply firing on everything.
# --------------------------------------------------------------------------
MEASURED = [
    pytest.param(
        "refactor(examples): remove the Isaac Replicator synth-data stub (closes #1891)",
        (),
        mod.TITLE_ONLY_CLAIM,
        (1891,),
        id="pr1894-claim-dropped-issue-stayed-open",
    ),
    pytest.param(
        "fix(training/rl): the action head is sized and named from the action keys (closes #1912)",
        (),
        mod.TITLE_ONLY_CLAIM,
        (1912,),
        id="pr1923-claim-dropped-issue-closed-by-hand",
    ),
    pytest.param(
        "fix(deps): floor lerobot at >=0.6.1 so bucket streaming is resolver-guaranteed (closes #1506, closes #1516)",
        (1506, 1516),
        mod.LINKED,
        (),
        id="pr1930-two-claims-both-linked",
    ),
    pytest.param(
        "ci: name the pull request whose only approval its own pusher supplied (closes #1905)",
        (1905,),
        mod.LINKED,
        (),
        id="pr1921-claim-linked",
    ),
    pytest.param(
        "docs(agents): PR Workflow step 1 names the rule that forces a fork",
        (1959,),
        mod.NO_CLAIM,
        (),
        id="pr1960-linked-without-claiming-in-the-title",
    ),
    pytest.param(
        "fix(training): val_episodes is one shared positive-count domain for both writers of eval_split",
        (),
        mod.NO_CLAIM,
        (),
        id="pr1955-no-claim-no-link",
    ),
    pytest.param(
        "fix(sim): bound the lock hold in step on every backend",
        (),
        mod.NO_CLAIM,
        (),
        id="pr1885-the-other-way-a-link-is-lost-stays-uncaught",
    ),
]


@pytest.mark.parametrize("title,linked,outcome,unlinked", MEASURED)
def test_the_measured_titles_classify_as_observed(
    title: str, linked: tuple[int, ...], outcome: str, unlinked: tuple[int, ...]
) -> None:
    verdict = mod.classify(title, linked)
    assert verdict.outcome == outcome
    assert verdict.unlinked == unlinked


def test_the_finding_is_two_of_the_measured_titles_and_not_the_rest() -> None:
    """Non-vacuity, stated as a count.

    A scanner that matched nothing would make every row above pass its own
    assertion by reporting ``no-claim``, so the corpus is also asserted as a
    population: two findings, and the claim-bearing rows outnumber them.
    """
    verdicts = [mod.classify(case.values[0], case.values[1]) for case in MEASURED]
    findings = [v for v in verdicts if v.is_finding]
    claiming = [v for v in verdicts if v.claimed]

    assert len(findings) == 2
    assert len(claiming) == 4
    assert sorted(n for v in findings for n in v.unlinked) == [1891, 1912]


@pytest.mark.parametrize("keyword", mod.CLOSING_KEYWORDS)
@pytest.mark.parametrize("case", [str.lower, str.upper, str.capitalize])
def test_every_documented_keyword_is_a_claim_in_any_case(keyword: str, case: Any) -> None:
    """All nine keywords GitHub acts on, in the three casings that occur in titles.

    Pinned against the constant rather than a hand-written list so the two cannot
    drift: a keyword added to CLOSING_KEYWORDS but not to the pattern fails here.
    """
    assert mod.title_claims(f"fix(sim): something {case(keyword)} #1234") == (1234,)


@pytest.mark.parametrize(
    "separator",
    ["closes #12", "closes  #12", "closes: #12", "closes:#12", "closes#12", "(closes #12)"],
)
def test_the_keyword_may_be_separated_from_the_number_by_punctuation(separator: str) -> None:
    assert mod.title_claims(f"docs: a change {separator}") == (12,)


@pytest.mark.parametrize(
    "title",
    [
        # The dominant title shape in this repository. "fix(" is a keyword
        # followed by a scope, not by a number, and the trailing reference has no
        # keyword before it -- both halves have to be got right for the 71
        # no-claim titles in the corpus to stay quiet.
        "fix(tools): pose_tool honors its steps / step_delay interpolation options",
        "fix(sim/newton): set_timestep applies the one shared timestep domain (#1931)",
        "refactor(mesh): migrate RosbridgeRobot onto MobileBaseRobot, follow-up to #1722",
        "docs: see #1234 for the rationale",
        "test(sim): a regression for the defect reported in #1823",
        "revert of #1840",
        # Word-boundary traps: both contain a keyword as a substring.
        "docs: the prefixes #12 uses",
        "fix: an unfixed #12 remains unfixed",
    ],
)
def test_a_reference_without_an_adjacent_keyword_is_not_a_claim(title: str) -> None:
    assert mod.title_claims(title) == ()


def test_one_keyword_governs_one_number() -> None:
    """How GitHub reads a body, applied to a title.

    ``closes #1506, closes #1516`` claims both -- that is #1930, and both were
    linked. ``fixes #12 and #13`` claims only #12, because #13 has no keyword of
    its own; a body written that way would also only have closed one, so
    reporting the second as a dropped claim would be wrong.
    """
    assert mod.title_claims("fix: closes #1506, closes #1516") == (1506, 1516)
    assert mod.title_claims("fix: fixes #12 and #13") == (12,)


def test_a_number_claimed_twice_is_reported_once() -> None:
    assert mod.title_claims("fix: closes #12, fixes #12") == (12,)


def test_a_body_that_mentions_the_keyword_cannot_clear_a_claim() -> None:
    """The design decision, pinned.

    #1894's body contains ``closes #1891`` inside a code span. GitHub does not
    link from a code span, so its link set is empty while its body text matches a
    keyword scan -- which means the obvious implementation of this check reports
    the incident it was written for as clean. Reading the published link set is
    what makes the two agree, and this test fails for any reimplementation that
    goes back to reading the body.
    """
    title = "refactor(examples): remove the Isaac Replicator synth-data stub (closes #1891)"
    verdict = mod.classify(title, ())

    assert verdict.is_finding
    assert verdict.unlinked == (1891,)


def test_a_claim_linked_to_a_different_issue_is_still_a_finding() -> None:
    """Linking *something* is not linking what the title promised."""
    verdict = mod.classify("fix: closes #1891", (1890,))
    assert verdict.is_finding
    assert verdict.unlinked == (1891,)


def test_a_partially_linked_claim_names_only_the_missing_issue() -> None:
    verdict = mod.classify("fix: closes #1506, closes #1516", (1506,))
    assert verdict.is_finding
    assert verdict.claimed == (1506, 1516)
    assert verdict.unlinked == (1516,)
    assert "#1516" in verdict.summary
    assert "#1506" not in verdict.summary


def test_an_unreadable_link_set_is_not_a_finding() -> None:
    """An unanswered question must not be reported as a dropped claim.

    #1961 measured ``PullRequest.projectItems`` returning a false ``0`` under the
    Actions token. Were ``closingIssuesReferences`` ever to behave that way, an
    empty-means-missing reading would put a red X on every branch whose title
    names an issue, so the two are kept apart: ``None`` is "GitHub did not
    answer", ``()`` is "GitHub answered none".
    """
    verdict = mod.classify("fix: closes #1891", None, "the API returned errors: ...")

    assert verdict.outcome == mod.UNKNOWN_LINKS
    assert verdict.is_finding is False
    assert verdict.claimed == (1891,)
    assert "not evidence" in verdict.summary


def test_a_truncated_link_set_is_reported_as_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A page cut short would report a linked issue as unlinked."""
    monkeypatch.setattr(
        mod,
        "_post",
        lambda *_a, **_k: {
            "data": {
                "repository": {
                    "pullRequest": {
                        "title": "fix: closes #12",
                        "closingIssuesReferences": {"totalCount": 101, "nodes": [{"number": 12}]},
                    }
                }
            }
        },
    )
    with pytest.raises(mod.LinkSetUnreadable, match="truncated"):
        mod.resolve_pull_request("strands-labs/robots", 1, "token")


def test_api_errors_are_unreadable_rather_than_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_post", lambda *_a, **_k: {"errors": [{"message": "Resource not accessible"}]})
    with pytest.raises(mod.LinkSetUnreadable, match="returned errors"):
        mod.resolve_pull_request("strands-labs/robots", 1, "token")


def test_a_complete_link_set_is_returned_with_the_title(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "_post",
        lambda *_a, **_k: {
            "data": {
                "repository": {
                    "pullRequest": {
                        "title": "fix: closes #1506, closes #1516",
                        "closingIssuesReferences": {"totalCount": 2, "nodes": [{"number": 1506}, {"number": 1516}]},
                    }
                }
            }
        },
    )
    title, linked = mod.resolve_pull_request("strands-labs/robots", 1930, "token")
    assert title == "fix: closes #1506, closes #1516"
    assert linked == (1506, 1516)


def test_the_finding_report_names_the_issue_and_both_remedies() -> None:
    verdict = mod.classify("refactor(examples): remove the stub (closes #1891)", ())
    report = mod.render(verdict, "strands-labs/robots", 1894, "refactor(examples): remove the stub (closes #1891)")

    assert "title-only-claim" in report
    assert "#1891" in report
    assert "Closes #1891" in report
    assert "reword the title" in report
    assert "code span" in report
    assert "never from the title" in verdict.summary


def test_a_clean_report_says_so_without_a_remedy_section() -> None:
    verdict = mod.classify("ci: a change (closes #1905)", (1905,))
    report = mod.render(verdict, "strands-labs/robots", 1921, "ci: a change (closes #1905)")

    assert "linked" in report
    assert "What clears this" not in report


def test_a_pipe_in_the_title_does_not_break_the_report_table() -> None:
    title = "fix: a | b (closes #12)"
    report = mod.render(mod.classify(title, ()), "strands-labs/robots", 1, title)
    assert r"a \| b" in report


@pytest.mark.parametrize(
    "title,linked,status",
    [
        ("refactor(examples): remove the stub (closes #1891)", (), 1),
        ("ci: a change (closes #1905)", (1905,), 0),
        ("fix(sim): bound the lock hold in step on every backend", (), 0),
    ],
)
def test_the_exit_status_is_one_only_for_the_finding(
    title: str, linked: tuple[int, ...], status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(mod, "resolve_pull_request", lambda *_a, **_k: (title, linked))

    argv = ["--repo", "strands-labs/robots", "--pr", "1", "--title", title, "--token", "token"]
    assert mod.main(argv) == status


def test_a_lookup_failure_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check reports nothing rather than accusing a branch it could not read."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    def explode(*_a: object, **_k: object) -> tuple[str, tuple[int, ...]]:
        raise mod.LinkSetUnreadable("no pull request in the response.")

    monkeypatch.setattr(mod, "resolve_pull_request", explode)
    argv = ["--repo", "strands-labs/robots", "--pr", "1", "--title", "fix: closes #1891", "--token", "token"]
    assert mod.main(argv) == 0


# --------------------------------------------------------------------------
# The workflow that runs it.
#
# Read as text rather than parsed, which is how every other workflow pin in this
# suite reads one (tests/test_dependabot_config_location.py,
# tests/test_codeql_query_filters.py): ``pyyaml`` is an optional dependency here,
# so a pin that imports it becomes a pin that skips, and skipping is how a
# structural guard stops guarding without anyone noticing.
# --------------------------------------------------------------------------
def _workflow() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def test_the_gate_reruns_when_the_title_or_body_is_edited() -> None:
    """``edited`` is the trigger that makes the remedy verifiable.

    The report asks the author to move the keyword into the body, which changes no
    code. #1914 narrowed the required test job away from exactly such events --
    correctly, since its input is the code -- so the opposite choice here is
    pinned rather than left as a line in a comment: without ``edited`` the only
    way to re-run this check would be an unrelated push, and a self-clearing gate
    that needs a push to clear is not self-clearing.
    """
    match = re.search(r"^\s*types:\s*\[([^\]]*)\]", _workflow(), re.MULTILINE)
    assert match, "the pull_request trigger declares no explicit types"
    types = {t.strip() for t in match.group(1).split(",")}

    assert "edited" in types
    assert {"opened", "reopened", "synchronize"} <= types


def test_the_gate_asks_for_the_scope_the_link_set_needs() -> None:
    """``pull-requests: read`` is what makes closingIssuesReferences readable.

    Without it the query returns errors, the script reports ``unknown-links``, and
    the check passes everything -- a silent no-op rather than a visible failure,
    which is the failure mode worth pinning.
    """
    assert re.search(r"^permissions:$", _workflow(), re.MULTILINE)
    assert re.search(r"^\s+pull-requests:\s*read\s*$", _workflow(), re.MULTILINE)


def test_the_gate_runs_the_script_from_the_base_checkout() -> None:
    """#1791: a branch that forked before this job landed does not carry the script.

    Checking out the base also means the job never executes the code it is
    reviewing, which matters for a job holding a token on a fork's pull request.
    """
    workflow = _workflow()
    assert "ref: ${{ github.base_ref }}" in workflow
    assert "python3 scripts/check_closing_reference.py" in workflow


def test_the_gate_passes_when_the_base_carries_no_copy_of_the_check() -> None:
    """The residual case a base checkout cannot cover: the branch that adds the script.

    Measured on this pull request's own first run, which died with
    ``can't open file ... check_closing_reference.py`` and exit 2 -- neither of the
    script's own statuses, and rendered by the checks UI as the same red X as a
    real finding, which is the argument #1791 makes about exit 2.

    This is a bounded condition and not a bypass: the ref checked out is the base
    branch tip rather than the merge base, so the file is missing only for runs
    that happen before this lands. A later deletion is caught by the module-level
    load at the top of this file, which fails at import, in the required check.
    """
    workflow = _workflow()
    assert "if [ ! -f scripts/check_closing_reference.py ]; then" in workflow
    assert "::notice title=No closing-reference rule on this base::" in workflow
    assert _SCRIPT.exists()


def test_the_title_is_not_handed_to_the_script_by_the_workflow() -> None:
    """#2216: the payload's title outranks the API's, and would go stale.

    ``main()`` resolves ``title = args.title or api_title``, so a ``PR_TITLE``
    from the event payload wins for the life of the run that received it. The
    workflow cancels nothing (it carries no ``concurrency`` block, pinned by
    ``test_the_gate_does_not_cancel_its_own_run``), so a run started by
    ``synchronize`` can still be going when an ``edited`` run passes; handed the
    payload title it would report the pre-edit verdict afterwards and strand a red
    context on a head that had already cleared.

    Not passing it makes ``resolve_pull_request``'s copy authoritative, so any run
    on a pull request computes the current verdict and two runs on one head agree.
    The script keeps reading ``PR_TITLE`` and ``--title`` for the command line,
    which is what the tests above exercise; this pin is about the workflow.
    """
    workflow = _workflow()
    setters = [
        line.strip()
        for line in workflow.splitlines()
        if not line.lstrip().startswith("#") and re.match(r"^\s*PR_TITLE\s*:", line)
    ]
    assert not setters, f"the workflow still hands the script a title: {setters}"

    invocation = [line for line in workflow.splitlines() if "check_closing_reference.py" in line]
    assert invocation
    assert all("${{" not in line for line in invocation)


def test_the_gate_does_not_cancel_its_own_run() -> None:
    """The premise the pin above leans on, and #2216's own finding.

    This workflow is the only one here started by an activity type that cannot
    change the head sha, so it is the only one that can have two runs on one head
    -- and a concurrency group keyed on the pull request number would hold both.
    Cancelling one leaves a permanent ``CANCELLED`` context on a head that
    satisfies the check, which is unclearable without a push and so defeats the
    self-clearing property ``test_the_gate_reruns_when_the_title_or_body_is_edited``
    exists to protect. The fleet-wide form of this rule is
    ``test_an_exempt_workflow_cannot_cancel_its_own_run``.
    """
    workflow = _workflow()
    assert not re.search(r"^concurrency:", workflow, re.MULTILINE)
    assert not re.search(r"^\s*cancel-in-progress:", workflow, re.MULTILINE)
