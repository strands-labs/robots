"""Pins the merge-blocker classifier against pull requests measured in this repo.

The property under test is not that a pull request is blocked -- ``mergeStateStatus``
already says that, and says nothing else. It is that the *rule* which is
unsatisfied is named, because that is what decides who owes the next action. Three
pull requests measured on 2026-08-21 all read ``blocked`` and needed three
different people:

===========  ==============================  ==========================
pull request unsatisfied                     who could clear it
===========  ==============================  ==========================
#2566        one unresolved review thread    the author
#2574        nothing -- a stale computation   anyone, by retrying
#2497        no approving review yet         any reviewer
===========  ==============================  ==========================

#2566 and #2574 each sat idle after approval (31 and 45 minutes) because a
scheduled pass read APPROVED plus green plus BLOCKED and concluded the next move
was somebody else's. So the fixtures below are those observations, and the
control triple is what carries the argument: the three differ in exactly one
input each and produce three different owners.

See scripts/check_merge_blockers.py, issue #1905, and the "PR Workflow" section
of AGENTS.md.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_merge_blockers.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_merge_blockers", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()

# Reached through the importlib load above, so these are module attributes at
# runtime rather than names mypy can resolve. The script itself is fully typed
# and checked (mypy scripts/check_merge_blockers.py); ``hatch run lint`` covers
# only the package and its tests.
Ruleset: Any = mod.Ruleset
PullRequestState: Any = mod.PullRequestState
Blocker: Any = mod.Blocker

REQUIRED = "call-test-lint / Test and Lint"


# The ruleset this repository's ``main`` actually carries, as returned by
# GET /repos/strands-labs/robots/rules/branches/main on 2026-08-21. Trimmed to
# the rule types the payload contains, values verbatim.
MEASURED_RULES_PAYLOAD: list[dict[str, Any]] = [
    {"type": "deletion", "parameters": {}},
    {
        "type": "pull_request",
        "parameters": {
            "required_approving_review_count": 1,
            "dismiss_stale_reviews_on_push": True,
            "required_reviewers": [],
            "require_code_owner_review": True,
            "require_last_push_approval": True,
            "required_review_thread_resolution": True,
            "require_extra_approval_for_unattributed_changes": True,
            "allowed_merge_methods": ["squash"],
        },
    },
    {"type": "non_fast_forward", "parameters": {}},
    {
        "type": "required_status_checks",
        "parameters": {
            "strict_required_status_checks_policy": False,
            "required_status_checks": [{"context": REQUIRED, "integration_id": 15368}],
        },
    },
]

MAIN = mod.parse_ruleset(MEASURED_RULES_PAYLOAD)


def state(**overrides: Any) -> Any:
    """A pull request that satisfies every rule, so each test varies one input."""
    base: dict[str, Any] = {
        "number": 1,
        "head_sha": "0123456789abcdef",
        "base_ref": "main",
        "draft": False,
        "mergeable": True,
        "merge_state": "blocked",
        "unresolved_threads": 0,
        "check_conclusions": {REQUIRED: "success"},
        "approvers": ("a-reviewer",),
        "pusher": "the-author",
    }
    base.update(overrides)
    return PullRequestState(**base)


def outcomes(blockers: Any) -> list[str]:
    return [b.outcome for b in blockers]


# --------------------------------------------------------------------------
# The ruleset is read, not assumed.
# --------------------------------------------------------------------------


def test_the_measured_ruleset_parses_to_the_rules_in_force() -> None:
    assert MAIN.required_approving_review_count == 1
    assert MAIN.required_review_thread_resolution is True
    assert MAIN.require_last_push_approval is True
    assert MAIN.required_contexts == (REQUIRED,)


def test_a_rule_the_branch_carries_but_this_check_cannot_answer_is_named() -> None:
    """Silence about a rule in force would read as the branch not having it.

    Both of these are set on ``main``. ``require_code_owner_review`` is vacuous
    here because the repository has no CODEOWNERS file, and
    ``require_extra_approval_for_unattributed_changes`` is not answerable from
    the fields this check reads. Neither may be silently dropped.
    """
    assert MAIN.unevaluated == (
        "require_code_owner_review",
        "require_extra_approval_for_unattributed_changes",
    )


def test_an_unknown_rule_type_does_not_break_the_parse() -> None:
    """A ruleset gaining a rule must not turn this check red."""
    rules = mod.parse_ruleset([*MEASURED_RULES_PAYLOAD, {"type": "future_rule", "parameters": {"x": 1}}])
    assert rules.required_contexts == (REQUIRED,)
    assert rules.required_approving_review_count == 1


def test_a_rule_the_branch_does_not_carry_is_never_named_as_a_blocker() -> None:
    """With thread resolution off, an unresolved thread is not a blocker."""
    without = Ruleset(required_approving_review_count=1, required_review_thread_resolution=False)
    assert mod.UNRESOLVED_THREADS not in outcomes(mod.evaluate(state(unresolved_threads=3), without))


def test_a_malformed_ruleset_payload_yields_no_rules_rather_than_raising() -> None:
    assert mod.parse_ruleset({"unexpected": "shape"}) == Ruleset()
    assert mod.parse_ruleset([None, "junk"]) == Ruleset()


# --------------------------------------------------------------------------
# The three measured pull requests: one input each, three owners.
# --------------------------------------------------------------------------


def test_2566_an_unresolved_thread_blocks_an_approved_green_pull_request() -> None:
    """The author owes this, and nothing on the pull request said so.

    #2566 was APPROVED with every check SUCCESS and read ``blocked`` purely
    because one thread was open -- a thread whose fix had already landed and
    which the reviewer had approved past.
    """
    blockers = mod.evaluate(state(unresolved_threads=1), MAIN)
    assert outcomes(blockers) == [mod.UNRESOLVED_THREADS]
    assert blockers[0].rule == "required_review_thread_resolution"
    assert blockers[0].owed_by == mod.AUTHOR
    assert blockers[0].is_finding is True


def test_2574_every_rule_satisfied_and_still_blocked_is_its_own_answer() -> None:
    """The #2574 case: APPROVED, 12 checks SUCCESS, zero threads, still ``blocked``.

    ``mergePullRequest`` refused with "Pull Request is not mergeable" and the
    REST merge then succeeded on the first attempt with no state having changed.
    So "no rule is unsatisfied" is a finding with a remedy, not an absence of
    one, and must not render as an empty report.
    """
    blockers = mod.evaluate(state(), MAIN)
    assert outcomes(blockers) == [mod.NO_UNSATISFIED_RULE]
    assert blockers[0].owed_by == mod.ANYONE
    assert blockers[0].is_finding is True


def test_2497_a_missing_first_review_is_reported_but_is_not_a_finding() -> None:
    """The ordinary state. A finding that fires on it would mean nothing."""
    blockers = mod.evaluate(state(approvers=()), MAIN)
    assert outcomes(blockers) == [mod.MISSING_APPROVAL]
    assert blockers[0].owed_by == mod.REVIEWER
    assert blockers[0].is_finding is False


def test_the_measured_triple_produces_three_different_owners() -> None:
    """The control: one varied input each, three parties owing the next action."""
    owners = [
        mod.primary(mod.evaluate(state(unresolved_threads=1), MAIN)).owed_by,
        mod.primary(mod.evaluate(state(), MAIN)).owed_by,
        mod.primary(mod.evaluate(state(approvers=()), MAIN)).owed_by,
    ]
    assert owners == [mod.AUTHOR, mod.ANYONE, mod.REVIEWER]
    assert len(set(owners)) == 3


# --------------------------------------------------------------------------
# Two further pull requests measured live on 2026-08-21.
# --------------------------------------------------------------------------


def test_1035_names_the_conflict_ahead_of_the_approval_it_sits_upstream_of() -> None:
    """#1035: CONFLICTING/DIRTY, one approval, from the account that pushed the head.

    Both rules are unsatisfied, but only one is actionable: #1905 records that an
    approval here is "necessary but no longer sufficient". So the conflict must
    own the next action, and the author must own the conflict.
    """
    blockers = mod.evaluate(
        state(mergeable=False, merge_state="dirty", approvers=("cagataycali",), pusher="cagataycali"),
        MAIN,
    )
    assert outcomes(blockers) == [mod.MERGE_CONFLICT, mod.PUSHER_ONLY_APPROVAL]
    assert mod.primary(blockers).outcome == mod.MERGE_CONFLICT
    assert mod.primary(blockers).owed_by == mod.AUTHOR


def test_2480_a_pending_check_does_not_mask_a_reviewer_who_can_act_now() -> None:
    """#2480: required check still running, no approval.

    The check is owed by nobody and binds first. Reporting that as the next
    action would park a pull request a reviewer could approve immediately.
    """
    blockers = mod.evaluate(state(approvers=(), check_conclusions={REQUIRED: None}), MAIN)
    assert outcomes(blockers) == [mod.REQUIRED_CHECK_PENDING, mod.MISSING_APPROVAL]
    assert blockers[0].owed_by == mod.NOBODY
    assert mod.primary(blockers).outcome == mod.MISSING_APPROVAL
    assert mod.primary(blockers).owed_by == mod.REVIEWER


# --------------------------------------------------------------------------
# The required check: three states, three owners.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("conclusions", "expected", "owner"),
    [
        ({REQUIRED: "failure"}, "required-check-failing", mod.AUTHOR),
        ({REQUIRED: "timed_out"}, "required-check-failing", mod.AUTHOR),
        ({REQUIRED: "cancelled"}, "required-check-failing", mod.AUTHOR),
        ({REQUIRED: None}, "required-check-pending", mod.NOBODY),
        ({}, "required-check-absent", mod.MAINTAINER),
        ({"some other check": "failure"}, "required-check-absent", mod.MAINTAINER),
    ],
)
def test_the_required_check_states_are_separated_by_who_clears_them(
    conclusions: dict[str, str | None], expected: str, owner: str
) -> None:
    blockers = mod.evaluate(state(check_conclusions=conclusions), MAIN)
    assert outcomes(blockers) == [expected]
    assert blockers[0].owed_by == owner


@pytest.mark.parametrize("conclusion", ["success", "neutral", "skipped"])
def test_a_check_that_did_not_fail_is_not_a_blocker(conclusion: str) -> None:
    """CodeQL reports NEUTRAL and ``deploy`` reports SKIPPED on a green head."""
    assert outcomes(mod.evaluate(state(check_conclusions={REQUIRED: conclusion}), MAIN)) == [mod.NO_UNSATISFIED_RULE]


def test_an_absent_required_check_names_the_held_fork_run() -> None:
    """A fork run held at ``action_required`` reads the same as never having run.

    #1722 carried nine such runs, and three passes over #1905 described it as a
    missing run rather than a held one. The remedy differs: authorisation, not a
    push, and it consumes no approval.
    """
    blockers = mod.evaluate(state(check_conclusions={}), MAIN)
    assert "action_required" in blockers[0].detail


# --------------------------------------------------------------------------
# Approvals, composed from the sibling rather than re-derived.
# --------------------------------------------------------------------------


def test_the_approval_semantics_come_from_the_sibling_check() -> None:
    """One owner for what a "current" approval is, so the two cannot drift.

    ``require_last_push_approval`` and the position-state rules belong to
    scripts/check_last_push_approval.py. This check needs the same primitive to
    count approvals at all, so it imports it. If that import is ever replaced by
    a copy, this fails.
    """
    # Asserted by defining module rather than by object identity: another test
    # file loading the sibling replaces the sys.modules entry, so identity is a
    # statement about import order and this is a statement about where the code
    # lives.
    for shared in (mod.current_approvers, mod.resolve_pusher, mod.resolve_reviews):
        assert shared.__module__ == "check_last_push_approval", shared
    assert mod._SIBLING.name == "check_last_push_approval.py"
    assert mod._SIBLING.is_file()


def test_a_commented_review_does_not_retract_an_approval_here_either() -> None:
    """The behaviour the shared primitive exists to keep identical."""
    review = mod._approval.Review
    reviews = [
        review("a-reviewer", "APPROVED", "2026-08-01T00:00:00Z"),
        review("a-reviewer", "COMMENTED", "2026-08-02T00:00:00Z"),
    ]
    assert mod.current_approvers(reviews) == ("a-reviewer",)


def test_an_approval_from_a_third_party_satisfies_the_last_push_rule() -> None:
    blockers = mod.evaluate(state(approvers=("someone-else",), pusher="the-author"), MAIN)
    assert outcomes(blockers) == [mod.NO_UNSATISFIED_RULE]


def test_two_approvals_where_only_one_is_eligible_still_satisfies_a_count_of_one() -> None:
    blockers = mod.evaluate(state(approvers=("the-author", "someone-else"), pusher="the-author"), MAIN)
    assert outcomes(blockers) == [mod.NO_UNSATISFIED_RULE]


def test_a_higher_required_count_is_honoured() -> None:
    rules = Ruleset(required_approving_review_count=2, require_last_push_approval=True)
    blockers = mod.evaluate(state(approvers=("someone-else",), pusher="the-author"), rules)
    assert outcomes(blockers) == [mod.MISSING_APPROVAL]
    assert "1 of 2" in blockers[0].detail


def test_the_last_push_rule_is_not_applied_when_the_branch_does_not_carry_it() -> None:
    rules = Ruleset(required_approving_review_count=1, require_last_push_approval=False)
    blockers = mod.evaluate(state(approvers=("the-author",), pusher="the-author"), rules)
    # The pusher's own approval counts here, so the count is met and nothing is
    # unsatisfied. Discounting it unconditionally invented a blocker; this is the
    # regression that caught it.
    assert outcomes(blockers) == [mod.NO_UNSATISFIED_RULE]


# --------------------------------------------------------------------------
# Precedence.
# --------------------------------------------------------------------------


def test_a_draft_gates_every_rule_after_it() -> None:
    blockers = mod.evaluate(state(draft=True, approvers=(), unresolved_threads=2), MAIN)
    assert outcomes(blockers)[0] == mod.DRAFT
    assert mod.primary(blockers).outcome == mod.DRAFT
    assert mod.primary(blockers).is_gating is True


def test_a_conflict_gates_the_thread_and_approval_rules() -> None:
    blockers = mod.evaluate(state(mergeable=False, unresolved_threads=2, approvers=()), MAIN)
    assert mod.primary(blockers).outcome == mod.MERGE_CONFLICT
    assert set(outcomes(blockers)) == {mod.MERGE_CONFLICT, mod.UNRESOLVED_THREADS, mod.MISSING_APPROVAL}


def test_an_unknown_mergeability_is_not_reported_as_a_conflict() -> None:
    """GitHub returns ``None`` while it computes. Guessing turns a wait into an accusation."""
    assert mod.MERGE_CONFLICT not in outcomes(mod.evaluate(state(mergeable=None), MAIN))


def test_the_next_action_line_names_one_owner_when_a_gating_blocker_is_present() -> None:
    blockers = mod.evaluate(state(mergeable=False, approvers=()), MAIN)
    rendered = mod.render(state(mergeable=False, approvers=()), MAIN, blockers, "o/r")
    assert f"Next action is owed by {mod.AUTHOR}" in rendered
    assert "necessary but not sufficient" in rendered
    # The reviewer must not be offered as an alternative next action.
    assert f"owed by {mod.AUTHOR} and {mod.REVIEWER}" not in rendered


def test_parallel_blockers_are_reported_together() -> None:
    """A pending check and a missing approval wait on different parties at once."""
    st = state(approvers=(), check_conclusions={REQUIRED: None})
    rendered = mod.render(st, MAIN, mod.evaluate(st, MAIN), "o/r")
    assert "necessary but not sufficient" not in rendered


# --------------------------------------------------------------------------
# Review threads.
# --------------------------------------------------------------------------


def test_the_thread_count_is_named_not_merely_flagged() -> None:
    blockers = mod.evaluate(state(unresolved_threads=1), MAIN)
    assert "1 review thread unresolved" in blockers[0].detail
    blockers = mod.evaluate(state(unresolved_threads=3), MAIN)
    assert "3 review threads unresolved" in blockers[0].detail


def test_an_outdated_thread_is_not_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """An outdated thread does not block a merge, so it is not a blocker.

    #1722 carries four threads, all resolved, and reads DIRTY for an unrelated
    reason -- counting them would have named the wrong rule.
    """
    payload = {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "nodes": [
                            {"isResolved": False, "isOutdated": False},
                            {"isResolved": False, "isOutdated": True},
                            {"isResolved": True, "isOutdated": False},
                        ]
                    }
                }
            }
        }
    }
    monkeypatch.setattr(mod.urllib.request, "urlopen", _fake_urlopen(payload))
    assert mod.resolve_unresolved_threads("o/r", 1, "t") == 1


def test_a_graphql_error_raises_rather_than_reporting_zero_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Zero would be a silent pass on the exact rule #2566 was blocked by."""
    monkeypatch.setattr(mod.urllib.request, "urlopen", _fake_urlopen({"errors": [{"message": "nope"}]}))
    with pytest.raises(ValueError, match="reviewThreads"):
        mod.resolve_unresolved_threads("o/r", 1, "t")


def test_a_null_pull_request_node_yields_zero_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GraphQL returns null, not {}, for an absent node."""
    monkeypatch.setattr(
        mod.urllib.request,
        "urlopen",
        _fake_urlopen({"data": {"repository": {"pullRequest": None}}}),
    )
    assert mod.resolve_unresolved_threads("o/r", 1, "t") == 0


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def read(self) -> bytes:
        import json

        return json.dumps(self._payload).encode()

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def _fake_urlopen(payload: Any) -> Any:
    def opener(request: Any, timeout: int = 0) -> _FakeResponse:
        return _FakeResponse(payload)

    return opener


# --------------------------------------------------------------------------
# Check-run parsing.
# --------------------------------------------------------------------------


def test_a_rerun_that_succeeded_does_not_retire_a_failing_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1035's head carried the required context twice. The worst answer stands."""
    calls: list[str] = []

    def fake_get(url: str, token: str) -> Any:
        calls.append(url)
        if "check-runs" in url:
            return {
                "check_runs": [
                    {"name": REQUIRED, "conclusion": "failure"},
                    {"name": REQUIRED, "conclusion": "success"},
                ]
            }
        return {"statuses": []}

    monkeypatch.setattr(mod, "_get", fake_get)
    assert mod.resolve_check_conclusions("o/r", "abc", "t") == {REQUIRED: "failure"}


def test_a_legacy_commit_status_can_supply_a_required_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ruleset names a context without saying which surface reports it."""

    def fake_get(url: str, token: str) -> Any:
        if "check-runs" in url:
            return {"check_runs": []}
        return {"statuses": [{"context": REQUIRED, "state": "success"}]}

    monkeypatch.setattr(mod, "_get", fake_get)
    assert mod.resolve_check_conclusions("o/r", "abc", "t") == {REQUIRED: "success"}


def test_a_pending_commit_status_is_kept_pending_not_coerced_to_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(url: str, token: str) -> Any:
        if "check-runs" in url:
            return {"check_runs": []}
        return {"statuses": [{"context": REQUIRED, "state": "pending"}]}

    monkeypatch.setattr(mod, "_get", fake_get)
    assert mod.resolve_check_conclusions("o/r", "abc", "t") == {REQUIRED: None}


# --------------------------------------------------------------------------
# Reports.
# --------------------------------------------------------------------------


def test_the_stale_state_report_names_the_remedy_and_the_token_caveat() -> None:
    """The #2574 remedy is a merge attempt, and the caveat prevents a false one.

    An Actions installation token reads ``blocked`` on any pull request touching
    ``.github/workflows/**`` whatever the rules say, so this outcome must carry
    the instruction to re-read before concluding the state is stale.
    """
    rendered = mod.render(state(), MAIN, mod.evaluate(state(), MAIN), "o/r")
    assert "Attempt the merge" in rendered
    assert "not authoritative" in rendered
    assert ".github/workflows/**" in rendered
    assert "personal" in rendered


def test_the_report_calls_the_cached_merge_state_what_it_is() -> None:
    rendered = mod.render(state(), MAIN, mod.evaluate(state(), MAIN), "o/r")
    assert "merge state (cached, not authoritative)" in rendered


def test_a_satisfied_pull_request_report_does_not_carry_the_stale_remedy() -> None:
    st = state(approvers=())
    rendered = mod.render(st, MAIN, mod.evaluate(st, MAIN), "o/r")
    assert "Attempt the merge" not in rendered


def test_the_conflict_detail_says_the_green_check_describes_an_older_head() -> None:
    """#1035's required check is SUCCESS on a head that predates the conflict."""
    st = state(mergeable=False, merge_state="dirty")
    rendered = mod.render(st, MAIN, mod.evaluate(st, MAIN), "o/r")
    assert "predates the conflict" in rendered


def test_no_report_string_carries_a_non_ascii_character() -> None:
    """AGENTS.md: no emojis in user-facing strings, plain ASCII only."""
    reports = [
        mod.render(state(), MAIN, mod.evaluate(state(), MAIN), "o/r"),
        mod.render(state(approvers=()), MAIN, mod.evaluate(state(approvers=()), MAIN), "o/r"),
        mod.render(
            state(mergeable=False),
            MAIN,
            mod.evaluate(state(mergeable=False), MAIN),
            "o/r",
        ),
        mod.render_sweep([mod.SweepRow(1, mod.evaluate(state(unresolved_threads=1), MAIN))], [2], "o/r"),
        mod.render_sweep([], [], "o/r"),
        "\n".join(mod._STALE_STATE_REMEDY),
    ]
    for report in reports:
        report.encode("ascii")


# --------------------------------------------------------------------------
# Sweep.
# --------------------------------------------------------------------------


def test_the_sweep_separates_the_author_clearable_rows(capsys: pytest.CaptureFixture[str]) -> None:
    rows = [
        mod.SweepRow(1035, mod.evaluate(state(mergeable=False), MAIN)),
        mod.SweepRow(2497, mod.evaluate(state(approvers=()), MAIN)),
    ]
    rendered = mod.render_sweep(rows, [], "o/r")
    assert "1 blocked on something no reviewer can clear:** #1035" in rendered
    assert f"| #1035 | {mod.MERGE_CONFLICT} | {mod.AUTHOR} |" in rendered
    assert f"| #2497 | {mod.MISSING_APPROVAL} | {mod.REVIEWER} |" in rendered


def test_a_clean_sweep_says_so_rather_than_printing_a_bare_table() -> None:
    rendered = mod.render_sweep([], [], "o/r")
    assert "No pull request is blocked on an author-clearable rule" in rendered


def test_an_unevaluated_pull_request_is_named_rather_than_omitted() -> None:
    """A silent gap in coverage must not read as a clean sweep."""
    rendered = mod.render_sweep([], [2497], "o/r")
    assert "#2497" in rendered
    assert "Not evaluated" in rendered


def test_a_draft_pull_request_is_not_swept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "_get",
        lambda url, token: [{"number": 1, "draft": True}, {"number": 2, "draft": False}],
    )
    assert mod.resolve_open_pull_requests("o/r", "t") == [2]


def test_the_sweep_rows_are_ordered_by_pull_request_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mod,
        "_get",
        lambda url, token: [{"number": 9}, {"number": 3}, {"number": 7}],
    )
    assert mod.resolve_open_pull_requests("o/r", "t") == [3, 7, 9]


def test_the_listing_stops_on_a_short_page(monkeypatch: pytest.MonkeyPatch) -> None:
    pages: list[str] = []

    def fake_get(url: str, token: str) -> Any:
        pages.append(url)
        return [{"number": 1}]

    monkeypatch.setattr(mod, "_get", fake_get)
    assert mod.resolve_open_pull_requests("o/r", "t") == [1]
    assert len(pages) == 1


def test_one_unreadable_pull_request_does_not_suppress_the_others(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mod, "resolve_open_pull_requests", lambda repo, token: [1, 2])

    def fake_state(repo: str, pr: int, token: str) -> Any:
        if pr == 1:
            raise ValueError("unreadable")
        return state(number=2, approvers=())

    monkeypatch.setattr(mod, "resolve_state", fake_state)
    monkeypatch.setattr(mod, "resolve_ruleset", lambda repo, ref, token: MAIN)
    rows, skipped, _ = mod.sweep("o/r", "t")
    assert [r.pr for r in rows] == [2]
    assert skipped == [1]


def test_the_ruleset_is_read_once_per_base_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sweep of thirty pull requests onto one branch asks once, not thirty times."""
    monkeypatch.setattr(mod, "resolve_open_pull_requests", lambda repo, token: [1, 2, 3])
    monkeypatch.setattr(mod, "resolve_state", lambda repo, pr, token: state(number=pr))
    calls: list[str] = []

    def fake_rules(repo: str, ref: str, token: str) -> Any:
        calls.append(ref)
        return MAIN

    monkeypatch.setattr(mod, "resolve_ruleset", fake_rules)
    mod.sweep("o/r", "t")
    assert calls == ["main"]


# --------------------------------------------------------------------------
# Exit status. Same contract as the sibling, so the two read alike.
# --------------------------------------------------------------------------


def test_a_finding_exits_one_and_the_ordinary_state_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "resolve_ruleset", lambda repo, ref, token: MAIN)

    monkeypatch.setattr(mod, "resolve_state", lambda repo, pr, token: state(unresolved_threads=1))
    assert mod.main(["--repo", "o/r", "--pr", "1", "--token", "t"]) == 1

    monkeypatch.setattr(mod, "resolve_state", lambda repo, pr, token: state(approvers=()))
    assert mod.main(["--repo", "o/r", "--pr", "1", "--token", "t"]) == 0


def test_a_lookup_failure_reports_nothing_rather_than_a_blocker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Red must never mean "this branch needs another human"."""

    def boom(repo: str, pr: int, token: str) -> Any:
        raise ValueError("no")

    monkeypatch.setattr(mod, "resolve_state", boom)
    assert mod.main(["--repo", "o/r", "--pr", "1", "--token", "t"]) == 0
    assert "lookup failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--repo", "o/r", "--token", "t"], "--pr is required"),
        (["--repo", "o/r", "--token", "t", "--pr", "1", "--all-open"], "mutually exclusive"),
        (["--repo", "o/r", "--pr", "1"], "--token is required"),
        (["--pr", "1", "--token", "t"], "--repo is required"),
    ],
)
def test_an_ambiguous_invocation_is_refused_rather_than_resolved(
    argv: list[str], expected: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently ignoring either flag reads as a successful run of the other thing."""
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("PR_NUMBER", raising=False)
    with pytest.raises(SystemExit):
        mod.main(argv)
    assert expected in capsys.readouterr().err


def test_the_step_summary_receives_the_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    mod._emit("## a report")
    assert "## a report" in summary.read_text(encoding="utf-8")
