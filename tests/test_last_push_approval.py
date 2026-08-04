"""Pins the last-push-approval classifier against four pull requests measured in this repo.

The interesting property of this check is not that it computes a boolean. It is
that it separates two states which every field a status sweep reads renders
identically -- ``reviewDecision: REVIEW_REQUIRED`` with
``mergeStateStatus: BLOCKED`` describes both "nobody has reviewed this yet" and
"the only approval can never count". So the fixtures below are the real
observations, not invented ones, and the control pair is what carries the
argument: #1920 and #1722 share a pusher and differ only in whether the
approving account is a different one. #1920 merged; #1722 has been blocked since
2026-08-01.

See scripts/check_last_push_approval.py, issue #1905, and the "PR Workflow"
section of AGENTS.md.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_last_push_approval.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_last_push_approval", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()
Review = mod.Review


# ``Review`` is reached through the importlib load below, so it is a module
# attribute at runtime and not a name mypy can resolve to a type. These helpers
# are annotated ``Any`` for that reason rather than to loosen anything: the
# script itself is fully typed and checked (mypy scripts/check_last_push_approval.py).
def approved(author: str, at: str = "2026-08-01T00:00:00Z") -> Any:
    return Review(author=author, state="APPROVED", submitted_at=at)


def commented(author: str, at: str = "2026-08-01T00:00:00Z") -> Any:
    return Review(author=author, state="COMMENTED", submitted_at=at)


# --------------------------------------------------------------------------
# The four measured pull requests.
#
# pull request | triggering_actor | approved by  | commit.author.login | reviewDecision
# #1894        | yinsong1986      | cagataycali  | yinsong1986         | APPROVED
# #1920        | cagataycali      | yinsong1986  | None                | APPROVED
# #1722        | cagataycali      | cagataycali  | cagataycali         | REVIEW_REQUIRED
# #1035        | cagataycali      | cagataycali  | cagataycali         | REVIEW_REQUIRED
# --------------------------------------------------------------------------

MEASURED = [
    pytest.param("yinsong1986", ["cagataycali"], mod.SATISFIED, id="pr1894-author-pushed-maintainer-approved"),
    pytest.param("cagataycali", ["yinsong1986"], mod.SATISFIED, id="pr1920-maintainer-pushed-other-approved"),
    pytest.param("cagataycali", ["cagataycali"], mod.PUSHER_ONLY_APPROVAL, id="pr1722-pusher-is-sole-approver"),
    pytest.param("cagataycali", ["cagataycali"], mod.PUSHER_ONLY_APPROVAL, id="pr1035-pusher-is-sole-approver"),
]


@pytest.mark.parametrize("pusher,approvers,expected", MEASURED)
def test_the_measured_pull_requests_classify_as_observed(pusher, approvers, expected):
    verdict = mod.classify(pusher, [approved(a) for a in approvers])
    assert verdict.outcome == expected


def test_the_control_pair_differs_only_in_who_approved():
    """#1920 and #1722 share a pusher; only the approving account differs.

    This is the whole isolation argument. If a future change made the verdict
    depend on anything else about these two, one of these assertions moves.
    """
    pusher = "cagataycali"
    merged = mod.classify(pusher, [approved("yinsong1986")])
    blocked = mod.classify(pusher, [approved("cagataycali")])

    assert merged.pusher == blocked.pusher == pusher
    assert merged.outcome == mod.SATISFIED
    assert blocked.outcome == mod.PUSHER_ONLY_APPROVAL
    assert not merged.is_finding
    assert blocked.is_finding


def test_an_unreviewed_pull_request_is_not_a_finding():
    """The state this check exists to distinguish itself from.

    #1899 and #1901 both read REVIEW_REQUIRED / BLOCKED with no approval and a
    head pushed by their own author. They are waiting on a reviewer, which is
    ordinary and already visible, so a red check here would be noise on every
    open pull request in the repository.
    """
    verdict = mod.classify("yinsong1986", [commented("github-advanced-security"), commented("yinsong1986")])
    assert verdict.outcome == mod.AWAITING_FIRST_REVIEW
    assert verdict.is_finding is False
    assert verdict.approvers == ()


def test_a_commented_review_does_not_retract_an_earlier_approval():
    """COMMENTED expresses no position, so it cannot supersede an approval.

    Every measured pull request here carries COMMENTED reviews from
    github-advanced-security interleaved with the human ones; if those counted
    as a position, #1894's approval would read as withdrawn and the check would
    pass a genuinely blocked branch.
    """
    reviews = [
        approved("cagataycali", "2026-08-01T10:00:00Z"),
        commented("cagataycali", "2026-08-01T11:00:00Z"),
    ]
    assert mod.current_approvers(reviews) == ("cagataycali",)


@pytest.mark.parametrize("superseding", ["CHANGES_REQUESTED", "DISMISSED"])
def test_a_later_position_supersedes_an_earlier_approval(superseding):
    reviews = [
        approved("cagataycali", "2026-08-01T10:00:00Z"),
        Review(author="cagataycali", state=superseding, submitted_at="2026-08-01T11:00:00Z"),
    ]
    assert mod.current_approvers(reviews) == ()


def test_an_approval_after_a_dismissal_counts_again():
    reviews = [
        Review(author="cagataycali", state="DISMISSED", submitted_at="2026-08-01T10:00:00Z"),
        approved("cagataycali", "2026-08-01T11:00:00Z"),
    ]
    assert mod.current_approvers(reviews) == ("cagataycali",)


def test_reviews_sharing_a_timestamp_are_ordered_by_position_in_the_list():
    """The API lists reviews chronologically; submitted_at resolves only to the second.

    #1899 carries two reviews submitted 14 seconds apart and two more within the
    same second, so a sort on the timestamp alone is not a total order and the
    surviving position would depend on dict iteration.
    """
    same = "2026-08-03T06:10:19Z"
    reviews = [
        approved("cagataycali", same),
        Review(author="cagataycali", state="CHANGES_REQUESTED", submitted_at=same),
    ]
    assert mod.current_approvers(reviews) == ()

    reviews_reversed = [
        Review(author="cagataycali", state="CHANGES_REQUESTED", submitted_at=same),
        approved("cagataycali", same),
    ]
    assert mod.current_approvers(reviews_reversed) == ("cagataycali",)


def test_a_second_approver_clears_the_finding():
    """Remedy 1 from the report, pinned: the finding is not sticky."""
    blocked = mod.classify("cagataycali", [approved("cagataycali")])
    assert blocked.outcome == mod.PUSHER_ONLY_APPROVAL

    cleared = mod.classify("cagataycali", [approved("cagataycali"), approved("yinsong1986")])
    assert cleared.outcome == mod.SATISFIED


def test_an_undetermined_pusher_is_not_a_finding():
    """A lookup that cannot attribute the push must not guess from the commit.

    #1920's head was committed under the strands-robots git identity, whose
    commit.author.login is None while its triggering_actor is cagataycali. A
    fallback to commit metadata would have read that pull request as having no
    pusher and, had the approver been the same account, as satisfied. So an
    unknown pusher is its own outcome and passes.
    """
    verdict = mod.classify(None, [approved("cagataycali")])
    assert verdict.outcome == mod.UNKNOWN_PUSHER
    assert verdict.is_finding is False
    assert verdict.pusher is None
    assert "commit metadata is not a sound substitute" in verdict.summary


def test_the_finding_summary_names_the_pusher_and_the_remedy():
    verdict = mod.classify("cagataycali", [approved("cagataycali")])
    assert "cagataycali" in verdict.summary
    assert "second approver" in verdict.summary

    report = mod.render(verdict, "strands-labs/robots", 1722, "3375c000")
    assert "pusher-only-approval" in report
    assert "A second reviewer approves" in report
    assert "Admin bypass. Not recommended" in report
    assert "3375c000" in report


def test_a_satisfied_report_names_the_approver_who_did_not_push():
    verdict = mod.classify("cagataycali", [approved("yinsong1986")])
    assert "yinsong1986" in verdict.summary
    report = mod.render(verdict, "strands-labs/robots", 1920, "d7f12fc1")
    assert mod.SATISFIED in report
    # The remedy block belongs only to a finding.
    assert "Admin bypass" not in report


def test_the_most_recent_workflow_run_names_the_pusher():
    """A re-push under a different account leaves the older runs in place.

    resolve_pusher takes the newest created_at rather than the first row, so a
    branch whose head was force-pushed by someone else is attributed to the
    account that pushed last.
    """
    payload = {
        "workflow_runs": [
            {
                "created_at": "2026-08-01T07:50:16Z",
                "event": "pull_request",
                "triggering_actor": {"login": "cagataycali"},
            },
            {"created_at": "2026-08-02T09:00:00Z", "event": "pull_request", "triggering_actor": {"login": "Vivek0712"}},
        ]
    }
    original = mod._get
    try:
        mod._get = lambda url, token: payload
        assert mod.resolve_pusher("strands-labs/robots", "deadbeef", "t") == "Vivek0712"
    finally:
        mod._get = original


def test_a_review_triggered_run_does_not_name_the_pusher():
    """The check's own `pull_request_review` trigger poisons the field it reads.

    Verbatim from #1921, this check's own pull request. Adding the review
    trigger creates a run on the same head sha attributed to the **reviewer**,
    newer than every run the push itself produced. Reading the newest run
    unfiltered named yinsong1986 as the pusher of a commit cagataycali pushed,
    and reported `pusher-only-approval` while GitHub read the pull request
    `APPROVED` / `UNSTABLE` -- a false positive that would have fired on every
    approved pull request in the repository.
    """
    payload = {
        "workflow_runs": [
            {
                "created_at": "2026-08-03T22:45:21Z",
                "event": "pull_request",
                "triggering_actor": {"login": "cagataycali"},
                "name": "Last Push Approval Check",
            },
            {
                "created_at": "2026-08-03T22:45:21Z",
                "event": "pull_request",
                "triggering_actor": {"login": "cagataycali"},
                "name": "Pull Request and Push Action",
            },
            {
                "created_at": "2026-08-03T23:11:27Z",
                "event": "pull_request_review",
                "triggering_actor": {"login": "yinsong1986"},
                "name": "Last Push Approval Check",
            },
        ]
    }
    original = mod._get
    try:
        mod._get = lambda url, token: payload
        pusher = mod.resolve_pusher("strands-labs/robots", "2714eacf", "t")
    finally:
        mod._get = original

    assert pusher == "cagataycali"
    # And therefore the verdict agrees with GitHub rather than contradicting it.
    assert mod.classify(pusher, [approved("yinsong1986")]).outcome == mod.SATISFIED


@pytest.mark.parametrize(
    "event",
    ["pull_request_review", "pull_request_review_comment", "issue_comment", "workflow_dispatch", "schedule"],
)
def test_only_push_producing_events_attribute_a_pusher(event):
    """Anything a push did not cause names whoever caused it instead."""
    original = mod._get
    try:
        mod._get = lambda url, token: {
            "workflow_runs": [
                {"created_at": "2026-08-03T23:00:00Z", "event": event, "triggering_actor": {"login": "someone-else"}}
            ]
        }
        assert mod.resolve_pusher("strands-labs/robots", "deadbeef", "t") is None
    finally:
        mod._get = original


def test_a_push_event_run_attributes_the_pusher():
    """A branch in this repository produces `push` runs rather than `pull_request` ones."""
    original = mod._get
    try:
        mod._get = lambda url, token: {
            "workflow_runs": [
                {"created_at": "2026-08-03T23:00:00Z", "event": "push", "triggering_actor": {"login": "cagataycali"}}
            ]
        }
        assert mod.resolve_pusher("strands-labs/robots", "deadbeef", "t") == "cagataycali"
    finally:
        mod._get = original


def test_a_head_with_no_workflow_run_yields_no_pusher():
    original = mod._get
    try:
        mod._get = lambda url, token: {"workflow_runs": []}
        assert mod.resolve_pusher("strands-labs/robots", "deadbeef", "t") is None
    finally:
        mod._get = original


def test_a_run_without_a_triggering_actor_is_skipped_not_trusted():
    original = mod._get
    try:
        mod._get = lambda url, token: {
            "workflow_runs": [
                {"created_at": "2026-08-02T09:00:00Z", "event": "pull_request", "triggering_actor": None},
                {
                    "created_at": "2026-08-01T09:00:00Z",
                    "event": "pull_request",
                    "triggering_actor": {"login": "cagataycali"},
                },
            ]
        }
        assert mod.resolve_pusher("strands-labs/robots", "deadbeef", "t") == "cagataycali"
    finally:
        mod._get = original


def test_reviews_are_parsed_from_the_rest_shape():
    original = mod._get
    try:
        mod._get = lambda url, token: [
            {
                "user": {"login": "github-advanced-security"},
                "state": "COMMENTED",
                "submitted_at": "2026-08-03T21:39:17Z",
            },
            {"user": {"login": "yinsong1986"}, "state": "APPROVED", "submitted_at": "2026-08-03T21:56:05Z"},
        ]
        reviews = mod.resolve_reviews("strands-labs/robots", 1920, "t")
    finally:
        mod._get = original

    assert mod.current_approvers(reviews) == ("yinsong1986",)


def test_a_lookup_failure_reports_nothing_rather_than_a_finding(capsys, monkeypatch):
    """An API error must not present as a deadlock.

    The check has no way to tell a rate limit from a green pull request, and a
    red X that a branch cannot clear is worse than no signal at all.
    """
    monkeypatch.setattr(mod, "resolve_head_sha", lambda *a, **k: "deadbeef")

    def boom(*_args, **_kwargs):
        raise mod.urllib.error.URLError("rate limited")

    monkeypatch.setattr(mod, "resolve_pusher", boom)
    exit_code = mod.main(["--repo", "strands-labs/robots", "--pr", "1722", "--token", "t"])
    assert exit_code == 0
    assert "reporting nothing" in capsys.readouterr().err


def test_the_exit_status_is_one_only_for_the_finding(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "resolve_head_sha", lambda *a, **k: "3375c000")
    monkeypatch.setattr(mod, "resolve_pusher", lambda *a, **k: "cagataycali")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    monkeypatch.setattr(mod, "resolve_reviews", lambda *a, **k: [approved("cagataycali")])
    assert mod.main(["--repo", "strands-labs/robots", "--pr", "1722", "--token", "t"]) == 1

    monkeypatch.setattr(mod, "resolve_reviews", lambda *a, **k: [approved("yinsong1986")])
    assert mod.main(["--repo", "strands-labs/robots", "--pr", "1920", "--token", "t"]) == 0

    monkeypatch.setattr(mod, "resolve_reviews", lambda *a, **k: [commented("yinsong1986")])
    assert mod.main(["--repo", "strands-labs/robots", "--pr", "1899", "--token", "t"]) == 0


def test_no_report_string_carries_a_non_ascii_character():
    """Log and report strings stay ASCII, per the repo's Unicode hygiene rule."""
    for approvers in (["cagataycali"], ["yinsong1986"], []):
        verdict = mod.classify("cagataycali", [approved(a) for a in approvers])
        report = mod.render(verdict, "strands-labs/robots", 1722, "3375c000")
        report.encode("ascii")
    mod.render(mod.classify(None, []), "strands-labs/robots", 1722, "3375c000").encode("ascii")


# --------------------------------------------------------------------------
# The workflow's bootstrapping guard.
#
# The job checks out the *base* branch, not the pull request head, so that a
# branch forked before this check landed does not die on a missing script
# (#1791). That has a mirror-image hole which was not theoretical: the base does
# not carry the script either until this change lands, so the introducing pull
# request exited 2 with `can't open file`, and the checks UI renders exit 2 and
# exit 1 as the same red X -- accusing a branch of a deadlock the job never
# computed. The guard makes that case pass, which is the same rule the script
# applies to an undetermined pusher and to a failed lookup.
# --------------------------------------------------------------------------

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "last-push-approval.yml"


def _run_step_body() -> str:
    """The shell body of the workflow's checking step, as text.

    Read with a line scan rather than a YAML parse because ``pyyaml`` is an
    optional dependency here, matching tests/test_pull_request_trigger_types.py.
    """
    lines = _WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "run: |")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith(" " * 10):
            break
        body.append(line[10:] if len(line) > 10 else "")
    return "\n".join(body)


def test_the_workflow_runs_the_script_from_a_guarded_path():
    body = _run_step_body()
    assert "check_last_push_approval.py" in body
    assert "if [ ! -f scripts/check_last_push_approval.py ]" in body
    assert "exit 0" in body


def test_the_workflow_passes_when_the_base_lacks_the_script(tmp_path):
    """Exit 0, not 2, when the checked-out base predates the script.

    Executes the workflow's own shell body in a directory without the script,
    so the pin is on the shipped text rather than on a paraphrase of it.
    """
    import subprocess

    body = _run_step_body()
    result = subprocess.run(
        ["sh", "-c", body],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"BASE_REF": "main", "GITHUB_REPOSITORY": "strands-labs/robots", "PATH": os.environ["PATH"]},
    )
    assert result.returncode == 0, result.stderr
    assert "nothing to report" in result.stdout


def test_the_guard_does_not_swallow_a_real_finding(tmp_path):
    """With the script present, the guard is transparent and the exit status is the script's.

    Otherwise the fix for the bootstrapping case would have turned the whole
    check into a permanent no-op, which is the failure mode that would be
    hardest to notice: a green check that never looks at anything.
    """
    import shutil
    import subprocess
    import textwrap

    (tmp_path / "scripts").mkdir()
    stub = tmp_path / "scripts" / "check_last_push_approval.py"
    stub.write_text(
        textwrap.dedent("""
        import sys
        print("stub ran")
        sys.exit(1)
    """)
    )
    assert shutil.which("sh")

    result = subprocess.run(
        ["sh", "-c", _run_step_body()],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"BASE_REF": "main", "GITHUB_REPOSITORY": "strands-labs/robots", "PATH": os.environ["PATH"]},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "stub ran" in result.stdout
    assert "nothing to report" not in result.stdout
