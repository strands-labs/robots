"""Pins the head-record classifier against the pull request that produced it.

The interesting property of this check is not that it compares two strings. It
is that the two strings are meant to be the same string, and every other signal
on the pull request is derived from the one that can be wrong. Measured on
#2508, which sat approved, green and unmergeable for over five hours:

    recorded ``headRefOid``   21ea097e   pushed 07:56:47   8 check suites, green
    tip of the head branch    271ec912   pushed 13:58:16   0 check suites

While that lag stood, ``reviewDecision`` read ``APPROVED``, the required check
read ``SUCCESS``, no thread was unresolved, and both sweep scripts reported no
finding -- each of them correctly, about ``21ea097e``. The merge refused with
``Head branch is out of date``, naming a staleness gate this repository does not
have: the ``default`` ruleset sets ``strict_required_status_checks_policy:
false`` and ``main`` has no classic branch protection.

So the fixtures below are the real observation and its control, and the control
is what carries the argument: the *same* pull request after a close/reopen
reconciled the record reads ``current`` on identical code, with the only change
being that the recorded head caught up to the tip.

The unresolvable cases are pinned for the opposite reason. A head this check
could not read must not be reported as a head that matches, because that is the
one failure mode that would silently turn the check into a no-op agreeing with
everything.

See scripts/check_pr_head_is_current.py, issue #2538, and the "PR Workflow"
section of AGENTS.md.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_pr_head_is_current.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_pr_head_is_current", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()

# The two heads #2508 carried, verbatim.
RECORDED = "21ea097e470a63002f9d0e63b22d2b1838770d21"
TIP = "271ec912b97857d8264a748d85d8f33eb3c2de5a"


class TestTheMeasuredPullRequest:
    """#2508 before and after the record was reconciled."""

    def test_a_recorded_head_behind_the_tip_is_the_finding(self) -> None:
        verdict = mod.classify(RECORDED, TIP)
        assert verdict.outcome == mod.STALE_HEAD_RECORD
        assert verdict.is_finding

    def test_the_same_pull_request_reads_current_once_reconciled(self) -> None:
        """The control. Only the recorded head moved; the branch tip did not."""
        verdict = mod.classify(TIP, TIP)
        assert verdict.outcome == mod.CURRENT
        assert not verdict.is_finding

    def test_the_finding_names_both_commits_and_refuses_the_push_remedy(self) -> None:
        """The summary has to carry the two facts that stop the expensive reflex.

        A reader who sees only "out of date" refreshes the branch, which on a
        contributor's branch spends the approval under
        ``require_last_push_approval``. So the summary names the remedy and names
        what not to do.
        """
        summary = mod.classify(RECORDED, TIP).summary
        assert RECORDED[:8] in summary
        assert TIP[:8] in summary
        assert "Reopen" in summary
        assert "do not push" in summary


class TestAnUnreadableHeadIsNotAMatch:
    """An unresolvable head is its own outcome, never a pass."""

    @pytest.mark.parametrize(
        ("recorded", "tip"),
        [
            (RECORDED, None),  # deleted fork or branch, or an unreadable ref
            (None, TIP),  # the pull request reported no head at all
            (None, None),
            ("", ""),  # both absent must not compare equal into ``current``
        ],
    )
    def test_missing_either_side_is_unresolvable(self, recorded, tip) -> None:
        verdict = mod.classify(recorded, tip)
        assert verdict.outcome == mod.UNRESOLVABLE_HEAD
        assert not verdict.is_finding

    def test_two_absent_heads_do_not_read_as_current(self) -> None:
        """The specific way this check would become a no-op that always agrees."""
        assert mod.classify(None, None).outcome != mod.CURRENT
        assert mod.classify("", "").outcome != mod.CURRENT


class TestTheTipIsReadFromTheHeadRepository:
    """The pull request's own view of its head cannot be the source here.

    That field is the value under suspicion, so a tip read through it would
    compare a field against itself and report ``current`` on exactly the
    pull requests this check exists to find.
    """

    def test_branch_tip_queries_the_head_repository_ref(self, monkeypatch) -> None:
        seen: dict[str, object] = {}

        def fake_graphql(query, variables, token):
            seen["query"] = query
            seen["variables"] = variables
            return {"repository": {"ref": {"target": {"oid": TIP}}}}

        monkeypatch.setattr(mod, "_graphql", fake_graphql)
        assert mod.branch_tip("yinsong1986/robots", "docs/data-augmentation-notebook", "t") == TIP

        # The ref is read off the head repository, by owner and name written out,
        # and never through pullRequest { headRefOid }.
        assert seen["variables"] == {
            "owner": "yinsong1986",
            "name": "robots",
            "ref": "refs/heads/docs/data-augmentation-notebook",
        }
        assert "headRefOid" not in str(seen["query"])
        assert "ref(qualifiedName:" in str(seen["query"]).replace(" ", "")

    def test_an_unreadable_ref_returns_none_rather_than_raising(self, monkeypatch) -> None:
        """A deleted fork must not abort a sweep over the other pull requests."""

        def boom(query, variables, token):
            raise ValueError("Could not resolve to a Repository")

        monkeypatch.setattr(mod, "_graphql", boom)
        assert mod.branch_tip("gone/robots", "some-branch", "t") is None

    def test_a_malformed_head_repository_is_unresolvable(self, monkeypatch) -> None:
        def unreached(query, variables, token):  # pragma: no cover - must not run
            raise AssertionError("a malformed repository must not reach the API")

        monkeypatch.setattr(mod, "_graphql", unreached)
        assert mod.branch_tip("not-a-repo", "branch", "t") is None


class TestEvaluateAndReport:
    """The node shapes both queries return, and what the reports must say."""

    @staticmethod
    def _node(**over):
        node = {
            "number": 2508,
            "isDraft": False,
            "headRefName": "docs/data-augmentation-notebook",
            "headRefOid": RECORDED,
            "headRepository": {"nameWithOwner": "yinsong1986/robots"},
            "mergeable": "UNKNOWN",
            "mergeStateStatus": "UNKNOWN",
            "reviewDecision": "APPROVED",
        }
        node.update(over)
        return node

    def test_the_lag_is_a_finding_even_though_every_other_field_reads_ready(self, monkeypatch) -> None:
        """The whole point: APPROVED plus UNKNOWN plus a moved tip is the finding."""
        monkeypatch.setattr(mod, "branch_tip", lambda *a, **k: TIP)
        row = mod.evaluate(self._node(), "t")
        assert row.verdict.is_finding
        assert row.review_decision == "APPROVED"
        assert row.mergeable == "UNKNOWN"

    def test_a_null_head_repository_is_unresolvable_not_a_finding(self, monkeypatch) -> None:
        monkeypatch.setattr(mod, "branch_tip", lambda *a, **k: TIP)
        row = mod.evaluate(self._node(headRepository=None), "t")
        assert row.verdict.outcome == mod.UNRESOLVABLE_HEAD

    def test_the_sweep_report_names_every_finding_and_the_remedy(self) -> None:
        stale = mod.Row(
            pr=2508,
            verdict=mod.classify(RECORDED, TIP),
            head_repo="yinsong1986/robots",
            head_ref="docs/data-augmentation-notebook",
            mergeable="UNKNOWN",
            merge_state_status="UNKNOWN",
            review_decision="APPROVED",
        )
        report = mod.render_sweep("strands-labs/robots", [stale], [])
        assert "#2508" in report
        assert "### What clears this" in report
        assert "Close and reopen" in report

    def test_a_clean_sweep_says_so_without_the_remedy_block(self) -> None:
        """A remedy printed under a clean report trains the reader to skip it."""
        current = mod.Row(
            pr=2536,
            verdict=mod.classify(TIP, TIP),
            head_repo="cagataycali/robots",
            head_ref="fix/benchmark-refused-default-robot",
            mergeable="MERGEABLE",
            merge_state_status="BLOCKED",
            review_decision="REVIEW_REQUIRED",
        )
        report = mod.render_sweep("strands-labs/robots", [current], [])
        assert "Every recorded head is its branch's tip." in report
        assert "### What clears this" not in report

    def test_an_unresolvable_row_is_named_rather_than_counted_clean(self) -> None:
        row = mod.Row(
            pr=1234,
            verdict=mod.classify(RECORDED, None),
            head_repo="gone/robots",
            head_ref="lost-branch",
            mergeable="UNKNOWN",
            merge_state_status="UNKNOWN",
            review_decision="REVIEW_REQUIRED",
        )
        report = mod.render_sweep("strands-labs/robots", [row], [])
        assert "#1234" in report
        assert "Unresolvable" in report


class TestExitStatus:
    """Exit 1 only on a finding, so a red run means one specific thing."""

    def test_a_missing_token_does_not_fail_the_caller(self) -> None:
        assert mod.main(["--repo", "strands-labs/robots", "--pr", "2508", "--token", ""]) == 0

    def test_a_missing_repo_does_not_fail_the_caller(self) -> None:
        assert mod.main(["--repo", "", "--all-open", "--token", "t"]) == 0

    def test_neither_pr_nor_all_open_does_not_fail_the_caller(self) -> None:
        assert mod.main(["--repo", "strands-labs/robots", "--token", "t"]) == 0

    def test_a_stale_record_exits_one(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(mod, "fetch_one", lambda *a, **k: TestEvaluateAndReport._node())
        monkeypatch.setattr(mod, "branch_tip", lambda *a, **k: TIP)
        code = mod.main(["--repo", "strands-labs/robots", "--pr", "2508", "--token", "t"])
        assert code == 1
        out = capsys.readouterr().out
        assert "::warning title=Recorded head is not the branch tip::" in out

    def test_a_current_record_exits_zero_and_annotates_nothing(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(mod, "fetch_one", lambda *a, **k: TestEvaluateAndReport._node(headRefOid=TIP))
        monkeypatch.setattr(mod, "branch_tip", lambda *a, **k: TIP)
        code = mod.main(["--repo", "strands-labs/robots", "--pr", "2508", "--token", "t"])
        assert code == 0
        assert "::warning" not in capsys.readouterr().out

    def test_a_draft_is_excluded_from_the_sweep(self, monkeypatch) -> None:
        """A draft cannot merge whatever its record says."""
        monkeypatch.setattr(
            mod,
            "fetch_open",
            lambda *a, **k: [TestEvaluateAndReport._node(isDraft=True)],
        )
        monkeypatch.setattr(mod, "branch_tip", lambda *a, **k: TIP)
        assert mod.main(["--repo", "strands-labs/robots", "--all-open", "--token", "t"]) == 0

    def test_one_unreadable_pull_request_does_not_suppress_a_finding(self, monkeypatch, capsys) -> None:
        """A sweep's value is the population, so one bad row must not end it."""
        good = TestEvaluateAndReport._node()
        bad = TestEvaluateAndReport._node(number=9999)

        def flaky(node, token):
            if node["number"] == 9999:
                raise ValueError("rate limited")
            return mod.Row(
                pr=node["number"],
                verdict=mod.classify(RECORDED, TIP),
                head_repo="yinsong1986/robots",
                head_ref="docs/data-augmentation-notebook",
                mergeable="UNKNOWN",
                merge_state_status="UNKNOWN",
                review_decision="APPROVED",
            )

        monkeypatch.setattr(mod, "fetch_open", lambda *a, **k: [bad, good])
        monkeypatch.setattr(mod, "evaluate", flaky)
        code = mod.main(["--repo", "strands-labs/robots", "--all-open", "--token", "t"])
        assert code == 1
        out = capsys.readouterr().out
        assert "#2508" in out
        assert "#9999" in out
