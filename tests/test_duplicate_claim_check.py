"""Pins the duplicate-claim check against the pairs this repository actually shipped.

The measurement behind the check is a corpus rather than an anecdote: over the last
100 pull requests (#1867 through #2016) three pairs claimed one issue each, and the
interesting property is not that they existed but that **every abandoned half had
already been approved**. :data:`_MEASURED_PAIRS` fixes all three in place, so the
scanner is tested against the real shapes rather than invented ones.

Two pins carry design decisions rather than behaviour:

``TestThereIsNoAgeRule``
    Issue #2017 proposed failing "the newer of the two". On this corpus the newer
    pull request is the one that merged in two of the three cases, so an age rule
    would have named the eventual survivor. The verdict is therefore symmetric,
    and that is asserted from both perspectives of every pair -- if someone later
    adds the age rule, these fail and say why not.

``TestAnIncompleteAnswerIsNotAFinding``
    A truncated link set, a truncated open-pull-request list and an API error must
    all reach ``unknown-claims``. The failure mode this guards is not a false
    accusation but a silent no-op: a check that reports clean because it could not
    see the other pull requests is worse than no check, because it looks like one.

See scripts/check_duplicate_claim.py, issue #2017, and the "PR Workflow" section
of AGENTS.md.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "check_duplicate_claim.py"
_AGENTS = _ROOT / "AGENTS.md"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_duplicate_claim", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The script is annotated and mypy-clean on its own
# (mypy scripts/check_duplicate_claim.py); it is reached through importlib here
# because scripts/ is not an importable package, so its members are module
# attributes at runtime rather than names mypy can resolve to types.
mod = _load()

#: The duplicate pairs measured over #1867 - #2016, as
#: ``(issue, older pull request, newer pull request, the one that merged)``.
#: Every abandoned half of these three was ``APPROVED`` when it was closed.
_MEASURED_PAIRS: tuple[tuple[int, int, int, int], ...] = (
    (1942, 1944, 1946, 1946),
    (1994, 1995, 1996, 1996),
    (2007, 2015, 2016, 2015),
)


#: Check scripts that can infer their repository from the environment, derived
#: rather than listed: a script that never reads ``$GITHUB_REPOSITORY`` (the local
#: git ones) has nothing to infer and so nothing to be given.
_INFERS_REPOSITORY: tuple[str, ...] = tuple(
    sorted(
        path.name
        for path in sorted((_ROOT / "scripts").glob("check_*.py"))
        if "GITHUB_REPOSITORY" in path.read_text(encoding="utf-8")
    )
)

#: The flag that names the API repository, and the argv marker that makes naming it
#: required, for each script that can infer one from ``GITHUB_REPOSITORY``.
#:
#: ``check_merge_base_overlap.py`` differs on both axes and is why this is a mapping
#: rather than a blanket ``--repo`` test. Its own ``--repo`` is the local checkout
#: path, so the API repository needs a distinct spelling; and only ``--all-open``
#: reaches the API at all, its single-branch mode being local git with nothing to
#: infer. Requiring the flag of that mode would be the same false rejection this
#: scope exists to avoid.
_NAMES_REPOSITORY: dict[str, tuple[str, str | None]] = {
    "check_closing_reference.py": ("--repo", None),
    "check_duplicate_claim.py": ("--repo", None),
    "check_last_push_approval.py": ("--repo", None),
    "check_merge_base_overlap.py": ("--github-repo", "--all-open"),
    "check_merge_blockers.py": ("--repo", None),
    "check_pr_head_is_current.py": ("--repo", None),
}


def _documented_intake_argv(issue: int) -> list[str]:
    """Return the intake command step 1 prints, as an ``argv`` list."""
    found = re.findall(r"python3 scripts/check_duplicate_claim\.py([^\n`]*)", _AGENTS.read_text(encoding="utf-8"))
    assert len(found) == 1, found
    return [part.replace("<N>", str(issue)) for part in found[0].split()]


def _documented_check_invocations() -> list[tuple[str, str]]:
    """Return ``(script, remaining argv)`` for every check AGENTS.md invokes."""
    return re.findall(r"python3 scripts/(check_[a-z_]+\.py)([^\n`]*)", _AGENTS.read_text(encoding="utf-8"))


def _pair_ids() -> list[str]:
    return [f"#{issue}" for issue, _, _, _ in _MEASURED_PAIRS]


class TestTheClassifierNamesTheCollisionAndNothingElse:
    """The four outcomes, and the boundary between a pass and a finding."""

    def test_a_pull_request_linking_nothing_cannot_collide(self) -> None:
        verdict = mod.classify([], {1: [7], 2: [7]})
        assert verdict.outcome == mod.NO_CLAIM
        assert not verdict.is_finding
        assert verdict.collisions == ()

    def test_a_claim_no_other_open_pull_request_shares_is_unique(self) -> None:
        verdict = mod.classify([1034], {1087: [], 1722: [], 2018: []})
        assert verdict.outcome == mod.UNIQUE_CLAIM
        assert not verdict.is_finding
        assert verdict.claimed == (1034,)
        assert verdict.scanned == 3

    def test_a_shared_claim_is_the_finding_and_names_the_rival(self) -> None:
        verdict = mod.classify([2007], {2015: [2007], 1722: []})
        assert verdict.outcome == mod.DUPLICATE_CLAIM
        assert verdict.is_finding
        assert verdict.collisions == ((2007, (2015,)),)
        assert verdict.rivals == (2015,)

    def test_a_rival_claiming_a_different_issue_is_not_a_collision(self) -> None:
        """The near-miss that a set-membership bug would report as a finding."""
        verdict = mod.classify([2007], {2015: [2008], 1722: [1034]})
        assert verdict.outcome == mod.UNIQUE_CLAIM

    def test_only_the_shared_issue_is_reported_when_a_claim_is_partly_unique(self) -> None:
        verdict = mod.classify([10, 20], {5: [20]})
        assert verdict.outcome == mod.DUPLICATE_CLAIM
        assert verdict.collisions == ((20, (5,)),)
        assert verdict.claimed == (10, 20)

    def test_the_pull_request_under_test_is_not_its_own_rival(self) -> None:
        """The bug that would make every claiming pull request a finding.

        :func:`resolve_open_claims` drops the number under test, and this is the
        assertion that says why: a pull request always shares its own claim.
        """
        verdict = mod.classify([2007], {})
        assert verdict.outcome == mod.UNIQUE_CLAIM

    def test_collisions_and_rivals_are_sorted_and_deduplicated(self) -> None:
        """A re-run must render the same report, so both axes are ordered."""
        verdict = mod.classify([30, 20, 20], {6: [30, 20], 5: [20]})
        assert verdict.claimed == (20, 30)
        assert verdict.collisions == ((20, (5, 6)), (30, (6,)))
        assert verdict.rivals == (5, 6)

    def test_find_collisions_reports_every_rival_of_one_issue(self) -> None:
        assert mod.find_collisions([7], {3: [7], 4: [7], 5: []}) == ((7, (3, 4)),)


class TestTheMeasuredPairsAreDuplicateClaims:
    """Each of the three shipped pairs must classify as the finding."""

    @pytest.mark.parametrize(("issue", "older", "newer", "merged"), _MEASURED_PAIRS, ids=_pair_ids())
    def test_the_pair_is_reported_from_the_newer_pull_requests_run(
        self, issue: int, older: int, newer: int, merged: int
    ) -> None:
        verdict = mod.classify([issue], {older: [issue]})
        assert verdict.outcome == mod.DUPLICATE_CLAIM
        assert verdict.rivals == (older,)

    @pytest.mark.parametrize(("issue", "older", "newer", "merged"), _MEASURED_PAIRS, ids=_pair_ids())
    def test_the_pair_is_reported_from_the_older_pull_requests_run_too(
        self, issue: int, older: int, newer: int, merged: int
    ) -> None:
        """The older side reports it as well, on its next ``edited`` run.

        A rule that decided by age would report clean here while a live collision
        stood, which is the one state a reporting check must not have.
        """
        verdict = mod.classify([issue], {newer: [issue]})
        assert verdict.outcome == mod.DUPLICATE_CLAIM
        assert verdict.rivals == (newer,)


class TestThereIsNoAgeRule:
    """Why #2017's "fail the newer" was not implemented, kept as a measurement."""

    def test_the_newer_pull_request_is_the_survivor_in_two_of_the_three_pairs(self) -> None:
        """The corpus fact that rules out an age rule.

        Not a property of this code -- a property of what shipped -- which is
        exactly why it belongs beside the check rather than only in its prose.
        """
        newer_survived = [issue for issue, _, newer, merged in _MEASURED_PAIRS if newer == merged]
        assert newer_survived == [1942, 1994], newer_survived
        assert len(newer_survived) == 2

    def test_the_verdict_does_not_depend_on_which_pull_request_is_older(self) -> None:
        younger_view = mod.classify([2007], {2015: [2007]})
        older_view = mod.classify([2007], {2016: [2007]})
        assert younger_view.outcome == older_view.outcome == mod.DUPLICATE_CLAIM

    def test_the_report_says_the_choice_is_not_the_checks_to_make(self) -> None:
        report = mod.render(mod.classify([2007], {2015: [2007]}), "strands-labs/robots", 2016)
        assert "newer of a duplicate pair is the one that merged in two" in report


class TestAnIncompleteAnswerIsNotAFinding:
    """An unreadable answer must reach neither a pass nor an accusation."""

    def test_an_unreadable_half_is_its_own_outcome(self) -> None:
        verdict = mod.classify(None, None, "the API returned errors: [...]")
        assert verdict.outcome == mod.UNKNOWN_CLAIMS
        assert not verdict.is_finding
        assert "the API returned errors" in verdict.summary

    def test_an_unreadable_open_list_does_not_clear_a_claim(self) -> None:
        """The silent no-op this guards: a claim read, the rivals not."""
        verdict = mod.classify([2007], None, "the open pull request list is truncated.")
        assert verdict.outcome == mod.UNKNOWN_CLAIMS
        assert verdict.claimed == (2007,)

    def test_a_truncated_link_set_is_refused_rather_than_read_short(self) -> None:
        with pytest.raises(mod.ClaimSetUnreadable, match="truncated"):
            mod.link_numbers({"number": 7, "closingIssuesReferences": {"totalCount": 2, "nodes": [{"number": 1}]}})

    def test_a_complete_link_set_is_read(self) -> None:
        assert mod.link_numbers(
            {"number": 7, "closingIssuesReferences": {"totalCount": 2, "nodes": [{"number": 9}, {"number": 1}]}}
        ) == (1, 9)

    def test_api_errors_are_unreadable_rather_than_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_post", lambda *_a, **_k: {"errors": [{"message": "nope"}]})
        with pytest.raises(mod.ClaimSetUnreadable, match="the API returned errors"):
            mod.resolve_claim("o/n", 7, "t")

    def test_a_repository_that_is_not_in_owner_name_form_is_refused(self) -> None:
        with pytest.raises(mod.ClaimSetUnreadable, match="owner/name"):
            mod.resolve_claim("nope", 7, "t")

    def test_a_missing_pull_request_is_unreadable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_post", lambda *_a, **_k: {"data": {"repository": {"pullRequest": None}}})
        with pytest.raises(mod.ClaimSetUnreadable, match="no pull request"):
            mod.resolve_claim("o/n", 7, "t")


class TestTheOpenListIsReadLiveAndPaginated:
    """The two ways a prefix of the open set would be a false clean."""

    def test_the_number_under_test_is_excluded_from_its_own_comparison(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = {
            "data": {
                "repository": {
                    "pullRequests": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [
                            {"number": 7, "closingIssuesReferences": {"totalCount": 1, "nodes": [{"number": 42}]}},
                            {"number": 8, "closingIssuesReferences": {"totalCount": 0, "nodes": []}},
                        ],
                    }
                }
            }
        }
        monkeypatch.setattr(mod, "_post", lambda *_a, **_k: page)
        assert mod.resolve_open_claims("o/n", "t", 7) == {8: ()}

    def test_every_page_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pages = [
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
                            "nodes": [{"number": 8, "closingIssuesReferences": {"totalCount": 0, "nodes": []}}],
                        }
                    }
                }
            },
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [
                                {
                                    "number": 9,
                                    "closingIssuesReferences": {"totalCount": 1, "nodes": [{"number": 42}]},
                                }
                            ],
                        }
                    }
                }
            },
        ]
        seen: list[object] = []

        def fake_post(_query: str, variables: dict[str, object], _token: str) -> object:
            seen.append(variables.get("after"))
            return pages[len(seen) - 1]

        monkeypatch.setattr(mod, "_post", fake_post)
        assert mod.resolve_open_claims("o/n", "t", 7) == {8: (), 9: (42,)}
        assert seen == [None, "c1"], seen

    def test_a_list_longer_than_the_page_bound_is_unreadable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A never-ending pager must not silently return the prefix it read."""
        endless = {
            "data": {"repository": {"pullRequests": {"pageInfo": {"hasNextPage": True, "endCursor": "c"}, "nodes": []}}}
        }
        monkeypatch.setattr(mod, "_post", lambda *_a, **_k: endless)
        with pytest.raises(mod.ClaimSetUnreadable, match="truncated"):
            mod.resolve_open_claims("o/n", "t", 7)

    def test_a_page_without_a_cursor_is_unreadable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        broken = {
            "data": {
                "repository": {"pullRequests": {"pageInfo": {"hasNextPage": True, "endCursor": None}, "nodes": []}}
            }
        }
        monkeypatch.setattr(mod, "_post", lambda *_a, **_k: broken)
        with pytest.raises(mod.ClaimSetUnreadable, match="cursor"):
            mod.resolve_open_claims("o/n", "t", 7)

    def test_the_open_set_is_read_from_the_repository_and_not_from_search(self) -> None:
        """``search`` is eventually consistent, and this job runs on ``opened``.

        A just-opened pull request may not be indexed yet, so a search-backed
        query would return a clean answer on exactly the event that matters.
        """
        assert "pullRequests(states: OPEN" in mod._OPEN_QUERY
        assert "search(" not in mod._OPEN_QUERY
        assert "search(" not in mod._SELF_QUERY

    def test_the_pull_request_under_test_is_read_by_number(self) -> None:
        """So the answer does not depend on it being indexed or on page order."""
        assert "pullRequest(number: $number)" in mod._SELF_QUERY


class TestTheReportNamesBothPullRequestsAndBothRemedies:
    """The report is the whole output, so its content is part of the contract."""

    def test_the_finding_names_the_issue_the_rival_and_both_remedies(self) -> None:
        report = mod.render(mod.classify([2007], {2015: [2007], 1722: []}), "strands-labs/robots", 2016)
        assert "duplicate-claim" in report
        assert "strands-labs/robots#2016" in report
        assert "| also claimed by | #2015 |" in report
        assert "drops the keyword from" in report
        assert "close it" in report

    def test_a_clean_report_carries_no_remedy_section(self) -> None:
        report = mod.render(mod.classify([1034], {1: []}), "o/n", 1035)
        assert "unique-claim" in report
        assert "What clears this" not in report

    def test_a_report_for_a_pull_request_claiming_nothing_omits_the_compared_count(self) -> None:
        """ "0 compared" would read as "the check could not see the other pull requests"."""
        report = mod.render(mod.classify([], {}), "o/n", 7)
        assert "no-claim" in report
        assert "other open pull requests compared" not in report

    def test_a_clean_report_states_how_many_were_compared(self) -> None:
        report = mod.render(mod.classify([1034], {1: [], 2: []}), "o/n", 1035)
        assert "| other open pull requests compared | 2 |" in report

    def test_one_clause_per_shared_issue(self) -> None:
        summary = mod.classify([20, 30], {5: [20], 6: [20, 30]}).summary
        assert "#20 is also claimed by #5 and #6" in summary
        assert "#30 is also claimed by #6" in summary
        # Semicolons, not "and", between clauses that already contain one.
        assert "#6 and #30" not in summary


class TestTheExitStatusIsOneOnlyForTheFinding:
    """What CI reads, as opposed to what the report says."""

    @pytest.mark.parametrize(
        ("claimed", "others", "expected"),
        [
            ((), {}, 0),
            ((1034,), {1: ()}, 0),
            ((2007,), {2015: (2007,)}, 1),
        ],
        ids=["no-claim", "unique-claim", "duplicate-claim"],
    )
    def test_the_exit_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        claimed: tuple[int, ...],
        others: dict[int, tuple[int, ...]],
        expected: int,
    ) -> None:
        monkeypatch.setattr(mod, "resolve_claim", lambda *_a, **_k: claimed)
        monkeypatch.setattr(mod, "resolve_open_claims", lambda *_a, **_k: others)
        status = mod.main(["--repo", "o/n", "--pr", "7", "--token", "t"])
        capsys.readouterr()
        assert status == expected

    def test_a_lookup_failure_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise mod.ClaimSetUnreadable("nope.")

        monkeypatch.setattr(mod, "resolve_claim", boom)
        assert mod.main(["--repo", "o/n", "--pr", "7", "--token", "t"]) == 0
        assert "unknown-claims" in capsys.readouterr().out

    def test_a_pull_request_claiming_nothing_does_not_read_the_open_list(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """One API call saved on the majority case, and the reason it is safe."""

        def unreached(*_a: object, **_k: object) -> None:
            raise AssertionError("the open list must not be read when nothing is claimed")

        monkeypatch.setattr(mod, "resolve_claim", lambda *_a, **_k: ())
        monkeypatch.setattr(mod, "resolve_open_claims", unreached)
        assert mod.main(["--repo", "o/n", "--pr", "7", "--token", "t"]) == 0
        assert "no-claim" in capsys.readouterr().out


class TestTheIntakeModeAsksAboutAnIssue:
    """``--issue`` is the half that prevents the work rather than capping it.

    Every observed pair opened inside one ~35-minute window, so the cheap question
    is asked before the second pull request exists -- at which point there is no
    pull request to compare, only an issue to look up.
    """

    def test_nothing_is_excluded_when_no_pull_request_exists_yet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ``--pr`` self-exclusion must not silently drop a rival at intake."""
        page = {
            "data": {
                "repository": {
                    "pullRequests": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [
                            {"number": 7, "closingIssuesReferences": {"totalCount": 1, "nodes": [{"number": 42}]}},
                            {"number": 8, "closingIssuesReferences": {"totalCount": 0, "nodes": []}},
                        ],
                    }
                }
            }
        }
        monkeypatch.setattr(mod, "_post", lambda *_a, **_k: page)
        assert mod.resolve_open_claims("o/n", "t") == {7: (42,), 8: ()}

    def test_an_issue_another_pull_request_claims_is_the_finding(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(mod, "resolve_open_claims", lambda *_a, **_k: {2015: (2007,)})
        status = mod.main(["--repo", "o/n", "--issue", "2007", "--token", "t"])
        out = capsys.readouterr().out
        assert status == 1
        assert "duplicate-claim" in out
        assert "| issue | o/n#2007 |" in out
        assert "| already claimed by | #2015 |" in out
        assert "Read it before" in out

    def test_an_unclaimed_issue_is_clean(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(mod, "resolve_open_claims", lambda *_a, **_k: {2015: (2007,), 1722: ()})
        status = mod.main(["--repo", "o/n", "--issue", "2017", "--token", "t"])
        out = capsys.readouterr().out
        assert status == 0
        assert "unique-claim" in out
        assert "| issue | o/n#2017 |" in out

    def test_the_intake_report_does_not_offer_the_review_remedy(self) -> None:
        """At intake there is no second pull request whose keyword could be dropped."""
        report = mod.render(mod.classify([2007], {2015: [2007]}), "o/n", None, 2007)
        assert "What this means" in report
        assert "What clears this" not in report

    def test_the_review_mode_excludes_the_pull_request_under_test(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The asymmetry between the two modes, asserted rather than implied."""
        seen: list[object] = []

        def spy(_repo: str, _token: str, pr: int | None = None) -> dict[int, tuple[int, ...]]:
            seen.append(pr)
            return {}

        monkeypatch.setattr(mod, "resolve_claim", lambda *_a, **_k: (2007,))
        monkeypatch.setattr(mod, "resolve_open_claims", spy)
        mod.main(["--repo", "o/n", "--pr", "2016", "--token", "t"])
        capsys.readouterr()
        assert seen == [2016], seen

    @pytest.mark.parametrize(
        "argv",
        [
            ["--repo", "o/n", "--token", "t"],
            ["--repo", "o/n", "--pr", "1", "--issue", "2", "--token", "t"],
        ],
        ids=["neither", "both"],
    )
    def test_exactly_one_subject_is_required(self, argv: list[str]) -> None:
        """Two subjects would silently answer only one of the two questions."""
        with pytest.raises(SystemExit):
            mod.main(argv)


class TestTheGuidanceRecordsTheIntakeCheck:
    """The CI half caps the cost; the intake half is what prevents it."""

    @staticmethod
    def _step_one() -> str:
        """Return step 1 of the PR Workflow, whitespace-collapsed.

        Bounded at step 2 so a qualifier reworded down into another step leaves
        the slice and fails, and collapsed so a reflow cannot.
        """
        text = _AGENTS.read_text(encoding="utf-8")
        start = text.index("1. Create the feature branch")
        end = text.index("2. Make changes", start)
        return " ".join(text[start:end].split())

    def test_the_slice_is_step_one_and_nothing_else(self) -> None:
        step_one = self._step_one()
        assert len(step_one) > 1500, len(step_one)
        assert "Make changes" not in step_one

    @pytest.mark.parametrize(
        "phrase",
        [
            "check that no open pull request already claims the issue",
            "every one of those had",
            "property of the *set* of open ones",
            "closingIssuesReferences",
            "check_duplicate_claim.py --repo strands-labs/robots --issue",
            "where the command is *running*",
            "refuses an inferred repository",
            "prevents the authoring rather than capping it",
        ],
    )
    def test_step_one_carries_the_intake_check(self, phrase: str) -> None:
        assert phrase in self._step_one()

    def test_the_guidance_names_the_escape_hatch(self) -> None:
        """Two deliberate implementations are allowed -- one claim between them."""
        step_one = self._step_one()
        assert "exactly one should claim the close" in step_one
        assert re.search(r"per #N|towards #N", step_one)


class TestIntakeModeMustNameTheRepository:
    """``$GITHUB_REPOSITORY`` names where a command runs, not what an issue belongs to.

    Intake is the mode that runs *before* any pull request exists, so it is a local
    invocation whose working directory says nothing about the target repository. The
    ``--pr`` mode is the opposite: a workflow reviewing a pull request runs in the
    repository that pull request lives in, so inference is right by construction
    there and the default is kept.

    Refused rather than warned about because the failure is silent and only misleads
    in the reassuring direction. Measured on this script with ``huggingface/lerobot``
    as the ambient repository, ``--issue 2029`` compared **405** unrelated open pull
    requests and reported ``unique-claim`` with exit ``0``; naming the repository
    compared 4. Both said no duplicate, so nothing in the report distinguished them.
    """

    def test_an_inferred_repository_is_refused_at_intake(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("GITHUB_REPOSITORY", "someone/elsewhere")
        with pytest.raises(SystemExit) as excinfo:
            mod.main(["--issue", "2029", "--token", "t"])
        assert excinfo.value.code == 2, excinfo.value.code
        err = capsys.readouterr().err
        assert "--repo owner/name" in err
        assert "'someone/elsewhere'" in err, err
        assert "reports no duplicate" in err

    def test_the_refusal_is_about_inference_not_the_value(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The very same repository, named explicitly, is accepted."""
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/n")
        monkeypatch.setattr(mod, "resolve_open_claims", lambda *_a, **_k: {})
        assert mod.main(["--repo", "o/n", "--issue", "2029", "--token", "t"]) == 0
        assert "| issue | o/n#2029 |" in capsys.readouterr().out

    def test_the_review_mode_keeps_the_inferred_default(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The asymmetry, asserted rather than implied by the intake tests."""
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/n")
        monkeypatch.setattr(mod, "resolve_claim", lambda *_a, **_k: (7,))
        monkeypatch.setattr(mod, "resolve_open_claims", lambda *_a, **_k: {})
        assert mod.main(["--pr", "5", "--token", "t"]) == 0
        assert "| pull request | o/n#5 |" in capsys.readouterr().out

    def test_the_refusal_precedes_every_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A refused invocation reads nothing -- not even to decide it is refused."""

        def fatal(*_a: object, **_k: object) -> object:
            raise AssertionError("the refused invocation reached the API")

        monkeypatch.setenv("GITHUB_REPOSITORY", "someone/elsewhere")
        monkeypatch.setattr(mod, "_post", fatal)
        monkeypatch.setattr(mod, "resolve_open_claims", fatal)
        monkeypatch.setattr(mod, "resolve_claim", fatal)
        with pytest.raises(SystemExit):
            mod.main(["--issue", "2029", "--token", "t"])

    def test_nothing_to_infer_still_names_the_flag(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """With no ambient repository the original precondition is what fires."""
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        with pytest.raises(SystemExit):
            mod.main(["--issue", "2029", "--token", "t"])
        assert "--repo is required" in capsys.readouterr().err

    def test_the_reason_names_what_was_inferred_and_what_follows(self) -> None:
        """A refusal that named neither would send the reader to the wrong place."""
        reason = mod.inferred_repository_refusal("someone/elsewhere")
        assert "--repo owner/name" in reason
        assert "'someone/elsewhere'" in reason
        assert "where this command is running" in reason
        assert "reports no duplicate" in reason

    def test_the_documented_command_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The invocation step 1 prints must survive this script's own preconditions.

        Pins guidance and code together from the guidance side: shortening the
        documented command back to an inferred repository fails here, and so does
        adding the refusal without updating the command it refuses.
        """
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        monkeypatch.setattr(mod, "resolve_open_claims", lambda *_a, **_k: {})
        assert mod.main([*_documented_intake_argv(2029), "--token", "t"]) == 0
        assert "| issue | strands-labs/robots#2029 |" in capsys.readouterr().out


class TestNoDocumentedInvocationLeavesTheRepositoryInferred:
    """The defect was a documented command, so the durable pin is on the guidance.

    Measured over the check scripts AGENTS.md invokes, exactly one omitted ``--repo``.
    ``check_last_push_approval.py`` names it in both of its invocations, and
    ``check_closing_reference.py`` has no local invocation at all -- its workflow calls
    it with no arguments, where the environment is correct by construction. Scoped to
    the scripts that can *infer* a repository, since the local-git checks have nothing
    to infer and requiring a flag of them would be a false rejection.

    The scope is per *invocation*, not per script: :data:`_NAMES_REPOSITORY` carries
    both the flag each script spells the repository with and the argv marker that makes
    naming it required, because one script's ``--repo`` means a local checkout path and
    one of its two modes never reaches the API.
    """

    def test_the_inferring_scripts_are_the_ones_measured(self) -> None:
        """Non-vacuity: the scope is a real set, and this script is in it."""
        assert "check_duplicate_claim.py" in _INFERS_REPOSITORY, _INFERS_REPOSITORY
        assert set(_INFERS_REPOSITORY) == set(_NAMES_REPOSITORY), (
            "a script that can infer the repository must declare how it names one",
            _INFERS_REPOSITORY,
            sorted(_NAMES_REPOSITORY),
        )

    def test_every_documented_invocation_names_the_repository(self) -> None:
        invocations = _documented_check_invocations()
        assert len(invocations) >= 3, invocations
        inferring = [(script, rest) for script, rest in invocations if script in _NAMES_REPOSITORY]
        assert inferring, invocations
        missing = [
            f"{script}{rest}"
            for script, rest in inferring
            for flag, marker in (_NAMES_REPOSITORY[script],)
            if (marker is None or marker in rest) and flag not in rest
        ]
        assert not missing, missing
