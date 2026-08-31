"""Pins the created-thing key against the claim-free duplicate pairs this repository shipped.

``scripts/check_duplicate_claim.py`` was built around one key,
``closingIssuesReferences``, and **249 of the last 300 pull requests (#2345
through #2708) link no issue at all** -- so for most of the traffic both
claim-keyed modes have nothing to collide on and report a unique claim while
looking straight at a duplicate pair. Issue #2709 is the third recorded instance.

:data:`_CLAIM_FREE_PAIRS` fixes the three measured ones in place, so the sweep is
tested against real shapes rather than invented ones. All three were reconstructed
from ``pulls/<n>/files``, which outlives the state change that closed one half of
two of them.

Three pins carry design decisions rather than behaviour:

``TestTheTwoKeysAreComplementary``
    The window holds five duplicate pairs: two reachable from a claim, three only
    from what both branches create, and none from both. So this is a second key
    rather than a replacement, and neither relation may be deleted in favour of the
    other.

``TestAFragmentCollidesOnItsSlugAndNotItsNumber``
    What a created path collides *on*. Almost always itself; a changelog fragment
    is named ``<number>-<slug>.md``, so it collides on the slug and the number is
    dropped. That replaced an assertion that a fragment can never collide at all,
    whose reason -- the number in its name -- was the half that had to go. The
    bound on the widening is asserted beside it, because the reason for the old
    boundary was a fear of firing on everything.

``TestOnlyACreatedPathCollides``
    Widening the relation from ``ADDED`` to every ``changeType`` turns 2 selected
    pairs of 1802 into 117, which is a composition question owned by
    ``scripts/check_merge_base_overlap.py`` and not this one. The narrowness is
    the whole claim, so it is asserted from both sides.

``TestAnIncompleteAnswerIsNotAFinding``
    A truncated file list, a truncated open-pull-request list and an API error
    must all reach ``unknown-additions``. The failure mode guarded is not a false
    accusation but a silent no-op: a sweep that reports clean because it could not
    read the open set is worse than none, because it looks like one.

See scripts/check_duplicate_claim.py, issue #2709, and the "PR Workflow" section
of AGENTS.md.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import textwrap
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


check = _load()


#: One fixture row: ``(label, {number: added paths}, the subjects both create)``.
#: The third element is a tuple rather than one path because a pair can share more
#: than one -- #2388/#2389 share both a test file and a changelog entry.
_ClaimFreePair = tuple[str, dict[int, tuple[str, ...]], tuple[str, ...]]

#: One row of the other key: ``(left, right, the issue both claim, added paths)``.
_IssueKeyedPair = tuple[int, int, int, dict[int, tuple[str, ...]]]

#: The three duplicate pairs in #2345..#2767 that claimed no issue, as
#: ``(label, {number: added paths}, the subjects both create)``. Every path is the
#: real one the pull request added, read back from ``pulls/<n>/files``, which
#: outlives the state change that closed one half of two of them.
#:
#: The three rows are deliberately one of each reachable shape, so no single
#: relation can satisfy the class:
#:
#: * #2388/#2389 collide on **both** a created test file and a changelog slug.
#: * #2707/#2708 collide on the **path only** -- #2707 added no fragment at all,
#:   so there is no slug on one side to compare.
#: * #2766/#2767 collide on the **slug only** -- one put its tests in a new file
#:   and the other into an existing one, so they create no common path.
_CLAIM_FREE_PAIRS: list[_ClaimFreePair] = [
    (
        "#2388/#2389 - un-count frames a discarded episode never wrote",
        {
            2388: (
                "changelog.d/2388-recorder-uncount-discarded-frames.md",
                "tests/test_recorder_counters_track_on_disk_frames.py",
            ),
            2389: (
                "changelog.d/2389-recorder-uncount-discarded-frames.md",
                "tests/test_recorder_counters_track_on_disk_frames.py",
            ),
        },
        (
            "changelog.d/*-recorder-uncount-discarded-frames.md",
            "tests/test_recorder_counters_track_on_disk_frames.py",
        ),
    ),
    (
        "#2707/#2708 - a value domain for TrainSpec.save_freq",
        {
            2707: ("tests/training/test_checkpoint_cadence_domain.py",),
            2708: (
                "changelog.d/2708-lerobot-trainer-checkpoint-cadence-domain.md",
                "tests/training/test_checkpoint_cadence_domain.py",
            ),
        },
        ("tests/training/test_checkpoint_cadence_domain.py",),
    ),
    (
        "#2766/#2767 - wire G1Driver.send_action to rt/lowcmd",
        {
            2766: (
                "changelog.d/2765-g1-send-action-wired.md",
                "tests/drivers/test_g1_send_action_wired.py",
            ),
            2767: ("changelog.d/2761-g1-send-action-wired.md",),
        },
        ("changelog.d/*-g1-send-action-wired.md",),
    ),
]

#: The two duplicate pairs in the same window that *were* reachable from a claim,
#: as ``(left, right, the issue both claim, each side's added paths)``. Their
#: added-path sets are disjoint, which is what makes the two keys complementary.
_ISSUE_KEYED_PAIRS: list[_IssueKeyedPair] = [
    (
        2570,
        2571,
        2569,
        {
            2570: ("changelog.d/2570-all-open-refuses-a-local-checkout-flag.md",),
            2571: ("changelog.d/2571-sweep-refuses-a-flag-it-cannot-honour.md",),
        },
    ),
    (
        2480,
        2508,
        2466,
        {
            2480: ("changelog.d/2466-dataset-transform-surface.md", "strands_robots/transforms/base.py"),
            2508: ("changelog.d/2467-data-augmentation-notebook.md", "examples/notebooks/07_data_augmentation.ipynb"),
        },
    ),
]


def _node(number: int, files: list[dict[str, str]], total: int | None = None) -> dict[str, Any]:
    """Build one pull-request node the way the GraphQL response shapes it."""
    return {
        "number": number,
        "files": {"totalCount": len(files) if total is None else total, "nodes": files},
    }


def _added(*paths: str) -> list[dict[str, str]]:
    return [{"path": path, "changeType": check.ADDED_CHANGE_TYPE} for path in paths]


def _edited(*paths: str) -> list[dict[str, str]]:
    return [{"path": path, "changeType": check.EDITED_CHANGE_TYPE} for path in paths]


def _creates(*paths: str) -> Any:
    """A pull request that creates these paths and edits nothing."""
    return check.PullFiles(created=tuple(sorted(paths)))


def _open_set(additions: dict[int, tuple[str, ...]]) -> dict[int, Any]:
    """``{number: the paths it added}`` as the open set the sweep classifies.

    The measured fixtures record what each branch *added*, which is the evidence
    that outlives the state change closing one half of a pair. The second key
    reads edited paths as well, so they are the created half of a file set here
    and the pairs stay reachable from the first key alone.
    """
    return {number: _creates(*paths) for number, paths in additions.items()}


_IDS = [row[0] for row in _CLAIM_FREE_PAIRS]


class TestTheMeasuredClaimFreePairsAreReported:
    """Each measured pair is the finding, and names what both branches create."""

    @pytest.mark.parametrize(("label", "additions", "subjects"), _CLAIM_FREE_PAIRS, ids=_IDS)
    def test_the_pair_is_the_finding(
        self, label: str, additions: dict[int, tuple[str, ...]], subjects: tuple[str, ...]
    ) -> None:
        verdict = check.classify_additions(_open_set(additions))
        assert verdict.outcome == check.DUPLICATE_ADDITION, label
        assert verdict.is_finding

    @pytest.mark.parametrize(("label", "additions", "subjects"), _CLAIM_FREE_PAIRS, ids=_IDS)
    def test_only_the_shared_subjects_are_reported(
        self, label: str, additions: dict[int, tuple[str, ...]], subjects: tuple[str, ...]
    ) -> None:
        left, right = sorted(additions)
        assert check.classify_additions(_open_set(additions)).collisions == ((left, right, subjects),)

    @pytest.mark.parametrize(("label", "additions", "subjects"), _CLAIM_FREE_PAIRS, ids=_IDS)
    def test_both_pull_requests_are_named(
        self, label: str, additions: dict[int, tuple[str, ...]], subjects: tuple[str, ...]
    ) -> None:
        summary = check.classify_additions(_open_set(additions)).summary
        for number in additions:
            assert f"#{number}" in summary
        for subject in subjects:
            assert subject in summary

    @pytest.mark.parametrize(("label", "additions", "subjects"), _CLAIM_FREE_PAIRS, ids=_IDS)
    def test_no_reported_subject_is_invented(
        self, label: str, additions: dict[int, tuple[str, ...]], subjects: tuple[str, ...]
    ) -> None:
        """Every subject is a real created path, or the slug-glob of two of them.

        The report must not name a file that exists on neither branch. A path
        subject has to have been added by both; a glob subject has to be what
        :func:`addition_key` makes of a path each side really added.
        """
        for left, right, reported in check.classify_additions(_open_set(additions)).collisions:
            for subject in reported:
                for number in (left, right):
                    matches = [path for path in additions[number] if check.addition_key(path) == subject]
                    assert matches, f"{label}: #{number} creates nothing keyed {subject!r}"
                if "*" not in subject:
                    assert subject in additions[left] and subject in additions[right], label


class TestAFragmentCollidesOnItsSlugAndNotItsNumber:
    """The exclusion this relation shipped with, and why the number was the wrong half.

    ``test_a_changelog_fragment_is_never_the_shared_path`` used to assert the
    opposite of the first case here. Its reason -- "its name embeds the number, so
    it cannot collide" -- is true of the raw path and is exactly why the number is
    dropped: #2766/#2767 wrote one slug under two numbers and shared no path at all.

    The second case is the bound that keeps the first from firing on the whole
    queue, and the measurement behind it is in :func:`check.addition_key`: 350
    distinct slugs across the 350 pull requests in #2345..#2767 that add a fragment.
    """

    def test_two_fragments_with_one_slug_collide_under_different_numbers(self) -> None:
        additions = {
            2766: check.file_sets(_node(2766, _added("changelog.d/2765-g1-send-action-wired.md"))),
            2767: check.file_sets(_node(2767, _added("changelog.d/2761-g1-send-action-wired.md"))),
        }
        verdict = check.classify_additions(additions)
        assert verdict.outcome == check.DUPLICATE_ADDITION
        assert verdict.collisions == ((2766, 2767, ("changelog.d/*-g1-send-action-wired.md",)),)

    def test_two_fragments_with_different_slugs_do_not_collide(self) -> None:
        """The over-reach control: every branch adds a fragment, so this is the norm."""
        additions = {
            11: check.file_sets(_node(11, _added("changelog.d/11-one-change.md"))),
            12: check.file_sets(_node(12, _added("changelog.d/12-another-change.md"))),
        }
        assert check.classify_additions(additions).outcome == check.UNIQUE_ADDITIONS

    def test_the_same_slug_under_the_same_number_still_collides(self) -> None:
        """Dropping the number must not lose the identical-path case it contained."""
        shared = _added("changelog.d/11-one-change.md")
        additions = {n: check.file_sets(_node(n, shared)) for n in (11, 12)}
        assert check.classify_additions(additions).outcome == check.DUPLICATE_ADDITION

    @pytest.mark.parametrize(
        "path",
        [
            "changelog.d/README.md",
            "changelog.d/no-leading-number.md",
            "changelog.d/2765-Not_A_Slug.md",
            "changelog.d/nested/2765-slug.md",
            "tests/test_x.py",
            "strands_robots/utils.py",
        ],
    )
    def test_anything_that_is_not_a_fragment_keys_on_itself(self, path: str) -> None:
        """Including a name the assembler would reject, which is the safe direction.

        Keying such a name on itself can only fail to report a pair. Inventing a
        slug for it could report one that is not there.
        """
        assert check.addition_key(path) == path

    def test_a_reserved_name_keys_on_itself_even_if_it_looks_like_a_fragment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The assembler's list is the authority, not the pattern's happy accident.

        ``README.md`` is excluded by the pattern anyway, having no leading digits,
        so the check reads as redundant against today's list. It is not redundant
        against the list: a reserved name that *did* carry a number would match
        the pattern, and the fragment directory's documentation file is not an
        entry two branches can duplicate.
        """
        monkeypatch.setattr(check._ASSEMBLER, "RESERVED_NAMES", frozenset({"0-release-notes.md"}))
        assert check._ASSEMBLER.FRAGMENT_NAME.match("0-release-notes.md"), "premise: the pattern accepts it"
        assert check.addition_key("changelog.d/0-release-notes.md") == "changelog.d/0-release-notes.md"

    def test_the_naming_rule_is_the_assemblers_and_not_a_second_copy(self) -> None:
        """A restated regex could drift from what a fragment actually is."""
        assert check._ASSEMBLER.FRAGMENT_NAME.match("2765-g1-send-action-wired.md")
        assert "README.md" in check._ASSEMBLER.RESERVED_NAMES
        source = _SCRIPT.read_text(encoding="utf-8")
        assert "_ASSEMBLER.FRAGMENT_NAME" in source
        assert "_ASSEMBLER.RESERVED_NAMES" in source


class TestTheTwoKeysAreComplementary:
    """Five duplicate pairs: two reachable from the claim, three from what is created."""

    @pytest.mark.parametrize(("label", "additions", "shared"), _CLAIM_FREE_PAIRS, ids=lambda value: str(value)[:40])
    def test_a_claim_free_pair_is_invisible_to_the_claim_key(
        self, label: str, additions: dict[int, tuple[str, ...]], shared: str
    ) -> None:
        """Neither half links an issue, so the claim-keyed verdict is ``no-claim``.

        Not a defect being pinned in place -- requiring a claim is what 18 of the
        last 30 merges would fail. It is why a second key had to exist.
        """
        left, right = sorted(additions)
        verdict = check.classify(claimed=(), others={right: ()})
        assert verdict.outcome == check.NO_CLAIM
        assert not verdict.is_finding

    @pytest.mark.parametrize(("left", "right", "issue", "additions"), _ISSUE_KEYED_PAIRS)
    def test_an_issue_keyed_pair_is_invisible_to_the_created_thing_key(
        self, left: int, right: int, issue: int, additions: dict[int, tuple[str, ...]]
    ) -> None:
        assert check.classify_additions(_open_set(additions)).outcome == check.UNIQUE_ADDITIONS

    @pytest.mark.parametrize(("left", "right", "issue", "additions"), _ISSUE_KEYED_PAIRS)
    def test_the_claim_key_still_reports_that_pair(
        self, left: int, right: int, issue: int, additions: dict[int, tuple[str, ...]]
    ) -> None:
        """The non-vacuity half: each key must still catch what it was built for."""
        verdict = check.classify(claimed=(issue,), others={right: (issue,)})
        assert verdict.outcome == check.DUPLICATE_CLAIM
        assert verdict.collisions == ((issue, (right,)),)

    def test_neither_outcome_vocabulary_reuses_the_others_names(self) -> None:
        """A reader has to be able to tell which relation produced a finding."""
        claim = {check.NO_CLAIM, check.UNIQUE_CLAIM, check.DUPLICATE_CLAIM, check.UNKNOWN_CLAIMS}
        addition = {check.UNIQUE_ADDITIONS, check.DUPLICATE_ADDITION, check.UNKNOWN_ADDITIONS}
        assert not claim & addition
        assert len(claim) == 4
        assert len(addition) == 3


class TestOnlyACreatedPathCollides:
    """The narrowness is the claim: a path that already exists is the sibling's question."""

    @pytest.mark.parametrize("change_type", ["MODIFIED", "REMOVED", "RENAMED", "COPIED", "CHANGED"])
    def test_a_path_both_branches_merely_touch_is_not_a_finding(self, change_type: str) -> None:
        shared = [{"path": "strands_robots/utils.py", "changeType": change_type}]
        additions = {n: check.file_sets(_node(n, shared)) for n in (11, 12)}
        assert check.classify_additions(additions).outcome == check.UNIQUE_ADDITIONS

    def test_the_same_path_created_by_both_is_the_finding(self) -> None:
        """The control for the row above: only ``changeType`` differs."""
        shared = _added("strands_robots/utils.py")
        additions = {n: check.file_sets(_node(n, shared)) for n in (11, 12)}
        assert check.classify_additions(additions).outcome == check.DUPLICATE_ADDITION

    def test_one_branch_creating_what_the_other_edits_is_not_a_finding(self) -> None:
        """Impossible on one base, and if the API says it the sibling sweep owns it."""
        creator = check.file_sets(_node(11, _added("docs/new.md")))
        editor = check.file_sets(_node(12, [{"path": "docs/new.md", "changeType": "MODIFIED"}]))
        assert check.classify_additions({11: creator, 12: editor}).outcome == check.UNIQUE_ADDITIONS

    def test_prose_is_not_exempt(self) -> None:
        """The sibling sweep's prose exemption does not transfer to this question.

        There it holds because a shared ``.md`` edit cannot change what the suite
        does and git reports the conflict anyway. Two branches each *writing* one
        new page is duplicated authoring whatever the suffix.
        """
        additions = {n: check.file_sets(_node(n, _added("docs/guide.md"))) for n in (11, 12)}
        verdict = check.classify_additions(additions)
        assert verdict.outcome == check.DUPLICATE_ADDITION
        assert verdict.collisions == ((11, 12, ("docs/guide.md",)),)


class TestEveryPairIsComparedAndTheReportIsStable:
    """Determinism in all three axes, and no reliance on adjacent numbers."""

    def test_pull_requests_that_are_not_adjacent_are_still_compared(self) -> None:
        additions = {
            11: ("tests/test_a.py",),
            12: ("tests/test_unrelated.py",),
            99: ("tests/test_a.py",),
        }
        assert check.find_addition_collisions(additions) == ((11, 99, ("tests/test_a.py",)),)

    def test_pairs_and_paths_are_sorted(self) -> None:
        additions = {
            99: ("tests/test_b.py", "tests/test_a.py"),
            11: ("tests/test_b.py", "tests/test_a.py"),
        }
        assert check.find_addition_collisions(additions) == ((11, 99, ("tests/test_a.py", "tests/test_b.py")),)

    def test_three_branches_creating_one_file_report_all_three_pairs(self) -> None:
        additions = {n: ("tests/test_a.py",) for n in (11, 12, 13)}
        assert check.find_addition_collisions(additions) == (
            (11, 12, ("tests/test_a.py",)),
            (11, 13, ("tests/test_a.py",)),
            (12, 13, ("tests/test_a.py",)),
        )
        assert check.classify_additions({n: _creates("tests/test_a.py") for n in (11, 12, 13)}).implicated == (
            11,
            12,
            13,
        )

    def test_an_empty_open_set_is_clean_and_says_so(self) -> None:
        verdict = check.classify_additions({})
        assert verdict.outcome == check.UNIQUE_ADDITIONS
        assert verdict.compared == 0

    def test_the_pair_count_is_the_number_of_comparisons(self) -> None:
        assert check.classify_additions({n: _creates() for n in range(1, 5)}).compared == 6


class TestAnIncompleteAnswerIsNotAFinding:
    """An unread file list is neither a pass nor a finding."""

    def test_an_unreadable_set_is_its_own_outcome(self) -> None:
        verdict = check.classify_additions(None, "the API returned errors.")
        assert verdict.outcome == check.UNKNOWN_ADDITIONS
        assert not verdict.is_finding
        assert "the API returned errors." in verdict.summary

    def test_a_truncated_file_list_is_refused_rather_than_read_short(self) -> None:
        node = _node(11, _added("tests/test_a.py"), total=check.FILE_PAGE_SIZE + 1)
        with pytest.raises(check.ClaimSetUnreadable, match="file list is truncated"):
            check.file_sets(node)

    def test_a_complete_file_list_is_read(self) -> None:
        assert check.file_sets(_node(11, _added("b", "a"))).created == ("a", "b")

    def test_a_node_without_a_file_list_is_unreadable(self) -> None:
        with pytest.raises(check.ClaimSetUnreadable, match="carried no file list"):
            check.file_sets({"number": 11, "files": "not a list"})

    def test_a_repository_that_is_not_in_owner_name_form_is_refused(self) -> None:
        with pytest.raises(check.ClaimSetUnreadable, match="owner/name form"):
            check.resolve_open_file_sets("robots", "token")

    def test_an_api_error_is_unreadable_rather_than_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(check, "_post", lambda *a, **k: {"errors": [{"message": "nope"}]})
        with pytest.raises(check.ClaimSetUnreadable, match="the API returned errors"):
            check.resolve_open_file_sets("owner/name", "token")

    def test_a_page_without_a_cursor_is_unreadable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "data": {
                "repository": {"pullRequests": {"pageInfo": {"hasNextPage": True, "endCursor": None}, "nodes": []}}
            }
        }
        monkeypatch.setattr(check, "_post", lambda *a, **k: payload)
        with pytest.raises(check.ClaimSetUnreadable, match="carried no cursor"):
            check.resolve_open_file_sets("owner/name", "token")

    def test_a_list_longer_than_the_page_bound_is_unreadable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "data": {
                "repository": {"pullRequests": {"pageInfo": {"hasNextPage": True, "endCursor": "next"}, "nodes": []}}
            }
        }
        monkeypatch.setattr(check, "_post", lambda *a, **k: payload)
        with pytest.raises(check.ClaimSetUnreadable, match="the list was truncated"):
            check.resolve_open_file_sets("owner/name", "token")


class TestTheOpenSetIsReadLiveAndWholeAndIncludesDrafts:
    """Every page, every open pull request, from the repository rather than search."""

    def test_every_page_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pages = [
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "pageInfo": {"hasNextPage": True, "endCursor": "c1"},
                            "nodes": [_node(11, _added("tests/test_a.py"))],
                        }
                    }
                }
            },
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": [_node(12, _added("tests/test_a.py"))],
                        }
                    }
                }
            },
        ]
        seen: list[object] = []

        def fake_post(query: str, variables: dict[str, object], token: str) -> object:
            seen.append(variables.get("after"))
            return pages[len(seen) - 1]

        monkeypatch.setattr(check, "_post", fake_post)
        additions = check.resolve_open_file_sets("owner/name", "token")
        assert seen == [None, "c1"]
        # The finding only exists across the page boundary, so a single-page read
        # would report clean here.
        assert check.classify_additions(additions).outcome == check.DUPLICATE_ADDITION

    def test_the_open_set_is_read_from_the_repository_and_not_from_search(self) -> None:
        """Search is eventually consistent, and a pull request opened seconds ago
        is exactly the row this sweep exists to find."""
        assert "pullRequests(states: OPEN" in check._ADDITIONS_QUERY
        assert "search(" not in check._ADDITIONS_QUERY

    def test_no_pull_request_is_excluded_for_being_a_draft(self) -> None:
        """This file's claim-keyed policy, not the sibling sweep's.

        A draft's new file is authored work whatever its merge state, so excluding
        one would hide a collision for as long as either side stayed a draft.
        """
        source = _SCRIPT.read_text(encoding="utf-8")
        sweep = source[source.index("def resolve_open_file_sets") : source.index("def render_additions")]
        assert "draft" not in sweep.lower()
        assert "draft" not in check._ADDITIONS_QUERY.lower()

    def test_the_whole_set_is_compared_with_nothing_under_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No pull request is "the subject", so none is left out of the comparison."""
        payload = {
            "data": {
                "repository": {
                    "pullRequests": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [_node(11, _added("a.py")), _node(12, _added("b.py"))],
                    }
                }
            }
        }
        monkeypatch.setattr(check, "_post", lambda *a, **k: payload)
        assert sorted(check.resolve_open_file_sets("owner/name", "token")) == [11, 12]


class TestTheReportNamesBothPullRequestsAndTheRemedy:
    """What a reader has to be able to act on."""

    @staticmethod
    def _finding() -> str:
        _, additions, _ = _CLAIM_FREE_PAIRS[1]
        return check.render_additions(check.classify_additions(_open_set(additions)), "strands-labs/robots")

    def test_the_finding_names_both_pull_requests_and_the_shared_file(self) -> None:
        report = self._finding()
        assert "#2707 + #2708" in report
        assert "tests/training/test_checkpoint_cadence_domain.py" in report

    def test_the_finding_offers_the_remedy_and_says_no_push_settles_it(self) -> None:
        report = self._finding()
        assert "What clears this" in report
        assert "Close whichever of the two is redundant" in report
        assert "no push by one of them settles it" in report

    def test_the_finding_separates_itself_from_the_composition_question(self) -> None:
        assert "not a merge order to decide" in self._finding()

    def test_a_clean_report_carries_no_remedy_section(self) -> None:
        report = check.render_additions(check.classify_additions({11: _creates("a.py")}), "owner/name")
        assert check.UNIQUE_ADDITIONS in report
        assert "What clears this" not in report

    def test_a_clean_report_states_what_it_looked_at(self) -> None:
        report = check.render_additions(check.classify_additions({n: _creates() for n in (11, 12, 13)}), "owner/name")
        assert "| open pull requests read | 3 |" in report
        assert "| pairs compared | 3 |" in report

    def test_one_row_per_colliding_pair(self) -> None:
        additions = {n: _creates("tests/test_a.py") for n in (11, 12, 13)}
        report = check.render_additions(check.classify_additions(additions), "owner/name")
        for pair in ("#11 + #12", "#11 + #13", "#12 + #13"):
            assert pair in report


class TestTheExitStatusIsOneOnlyForTheFinding:
    """The contract the sibling gate scripts share: 1 is a finding, 2 is a usage error."""

    @pytest.mark.parametrize(
        ("additions", "expected"),
        [
            ({11: _creates("a.py"), 12: _creates("a.py")}, 1),
            ({11: _creates("a.py"), 12: _creates("b.py")}, 0),
        ],
    )
    def test_the_exit_status(self, monkeypatch: pytest.MonkeyPatch, additions: dict[int, Any], expected: int) -> None:
        monkeypatch.setattr(check, "resolve_open_file_sets", lambda *a, **k: additions)
        argv = ["--repo", "owner/name", "--all-open", "--token", "t"]
        assert check.main(argv) == expected

    def test_a_lookup_failure_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> dict[int, Any]:
            raise check.ClaimSetUnreadable("the response carried no repository.")

        monkeypatch.setattr(check, "resolve_open_file_sets", boom)
        assert check.main(["--repo", "owner/name", "--all-open", "--token", "t"]) == 0

    @pytest.mark.parametrize(
        "argv",
        [
            ["--repo", "owner/name", "--token", "t"],
            ["--repo", "owner/name", "--all-open", "--pr", "5", "--token", "t"],
            ["--repo", "owner/name", "--all-open", "--issue", "5", "--token", "t"],
            ["--repo", "owner/name", "--all-open", "--pr", "5", "--issue", "6", "--token", "t"],
        ],
    )
    def test_exactly_one_subject_is_required(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit) as raised:
            check.main(argv)
        assert raised.value.code == 2

    def test_the_sweep_keeps_the_inferred_repository_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Its caller is a workflow running where the pull requests live, and a
        sweep of the wrong repository is visible in its own report."""
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/name")
        monkeypatch.setattr(check, "resolve_open_file_sets", lambda *a, **k: {})
        assert check.main(["--all-open", "--token", "t"]) == 0

    def test_the_sweep_reads_no_claim_and_no_single_pull_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The two keys are independent: neither lookup of the other runs here."""

        def forbidden(*args: object, **kwargs: object) -> object:
            raise AssertionError("the added-path sweep must not read a claim")

        monkeypatch.setattr(check, "resolve_claim", forbidden)
        monkeypatch.setattr(check, "resolve_open_claims", forbidden)
        monkeypatch.setattr(check, "resolve_open_file_sets", lambda *a, **k: {11: _creates("a.py")})
        assert check.main(["--repo", "owner/name", "--all-open", "--token", "t"]) == 0


class TestTheGuidanceRecordsTheSecondKey:
    """AGENTS.md step 1 owns the duplicate-work convention, both keys of it."""

    @staticmethod
    def _step_one() -> str:
        text = _AGENTS.read_text(encoding="utf-8")
        start = text.index("1. Create the feature branch")
        end = text.index("2. Make changes", start)
        return " ".join(text[start:end].split())

    @pytest.mark.parametrize(
        "phrase",
        [
            "check_duplicate_claim.py --repo strands-labs/robots --all-open",
            "link no issue at all",
            "create the same thing",
            "350 distinct slugs",
            "a path set is a property of a pushed branch",
            "check_merge_base_overlap.py",
        ],
    )
    def test_step_one_carries_the_second_key(self, phrase: str) -> None:
        assert phrase in self._step_one()

    def test_the_guidance_keeps_the_intake_check_it_already_had(self) -> None:
        """The second key is additive: the claim-keyed intake question stays."""
        step_one = self._step_one()
        assert "check_duplicate_claim.py --repo strands-labs/robots --issue" in step_one
        assert "check that no open pull request already claims the issue" in step_one


class TestTheModuleStatesTheMeasurementBehindTheRelation:
    """A relation over paths is only worth wiring up if its precision is stated."""

    @staticmethod
    def _doc() -> str:
        """The docstring with runs of whitespace collapsed.

        So a pin is on the sentence rather than on where the paragraph happens to
        wrap: re-flowing prose must not be able to drop a measurement silently,
        and must not be able to fail this class either.
        """
        assert check.__doc__ is not None
        return " ".join(check.__doc__.split())

    @pytest.mark.parametrize(
        "phrase",
        [
            "2002 pairs",
            "both add a path",
            "both create the same thing",
            "350 distinct slugs",
            "none from both",
            "a path set is a property of a *pushed",
        ],
    )
    def test_the_docstring_carries_the_measurement(self, phrase: str) -> None:
        assert phrase in self._doc()

    @pytest.mark.parametrize(
        "phrase",
        [
            "would fire on every pair in the queue",
            "40 of those 350",
            "selected the same two pairs plus one and lost none",
        ],
    )
    def test_the_docstring_states_why_the_number_is_dropped(self, phrase: str) -> None:
        """The widening replaced a stated certainty, so its own bound is stated too.

        The exclusion it lifted was justified by a fear -- that keying on a
        fragment fires on the whole queue -- and a widening that does not answer
        the reason for the boundary it moved is indistinguishable from having
        missed it.
        """
        assert phrase in self._doc()

    def test_the_stale_scope_bullet_no_longer_calls_the_pair_unreachable(self) -> None:
        """The 18-of-30 measurement stands; the conclusion drawn from it was wider.

        It rules out *requiring* a claim, which neither claim-keyed mode does. It
        never ruled out colliding the pair on a different key.
        """
        doc = self._doc()
        assert "18 of the last 30 merges" in doc
        assert "collides it on what the two branches create" in doc


class TestALongFileListIsCompletedRatherThanRefused:
    """One pull request changing more files than a page does not blind the sweep.

    :func:`file_sets` refuses a node whose file list stopped short of its own
    ``totalCount``, because reading a prefix would report a clean answer computed
    from part of a branch. That refusal is right about the node. What was missing
    is that :func:`resolve_open_file_sets` handed it a node it had made no attempt
    to complete, so one oversized pull request raised out of the whole sweep and
    every pair on the board went ungraded -- including pairs that have nothing to
    do with the long branch. The paths are still read whole; the reader now pages
    until the node's own total is in hand, and the refusal stays as the backstop
    for a list that cannot be completed inside the bound.
    """

    @staticmethod
    def _open_page(nodes: list[dict[str, Any]]) -> dict[str, Any]:
        """The single page of the open-pull-request query."""
        return {
            "data": {
                "repository": {
                    "pullRequests": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": nodes,
                    }
                }
            }
        }

    @staticmethod
    def _files_page(paths: list[str], *, cursor: str | None) -> dict[str, Any]:
        """One page of a single pull request's own changed-file list."""
        return {
            "data": {
                "repository": {
                    "pullRequest": {
                        "files": {
                            "pageInfo": {
                                "hasNextPage": cursor is not None,
                                "endCursor": cursor,
                            },
                            "nodes": _added(*paths),
                        }
                    }
                }
            }
        }

    @staticmethod
    def _long_node(number: int, paths: list[str], *, total: int, cursor: str | None) -> dict[str, Any]:
        """A node whose file list is the first page of a longer list."""
        node = _node(number, _added(*paths), total=total)
        node["files"]["pageInfo"] = {
            "hasNextPage": cursor is not None,
            "endCursor": cursor,
        }
        return node

    def _wire(
        self,
        monkeypatch: pytest.MonkeyPatch,
        nodes: list[dict[str, Any]],
        file_pages: list[dict[str, Any]],
    ) -> list[object]:
        """Answer the open-set query once and the per-pull file query from a queue.

        The two queries are told apart by their variables: only the file query
        carries the pull request's own number.
        """
        cursors: list[object] = []

        def fake_post(query: str, variables: dict[str, object], token: str) -> dict[str, Any]:
            if "number" in variables:
                cursors.append(variables.get("after"))
                return file_pages[len(cursors) - 1]
            return self._open_page(nodes)

        monkeypatch.setattr(check, "_post", fake_post)
        return cursors

    def test_a_collision_in_a_later_page_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The finding lives past the first page, which is the whole point.

        A pull request long enough to be paged collides with a one-file sibling on
        a path the first page does not carry. Reading the first page alone reports
        a clean board; refusing the node reports nothing at all.
        """
        first = [f"src/module_{index}.py" for index in range(check.FILE_PAGE_SIZE)]
        shared = "tests/test_shared_surface.py"
        nodes = [
            self._long_node(11, first, total=len(first) + 1, cursor="cursor-1"),
            _node(12, _added(shared)),
        ]
        cursors = self._wire(monkeypatch, nodes, [self._files_page([shared], cursor=None)])

        file_sets = check.resolve_open_file_sets("owner/name", "token")

        assert cursors == ["cursor-1"], "the node's own cursor, followed once"
        assert shared in file_sets[11].created
        verdict = check.classify_additions(file_sets)
        assert verdict.outcome == check.DUPLICATE_ADDITION
        assert (11, 12) in [(low, high) for low, high, _ in verdict.collisions]

    def test_one_long_pull_request_no_longer_blinds_the_rest_of_the_board(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The measured harm: a pair that shares nothing with the long branch.

        Before this, the long node raised before any pair was compared, so the
        sweep answered ``unknown-additions`` for the whole board and a duplicate
        between two unrelated one-file pull requests went unreported.
        """
        first = [f"src/module_{index}.py" for index in range(check.FILE_PAGE_SIZE)]
        shared = "tests/test_unrelated_pair.py"
        nodes = [
            self._long_node(11, first, total=len(first) + 1, cursor="cursor-1"),
            _node(21, _added(shared)),
            _node(22, _added(shared)),
        ]
        self._wire(
            monkeypatch,
            nodes,
            [self._files_page(["src/only_mine.py"], cursor=None)],
        )

        verdict = check.classify_additions(check.resolve_open_file_sets("owner/name", "token"))

        assert verdict.outcome == check.DUPLICATE_ADDITION
        assert (21, 22) in [(low, high) for low, high, _ in verdict.collisions]

    def test_a_single_page_pull_request_costs_no_extra_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ordinary traffic is unchanged, which is the bound on the widening.

        A node whose list is already whole reports ``hasNextPage: false``, so the
        reader stops without asking for a page it does not need.
        """
        nodes = [_node(11, _added("tests/test_a.py"))]
        cursors = self._wire(monkeypatch, nodes, [])

        file_sets = check.resolve_open_file_sets("owner/name", "token")

        assert cursors == [], "no page was requested for a list already in hand"
        assert file_sets[11].created == ("tests/test_a.py",)

    def test_a_list_longer_than_the_bound_is_still_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The refusal survives as the backstop, naming the bound it hit.

        Paging is bounded, so a list that cannot be completed inside it is
        reported rather than read as a prefix -- the property the shipped refusal
        was protecting, kept.
        """
        first = [f"src/module_{index}.py" for index in range(check.FILE_PAGE_SIZE)]
        nodes = [self._long_node(11, first, total=10_000, cursor="cursor-1")]
        pages = [
            self._files_page([f"src/more_{page}.py"], cursor=f"cursor-{page + 2}")
            for page in range(check.MAX_FILE_PAGES + 2)
        ]
        self._wire(monkeypatch, nodes, pages)

        with pytest.raises(check.ClaimSetUnreadable) as raised:
            check.resolve_open_file_sets("owner/name", "token")

        message = str(raised.value)
        assert "#11" in message
        assert str(check.MAX_FILE_PAGES * check.FILE_PAGE_SIZE) in message

    def test_a_list_that_runs_out_of_pages_short_of_its_total_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Paging is not a licence to read a prefix.

        A node that stops offering pages while its own ``totalCount`` is unmet is
        reported, not treated as complete -- the refusal this reader exists to
        stop reaching for an ordinary long branch is still the answer for a list
        the API will not finish handing over.
        """
        nodes = [self._long_node(11, ["src/a.py"], total=500, cursor="cursor-1")]
        self._wire(monkeypatch, nodes, [self._files_page(["src/b.py"], cursor=None)])

        with pytest.raises(check.ClaimSetUnreadable) as raised:
            check.resolve_open_file_sets("owner/name", "token")

        message = str(raised.value)
        assert "#11" in message
        assert "500" in message and "2" in message, "the shortfall is quantified"

    def test_a_page_that_promises_more_and_names_no_cursor_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reader that looped on a missing cursor would ask for the same page
        forever, so the unusable answer is reported instead."""
        node = _node(11, _added("src/a.py"), total=500)
        node["files"]["pageInfo"] = {"hasNextPage": True, "endCursor": None}
        self._wire(monkeypatch, [node], [])

        with pytest.raises(check.ClaimSetUnreadable) as raised:
            check.resolve_open_file_sets("owner/name", "token")

        assert "#11" in str(raised.value)

    def test_the_completion_precedes_the_read_that_would_refuse(self) -> None:
        """Order, not presence: a node read before it is completed hits the
        refusal the completion exists to make unnecessary."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(check.resolve_open_file_sets)))
        function = tree.body[0]
        assert isinstance(function, ast.FunctionDef)
        # Drop the docstring: it names ``file_sets`` in prose, so an offset
        # comparison over the raw source would grade that mention.
        body = function.body[1:] if ast.get_docstring(function) else function.body
        called = [
            node.func.id
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assert "complete_file_nodes" in called, "the reader no longer completes a node's file list"
        assert "file_sets" in called, "the reader no longer reads the file set"
        assert called.index("complete_file_nodes") < called.index("file_sets"), (
            "the node is read before it is completed, so the refusal fires first"
        )
