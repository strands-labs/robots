"""Contract pin for the two states a ``null`` ``reviewDecision`` conflates.

``AGENTS.md`` > PR Workflow > step 8 documents ``reviewDecision: null`` as a
third reading meaning "one resolve from merging". That was recorded from #1974,
where resolving the sole unresolved thread moved ``mergeStateStatus`` to
``CLEAN`` and the decision to ``APPROVED``. The measurement is real. What the
passage did not name is the input that made the resolve *sufficient*, and it was
then acted on where that input was absent.

#2328 presented the same signature - ``mergeable: MERGEABLE``,
``mergeStateStatus: BLOCKED``, ``reviewDecision: null``, one unresolved
``github-advanced-security`` thread, ``call-test-lint / Test and Lint``
``SUCCESS`` - and resolving that thread settled the decision the other way:

===============  ===================================  ================================
pull request     approving review present             after ``resolveReviewThread``
===============  ===================================  ================================
#1974            one ``APPROVED``, post-dating head   ``APPROVED`` / ``CLEAN``
#2328            none - every review ``COMMENTED``     ``REVIEW_REQUIRED`` / ``BLOCKED``
===============  ===================================  ================================

``null`` is also what a pull request with no approving review at all reads,
because a ``COMMENTED`` review contributes no approval. So the resolve was
necessary on both and sufficient on one, and the documented reading - "this one
needs no review at all" - was true of #1974 and false of #2328 while the field
was byte-identical on the two. It misreads in the reassuring direction, which is
the property step 8 already flags for ``null``: acting on it yields a resolve, a
re-read expecting ``CLEAN``, and a pull request reported as ready while it waits
on a first approving review - the presentation #1905 records for another cause.

Two halves are pinned, for the reason ``c56ab08`` pinned both halves of the
merge-mutation bullet. The *arithmetic*: a verdict keyed on the decision and the
threads alone is wrong on one of the two recorded rows, while one that also
reads the review set is right on both. The *prose*, asserted inside the same
bullet, so a later tidy-up cannot leave the bare instruction behind.

The ordering assertion is the one with teeth. #2328's decision moved from
``null`` to ``REVIEW_REQUIRED`` **on the resolve**, so the single value that
identifies which case a reader is in is destroyed by the action the passage
prescribes. A re-read afterwards cannot recover it, which is why the review set
has to be read first rather than merely read. See #2343.

These are text assertions rather than a parsed document because the claim being
pinned is prose and that is the shape the existing ``AGENTS.md`` pins use
(``tests/test_merge_gate_viewer_scope.py`` and
``tests/test_codeql_query_filters.py`` read the file the same way).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"

#: What was measured on each pull request, before and after the resolve the
#: passage prescribes. ``review_states`` is the whole review set, because the
#: correction turns on a set containing no ``APPROVED`` rather than on a count.
_OBSERVED: dict[int, dict[str, Any]] = {
    1974: {
        "review_decision_before": None,
        "merge_state_status_before": "BLOCKED",
        "review_states": ("APPROVED",),
        "unresolved_threads_before": 1,
        "review_decision_after_resolve": "APPROVED",
        "merge_state_status_after_resolve": "CLEAN",
        "resolve_was_sufficient": True,
    },
    2328: {
        "review_decision_before": None,
        "merge_state_status_before": "BLOCKED",
        "review_states": ("COMMENTED", "COMMENTED", "COMMENTED"),
        "unresolved_threads_before": 1,
        "review_decision_after_resolve": "REVIEW_REQUIRED",
        "merge_state_status_after_resolve": "BLOCKED",
        "resolve_was_sufficient": False,
    },
}


def _agents_text() -> str:
    return _AGENTS_PATH.read_text(encoding="utf-8")


def _resolve_is_sufficient_by_decision_alone(observation: dict[str, Any]) -> bool:
    """The uncorrected reading: ``null`` plus a thread means one resolve from merging."""
    return observation["review_decision_before"] is None and observation["unresolved_threads_before"] > 0


def _resolve_is_sufficient_by_review_set(observation: dict[str, Any]) -> bool:
    """The corrected reading: an approving review has to be there already."""
    return (
        observation["review_decision_before"] is None
        and observation["unresolved_threads_before"] > 0
        and "APPROVED" in observation["review_states"]
    )


class TestTheDecisionAloneDoesNotPredictTheResolve:
    """The arithmetic half: which reading matches the two recorded outcomes."""

    def test_the_two_rows_are_indistinguishable_before_the_resolve(self) -> None:
        # The premise of the whole bullet: if the pre-resolve fields differed, a
        # reader could tell the cases apart without the review set and none of
        # this would be worth documenting.
        before = {
            number: (
                observation["review_decision_before"],
                observation["merge_state_status_before"],
                observation["unresolved_threads_before"],
            )
            for number, observation in _OBSERVED.items()
        }
        assert before[1974] == before[2328], (
            "The recorded pre-resolve fields for #1974 and #2328 now differ, which would "
            f"mean the null reading is distinguishable without the review set: {before}. "
            "If a measurement was corrected, re-derive the AGENTS.md claim rather than "
            "adjusting this fixture to keep it passing."
        )

    def test_the_uncorrected_reading_is_wrong_on_one_of_the_two_rows(self) -> None:
        wrong = [
            number
            for number, observation in _OBSERVED.items()
            if _resolve_is_sufficient_by_decision_alone(observation) is not observation["resolve_was_sufficient"]
        ]
        assert wrong == [2328], (
            "A verdict keyed on reviewDecision and the threads alone should predict the "
            f"resolve correctly on #1974 and incorrectly on #2328; mispredicted: {wrong}. "
            "That asymmetry is the reason step 8 needs the review set."
        )

    def test_reading_the_review_set_is_right_on_both_rows(self) -> None:
        for number, observation in _OBSERVED.items():
            assert _resolve_is_sufficient_by_review_set(observation) is observation["resolve_was_sufficient"], (
                f"Reading the review set alongside the threads mispredicts #{number}. "
                "The corrected AGENTS.md instruction rests on this being right on both."
            )

    def test_a_commented_review_contributes_no_approval(self) -> None:
        # Why null cannot be taken to imply an approval exists: #2328 read null
        # with no APPROVED review anywhere in its set.
        assert "APPROVED" not in _OBSERVED[2328]["review_states"]
        assert _OBSERVED[2328]["review_decision_before"] is None, (
            "#2328 is the counterexample only because it read null while carrying no "
            "approving review. Without that, the bullet's correction has no evidence."
        )

    def test_the_resolve_destroys_the_value_that_identifies_the_case(self) -> None:
        observation = _OBSERVED[2328]
        assert observation["review_decision_before"] is None
        assert observation["review_decision_after_resolve"] == "REVIEW_REQUIRED", (
            "The ordering instruction in step 8 - read the review set before resolving - "
            "rests on the null being gone afterwards. If #2328's decision is recorded as "
            "still null after the resolve, the reason for the ordering has changed."
        )


#: The sentence the correction introduces. Everything else is positioned from
#: it, so its absence fails loudly rather than making the rest vacuous.
_CORRECTION = "at least two states"

#: The read the correction prescribes, and the fact that makes its ordering
#: load-bearing.
_REVIEW_SET_READ = "reviews(last: 20)"
_ORDERING_RATIONALE = "gone by the"

#: The claim that was wrong when unqualified. It may still appear - the
#: correction quotes it to say which case it holds for - but only next to the
#: qualification.
_UNQUALIFIED_CLAIM = "needs no review at all"
_QUALIFICATION = "false of #2328"

#: How far apart two phrases may sit while still reading as one instruction.
_ADJACENCY_WINDOW = 1400


def _window_after(text: str, anchor: str) -> str | None:
    position = text.find(anchor)
    if position < 0:
        return None
    return text[position : position + _ADJACENCY_WINDOW]


class TestTheNullReadingNamesItsSecondState:
    """The prose half: the instruction and its qualifier stay in one breath."""

    def test_the_correction_is_still_present(self) -> None:
        # Context guard: the assertions below are positioned from this phrase, so
        # a silent rewording would move the pin rather than break it.
        assert _CORRECTION in _agents_text(), (
            f"AGENTS.md no longer contains {_CORRECTION!r}, which this class uses to locate "
            "the null-reviewDecision correction. If it was deliberately reworded, update "
            "_CORRECTION to match rather than deleting these tests - the point is that the "
            "null reading and its second state stay together. See #2343."
        )

    def test_the_second_state_is_named_with_its_counterexample(self) -> None:
        window = _window_after(_agents_text(), _CORRECTION)
        assert window is not None and "#2328" in window, (
            "AGENTS.md says a null reviewDecision is at least two states without naming the "
            "pull request that measured the second one. The claim is only actionable with "
            "the case attached: #2328 read null with no approving review, and resolving its "
            "thread produced REVIEW_REQUIRED rather than APPROVED. See #2343."
        )

    def test_the_review_set_read_is_prescribed(self) -> None:
        window = _window_after(_agents_text(), _CORRECTION)
        assert window is not None and _REVIEW_SET_READ in window, (
            "AGENTS.md names the second null state but does not give the query that tells the "
            "two apart, leaving a reader knowing the field is ambiguous and not what to read "
            "instead. The distinguishing read is the review set beside the threads. See #2343."
        )

    def test_the_read_is_ordered_before_the_resolve(self) -> None:
        window = _window_after(_agents_text(), _CORRECTION)
        assert window is not None and _ORDERING_RATIONALE in window, (
            "AGENTS.md prescribes reading the review set but no longer says why the order "
            "matters. #2328's decision moved from null to REVIEW_REQUIRED on the resolve, so "
            "a reader who resolves first cannot recover the value that identified the case, "
            "and the read is only useful beforehand. See #2343."
        )

    def test_the_unqualified_claim_is_never_restored(self) -> None:
        text = _agents_text()
        position = text.find(_UNQUALIFIED_CLAIM)
        while position >= 0:
            neighbourhood = text[max(0, position - _ADJACENCY_WINDOW) : position + _ADJACENCY_WINDOW]
            assert _QUALIFICATION in neighbourhood, (
                f"AGENTS.md asserts {_UNQUALIFIED_CLAIM!r} without the qualification that it "
                "holds for #1974 and not for #2328. Unqualified, that sentence sends a reader "
                "to resolve a thread and expect CLEAN on a pull request that has no approving "
                "review, which is the misread #2343 records. Keep the phrase beside "
                f"{_QUALIFICATION!r}, or reword it."
            )
            position = text.find(_UNQUALIFIED_CLAIM, position + 1)
