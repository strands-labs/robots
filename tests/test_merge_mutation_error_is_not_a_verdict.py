"""Contract pins for what a ``mergePullRequest`` error does and does not decide.

``AGENTS.md`` > PR Workflow > step 8 covers the read-back a mutation needs on both
sides, and its *After.* half names the case where the mutation reports the wrong
thing: ``Pull Request is not mergeable`` returned for a squash that is already on
``main``. Until now that read as a rare race worth a footnote, evidenced by one
pull request from 2026-07-30 - #1756, merge commit ``4bf139c``.

It is not rare. Three of three recorded merges returned it, and the two most
recent were thirty seconds apart:

    pull request  mutation error                  merge commit   merged at
    #1756         Pull Request is not mergeable   4bf139c        2026-07-30 06:55:29Z
    #2249         Pull Request is not mergeable   926beb9        2026-08-13 19:24:50Z
    #2250         Pull Request is not mergeable   07a759d        2026-08-13 19:25:20Z

Every cheap explanation is ruled out by the payload that produced those two rows.
Each was a single call - no earlier attempt against the same pull request - and
each carried ``expectedHeadOid``, so a stale oid cannot be the refusal's cause;
each target read ``mergeStateStatus: CLEAN``, ``mergeable: MERGEABLE``,
``reviewDecision: APPROVED`` with every required context ``SUCCESS`` moments
before. There is nothing in the response to key a verdict on either: ``errors[0]``
is that sentence and ``data.mergePullRequest`` is ``null`` whether the squash
landed or not.

Which makes the error worse than merely noisy. An agent that believes it does not
stop at a wrong status line - it re-derives work that is already on ``main``, and
the natural first move, retrying the mutation, cannot correct the belief: a second
call against #2249 *after* it had merged returned the identical error beside the
identical ``null``. Two identical failures read as corroboration, so the retry
converts one misread into a confident one.

The read-back that step 8 already prescribes is the only thing that separates the
two outcomes, and it also names the likely mechanism, in the very field the error
is worded about: once #2249 was merged its ``mergeStateStatus`` and ``mergeable``
both report ``UNKNOWN``, which is what a mutation re-reading the pull request it
has just closed would see on its way out.

Three classes:

``TestTheErrorDoesNotSeparateTheOutcomes`` runs both verdicts over the recorded
observations - one keyed on the mutation response, one on the ``state``/``merged``
read-back - and asserts the first is wrong on every row while the second is right
on every row. That is the arithmetic behind the guidance, so the pin says *why*
the error is uninformative rather than only that the prose mentions it.

``TestARetryCannotCorrectTheMisread`` pins the repeat call: same input, same
error, same ``null``, against a pull request that by then could not be merged by
anyone. It is the response an agent would treat as confirmation.

``TestTheGuidanceKeepsTheCorrectionAdjacent`` pins the prose, because the prose is
the deliverable: an agent reads ``AGENTS.md``, not this module. What is asserted is
*adjacency* - the recurrence and the retry warning have to stay inside the same
bullet as the instruction they qualify, since "can report" reads perfectly well
alone and is exactly what a later tidy-up would leave behind. Same shape as
``tests/test_timeline_filter_count_is_unfiltered.py``,
``tests/test_merge_gate_viewer_scope.py`` and
``tests/test_graphql_node_id_targeting.py`` use for step 8's other reading
disciplines.

Negative control: with ``origin/main``'s ``AGENTS.md`` restored, 4 of the 5 tests
in ``TestTheGuidanceKeepsTheCorrectionAdjacent`` fail - the fifth reads the
read-back instruction, which that tree already carries, and the anchor guard
locates the bullet on both trees - while all 17 offline tests pass unchanged: the
API's behaviour is a property of GitHub, not of this change, and only the guidance
is new.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"

#: The error message GitHub returns, verbatim.
_ERROR = "Pull Request is not mergeable"

#: One ``mergePullRequest`` mutation per row, with the response it returned and
#: the state a read-back found afterwards. #2249 and #2250 were recorded
#: 2026-08-13 with ``PAT_TOKEN``; #1756 is the case ``AGENTS.md`` already carried,
#: its merge commit and timestamp read back from the API.
_OBSERVED: dict[int, dict[str, Any]] = {
    1756: {
        "expected_head_oid_supplied": True,
        "errors": [_ERROR],
        "data": {"mergePullRequest": None},
        "read_back": {
            "state": "MERGED",
            "merged": True,
            "merge_commit": "4bf139cc311ee18ae062519bac8c9cfeebda39b2",
            "merged_at": "2026-07-30T06:55:29Z",
        },
    },
    2249: {
        "expected_head_oid_supplied": True,
        "errors": [_ERROR],
        "data": {"mergePullRequest": None},
        "read_back": {
            "state": "MERGED",
            "merged": True,
            "merge_commit": "926beb930e8f68336a8211ee0bbbb11f4665795f",
            "merged_at": "2026-08-13T19:24:50Z",
            # Read in the same query as ``state``: the fields the error is worded
            # about no longer answer, because the pull request is closed.
            "merge_state_status": "UNKNOWN",
            "mergeable": "UNKNOWN",
        },
    },
    2250: {
        "expected_head_oid_supplied": True,
        "errors": [_ERROR],
        "data": {"mergePullRequest": None},
        "read_back": {
            "state": "MERGED",
            "merged": True,
            "merge_commit": "07a759d6491c9a76c6655d2d9454fa00ddc66d4c",
            "merged_at": "2026-08-13T19:25:20Z",
        },
    },
}

#: The pre-mutation gate read on #2249 and #2250, moments before each call. Every
#: field step 8 tells you to read said "yes".
_GATE_BEFORE: dict[int, dict[str, Any]] = {
    2249: {
        "mergeStateStatus": "CLEAN",
        "mergeable": "MERGEABLE",
        "reviewDecision": "APPROVED",
        "required_contexts_all_success": True,
    },
    2250: {
        "mergeStateStatus": "CLEAN",
        "mergeable": "MERGEABLE",
        "reviewDecision": "APPROVED",
        "required_contexts_all_success": True,
    },
}

#: The repeat call against #2249 after it had merged, same input as the first.
_RETRY_ON_MERGED: dict[str, Any] = {
    "pull_request": 2249,
    "expected_head_oid": "9decef7e6199d695ba928fd45e89f42f055193e5",
    "errors": [_ERROR],
    "data": {"mergePullRequest": None},
}


def _verdict_from_mutation_response(observation: dict[str, Any]) -> bool:
    """Did the merge land, judged only by what the mutation returned?

    This is the derivation step 8 exists to forbid, written out so the pin can
    show what it gets wrong.
    """
    return not observation["errors"] and observation["data"]["mergePullRequest"] is not None


def _verdict_from_read_back(observation: dict[str, Any]) -> bool:
    """Did the merge land, judged by reading the pull request back?"""
    read_back = observation["read_back"]
    return read_back["merged"] and read_back["state"] == "MERGED"


class TestTheErrorDoesNotSeparateTheOutcomes:
    """The mutation response is the same on a refusal and on a success."""

    @pytest.mark.parametrize("number", sorted(_OBSERVED))
    def test_the_merge_landed(self, number: int) -> None:
        assert _verdict_from_read_back(_OBSERVED[number]), (
            f"#{number} is recorded here because its squash reached main; a row whose "
            "merge did not land belongs in a different fixture"
        )

    @pytest.mark.parametrize("number", sorted(_OBSERVED))
    def test_the_mutation_response_says_it_did_not(self, number: int) -> None:
        assert _verdict_from_mutation_response(_OBSERVED[number]) is False, (
            f"#{number}'s mutation response must read as a failure - that is the whole defect being pinned"
        )

    def test_the_response_keyed_verdict_is_wrong_on_every_row(self) -> None:
        wrong = [
            number
            for number, observation in _OBSERVED.items()
            if _verdict_from_mutation_response(observation) != _verdict_from_read_back(observation)
        ]
        assert sorted(wrong) == sorted(_OBSERVED), (
            "the point of the guidance is that the mutation response misleads every time, "
            f"not occasionally; it agreed with the truth on {sorted(set(_OBSERVED) - set(wrong))}"
        )

    def test_the_error_text_carries_no_discriminator(self) -> None:
        assert {tuple(observation["errors"]) for observation in _OBSERVED.values()} == {(_ERROR,)}
        assert {observation["data"]["mergePullRequest"] for observation in _OBSERVED.values()} == {None}

    @pytest.mark.parametrize("number", sorted(_GATE_BEFORE))
    def test_the_gate_read_before_the_call_said_yes(self, number: int) -> None:
        gate = _GATE_BEFORE[number]
        assert gate == {
            "mergeStateStatus": "CLEAN",
            "mergeable": "MERGEABLE",
            "reviewDecision": "APPROVED",
            "required_contexts_all_success": True,
        }, f"#{number} was merged on a fully green gate, so the error cannot be read as the gate having refused"

    @pytest.mark.parametrize("number", sorted(_OBSERVED))
    def test_a_stale_expected_head_oid_is_ruled_out(self, number: int) -> None:
        assert _OBSERVED[number]["expected_head_oid_supplied"] is True

    def test_the_fields_the_error_is_worded_about_stop_answering(self) -> None:
        after = _OBSERVED[2249]["read_back"]
        assert after["merge_state_status"] == "UNKNOWN"
        assert after["mergeable"] == "UNKNOWN"


class TestARetryCannotCorrectTheMisread:
    """Retrying returns the same payload, so it reads as corroboration."""

    def test_the_retry_targets_an_already_merged_pull_request(self) -> None:
        assert _OBSERVED[_RETRY_ON_MERGED["pull_request"]]["read_back"]["merged"] is True

    def test_the_retry_returns_the_identical_response(self) -> None:
        first = _OBSERVED[_RETRY_ON_MERGED["pull_request"]]
        assert _RETRY_ON_MERGED["errors"] == first["errors"]
        assert _RETRY_ON_MERGED["data"] == first["data"]

    def test_the_retry_verdict_is_wrong_in_the_same_direction(self) -> None:
        assert _verdict_from_mutation_response(_RETRY_ON_MERGED) is False, (
            "two identical failures read as confirmation, which is what makes the retry "
            "reflex more expensive than the single misread"
        )


#: Locates step 8's *After.* bullet in ``AGENTS.md``. The heading, not the correction.
_ANCHOR = "- *After.* A `mergePullRequest` mutation can report `Pull Request is not"

#: The next bullet, which bounds the passage.
_NEXT_BULLET = "- *And on `main` afterwards.*"


@pytest.fixture(scope="module")
def bullet() -> str:
    """The whole *After.* bullet, so every prose assertion also asserts adjacency."""
    text = _AGENTS_PATH.read_text(encoding="utf-8")
    assert _ANCHOR in text, (
        f"AGENTS.md no longer contains {_ANCHOR!r}, which this module uses to locate step 8's "
        "after-the-mutation read. Re-point the anchor at the passage that replaced it rather "
        "than deleting these tests"
    )
    start = text.index(_ANCHOR)
    end = text.index(_NEXT_BULLET, start)
    return text[start:end]


class TestTheGuidanceKeepsTheCorrectionAdjacent:
    """The prose is the deliverable. Assert it, and assert it stays in place."""

    def test_it_still_prescribes_the_read_back(self, bullet: str) -> None:
        assert "`state`/`merged`" in bullet
        assert "git log origin/main" in bullet

    def test_it_says_the_error_recurs(self, bullet: str) -> None:
        assert "#2249" in bullet and "#2250" in bullet, (
            "AGENTS.md must carry the recurrence, because one case from a fortnight earlier "
            "reads as a race worth ignoring and three in a row do not"
        )
        assert "926beb9" in bullet and "07a759d" in bullet, (
            "name the merge commits: the claim is that the squashes landed, and a commit on "
            "main is the only evidence a reader can check"
        )

    def test_it_rules_out_the_cheap_explanation(self, bullet: str) -> None:
        assert "expectedHeadOid" in bullet, (
            "a reader's first hypothesis is a stale head oid; the guidance has to close it "
            "or the recurrence looks like the caller's own bug"
        )

    def test_it_says_a_retry_is_not_the_remedy(self, bullet: str) -> None:
        assert "retry" in bullet.lower(), (
            "the read-back instruction alone does not stop the retry reflex, and the retry is "
            "what turns one misread into a confident one"
        )

    def test_it_names_this_pin(self, bullet: str) -> None:
        assert Path(__file__).name in bullet, (
            "AGENTS.md names the module that pins each measured claim, so a reader can find "
            "the payloads behind these numbers"
        )
