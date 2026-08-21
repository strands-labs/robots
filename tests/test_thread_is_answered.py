"""Pins for scripts/check_thread_is_answered.py.

The payloads here are not invented. Each one is the shape measured from a real
review thread in this repository, with the resolution flag set to what it was
during the window the check is meant to be run in -- which is the only window in
which a thread is not yet ``isResolved``. A sweep of the 150 most recently updated
closed pull requests finds 26 review threads and every one of them is resolved, so
the two discriminating outcomes cannot be covered by real closed data and are
covered here instead.

Measured sources:

- #2577, thread ``PRRT_kwDORUMiZs6bQMZz``: one reviewer comment plus two author
  replies (19:17:50Z, 19:28:20Z), head ``d04a8969``, thread commit ``b966ce64``,
  ``isOutdated: true``.
- #2511, thread ``PRRT_kwDORUMiZs6arq8z``: one reviewer comment plus three author
  replies (04:19, 04:31, 04:46), head ``0a69634d``, thread commit ``ee3e4526``.
- #2480, thread ``PRRT_kwDORUMiZs6aqPJ-``: a ``github-advanced-security`` comment
  answered by the author, ``isOutdated: false`` even after ``e83cf51`` fixed it.
- #1722: four threads carrying a single bot comment each.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_thread_is_answered.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_thread_is_answered", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()

# Measured on the pull requests named in the module docstring.
AUTHOR = "cagataycali"
REVIEWER = "yinsong1986"
BOT = "github-advanced-security"

HEAD_2577 = "d04a896937dbec1281e39eaa31eeffecf950889d"
COMMIT_2577 = "b966ce64b75edfe69e4f8ba73bfc391306703b1e"
HEAD_2511 = "0a69634d7f0d037b3c1d587744acb92e9f99bf59"
COMMIT_2511 = "ee3e452681fe4f649b87f50bdba40a07e706a192"


def _comment(login: str, *, bot: bool = False, oid: str = COMMIT_2577) -> dict:
    """One review-thread comment, in the shape the GraphQL query returns."""
    return {
        "author": {"login": login, "__typename": "Bot" if bot else "User"},
        "originalCommit": {"oid": oid},
    }


def _thread(
    comments: list[dict],
    *,
    resolved: bool = False,
    outdated: bool = False,
    path: str = "tests/simulation/mujoco/test_randomize_persistence_boundary.py",
    thread_id: str = "PRRT_kwDORUMiZs6bQMZz",
) -> dict:
    return {
        "id": thread_id,
        "isResolved": resolved,
        "isOutdated": outdated,
        "path": path,
        "comments": {"totalCount": len(comments), "nodes": comments},
    }


def _pr(threads: list[dict], *, number: int = 2577, author: str = AUTHOR, head: str = HEAD_2577) -> dict:
    return {
        "number": number,
        "isDraft": False,
        "headRefOid": head,
        "author": {"login": author},
        "reviewThreads": {"totalCount": len(threads), "nodes": threads},
    }


# --------------------------------------------------------------------------
# The two incidents this check was written for.
# --------------------------------------------------------------------------


def test_the_2577_thread_is_answered_when_its_third_comment_was_posted() -> None:
    """The reply that should not have been sent.

    At 19:28:20Z the thread held a reviewer comment and the author's own reply
    from 19:17:50Z, which already named ``d04a8969``. The check has to call that
    answered, because the reviewer's question is still sitting in the payload
    verbatim and is not evidence that anything is outstanding.
    """
    thread = _thread(
        [_comment(REVIEWER), _comment(AUTHOR)],
        outdated=True,
    )
    verdict = mod.classify(thread, AUTHOR, HEAD_2577)
    assert verdict.outcome == mod.ANSWERED
    assert verdict.is_owed is False
    assert verdict.last_author == AUTHOR


@pytest.mark.parametrize("replies", [1, 2, 3])
def test_the_2511_thread_is_answered_from_the_first_reply_onwards(replies: int) -> None:
    """One reviewer comment, then N author replies -- answered at every N.

    #2511 collected three author replies over 27 minutes, all announcing the same
    commit. The check must return the same verdict after the first as after the
    third, or it would license the second.
    """
    comments = [_comment(REVIEWER, oid=COMMIT_2511)]
    comments += [_comment(AUTHOR, oid=COMMIT_2511) for _ in range(replies)]
    verdict = mod.classify(
        _thread(comments, thread_id="PRRT_kwDORUMiZs6arq8z"),
        AUTHOR,
        HEAD_2511,
    )
    assert verdict.outcome == mod.ANSWERED


# --------------------------------------------------------------------------
# The authorship rule, and that it is what decides.
# --------------------------------------------------------------------------


def test_a_reviewer_comment_with_no_reply_is_owed() -> None:
    """The case that is work: the last word is the reviewer's."""
    verdict = mod.classify(_thread([_comment(REVIEWER)]), AUTHOR, HEAD_2577)
    assert verdict.outcome == mod.AWAITING_THE_AUTHOR
    assert verdict.is_owed is True


def test_flipping_only_the_last_comments_author_flips_the_outcome() -> None:
    """Non-vacuity: the authorship test is doing the deciding, not the payload shape.

    Same thread, same head, same commit, same comment count -- only the login on
    the final comment differs. A check that always agreed would not move here.
    """
    answered = mod.classify(_thread([_comment(REVIEWER), _comment(AUTHOR)]), AUTHOR, HEAD_2577)
    owed = mod.classify(_thread([_comment(AUTHOR), _comment(REVIEWER)]), AUTHOR, HEAD_2577)
    assert (answered.outcome, owed.outcome) == (mod.ANSWERED, mod.AWAITING_THE_AUTHOR)


def test_a_resolved_thread_is_settled_whoever_spoke_last() -> None:
    """Resolution is a reviewer action, so it outranks the authorship test.

    A resolved thread whose last comment is the reviewer's is closed business, not
    a demand: the reviewer had the last word and then resolved it anyway.
    """
    verdict = mod.classify(_thread([_comment(AUTHOR), _comment(REVIEWER)], resolved=True), AUTHOR, HEAD_2577)
    assert verdict.outcome == mod.SETTLED
    assert verdict.is_owed is False


def test_a_reviewer_demand_on_an_outdated_thread_is_owed() -> None:
    """The direction the ``isOutdated`` proxy gets wrong.

    #2577's thread went on taking comments after ``d04a8969`` flipped it to
    ``isOutdated: true``. A rule that read outdated as terminal would file a
    reviewer's demand arriving that way as settled; the authorship test does not.
    """
    verdict = mod.classify(_thread([_comment(AUTHOR), _comment(REVIEWER)], outdated=True), AUTHOR, HEAD_2577)
    assert verdict.outcome == mod.AWAITING_THE_AUTHOR
    assert verdict.outdated is True


# --------------------------------------------------------------------------
# Bots.
# --------------------------------------------------------------------------


def test_a_bot_comment_after_the_authors_reply_does_not_reopen_the_thread() -> None:
    """``github-advanced-security`` posts into threads (#2480, all four on #1722).

    A bot speaking after the author must not turn an answered thread back into
    work, which is why the *last non-bot* comment decides.
    """
    thread = _thread([_comment(REVIEWER), _comment(AUTHOR), _comment(BOT, bot=True)])
    verdict = mod.classify(thread, AUTHOR, HEAD_2577)
    assert verdict.outcome == mod.ANSWERED
    assert verdict.last_author == AUTHOR


def test_a_thread_of_only_bot_comments_is_owed() -> None:
    """#1722's four threads are one bot comment each, and nobody has answered them."""
    verdict = mod.classify(_thread([_comment(BOT, bot=True)]), "Vivek0712", HEAD_2577)
    assert verdict.outcome == mod.AWAITING_THE_AUTHOR
    assert verdict.last_author is None


def test_last_non_bot_author_skips_a_run_of_trailing_bots() -> None:
    assert mod.last_non_bot_author([_comment(AUTHOR), _comment(BOT, bot=True), _comment(BOT, bot=True)]) == AUTHOR
    assert mod.last_non_bot_author([]) is None
    assert mod.last_non_bot_author([_comment(BOT, bot=True)]) is None


# --------------------------------------------------------------------------
# What is reported rather than decided on.
# --------------------------------------------------------------------------


def test_an_answer_with_no_commit_after_it_is_still_answered() -> None:
    """An author reply that explains rather than changes is a complete answer.

    Here the head is still the commit the thread was written against, so no commit
    followed the reply. Folding that into the verdict would call the thread
    unanswered and license exactly the duplicate reply this check exists to stop.
    """
    verdict = mod.classify(_thread([_comment(REVIEWER), _comment(AUTHOR)]), AUTHOR, COMMIT_2577)
    assert verdict.outcome == mod.ANSWERED
    assert verdict.head_moved is False


def test_head_moved_is_reported_and_changes_no_verdict() -> None:
    """The same thread under two heads: the column moves, the outcome does not."""
    thread = _thread([_comment(REVIEWER), _comment(AUTHOR)])
    moved = mod.classify(thread, AUTHOR, HEAD_2577)
    unmoved = mod.classify(thread, AUTHOR, COMMIT_2577)
    assert moved.head_moved is True
    assert unmoved.head_moved is False
    assert moved.outcome == unmoved.outcome == mod.ANSWERED


def test_an_unreadable_head_reports_no_movement_rather_than_movement() -> None:
    """``None`` is not a commit that differs from the thread's."""
    verdict = mod.classify(_thread([_comment(REVIEWER)]), AUTHOR, None)
    assert verdict.head_moved is False


def test_the_thread_commit_is_read_from_the_thread_not_from_a_reply() -> None:
    """``originalCommit`` is a property of the thread, measured identical on all four
    comments of #2511 across 27 minutes and two pushes.

    So it is read from the thread's first comment. A payload whose last comment
    carried a different oid must not change the reported thread commit -- if it
    ever did, the field would be describing the reply instead.
    """
    thread = _thread([_comment(REVIEWER, oid=COMMIT_2511), _comment(AUTHOR, oid=HEAD_2511)])
    assert mod.classify(thread, AUTHOR, HEAD_2511).original_commit == COMMIT_2511


# --------------------------------------------------------------------------
# Pull-request level aggregation.
# --------------------------------------------------------------------------


def test_a_pull_request_with_no_threads_owes_nothing() -> None:
    row = mod.evaluate(_pr([]))
    assert row.outcome == mod.NOTHING_OWED
    assert row.is_finding is False
    assert "No review threads" in row.summary


def test_one_owed_thread_among_settled_ones_is_still_a_finding() -> None:
    """A single unanswered thread is not diluted by its resolved neighbours."""
    row = mod.evaluate(
        _pr(
            [
                _thread([_comment(REVIEWER), _comment(AUTHOR)], resolved=True),
                _thread([_comment(REVIEWER)], path="strands_robots/utils.py"),
            ]
        )
    )
    assert row.outcome == mod.AUTHOR_OWES_A_REPLY
    assert row.is_finding is True
    assert [t.path for t in row.owed] == ["strands_robots/utils.py"]
    assert "strands_robots/utils.py" in row.summary


def test_threads_beyond_the_read_ceiling_are_named_rather_than_cleared() -> None:
    """The unread tail is the newest threads, which are the likeliest to be owed.

    Reporting ``nothing-owed`` while silently dropping them would be wrong in the
    one direction that costs work, so the shortfall is stated.
    """
    node = _pr([_thread([_comment(REVIEWER), _comment(AUTHOR)])])
    node["reviewThreads"]["totalCount"] = 103
    row = mod.evaluate(node)
    assert row.unread_threads == 102
    assert "were not read" in mod.render_one("strands-labs/robots", row)


def test_an_absent_thread_total_is_not_read_as_a_shortfall() -> None:
    node = _pr([_thread([_comment(REVIEWER)])])
    del node["reviewThreads"]["totalCount"]
    assert mod.evaluate(node).unread_threads == 0


# --------------------------------------------------------------------------
# Reporting and exit status.
# --------------------------------------------------------------------------


def test_the_remedy_appears_only_when_something_is_owed() -> None:
    owed = mod.evaluate(_pr([_thread([_comment(REVIEWER)])]))
    answered = mod.evaluate(_pr([_thread([_comment(REVIEWER), _comment(AUTHOR)])]))
    assert "### What clears this" in mod.render_one("strands-labs/robots", owed)
    assert "### What clears this" not in mod.render_one("strands-labs/robots", answered)


def test_the_single_report_names_the_outcome_and_both_commits() -> None:
    row = mod.evaluate(_pr([_thread([_comment(REVIEWER), _comment(AUTHOR)])]))
    report = mod.render_one("strands-labs/robots", row)
    assert f"Outcome: **{mod.NOTHING_OWED}**" in report
    assert "`d04a8969`" in report
    assert "`b966ce64`" in report


def test_the_sweep_names_every_pull_request_and_only_findings_in_detail() -> None:
    rows = [
        mod.evaluate(_pr([_thread([_comment(REVIEWER), _comment(AUTHOR)])], number=2577)),
        mod.evaluate(_pr([_thread([_comment(REVIEWER)])], number=2511)),
    ]
    report = mod.render_sweep("strands-labs/robots", rows, [])
    assert "#2577" in report and "#2511" in report
    assert "### #2511" in report
    assert "### #2577" not in report


@pytest.mark.parametrize(
    ("threads", "expected"),
    [
        ([_thread([_comment(REVIEWER), _comment(AUTHOR)])], 0),
        ([_thread([_comment(REVIEWER), _comment(AUTHOR)], resolved=True)], 0),
        ([], 0),
        ([_thread([_comment(REVIEWER)])], 1),
    ],
)
def test_exit_status_is_one_only_when_a_thread_awaits_its_author(
    threads: list[dict], expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "fetch_one", lambda *_a, **_k: _pr(threads))
    assert mod.main(["--repo", "strands-labs/robots", "--pr", "2577", "--token", "t"]) == expected


def test_the_sweep_skips_drafts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A draft is not waiting on its author in the sense this reports."""
    draft = _pr([_thread([_comment(REVIEWER)])], number=9999)
    draft["isDraft"] = True
    monkeypatch.setattr(mod, "fetch_open", lambda *_a, **_k: [draft])
    assert mod.main(["--repo", "strands-labs/robots", "--all-open", "--token", "t"]) == 0


def test_a_lookup_failure_is_not_a_finding(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Never accuse an author of a reply this run could not observe."""

    def boom(*_a: object, **_k: object) -> dict:
        raise ValueError("GraphQL errors: [{'message': 'nope'}]")

    monkeypatch.setattr(mod, "fetch_one", boom)
    assert mod.main(["--repo", "strands-labs/robots", "--pr", "2577", "--token", "t"]) == 0
    assert "could not evaluate" in capsys.readouterr().err


def test_a_failed_listing_is_not_a_finding(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    def boom(*_a: object, **_k: object) -> list[dict]:
        raise ValueError("nope")

    monkeypatch.setattr(mod, "fetch_open", boom)
    assert mod.main(["--repo", "strands-labs/robots", "--all-open", "--token", "t"]) == 0
    assert "could not list open pull requests" in capsys.readouterr().err


def test_one_unreadable_pull_request_does_not_suppress_a_finding_on_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = _pr([_thread([_comment(REVIEWER)])], number=1)
    del broken["number"]
    good = _pr([_thread([_comment(REVIEWER)])], number=2511)
    monkeypatch.setattr(mod, "fetch_open", lambda *_a, **_k: [broken, good])
    assert mod.main(["--repo", "strands-labs/robots", "--all-open", "--token", "t"]) == 1


def test_all_open_and_pr_are_refused_together() -> None:
    """Silently ignoring either flag reads as a successful run of the other thing."""
    with pytest.raises(SystemExit) as excinfo:
        mod.main(["--repo", "strands-labs/robots", "--pr", "2577", "--all-open", "--token", "t"])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("argv", [["--pr", "2577"], ["--all-open"]])
def test_a_missing_token_reports_nothing_rather_than_failing(argv: list[str], capsys: pytest.CaptureFixture) -> None:
    assert mod.main(["--repo", "strands-labs/robots", *argv, "--token", ""]) == 0
    assert "no token" in capsys.readouterr().err


def test_neither_a_pull_request_nor_a_sweep_was_asked_for(capsys: pytest.CaptureFixture) -> None:
    assert mod.main(["--repo", "strands-labs/robots", "--token", "t"]) == 0
    assert "--pr or --all-open is required" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Query shape.
# --------------------------------------------------------------------------


def test_both_query_shapes_carry_the_whole_field_list() -> None:
    """The two documents share one field list, so the sweep cannot drift from the
    single-pull-request path -- and neither may ship an unsubstituted placeholder.
    """
    for document in (mod._ONE_PR, mod._OPEN_PRS):
        assert mod._FIELDS_PLACEHOLDER not in document
        assert "__THREADS__" not in document and "__COMMENTS__" not in document
        for field in ("headRefOid", "isResolved", "isOutdated", "originalCommit", "__typename", "totalCount"):
            assert field in document, field


def test_the_comment_window_is_the_tail_and_the_thread_window_is_the_head() -> None:
    """Only the last non-bot comment decides, so comments are read from the end;
    the newest threads are the likeliest to be owed, so threads are read from the
    start and the shortfall is reported rather than dropped.
    """
    assert f"comments(last: {mod._COMMENTS_PER_THREAD})" in mod._ONE_PR
    assert f"reviewThreads(first: {mod._THREADS_PER_PR})" in mod._ONE_PR


@pytest.mark.parametrize("repo", ["", "owner", "owner/", "/name", "a/b/c"])
def test_a_repository_that_is_not_owner_slash_name_is_refused(repo: str) -> None:
    with pytest.raises(ValueError, match="must be owner/name"):
        mod._split_repo(repo)
