"""Contract pins for the changelog fragment check.

``scripts/check_changelog_fragment.py`` exists because the one rule the changelog
convention rests on had no gate behind it. ``changelog.d/README.md`` and
``AGENTS.md`` both say a behavioural change records itself as a
``changelog.d/<number>-<slug>.md`` fragment and never appends to
``## [Unreleased]``; sixty-seven fragments follow it; and with three entries
inserted directly beneath that heading, ``assemble_changelog.py --check`` exits 0
and the 30 tests in ``tests/test_changelog_format.py`` +
``tests/test_changelog_fragments.py`` all pass. Neither suite can see a fragment
that was never written, because a missing file leaves nothing to inspect.

The checks below pin the four properties that make the script worth having:

- it **refuses** a direct append, replayed as real commits in a real repository
  rather than as a hand-built string pair;
- it is **self-clearing** -- moving the entry into a fragment makes it pass, so
  the remedy it asks for is the remedy that satisfies it, with no override;
- it **does not obstruct a release**: the assembler renders a fragment verbatim
  and deletes what it consumed, so an assemble run's entries are accounted for
  per entry -- and a release that *also* hand-writes one is still refused, for
  only that one;
- it is **quiet** where nothing was added: editing, rewording, or reordering an
  entry already in the log, and the 168 legacy entries on ``main``, are not
  additions.

The git-topology tests are the load-bearing ones. The question is what a branch
*added*, which is a statement about a merge base, and a merge base is exactly the
thing a hand-built fixture would assume rather than exercise.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_changelog_fragment.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_changelog_fragment", _SCRIPT_PATH)
    assert spec and spec.loader, f"cannot load {_SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_changelog_fragment"] = module
    spec.loader.exec_module(module)
    return module


check = _load_module()


# --- git fixtures ---------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def _write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


#: A log shaped like the real one: an ``[Unreleased]`` section that already holds
#: entries -- the 168 on ``main`` are why no static assertion about this section
#: can be written -- followed by a released, dated section.
_LEGACY_LOG = """# CHANGELOG

All notable behavioural changes to `strands-robots` are logged here.

## [Unreleased]

### Fixed: a legacy entry that predates the fragment convention

Body of the legacy entry.

### Fixed: a second legacy entry

Body of the second legacy entry.

## [0.1.0] - 2026-01-01

### Added: the first release

Body.
"""


def _unreleased_body(log_text: str) -> str:
    """The text between ``## [Unreleased]`` and the next level-2 heading."""
    _, _, rest = log_text.partition("## [Unreleased]\n")
    body, _, _ = rest.partition("\n## ")
    return body


def _append_to_unreleased(repo: Path, entry: str) -> None:
    """Append an entry at the top of ``[Unreleased]``, as a direct edit does."""
    path = repo / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("## [Unreleased]\n", f"## [Unreleased]\n\n{entry}", 1), encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository on ``main`` with a populated log and one pending fragment."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "checks@example.invalid")
    _git(root, "config", "user.name", "Changelog Fragment Tests")
    _git(root, "config", "commit.gpgsign", "false")
    _write(root, "CHANGELOG.md", _LEGACY_LOG)
    _write(root, "changelog.d/README.md", "# changelog.d - news fragments\n")
    _write(
        root,
        "changelog.d/1700-a-pending-change.md",
        "### Fixed: a change already recorded as a fragment\n\nBody of the pending fragment.\n",
    )
    _commit(root, "initial commit")
    return root


def _run(repo: Path, head: str = "HEAD") -> int:
    return int(check.main(["--repo", str(repo), "--base-ref", "main", "--head", head]))


#: The entry text a direct append inserts, matching the shape of a real one.
_APPENDED = "### Fixed: an entry written straight into the log\n\nBody of the appended entry.\n"


def _branch(repo: Path, name: str = "pr") -> None:
    _git(repo, "checkout", "-q", "-b", name)


# --- the escape this exists to close --------------------------------------------


def test_a_direct_unreleased_append_is_refused(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The #1783 shape: a behavioural change recorded in the log, with no fragment."""
    _branch(repo)
    _write(repo, "strands_robots/dataset_recorder.py", "# the behavioural change\n")
    _append_to_unreleased(repo, _APPENDED)
    _commit(repo, "fix something, and record it in CHANGELOG.md")

    assert _run(repo) == 1
    output = capsys.readouterr().out
    assert "### Fixed: an entry written straight into the log" in output
    assert "changelog.d/<number>-<slug>.md" in output


def test_three_appended_entries_are_each_named(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """#1783 added three entries in one diff; the report must not name only the first."""
    _branch(repo)
    for index in range(3):
        _append_to_unreleased(repo, f"### Fixed: appended entry number {index}\n\nBody.\n")
    _commit(repo, "record three changes in CHANGELOG.md")

    assert _run(repo) == 1
    output = capsys.readouterr().out
    for index in range(3):
        assert f"### Fixed: appended entry number {index}" in output


def test_moving_the_entry_into_a_fragment_clears_the_check(repo: Path) -> None:
    """The remedy the report asks for is the remedy that satisfies the check."""
    _branch(repo)
    _write(repo, "strands_robots/dataset_recorder.py", "# the behavioural change\n")
    _append_to_unreleased(repo, _APPENDED)
    _commit(repo, "fix something, and record it in CHANGELOG.md")
    assert _run(repo) == 1

    # Exactly what the report says to do: drop it from the log, write a fragment.
    _git(repo, "checkout", "-q", "HEAD~1", "--", "CHANGELOG.md")
    _write(repo, "changelog.d/1783-a-fragment-instead.md", _APPENDED)
    _commit(repo, "record the change as a fragment instead")

    assert _run(repo) == 0


def test_a_fragment_only_branch_passes(repo: Path) -> None:
    """The path every recent merged pull request takes."""
    _branch(repo)
    _write(repo, "strands_robots/dataset_recorder.py", "# the behavioural change\n")
    _write(repo, "changelog.d/1785-the-normal-path.md", _APPENDED)
    _commit(repo, "fix something, recorded as a fragment")

    assert _run(repo) == 0


# --- the release path stays open ------------------------------------------------


def test_an_assemble_run_that_deletes_its_fragment_passes(repo: Path) -> None:
    """``--apply`` folds a fragment into the log and deletes it; that is not an append."""
    _branch(repo, "release")
    folded = (repo / "changelog.d/1700-a-pending-change.md").read_text(encoding="utf-8")
    _append_to_unreleased(repo, folded)
    (repo / "changelog.d/1700-a-pending-change.md").unlink()
    _commit(repo, "assemble the pending fragments into the log")

    assert _run(repo) == 0


def test_an_assemble_run_that_also_hand_writes_an_entry_is_refused_for_only_that_entry(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exemption is per entry, so a release cannot carry an unaccounted one along."""
    _branch(repo, "release")
    folded = (repo / "changelog.d/1700-a-pending-change.md").read_text(encoding="utf-8")
    _append_to_unreleased(repo, folded)
    (repo / "changelog.d/1700-a-pending-change.md").unlink()
    _append_to_unreleased(repo, _APPENDED)
    _commit(repo, "assemble the fragments, and smuggle one entry in by hand")

    assert _run(repo) == 1
    output = capsys.readouterr().out
    unaccounted, _, accounted = output.partition("Accounted for by a fragment")
    assert "### Fixed: an entry written straight into the log" in unaccounted
    assert "### Fixed: a change already recorded as a fragment" not in unaccounted
    assert "### Fixed: a change already recorded as a fragment" in accounted


def test_two_added_entries_need_two_deleted_fragments(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """One consumed fragment cannot license two copies of its entry."""
    _branch(repo, "release")
    folded = (repo / "changelog.d/1700-a-pending-change.md").read_text(encoding="utf-8")
    _append_to_unreleased(repo, folded)
    _append_to_unreleased(repo, folded)
    (repo / "changelog.d/1700-a-pending-change.md").unlink()
    _commit(repo, "fold one fragment in twice")

    assert _run(repo) == 1
    assert "that no fragment accounts for" in capsys.readouterr().out


def test_a_deleted_readme_does_not_account_for_an_added_entry(repo: Path) -> None:
    """``README.md`` is documentation, not a consumed fragment."""
    _branch(repo)
    _append_to_unreleased(repo, _APPENDED)
    (repo / "changelog.d/README.md").unlink()
    _commit(repo, "append an entry and delete the fragment directory's README")

    assert _run(repo) == 1


def test_collapsing_unreleased_into_a_dated_section_passes(repo: Path) -> None:
    """Release bookkeeping moves entries out of ``[Unreleased]``, and out is not in."""
    _branch(repo, "release")
    path = repo / "CHANGELOG.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "## [Unreleased]\n", "## [Unreleased]\n\n## [0.2.0] - 2026-02-01\n", 1
        ),
        encoding="utf-8",
    )
    _commit(repo, "cut 0.2.0")

    assert _run(repo) == 0


def test_an_entry_added_under_a_released_version_is_not_flagged(repo: Path) -> None:
    """Only the ``[Unreleased]`` anchor has the conflict property this protects."""
    _branch(repo, "release")
    path = repo / "CHANGELOG.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "## [0.1.0] - 2026-01-01\n", f"## [0.1.0] - 2026-01-01\n\n{_APPENDED}", 1
        ),
        encoding="utf-8",
    )
    _commit(repo, "correct the 0.1.0 section")

    assert _run(repo) == 0


# --- quiet where nothing was added ----------------------------------------------


def test_an_unchanged_branch_adds_nothing(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The 168 legacy entries appear on both sides of the diff and cancel."""
    _branch(repo)
    _write(repo, "strands_robots/dataset_recorder.py", "# a change with no log entry at all\n")
    _commit(repo, "a branch that does not touch the log")

    assert _run(repo) == 0
    assert "No entry was added" in capsys.readouterr().out


def test_rewording_an_existing_entry_heading_is_reported(repo: Path) -> None:
    """A typo fix in a heading replaces an entry rather than adding one.

    The replaced heading is a *different string*, so this is the one shape where
    the multiset difference reports an addition on an edit. It is allowed
    deliberately: the guard is against a branch introducing an entry, and a
    reworded heading is one the log already carried -- pinned so a future
    tightening to "any heading not present at the base" is a conscious choice
    rather than an accident.
    """
    _branch(repo)
    path = repo / "CHANGELOG.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "### Fixed: a legacy entry that predates the fragment convention",
            "### Fixed: a legacy entry that predates the fragment convention (typo fixed)",
        ),
        encoding="utf-8",
    )
    _commit(repo, "fix a typo in an existing entry heading")

    # Reported, because the string differs; the point of the pin is that this is
    # the *only* edit shape that is, and that it is visibly a rewording.
    assert _run(repo) == 1


def test_editing_an_entry_body_is_not_an_addition(repo: Path) -> None:
    """Prose repair beneath an existing heading changes no heading at all."""
    _branch(repo)
    path = repo / "CHANGELOG.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("Body of the legacy entry.", "Body of the legacy entry, clarified."),
        encoding="utf-8",
    )
    _commit(repo, "clarify the body of an existing entry")

    assert _run(repo) == 0


def test_reordering_existing_entries_is_not_an_addition(repo: Path) -> None:
    """A multiset comparison is order-insensitive, so a reshuffle is silent."""
    _branch(repo)
    path = repo / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    body = _unreleased_body(text)
    first, _, second = body.partition("### Fixed: a second legacy entry")
    path.write_text(
        text.replace(body, "\n### Fixed: a second legacy entry" + second + first.rstrip("\n") + "\n"),
        encoding="utf-8",
    )
    _commit(repo, "reorder the unreleased entries")

    assert _run(repo) == 0


def test_deleting_an_existing_entry_is_not_an_addition(repo: Path) -> None:
    """Removal is the opposite direction and must not be reported as an append."""
    _branch(repo)
    path = repo / "CHANGELOG.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "### Fixed: a second legacy entry\n\nBody of the second legacy entry.\n\n", ""
        ),
        encoding="utf-8",
    )
    _commit(repo, "retire an unreleased entry")

    assert _run(repo) == 0


def test_a_duplicate_of_an_existing_heading_is_flagged(repo: Path) -> None:
    """Why the comparison is a multiset: ``[Unreleased]`` really does repeat headings.

    Two entries on ``main`` are both a bare ``### Fixed:``, so a set difference
    would let a branch add a third copy of a heading already present.
    """
    _branch(repo)
    _append_to_unreleased(repo, "### Fixed: a second legacy entry\n\nA third copy of an existing heading.\n")
    _commit(repo, "append an entry whose heading is already in the log")

    assert _run(repo) == 1


# --- git topology ---------------------------------------------------------------


def test_an_entry_landing_on_the_base_is_not_attributed_to_the_branch(repo: Path) -> None:
    """An entry added on ``main`` after the branch diverged is not the branch's."""
    _branch(repo)
    _write(repo, "strands_robots/dataset_recorder.py", "# an unrelated change\n")
    _commit(repo, "the branch's own commit")

    _git(repo, "checkout", "-q", "main")
    _append_to_unreleased(repo, _APPENDED)
    _commit(repo, "someone else appends to the log on main")
    _git(repo, "checkout", "-q", "pr")

    assert _run(repo) == 0


def test_a_merge_commit_head_still_sees_the_branchs_append(repo: Path) -> None:
    """Measured, not assumed: the ``refs/pull/<n>/merge`` head does not hide it.

    The sibling merge-base overlap check is defeated by that commit, because it
    compares two path sets either side of the merge base and the merge commit
    drives the merge base to the base tip, emptying the base side. This check
    asks a different question -- what entries does the head carry that the base
    does not -- and the branch's appended entry is on the head and not on the
    base whichever of the two commits is used. Pinned so the workflow's use of
    the head SHA is understood as consistency with its sibling and as reviewing
    the tree under review, not as load-bearing for correctness here.
    """
    _branch(repo)
    _append_to_unreleased(repo, _APPENDED)
    _commit(repo, "append an entry")

    _git(repo, "checkout", "-q", "main")
    _write(repo, "strands_robots/robot.py", "# main moves on\n")
    _commit(repo, "an unrelated commit on main")
    _git(repo, "checkout", "-q", "pr")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge main into pr", "main")

    assert _run(repo, head="HEAD") == 1


def test_a_branch_that_introduces_the_changelog_is_handled(tmp_path: Path) -> None:
    """An absent base-side log yields an empty base-side set, not a crash."""
    root = tmp_path / "fresh"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "checks@example.invalid")
    _git(root, "config", "user.name", "Changelog Fragment Tests")
    _git(root, "config", "commit.gpgsign", "false")
    _write(root, "README.md", "# fresh\n")
    _commit(root, "initial commit")

    _branch(root)
    _write(root, "CHANGELOG.md", _LEGACY_LOG)
    _commit(root, "introduce the changelog")

    # Every entry in the new log is an addition, and none is accounted for.
    assert _run(root) == 1


def test_an_unresolvable_base_ref_fails_loudly(repo: Path) -> None:
    """A check that cannot compute its answer must not report the reassuring one."""
    assert int(check.main(["--repo", str(repo), "--base-ref", "no-such-branch", "--head", "HEAD"])) == 1


def test_the_remote_tracking_ref_wins_over_a_local_branch(repo: Path, tmp_path: Path) -> None:
    """CI has ``origin/main`` and no local ``main``; the resolution order matches."""
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True)
    assert check.resolve_base_ref("main", repo=clone) == "origin/main"


# --- unit pins on the pieces ----------------------------------------------------


def test_unreleased_entries_stops_at_the_next_version_heading() -> None:
    assert check.unreleased_entries(_LEGACY_LOG) == (
        "### Fixed: a legacy entry that predates the fragment convention",
        "### Fixed: a second legacy entry",
    )


def test_unreleased_entries_is_empty_without_the_anchor() -> None:
    assert check.unreleased_entries("# CHANGELOG\n\n## [0.1.0] - 2026-01-01\n\n### Added: a thing\n") == ()


def test_unreleased_entries_ignores_deeper_headings() -> None:
    log = "## [Unreleased]\n\n### Fixed: an entry\n\n#### A sub-heading in its body\n"
    assert check.unreleased_entries(log) == ("### Fixed: an entry",)


def test_fragment_entry_reads_the_first_non_blank_line() -> None:
    assert check.fragment_entry("\n\n### Fixed: a thing\n\nBody.\n") == "### Fixed: a thing"


def test_fragment_entry_is_none_when_the_first_line_is_not_an_entry() -> None:
    assert check.fragment_entry("Some prose with no heading.\n") is None
    assert check.fragment_entry("") is None


def test_fragment_entry_and_log_entries_agree_on_trailing_whitespace() -> None:
    """Both sides ``rstrip``, so a trailing space cannot fake an unaccounted entry."""
    assert (
        check.fragment_entry("### Fixed: a thing   \n")
        == check.unreleased_entries("## [Unreleased]\n\n### Fixed: a thing\n")[0]
    )


def test_added_entries_is_a_multiset_difference() -> None:
    assert check.added_entries(["### a"], ["### a", "### a", "### b"]) == ("### a", "### b")


def test_added_entries_is_empty_when_nothing_was_added() -> None:
    assert check.added_entries(["### a", "### b"], ["### b", "### a"]) == ()


def test_unaccounted_entries_is_a_multiset_difference() -> None:
    assert check.unaccounted_entries(["### a", "### a"], ["### a"]) == ("### a",)
    assert check.unaccounted_entries(["### a"], ["### a", "### b"]) == ()


def test_the_report_names_the_entry_and_the_remedy() -> None:
    report = check.render_report(
        base_ref="main",
        merge_base_sha="0123456789abcdef",
        added=("### Fixed: an entry",),
        accounted=(),
        unaccounted=("### Fixed: an entry",),
    )
    assert "### Fixed: an entry" in report
    assert "changelog.d/<number>-<slug>.md" in report
    assert "assemble_changelog.py --apply" in report
    assert "re-approval round" in report


def test_a_report_paragraph_written_as_several_literals_stays_one_line() -> None:
    """Each paragraph is one report line, so a sentence is never split mid-clause.

    ``render_report`` builds the report as a list of lines and joins it with
    newlines, so one element is one line. Three of its paragraphs are written as
    several adjacent literals for source width, and each is a single element --
    the explicit ``+`` between the parts says so rather than leaving it to be
    read as a missing comma.

    Adding the commas instead splits those paragraphs across lines, breaking a
    sentence in the middle of a clause. Every substring assertion above still
    passes when that happens, since each fragment survives on a line of its own,
    which is why the shape is pinned here rather than left to the wording checks.
    """
    report = check.render_report(
        base_ref="main",
        merge_base_sha="0123456789abcdef",
        added=("### Fixed: an entry",),
        accounted=(),
        unaccounted=("### Fixed: an entry",),
    )
    lines = report.splitlines()
    paragraphs = (
        ("Every branch appends at the same anchor", "so there is nothing to reconcile."),
        ("Record each one as", "/README.md`."),
        ("is assembled from the accumulated fragments", "--apply`."),
    )
    for opening, ending in paragraphs:
        holding = [line for line in lines if opening in line]
        assert len(holding) == 1, f"{opening!r} appears on {len(holding)} lines, expected 1"
        assert holding[0].endswith(ending), f"paragraph starting {opening!r} does not end with {ending!r}"


def test_the_report_says_so_when_nothing_was_added() -> None:
    report = check.render_report(
        base_ref="main",
        merge_base_sha="0123456789abcdef",
        added=(),
        accounted=(),
        unaccounted=(),
    )
    assert "No entry was added" in report
    assert "changelog.d/<number>-<slug>.md" not in report


def test_reserved_names_match_the_assembler() -> None:
    """A name the assembler skips must not be counted as a consumed fragment."""
    assembler = (_REPO_ROOT / "scripts" / "assemble_changelog.py").read_text(encoding="utf-8")
    for name in check.RESERVED_NAMES:
        assert f'"{name}"' in assembler, f"{name} is not reserved by the assembler"


def test_no_emoji_in_the_script_or_its_output() -> None:
    """Project rule: agent-read strings are plain ASCII, including U+FE0F."""
    for path in (_SCRIPT_PATH,):
        text = path.read_text(encoding="utf-8")
        offenders = [(index, char) for index, char in enumerate(text) if ord(char) > 0x7F]
        assert not offenders, f"{path.name} holds non-ASCII characters: {offenders[:5]}"


# --- the tree the gate is read from ---------------------------------------------


def test_the_verdict_does_not_depend_on_the_checked_out_tree(repo: Path) -> None:
    """Same commits, same verdict, whichever tree happens to be checked out.

    This is the property that lets CI run the script from the *base* checkout while
    judging the branch, and CI must: a branch that forked before this check landed
    carries no copy of the script, so running it out of the head tree exits 2 before
    the check begins -- a red X indistinguishable from exit 1, which the script
    reserves for a real unaccounted entry. Measured on #1786, whose head forked one
    commit before the gate it failed; see issue #1791.

    It holds because nothing here reads the working tree: ``file_at`` is
    ``git show <rev>:<path>`` and ``deleted_fragments`` is ``git diff <a>..<b>``, so
    only reachability matters and the checkout is irrelevant to the answer.
    """
    _branch(repo)
    _append_to_unreleased(repo, _APPENDED)
    head = _commit(repo, "append straight to the log")

    # The branch checked out, as the workflow used to do.
    assert _run(repo, head) == 1

    # The base checked out, as the workflow now does. The appended entry is not in
    # the tree on disk at all, and it is still found and still refused.
    _git(repo, "checkout", "-q", "main")
    appended_heading = _APPENDED.splitlines()[0]
    assert appended_heading not in (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert _run(repo, head) == 1


def test_a_recorded_change_still_passes_from_the_base_checkout(repo: Path) -> None:
    """The other direction: reading from the base does not fail every branch.

    Without this, the test above is satisfied by a check that refuses everything.
    """
    _branch(repo)
    _write(repo, "changelog.d/1791-a-recorded-change.md", _APPENDED)
    head = _commit(repo, "record the change as a fragment")

    _git(repo, "checkout", "-q", "main")
    assert _run(repo, head) == 0


def test_the_workflow_names_the_head_rather_than_checking_it_out() -> None:
    """The step must grade the pull request head without checking it out.

    The gate used to be its own workflow, checked out from the *base* so a head
    that predates the script could not exit 2 (#1791). It now runs as a guard
    as a step inside the required check (test-lint.yml), whose checkout is the pull
    request's merge commit -- a tree that carries every script on the base tip by
    construction, so the same property holds without a base checkout. What has to
    stay explicit is the commit under test: the head sha is named with ``--head``
    and never checked out, because a merge-commit head sees the branch's append
    but reports it against the wrong tree.
    """
    workflow = (_REPO_ROOT / ".github" / "workflows" / "test-lint.yml").read_text(encoding="utf-8")
    assert "HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in workflow
    assert "ref: ${{ github.event.pull_request.head.sha }}" not in workflow, (
        "checking the head out is what made the script's presence a precondition"
    )
    assert 'scripts/check_changelog_fragment.py --base-ref "${BASE_REF}" --head "${HEAD_SHA}"' in workflow, (
        "the commit under test is named, not checked out"
    )


# --- the fragment the branch actually wrote -------------------------------------
#
# The half above asks whether an entry bypassed the fragment convention. It cannot
# ask whether the fragment that *was* written is valid: a branch that records its
# change correctly adds no heading to the log, so the added-entry set is empty and
# the check is silent. Fragment validity was enforced only by
# tests/test_changelog_fragments.py, inside the one required check, and on #2144 --
# whose sole defect was a fragment first line reading `Fixed: ...` -- this job
# reported SUCCESS in 10s while `call-test-lint / Test and Lint` reported FAILURE
# 20.8 minutes later, worded as a test failure. See issue #2163.
#
# The two boundaries below are the ones worth guarding, because getting either
# wrong costs more than the gap did: judging the whole directory would fail a
# branch for a fragment already on the base (#1879), and reading the working tree
# would break the property that lets CI judge the branch from a base checkout
# (#1791).


#: A fragment body in the shape the assembler accepts.
_VALID_FRAGMENT = "### Fixed: a change recorded the normal way\n\nBody of the fragment.\n"

#: The #2144 shape: the heading text is there, the `### ` that makes it a heading
#: is not. This is what a first-time fragment author writes.
_MALFORMED_FRAGMENT = "Fixed: a change recorded with no entry heading\n\nBody of the fragment.\n"


def _load_assembler():  # type: ignore[no-untyped-def]
    """Load ``assemble_changelog`` the way its own suite does."""
    path = _REPO_ROOT / "scripts" / "assemble_changelog.py"
    spec = importlib.util.spec_from_file_location("assemble_changelog", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["assemble_changelog"] = module
    spec.loader.exec_module(module)
    return module


def test_a_malformed_fragment_is_refused(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The #2144 shape, which reached the required suite and nothing faster."""
    _branch(repo)
    _write(repo, "strands_robots/utils.py", "# the behavioural change\n")
    _write(repo, "changelog.d/2144-a-malformed-fragment.md", _MALFORMED_FRAGMENT)
    _commit(repo, "fix something, and record it in a fragment with no heading")

    assert _run(repo) == 1
    output = capsys.readouterr().out
    assert "changelog.d/2144-a-malformed-fragment.md" in output
    assert "first line must be a level-3 entry heading" in output


def test_adding_the_missing_heading_prefix_clears_the_check(repo: Path) -> None:
    """Self-clearing, like the append half: the remedy asked for is the one that satisfies it."""
    _branch(repo)
    _write(repo, "changelog.d/2144-a-malformed-fragment.md", _MALFORMED_FRAGMENT)
    _commit(repo, "record the change in a fragment with no heading")
    assert _run(repo) == 1

    _write(repo, "changelog.d/2144-a-malformed-fragment.md", f"### {_MALFORMED_FRAGMENT}")
    _commit(repo, "make the first line an entry heading")

    assert _run(repo) == 0


def test_a_misnamed_fragment_is_refused(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The other half of the class, reached by a different path.

    A bad *name* makes ``collect_fragments`` raise, so ``validate_fragments``
    returns that one message and never inspects any body. ``changelog.d/README.md``
    calls it out specifically: a stray or misnamed file is a hard error, so an entry
    can never be silently dropped.
    """
    _branch(repo)
    _write(repo, "changelog.d/Bad_Name.md", _VALID_FRAGMENT)
    _commit(repo, "record the change in a badly named file")

    assert _run(repo) == 1
    assert "not a valid fragment name" in capsys.readouterr().out


def test_renaming_the_fragment_clears_the_check(repo: Path) -> None:
    """The name half is self-clearing too."""
    _branch(repo)
    _write(repo, "changelog.d/Bad_Name.md", _VALID_FRAGMENT)
    _commit(repo, "record the change in a badly named file")
    assert _run(repo) == 1

    _git(repo, "mv", "changelog.d/Bad_Name.md", "changelog.d/2163-a-good-name.md")
    _commit(repo, "give the fragment a valid name")

    assert _run(repo) == 0


def test_an_uppercase_slug_is_refused(repo: Path) -> None:
    """The naming rule is the assembler's regex, not a substring check on the extension."""
    _branch(repo)
    _write(repo, "changelog.d/2163-Bad-Slug.md", _VALID_FRAGMENT)
    _commit(repo, "record the change with an uppercase slug")

    assert _run(repo) == 1


# --- only the branch's own fragments --------------------------------------------


def test_a_malformed_fragment_already_on_the_base_is_not_attributed_to_the_branch(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The #1879 boundary, and the reason ``validate_fragments()`` is not wired in whole.

    That function walks the entire directory -- 315 pending fragments on ``main`` --
    so a branch that touched nothing under ``changelog.d/`` would be failed for
    somebody else's fragment, and the report would name a file absent from its diff.
    Accusing a branch of a pre-existing defect is what #1879 spent a review round on.
    """
    _write(repo, "changelog.d/1699-already-broken.md", _MALFORMED_FRAGMENT)
    _commit(repo, "a malformed fragment that is already on the base")

    _branch(repo)
    _write(repo, "strands_robots/utils.py", "# an unrelated behavioural change\n")
    _write(repo, "changelog.d/2163-a-good-fragment.md", _VALID_FRAGMENT)
    _commit(repo, "an unrelated change, recorded correctly")

    assert _run(repo) == 0
    assert "1699-already-broken.md" not in capsys.readouterr().out


def test_a_branch_that_breaks_an_existing_fragment_is_answerable_for_it(repo: Path) -> None:
    """``--diff-filter=AM``: modifying a fragment makes it the branch's, not only adding one."""
    _branch(repo)
    _write(repo, "changelog.d/1700-a-pending-change.md", _MALFORMED_FRAGMENT)
    _commit(repo, "strip the heading off a fragment that was already valid")

    assert _run(repo) == 1


def test_deleting_a_malformed_fragment_is_not_refused(repo: Path) -> None:
    """A deletion is not the branch adopting it, and the release path deletes fragments."""
    _write(repo, "changelog.d/1699-already-broken.md", _MALFORMED_FRAGMENT)
    _commit(repo, "a malformed fragment on the base")

    _branch(repo, "release")
    (repo / "changelog.d/1699-already-broken.md").unlink()
    _commit(repo, "drop the malformed fragment")

    assert _run(repo) == 0


def test_the_readme_is_not_judged_as_a_fragment(repo: Path) -> None:
    """Reserved names are documentation. The skip mirrors ``collect_fragments``."""
    _branch(repo)
    _write(repo, "changelog.d/README.md", "# changelog.d - news fragments\n\nRevised prose.\n")
    _commit(repo, "revise the fragment directory's README")

    assert _run(repo) == 0


def test_a_dotfile_is_not_judged_as_a_fragment(repo: Path) -> None:
    """``collect_fragments`` skips dotfiles, so a `.gitkeep` is not a malformed entry."""
    _branch(repo)
    _write(repo, "changelog.d/.gitkeep", "")
    _commit(repo, "keep the directory")

    assert _run(repo) == 0


# --- every rule the assembler has, not only the first line ----------------------


def test_the_gate_refuses_exactly_what_the_assembler_refuses(repo: Path, tmp_path: Path) -> None:
    """One rule, one owner.

    The gate imports ``FRAGMENT_NAME`` and ``validate_fragment`` instead of
    restating either, because a second copy could refuse a fragment the assembler
    accepts -- the contradiction #2139 had to remove for ``save_episode``. This
    compares the two verdicts over every rule the assembler has, so a future
    narrowing to "check the first line" fails here rather than shipping a gate that
    silently stops asking three of its four questions.
    """
    assembler = _load_assembler()
    cases = (
        ("2200-a-valid-fragment.md", _VALID_FRAGMENT),
        ("2200-no-entry-heading.md", _MALFORMED_FRAGMENT),
        ("2200-an-empty-fragment.md", "\n\n"),
        ("2200-two-entries.md", "### Fixed: one\n\nBody.\n\n### Added: two\n\nBody.\n"),
        ("2200-a-forged-version.md", "### Fixed: one\n\n## [9.9.9] - 2026-01-01\n"),
        ("Bad_Name.md", _VALID_FRAGMENT),
        ("2200-Bad-Slug.md", _VALID_FRAGMENT),
    )

    for index, (name, body) in enumerate(cases):
        directory = tmp_path / f"case{index}"
        directory.mkdir()
        (directory / name).write_text(body, encoding="utf-8")
        assembler_refuses = bool(assembler.validate_fragments(directory))

        branch = f"case{index}"
        _git(repo, "checkout", "-q", "main")
        _git(repo, "checkout", "-q", "-b", branch)
        _write(repo, f"changelog.d/{name}", body)
        _commit(repo, f"add {name}")
        gate_refuses = _run(repo) == 1

        assert gate_refuses == assembler_refuses, (
            f"{name}: gate refuses={gate_refuses}, assembler refuses={assembler_refuses}"
        )


# --- what the reader is told ----------------------------------------------------


def test_the_annotation_reproduces_the_assemblers_message(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The annotation is the assembler's wording byte for byte, so the local command matches.

    ``python scripts/assemble_changelog.py --check`` is what a contributor runs to
    reproduce this, and a paraphrase here would leave them searching its output for
    a sentence it does not print. The report line drops only the leading file name,
    which it prints itself.
    """
    assembler = _load_assembler()
    _branch(repo)
    _write(repo, "changelog.d/2144-a-malformed-fragment.md", _MALFORMED_FRAGMENT)
    _commit(repo, "record the change in a fragment with no heading")

    assert _run(repo) == 1
    output = capsys.readouterr().out

    directory = repo / "changelog.d"
    expected = assembler.validate_fragments(directory)
    assert len(expected) == 1, f"expected one assembler verdict, got {expected}"
    assert f"::error file=changelog.d/2144-a-malformed-fragment.md,line=1::{expected[0]}" in output
    assert "- `changelog.d/2144-a-malformed-fragment.md`: 2144-" not in output, (
        "the report line must not print the fragment name twice"
    )


def test_the_report_names_the_command_that_reproduces_it(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _branch(repo)
    _write(repo, "changelog.d/2144-a-malformed-fragment.md", _MALFORMED_FRAGMENT)
    _commit(repo, "record the change in a fragment with no heading")

    assert _run(repo) == 1
    output = capsys.readouterr().out
    assert "assemble_changelog.py --check" in output
    assert "hard error rather than a skip" in output


def test_both_halves_are_reported_in_one_run(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A branch can break the convention twice, and one run says so once.

    The report used to return early when no entry had been added, which is exactly
    the state the #2144 shape is in.
    """
    _branch(repo)
    _append_to_unreleased(repo, _APPENDED)
    _write(repo, "changelog.d/2144-a-malformed-fragment.md", _MALFORMED_FRAGMENT)
    _commit(repo, "append to the log and add a malformed fragment")

    assert _run(repo) == 1
    output = capsys.readouterr().out
    assert "that no fragment accounts for" in output
    assert "the assembler would refuse" in output


def test_a_valid_fragment_reports_nothing_about_fragments(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Quiet on the path every merged pull request takes."""
    _branch(repo)
    _write(repo, "changelog.d/2163-the-normal-path.md", _VALID_FRAGMENT)
    _commit(repo, "record the change as a valid fragment")

    assert _run(repo) == 0
    output = capsys.readouterr().out
    assert "the assembler would refuse" not in output
    assert "No entry was added" in output


# --- the tree the rules are read from -------------------------------------------


def test_the_fragment_verdict_does_not_depend_on_the_checked_out_tree(repo: Path) -> None:
    """Same property as the append half, and CI depends on it for the same reason.

    The workflow checks out the *base* branch and names the head with ``--head``, so
    the branch's fragment bodies must come from the object database. They do:
    ``fragment_problems`` reads each one with ``git show <head>:<path>``.
    """
    _branch(repo)
    _write(repo, "changelog.d/2144-a-malformed-fragment.md", _MALFORMED_FRAGMENT)
    head = _commit(repo, "record the change in a fragment with no heading")

    _git(repo, "checkout", "-q", "main")
    assert not (repo / "changelog.d/2144-a-malformed-fragment.md").exists()

    assert _run(repo, head) == 1


def test_the_validator_survives_a_first_load_of_its_own(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate must work when it is the thing that first imports the assembler.

    ``@dataclass`` resolves its own module through ``sys.modules`` to decide whether
    an annotation is a ``KW_ONLY`` sentinel, so a module executed by path but never
    registered there dies on its first dataclass with ``AttributeError: 'NoneType'
    object has no attribute '__dict__'`` -- naming neither the loader nor the
    reason. Nothing else here would catch it: every other test in this file runs
    after some importer has already put ``assemble_changelog`` in the table.
    """
    monkeypatch.delitem(sys.modules, "assemble_changelog", raising=False)

    spec = importlib.util.spec_from_file_location("check_changelog_fragment_fresh", _SCRIPT_PATH)
    assert spec and spec.loader
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "check_changelog_fragment_fresh", fresh)
    spec.loader.exec_module(fresh)

    _branch(repo)
    _write(repo, "changelog.d/2144-a-malformed-fragment.md", _MALFORMED_FRAGMENT)
    head = _commit(repo, "record the change in a fragment with no heading")

    problems = fresh.fragment_problems(fresh.changed_fragments("main", head, repo=repo), head, repo=repo)
    assert len(problems) == 1
    assert problems[0][0] == "changelog.d/2144-a-malformed-fragment.md"


# --- the readers, directly ------------------------------------------------------


def test_changed_fragments_reports_adds_and_modifications_only(repo: Path) -> None:
    _write(repo, "changelog.d/1699-to-be-deleted.md", _VALID_FRAGMENT)
    base = _commit(repo, "a fragment on the base")

    _branch(repo)
    _write(repo, "changelog.d/2163-added.md", _VALID_FRAGMENT)
    _write(repo, "changelog.d/1700-a-pending-change.md", _VALID_FRAGMENT.replace("Body", "Revised body"))
    (repo / "changelog.d/1699-to-be-deleted.md").unlink()
    head = _commit(repo, "add one, modify one, delete one")

    assert check.changed_fragments(base, head, repo=repo) == (
        "changelog.d/1700-a-pending-change.md",
        "changelog.d/2163-added.md",
    )


def test_fragment_problems_is_empty_for_a_valid_fragment(repo: Path) -> None:
    _branch(repo)
    _write(repo, "changelog.d/2163-valid.md", _VALID_FRAGMENT)
    head = _commit(repo, "add a valid fragment")

    assert check.fragment_problems(("changelog.d/2163-valid.md",), head, repo=repo) == ()


def test_a_placeholder_numbered_fragment_is_refused_on_the_pull_request(repo: Path) -> None:
    """The job named for the convention must refuse the placeholder name.

    A ``0000-`` fragment is valid in every other respect, so while the rule lived
    only in ``tests/test_changelog_fragments.py`` this job reported SUCCESS and
    the branch learned about the rename from the required suite twenty minutes
    later - and a branch that merged in between put an untraceable entry on the
    log. The verdict is the assembler's, so asking for the rename here costs no
    second copy of the rule.
    """
    _branch(repo)
    _write(repo, "changelog.d/0000-a-placeholder.md", _VALID_FRAGMENT)
    head = _commit(repo, "add a placeholder-numbered fragment")

    problems = check.fragment_problems(("changelog.d/0000-a-placeholder.md",), head, repo=repo)

    assert [path for path, _ in problems] == ["changelog.d/0000-a-placeholder.md"]
    assert "placeholder" in problems[0][1], problems
    assert _run(repo) == 1, "the job must exit non-zero on a fragment the assembler would refuse"


def test_fragment_problems_names_the_path_it_was_given(repo: Path) -> None:
    """The path in the tuple is the one the annotation needs, not the basename."""
    _branch(repo)
    _write(repo, "changelog.d/2163-malformed.md", _MALFORMED_FRAGMENT)
    head = _commit(repo, "add a malformed fragment")

    problems = check.fragment_problems(("changelog.d/2163-malformed.md",), head, repo=repo)
    assert [path for path, _ in problems] == ["changelog.d/2163-malformed.md"]


def test_the_report_names_a_fragment_problem() -> None:
    report = check.render_report(
        base_ref="main",
        merge_base_sha="0123456789abcdef",
        added=(),
        accounted=(),
        unaccounted=(),
        problems=(("changelog.d/2163-a-fragment.md", "2163-a-fragment.md: fragment is empty"),),
    )
    assert "`changelog.d/2163-a-fragment.md`: fragment is empty" in report
    assert "the assembler would refuse" in report
