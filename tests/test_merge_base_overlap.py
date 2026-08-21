"""Contract pins for the merge-base overlap check.

``scripts/check_merge_base_overlap.py`` exists because four green signals are not
enough to merge safely. #1766 and #1763 both edited
``strands_robots/simulation/mujoco/scene_ops.py``; #1766 landed first; #1763 then
read ``APPROVED`` / ``SUCCESS`` / ``MERGEABLE`` / ``CLEAN`` and its squash still
broke ``main``, because it carried a premise test asserting the defect #1766 had
just fixed. Every one of those signals is computed against the base the branch
was tested on, so none of them can see it.

The checks below pin the three properties that make the script worth having:

- it **flags** the #1763/#1766 topology, replayed here as real commits in a real
  repository rather than as a hand-built path set;
- it is **self-clearing** -- merging the base branch makes it pass, so the
  remedy it asks for is the remedy that satisfies it, with no override to add;
- it is **quiet** where an overlap cannot matter: disjoint edits and prose-only
  overlaps do not block.

The git-topology tests are the load-bearing ones. The overlap is a statement
about merge bases, and a merge base is exactly the thing a hand-built fixture
would assume rather than exercise -- including the one way to get this wrong in
CI, pinned by ``test_a_merge_commit_head_defeats_the_check``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_merge_base_overlap.py"

#: The file both #1766 and #1763 edited. Used verbatim in the replay so the
#: pinned scenario is recognisable as the incident it came from.
_SHARED = "strands_robots/simulation/mujoco/scene_ops.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_merge_base_overlap", _SCRIPT_PATH)
    assert spec and spec.loader, f"cannot load {_SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_merge_base_overlap"] = module
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


#: A file with enough distinct lines that two branches can edit it in regions far
#: enough apart for git to merge them without a conflict -- which is the whole
#: point of the incident: the text merged cleanly and the meaning did not.
_SHARED_BODY = "\n".join(f"line {index}" for index in range(1, 41)) + "\n"


def _edit_line(repo: Path, relative: str, line_number: int, text: str) -> None:
    path = repo / relative
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[line_number - 1] = text
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository on ``main`` with one commit, ready to branch from."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "checks@example.invalid")
    _git(root, "config", "user.name", "Merge Base Overlap Tests")
    _git(root, "config", "commit.gpgsign", "false")
    _write(root, _SHARED, _SHARED_BODY)
    _write(root, "docs/simulation/world-building.md", "# World building\n\nOriginal prose.\n")
    _commit(root, "initial commit")
    return root


def _branch_editing_shared(repo: Path) -> None:
    """Branch off ``main`` and edit the shared file, as #1763 did."""
    _git(repo, "checkout", "-q", "-b", "pr")
    _edit_line(repo, _SHARED, 40, "line 40  # edited by the pull request")
    _write(repo, "tests/simulation/mujoco/test_add_robot_preserves_scene_state.py", "# premise test\n")
    _commit(repo, "the pull request's commit")


def _land_on_main_editing_shared(repo: Path) -> None:
    """Land a commit on ``main`` touching the shared file, as #1766 did."""
    _git(repo, "checkout", "-q", "main")
    _edit_line(repo, _SHARED, 1, "line 1  # edited by the commit that landed first")
    _commit(repo, "the commit that landed on main first")
    _git(repo, "checkout", "-q", "pr")


def _run_at(repo: Path, head: str) -> int:
    """Check a named commit, which is what CI does: the head SHA, never ``HEAD``."""
    return int(check.main(["--repo", str(repo), "--base-ref", "main", "--head", head]))


def _run(repo: Path) -> int:
    return _run_at(repo, "HEAD")


# --- the incident, replayed -----------------------------------------------------


def test_the_1763_1766_pair_is_flagged(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The acceptance criterion: replayed, the pair that broke main is caught.

    Both branches edit the shared file in regions git merges without complaint,
    so this is not a conflict the existing gate would have stopped.
    """
    _branch_editing_shared(repo)
    _land_on_main_editing_shared(repo)

    assert _run(repo) == 1, "an untested combination must not report success"
    out = capsys.readouterr().out
    assert _SHARED in out, "the report must name the overlapping path"
    assert "merge" in out and "main" in out, "the report must state the remedy"


def test_the_pair_merges_without_a_text_conflict(repo: Path) -> None:
    """Non-vacuity for the test above: git itself has nothing to report here.

    If the two edits conflicted, the existing merge gate would already block the
    merge and this check would be redundant. The incident happened precisely
    because they did not.
    """
    _branch_editing_shared(repo)
    _land_on_main_editing_shared(repo)

    _git(repo, "merge", "--no-edit", "-q", "main")  # raises CalledProcessError on conflict
    assert "line 1  # edited by the commit that landed first" in (repo / _SHARED).read_text(encoding="utf-8")
    assert "line 40  # edited by the pull request" in (repo / _SHARED).read_text(encoding="utf-8")


def test_merging_the_base_clears_the_check(repo: Path) -> None:
    """Self-clearing: the remedy the report asks for is what makes it pass.

    This is why the check needs no bypass label. Merging the base advances the
    merge base to the base tip, which empties the base-side path set, and the
    re-run then happens against a base containing the landed commits.
    """
    _branch_editing_shared(repo)
    _land_on_main_editing_shared(repo)
    assert _run(repo) == 1

    _git(repo, "merge", "--no-edit", "-q", "main")

    assert _run(repo) == 0, "after merging the base there is nothing untested left to flag"


# --- where an overlap cannot matter ---------------------------------------------


def test_an_unmoved_base_does_not_overlap(repo: Path) -> None:
    _branch_editing_shared(repo)
    assert _run(repo) == 0


def test_disjoint_edits_do_not_overlap(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A base that moved under *other* files invalidates nothing."""
    _branch_editing_shared(repo)
    _git(repo, "checkout", "-q", "main")
    _write(repo, "strands_robots/policies/mock.py", "# unrelated landing\n")
    _commit(repo, "an unrelated commit on main")
    _git(repo, "checkout", "-q", "pr")

    assert _run(repo) == 0
    assert "No overlap" in capsys.readouterr().out


def test_a_prose_only_overlap_is_reported_but_does_not_block(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Documentation cannot change what the suite does, so it does not gate."""
    doc = "docs/simulation/world-building.md"
    _git(repo, "checkout", "-q", "-b", "pr")
    _write(repo, doc, "# World building\n\nOriginal prose.\n\nAdded by the pull request.\n")
    _commit(repo, "docs from the pull request")

    _git(repo, "checkout", "-q", "main")
    _write(repo, doc, "# World building\n\nRewritten on main.\n")
    _commit(repo, "docs on main")
    _git(repo, "checkout", "-q", "pr")

    assert _run(repo) == 0, "a prose overlap must not demand a full re-run"
    out = capsys.readouterr().out
    assert doc in out, "it is still reported - the reader may want to know"
    assert "not blocking" in out


# --- the way to get this wrong in CI -------------------------------------------


def test_a_merge_commit_head_defeats_the_check(repo: Path) -> None:
    """Pins why the workflow must check out the head SHA, not the merge commit.

    ``actions/checkout`` defaults to ``refs/pull/<n>/merge`` on a pull request.
    That commit already contains the base tip, so the merge base *is* the base
    tip, the base-side set is empty, and the check would pass unconditionally --
    silently, with no signal that it had stopped testing anything.
    """
    _branch_editing_shared(repo)
    _land_on_main_editing_shared(repo)
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert _run(repo) == 1

    _git(repo, "checkout", "-q", "-b", "simulated-merge-ref")
    _git(repo, "merge", "--no-edit", "-q", "main")
    merge_commit = _git(repo, "rev-parse", "HEAD").strip()
    assert merge_commit != head

    against_merge_commit = int(check.main(["--repo", str(repo), "--base-ref", "main", "--head", merge_commit]))
    assert against_merge_commit == 0, (
        "documents the trap: run against the merge commit and the check is vacuous, "
        "which is why the workflow pins the head SHA"
    )


def test_a_rename_on_the_base_still_overlaps_the_old_path(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--no-renames`` keeps a renamed file's old name in the base-side set."""
    _branch_editing_shared(repo)
    _git(repo, "checkout", "-q", "main")
    _git(repo, "mv", _SHARED, "strands_robots/simulation/mujoco/scene_operations.py")
    _commit(repo, "rename on main")
    _git(repo, "checkout", "-q", "pr")

    assert _run(repo) == 1
    assert _SHARED in capsys.readouterr().out


def test_an_unresolvable_base_ref_fails_loudly(repo: Path) -> None:
    """A check that cannot compute its answer must not report the reassuring one."""
    _branch_editing_shared(repo)
    assert int(check.main(["--repo", str(repo), "--base-ref", "no-such-branch", "--head", "HEAD"])) == 1


def test_the_remote_tracking_ref_wins_over_a_local_branch(repo: Path) -> None:
    """CI has only ``origin/<base>``, so that is what must be preferred.

    Pinned by pointing ``origin/main`` at a commit that overlaps while the local
    ``main`` does not: the check must follow the remote-tracking ref.
    """
    _branch_editing_shared(repo)
    _git(repo, "checkout", "-q", "main")
    _edit_line(repo, _SHARED, 1, "line 1  # only on the remote-tracking ref")
    landed = _commit(repo, "a commit reachable only via origin/main")
    _git(repo, "update-ref", "refs/remotes/origin/main", landed)
    _git(repo, "reset", "-q", "--hard", "HEAD~1")  # local main no longer has it
    _git(repo, "checkout", "-q", "pr")

    assert check.resolve_base_ref("main", repo=repo) == "origin/main"
    assert _run(repo) == 1, "the overlap is only visible through origin/main"


# --- the pure core --------------------------------------------------------------


def test_overlap_is_the_sorted_intersection() -> None:
    assert check.overlapping_paths(["b.py", "a.py", "c.py"], ["c.py", "a.py"]) == ("a.py", "c.py")


def test_overlap_is_empty_for_disjoint_sets() -> None:
    assert check.overlapping_paths(["a.py"], ["b.py"]) == ()


@pytest.mark.parametrize("path", ["CHANGELOG.md", "docs/guide.md", "notes.rst", "requirements.txt", "A.MD"])
def test_prose_paths_are_classified_as_prose(path: str) -> None:
    assert check.is_prose(path)


@pytest.mark.parametrize(
    "path",
    [
        "strands_robots/simulation/mujoco/scene_ops.py",
        "tests/test_thing.py",
        "strands_robots/registry/robots.json",
        "pyproject.toml",
        "mkdocs.yml",
        ".github/workflows/test-lint.yml",
    ],
)
def test_behaviour_bearing_paths_are_not_classified_as_prose(path: str) -> None:
    """Non-vacuity: the prose carve-out must not swallow anything that runs."""
    assert not check.is_prose(path)


def test_partition_splits_and_sorts() -> None:
    blocking, prose = check.partition_overlap(["z.md", "b.py", "a.py", "a.md", "b.py"])
    assert blocking == ("a.py", "b.py"), "deduplicated and sorted"
    assert prose == ("a.md", "z.md")


def test_report_names_every_blocking_path_and_the_remedy() -> None:
    report = check.render_report(
        base_ref="main",
        merge_base_sha="32dc3f5b2ca5f226842e4f8e40aaa8e64108e383",
        blocking=[_SHARED],
        prose=["CHANGELOG.md"],
        base_change_count=5,
    )
    assert _SHARED in report
    assert "CHANGELOG.md" in report
    assert "not blocking" in report
    assert "32dc3f5b" in report, "the merge base is what makes the claim checkable"
    assert "merge" in report


def test_report_says_so_when_there_is_no_overlap() -> None:
    report = check.render_report(
        base_ref="main",
        merge_base_sha="0123456789abcdef",
        blocking=[],
        prose=[],
        base_change_count=0,
    )
    assert "No overlap" in report


def test_no_emoji_in_the_script_or_its_output() -> None:
    """Project rule: plain ASCII in anything an agent reads programmatically."""
    source = _SCRIPT_PATH.read_text(encoding="utf-8")
    offenders = [(index, char) for index, char in enumerate(source) if ord(char) > 0x7F]
    assert offenders == [], f"non-ASCII in {_SCRIPT_PATH.name}: {offenders[:5]}"


# --- the tree the gate is read from ---------------------------------------------


def test_the_overlap_does_not_depend_on_the_checked_out_tree(repo: Path) -> None:
    """Same commits, same overlap, whichever tree happens to be checked out.

    This is the property that lets CI run the script from the *base* checkout while
    judging the branch, and CI must: a branch that forked before a gate landed
    carries no copy of that gate's script, so running it out of the head tree exits
    2 before the check begins -- a red X indistinguishable from exit 1, which the
    script reserves for a real untested overlap. That failure was measured on the
    sibling changelog gate in issue #1791; this one shares its shape and is latent
    only because every currently open branch happens to postdate it.

    It holds because nothing here reads the working tree: the path sets come from
    ``git diff --name-only <a>..<b>``, so only reachability matters.
    """
    _branch_editing_shared(repo)
    _land_on_main_editing_shared(repo)
    head = _git(repo, "rev-parse", "HEAD").strip()

    # The branch checked out, as the workflow used to do.
    assert _run_at(repo, head) == 1

    # The base checked out, as the workflow now does. The branch's edit is not in
    # the tree on disk, and the overlap is still reported.
    _git(repo, "checkout", "-q", "main")
    assert "line 40  # edited by the pull request" not in (repo / _SHARED).read_text(encoding="utf-8")
    assert _run_at(repo, head) == 1


def test_a_branch_with_no_overlap_still_passes_from_the_base_checkout(repo: Path) -> None:
    """The other direction: reading from the base does not fail every branch.

    Without this, the test above is satisfied by a check that refuses everything.
    """
    _branch_editing_shared(repo)
    head = _git(repo, "rev-parse", "HEAD").strip()

    _git(repo, "checkout", "-q", "main")
    assert _run_at(repo, head) == 0


def test_the_workflow_reads_the_script_from_the_base_and_names_the_head() -> None:
    """The workflow must not require its own script in the tree under review.

    A ``pull_request`` workflow definition is read from the merge commit, so a gate
    runs against heads that contain neither it nor its script. The sibling changelog
    gate exited 2 for exactly that reason (issue #1791); this workflow had the same
    shape.
    """
    workflow = (_REPO_ROOT / ".github" / "workflows" / "merge-base-overlap.yml").read_text(encoding="utf-8")

    assert "ref: ${{ github.base_ref }}" in workflow, "the gate's script must come from the base branch"
    assert "ref: ${{ github.event.pull_request.head.sha }}" not in workflow, (
        "checking the head out is what made the script's presence a precondition"
    )
    assert "HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in workflow
    assert '--head "$HEAD_SHA"' in workflow, "the commit under test is named, not checked out"


# --- the open set ---------------------------------------------------------------
#
# ``--all-open`` answers a question no per-branch run can: ``M..base`` is empty by
# construction whenever ``M`` is the base a branch was evaluated against, so two
# pull requests that are both still open are invisible to each other, and a
# sibling merging afterwards invalidates nothing. These pins are API-shaped rather
# than git-shaped because the sweep reads no checkout -- ``_get`` is its single
# seam, so faking that is faking the network and nothing else.


def _files(names: list[str], renamed: dict[str, str] | None = None) -> list[dict[str, object]]:
    """Build a file list in the shape both file-carrying endpoints return.

    ``renamed`` maps a new path to the path it was renamed from, which the API
    reports as ``previous_filename`` on that same entry rather than as a second
    entry -- so a rename is one row naming two paths, and reading only one of
    them is invisible in the payload.
    """
    previous = renamed or {}
    rows: list[dict[str, object]] = []
    for name in names:
        row: dict[str, object] = {"filename": name}
        if name in previous:
            row["previous_filename"] = previous[name]
        rows.append(row)
    return rows


def _compare(
    files: list[str],
    *,
    merge_base: str = "aaaa1111",
    behind_by: int = 0,
    renamed: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build one ``compare`` payload in the shape the endpoint returns."""
    return {
        "merge_base_commit": {"sha": merge_base},
        "behind_by": behind_by,
        "files": _files(files, renamed),
    }


def _api(
    pulls: list[dict[str, object]],
    compares: dict[str, object],
    pull_files: dict[int, list[dict[str, object]]] | None = None,
    base: str = "main",
) -> Callable[[str, str], object]:
    """Return a ``_get`` stand-in serving the open set, compares, and head file lists.

    A compare with no recorded payload raises, which is deliberate: an unrecorded
    lookup is a test that has not said what it means, and silently returning an
    empty path set would make it pass as "no overlap".

    ``pull_files`` records a pull request's own paginated file list. A number with
    no entry is served, paginated, from that pull request's ``base...head``
    compare recording -- sound below the compare cap because the two endpoints
    return the same set there, measured on #1035 (7 files), #1722 (10) and #1667
    (153), identical both ways. The tests where the two must differ record the
    divergence explicitly, which is the only place the distinction carries
    meaning.
    """
    heads = {int(cast(int, row["number"])): str(cast(dict[str, object], row["head"])["sha"]) for row in pulls}
    recorded = dict(pull_files or {})

    def get(url: str, token: str) -> object:
        if "/pulls?" in url:
            return pulls if url.endswith("page=1") else []
        if "/files?" in url:
            number = int(url.split("/pulls/", 1)[1].split("/files", 1)[0])
            entries = recorded.get(number)
            if entries is None:
                key = f"{base}...{heads[number]}"
                if key not in compares:
                    raise check.ApiError(f"no recorded file list for #{number}")
                entries = cast(list[dict[str, object]], cast(dict[str, object], compares[key])["files"])
            page = int(url.rsplit("page=", 1)[1])
            start = (page - 1) * check._PULL_FILE_PAGE
            return entries[start : start + check._PULL_FILE_PAGE]
        key = url.split("/compare/", 1)[1]
        if key not in compares:
            raise check.ApiError(f"no recorded compare for {key}")
        return compares[key]

    return get


def _pull(number: int, head: str, *, draft: bool = False) -> dict[str, object]:
    return {"number": number, "head": {"sha": head}, "draft": draft}


def _sweep(monkeypatch: pytest.MonkeyPatch, get: Callable[[str, str], object], tmp_path: Path) -> int:
    """Run the sweep from a directory that is not a repository at all.

    Every sweep test runs from ``tmp_path`` rather than the checkout, which pins
    the property the mode depends on: the open set comes from the API, so a caller
    reporting repository health needs no clone and no fetch of ten pull request
    heads.
    """
    monkeypatch.setattr(check, "_get", get)
    monkeypatch.chdir(tmp_path)
    return int(check.main(["--all-open", "--github-repo", "owner/name", "--token", "t"]))


def test_a_pair_of_open_pull_requests_editing_one_path_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The finding the single-branch mode cannot make: two open branches, one file.

    This is the #1763/#1766 topology before either has merged. Each pull request's
    checks ran against a base holding neither change, so both read clean and the
    first tree in which the two are compiled together is ``main``.
    """
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20")],
        {
            "main...head10": _compare([_SHARED, "tests/a_test.py"]),
            "head10...main": _compare([]),
            "main...head20": _compare([_SHARED, "tests/b_test.py"]),
            "head20...main": _compare([]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 1

    report = capsys.readouterr().out
    assert "#10 + #20" in report
    assert _SHARED in report
    # Only the shared path is named: the two private files are not the finding.
    assert "tests/a_test.py" not in report
    assert "tests/b_test.py" not in report


def test_a_prose_only_pair_is_listed_but_does_not_set_the_exit_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same suppression the single-branch mode applies, and for the same reason.

    Reported so a reader can see it was considered, not blocking because prose
    cannot change what the package or the suite does -- and a genuine collision
    inside one paragraph arrives as a merge conflict, which is a signal already.
    """
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20")],
        {
            "main...head10": _compare(["docs/guide.md"]),
            "head10...main": _compare([]),
            "main...head20": _compare(["docs/guide.md"]),
            "head20...main": _compare([]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 0

    report = capsys.readouterr().out
    assert "prose-only" in report
    assert "#10 + #20: `docs/guide.md`" in report
    assert "behaviour-bearing path" not in report


def test_disjoint_open_pull_requests_report_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without this, the pins above are satisfied by a sweep that flags everything."""
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20")],
        {
            "main...head10": _compare(["strands_robots/one.py"]),
            "head10...main": _compare([]),
            "main...head20": _compare(["strands_robots/two.py"]),
            "head20...main": _compare([]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 0
    assert "shares a changed path" in capsys.readouterr().out


def test_a_clean_sweep_claims_only_the_paths_it_compared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A clean report may not claim there is no untested composition.

    It compared changed *paths*, and a test resolving its population from a
    filesystem walk is coupled to files it never names -- so its intersection
    with the sibling it grades is empty and this mode cannot describe that
    composition at all. #2557's whole-tree ASCII grader merged in a batch with
    two siblings that added exactly the prose it scores, at a pairwise path
    intersection of ``[]``.

    The wording is the whole finding: "no untested composition in the open set"
    is a claim about compositions, which is strictly more than the relation
    measured, and a reader who trusts it stops looking. Widening the path set to
    the walked root was measured and is not the remedy -- it selects 11 of 36
    pairs on the live open set and names no defect -- and the relation that does
    settle it needs a checkout ``_sweep`` deliberately does not have. So the
    report is scoped instead. See issue #2561.
    """
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20")],
        {
            "main...head10": _compare(["strands_robots/one.py"]),
            "head10...main": _compare([]),
            "main...head20": _compare(["strands_robots/two.py"]),
            "head20...main": _compare([]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 0
    report = capsys.readouterr().out
    assert "No untested composition" not in report
    assert "resolves its population from a filesystem walk" in report
    assert "#2561" in report


def test_the_named_blind_spot_survives_a_populated_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The caveat is unconditional, because the blind spot is.

    A populated report is the case where a reader is most likely to treat the
    rows as the full set of untested compositions, so the paragraph that says
    which class is missing cannot be attached to the clean branch only.
    """
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20")],
        {
            "main...head10": _compare(["strands_robots/shared.py"]),
            "head10...main": _compare([]),
            "main...head20": _compare(["strands_robots/shared.py"]),
            "head20...main": _compare([]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 1
    report = capsys.readouterr().out
    assert "#10 + #20" in report
    assert "resolves its population from a filesystem walk" in report


def test_a_draft_pull_request_is_excluded_from_the_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A draft cannot merge whatever else is true of it.

    So a pair involving one does not mean what a pair of ready pull requests
    means, and reporting it would spend a merge-order decision on a change nobody
    has offered yet. The excluded pull request's compares are deliberately absent
    from the recorded set: reaching for them at all would raise.
    """
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20", draft=True)],
        {"main...head10": _compare([_SHARED]), "head10...main": _compare([])},
    )

    assert _sweep(monkeypatch, get, tmp_path) == 0

    report = capsys.readouterr().out
    assert "1 open non-draft pull request(s)" in report
    assert "#20" not in report


def test_a_stale_base_overlap_is_reported_with_its_distance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The second mode: the single-branch computation, applied to the population.

    Nothing invalidates a pull request's pass when its sibling merges. Stale
    approvals are dismissed on push; a stale pass has no equivalent, and a pull
    request idle in review never re-runs -- so the exposure runs until its next
    push, not until the next merge.
    """
    get = _api(
        [_pull(10, "head10")],
        {
            "main...head10": _compare([_SHARED], behind_by=7),
            "head10...main": _compare([_SHARED, "strands_robots/other.py"]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 1

    report = capsys.readouterr().out
    assert "base moved under a path they edit" in report
    assert "| #10 | 7 |" in report
    assert _SHARED in report
    assert "strands_robots/other.py" not in report


def test_a_pull_request_level_with_its_base_is_not_a_stale_base_finding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``behind_by == 0`` is the state the single-branch check already cleared.

    Reporting it would re-report every green pull request in the repository, which
    is how a sweep becomes noise nobody reads.
    """
    get = _api(
        [_pull(10, "head10")],
        {
            "main...head10": _compare([_SHARED], behind_by=0),
            "head10...main": _compare([_SHARED]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 0
    assert "base moved under a path" not in capsys.readouterr().out


def test_a_truncated_head_side_path_set_is_named_rather_than_read_as_no_overlap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file list that stopped at its ceiling looks complete in the payload.

    This check's failure mode is a *missed* overlap, so a path set that reached
    the ceiling is named as unevaluated. Quietly intersecting a truncated set is
    exactly how one goes missing, and the report would say "clean" while meaning
    "did not look". The ceiling that applies to the head side is the paginated
    endpoint's, ten times the compare cap.
    """
    at_ceiling = _files([f"strands_robots/f{index}.py" for index in range(check._PULL_FILE_CAP)])
    get = _api(
        [_pull(10, "head10")],
        {"main...head10": _compare([]), "head10...main": _compare([])},
        pull_files={10: at_ceiling},
    )

    assert _sweep(monkeypatch, get, tmp_path) == 0

    report = capsys.readouterr().out
    assert "Unevaluated (1)" in report
    assert "#10: not evaluated in either mode" in report
    assert f"{check._PULL_FILE_CAP}-entry ceiling" in report
    assert "0 open non-draft pull request(s)" in report


def test_a_head_side_at_the_compare_cap_stays_in_the_pairwise_comparison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The compare cap must not remove a pull request from the mode that finds defects.

    The compare call survives for ``merge_base_commit`` and ``behind_by``, which
    the paginated endpoint does not carry, and its ``files`` list is not read --
    so a head side at the 300-entry cap is not a reason to drop the pull request.
    It used to be, and a large diff is the most likely thing on a queue to collide
    with something, so the old behaviour discarded findings in proportion to how
    much they were worth.
    """
    capped = [f"strands_robots/f{index}.py" for index in range(check._COMPARE_FILE_CAP)]
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20")],
        {
            "main...head10": _compare(capped),
            "head10...main": _compare([]),
            "main...head20": _compare([_SHARED]),
            "head20...main": _compare([]),
        },
        pull_files={10: _files([_SHARED, "strands_robots/private.py"])},
    )

    assert _sweep(monkeypatch, get, tmp_path) == 1

    report = capsys.readouterr().out
    assert "#10 + #20" in report, "a capped compare must not drop the pull request from the pairwise mode"
    assert "Unevaluated" not in report
    assert "2 open non-draft pull request(s)" in report


def test_a_head_side_path_set_spanning_several_pages_is_read_whole(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pagination is the point, so a set larger than one page must be followed.

    A read that stopped after the first page would raise the ceiling on paper and
    lower it in fact: the overlap here sits on the last page, past two full ones.
    """
    filler = [f"strands_robots/f{index}.py" for index in range(2 * check._PULL_FILE_PAGE)]
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20")],
        {
            "main...head10": _compare([]),
            "head10...main": _compare([]),
            "main...head20": _compare([_SHARED]),
            "head20...main": _compare([]),
        },
        pull_files={10: _files([*filler, _SHARED])},
    )

    assert _sweep(monkeypatch, get, tmp_path) == 1

    report = capsys.readouterr().out
    assert "#10 + #20" in report
    assert _SHARED in report


def test_a_renamed_path_overlaps_a_sibling_editing_the_old_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rename and an edit of the old name compose with no conflict marker.

    The API reports a rename as one entry carrying both names, so collecting only
    ``filename`` leaves the two branches sharing no path -- while git, which does
    detect the rename, applies the edit to the new name and merges cleanly. That
    is the silent composition the sweep exists to report, so it must not be the
    one shape the sweep cannot see. Merged #2057 renamed a test file, and a
    sibling still editing the old path would have been invisible.
    """
    renamed_to = "tests/test_args_docstring_completeness.py"
    old_name = "tests/simulation/test_args_docstring_completeness.py"
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20")],
        {
            "main...head10": _compare([renamed_to], renamed={renamed_to: old_name}),
            "head10...main": _compare([]),
            "main...head20": _compare([old_name]),
            "head20...main": _compare([]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 1

    report = capsys.readouterr().out
    assert "#10 + #20" in report
    assert old_name in report


def test_the_two_modes_agree_about_a_rename() -> None:
    """``--no-renames`` and ``previous_filename`` must reach the same path set.

    ``test_a_rename_on_the_base_still_overlaps_the_old_path`` pins the git side of
    this from real commits. The two modes reading a rename differently is the
    class of divergence that makes one of them wrong without either looking it,
    so the equality is asserted rather than inferred from two separate verdicts.
    """
    renamed_to = "strands_robots/b.py"
    old_name = "strands_robots/a.py"
    entries = _files([renamed_to], renamed={renamed_to: old_name})

    assert check.paths_from_entries(entries) == frozenset({renamed_to, old_name})


def test_a_truncated_base_side_set_still_leaves_the_pair_comparison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two path sets are inputs to two independent modes, so they skip apart.

    Measured on the live queue: the base-side set is the one that grows without
    bound, so it is the one that reaches the cap -- on #1035, 265 commits behind.
    Dropping the whole pull request for it would have discarded the pairwise
    finding against #1722, which is the finding this mode exists to make. The base
    side is also the only side still read from the capped endpoint: it has no
    paginated equivalent, so this is the one cap the sweep cannot route around.
    """
    capped = [f"strands_robots/f{index}.py" for index in range(check._COMPARE_FILE_CAP)]
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20")],
        {
            "main...head10": _compare([_SHARED], behind_by=265),
            "head10...main": _compare(capped),
            "main...head20": _compare([_SHARED]),
            "head20...main": _compare([]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 1

    report = capsys.readouterr().out
    assert "#10 + #20" in report, "the pair comparison must survive an unreadable base side"
    assert "stale-base mode only" in report
    assert "base moved under a path they edit" not in report


def test_one_unreadable_pull_request_does_not_suppress_the_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sweep exists to report across a population, so one failure is not fatal.

    The unreadable pull request is named rather than counted as clean: one this
    check could not read is not one it cleared, which is the failure mode the
    whole file is written against.
    """
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20"), _pull(30, "head30")],
        {
            "main...head10": _compare([_SHARED]),
            "head10...main": _compare([]),
            "main...head30": _compare([_SHARED]),
            "head30...main": _compare([]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 1

    report = capsys.readouterr().out
    assert "#10 + #30" in report
    assert "#20: not evaluated in either mode" in report


def test_the_sweep_and_the_single_branch_mode_share_one_prose_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both modes call ``partition_overlap``, so they cannot disagree about prose.

    Pinned by construction rather than by comparing two reports: narrowing
    ``PROSE_SUFFIXES`` moves both verdicts together, which is the point of the
    sweep reusing the path-set helpers instead of carrying its own copy.
    """
    monkeypatch.setattr(check, "PROSE_SUFFIXES", frozenset())
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20")],
        {
            "main...head10": _compare(["docs/guide.md"]),
            "head10...main": _compare([]),
            "main...head20": _compare(["docs/guide.md"]),
            "head20...main": _compare([]),
        },
    )

    # The same prose-only pair the test above cleared now blocks, because the one
    # classification both modes read has changed.
    assert _sweep(monkeypatch, get, tmp_path) == 1


def test_all_open_and_head_are_mutually_exclusive(tmp_path: Path) -> None:
    """``--head`` names one commit and the sweep reads none.

    Ignoring it would answer a question the caller did not ask while looking like
    it had, which is the same reasoning the sibling sweep gives for refusing
    ``--all-open --pr``.
    """
    with pytest.raises(SystemExit) as raised:
        check.main(["--all-open", "--head", "abc123", "--github-repo", "owner/name", "--token", "t"])
    assert raised.value.code == 2


@pytest.mark.parametrize("missing", ["--github-repo", "--token"])
def test_the_sweep_refuses_to_run_without_its_own_inputs(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """Neither input has a local fallback, so a missing one is refused, not guessed.

    ``--repo`` is not consulted for either: in this script it is a checkout path,
    and the sweep reads no checkout.
    """
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    argv = ["--all-open", "--github-repo", "owner/name", "--token", "t"]
    argv.remove(missing)
    argv.remove("owner/name" if missing == "--github-repo" else "t")

    with pytest.raises(SystemExit) as raised:
        check.main(argv)
    assert raised.value.code == 2
