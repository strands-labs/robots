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

import ast
import importlib.util
import inspect
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
        named=[],
        orphaned=[],
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
        named=[],
        orphaned=[],
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


def _files(
    names: list[str],
    renamed: dict[str, str] | None = None,
    patches: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Build a file list in the shape both file-carrying endpoints return.

    ``renamed`` maps a new path to the path it was renamed from, which the API
    reports as ``previous_filename`` on that same entry rather than as a second
    entry -- so a rename is one row naming two paths, and reading only one of
    them is invisible in the payload.

    ``patches`` maps a path to its unified diff, the ``patch`` field the endpoint
    returns alongside ``filename``. Omitted by default, and omitting it is
    meaningful rather than merely terse: an entry with no diff contributes no
    module literal, which is the shape every path-relation test here wants.
    """
    previous = renamed or {}
    diffs = patches or {}
    rows: list[dict[str, object]] = []
    for name in names:
        row: dict[str, object] = {"filename": name}
        if name in previous:
            row["previous_filename"] = previous[name]
        if name in diffs:
            row["patch"] = diffs[name]
        rows.append(row)
    return rows


def _compare(
    files: list[str],
    *,
    merge_base: str = "aaaa1111",
    behind_by: int = 0,
    renamed: dict[str, str] | None = None,
    patches: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build one ``compare`` payload in the shape the endpoint returns."""
    return {
        "merge_base_commit": {"sha": merge_base},
        "behind_by": behind_by,
        "files": _files(files, renamed, patches),
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


def _flag_partition() -> tuple[dict[str, str], set[str]]:
    """Split ``main``'s value-bearing flags into what the sweep reads and what it does not.

    Derived from the source rather than listed. Every value the sweep reads is
    passed to ``_run_sweep``, so its complement is exactly the set of flags only
    the single-branch path consults -- which is the partition the mutual-exclusion
    checks have to cover. A flag added later lands on one side or the other
    without this test being edited.

    ``store_true`` flags are skipped: they select the mode rather than carry an
    input it could read.
    """
    options: dict[str, str] = {}
    swept: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(check.main))):
        if not isinstance(node, ast.Call):
            continue
        called = ast.unparse(node.func)
        if called.endswith("add_argument") and node.args and isinstance(node.args[0], ast.Constant):
            option = str(node.args[0].value)
            if not option.startswith("--"):
                continue
            if any(
                keyword.arg == "action"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "store_true"
                for keyword in node.keywords
            ):
                continue
            options[option.removeprefix("--").replace("-", "_")] = option
        elif called == "_run_sweep":
            swept.update(
                argument.attr
                for argument in node.args
                if isinstance(argument, ast.Attribute) and ast.unparse(argument).startswith("args.")
            )
    return options, swept


_FLAG_OPTIONS, _SWEEP_READS = _flag_partition()

#: Flags the sweep is handed none of, so passing one to ``--all-open`` describes a
#: run that is not happening.
_SWEEP_IGNORES = tuple(sorted(set(_FLAG_OPTIONS) - _SWEEP_READS))


def _refusal(argv: list[str], captured: pytest.CaptureFixture[str]) -> str:
    """Run ``main`` expecting an argparse refusal and return the message it gave.

    Only the text after ``error:`` -- argparse prints the whole usage line first,
    and the usage line names every flag, so asserting against the full stderr
    would pass whatever the refusal said.

    An accepted call is reported as what it did rather than as a missing raise:
    the run went ahead reading a value it was never handed, which is the thing
    being pinned.
    """
    try:
        code = check.main(argv)
    except SystemExit as refused:
        assert refused.code == 2, f"expected an argparse refusal, got exit {refused.code}"
        return captured.readouterr().err.split("error:", 1)[1].strip()
    raise AssertionError(
        f"argparse accepted {' '.join(argv)} and the run went ahead (exit {code}): the sweep is handed "
        "none of that flag, so the value named nothing and the report describes whatever "
        "$GITHUB_REPOSITORY names instead"
    )


def test_all_open_and_repo_are_mutually_exclusive(capsys: pytest.CaptureFixture[str]) -> None:
    """``--repo`` names one checkout and the sweep reads none, so it is refused too.

    The same reasoning as ``--head`` above, on the flag a caller is far more
    likely to reach for: the sibling gate scripts spell ``owner/name`` ``--repo``,
    so ``--repo strands-labs/robots --all-open`` reads as naming the repository.
    Accepted as a path it named nothing -- the sweep read ``$GITHUB_REPOSITORY``
    and reported on whatever repository the command happened to be running in,
    exiting ``0`` with a report shaped exactly like the right one. Measured before
    this refusal, run from a checkout of a fork: the same command reported ``0 open
    non-draft pull request(s)`` for the fork, where naming the intended repository
    reports 10.

    So the refusal also names ``--github-repo``: a caller who passed ``--repo``
    wanted the repository named, and that is the flag that names it.
    """
    message = _refusal(
        ["--all-open", "--repo", "strands-labs/robots", "--github-repo", "owner/name", "--token", "t"],
        capsys,
    )
    assert "--repo" in message
    assert "--github-repo" in message, f"the refusal must name the flag that does name a repository: {message!r}"


@pytest.mark.parametrize("dest", _SWEEP_IGNORES)
def test_a_flag_the_sweep_is_handed_none_of_is_refused_rather_than_ignored(
    dest: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every flag outside ``_run_sweep``'s arguments is refused by ``--all-open``.

    The root-cause form of the two named tests around it: the rule is a property
    of the partition, not of the two flags that happen to be on the local side
    today, so a third one is graded on arrival rather than silently ignored.
    """
    option = _FLAG_OPTIONS[dest]
    message = _refusal(
        ["--all-open", option, "placeholder", "--github-repo", "owner/name", "--token", "t"],
        capsys,
    )
    assert option in message, f"the refusal must name the flag it refused: {message!r}"


def test_the_partition_is_derived_and_finds_both_sides() -> None:
    """Non-vacuity: a derivation that found nothing would grade nothing.

    Pinned as the two sets rather than a count, because each side carries a
    different obligation -- the sweep's own inputs are refused when *missing*
    (above), and the local ones when *present*.
    """
    assert _SWEEP_READS == {"github_repo", "base_ref", "token"}, _SWEEP_READS
    assert set(_SWEEP_IGNORES) == {"head", "repo"}, _SWEEP_IGNORES


def test_the_flag_both_modes_read_is_not_refused_by_the_sweep(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--base-ref`` is handed to the sweep, so naming it is not a mistake.

    The boundary: the refusal is scoped to flags the sweep is handed none of, not
    to every flag the single-branch path also happens to read.
    """
    monkeypatch.setattr(check, "_get", _api([], {}, base="release"))
    monkeypatch.chdir(tmp_path)
    assert check.main(["--all-open", "--base-ref", "release", "--github-repo", "owner/name", "--token", "t"]) == 0


def test_one_branch_still_reads_the_checkout_path(repo: Path) -> None:
    """``--repo`` is refused by the sweep only; it stays the single-branch input.

    Which is what makes the refusal a scoping statement rather than a removal.
    """
    _branch_editing_shared(repo)
    assert check.main(["--repo", str(repo), "--base-ref", "main", "--head", "HEAD"]) == 0


def test_the_head_refusal_is_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    """Each refusal stays specific to the flag it refused.

    Collapsing them into one message would lose the ``--github-repo`` pointer,
    which is the half a caller who reached for ``--repo`` needs.
    """
    message = _refusal(
        ["--all-open", "--head", "abc123", "--github-repo", "owner/name", "--token", "t"],
        capsys,
    )
    assert message == "--all-open sweeps the open set and reads no local commit; --head names one branch"


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


# --- the named-module relation ---------------------------------------------------
#
# `main` went red at `828f80eb` from #2762 and #2774 with the pairwise path
# intersection empty: #2762's fixture reaches into
# `strands_robots.device_connect` by string, #2774 rewrote that package's
# `__init__`, and neither branch changed a path the other did. The relation these
# pin keys on the name a test writes down rather than on a path, which is why it
# reaches that composition. Issues #2791, #2795.

#: The package #2774 rewrote, named by #2762's fixture two segments deeper.
_NAMED_PACKAGE = "strands_robots/device_connect/__init__.py"

#: The literal #2762's fixture patches, verbatim.
_NAMED_LITERAL = "strands_robots.device_connect.reachy_transport.api"


def _branch_naming_a_module(repo: Path, literal: str = _NAMED_LITERAL, *, in_test: bool = True) -> None:
    """Branch off ``main`` and add a file reaching into ``literal`` by string.

    ``in_test`` places the file under ``tests/`` or under the package. The
    distinction is the relation's whole scope: production code that needs another
    module imports it, and an import is a path the diff already carries.
    """
    _git(repo, "checkout", "-q", "-b", "pr")
    body = f'monkeypatch.setattr("{literal}", stand_in)\n'
    relative = "tests/drivers/test_reachy_transport_guards.py" if in_test else "strands_robots/drivers/reachy.py"
    _write(repo, relative, body)
    _commit(repo, "the pull request's commit")


def _land_on_main_changing(repo: Path, relative: str) -> None:
    """Land a commit on ``main`` rewriting ``relative``, as #2774 did."""
    _git(repo, "checkout", "-q", "main")
    _write(repo, relative, "def __getattr__(name):\n    raise AttributeError(name)\n")
    _commit(repo, "the commit that landed on main first")
    _git(repo, "checkout", "-q", "pr")


def test_a_branch_whose_test_names_a_module_the_base_rewrote_is_flagged(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The #2762/#2774 topology in the single-branch mode: no shared path at all.

    The branch adds a test that patches ``_NAMED_LITERAL`` by string; the base then
    rewrites the package two segments up, which is the module that name resolves
    through. The path intersection is empty -- the branch edits one file under
    ``tests/`` and the base one under ``strands_robots/`` -- so the relation that
    reaches this is the name, and the report has to say which module it was.
    """
    _branch_naming_a_module(repo)
    _land_on_main_changing(repo, _NAMED_PACKAGE)

    assert _run(repo) == 1

    report = capsys.readouterr().out
    assert _NAMED_PACKAGE in report
    assert "name" in report.lower()


def test_the_named_module_finding_shares_no_path_with_the_base(repo: Path) -> None:
    """Premise: the path relation really is empty here, so it is not doing the work.

    Without this the test above would pass on a tree where the two branches happen
    to share a file, and the finding would be attributable to the relation that
    already existed.
    """
    _branch_naming_a_module(repo)
    _land_on_main_changing(repo, _NAMED_PACKAGE)

    fork_point = check.merge_base("main", "HEAD", repo=repo)
    branch_paths = check.changed_paths(fork_point, "HEAD", repo=repo)
    base_paths = check.changed_paths(fork_point, "main", repo=repo)
    assert check.overlapping_paths(branch_paths, base_paths) == ()
    assert _NAMED_PACKAGE in base_paths


def test_merging_the_base_clears_a_named_module_finding(repo: Path) -> None:
    """Self-clearing, like the path relation: the remedy asked for is the remedy.

    Merging the base advances the merge base, so the base-side path set is empty
    and no module the branch names can be in it.
    """
    _branch_naming_a_module(repo)
    _land_on_main_changing(repo, _NAMED_PACKAGE)
    assert _run(repo) == 1

    _git(repo, "merge", "-q", "--no-edit", "main")
    assert _run(repo) == 0


def test_a_module_named_by_a_test_and_changed_by_nobody_does_not_block(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Quiet where it cannot matter: a name is only a finding against a change.

    Naming a module is the normal way a test reaches into one, so a relation that
    fired on the name alone would fire on most test diffs in the tree.
    """
    _branch_naming_a_module(repo)
    _git(repo, "checkout", "-q", "main")
    _write(repo, "strands_robots/unrelated.py", "value = 1\n")
    _commit(repo, "an unrelated commit on main")
    _git(repo, "checkout", "-q", "pr")

    assert _run(repo) == 0
    assert _NAMED_PACKAGE not in capsys.readouterr().out


def test_prose_naming_a_module_is_not_a_reach_into_it(repo: Path) -> None:
    """A docstring that mentions a module is not a test that patches one.

    The literal has to be the entire contents of the string for the same reason
    the walked-root widening was rejected: a relation that fires on the characters
    appearing anywhere would select every test whose docstring explains what it
    covers, which is most of them.
    """
    _git(repo, "checkout", "-q", "-b", "pr")
    _write(
        repo,
        "tests/drivers/test_reachy_transport_guards.py",
        '"""Covers strands_robots.device_connect.reachy_transport.api indirectly."""\n',
    )
    _commit(repo, "a test whose prose names the module")
    _land_on_main_changing(repo, _NAMED_PACKAGE)

    assert _run(repo) == 0


def test_a_literal_in_production_code_is_not_read(repo: Path) -> None:
    """Scoped to test modules, which is where a name is the only coupling.

    Production code that needs another module imports it, and an import puts the
    importing file in the diff -- so the path relation already carries it and a
    second reading of the same coupling would double-report it.
    """
    _branch_naming_a_module(repo, in_test=False)
    _land_on_main_changing(repo, _NAMED_PACKAGE)

    assert _run(repo) == 0


def test_the_bare_package_root_is_not_resolved(repo: Path) -> None:
    """``strands_robots.mesh`` resolves; ``strands_robots`` alone does not.

    Measured over 2676 co-open pairs from 400 pull requests: resolving the root as
    well adds four findings, every one of them a pair with #2486's edit to
    ``strands_robots/__init__.py``, which every literal in the tree names by its
    first segment. That is the shallowest coupling expressible and the one a
    reader can least act on.
    """
    assert "strands_robots/__init__.py" not in check.named_module_paths({"strands_robots.mesh.core"})
    assert "strands_robots/mesh/__init__.py" in check.named_module_paths({"strands_robots.mesh.core"})
    assert check.named_module_paths({"strands_robots"}) == frozenset()


def test_a_prefix_of_the_literal_resolves_because_importing_runs_it(repo: Path) -> None:
    """The #2774 edit was two segments shallower than the name #2762 wrote.

    Importing ``a.b.c`` executes ``a/b/__init__.py``, so a change to the package
    is a change to what the name resolves through. A relation reading only the
    full dotted path would have missed the composition that broke ``main``.
    """
    resolved = check.named_module_paths({_NAMED_LITERAL})
    assert _NAMED_PACKAGE in resolved
    assert "strands_robots/device_connect/reachy_transport.py" in resolved


def test_a_named_module_the_branch_also_edits_is_reported_once(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """One coupling, one row: the path relation already reports this one.

    A branch that both names a module and edits it is the case where the two
    relations agree, and reporting it twice would spend a reader's attention on a
    second row carrying no further information.
    """
    _git(repo, "checkout", "-q", "-b", "pr")
    _write(repo, "tests/drivers/test_reachy_transport_guards.py", f'monkeypatch.setattr("{_NAMED_LITERAL}", x)\n')
    _write(repo, _NAMED_PACKAGE, "value = 1\n")
    _commit(repo, "a branch that both names and edits the module")
    _land_on_main_changing(repo, _NAMED_PACKAGE)

    assert _run(repo) == 1
    report = capsys.readouterr().out
    # It is in the shared-path finding, and there is no second section for it: the
    # named-module list is what this branch's own edit already accounts for.
    assert "behaviour-bearing path(s)" in report
    assert "tests name" not in report


def test_the_2762_2774_pair_is_flagged_in_the_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The incident, replayed in the mode that could have reported it before merge.

    Both were open at once, so this is the pairwise relation's case, and the path
    sets and the literal below are the real ones: #2762's ten files, #2774's five,
    intersecting to nothing.
    """
    fixture = "tests/drivers/test_reachy_transport_guards_are_reachable.py"
    patch = (
        "@@ -0,0 +1,2 @@\n"
        '+    for name in [n for n in list(sys.modules) if n.startswith("strands_robots.device_connect")]:\n'
        f'+    monkeypatch.setattr("{_NAMED_LITERAL}", lambda *a, **k: dict(_LITE_STATUS))\n'
    )
    get = _api(
        [_pull(2762, "head2762"), _pull(2774, "head2774")],
        {
            "main...head2762": _compare(
                ["strands_robots/drivers/reachy.py", "strands_robots/drivers/__init__.py", fixture],
                patches={fixture: patch},
            ),
            "head2762...main": _compare([]),
            "main...head2774": _compare([_NAMED_PACKAGE, "strands_robots/device_connect/_impl.py"]),
            "head2774...main": _compare([]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 1

    report = capsys.readouterr().out
    assert "#2762 + #2774" in report
    assert _NAMED_PACKAGE in report
    # The path relation is silent, so the pair cannot be in the shared-path table.
    assert "Pairs editing the same behaviour-bearing path" not in report


def test_the_named_module_relation_costs_no_additional_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The literals come from the payload the path set is already built from.

    ``patch`` arrives beside ``filename`` on the same entries, so a second reading
    of them is free. A relation that fetched file contents per test module would
    multiply the sweep's request count by the size of the queue, which is the cost
    that keeps this mode runnable without a clone.
    """
    fixture = "tests/drivers/test_guards.py"
    patch = f'@@ -0,0 +1 @@\n+monkeypatch.setattr("{_NAMED_LITERAL}", x)\n'
    recorded = _api(
        [_pull(10, "head10"), _pull(20, "head20")],
        {
            "main...head10": _compare([fixture], patches={fixture: patch}),
            "head10...main": _compare([]),
            "main...head20": _compare([_NAMED_PACKAGE]),
            "head20...main": _compare([]),
        },
    )
    urls: list[str] = []

    def counting(url: str, token: str) -> object:
        urls.append(url)
        return recorded(url, token)

    assert _sweep(monkeypatch, counting, tmp_path) == 1
    # Two pull requests, one page of files each: the relation adds no third fetch.
    assert len([url for url in urls if "/files?" in url]) == 2


def test_a_base_that_changed_a_module_the_branchs_tests_name_is_reported_with_its_distance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The stale-base half, mirroring ``stale_base_overlaps``.

    Nothing re-runs a pull request idle in review, so a module its tests name can
    be rewritten under it with its green result standing.
    """
    fixture = "tests/drivers/test_guards.py"
    patch = f'@@ -0,0 +1 @@\n+import_module("{_NAMED_LITERAL}")\n'
    get = _api(
        [_pull(10, "head10")],
        {
            "main...head10": _compare([fixture], behind_by=4, patches={fixture: patch}),
            "head10...main": _compare([_NAMED_PACKAGE]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 1

    report = capsys.readouterr().out
    assert _NAMED_PACKAGE in report
    assert "| #10 | 4 |" in report


def test_both_modes_read_module_literals_through_one_extractor() -> None:
    """Parity, for the reason the prose rule is shared: two readings would drift.

    The sweep reads the API's ``patch`` field and the single-branch mode reads
    ``git diff``, but both hand ``(path, patch)`` pairs to ``module_literals``, so
    neither mode can disagree with the other about what a literal is.
    """
    sweep_side = inspect.getsource(check.literals_from_entries)
    branch_side = inspect.getsource(check.main)
    assert "module_literals(" in sweep_side
    assert "module_literals(" in branch_side
    assert "diff_entries(" in branch_side
    # One definition of the pattern, so a change to it moves both modes together.
    tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
    compiled = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_MODULE_LITERAL" for t in node.targets)
    ]
    assert len(compiled) == 1


def test_the_walk_blind_spot_is_still_named_now_that_a_name_is_reachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reaching a named coupling must not read as reaching every coupling.

    A population resolved by ``rglob`` writes nothing down, so it shares neither a
    path nor a literal with the siblings it grades. The caveat stays unconditional,
    and it stays true: this relation reaches couplings a test states, not couplings
    inferred from a glob.
    """
    fixture = "tests/drivers/test_guards.py"
    patch = f'@@ -0,0 +1 @@\n+import_module("{_NAMED_LITERAL}")\n'
    get = _api(
        [_pull(10, "head10"), _pull(20, "head20")],
        {
            "main...head10": _compare([fixture], patches={fixture: patch}),
            "head10...main": _compare([]),
            "main...head20": _compare([_NAMED_PACKAGE]),
            "head20...main": _compare([]),
        },
    )

    assert _sweep(monkeypatch, get, tmp_path) == 1
    report = capsys.readouterr().out
    assert "resolves its population from a filesystem walk" in report
    assert "#2561" in report


# --- the role relation, over a removal -------------------------------------------
#
# `main` went red at `8d0298345` from a ninth pull request and eight that had
# already landed. #3037 removed 96 g1 lookup modules; the eight verbs cited them
# in docstring roles. The path intersection was empty in both directions - the
# removal touches no verb, the verbs cite from files the removal does not open -
# and git merged the two sides with no conflict, so the first tree holding both
# the role and the deletion was `main`, where
# `tests/test_docstring_xref_roles_resolve.py` reported 44 offending docstrings.
#
# The relation these pin reads a removal rather than a change, and reads both
# trees rather than a diff, because a role only counts inside a docstring and a
# hunk cannot be parsed for that. Issues #2791, #3065.

#: The lookup module #3037 removed, standing for all 96.
_CITED_MODULE = "strands_robots/tools/g1/g1_fsm_targets.py"

#: Its dotted name, which is what a role spells.
_CITED_TARGET = "strands_robots.tools.g1.g1_fsm_targets"

#: The verb whose docstring cites it, standing for all eight.
_CITING_VERB = "strands_robots/tools/g1/g1_set_fsm.py"

#: The report heading the relation renders, used to tell "reported by this
#: relation" from "reported by the path relation on the same run".
_ROLE_HEADING = "compose to a tree carrying"


def _module_citing(target: str, *, in_docstring: bool = True) -> str:
    """Source for a verb whose module docstring - or a comment - cites ``target``."""
    citation = f"See :mod:`{target}` for the admitted set."
    head = f'"""One verb.\n\n{citation}\n"""' if in_docstring else f"# {citation}"
    return f"{head}\n\nvalue = 1\n"


def _land_the_cited_module(repo: Path) -> None:
    """Put the lookup module on ``main``, so both sides fork with it present."""
    _write(repo, _CITED_MODULE, '"""The admitted FSM targets."""\n\nTARGETS = (1, 2)\n')
    _commit(repo, "the lookup module both sides know about")


def _branch_deleting_the_cited_module(repo: Path) -> None:
    """Branch off ``main`` and remove the lookup module, as #3037 did."""
    _git(repo, "checkout", "-q", "-b", "pr")
    _git(repo, "rm", "-q", _CITED_MODULE)
    _commit(repo, "consolidate the lookup modules into the dispatcher")


def _land_on_main_citing(repo: Path, *, in_docstring: bool = True) -> None:
    """Land a verb on ``main`` whose docstring cites the module, as the eight did."""
    _git(repo, "checkout", "-q", "main")
    _write(repo, _CITING_VERB, _module_citing(_CITED_TARGET, in_docstring=in_docstring))
    _commit(repo, "port a verb that cites the lookup module")
    _git(repo, "checkout", "-q", "pr")


def test_a_branch_deleting_a_module_the_base_cites_is_flagged(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The #3037 topology: the branch removes what the base went on to cite.

    Neither side is wrong alone, and neither side's checks can see it - the role
    and the deletion are in different trees until the merge. The report has to
    name the citing file and the target, because the remedy is an edit to that
    docstring rather than a merge.
    """
    _land_the_cited_module(repo)
    _branch_deleting_the_cited_module(repo)
    _land_on_main_citing(repo)

    assert _run(repo) == 1

    report = capsys.readouterr().out
    assert _ROLE_HEADING in report
    assert _CITING_VERB in report
    assert _CITED_TARGET in report


def test_the_role_finding_shares_no_path_and_no_literal_with_the_base(repo: Path) -> None:
    """Premise: neither relation that already existed can reach this composition.

    Without this the test above would pass on a tree where the two sides happen to
    share a file, and the finding would be attributable to the path relation. The
    literal relation is empty too: a docstring role is not a quoted whole-string
    name, and the citing file is not under ``tests/``.
    """
    _land_the_cited_module(repo)
    _branch_deleting_the_cited_module(repo)
    _land_on_main_citing(repo)

    fork_point = check.merge_base("main", "HEAD", repo=repo)
    branch_paths = check.changed_paths(fork_point, "HEAD", repo=repo)
    base_paths = check.changed_paths(fork_point, "main", repo=repo)
    assert check.overlapping_paths(branch_paths, base_paths) == ()
    literals = check.module_literals(check.diff_entries(fork_point, "HEAD", repo=repo))
    assert check.named_module_overlaps(literals, base_paths) == ()
    assert _CITED_MODULE in branch_paths
    assert _CITING_VERB in base_paths


def test_the_composition_merges_without_a_text_conflict(repo: Path) -> None:
    """Non-vacuity: git has nothing to report, which is why a check is needed.

    A removal on one side and a new file on the other is the cleanest merge there
    is. The merge gate reports ``CLEAN`` and the merged tree carries a role
    naming a module that is not in it.
    """
    _land_the_cited_module(repo)
    _branch_deleting_the_cited_module(repo)
    _land_on_main_citing(repo)

    _git(repo, "merge", "--no-edit", "-q", "main")  # raises CalledProcessError on conflict
    assert not (repo / _CITED_MODULE).exists()
    assert _CITED_TARGET in (repo / _CITING_VERB).read_text(encoding="utf-8")


def test_a_branch_citing_a_module_the_base_deleted_is_flagged(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The mirror direction, because either side can land last.

    Same composition with the roles of the two branches swapped: here the branch
    writes the docstring and the base removes the module. Reading only the
    direction the incident arrived from would leave the citing author's own branch
    unguarded, which is the half that is easiest to fix.
    """
    _land_the_cited_module(repo)
    _git(repo, "checkout", "-q", "-b", "pr")
    _write(repo, _CITING_VERB, _module_citing(_CITED_TARGET))
    _commit(repo, "port a verb that cites the lookup module")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "rm", "-q", _CITED_MODULE)
    _commit(repo, "consolidate the lookup modules into the dispatcher")
    _git(repo, "checkout", "-q", "pr")

    assert _run(repo) == 1
    report = capsys.readouterr().out
    assert _ROLE_HEADING in report
    assert _CITED_TARGET in report


def test_merging_the_base_is_not_enough_and_editing_the_role_clears_it(repo: Path) -> None:
    """This relation is not self-clearing, and it must not pretend to be.

    The path relation clears on a merge because the merge is the whole remedy.
    Here the merge produces exactly the tree the report is about: a role and no
    module. The suite is red on that tree, so a check that went quiet would be
    disagreeing with the thing it predicts. Dropping the role is what clears it,
    which is what the remedy sentence asks for.
    """
    _land_the_cited_module(repo)
    _branch_deleting_the_cited_module(repo)
    _land_on_main_citing(repo)
    assert _run(repo) == 1

    _git(repo, "merge", "-q", "--no-edit", "main")
    assert _run(repo) == 1

    _write(repo, _CITING_VERB, '"""One verb."""\n\nvalue = 1\n')
    _commit(repo, "drop the role that cited the removed module")
    assert _run(repo) == 0


def test_a_removal_nobody_cites_does_not_block(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Quiet where it cannot matter: removing a module is not itself a finding.

    Consolidation is routine, and a relation that fired on any deletion would fire
    on most refactors in the tree.
    """
    _land_the_cited_module(repo)
    _branch_deleting_the_cited_module(repo)
    _git(repo, "checkout", "-q", "main")
    _write(repo, "strands_robots/unrelated.py", "value = 1\n")
    _commit(repo, "an unrelated commit on main")
    _git(repo, "checkout", "-q", "pr")

    assert _run(repo) == 0
    assert _ROLE_HEADING not in capsys.readouterr().out


def test_a_role_the_composition_keeps_does_not_block(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The other half of quiet: citing a module is the normal way to reference one.

    The base adds the same citing verb, and nothing removes the module. Every
    docstring role in the tree would be a row if the relation keyed on the role
    alone.
    """
    _land_the_cited_module(repo)
    _git(repo, "checkout", "-q", "-b", "pr")
    _edit_line(repo, _SHARED, 40, "line 40  # edited by the pull request")
    _commit(repo, "the pull request's commit")
    _land_on_main_citing(repo)

    assert _run(repo) == 0
    assert _ROLE_HEADING not in capsys.readouterr().out


def test_a_role_outside_a_docstring_is_not_this_relations_business(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The population is the grader's: docstrings, not every line mentioning a name.

    A role in a comment is a dead pointer on its own account, and it is not what
    turns the suite red. Reporting it here would block a merge for something no
    test grades, and the ``git grep`` narrowing step finds such a file - so this
    pins the parse, not the search.
    """
    _land_the_cited_module(repo)
    _branch_deleting_the_cited_module(repo)
    _land_on_main_citing(repo, in_docstring=False)

    assert _run(repo) == 0
    assert _ROLE_HEADING not in capsys.readouterr().out


def test_a_citing_file_the_branch_changes_is_left_to_the_branch_s_own_suite_run(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One tree, one report: a branch holding both halves is graded by its own suite.

    When the branch itself changes the citing file, the role and the deletion are
    already in the same tree, so ``call-test-lint`` on that head reports it. A
    second row here would spend a reader's attention describing a composition
    that is not a composition.
    """
    _land_the_cited_module(repo)
    _write(repo, _CITING_VERB, _module_citing(_CITED_TARGET))
    _commit(repo, "a verb citing the lookup module, on main before either side")
    _git(repo, "checkout", "-q", "-b", "pr")
    _git(repo, "rm", "-q", _CITED_MODULE)
    _write(repo, _CITING_VERB, _module_citing(_CITED_TARGET) + "# touched by this branch\n")
    _commit(repo, "remove the module and edit the citing verb")

    assert _run(repo) == 0
    assert _ROLE_HEADING not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(f'"""Head.\n\n:mod:`{_CITED_TARGET}`\n"""\n', {_CITED_TARGET}, id="module-docstring"),
        pytest.param(f'class C:\n    """:class:`{_CITED_TARGET}.T`"""\n', {f"{_CITED_TARGET}.T"}, id="class"),
        pytest.param(f'def f():\n    """:func:`{_CITED_TARGET}.g`"""\n', {f"{_CITED_TARGET}.g"}, id="function"),
        pytest.param(f"# :mod:`{_CITED_TARGET}`\n", set(), id="comment"),
        pytest.param(f'X = ":mod:`{_CITED_TARGET}`"\n', set(), id="runtime-string"),
        pytest.param(f'""":mod:`~{_CITED_TARGET}`"""\n', {_CITED_TARGET}, id="display-tilde"),
        pytest.param('""":mod:`.protocol`"""\n', set(), id="relative-target"),
        pytest.param('""":meth:`Cls.method`"""\n', set(), id="short-form"),
        pytest.param('"""not a docstring below"""\nif True:\n    pass\n', set(), id="no-role"),
        pytest.param("def f(:\n", set(), id="unparseable"),
    ],
)
def test_role_targets_reads_the_docstrings_a_grader_would(source: str, expected: set[str]) -> None:
    """Which spellings this relation can decide, and which it leaves to the grader.

    A wrapped or short-form target has no module path to intersect, so admitting it
    would mean guessing at a file. The grader reports both on their own account;
    this relation only claims the contiguous qualified form.
    """
    assert check.role_targets(source) == expected


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        pytest.param([_CITED_MODULE], {_CITED_TARGET}, id="module"),
        pytest.param(["strands_robots/tools/g1/__init__.py"], {"strands_robots.tools.g1"}, id="package"),
        pytest.param(["strands_robots/__init__.py"], set(), id="bare-root"),
        pytest.param(["strands_robots/one.py"], {"strands_robots.one"}, id="shallowest-resolvable"),
        pytest.param(["tests/drivers/test_x.py"], set(), id="outside-the-package"),
        pytest.param(["docs/guide.md"], set(), id="prose"),
    ],
)
def test_a_deleted_path_becomes_a_search_key_only_when_a_role_could_name_it(
    paths: list[str], expected: set[str]
) -> None:
    """The bare package root is excluded, for cost as well as for correctness.

    ``named_module_paths`` cannot resolve a one-segment name, so such a key could
    never produce a finding - and as a search key it matches every citing file in
    the tree, which is the one input that would make this relation expensive.
    """
    assert check.dotted_module_names(paths) == expected


def test_a_branch_deleting_no_module_asks_git_nothing(repo: Path) -> None:
    """The relation costs nothing on the branches that are not about a removal.

    Pinned through a revision that does not exist: reaching git at all would raise
    rather than return an empty result, so an empty key set provably short-circuits
    before the search.
    """
    assert check.files_naming("no-such-revision", (), ("*.py",), repo=repo) == ()
    assert check.orphaned_roles("no-such-revision", (), ("*.py",), repo=repo) == ()


def test_a_name_is_resolved_to_a_path_by_one_resolver() -> None:
    """Both name relations answer "which files does this name need" the same way.

    A role and a string literal are different spellings of one coupling, and the
    prefix rule is the subtle part of resolving either: importing ``a.b.c`` runs
    ``a/b/__init__.py``. Two resolvers would drift, and the drift would show up as
    a relation that misses a package-level removal.
    """
    assert "named_module_paths(" in inspect.getsource(check.orphaned_roles)
    resolved = check.named_module_paths({f"{_CITED_TARGET}.g1_fsm_target_admits"})
    assert _CITED_MODULE in resolved
    assert "strands_robots/tools/g1/__init__.py" in resolved
