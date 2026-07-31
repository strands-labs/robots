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
from pathlib import Path
from types import ModuleType

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
