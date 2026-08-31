"""Repo hygiene: block host-specific absolute paths from being committed.

History: PR #85 shipped a hardcoded ``/Users/cagatay/robots/...`` in
``tests/simulation/mujoco/test_agenttool_contract.py`` that passed on the
author's laptop, got committed, and was only caught by CI because CI happens
to not live at that path.

This test is a cheap regex sweep over every top-level directory of the
repository that ships Python - ``strands_robots/``, ``tests/``, ``tests_integ/``,
``examples/`` and ``scripts/`` today - that fails fast if anyone re-introduces a
``/Users/<name>``, ``/home/<name>`` or ``C:\\Users\\<name>`` string. Prefer
module-relative paths, ``pathlib.Path`` + ``__file__``, ``importlib.resources``,
or fixtures.

The area list is derived rather than written down. It was a hardcoded three-tuple
that named the two directories the PR #85 defect happened to land in, and
``examples/`` and ``scripts/`` shipped 109 files outside it. ``examples/`` is the
worst place to miss: it is the code a reader copies, so a host path there is the
same defect propagated rather than merely committed.

Allowlist patterns live below - keep it narrow.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Areas that must be swept however the tree grows. A directory added later that
# ships Python is picked up by _scanned_areas() without an edit here; this set
# only refuses the reverse, an area silently dropping out of the sweep.
_REQUIRED_AREAS = frozenset({"strands_robots", "tests", "tests_integ", "examples", "scripts"})

# Patterns that indicate a hardcoded host-specific user path.
#
# None of these requires a trailing separator. A home directory is host-specific
# whether or not a further segment follows it, and requiring the separator let a
# *terminal* literal through untouched: ``HOME = "/Users/alice"`` on one line
# with ``Path(HOME) / "robots"`` on the next is the PR #85 defect split across
# two statements, and it swept clean. The user segment is what makes the path
# host-specific, so the pattern ends where that segment does.
HOST_PATH_PATTERNS = [
    # POSIX home directories with a specific user segment
    re.compile(r"/Users/[A-Za-z0-9._-]+"),
    re.compile(r"/home/[A-Za-z0-9._-]+"),
    # Windows user profile, as it appears source-escaped and raw
    re.compile(r"[A-Za-z]:\\\\Users\\\\[A-Za-z0-9._-]+"),
    re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+"),
]

# Explicit allowlist - files or string occurrences that are ABOUT these patterns
# (documentation, validators themselves, regex sources).
ALLOWED_FILES = {
    # This test itself defines the patterns above.
    "tests/test_no_host_paths.py",
    # Path validation logic *contains* Windows system paths as blocklist entries;
    # those are C:\Windows\, C:\Program Files\ - not user profiles.
    "strands_robots/tools/_path_validation.py",
    "tests/tools/test_path_validation.py",
    # Container volume-safety tests contain protected host paths as test data
    # (the test asserts that the production code REJECTS these paths).
    "tests/tools/test_gr00t_container_hardening.py",
    # Protected host paths (incl. //home/<u>/.aws as a
    # leading-double-slash bypass vector) as attack input; each assertion proves
    # the production guard REJECTS the path rather than using it.
    "tests/tools/test_gr00t_pentest_regressions.py",
}


def _is_repo_owned_python_area(entry: Path) -> bool:
    """Whether ``entry`` is a top-level directory whose Python this gate owns.

    One owner for the three conditions, because "an area the sweep should read"
    was derived twice and the two copies skewed on the third: the reach cell
    below built its own expectation and omitted the virtualenv marker, so a
    checkout carrying ``venv/`` failed with a message naming the single directory
    the sweep deliberately refuses to read.

    Two kinds of directory are excluded, and both are the cost of deriving the
    area list rather than writing it down. Dot-directories hold caches and
    tooling, not committed source. A virtual environment holds third-party code
    full of the packager's own home directory, so reading one would fail the gate
    for a reason the author cannot fix; it is recognised by its PEP 405
    ``pyvenv.cfg`` marker rather than by name, since ``.venv`` is a convention
    and ``venv/`` is just as common.

    Args:
        entry: Path to classify. A non-directory is never an area.

    Returns:
        True when the directory ships Python that a committed-path gate owns.
    """
    return (
        entry.is_dir()
        and not entry.name.startswith(".")
        and not (entry / "pyvenv.cfg").exists()
        and any(entry.rglob("*.py"))
    )


def _scanned_areas(root: Path = REPO_ROOT) -> tuple[str, ...]:
    """Return every top-level directory of ``root`` whose Python this gate owns.

    Deriving the list means a directory added later is swept on arrival. A
    hardcoded tuple equal to today's set fires on nothing when the tree grows,
    which is the silent hole this avoids: the tuple this replaced named three
    areas and the repository ships five.

    Args:
        root: Repository root to enumerate. Defaults to this repository.

    Returns:
        Directory names, sorted, per :func:`_is_repo_owned_python_area`.
    """
    return tuple(entry.name for entry in sorted(root.iterdir()) if _is_repo_owned_python_area(entry))


def _iter_source_files(root: Path = REPO_ROOT) -> list[Path]:
    """Return every ``.py`` file the sweep reads, under every area that ships one.

    Args:
        root: Repository root to walk. Defaults to this repository.

    Returns:
        Paths, in area order, excluding bytecode caches and virtualenvs.
    """
    files: list[Path] = []
    for d in _scanned_areas(root):
        for p in (root / d).rglob("*.py"):
            # Skip bytecode caches and anything inside .venv / build dirs
            if "__pycache__" in p.parts or ".venv" in p.parts:
                continue
            files.append(p)
    return files


def _areas_missed(root: Path = REPO_ROOT) -> list[str]:
    """Return every area of ``root`` that this gate owns and the sweep never reads.

    The sweep's reach, which the pattern cells cannot speak to: they grade what
    the patterns match, never where the patterns are applied. Both halves resolve
    an area through :func:`_is_repo_owned_python_area`, so what stays gradeable
    here is the file walk - whether :func:`_iter_source_files`'s own filters drop
    a whole area on the way - while the derivation is graded on constructed trees,
    where an area can be added or made third-party on purpose.

    Args:
        root: Repository root to compare. Defaults to this repository.

    Returns:
        Area names, sorted. Empty when the sweep reads every area it owns.
    """
    swept = {path.relative_to(root).parts[0] for path in _iter_source_files(root)}
    owned = {entry.name for entry in root.iterdir() if _is_repo_owned_python_area(entry)}
    return sorted(owned - swept)


def _offenders(root: Path = REPO_ROOT) -> list[tuple[str, int, str]]:
    """Return every host-specific path literal committed under ``root``.

    Extracted from the sweep so the rule can be graded against a constructed
    tree: the shipped corpus is clean, so a cell that only walks it cannot show
    that a host path in a newly-swept area would be caught.

    Args:
        root: Repository root to sweep. Defaults to this repository.

    Returns:
        ``(relative path, 1-based line number, trimmed line)`` per hit.
    """
    offenders: list[tuple[str, int, str]] = []
    for path in _iter_source_files(root):
        rel = path.relative_to(root).as_posix()
        if rel in ALLOWED_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(pat.search(line) for pat in HOST_PATH_PATTERNS):
                offenders.append((rel, lineno, line.strip()[:120]))
    return offenders


# This sweep walks a few hundred small .py files and completes in well under a
# second. Its only failure mode under the global ``--timeout=120`` budget is a
# transient runner I/O stall on ``Path.read_text`` - an environmental hiccup,
# not an algorithmic hang. With the suite running fail-fast (``-x``), one such
# stall aborts the entire job and red-flags otherwise-green PRs. Disable the
# per-test timeout here (``timeout(0)``) so this deterministic hygiene check is
# never governed by the wall-clock budget; the strict 120s budget still
# protects every other test from genuine hangs.
@pytest.mark.timeout(0)
def test_no_host_specific_absolute_paths() -> None:
    """Fail if any .py file contains ``/Users/<name>`` or ``/home/<name>``.

    If you need a path in a test, use module-relative resolution:

        Path(__file__).parent / "fixture.json"

    or the existing module constants:

        from strands_robots.simulation.mujoco import simulation
        simulation._TOOL_SPEC_PATH
    """
    offenders = _offenders()

    if offenders:
        msg = ["Host-specific absolute paths detected (use Path(__file__) or fixtures instead):"]
        for rel, lineno, snippet in offenders:
            msg.append(f"  {rel}:{lineno}: {snippet}")
        raise AssertionError("\n".join(msg))


def test_host_path_sweep_disables_global_timeout() -> None:
    """Guard the flake fix: the sweep must opt out of the global per-test timeout.

    ``test_no_host_specific_absolute_paths`` is a deterministic, sub-second regex
    sweep whose only way to exceed the global ``--timeout=120`` budget is a
    transient runner I/O stall. Under fail-fast (``-x``), one such stall aborts
    the whole suite. We pin ``@pytest.mark.timeout(0)`` so the wall-clock budget
    cannot govern it. This regression asserts that opt-out stays in place; it
    fails if the marker is dropped or set to a finite budget.
    """
    pytestmark = getattr(test_no_host_specific_absolute_paths, "pytestmark", [])
    marks = [m for m in pytestmark if m.name == "timeout"]
    assert marks, "expected a @pytest.mark.timeout marker on the host-path sweep"
    assert marks[0].args == (0,), f"expected timeout(0) to disable the budget, got {marks[0].args!r}"


# Source lines the sweep must flag. Each host path appears twice: once terminal
# and once with a further segment. The terminal form is the escape this pins --
# it carried no trailing separator, so the patterns did not match it -- and the
# with-segment form is kept beside it so a future re-narrowing of the patterns
# fails here instead of quietly restoring the gap. The Windows entries appear in
# both the shapes real source takes: backslash-escaped, and raw.
HOST_PATH_LINES = [
    'HOME = "/Users/cagatay"',
    'HOME = "/Users/cagatay/robots/policy.pt"',
    "root = '/home/cagatay'",
    "root = '/home/cagatay/datasets'",
    r'win = "C:\\Users\\cagatay"',
    r'win = "C:\\Users\\cagatay\\robots"',
    r'win = r"C:\Users\cagatay"',
    r'win = r"C:\Users\cagatay\robots"',
]

# Source lines the sweep must leave alone. These pin the boundary the patterns
# must not cross now that a trailing separator is no longer required: the user
# segment is mandatory, and ``/home`` is not a prefix match for ``/homebrew``.
NON_HOST_PATH_LINES = [
    'anchor = "/Users/"',
    'anchor = "/home/"',
    'brew = "/homebrew/bin/mjpython"',
    'shared = "/usr/local/share/robots"',
    'fixture = str(tmp_path / "calibration" / "so101.json")',
]


@pytest.mark.parametrize("line", HOST_PATH_LINES)
def test_a_host_path_is_flagged_with_or_without_a_trailing_segment(line: str) -> None:
    """A home directory is host-specific whether or not a segment follows it.

    Every pattern required a trailing separator, so a literal that *ended* at
    the user segment was not host-specific as far as the gate was concerned.
    That splits the PR #85 defect across two statements and ships it:

        HOME = "/Users/cagatay"        # swept clean
        model = Path(HOME) / "robots" / "policy.pt"

    which fails on every machine that is not the author's exactly as the
    single-line form does.
    """
    assert any(pat.search(line) for pat in HOST_PATH_PATTERNS), f"host-specific path not detected: {line!r}"


@pytest.mark.parametrize("line", NON_HOST_PATH_LINES)
def test_a_path_without_a_user_segment_is_not_flagged(line: str) -> None:
    """Dropping the trailing separator must not widen the gate past a user name.

    ``/Users/`` with nothing after it names no host, and ``/homebrew`` is not a
    home directory. Without these the patterns could be "fixed" into matching
    the prefix alone, which flags portable paths and trains the next author to
    reach for the allowlist.
    """
    assert not any(pat.search(line) for pat in HOST_PATH_PATTERNS), (
        f"portable path wrongly flagged as host-specific: {line!r}"
    )


class TestEveryAreaThatShipsPythonIsSwept:
    """The sweep's reach, which the pattern cells cannot speak to.

    ``HOST_PATH_LINES`` grades what the patterns match. Nothing graded *where*
    they are applied, so a hardcoded area tuple could name any subset of the
    tree and every pattern cell would still pass.
    """

    def test_no_area_that_ships_python_is_left_out(self) -> None:
        """A directory carrying this repository's ``.py`` files must be reached.

        This is the regression: the tuple this replaced named
        ``strands_robots``, ``tests`` and ``tests_integ``, so ``examples/`` and
        ``scripts/`` were outside the gate entirely.

        The expectation resolves an area the same way the sweep does, which is
        deliberate: deriving it a second time here is what let the two skew on
        the virtualenv marker, and a gate that fails on the author's own checkout
        layout is worse than one that grades a narrower claim. The claim left is
        the file walk, and the derivation is graded on constructed trees below.
        """
        owned = {entry.name for entry in REPO_ROOT.iterdir() if _is_repo_owned_python_area(entry)}
        assert owned, "no top-level directory ships Python; the derivation has gone blind"
        missing = _areas_missed()
        assert not missing, f"these areas ship Python and the sweep never reads them: {missing}"

    def test_every_area_derivation_resolves_through_the_one_predicate(self) -> None:
        """No second copy of the ownership rule may exist in this module.

        The skew this replaced was invisible on a checkout carrying no
        virtualenv, so no cell that walks the shipped repository can catch its
        return: re-deriving the rule inline here passes on this machine and
        fails on the contributor's. What does catch it is the shape - every
        place that turns a directory listing into a set of areas has to ask
        :func:`_is_repo_owned_python_area` rather than spelling the rule out
        again and drifting from the sweep.
        """
        module = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        derivations = []
        for node in ast.walk(module):
            if not isinstance(node, ast.SetComp | ast.ListComp | ast.GeneratorExp):
                continue
            for generator in node.generators:
                iterable = generator.iter
                # sorted(root.iterdir()) and root.iterdir() are the same listing.
                if isinstance(iterable, ast.Call) and iterable.args:
                    iterable = iterable.args[0]
                if not (
                    isinstance(iterable, ast.Call)
                    and isinstance(iterable.func, ast.Attribute)
                    and iterable.func.attr == "iterdir"
                ):
                    continue
                conditions = " ".join(ast.unparse(test) for test in generator.ifs)
                derivations.append((node.lineno, conditions))

        assert len(derivations) >= 2, f"the derivation scan has gone blind: {derivations}"
        rogue = [line for line, conditions in derivations if "_is_repo_owned_python_area" not in conditions]
        assert not rogue, f"these area derivations do not resolve through the one predicate: lines {rogue}"

    def test_the_required_areas_are_all_present_and_ship_python(self) -> None:
        """Non-vacuity for the floor: each named area exists and carries Python.

        Without this, ``_REQUIRED_AREAS`` could name a directory that has been
        renamed away and the subset check below would be asserting nothing.
        """
        for area in sorted(_REQUIRED_AREAS):
            path = REPO_ROOT / area
            assert path.is_dir(), f"{area} is named in _REQUIRED_AREAS and is not a directory"
            assert any(path.rglob("*.py")), f"{area} is named in _REQUIRED_AREAS and ships no Python"

    def test_an_area_cannot_silently_drop_out_of_the_sweep(self) -> None:
        """The floor's own claim: every required area is reached today."""
        reached = set(_scanned_areas())
        assert _REQUIRED_AREAS <= reached, f"these areas dropped out of the sweep: {sorted(_REQUIRED_AREAS - reached)}"


class TestAHostPathInANewlySweptAreaIsCaught:
    """Compose the two halves on a constructed tree.

    The shipped corpus is clean, so walking it cannot show that a host path in
    ``examples/`` would be reported - only that none is there today. These build
    a repository shaped like this one and put the literal in the area that was
    outside the old tuple.
    """

    @staticmethod
    def _tree(tmp_path: Path) -> Path:
        """Build a miniature repository with one host path, under ``examples/``."""
        (tmp_path / "strands_robots").mkdir()
        (tmp_path / "strands_robots" / "ok.py").write_text(
            'CONFIG = Path(__file__).parent / "config.json"' + "\n", encoding="utf-8"
        )
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "tool.py").write_text('SHARED = "/usr/local/share/robots"' + "\n", encoding="utf-8")
        (tmp_path / "examples").mkdir()
        (tmp_path / "examples" / "demo.py").write_text(
            'HOME = "/Users/cagatay"' + "\n" + 'model = Path(HOME) / "robots" / "policy.pt"' + "\n",
            encoding="utf-8",
        )
        return tmp_path

    def test_the_literal_under_examples_is_reported(self, tmp_path: Path) -> None:
        """The offender list names the example, its line and the line's text."""
        offenders = _offenders(self._tree(tmp_path))
        assert [(rel, lineno) for rel, lineno, _ in offenders] == [("examples/demo.py", 1)]
        assert '"/Users/cagatay"' in offenders[0][2]

    def test_the_portable_siblings_are_left_alone(self, tmp_path: Path) -> None:
        """Reaching further areas must not start flagging portable paths.

        The tree carries a module-relative path in ``strands_robots/`` and a
        shared system path in ``scripts/``; widening the sweep is only useful if
        neither is reported.
        """
        reported = {rel for rel, _, _ in _offenders(self._tree(tmp_path))}
        unexpected = sorted(reported - {"examples/demo.py"})
        assert not unexpected, f"portable paths wrongly reported: {unexpected}"

    def test_an_area_added_later_is_swept_on_arrival(self, tmp_path: Path) -> None:
        """A directory nobody has written down yet is graded the day it lands.

        This is what deriving the list buys over a tuple equal to today's set:
        the tuple would report nothing here.
        """
        tree = self._tree(tmp_path)
        (tree / "benchmarks").mkdir()
        (tree / "benchmarks" / "run.py").write_text("root = '/home/cagatay/datasets'" + "\n", encoding="utf-8")
        assert "benchmarks" in _scanned_areas(tree)
        assert ("benchmarks/run.py", 1) in [(rel, lineno) for rel, lineno, _ in _offenders(tree)]

    def test_a_virtualenv_in_the_checkout_is_not_swept(self, tmp_path: Path) -> None:
        """Deriving the areas must not start sweeping third-party code.

        A hardcoded tuple could not reach a virtual environment; a derived one
        can, and site-packages is full of the packager's own home directory. The
        gate would then fail for a reason the author cannot fix. The PEP 405
        marker is the test rather than the name, because ``venv/`` is as common
        as ``.venv/`` and only the latter is a dot-directory.
        """
        tree = self._tree(tmp_path)
        venv = tree / "venv"
        (venv / "lib").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = /usr/bin" + "\n", encoding="utf-8")
        (venv / "lib" / "third_party.py").write_text('CACHE = "/home/packager/.cache"' + "\n", encoding="utf-8")
        assert "venv" not in _scanned_areas(tree)
        reported = {rel for rel, _, _ in _offenders(tree)}
        assert reported == {"examples/demo.py"}, f"the virtualenv was swept: {sorted(reported)}"

    def test_the_reach_expectation_skips_a_virtualenv_too(self, tmp_path: Path) -> None:
        """The reach comparison must apply the same marker filter as the sweep.

        The cell above grades that the sweep leaves a virtualenv alone. This
        grades the other half, which used to derive its own expectation and omit
        the PEP 405 check, so the two disagreed on exactly this layout: on any
        checkout carrying a non-dot virtualenv - ``python -m venv venv``, which
        this module's own docstring calls as common as ``.venv/`` - the reach cell
        failed with ``['venv']`` and instructed the developer to sweep the one
        directory the module refuses to sweep. That turns the pre-push gate red
        for a reason the author cannot fix and points at their environment rather
        than at their diff, and a fresh CI checkout never sees it.
        """
        tree = self._tree(tmp_path)
        venv = tree / "venv"
        (venv / "lib").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = /usr/bin" + "\n", encoding="utf-8")
        (venv / "lib" / "third_party.py").write_text('CACHE = "/home/packager/.cache"' + "\n", encoding="utf-8")

        # The virtualenv ships Python and is not owned; the sibling areas are.
        assert not _is_repo_owned_python_area(venv)
        assert _is_repo_owned_python_area(tree / "examples")
        assert _areas_missed(tree) == []

    def test_an_area_the_walk_drops_is_reported(self, tmp_path: Path) -> None:
        """Non-vacuity for the reach check: a real gap is still named.

        One predicate now answers both halves, so the classification agrees by
        construction and cannot be what the check catches. What is left is the
        file walk, and that has to be shown to still be graded: here an area's
        only Python sits under a nested ``.venv/``, so the area is owned (a
        ``.py`` is there) and every file in it is dropped on the way. Without a
        case like this, ``_areas_missed`` could answer "no gap" unconditionally
        and no cell in this module would notice.
        """
        tree = self._tree(tmp_path)
        nested = tree / "stale" / ".venv" / "lib"
        nested.mkdir(parents=True)
        (nested / "vendored.py").write_text("value = 1" + "\n", encoding="utf-8")
        assert "stale" in _scanned_areas(tree), "the area is not owned, so any gap is not the walk's"
        assert _areas_missed(tree) == ["stale"]

    def test_a_bytecode_cache_is_not_swept(self, tmp_path: Path) -> None:
        """The cache filter survives the rewrite; a stale .pyc names no author."""
        tree = self._tree(tmp_path)
        cache = tree / "examples" / "__pycache__"
        cache.mkdir()
        (cache / "stale.py").write_text('HOME = "/Users/someone"' + "\n", encoding="utf-8")
        assert not any("__pycache__" in path.parts for path in _iter_source_files(tree))
