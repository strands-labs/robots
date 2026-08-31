"""Contract pins for the ``changelog.d/`` news-fragment convention.

A behavioural PR records its entry as a file under ``changelog.d/`` instead of
appending to ``## [Unreleased]`` in ``CHANGELOG.md``. That is what keeps two
concurrently open PRs from conflicting on the changelog: they never touch the
same path. ``scripts/assemble_changelog.py`` folds the accumulated fragments
into the log at release time.

These checks pin the two halves of that trade:

- the convention is *declared* (``changelog.d/README.md`` exists and documents
  it), so a future directory cleanup cannot quietly drop the policy while
  leaving the tooling in place;
- assembly is *lossless and fail-safe* -- fragments are ordered
  deterministically, every body lands exactly once, a malformed or misnamed
  fragment is refused rather than skipped, and a refusal writes nothing and
  deletes nothing.

The last point is the one that matters most: a fragment that is silently
ignored is a behavioural change that never reaches the log, and a half-applied
assembly can delete a fragment whose content never landed.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "assemble_changelog.py"
_FRAGMENT_DIR = _REPO_ROOT / "changelog.d"
_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"

_VERSION_HEADING = re.compile(r"^## \[(?P<ver>\d+\.\d+\.\d+)\] - (?P<date>\d{4}-\d{2}-\d{2})\s*$")


def _load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("assemble_changelog", _SCRIPT_PATH)
    assert spec and spec.loader, f"cannot load {_SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["assemble_changelog"] = module
    spec.loader.exec_module(module)
    return module


assemble = _load_module()

_MINIMAL_LOG = """# CHANGELOG

## [Unreleased]

### Fixed: an entry that was already in the log

Body of the pre-existing entry.


## [0.4.1] - 2026-01-02

### Fixed: something released

Released body.
"""


def _fragment(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    """A fragment directory plus a changelog, isolated from the repo copy."""
    fragment_dir = tmp_path / "changelog.d"
    fragment_dir.mkdir()
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(_MINIMAL_LOG, encoding="utf-8")
    return fragment_dir, changelog


# --- the convention is declared -------------------------------------------------


def test_fragment_directory_documents_the_convention() -> None:
    """The policy lives in the repo, not only in reviewers' heads."""
    readme = _FRAGMENT_DIR / "README.md"
    assert readme.is_file(), "changelog.d/README.md must document the fragment convention"
    text = readme.read_text(encoding="utf-8")
    assert "<number>-<slug>.md" in text, "README must state the fragment naming contract"
    assert "assemble_changelog.py" in text, "README must point at the assembly step"


def test_repository_fragments_are_valid() -> None:
    """Pending fragments are checked at PR time, not at release time."""
    assert assemble.validate_fragments(_FRAGMENT_DIR) == []


# --- ordering and losslessness --------------------------------------------------


def test_fragments_are_ordered_newest_first_and_deterministically(workspace: tuple[Path, Path]) -> None:
    fragment_dir, _ = workspace
    _fragment(fragment_dir, "9-old.md", "### Fixed: old\n\nOld body.\n")
    _fragment(fragment_dir, "1200-new.md", "### Added: new\n\nNew body.\n")
    _fragment(fragment_dir, "100-middle.md", "### Changed: middle\n\nMiddle body.\n")

    names = [fragment.name for fragment in assemble.collect_fragments(fragment_dir)]
    assert names == ["1200-new.md", "100-middle.md", "9-old.md"], "numeric descending, not lexicographic"


def test_render_keeps_every_body_exactly_once(workspace: tuple[Path, Path]) -> None:
    fragment_dir, _ = workspace
    _fragment(fragment_dir, "10-one.md", "### Fixed: one\n\nFirst body.\n")
    _fragment(fragment_dir, "11-two.md", "### Added: two\n\nSecond body.\n")

    rendered = assemble.render(assemble.collect_fragments(fragment_dir))
    assert rendered.count("First body.") == 1
    assert rendered.count("Second body.") == 1
    assert rendered.count("### ") == 2


def test_readme_is_not_treated_as_a_fragment(workspace: tuple[Path, Path]) -> None:
    fragment_dir, _ = workspace
    _fragment(fragment_dir, "README.md", "# changelog.d\n\nHow to add a fragment.\n")
    assert assemble.collect_fragments(fragment_dir) == []
    assert assemble.validate_fragments(fragment_dir) == []


# --- apply: the happy path ------------------------------------------------------


def test_apply_folds_fragments_in_and_consumes_them(workspace: tuple[Path, Path]) -> None:
    fragment_dir, changelog = workspace
    first = _fragment(fragment_dir, "20-newer.md", "### Fixed: newer thing\n\nNewer body.\n")
    second = _fragment(fragment_dir, "15-older.md", "### Added: older thing\n\nOlder body.\n")

    consumed = assemble.apply(fragment_dir, changelog)

    assert [fragment.name for fragment in consumed] == ["20-newer.md", "15-older.md"]
    assert not first.exists() and not second.exists(), "consumed fragments must be removed"

    text = changelog.read_text(encoding="utf-8")
    assert "Newer body." in text and "Older body." in text
    assert "an entry that was already in the log" in text, "pre-existing entries must survive"
    assert text.index("Newer body.") < text.index("Older body.") < text.index("Body of the pre-existing entry.")
    assert text.index("## [Unreleased]") < text.index("Newer body.")
    assert text.endswith("\n")


def test_apply_with_no_fragments_leaves_the_log_byte_identical(workspace: tuple[Path, Path]) -> None:
    fragment_dir, changelog = workspace
    before = changelog.read_text(encoding="utf-8")
    assert assemble.apply(fragment_dir, changelog) == []
    assert changelog.read_text(encoding="utf-8") == before


def test_apply_into_an_empty_unreleased_section(tmp_path: Path) -> None:
    """The seam has the same shape whether or not the section already had entries."""
    fragment_dir = tmp_path / "changelog.d"
    fragment_dir.mkdir()
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# CHANGELOG\n\n## [Unreleased]\n\n## [0.4.1] - 2026-01-02\n\nBody.\n", encoding="utf-8")
    _fragment(fragment_dir, "30-first.md", "### Fixed: the first entry\n\nFirst body.\n")

    assemble.apply(fragment_dir, changelog)
    text = changelog.read_text(encoding="utf-8")

    assert "## [Unreleased]\n\n### Fixed: the first entry" in text
    assert text.count("## [Unreleased]") == 1


def test_assembled_log_still_satisfies_the_structural_contract(tmp_path: Path) -> None:
    """Assembly must not disturb what tests/test_changelog_format.py pins."""
    fragment_dir = tmp_path / "changelog.d"
    fragment_dir.mkdir()
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(_CHANGELOG.read_text(encoding="utf-8"), encoding="utf-8")
    _fragment(fragment_dir, "40-structural.md", "### Fixed: a freshly assembled entry\n\nBody.\n")

    assemble.apply(fragment_dir, changelog)

    h2 = [line for line in changelog.read_text(encoding="utf-8").splitlines() if line.startswith("## ")]
    assert h2[0].strip() == "## [Unreleased]"
    assert sum(1 for line in h2 if line.strip() == "## [Unreleased]") == 1

    versions = []
    for line in h2[1:]:
        match = _VERSION_HEADING.match(line)
        assert match, f"assembly forged a malformed version heading: {line!r}"
        _dt.date.fromisoformat(match.group("date"))
        versions.append(tuple(int(part) for part in match.group("ver").split(".")))
    assert versions == sorted(versions, reverse=True)


# --- refusals: nothing written, nothing deleted ---------------------------------


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("50-empty.md", "\n\n", "empty"),
        ("51-no-heading.md", "Just prose, no heading.\n", "level-3 entry heading"),
        ("52-deeper-heading-first.md", "#### Fixed: too deep\n\nBody.\n", "level-3 entry heading"),
        ("54-two-entries.md", "### Fixed: one\n\nA.\n\n### Added: two\n\nB.\n", "exactly one"),
        ("55-version-heading.md", "### Fixed: one\n\n## [9.9.9] - 2026-01-01\n\nA.\n", "level-2 heading"),
    ],
)
def test_invalid_fragment_is_reported(workspace: tuple[Path, Path], name: str, text: str, expected: str) -> None:
    fragment_dir, _ = workspace
    _fragment(fragment_dir, name, text)
    problems = assemble.validate_fragments(fragment_dir)
    assert problems, f"{name} must be refused"
    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize("category", ["Fixed", "Added", "Docs", "Quality", "Security", "Internal Refactor"])
def test_the_categories_already_in_the_log_are_accepted(workspace: tuple[Path, Path], category: str) -> None:
    """Structure is policed, taxonomy is not.

    ``CHANGELOG.md`` carries a dozen entry categories in practice (``Docs``,
    ``Quality``, ``Internal Refactor``, ...). A fragment validator that accepted
    only the Keep a Changelog six would legislate a new content policy, and
    would refuse entries in the style the log already uses.
    """
    fragment_dir, _ = workspace
    _fragment(fragment_dir, "56-category.md", f"### {category}: a summary\n\nBody.\n")
    assert assemble.validate_fragments(fragment_dir) == []


def test_misnamed_fragment_is_an_error_not_a_skip(workspace: tuple[Path, Path]) -> None:
    """A file that does not match the naming contract must never be ignored."""
    fragment_dir, _ = workspace
    _fragment(fragment_dir, "no-leading-number.md", "### Fixed: something\n\nBody.\n")
    problems = assemble.validate_fragments(fragment_dir)
    assert any("not a valid fragment name" in problem for problem in problems), problems


def test_stray_non_markdown_file_is_an_error_not_a_skip(workspace: tuple[Path, Path]) -> None:
    fragment_dir, _ = workspace
    _fragment(fragment_dir, "60-notes.txt", "### Fixed: something\n\nBody.\n")
    problems = assemble.validate_fragments(fragment_dir)
    assert any("not a valid fragment name" in problem for problem in problems), problems


def test_apply_refuses_invalid_fragment_without_touching_anything(workspace: tuple[Path, Path]) -> None:
    fragment_dir, changelog = workspace
    good = _fragment(fragment_dir, "70-good.md", "### Fixed: a good entry\n\nGood body.\n")
    bad = _fragment(fragment_dir, "71-bad.md", "no heading at all\n")
    before = changelog.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to assemble"):
        assemble.apply(fragment_dir, changelog)

    assert changelog.read_text(encoding="utf-8") == before, "a refusal must not write the log"
    assert good.exists() and bad.exists(), "a refusal must not delete fragments"


def test_apply_refuses_a_log_without_an_unreleased_anchor(tmp_path: Path) -> None:
    fragment_dir = tmp_path / "changelog.d"
    fragment_dir.mkdir()
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# CHANGELOG\n\n## [0.4.1] - 2026-01-02\n\nBody.\n", encoding="utf-8")
    pending = _fragment(fragment_dir, "80-pending.md", "### Fixed: pending\n\nBody.\n")
    before = changelog.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        assemble.apply(fragment_dir, changelog)

    assert changelog.read_text(encoding="utf-8") == before
    assert pending.exists(), "the fragment must survive a failed assembly"


# --- CLI ------------------------------------------------------------------------


def test_check_mode_reports_and_writes_nothing(workspace: tuple[Path, Path], capsys: pytest.CaptureFixture) -> None:
    fragment_dir, changelog = workspace
    kept = _fragment(fragment_dir, "90-kept.md", "### Fixed: kept\n\nBody.\n")
    before = changelog.read_text(encoding="utf-8")

    code = assemble.main(["--check", "--fragment-dir", str(fragment_dir), "--changelog", str(changelog)])

    assert code == 0
    assert "1 pending" in capsys.readouterr().out
    assert kept.exists()
    assert changelog.read_text(encoding="utf-8") == before


def test_check_mode_exits_non_zero_on_an_invalid_fragment(workspace: tuple[Path, Path]) -> None:
    fragment_dir, changelog = workspace
    _fragment(fragment_dir, "91-bad.md", "no heading\n")
    code = assemble.main(["--check", "--fragment-dir", str(fragment_dir), "--changelog", str(changelog)])
    assert code == 1


# --- fragment numbers name the change that landed them --------------------------

#: Fragment numbers at or above this are treated as pre-PR placeholders and
#: refused. The repository's PR/issue counter is a single monotonic sequence
#: (highest so far: the 2900s), so this floor leaves several years of headroom
#: while still catching the ``999x`` names a branch reaches for before a PR
#: number exists.
_PLACEHOLDER_FLOOR = 9000


def _real_fragment_numbers() -> list[tuple[int, str]]:
    """Every ``(number, name)`` pair in the repository's own fragment directory."""
    pairs = []
    for path in sorted(_FRAGMENT_DIR.iterdir()):
        if path.name in assemble.RESERVED_NAMES or path.is_dir():
            continue
        match = assemble.FRAGMENT_NAME.match(path.name)
        if match is not None:
            pairs.append((int(match.group("number")), path.name))
    return pairs


def test_no_fragment_uses_a_reserved_placeholder_number() -> None:
    """A fragment is named for the PR (or issue) that landed it, per README.

    A placeholder like ``9999-`` is a natural thing to write while a branch has
    no PR number yet, and renaming it before merge is easy to forget: nothing
    downstream fails, so the fragment survives review and merges under the
    placeholder. Three did, from two separate PRs. The cost only becomes visible
    at release time, and by then ``--apply`` has deleted the fragment, so the
    number cannot be recovered from the log it produced.

    Two things are lost. The number is the only pointer from a release-note
    entry back to the change that made it, so a placeholder makes an entry
    untraceable. And because assembly orders by *descending* number, the
    placeholder sorts above every real entry - see the companion cell below.
    """
    offenders = [name for number, name in _real_fragment_numbers() if number >= _PLACEHOLDER_FLOOR]
    assert not offenders, (
        f"fragment(s) named with a reserved placeholder number (>= {_PLACEHOLDER_FLOOR}): "
        f"{offenders}. Rename each to the PR or issue number that carries it, so the "
        "release-note entry points back at the change and sorts in the right place."
    )


def test_a_placeholder_number_sorts_above_every_real_entry(workspace: tuple[Path, Path]) -> None:
    """The ordering consequence, pinned so the rule above keeps its reason.

    ``collect_fragments`` orders by descending number so the assembled section
    reads newest-first. A placeholder number is not a position in that sequence,
    so it does not read as "newest" - it reads as *first*, ahead of everything,
    however old the change behind it is.
    """
    fragment_dir, _ = workspace
    _fragment(fragment_dir, "2953-a-real-recent-change.md", "### Fixed: recent\n\nBody.\n")
    _fragment(fragment_dir, "12-a-genuinely-old-change.md", "### Fixed: old\n\nBody.\n")
    _fragment(fragment_dir, "9999-a-placeholder.md", "### Fixed: placeholder\n\nBody.\n")

    order = [fragment.name for fragment in assemble.collect_fragments(fragment_dir)]

    assert order[0] == "9999-a-placeholder.md", (
        "expected the placeholder to sort first - if this no longer holds, the "
        "ordering rationale in the companion cell needs rewriting"
    )
    assert order == [
        "9999-a-placeholder.md",
        "2953-a-real-recent-change.md",
        "12-a-genuinely-old-change.md",
    ]
