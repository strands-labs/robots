"""CHANGELOG integrity guard (Keep a Changelog conventions).

`CHANGELOG.md` is the human-facing record of behavioural changes and the source
material for release notes. When a tag is cut, the accumulated `## [Unreleased]`
section must be collapsed into a dated `## [x.y.z] - YYYY-MM-DD` heading and a
fresh empty `[Unreleased]` opened for the next wave. Skipping that collapse
leaves shipped content mislabelled as unreleased -- the published tag then has
no dated section in the log.

These checks pin the structural contract so the collapse is not forgotten:

- exactly one `## [Unreleased]` heading, and it comes first;
- every other version heading is `## [MAJOR.MINOR.PATCH] - YYYY-MM-DD` with a
  real ISO date;
- version sections appear in strictly descending semantic-version order with no
  duplicates;
- every published release (``PUBLISHED_RELEASES``) has its own dated section --
  i.e. its content is not still sitting under `[Unreleased]`.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

# Every git tag that has been published MUST have a dated CHANGELOG section.
# Append the version here whenever a new tag is cut (and collapse `[Unreleased]`
# into the matching dated heading in the same change).
PUBLISHED_RELEASES = frozenset({"0.4.0", "0.4.1"})

_CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
_VERSION_HEADING = re.compile(r"^## \[(?P<ver>\d+\.\d+\.\d+)\] - (?P<date>\d{4}-\d{2}-\d{2})\s*$")
_UNRELEASED_HEADING = re.compile(r"^## \[Unreleased\]\s*$")
_ANY_H2 = re.compile(r"^## ")


def _h2_lines() -> list[str]:
    text = _CHANGELOG.read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if _ANY_H2.match(ln)]


def test_changelog_exists() -> None:
    assert _CHANGELOG.is_file(), f"CHANGELOG.md not found at {_CHANGELOG}"


def test_single_unreleased_heading_first() -> None:
    h2 = _h2_lines()
    unreleased = [ln for ln in h2 if _UNRELEASED_HEADING.match(ln)]
    assert len(unreleased) == 1, f"expected exactly one '## [Unreleased]', found {len(unreleased)}"
    assert _UNRELEASED_HEADING.match(h2[0]), f"first level-2 heading must be '## [Unreleased]', got {h2[0]!r}"


def test_version_headings_well_formed_and_descending() -> None:
    versions: list[tuple[int, int, int]] = []
    for ln in _h2_lines():
        if _UNRELEASED_HEADING.match(ln):
            continue
        m = _VERSION_HEADING.match(ln)
        assert m, f"malformed version heading (want '## [x.y.z] - YYYY-MM-DD'): {ln!r}"
        # date must be a real calendar date
        _dt.date.fromisoformat(m.group("date"))
        versions.append(tuple(int(p) for p in m.group("ver").split(".")))  # type: ignore[arg-type]

    assert versions, "no dated version sections found"
    assert len(versions) == len(set(versions)), f"duplicate version heading(s): {versions}"
    assert versions == sorted(versions, reverse=True), f"version sections must be in descending order, got {versions}"


def test_published_releases_have_dated_sections() -> None:
    """Each published tag must be collapsed into its own dated section.

    Fails while a shipped release's content is still under `[Unreleased]`.
    """
    documented = {m.group("ver") for ln in _h2_lines() if (m := _VERSION_HEADING.match(ln))}
    missing = PUBLISHED_RELEASES - documented
    assert not missing, (
        f"published release(s) {sorted(missing)} have no dated CHANGELOG section - "
        "collapse '[Unreleased]' into the matching '## [x.y.z] - YYYY-MM-DD' heading"
    )
