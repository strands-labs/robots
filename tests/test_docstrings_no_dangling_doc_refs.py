"""Guard: published ``strands_robots`` source must be self-contained.

Docstrings and comments in the shipped package must not point readers at
documents that are not part of the repository, nor at line-number
self-references that rot the moment a file is edited.

Two dangling-reference classes are pinned here:

1. References to an internal design memo
   (``reports/STREAMING_DATA_LOOP_DEEP_DIVE.md``) that is not shipped in the
   distribution - every such pointer is a dead end for a reader.
2. ``~L<line>`` self-references, which silently drift out of date as soon as
   the surrounding file changes.

Both fail loudly here so they cannot creep back into the package.
"""

from __future__ import annotations

import re
from pathlib import Path

import strands_robots

_PKG_ROOT = Path(strands_robots.__file__).resolve().parent

# The unpublished internal memo earlier docstrings pointed at.
_UNPUBLISHED_MEMO = "STREAMING_DATA_LOOP_DEEP_DIVE"
# "~L1234"-style pointers into a file break the instant lines shift.
_ROTTING_LINE_REF = re.compile(r"~L\d+")


def _package_sources() -> list[Path]:
    return sorted(_PKG_ROOT.rglob("*.py"))


def test_no_reference_to_unpublished_deep_dive_memo() -> None:
    offenders = [
        str(path.relative_to(_PKG_ROOT))
        for path in _package_sources()
        if _UNPUBLISHED_MEMO in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"source references the unpublished '{_UNPUBLISHED_MEMO}' memo: {offenders}. "
        "Inline the rationale instead of pointing at a doc that is not shipped."
    )


def test_no_rotting_line_number_self_references() -> None:
    offenders = [
        str(path.relative_to(_PKG_ROOT))
        for path in _package_sources()
        if _ROTTING_LINE_REF.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        f"source uses rotting '~L<line>' self-references: {offenders}. "
        "Describe the location by symbol or behavior, not a line number."
    )
