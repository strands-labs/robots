"""Regression: no ``strands_robots`` module may cite an internal review thread
by ``<module>.py:<line>`` back-pointer.

AGENTS.md's review-archaeology rule (and the "code is the single source of
truth" doctrine) both reject comments that narrate the pull-request review that
produced a change. The worst offenders point at a sibling source location by
line number -- ``review thread session.py:296`` -- which is guaranteed to rot:
the referenced line moves on the very next edit, leaving a comment that
misdirects the reader to unrelated code. The rationale a comment needs to
convey is *why the code is shaped this way*, which stands on its own; the
provenance ("a reviewer asked for this in thread X at line Y") is noise that
belongs in git history, not the source.

This scan walks every Python module under the ``strands_robots`` package and
rejects three review-archaeology shapes:

  * ``review thread`` / ``review threads`` (the literal provenance framing),
  * ``flagged in review`` (its passive twin),
  * ``review <module>.py:<line>`` (a review citation carrying a rot-prone
    internal source-location back-pointer).

References to *upstream* files (e.g. ``run_mujoco_gear_wbc.py:47-50``) and to
stable, named anchors (``AGENTS.md > Review Learnings``, issue numbers) are not
matched -- only the review-thread-by-line-pointer pattern is. The scan flattens
comment continuations so a citation wrapped across ``#`` lines is still caught.
It would have failed when 20+ comment blocks across the ``mesh`` package still
carried ``review thread <file>.py:<line>`` pointers.
"""

from __future__ import annotations

import re
from pathlib import Path

import strands_robots

# Review-archaeology framing that cites an internal review thread. The third
# alternative is the load-bearing one: a ``review`` citation immediately
# followed by an internal ``<module>.py:<line>`` back-pointer that rots on the
# next edit. Upstream ``<file>.py:<line>`` references (not preceded by
# ``review``) and named anchors like ``Review Learnings`` are intentionally not
# matched.
_REVIEW_ARCHAEOLOGY = re.compile(
    r"(?i)(?:review\s+threads?\b|flagged\s+in\s+review|\breview\s+[A-Za-z_][A-Za-z0-9_]*\.py:\d+)"
)

# Collapse comment-continuation runs (`\n    # `) and other whitespace to a
# single space so a citation split across multiple ``#`` lines still matches.
_FLATTEN = re.compile(r"[\s#]+")

_PACKAGE_DIR = Path(strands_robots.__file__).resolve().parent


def _python_sources() -> list[Path]:
    return sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_package_sources_discovered() -> None:
    """Guard: the scan actually walked the whole package, not one subtree."""
    sources = _python_sources()
    assert len(sources) > 50
    rel_dirs = {p.relative_to(_PACKAGE_DIR).parts[0] for p in sources if p.parent != _PACKAGE_DIR}
    assert {"simulation", "tools", "registry", "mesh"} <= rel_dirs


def test_no_review_thread_line_pointers_in_package_sources() -> None:
    """No module may cite an internal review thread by ``<module>.py:<line>``."""
    offenders: list[str] = []
    for path in _python_sources():
        flattened = _FLATTEN.sub(" ", path.read_text(encoding="utf-8"))
        for match in _REVIEW_ARCHAEOLOGY.finditer(flattened):
            start = max(match.start() - 30, 0)
            snippet = flattened[start : match.end() + 30].strip()
            offenders.append(f"{path.relative_to(_PACKAGE_DIR.parent)}: ...{snippet}...")
    assert not offenders, (
        "Review-thread archaeology found in strands_robots sources. Explain WHY the "
        "code is shaped this way and reference a stable symbol, not a rot-prone "
        "``<file>.py:<line>`` review pointer:\n" + "\n".join(offenders)
    )
