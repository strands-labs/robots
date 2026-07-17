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
rejects these review-archaeology shapes:

  * ``review thread`` / ``review threads`` (the literal provenance framing),
  * ``flagged in review`` (its passive twin),
  * ``review <module>.py:<line>`` (a review citation carrying a rot-prone
    internal source-location back-pointer),
  * review-*round* provenance -- ``review [feedback] round N`` and a PR/issue
    number narrating a ``round N`` (e.g. ``#168 round 38``, ``#175 round 46d``).
    Which review pass produced a change is exactly the provenance that belongs
    in git history, not the source; the number rots and misdirects the reader.

References to *upstream* files (e.g. ``run_mujoco_gear_wbc.py:47-50``) and to
stable, named anchors (``AGENTS.md > Review Learnings``, a bare issue number
like ``#179``) are not matched -- only review-thread-by-line-pointer and
review-round provenance are. The scan flattens comment continuations so a
citation wrapped across ``#`` lines is still caught (the flattener also strips
the leading ``#`` of a ``#NNN`` PR reference, so the round pattern matches the
bare ``NNN round N`` that remains). It would have failed when 20+ comment
blocks across the ``mesh`` package still carried ``review thread
<file>.py:<line>`` pointers, and when ``mesh``/``simulation``/``benchmarks``
sources narrated the PR review round that produced a change.
"""

from __future__ import annotations

import re
from pathlib import Path

import strands_robots

# Review-archaeology framing that cites an internal review thread or PR review
# round. Loaded alternatives:
#   * ``review thread(s)`` / ``flagged in review`` -- literal provenance framing.
#   * ``review <module>.py:<line>`` -- a review citation carrying a rot-prone
#     internal source-location back-pointer.
#   * ``review [feedback] round N`` and ``<digits> round N`` -- narration of the
#     PR review pass that produced a change. The flattener strips the ``#`` of a
#     ``#NNN`` PR reference, so ``#168 round 38`` arrives here as ``168 round
#     38`` and is caught by the digits-prefixed alternative.
# Upstream ``<file>.py:<line>`` references (not preceded by ``review``), named
# anchors like ``Review Learnings``, and bare issue numbers are not matched.
_REVIEW_ARCHAEOLOGY = re.compile(
    r"(?i)(?:"
    r"review\s+threads?\b"
    r"|flagged\s+in\s+review"
    r"|\breview\s+[A-Za-z_][A-Za-z0-9_]*\.py:\d+"
    r"|review\s+(?:feedback\s+)?round\s+\d+"
    r"|\b\d+\s+round\s+\d+[a-z]?\b"
    r")"
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
        "Review archaeology found in strands_robots sources. Explain WHY the code "
        "is shaped this way and reference a stable symbol -- not a rot-prone "
        "``<file>.py:<line>`` review pointer nor the PR review round that produced "
        "the change:\n" + "\n".join(offenders)
    )


def test_review_round_provenance_is_matched() -> None:
    """The pattern rejects PR-review-round narration.

    The flattener strips the ``#`` of a ``#NNN`` PR reference, so these are the
    shapes the scan actually sees after flattening.
    """
    offenders = [
        "168 round 38",  # was "#168 round 38"
        "175 round 46d body-name",  # was "#175 round 46d ..."
        "See review feedback round 4 (symlink-swap defence).",
        "review round 2",
    ]
    for text in offenders:
        assert _REVIEW_ARCHAEOLOGY.search(text), f"should be flagged: {text!r}"


def test_legitimate_references_are_not_matched() -> None:
    """Stable references and incidental prose must not trip the round pattern."""
    allowed = [
        "run_mujoco_gear_wbc.py:47-50",  # upstream file:line, not review-prefixed
        "See AGENTS.md > Review Learnings for the rationale.",
        "Public since #179: standalone integration tests bypass this.",
        "round-trip the dataset and assert it is non-empty",
        "round the value to the nearest integer",
        "the first round of IK refinement converges quickly",  # 'round' without digits
    ]
    for text in allowed:
        assert not _REVIEW_ARCHAEOLOGY.search(text), f"false positive: {text!r}"
